import os
import shutil
import re
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime
from omegaconf import DictConfig
from math_verify import parse, verify
from trl import GRPOConfig, get_peft_config
from transformers import get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    RichProgressBar,
)
from lmm_icl_interface import LMMInterface

from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from agent_ppoLoRA.rldataloader import VQAICVRLDataModule
from icv_src.icv_module_outer import VQAICVModuleOuter
from icv_src.icv_model.icv_intervention import LearnableICVInterventionLMM
from icv_src.icv_encoder.global_lora_encoder import GlobalICVEncoderOuter


from utils import get_icv_cpk_path, init_interface

from agent_ppoLoRA.process_text import postprocess_completions


from open_r1_multimodal.src.open_r1.trainer import Qwen2VLGRPOTrainer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


from transformers import TrainerCallback
# class GradientLoggingCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         model = kwargs.get("model", None)
#         for name, param in model.icv_encoder.named_parameters():
#             if param.grad is None:
#                 print(f"{name}: gradient is None")
#             else:
#                 grad_norm = param.grad.norm().item()
#                 print(f"{name}: grad norm = {grad_norm}")
#                 if torch.isnan(param.grad).any():
#                     print(f"NaNs found in gradient for {name}")
            

class VILMWithSteering(nn.Module):
    def __init__(self, interface: LMMInterface, cfg, module_cfg, lmm_cfg, num_intervention_layers, icv_cpk):
        super().__init__()
        self.interface = interface
        self.interface.requires_grad_(False)
        self.model = LearnableICVInterventionLMM(
            interface,
            enable_intervention=True,
            intervention_layer=cfg.lmm.intervention_layer,
            layer_format=cfg.lmm.layer_format,
            total_layers=cfg.lmm.total_layers,
        )
        self.module_cfg = module_cfg
        self.lmm_cfg = lmm_cfg 
        self.num_intervention_layers = num_intervention_layers
        self.config = self.model.lmm.model.config  # Forward config from the underlying model
        self.warnings_issued = {}  # Add this line to prevent the errors
        self.add_model_tags = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.icv_cpk = icv_cpk

        icv_encoder_factor: GlobalICVEncoderOuter = hydra.utils.instantiate(
            module_cfg.icv_encoder, _partial_=True, icv_cpk=icv_cpk
        )
        icv_layer_num = len(self.model.intervention_layer_names)
        hidden_dim = self.lmm_cfg.hidden_size
        self.icv_encoder = icv_encoder_factor(
            n_layers=icv_layer_num, hidden_dim=hidden_dim
        )

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # # check which module requires grad
                
    def forward(self, input_ids, attention_mask, pixel_values, image_attention_mask, labels, is_ref, generation_config=None):
        if not is_ref:
            icv_encoder_output = self.icv_encoder()
            icv = (
                icv_encoder_output.alpha.unsqueeze(dim=-1)
                * icv_encoder_output.in_context_vector
            )
            self.model.toggle_intervention(True)
            # print("ICV: ", icv)
            return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_attention_mask=image_attention_mask, labels=labels, icv=icv, is_ref=is_ref, retain_grad=True, use_cache=False)
        else:
            with torch.inference_mode():
                self.model.toggle_intervention(False)
                return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_attention_mask=image_attention_mask, labels=None, icv=None, is_ref=is_ref, retain_grad=False, use_cache=False)
    
    def generate(self, input_ids, attention_mask, pixel_values, image_attention_mask, labels, generation_config=None):
        self.model.toggle_intervention(True)
        icv_encoder_output = self.icv_encoder()
        icv = (
            icv_encoder_output.alpha.unsqueeze(dim=-1)
            * icv_encoder_output.in_context_vector
        )
        # icv here does not require grad
        with torch.inference_mode():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_attention_mask=image_attention_mask, labels=None, icv=icv, generation_config=generation_config, use_cache=False)
            return out

    def configure_optimizers(self):
        params = []
        for name, param in self.icv_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if "alpha" in name:
                params.append({"params": param, "lr": self.module_cfg.alpha_lr})
            else:
                print("Param requires grad: ", name)
                params.append({"params": param})

        if "deepspeed" in self.module_cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                params,
                lr=self.module_cfg.icv_lr,
                weight_decay=self.module_cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                params,
                lr=self.module_cfg.icv_lr,
                weight_decay=self.module_cfg.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        if isinstance(self.module_cfg.warm_steps, float):
            warm_steps = self.module_cfg.warm_steps * step_batches
        elif isinstance(self.module_cfg.warm_steps, int):
            warm_steps = self.module_cfg.warm_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(self.module_cfg.warm_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@hydra.main(config_path="config", config_name="train.yaml")
def main(cfg: DictConfig):
    global logger
    pl.seed_everything(cfg.seed)
    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)
    result_dir = Path(cfg.result_dir)
    model_name = cfg.lmm.model_name

    training_args = GRPOConfig(per_device_train_batch_size=cfg.per_device_train_batch_size, num_train_epochs=cfg.nepochs, learning_rate=5e-7)

    save_grpo_dir = result_dir / "icv_cpk_grpo.pth"

    wb_logger = WandbLogger(
        save_dir=cfg.result_dir,
        name=cfg.run_name,
        project="VQAInContextVectorOuter",
        log_model=False,
    )
    wb_logger.log_hyperparams(dict(cfg))
    prompt_manager, interface, processor = init_interface(cfg)

    # model = VQAICVModuleOuter(
    #     interface=interface, module_cfg=cfg.icv_module, lmm_cfg=cfg.lmm
    # )
    data_module = VQAICVRLDataModule(
        data_cfg=cfg.data_cfg, prompt_manager=prompt_manager, prompt_processor=processor
    )
    data_module.setup("fit")
    dataset = data_module.train_dataset()

    model_cpk_dir = get_icv_cpk_path(
        result_dir,
        model_name=model_name,
        dataset_name=cfg.data_cfg.task.datasets.name,
        run_name=cfg.run_name,
    )

    # we continue training 2 vectors from the checkpoint with GRPO
    icv_cpk = torch.load(model_cpk_dir / "icv_cpk.pth")
    
    # load the 2 trained vectors


    # model.icv_model refers to the model with LeanableICVInterventionLMM
    combined_model = VILMWithSteering(interface, cfg=cfg, module_cfg=cfg.icv_module, lmm_cfg=cfg.lmm, num_intervention_layers=cfg.lmm.total_layers, icv_cpk=icv_cpk)
    optimizers = None

    # put this model through GRPO trainer
    trainer_cls = Qwen2VLGRPOTrainer

    reward_funcs = [accuracy_reward]

    trainer = trainer_cls(
        model=combined_model,
        reward_funcs=reward_funcs,
        processing_class=processor,
        peft_config=get_peft_config(cfg),
        cfg=cfg,
        prompt_manager=prompt_manager,
        args=training_args,
        train_dataset=dataset,
        attn_implementation=cfg.attn_implementation,
        max_pixels=cfg.max_pixels,
        min_pixels=cfg.min_pixels,
        model_name=cfg.model_name,
        optimizers=optimizers,
        # callbacks=[GradientLoggingCallback()],
        logger=wb_logger,
    )

    trainer.train()

    # save the model
    torch.save(combined_model.state_dict(), save_grpo_dir)




def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion for completion in completions]
    # process completions
    contents = postprocess_completions(contents)
    # print("Completion: ", contents)
    # print("Solution: ", solution)
    # print("+++++")
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]



if __name__ == "__main__":
    load_dotenv()
    main()
