import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import copy
import torch.nn as nn
import pytorch_lightning as pl
import logging
from contextlib import nullcontext
from typing import Dict
from omegaconf import DictConfig
import hydra
import wandb
from rldataloader import VQAICVRLDataModule
from rlmodule import VQAICVModule  
from utils import init_interface
from icv_src.icv_model.icv_intervention import LearnableICVInterventionLMM
import os
import re

PATH_TO_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__)))


logger = logging.getLogger(__name__)


class WandbTrainingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)


# ========================================================
# Define the steering networks and controller.
# Two small networks that take a prompt embedding and produce two vectors.
class SteeringNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class SteeringICVController(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim):
        super().__init__()
        self.steering_net_A = SteeringNetwork(input_dim, n_layers)  # Outputs [batch, n_layers]
        self.steering_net_B = SteeringNetwork(input_dim, hidden_dim)  # Outputs [batch, hidden_dim]
    
    def forward(self, prompt_embedding):
        # prompt_embedding: [batch, input_dim]
        vec_A = self.steering_net_A(prompt_embedding)  # [batch, n_layers]
        vec_B = self.steering_net_B(prompt_embedding)  # [batch, hidden_dim]

        # Outer product: [batch, n_layers, hidden_dim]
        steering_matrix = torch.bmm(vec_A.unsqueeze(2), vec_B.unsqueeze(1))

        return steering_matrix

class VILMWithSteering(nn.Module):
    def __init__(self, vilm, steering_controller, num_intervention_layers):
        super().__init__()
        self.vilm = vilm
        self.steering_controller = steering_controller
        self.num_intervention_layers = num_intervention_layers
        self.config = self.vilm.lmm.model.config  # Forward config from the underlying model
        self.warnings_issued = {}  # Add this line to prevent the errors
        self.add_model_tags = {}

        
    def forward(self, input_ids, **kwargs):
        # Compute the ICV from the prompt.
        icv = compute_icv(input_ids, self.vilm, self.steering_controller, self.num_intervention_layers)
        return self.vilm(input_ids=input_ids, icv=icv, **kwargs)
    
    def generate(self, input_ids, **kwargs):
        icv = compute_icv(input_ids, self.vilm, self.steering_controller, self.num_intervention_layers)
        return self.vilm.generate(input_ids=input_ids, icv=icv, **kwargs)


def compute_icv(input_ids, vilm, steering_controller, num_intervention_layers):
    # Get a prompt embedding from the base model.
    prompt_embedding = get_prompt_embedding(vilm.lmm, input_ids)  # [batch, hidden_dim]
    # Compute the steering matrix (outer product of two generated vectors).
    steering_matrix = steering_controller(prompt_embedding)  # [batch, hidden_dim, hidden_dim]
    # Replicate the matrix across all intervention layers.
    icv = steering_matrix.unsqueeze(1).repeat(1, num_intervention_layers, 1, 1)
    return icv


def get_prompt_embedding(lmm, input_ids):
    # Use a forward pass through the base model to extract a prompt representation.
    # We assume that the first token's hidden state from the last layer is a good summary.
    with torch.no_grad():
        outputs = lmm(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
        prompt_embedding = hidden_states[:, 0, :]    # shape: [batch, hidden_dim]
    return prompt_embedding


def exactmatch_reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]



@hydra.main(config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    # set seed
    # torch.manual_seed(cfg.seed)
    # torch.cuda.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    input_dim = 4096  # Adjust as needed.
    total_layers = cfg.lmm.total_layers
    hidden_dim = cfg.lmm.hidden_size
    
    wandb.init(
        project="RLVQAInContextVector",
        name=cfg.run_name,
        config=dict(cfg),
        dir=cfg.result_dir
    )

    prompt_manager, interface, processor = init_interface(cfg)

    data_module = VQAICVRLDataModule(cfg.data_cfg, prompt_manager, processor)
    data_module.setup("fit")
    train_dataset = data_module.train_dataset()

    vilm = LearnableICVInterventionLMM(
        interface,
        enable_intervention=True,
        intervention_layer=cfg.lmm.intervention_layer,
        layer_format=cfg.lmm.layer_format,
        total_layers=cfg.lmm.total_layers,
    )
    vilm.lmm.eval()


    # ========================================================
    # Create the steering controller.
    steering_controller = SteeringICVController(input_dim=hidden_dim, hidden_dim=hidden_dim)

    # The VILM class expects an ICV tensor of shape:
    # [batch, num_intervention_layers, hidden_dim, hidden_dim]
    num_intervention_layers = len(vilm.intervention_layer_names)

    # Instantiate the combined model.
    combined_model = VILMWithSteering(vilm, steering_controller, num_intervention_layers)

    # Freeze the base model parameters so only the steering controller is trainable.
    for param in combined_model.vilm.lmm.parameters():
        param.requires_grad = False

    # ========================================================
    # Create a reference model for PPOTrainer (a copy of combined_model).
    ref_model = copy.deepcopy(combined_model)
    ref_model.eval()

    output_dir = f"{PATH_TO_REPO}/output/{cfg.lmm}-PPO-seed{cfg.seed}"
    run_name = f"{cfg.lmm}-PPO-seed{cfg.seed}"

    do_eval = True
    eval_strategy ="steps"
    if cfg.do_eval==0:
        do_eval = False
        eval_strategy = "no"

    reward_list = []
    reward_list.append(exactmatch_reward_func)



    # Configure PPOTrainer.
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        eval_strategy=eval_strategy,
        eval_steps=50,
        do_eval=do_eval,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=cfg.bs,
        gradient_accumulation_steps=cfg.gc,
        num_generations=cfg.n_actions,
        max_prompt_length=256,
        max_completion_length=cfg.generate_kwargs.max_new_tokens,
        num_train_epochs=cfg.nepochs,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
    )

    ppo_trainer = GRPOTrainer(
        model=combined_model,
        processing_class=processor.processor.tokenizer,
        args=training_args,
        reward_funcs=reward_list,
        train_dataset=train_dataset,
    )

    # ========================================================
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_model.to(device)
    ref_model.to(device)

    # ========================================================
    # Training loop using PPOTrainer.
    num_epochs = 3  # You can also use cfg.train.num_epochs if defined.
    logger.info("Starting training loop...")

    ppo_trainer.add_callback(WandbTrainingCallback()) 

    ppo_trainer.train()

if __name__ == "__main__":
    main()
