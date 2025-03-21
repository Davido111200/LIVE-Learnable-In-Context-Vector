# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hydra
import wandb
import torch
import torch.nn as nn
import string


from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration
import pytorch_lightning as pl

from rldataloader import VQAICVRLDataModule
from rlmodule import VQAICVModule  
from utils import init_interface
from icv_src.icv_model.icv_intervention import LearnableICVInterventionLMM
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from math_verify import parse, verify
from process_text import postprocess_completions

from open_r1_multimodal.src.open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config




class SteeringNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(torch.float16)  # Ensure dtype consistency
        
    def forward(self, x):
        x = x.to(torch.float16)  # Ensure dtype consistency
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
    def __init__(self, vilm, num_intervention_layers):
        super().__init__()
        self.vilm = vilm
        self.num_intervention_layers = num_intervention_layers
        self.config = self.vilm.lmm.model.config  # Forward config from the underlying model
        self.warnings_issued = {}  # Add this line to prevent the errors
        self.add_model_tags = {}


        
    def forward(self, input_ids, attention_mask, pixel_values, image_attention_mask, icv=None, generation_config=None):
        self.vilm.toggle_intervention(True)
        # icv = compute_icv(input_ids, attention_mask, pixel_values, image_attention_mask, self.vilm, self.steering_controller)

        # print if steering controller requires gradients
        return self.vilm(input_ids, attention_mask, pixel_values, image_attention_mask, icv=icv)
    
    def generate(self, input_ids, attention_mask, pixel_values, image_attention_mask, icv=None, generation_config=None):
        self.vilm.toggle_intervention(True)
        # icv = compute_icv(input_ids, attention_mask, pixel_values, image_attention_mask, self.vilm, self.steering_controller)
        return self.vilm.generate(input_ids, attention_mask, pixel_values, image_attention_mask, icv=icv)


def compute_icv(input_ids, attention_mask, pixel_values, image_attention_mask, vilm, steering_controller):
    # Get a prompt embedding from the base model.
    prompt_embedding = get_prompt_embedding(vilm, input_ids, attention_mask, pixel_values, image_attention_mask)  # [batch, hidden_dim]
    # Compute the steering matrix (outer product of two generated vectors).
    steering_matrix = steering_controller(prompt_embedding)  # [batch, hidden_dim, hidden_dim]
    icv = steering_matrix
    return icv


def get_prompt_embedding(vilm, input_ids, attention_mask, pixel_values, image_attention_mask):
    lmm = vilm.lmm
    # put all inputs on the same device as the model
    input_ids = input_ids.to(lmm.model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(lmm.model.device)
    pixel_values = pixel_values.to(lmm.model.device)
    image_attention_mask = image_attention_mask.to(lmm.model.device)

    # reshape the input_ids to have a batch size of 1 if not already
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        pixel_values = pixel_values.unsqueeze(0)
        image_attention_mask = image_attention_mask.unsqueeze(0)
        

    # Run the model with additional outputs
    vilm.toggle_intervention(False) 
    outputs = lmm.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_attention_mask=image_attention_mask, return_dict=True, output_hidden_states=True)
    # Hidden states from the last layer: [batch, seq_len, hidden_dim]
    last_hidden_state = outputs.hidden_states[-1]

    weights_for_non_padding = attention_mask.to(last_hidden_state.device) * \
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1, device=last_hidden_state.device).unsqueeze(0)
    sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_non_padding_tokens = torch.sum(weights_for_non_padding, dim=-1, keepdim=True)
    sentence_embeddings = sum_embeddings / num_of_non_padding_tokens
    return sentence_embeddings


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=16,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion for completion in completions]
    # process completions
    contents = postprocess_completions(contents)
    print("Completion: ", contents)
    print("Solution: ", solution)
    print("+++++")
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


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

@hydra.main(config_path="../config", config_name="train.yaml")
def main(cfg):
    training_args = GRPOConfig(per_device_train_batch_size=cfg.per_device_train_batch_size, num_train_epochs=cfg.nepochs)
    save_dir = "/home/s223540177/LIVE-Learnable-In-Context-Vector/checkpoints/HuggingFaceM4/idefics-9b/model_grpo.pth"
    
    # Parse additional arguments inside the function
    pl.seed_everything(cfg.seed)

    n_layers = cfg.lmm.total_layers
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

    vilm = LearnableICVInterventionLMM(
        interface,
        enable_intervention=True,
        intervention_layer=cfg.lmm.intervention_layer,
        layer_format=cfg.lmm.layer_format,
        total_layers=cfg.lmm.total_layers,
    )
    # vilm.lmm.eval()


    # The VILM class expects an ICV tensor of shape:
    # [batch, num_intervention_layers, hidden_dim, hidden_dim]
    num_intervention_layers = len(vilm.intervention_layer_names)

    # Instantiate the combined model.
    combined_model = VILMWithSteering(vilm, num_intervention_layers)

    # Freeze the base model parameters so only the steering controller is trainable.
    for param in combined_model.vilm.lmm.parameters():
        param.requires_grad = False

    # Get reward functions
    reward_funcs = [accuracy_reward, format_reward]

    # Load the dataset
    dataset = data_module.train_dataset()

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=combined_model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=dataset["eval"] if training_args.eval_strategy != "no" else None,
        cfg=cfg,
        prompt_manager=prompt_manager,
        peft_config=get_peft_config(cfg),
        processing_class=processor,
        attn_implementation=cfg.attn_implementation,
        max_pixels=cfg.max_pixels,
        min_pixels=cfg.min_pixels,
        model_name=cfg.model_name,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    torch.save(combined_model, save_dir)

    # trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=cfg.dataset_name)


if __name__ == "__main__":
    main()


