import torch
import hydra
from omegaconf import DictConfig
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.distributions import Normal
import torch.nn.functional as F
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
import os

from rldataloader import VQAICVRLDataModule
from rlmodule import VQAICVModule  
from utils import init_interface
from inference import generate_answers_fixed_alpha


# -----------------------------
# Outer Product Policy Network
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class OuterProductPolicyNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        total_layers, 
        hidden_size, 
        num_actions=64,
    ):
        super().__init__()
        self.total_layers = total_layers
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # MLP for computing layer coefficients.
        self.layer_coeffs_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, total_layers)
        )

        # MLPs for the Gaussian parameters of the edit coefficients.
        self.edit_mean = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, hidden_size)
        )
        self.edit_log_std = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, hidden_size)
        )

        for param in self.layer_coeffs_mlp.parameters():
            param.requires_grad = True
        # require grad for edit_mean and edit_log_std
        for param in self.edit_mean.parameters():
            param.requires_grad = True
        for param in self.edit_log_std.parameters():
            param.requires_grad = True

        # Initialize parameters
        self._initialize_weights()


    def _initialize_weights(self):
        # Initialize last layer of edit_mean to output ~0
        nn.init.zeros_(self.edit_mean[-1].bias)
        nn.init.xavier_uniform_(self.edit_mean[-1].weight, gain=0.01)

        # Initialize last layer of edit_log_std to output log(0.01) = -4.6
        nn.init.constant_(self.edit_log_std[-1].bias, -4.6)
        nn.init.xavier_uniform_(self.edit_log_std[-1].weight, gain=0.01)



    def forward(self, x):
        B = x.size(0)

        # Compute layer coefficients (B, total_layers, 1)
        coeffs = self.layer_coeffs_mlp(x)
        layer_coeffs = torch.sigmoid(coeffs)  # Normalize to [0, 1]
        # layer_coeffs = 0.1 + (2 - 0.1) * layer_coeffs  # Scale to [0.1, 1]
        layer_coeffs = layer_coeffs.unsqueeze(-1)  # (B, total_layers, 1)
        # print("layer_coeffs: ", layer_coeffs)

        # Compute Gaussian parameters for edit coefficients.
        mean = self.edit_mean(x)
        log_std = self.edit_log_std(x)
        std = torch.exp(log_std)
        edit_dist = Normal(mean, std)

        # Sample a single action per batch item
        raw_edit = edit_dist.rsample()  # (B, hidden_size)
        # print("raw_edit: ", raw_edit)
        log_prob_edit = edit_dist.log_prob(raw_edit).sum(dim=-1)  # (B,)
        entropy = edit_dist.entropy().sum(dim=-1)  # (B,)

        # Reshape `raw_edit` for outer product (B, 1, hidden_size)
        edit_coeffs = raw_edit.unsqueeze(1)  
        # print("edit_coeffs: ", edit_coeffs)

        # Compute the outer product (B, total_layers, hidden_size)
        outer = layer_coeffs * edit_coeffs

        return outer, log_prob_edit, entropy, raw_edit

    def evaluate_actions(self, x, raw_edit):
        coeffs = self.layer_coeffs_mlp(x)
        layer_coeffs = coeffs.unsqueeze(-1)  # (B, total_layers, 1)

        mean = self.edit_mean(x)
        log_std = self.edit_log_std(x)
        std = torch.exp(log_std)
        edit_dist = Normal(mean, std)

        log_prob_edit = edit_dist.log_prob(raw_edit).sum(dim=-1)  # (B,)

        return log_prob_edit, layer_coeffs

# -----------------------------
# Baseline Network
# -----------------------------
class BaselineNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# -----------------------------
# Main Training Routine
# -----------------------------
@hydra.main(config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    # check if dir already exists
    # if os.path.exists("/home/s223540177/LIVE-Learnable-In-Context-Vector/saves/outer_policy_net.pth"):
    #     raise Exception("OuterProductPolicyNet already exists at /home/s223540177/LIVE-Learnable-In-Context-Vector/saves/outer_policy_net.pth")
    # if os.path.exists("/home/s223540177/LIVE-Learnable-In-Context-Vector/saves/baseline_net.pth"):
    #     raise Exception("BaselineNet already exists at /home/s223540177/LIVE-Learnable-In-Context-Vector/saves/baseline_net.pth")
    
    pl.seed_everything(cfg.seed)
    
    input_dim = 4096  # Adjust as needed.
    total_layers = cfg.lmm.total_layers
    hidden_size = cfg.lmm.hidden_size
    num_actions = 64

    policy_net = OuterProductPolicyNet(input_dim=input_dim, total_layers=total_layers, hidden_size=hidden_size, num_actions=num_actions)
    baseline_net = BaselineNet(input_dim=input_dim)
    
    wandb.init(
        project="RLVQAInContextVector",
        name=cfg.run_name,
        config=dict(cfg),
        dir=cfg.result_dir
    )

    prompt_manager, interface, processor = init_interface(cfg)
    data_module = VQAICVRLDataModule(cfg.data_cfg, prompt_manager, processor)
    model = VQAICVModule(interface, cfg, cfg.icv_module, cfg.lmm, policy_net, baseline_net, processor)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=0.5
    )

    trainer.fit(model, data_module)

    # Save the final model checkpoint
    torch.save(policy_net.state_dict(), "/home/s223540177/LIVE-Learnable-In-Context-Vector/saves/outer_policy_net.pth")
    print("OuterProductPolicyNet saved as /home/s223540177/LIVE-Learnable-In-Context-Vector/saves/outer_policy_net.pth")

    # save the final baseline checkpoint
    torch.save(baseline_net.state_dict(), "/home/s223540177/LIVE-Learnable-In-Context-Vector/saves/baseline_net.pth")
    print("BaselineNet saved as /home/s223540177/LIVE-Learnable-In-Context-Vector/saves/baseline_net.pth")


if __name__ == "__main__":
    main()
