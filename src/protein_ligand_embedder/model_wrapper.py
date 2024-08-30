import yaml
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from argparse import ArgumentParser, Namespace, FileType
from DockingModels import EquivariantElucidatedDiffusion, en_score_model_l1_4M_drop01, en_score_model_l1_21M_drop01, model_entrypoint


class ConfidenceHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.linear(x)

class EGNNModelWrapper(nn.Module):
    def __init__(
        self,
        model_dir,
        ckpt_path,
        device,
        sigma_max=160,
        sigma_min=0.002,
        rho=7,
        S_churn=80,
        S_min=0.05,
        S_max=50,
        S_noise=1.003,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.ckpt_path = ckpt_path
        self.device = device
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.load_model()
        self.confidence_head = ConfidenceHead(3, 1)

    def load_model(self):
        with open(f'{self.model_dir}/model_parameters.yml') as f:
            model_args = Namespace(**yaml.full_load(f))

        create_model = model_entrypoint(model_args.model_name)
        score_model = create_model(device=self.device, lm_embedding_type='esm')
        model = EquivariantElucidatedDiffusion(
            net=score_model,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            sigma_data=model_args.sigma_data,
            rho = self.rho,
            S_churn=self.S_churn,
            S_min=self.S_min,
            S_max=self.S_max,
            S_noise=self.S_noise,
        )

        state_dict = torch.load(f'{self.model_dir}/{self.ckpt_path}', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        self.diffusion_model = model.to(self.device)

    def forward(self, data, num_steps, dtype):

        _, x_t = self.diffusion_model.sample(data, num_steps, dtype)
        lig_seq_len = torch.bincount(data['ligand'].batch).tolist()
        lig_coords = torch.split(x_t, lig_seq_len)
        pred = self.confidence_head(scatter_mean(x_t, data['ligand'].batch, dim=0))

        return pred, lig_coords