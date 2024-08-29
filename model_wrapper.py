import yaml
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from argparse import ArgumentParser, Namespace, FileType
from DockingModels import EquivariantElucidatedDiffusion, en_score_model_l1_4M_drop01, en_score_model_l1_21M_drop01, model_entrypoint, CustomConfig


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
        model_name,
        config
    ):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.confidence_head = ConfidenceHead(3, 1)

    def load_model(self):
        model = EquivariantElucidatedDiffusion.from_pretrained(self.model_name, config=self.config, subfolder="ckpts")
        self.diffusion_model = model.to(self.device)

    def forward(self, data, num_steps, dtype):

        _, x_t = self.diffusion_model.sample(data, num_steps, dtype)
        lig_seq_len = torch.bincount(data['ligand'].batch).tolist()
        lig_coords = torch.split(x_t, lig_seq_len)
        pred = self.confidence_head(scatter_mean(x_t, data['ligand'].batch, dim=0))

        return pred, lig_coords