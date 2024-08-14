import math
import torch
import numpy as np
from torch import nn
from datasets import Dataset
from torch.nn import functional as F
from EquivariantElucidatedDiffusion import *

class EGNNModelWrapper(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def load(self, model_name, ckpt_path, device, yaml, lm_embedding_type='esm'):
        create_model = model_entrypoint(model_name)
        score_model = create_model(device=device, lm_embedding_type=lm_embedding_type)
        self.original_model = score_model
        if ckpt_path:
            state_dict = torch.load(f'ckpt_path', map_location=torch.device('cpu'))
            self.original_model.load_state_dict(state_dict['net'], strict=True)

        self.original_model.to(device)

        with open(f'{args.model_dir}/model_parameters.yml') as f:
            model_args = Namespace(**yaml.full_load(f))

        self.model_args = model_args

    def forward(self, data, times, dtype):

        x, sigma = data.noised_atom_pos, data.padded_sigmas

        batch = data['ligand'].batch
        c_skip = self.model_args.sigma_data ** 2 / (sigma[batch] ** 2 + self.model_args.sigma_data ** 2)
        c_out = sigma[batch] * self.model_args.sigma_data / (sigma[batch] ** 2 + self.model_args.sigma_data ** 2).sqrt()
        c_in = 1 / (self.model_args.sigma_data ** 2 + sigma[batch] ** 2).sqrt()
        c_noise = (sigma/self.model_args.sigma_data).log() / 4

        data['ligand'].pos = c_in * x

        _, node_attr, coords_attr = self.original_model(data, c_noise, dtype)

        # Return only the coords and node_attr
        emb_dict = {
            'coords_attr': coords_attr.detach().cpu(),
            'node_attr': node_attr.detach().cpu()
        }

        emb_dataset = Dataset.from_dict(emb_dict)
        return emb_dataset
