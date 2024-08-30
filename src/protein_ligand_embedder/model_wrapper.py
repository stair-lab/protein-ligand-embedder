import gc
import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from DockingModels import EquivariantElucidatedDiffusion
from datasets import Dataset


class ProteinLigandWrapper(nn.Module):
    def __init__(
        self,
        model_name,
        config
    ):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.diffusion_model = None

    def load_model(self):
        model = EquivariantElucidatedDiffusion.from_pretrained(self.model_name, config=self.config, subfolder="ckpts")
        self.diffusion_model = model.to('cuda')

    def unload_model(self):
        del self.diffusion_model
        self.diffusion_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_emnbeddings(
          self, dataloader:DataLoader, num_steps:int, dtype: torch.dtype
        ):
      
        emb_dict = {}
        col_emb = []
        tqdm_dataloader = tqdm(dataloader)
        for batch in tqdm_dataloader:
            batch = batch.to(self.diffusion_model.device)
            _, x_t = self.diffusion_model.sample(batch, num_steps, dtype)
            lig_seq_len = torch.bincount(batch['ligand'].batch).tolist()
            lig_coords = torch.split(x_t, lig_seq_len)
        
            col_emb.append(lig_coords)
        emb_dict['embedding'] = col_emb

        return Dataset.from_dict(emb_dict)