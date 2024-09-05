import gc
import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from DockingModels import EquivariantElucidatedDiffusion
from datasets import Dataset
from torch_scatter import scatter, scatter_mean


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
            x_t, rec_node_attr, lig_node_attr = self.diffusion_model.sample(batch, num_steps, dtype)
            rec_node_attr = scatter_mean(rec_node_attr, batch['receptor'].batch, dim=0)
            lig_node_attr = scatter_mean(lig_node_attr, batch['ligand'].batch, dim=0)
            embedding = torch.cat([lig_node_attr, rec_node_attr], dim=1)
            col_emb.append(embedding)
        emb_dict['embedding'] = col_emb

        return Dataset.from_dict(emb_dict)