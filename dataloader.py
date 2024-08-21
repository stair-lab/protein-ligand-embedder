import os
import torch

from rdkit import Chem
from torch_geometric.data import Dataset


class MiniDataset(Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__(root, transform)
        self.processed_dir_path = root
        self.split = split
        self.csv = 'test_200' if split == 'val' else 'train_5k'
        # Set the token for authentication
        self.cache = os.path.join(self.processed_dir_path, 'cache', f'{self.split}_PDBSCAN22_limit0_INDEX{self.split}_recRad15.0_recMax24_esmEmbeddings_knnOnly_recSizeMax3000_{self.csv}')

    @property
    def processed_file_names(self):
        # Dynamically list all .pt files in the processed directory
        return [f for f in os.listdir(self.cache) if f.endswith('.pt')]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load the processed data from .pt files
        data_path = os.path.join(self.cache, self.processed_file_names[idx])
        data = torch.load(data_path)

        name = data.name.split(".pdb_")[1]
        lig_path = os.path.join(self.processed_dir_path, 'pair_processing', self.split + '_ligands', name)
        supplier = Chem.SDMolSupplier(lig_path, sanitize=False, removeHs=False)
        lig = supplier[0]
        data.mol = lig

        return data
