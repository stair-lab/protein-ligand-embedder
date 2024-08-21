import torch
import signal
import numpy as np
from spyrmsd import rmsd, molecule
from rdkit.Chem import RemoveAllHs
from contextlib import contextmanager

from argparse import ArgumentParser, Namespace, FileType

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    with time_limit(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_rmsds(lig_coords, batch):

    data_list = batch.to_data_list()

    filterHs = [torch.not_equal(data['ligand'].x[:, 0], 0).cpu().numpy() for data in data_list]

    ligand_pos = np.asarray([pos.cpu().numpy()[filterHs[i]] for i, pos in enumerate(lig_coords)], dtype=object)

    mol = [RemoveAllHs(complex_graph.mol) for complex_graph in data_list]

    orig_ligand_pos = np.array([complex_graph['ligand'].orig_pos[filterHs[i]] - complex_graph.original_center.cpu().numpy() for i, complex_graph in enumerate(data_list)], dtype=object)

    rmsds = []
    for i in range(len(orig_ligand_pos)):
        try:
            rmsd = get_symmetry_rmsd(mol[i], orig_ligand_pos[i], [ligand_pos[i]])
        except Exception as e:
            print("Using non corrected RMSD because of the error:", e)
            rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append(rmsd)
    rmsds = np.array(rmsds)

    return torch.from_numpy(rmsds)
