import torch
import numpy as np
from tqdm import tqdm
from timm.utils import ModelEmaV2

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from dataloader import MiniDataset
from model_wrapper import EGNNModelWrapper
from utils import get_rmsds, AverageMeter
from DockingModels import CustomConfig

from argparse import ArgumentParser

def get_args_parser():

    parser = ArgumentParser()

    # Model
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Load pretrained weights')
    parser.add_argument('--model_dir', type=str, default=None, help='Model path')
    parser.add_argument('--root', type=str, default=None, help='data path')
    parser.add_argument('--use_bfp16', action='store_true', default=False, help='Use bfloat16')
    parser.add_argument('--num_steps', type=int, default=20, help='sampling steps')

    args = parser.parse_args()

    return args


def evaluate(args, model, val_loader, dtype):
    model.eval()
    val_losses = AverageMeter()

    for batch in val_loader:
        batch = batch.to(args.device)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
            pred, lig_coords = model(batch, args.num_steps, dtype=dtype)
            target = get_rmsds(lig_coords, batch).to(args.device)
            val_loss = F.mse_loss(pred.float(), target.float())
        val_losses.update(val_loss.detach().cpu().item(), n=args.batch_size)

    return val_losses.avg

def train(args, device, model, optimizer, scheduler, model_ema, train_loader, val_loader, dtype):
    model.train()
    train_losses = []
    train_loss = AverageMeter()
    val_losses = []

    for epoch in tqdm(range(args.epochs), desc="Epochs Progress"):
        batch_idx = 0

        for batch in tqdm(train_loader):
            batch = batch.to(args.device)
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            with torch.autocast(device_type='cuda', dtype=dtype):
                pred, lig_coords = model(batch, args.num_steps, dtype=dtype)
                target = get_rmsds(lig_coords, batch).to(args.device)
                loss = F.mse_loss(pred.float(), target.float())
                loss /= args.accum_steps
                batch_idx += 1
            loss.backward()
            batch_loss += loss

            if (batch_idx + 1) % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_idx = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                model_ema.update(model)
                train_loss.update(batch_loss.detach().cpu().item(), n=args.batch_size)

        epoch_loss = train_loss.avg
        train_loss.reset()
        train_losses.append(epoch_loss)

        val_loss = evaluate(args, model_ema, val_loader, dtype)
        val_losses.append(val_loss)

        scheduler.step(val_losses if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    dtype = torch.bfloat16 if (args.use_bfp16 and device.type == 'cuda') else torch.float32

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    ''' Dataset '''
    train_dataset, val_dataset = MiniDataset(args.root, 'train'), MiniDataset(args.root, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, prefetch_factor=1, shuffle=True,
                              num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, prefetch_factor=1, shuffle=False,
                              num_workers=args.num_dataloader_workers, pin_memory=args.pin_memory)

    ''' Model '''
    congig = CustomConfig.from_pretrained(args.model_dir, subfolder="ckpts") #change the params of diffusion sampling process as needed
    model = EGNNModelWrapper(args.model_dir, congig).to(device)

    model_ema = ModelEmaV2(model, decay=0.9999, device=device)

    ''' Optimizer and LR Scheduler '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train(args, device, model, optimizer, scheduler, model_ema, train_loader, val_loader, dtype)

if __name__ == "__main__":

    args = get_args_parser()

    main(args)