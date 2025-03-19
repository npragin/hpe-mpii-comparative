import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from config import config
from datasets.mpii import getMPIIDataloaders
from models.SwinKeypointDetector import SwinKeypointDetector

import argparse

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    print("Using GPU")
    device = "cuda"
else:
    print("Using CPU")
    device = "cpu"

kpt_k = (torch.tensor([
        .089,  # 0: r_ankle
        .087,  # 1: r_knee
        .107, # 2: r_hip
        .107, # 3: l_hip
        .087,  # 4: l_knee
        .089,  # 5: l_ankle
        .107, # 6: pelvis
        .107, # 7: thorax
        .087,  # 8: upper_neck
        .089,  # 9: head_top
        .062,  # 10: r_wrist # Changed from 0.79
        .072,  # 11: r_elbow
        .079,  # 12: r_shoulder
        .079,  # 13: l_shoulder
        .072,  # 14: l_elbow
        .062   # 15: l_wrist # Changed from 0.79
    ], dtype=torch.float32) * 2).to(device)

def plotVisualization(model: SwinKeypointDetector, datapoint):
    model.eval()
    datapoint = datapoint.to(device)
    datapoint = datapoint.unsqueeze(0)
    with torch.no_grad():
        coords_out, visibility_out = model(datapoint)
    coords_out = coords_out.to("cpu")
    visibility_out = visibility_out.to("cpu")
    datapoint = datapoint.to("cpu")
    datapoint = datapoint.squeeze(0)

    visibility_out = visibility_out.squeeze()
    visibility = F.sigmoid(visibility_out)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_tensor = torch.tensor(mean, dtype=datapoint.dtype, device=datapoint.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=datapoint.dtype, device=datapoint.device).view(3, 1, 1)
    unnormalized_datapoint = datapoint * std_tensor + mean_tensor
    unnormalized_datapoint = unnormalized_datapoint.permute(1,2,0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(unnormalized_datapoint)

    coords_out = coords_out.squeeze()

    for i, (coord, vis) in enumerate(zip(coords_out, visibility)):
        if vis.item() > 0.5:
            # Assuming coords are [x, y]
            x, y = coord
            ax.scatter(x, y, c='r', s=50)

    plt.title('Keypoint Visualization')
    wandb.log({"KP Visualization": fig})

# TODO: Use the actual k metric and use head bounding box area instead of actual
def getOKS(x, y, bb):
    # Denormalize keypoints
    x[:,:,0] *= (bb.unsqueeze(1)[:,:,2]).to(device)
    x[:,:,1] *= (bb.unsqueeze(1)[:,:,3]).to(device)
    y[:,:,0] *= (bb.unsqueeze(1)[:,:,2]).to(device)
    y[:,:,1] *= (bb.unsqueeze(1)[:,:,3]).to(device)

    epsilon = torch.finfo(torch.float32).eps
    distances_squared = ((x - y[:,:,:2])**2).sum(dim=-1)
    vis_mask = y[:, :, 2] != 0
    vis_mask_sum = vis_mask.sum(-1)
    areas = (bb[:, 2] * bb[:, 3]).unsqueeze(-1).to(device)

    denom = 2 * kpt_k ** 2 * (areas + epsilon)

    exp_term = distances_squared / denom
    oks = torch.exp(-exp_term).sum(-1)
    oks /= vis_mask_sum
    total_oks = torch.sum(oks)

    return total_oks

def evaluate(model: SwinKeypointDetector, test_loader: DataLoader):
    running_loss = 0
    running_oks = 0
    coordinates_criterion = nn.SmoothL1Loss()
    visibility_criterion = nn.BCEWithLogitsLoss()

    plotVisualization(model, test_loader.dataset[torch.randint(0, len(test_loader.dataset), (1,))][0])

    model.train()
    with torch.no_grad():
        for x, y, bb in test_loader:
            x = x.to(device)
            y = y.to(device)
            bb = bb.to(device)

            coords_out, visibility_out = model(x)
            visibility_out = visibility_out.squeeze(-1)

            coords_loss = coordinates_criterion(coords_out, y[:, :, :2])
            scaled_coords_loss = coords_loss * config["coords_loss_weight"]
            visibility_loss = visibility_criterion(visibility_out, y[:, :, 2])
            scaled_visibility_loss = visibility_loss * config["vis_loss_weight"]

            loss = scaled_coords_loss + scaled_visibility_loss
            
            oks = getOKS(coords_out, y, bb)

            running_loss += loss
            running_oks += oks

    return running_loss / len(test_loader.dataset), running_oks / len(test_loader.dataset)

def train(model: SwinKeypointDetector, train_loader: DataLoader, val_loader: DataLoader):
    # Log number of parameters in model
    config["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Setup wandb
    wandb.login()
    wandb.init(project="MPII HPE", name=datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), config=config)

    # Model -> GPU
    model.to(device)

    # Set up optimizer with different LR for SWIN and MLP components
    optimizer = AdamW([
        {"params": model.swin.parameters(), "lr": config["swin_lr"]},
        {"params": model.coords_head.parameters()},
        {"params": model.visibility_head.parameters()}
    ], lr=config["mlp_lr"])

    # Set up learning rate scheduler with linear warmup period
    warmup = LinearLR(
        optimizer,
        start_factor=config["warmup_lr_factor"],
        total_iters=config["warmup_epochs"]
    )

    annealing = CosineAnnealingLR(
        optimizer,
        T_max=config["max_epoch"] - config["warmup_epochs"]
    )

    scheduler = SequentialLR(optimizer,
        schedulers=[warmup, annealing],
        milestones=[config["warmup_epochs"]]
    )

    coordinates_criterion = nn.SmoothL1Loss()
    visibility_criterion = nn.BCEWithLogitsLoss()

    # TODO: Implement convergence monitoring of MLP vs SWIN (gradient magnitudes or weight deltas)
    iteration = 0
    best_val = 0
    pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Iterations", unit="batch")
    for epoch in range(config["max_epoch"]):
        model.train()

        # Log LR
        wandb.log({"SWIN_LR": scheduler.get_last_lr()[0], "MLP_LR": scheduler.get_last_lr()[1]}, step=iteration)

        for x, y, bb in train_loader:
            x = x.to(device)
            y = y.to(device)

            coords_out, visibility_out = model(x)
            visibility_out = visibility_out.squeeze(-1)

            invisible_coords_loss_mask = (y[:, :, 2] != 0).unsqueeze(-1)
            coords_out *= invisible_coords_loss_mask

            coords_loss = coordinates_criterion(coords_out, y[:, :, :2])
            scaled_coords_loss = coords_loss * config["coords_loss_weight"]
            visibility_loss = visibility_criterion(visibility_out, y[:, :, 2])
            scaled_visibility_loss = visibility_loss * config["vis_loss_weight"]

            loss = scaled_coords_loss + scaled_visibility_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            mean_oks = getOKS(coords_out, y, bb) / len(x)

            wandb.log({"Train Loss": loss.item() / len(x), "Train OKS": mean_oks, "Train Viz Loss (scaled)": scaled_visibility_loss.item() / len(x), "Train Coords Loss (scaled)": scaled_coords_loss.item() / len(x)}, step=iteration)
            pbar.update(1)
            iteration += 1

        val_loss, val_oks = evaluate(model, val_loader)
        wandb.log({"Val Loss": val_loss, "Val OKS": val_oks}, step=iteration)

        if val_oks > best_val:
            best_val = val_oks
            torch.save(model.state_dict(),"chkpts/" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + "_epoch_" + str(epoch))

        scheduler.step()

    wandb.finish()
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description='Training script with hyperparameters')
    parser.add_argument('--swin_lr', type=float, default=config['swin_lr'], help='Learning rate for SWIN')
    parser.add_argument('--mlp_lr', type=float, default=config['mlp_lr'])

    # Parse arguments
    args = parser.parse_args()

    config["swin_lr"] = args.swin_lr
    config["mlp_lr"] = args.mlp_lr

    train_loader, val_loader, _ = getMPIIDataloaders(config)

    model = SwinKeypointDetector(
        config["num_keypoints"],
        config["head_hidden_dim"],
        config["swin_variant"],
        config["pretrained"]
    )

    torch.compile(model)

    train(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
