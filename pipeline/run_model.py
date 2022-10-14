import os
os.environ["WANDB_SILENT"] = "true"

import torch
from torch.optim import Adam, SGD
from torch.nn.functional import binary_cross_entropy
import wandb

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("g2net-detecting-continuous-gravitational-waves")+1]))

from src.short_time_fourier_transform import ShortTimeFourierTransform_v3
from src.utils import dataset_split
from model.model_short_time_fourier_transform import NeuralNet_v3
from model.step_epoch import train

def main():
    dataset = ShortTimeFourierTransform_v3()
    train_set, valid_set = dataset_split(dataset, valid_size=CONFIG["valid_size"])
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    ) if len(valid_set.indices) else None
    
    model = NeuralNet_v3().to(DEVICE)
    optimizer = eval(CONFIG["optimizer"])
    scheduler = CONFIG["scheduler"]
    loss_function = CONFIG["loss_function"]

    if REPORTS:
            wandb.init(project="Test-MNIST", entity="guychahine", config=CONFIG, tags=TAGS)
            
    for epoch in range(1, CONFIG['nb_epoch']+1):
        train(
            model,
            DEVICE,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            loss_function,
            [epoch, CONFIG['nb_epoch']],
            LOG_INTERVAL,
        )
    wandb.finish() if wandb.run else None
    
if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    REPORTS = False
    LOG_INTERVAL = 1
    
    CONFIG = {
        "nb_epoch": 50,
        "valid_size": 0,
        "batch_size": 48,
        "optimizer": "SGD(model.parameters(), lr=1e-3)",
        "scheduler": None,
        "loss_function": binary_cross_entropy,
    }
    TAGS = ["Pytorch", "C2DNN_v1", "NeuralNet_v1", "MelSpectrogram_v1"]
    
    main()