import torch

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("g2net-detecting-continuous-gravitational-waves")+1]))

from src.data_for_torch import G2NETDataset


def main():
    train_set = G2NETDataset()
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
    )
    for i_batch, (x1, x2, y) in enumerate(train_loader):
        print(x1.shape, x2.shape, y.shape)


if __name__ == "__main__":
    main()