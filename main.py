import torch
from torch import nn

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import optim

from load_data import get_dataset
from model import ConvNet
from train import train_epochs

from constants import device

if __name__ == '__main__':
    model = ConvNet()
    model.to(device)
    dataset = get_dataset()

    data_train, data_val = random_split(dataset, [0.8, 0.2])
    print(f'Training samples:{len(data_train)}\nValidation samples:{len(data_val)}\n')

    train_loader = DataLoader(data_train, 16, shuffle=True)
    val_loader = DataLoader(data_val, 16, shuffle=True)

    train_epochs(15, optim.SGD(model.parameters(), lr=1e-2), model, nn.CrossEntropyLoss(), train_loader, val_loader)

