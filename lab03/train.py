from dataloader import read_bci_data
from Network import EEGNet, DeepConvNet
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label) -> None:
        super().__init__()
        self.dataset = dataset
        self.label = label

    def __getitem__(self, idx):
        label = np.zeros((2,))
        label[int(self.label[idx])] = 1
        return torch.Tensor(self.dataset[idx]), torch.Tensor(label)

    def __len__(self):
        return len(self.dataset)


def train(train_data, train_label, test_data, test_label, lr=1e-2, batch_size=64, epochs=300):
    writer = SummaryWriter(f'tb_EEG')
    model = EEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs // 2)

    train_loader = DataLoader(dataset=Dataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=Dataset(test_data, test_label), batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        # ------------------------------ Training ------------------------------
        model.train()
        train_loss = []
        train_acc = []
        for batch in train_loader:
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels[:, 1].to(device)).float().mean()
            train_loss.append(loss)
            train_acc.append(acc)
        lr_scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)

        # record to tensorboard
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('accuracy', train_acc, epoch)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"[ Train | {epoch:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ------------------------------ Testing ------------------------------
        model.train()
        test_acc = []
        with torch.no_grad():
            for batch in test_loader:
                imgs, labels = batch
                logits = model(imgs.to(device))
                acc = (logits.argmax(dim=-1) == labels[:, 1].to(device)).float().mean()
                test_acc.append(acc)

        test_acc = sum(test_acc) / len(test_acc)

        # record to tensorboard
        writer.add_scalar('accuracy', test_acc, epoch)

        print(f"[ Test | {epoch:03d}/{epochs:03d} ] acc = {test_acc:.5f}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, train_label, test_data, test_label = read_bci_data()

    train(train_data, train_label, test_data, test_label, lr=1e-2, batch_size=64, epochs=300)
