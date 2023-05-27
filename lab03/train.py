from dataloader import read_bci_data
from Network import EEGNet, DeepConvNet, ShallowConvNet
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparse import ArgumentParser

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


def train(train_data, train_label, test_data, test_label, model, lr=1e-2, batch_size=64, epochs=300, reg_lambda = 0.0015):
    writer = SummaryWriter(f'./log/tb_DeepConvNet')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg_lambda)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs // 2)

    train_loader = DataLoader(dataset=Dataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=Dataset(test_data, test_label), batch_size=batch_size, shuffle=True)

    max_acc = 0
    train_acc_all = []
    test_acc_all = []

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
        train_acc_all.append(train_acc.cpu().item())

        # record to tensorboard
        writer.add_scalars('loss', {'train' : train_loss}, epoch)
        writer.add_scalars('accuracy', {'train': train_acc}, epoch)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"[ Train | {epoch:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ------------------------------ Testing ------------------------------
        model.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for batch in test_loader:
                imgs, labels = batch
                logits = model(imgs.to(device))
                acc = (logits.argmax(dim=-1) == labels[:, 1].to(device)).float().mean()
                test_loss.append(loss)
                test_acc.append(acc)

        test_loss = sum(test_loss) / len(test_loss)
        test_acc = sum(test_acc) / len(test_acc)
        test_acc_all.append(test_acc.cpu().item())

        # record to tensorboard
        writer.add_scalars('loss', {'test' : test_loss}, epoch)
        writer.add_scalars('accuracy', {'test': test_acc}, epoch)

        print(f"[ Test  | {epoch:03d}/{epochs:03d} ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")

        # save model
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), 'model.dump')
            print('Save model')

        if test_acc >= 0.87:
            torch.save(model.state_dict(), f'ShallowConvNet{test_acc:.4f}_{lr}_{batch_size}_{reg_lambda}.dump')

    print(f'highest accuracy: {max_acc}')
    np.savez('./DeepConvNet/LeakyReLU.npz', train=train_acc_all, test=test_acc_all)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", dest="batch_size", default=64)
    parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=0.0007)
    parser.add_argument("--reg_lambda", type=float, help="regularization factor", dest="reg_lambda", default=0.0017)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.autograd.set_detect_anomaly(True)

    train_data, train_label, test_data, test_label = read_bci_data()

    # model = EEGNet().to(device)
    # model = DeepConvNet(C=2, T=750).to(device)
    model = ShallowConvNet().to(device)

    train(train_data, train_label, test_data, test_label, model, lr=args.lr, batch_size=args.batch_size, epochs=800, reg_lambda=args.reg_lambda)
