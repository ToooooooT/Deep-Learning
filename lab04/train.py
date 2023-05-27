from dataloader import RetinopathyLoader
from Network import ResNet18, ResNet50
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils import data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from argparse import ArgumentParser
import torchvision
from tqdm import tqdm

def train(model: nn.Module, lr=1e-3, batch_size=4, epochs=10, weight_decay=5e-4, name='ResNet18'):
    writer = SummaryWriter(f'./log/tb_{name}')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs // 2)

    train_loader = data.DataLoader(dataset=RetinopathyLoader('./data/new_train/', 'train'), batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=RetinopathyLoader('./data/new_test/', 'test'), batch_size=batch_size, shuffle=False)

    max_acc = 0
    train_acc_all = []
    test_acc_all = []

    for epoch in range(1, epochs + 1):
        # ------------------------------ Training ------------------------------
        model.train()
        train_loss = []
        train_acc = []
        for batch in tqdm(train_loader):
            imgs: torch.Tensor
            labels: torch.Tensor
            imgs, labels = batch
            logits: torch.Tensor = model(imgs.to(device))
            loss: torch.Tensor = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.argmax(dim=-1).to(device)).float().mean()
            train_loss.append(loss)
            train_acc.append(acc)
        lr_scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_acc: torch.Tensor = sum(train_acc) / len(train_acc)
        train_acc_all.append(train_acc.cpu().item())

        # record to tensorboard
        # writer.add_scalars('loss', {'train' : train_loss}, epoch)
        # writer.add_scalars('accuracy', {'train': train_acc}, epoch)
        # writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"[ Train | {epoch:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ------------------------------ Testing ------------------------------
        model.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                imgs, labels = batch
                logits = model(imgs.to(device))
                acc = (logits.argmax(dim=-1) == labels.argmax(dim=-1).to(device)).float().mean()
                test_loss.append(loss)
                test_acc.append(acc)

        test_loss = sum(test_loss) / len(test_loss)
        test_acc: torch.Tensor = sum(test_acc) / len(test_acc)
        test_acc_all.append(test_acc.cpu().item())

        # record to tensorboard
        # writer.add_scalars('loss', {'test' : test_loss}, epoch)
        # writer.add_scalars('accuracy', {'test': test_acc}, epoch)

        print(f"[ Test  | {epoch:03d}/{epochs:03d} ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")

        # save model
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), f'{name}w.dump')
            print('Save model')

        if test_acc >= 0.82:
            torch.save(model.state_dict(), f'{name}w{test_acc:.4f}_{lr}_{batch_size}_{weight_decay}.dump')

    print(f'highest accuracy: {max_acc}')
    np.savez(f'./{name}/w.npz', train=train_acc_all, test=test_acc_all)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="batch size", dest="batch_size", default=4)
    parser.add_argument("--lr", type=float, help="learning rate", dest="lr", default=0.001)
    parser.add_argument("--weight_decay", type=float, help="regularization factor", dest="weight_decay", default=0.0005)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet18_n = ResNet18().to(device)
    # resnet50_n = ResNet50().to(device)
    # resnet18_y = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(device)
    # resnet18_y.fc = nn.Linear(512, 5, bias=True)
    # resnet18_y.to(device)
    # resnet50_y = torchvision.models.resnet50(weights='IMAGENET1K_V2').to(device)
    # resnet50_y.fc = nn.Linear(2048, 5, bias=True)
    # resnet50_y.to(device)

    train(resnet18_n, lr=args.lr, batch_size=args.batch_size, epochs=5, weight_decay=args.weight_decay, name='ResNet18')