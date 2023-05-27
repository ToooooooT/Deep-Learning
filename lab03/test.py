from dataloader import read_bci_data
from Network import EEGNet, DeepConvNet, ShallowConvNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
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


def test(test_data, test_label, model, batch_size=64):
    test_loader = DataLoader(dataset=Dataset(test_data, test_label), batch_size=batch_size, shuffle=True)
    model.eval()
    test_acc = []
    with torch.no_grad():
        for batch in test_loader:
            imgs, labels = batch
            logits = model(imgs.to(device))
            acc = (logits.argmax(dim=-1) == labels[:, 1].to(device)).float().mean()
            test_acc.append(acc)

    test_acc = sum(test_acc) / len(test_acc)
    return test_acc



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, train_label, test_data, test_label = read_bci_data()

    model = EEGNet()
    model.load_state_dict(torch.load('EEGNet/best.dump', map_location=device))
    model = model.to(device)

    EEG_acc = test(test_data, test_label, model, batch_size=64)
    print(f'EEGNet test accuracy: {EEG_acc * 100:.3f}%')

    model = DeepConvNet(C=2, T=750)
    model.load_state_dict(torch.load('DeepConvNet/best.dump', map_location=device))
    model = model.to(device)

    DeepConvNet_acc = test(test_data, test_label, model, batch_size=16)
    print(f'DeepConvNet test accuracy: {DeepConvNet_acc * 100:.3f}%')

    model = ShallowConvNet()
    model.load_state_dict(torch.load('ShallowConvNet/best.dump', map_location=device))
    model = model.to(device)

    ShallowConvNet_acc = test(test_data, test_label, model, batch_size=16)
    print(f'ShallowConvNet test accuracy: {ShallowConvNet_acc * 100:.3f}%')