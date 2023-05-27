from dataloader import RetinopathyLoader
from Network import ResNet18, ResNet50
import torch
import torchvision
from torch.utils import data
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt
from tqdm import tqdm

def test(model: nn.Module, batch_size=64, name='ResNet18'):
    test_loader = data.DataLoader(dataset=RetinopathyLoader('./data/new_test/', 'test'), batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    test_acc = []
    test_label = []
    test_predict = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs: torch.Tensor
            labels: torch.Tensor
            imgs, labels = batch
            logits: torch.Tensor = model(imgs.to(device))
            acc = (logits.argmax(dim=-1) == labels.argmax(dim=-1).to(device)).float().mean()
            test_acc.append(acc)
            test_predict += logits.argmax(dim=-1).view(-1,).tolist()
            test_label += labels.argmax(dim=-1).view(-1,).tolist()

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(test_label, test_predict) / 7025, display_labels=['0', '1', '2', '3', '4'])
    disp.plot()
    plt.title('Normalized Confusion Matrix')
    plt.savefig(f'./{name}/confusion_matrix')
    test_acc = sum(test_acc) / len(test_acc)
    return test_acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet18_wo = ResNet18()
    resnet18_wo.load_state_dict(torch.load('ResNet18/my.dump'))
    resnet18_wo = resnet18_wo.to(device)

    ResNet18wo_acc = test(resnet18_wo, batch_size=4, name='ResNet18')
    print(f'ResNet18 without pretrained test accuracy: {ResNet18wo_acc * 100:.3f}%')

    resnet18_w = torchvision.models.resnet18()
    resnet18_w.fc = nn.Linear(512, 5, bias=True)
    resnet18_w.load_state_dict(torch.load('ResNet18/w.dump'))
    resnet18_w = resnet18_w.to(device)

    ResNet18w_acc = test(resnet18_w, batch_size=4, name='ResNet18')
    print(f'ResNet18 test accuracy: {ResNet18w_acc * 100:.3f}%')

    resnet50_wo = ResNet50()
    resnet50_wo.load_state_dict(torch.load('ResNet50/my.dump'))
    resnet50_wo = resnet50_wo.to(device)

    ResNet50wo_acc = test(resnet50_wo, batch_size=4, name='ResNet50')
    print(f'ResNet50 without pretrained test accuracy: {ResNet50wo_acc * 100:.3f}%')

    resnet50_w = torchvision.models.resnet50()
    resnet50_w.fc = nn.Linear(2048, 5, bias=True)
    resnet50_w.load_state_dict(torch.load('ResNet50/w.dump'))
    resnet50_w = resnet50_w.to(device)

    ResNet50w_acc = test(resnet50_w, batch_size=4, name='ResNet50')
    print(f'ResNet50 test accuracy: {ResNet50w_acc * 100:.3f}%')