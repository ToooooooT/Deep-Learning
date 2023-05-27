import numpy as np
import matplotlib.pyplot as plt

def plot(train_n, test_n, train_y, test_y, name):
    x = np.arange(1, len(train_n) + 1)
    plt.plot(x, train_n, label='Train(w/o pretraining)')
    plt.plot(x, test_n, label='Test(w/o pretraining)')
    plt.plot(x, train_y, label='Train(with pretraining)')
    plt.plot(x, test_y, label='Test(with pretraining)')
    plt.title(f'Result comparision({name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend(loc='best')
    plt.savefig(f'./{name}/comparision')
    plt.close()

if __name__ == '__main__':
    # ResNet18
    ResNet18_w = np.load('./ResNet18/w.npz')
    ResNet18_wo = np.load('./ResNet18/my.npz')
    # ResNet18_my = np.load('./ResNet18/my.npz')
    plot(ResNet18_wo['train'] * 100, ResNet18_wo['test'] * 100, ResNet18_w['train'] * 100, ResNet18_w['test'] * 100, 'ResNet18')

    # ResNet50
    ResNet50_w = np.load('./ResNet50/w.npz')
    ResNet50_wo = np.load('./ResNet50/my.npz')
    # ResNet50_my = np.load('./ResNet50/my.npz')
    plot(ResNet50_wo['train'] * 100, ResNet50_wo['test'] * 100, ResNet50_w['train'] * 100, ResNet50_w['test'] * 100, 'ResNet50')