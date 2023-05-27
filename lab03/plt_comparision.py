import numpy as np
import matplotlib.pyplot as plt

def plot(train_ELU, test_ELU, train_ReLU, test_ReLU, train_LeakyReLU, test_LeakyReLU, name):
    x = np.arange(1, len(train_ELU) + 1)
    plt.plot(x, train_ELU, label='elu_train')
    plt.plot(x, test_ELU, label='elu_test')
    plt.plot(x, train_ReLU, label='relu_train')
    plt.plot(x, test_ReLU, label='relu_test')
    plt.plot(x, train_LeakyReLU, label='leakyrelu_train')
    plt.plot(x, test_LeakyReLU, label='leakyrelu_test')
    plt.title(f'Activation function comparision({name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend(loc='best')
    plt.savefig(f'./{name}/comparision')

if __name__ == '__main__':
    # EEGNet
    # EEG_ELU = np.load('./EEGNet/ELU.npz')
    # EEG_train_ELU = EEG_ELU['train'] * 100
    # EEG_test_ELU = EEG_ELU['test'] * 100
    # EEG_ReLU = np.load('./EEGNet/ReLU.npz')
    # EEG_train_ReLU = EEG_ReLU['train'] * 100
    # EEG_test_ReLU = EEG_ReLU['test'] * 100
    # EEG_LeakyReLU = np.load('./EEGNet/LeakyReLU.npz')
    # EEG_train_LeakyReLU = EEG_LeakyReLU['train'] * 100
    # EEG_test_LeakyReLU = EEG_LeakyReLU['test'] * 100
    # plot(EEG_train_ELU, EEG_test_ELU, EEG_train_ReLU, EEG_test_ReLU, EEG_train_LeakyReLU, EEG_test_LeakyReLU, 'EEGNet')

    # DeepConvNet
    DeepConvNet_ELU = np.load('./DeepConvNet/ELU.npz')
    DeepConvNet_train_ELU = DeepConvNet_ELU['train'] * 100
    DeepConvNet_test_ELU = DeepConvNet_ELU['test'] * 100
    DeepConvNet_ReLU = np.load('./DeepConvNet/ReLU.npz')
    DeepConvNet_train_ReLU = DeepConvNet_ReLU['train'] * 100
    DeepConvNet_test_ReLU = DeepConvNet_ReLU['test'] * 100
    DeepConvNet_LeakyReLU = np.load('./DeepConvNet/LeakyReLU.npz')
    DeepConvNet_train_LeakyReLU = DeepConvNet_LeakyReLU['train'] * 100
    DeepConvNet_test_LeakyReLU = DeepConvNet_LeakyReLU['test'] * 100
    plot(DeepConvNet_train_ELU, DeepConvNet_test_ELU, DeepConvNet_train_ReLU, DeepConvNet_test_ReLU, \
         DeepConvNet_train_LeakyReLU, DeepConvNet_test_LeakyReLU, 'DeepConvNet')
