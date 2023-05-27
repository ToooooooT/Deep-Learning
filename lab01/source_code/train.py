from gen import generate_linear, generate_XOR_easy
import numpy as np
import matplotlib.pyplot as plt
from model import Model, ConvModel

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


def plot_lr_curve(loss, name):
    fig, ax = plt.subplots()
    ax.plot(loss)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'{name} learning curve')
    plt.savefig(f'{name}.png')


def train (model, x, y, name, epoch, batch_size):
    # hyperparameter
    losses = []

    for i in range(epoch):
        index = np.random.choice(x.shape[0], batch_size)
        input = x[index, :].T
        y_hat = y[index, :].T
        y_pred = model.forward(input)
        loss = model.cross_entropy_loss(y_pred, y_hat)
        losses.append(loss)
        model.backward()
        model.update()
        if i % 1000 == 0:
            print(f'epoch {i} loss: {loss}')

    plot_lr_curve(losses, name)


def test(model, x, y):
    acc = 0
    total_loss = 0
    n = x.shape[0]
    y_pred = model.forward(x.T)
    for i in range(n):
        acc = acc + 1 if (y_pred[0, i] >= 0.5 and y[i] == 1) or (y_pred[0, i] < 0.5 and y[i] == 0) else acc
        print(f'Iter{i} |   Ground truth: {y[i]}    prediction: {y_pred[0, i]}')
    total_loss = model.cross_entropy_loss(y_pred, y.T) * n
    print(f'loss={total_loss:.5f} accuracy={acc / n * 100:.2f}%\n')
    show_result(x, y.reshape(-1,), y_pred.reshape(-1,))


if __name__ == '__main__':
    linear_x, linear_y = generate_linear()
    XOR_x, XOR_y = generate_XOR_easy()

    # initialize network weights (often small random values)
    linear_model = Model(input_size=2, hidden1_size=32, hidden2_size=32, output_size=1, lr=0.001, activation=0)
    XOR_model = Model(input_size=2, hidden1_size=32, hidden2_size=32, output_size=1, lr=0.01, activation=0)

    print('Linear Classification\n')
    train(linear_model, linear_x, linear_y, 'linear', epoch=40000, batch_size=16)
    test(linear_model, linear_x, linear_y)

    print('XOR Classification\n')
    train(XOR_model, XOR_x, XOR_y, 'XOR', epoch=200000, batch_size=8)
    test(XOR_model, XOR_x, XOR_y)