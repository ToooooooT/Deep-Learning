import matplotlib.pyplot as plt
import numpy as np

path = './logs/fp/CCONV-rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0005-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000-anneal_cycle=2-anneal_ratio=0.25-tfr_decay_step=0.990/train_record.txt'
# path = './logs/fp/HAF-SVG-rnn_size=256-predictor-posterior-rnn_layers=2-1-n_past=2-n_future=10-lr=0.0005-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000-anneal_cycle=2-anneal_ratio=0.25-tfr_decay_step=0.990/train_record.txt'

def plt_curve (y, ylabel, title, pathname):
    if ylabel == 'psnr':
        x = np.arange(len(y)) * 5
    else:
        x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.savefig(pathname)
    plt.close()


psnr = []
kl_loss = []
with open(path, 'r') as f:
    for line in f:
        if 'validate' in line:
            psnr.append(float(line.split(' ')[-2]))
        elif 'kld' in line:
            kl_loss.append(float(line.split(' ')[-1]))

plt_curve(psnr, 'psnr', 'psnr curves', './psnr_curves')
plt_curve(kl_loss, 'kl loss', 'kl loss curves', './kl_loss_curves')

