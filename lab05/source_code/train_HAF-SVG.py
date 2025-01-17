import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio

from dataset import bair_robot_pushing_dataset
from models.lstm import lstm, HIM
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, finn_eval_seq

torch.backends.cudnn.benchmark = True

def kl_criterion(mu1, logvar1, mu2, logvar2, args):
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / args.batch_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='../logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='../data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.99, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.1, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing during training (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=29, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--c_dim', type=int, default=7, help='dimensionality of condtion embedding')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=32, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=False, action='store_true')  
    parser.add_argument('--group', type=int, default=2, help='number of groups in Hierarchical Inference Module')  

    args = parser.parse_args()
    return args

def train(x, cond, modules, optimizer, kl_anneal, args, device):
    '''
    x       : (batch_size, seq_num, C, H, W)
    cond    : (batch_size, seq_num, 4 + 3)
    '''
    modules['frame_predictor'].zero_grad()
    modules['posterior'].zero_grad()
    modules['prior'].zero_grad()
    modules['encoder'].zero_grad()
    modules['decoder'].zero_grad()
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].init_hidden()
    modules['prior'].init_hidden()
    mse: torch.Tensor = 0
    kld: torch.Tensor = 0
    y_pred: torch.tensor
    x = x.to(device)
    cond = cond.to(device)
    h_seq = [modules['encoder'](x[:, i]) for i in range(args.n_past + args.n_future + 1)]
    beta = kl_anneal.get_beta()
    for i in range(1, args.n_past + args.n_future):
        use_teacher_forcing = True if random.random() < args.tfr else False
        if i <= args.n_past or use_teacher_forcing:
            h, skip = h_seq[i - 1]
        else:
            h, skip = modules['encoder'](y_pred)
        z_post, mu_post, logvar_post = modules['posterior'](h_seq[i][0])
        z_pri, mu_pri, logvar_pri = modules['prior'](h)
        # TODO : embedding condition
        g = modules['frame_predictor'](torch.cat((h, z_post, cond[:, i - 1]), 1))
        y_pred = modules['decoder']([g, skip])

        mse += F.mse_loss(y_pred, x[:, i])
        kld += kl_criterion(mu_post, logvar_post, mu_pri, logvar_pri, args)
        
    beta = kl_anneal.get_beta()
    loss: torch.Tensor = mse + kld * beta
    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy() / (args.n_past + args.n_future), mse.detach().cpu().numpy() / (args.n_past + args.n_future), kld.detach().cpu().numpy() / (args.n_future + args.n_past)

def pred(validate_seq, validate_cond, modules, args, device):
    y_pred: torch.Tensor
    pred_seq = [torch.rand((args.batch_size, 3, 64, 64))]
    validate_seq = validate_seq.to(device)
    validate_cond = validate_cond.to(device)
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].init_hidden()
    modules['prior'].init_hidden()
    with torch.no_grad():
        for i in range(1, args.n_past + args.n_future):
            if i <= args.n_past:
                h, skip = modules['encoder'](validate_seq[:, i - 1])
                h_target = modules['encoder'](validate_seq[:, i])
                h_target = h_target[0].detach()
                z, _, _ = modules['posterior'](h_target)
            else:
                h, skip = modules['encoder'](pred_seq[-1])
                z, _, _ = modules['prior'](h)
            # TODO : embedding condition
            g = modules['frame_predictor'](torch.cat((h, z, validate_cond[:, i - 1]), 1))
            y_pred = modules['decoder']([g, skip])
            pred_seq.append(y_pred)

    pred_seq = [x.cpu() for x in pred_seq]
    return torch.stack(pred_seq, dim=1)


def plot_pred(validate_seq, validate_cond, modules, epoch, args, device):
    pred_seq = pred(validate_seq, validate_cond, modules, args, device)
    dir = '%s/gen/epoch\=%d/' % (args.log_dir, epoch)
    os.makedirs(dir, exist_ok=True)
    # change pred seq dimesion from (batch_size, 30, C, H. W) -> (batch_size, 30, H, W, C)
    pred_seq = pred_seq.permute(0, 1, 3, 4, 2).numpy()
    for i in range(pred_seq.shape[0]):
        frames = [(pred_seq[i, j] * 255).astype('uint8') for j in range(1, pred_seq.shape[1])]
        imageio.mimsave(f'{dir}batch_size\={i}.gif', frames, duration=0.1)

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        self.cyclical = args.kl_anneal_cyclical
        self.beta = self.frange_cycle_linear(args.niter, start=0.0, stop=1.0,  n_cycle=args.kl_anneal_cycle, ratio=args.kl_anneal_ratio)
        self.idx = 0

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
        # ref: https://github.com/haofuml/cyclical_annealing
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and int(i + c * period) < n_iter:
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
    
    def update(self):
        self.idx += 1
    
    def get_beta(self):
        return self.beta[self.idx]


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        assert False, 'Using CPU.'
    
    assert args.n_past + args.n_future <= 29 and args.n_eval <= 29 # cannot predict first frame ?
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        niter = args.niter
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = '%s/continued' % args.log_dir
        start_epoch = saved_model['last_epoch']
    else:
        name = 'HAF-SVG-rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f-anneal_cycle=%d-anneal_ratio=%.2f-tfr_decay_step=%.3f'\
            % (args.rnn_size, args.predictor_rnn_layers, args.posterior_rnn_layers, args.n_past, args.n_future, args.lr, args.g_dim, args.z_dim, args.last_frame_skip, args.beta, args.kl_anneal_cycle, args.kl_anneal_ratio, args.tfr_decay_step)

        args.log_dir = '%s/%s' % (args.log_dir, name)
        niter = args.niter
        start_epoch = 1

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
        os.remove('./{}/train_record.txt'.format(args.log_dir))
    
    print(args)

    with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------

    if args.model_dir != '':
        frame_predictor = saved_model['frame_predictor']
        posterior = saved_model['posterior']
        prior = saved_model['prior']
    else:
        frame_predictor = lstm(args.g_dim + args.z_dim * args.group + args.c_dim, args.g_dim, args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        posterior = HIM(args.g_dim, args.z_dim, args.rnn_size, args.posterior_rnn_layers, args.group, args.batch_size, device)
        prior = HIM(args.g_dim, args.z_dim, args.rnn_size, args.prior_rnn_layers, args.group, args.batch_size, device)
        frame_predictor.apply(init_weights)
        posterior.apply(init_weights)
        prior.apply(init_weights)
            
    if args.model_dir != '':
        decoder = saved_model['decoder']
        encoder = saved_model['encoder']
    else:
        encoder = vgg_encoder(args.g_dim)
        decoder = vgg_decoder(args.g_dim)
        encoder.apply(init_weights)
        decoder.apply(init_weights)
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    prior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    train_data = bair_robot_pushing_dataset(args, 'train')
    validate_data = bair_robot_pushing_dataset(args, 'validate')
    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    train_iterator = iter(train_loader)

    validate_loader = DataLoader(validate_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    validate_iterator = iter(validate_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    params = list(frame_predictor.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(posterior.parameters()) + list(prior.parameters())
    optimizer = args.optimizer(params, lr=args.lr, betas=(args.beta1, 0.999))
    kl_anneal = kl_annealing(args)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    progress = tqdm(total=args.niter)
    best_val_psnr = 0
    for epoch in range(start_epoch, start_epoch + niter):
        frame_predictor.train()
        posterior.train()
        prior.train()
        encoder.train()
        decoder.train()

        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0

        for _ in range(args.epoch_size):
            try:
                seq, cond = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                seq, cond = next(train_iterator)
            
            loss, mse, kld = train(seq, cond, modules, optimizer, kl_anneal, args, device)
            epoch_loss += loss
            epoch_mse += mse
            epoch_kld += kld
        
        if epoch >= args.tfr_start_decay_epoch:
            ### Update teacher forcing ratio ###
            # use exponential decay
            args.tfr = max(args.tfr_decay_step ** (epoch - args.tfr_start_decay_epoch), args.tfr_lower_bound)
            # use linear decay
            # args.tfr = max(args.tfr - args.tfr_decay_step, args.tfr_lower_bound)

        kl_anneal.update()
        progress.update(1)
        with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
            train_record.write(('[epoch: %02d] loss: %.5f | mse loss: %.5f | kld loss: %.5f\n' % (epoch, epoch_loss  / args.epoch_size, epoch_mse / args.epoch_size, epoch_kld / args.epoch_size)))
        
        frame_predictor.eval()
        encoder.eval()
        decoder.eval()
        posterior.eval()
        prior.eval()

        if epoch % 5 == 0:
            psnr_list = []
            for _ in range(len(validate_data) // args.batch_size):
                try:
                    validate_seq, validate_cond = next(validate_iterator)
                except StopIteration:
                    validate_iterator = iter(validate_loader)
                    validate_seq, validate_cond = next(validate_iterator)

                pred_seq = pred(validate_seq, validate_cond, modules, args, device)
                # change validate and pred seq dimesion from (batch_size, 30, C, H. W) -> (30, batch_size, C, H. W)
                validate_seq = validate_seq.permute((1, 0, 2, 3, 4))
                pred_seq = pred_seq.permute((1, 0, 2, 3, 4))
                _, _, psnr = finn_eval_seq(validate_seq[args.n_past:args.n_past + args.n_future], pred_seq[args.n_past:])
                psnr_list.append(psnr)
                
            ave_psnr = np.mean(np.concatenate(psnr_list))


            with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('====================== validate psnr = {:.5f} ========================\n'.format(ave_psnr)))

            if ave_psnr > best_val_psnr:
                best_val_psnr = ave_psnr
                # save the model
                torch.save({
                    'encoder': encoder,
                    'decoder': decoder,
                    'frame_predictor': frame_predictor,
                    'posterior': posterior,
                    'prior': prior,
                    'args': args,
                    'last_epoch': epoch},
                    '%s/model.pth' % args.log_dir)


if __name__ == '__main__':
    main()
        
