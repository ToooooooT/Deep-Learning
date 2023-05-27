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
from PIL import Image

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import init_weights, kl_criterion, finn_eval_seq

torch.backends.cudnn.benchmark = True

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
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--c_dim', type=int, default=7, help='dimensionality of condtion embedding')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=False, action='store_true')  

    args = parser.parse_args()
    return args

def pred(validate_seq, validate_cond, modules, args, device):
    y_pred: torch.Tensor
    pred_seq = [torch.rand((12, 3, 64, 64))]
    validate_seq = validate_seq.to(device)
    validate_cond = validate_cond.to(device)
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    modules['prior'].hidden = modules['posterior'].init_hidden()
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


def plot_pred(validate_seq, validate_cond, modules, args, device):
    pred_seq = pred(validate_seq, validate_cond, modules, args, device)
    ret = pred_seq
    
    return ret


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        assert False, 'Using CPU.'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        model_dir = args.model_dir
        args = saved_model['args']
        args.model_dir = model_dir
        args.log_dir = '%s/test' % args.log_dir
    else:
        assert False # no model dir argument

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(args)

    # ------------ build the models  --------------

    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'encoder': encoder,
        'decoder': decoder,
    }
    # --------- training loop ------------------------------------

    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()

    psnr_list = []
    for validate_seq, validate_cond in tqdm(test_loader):
        pred_seq = pred(validate_seq, validate_cond, modules, args, device)
        # change validate and pred seq dimesion from (batch_size, 30, C, H. W) -> (30, batch_size, C, H. W)
        validate_seq = validate_seq.permute((1, 0, 2, 3, 4))
        pred_seq = pred_seq.permute((1, 0, 2, 3, 4))
        _, _, psnr = finn_eval_seq(validate_seq[args.n_past:args.n_past + args.n_future], pred_seq[args.n_past:])
        psnr_list.append(psnr)
        
    ave_psnr = np.mean(np.concatenate(psnr_list))

    # plot image and gif
    dir = '%s/gen/' % (args.log_dir)
    os.makedirs(dir, exist_ok=True)
    # change pred seq dimesion from (30, batch_size, C, H, W) -> (batch_size, 30, H, W, C)
    pred_seq = pred_seq.permute(1, 0, 3, 4, 2).numpy()
    validate_seq = validate_seq.permute(1, 0, 3, 4, 2).numpy()
    frames = [(validate_seq[-1, j] * 255).astype(np.uint8) for j in range(args.n_past)] + [(pred_seq[-1, j] * 255).astype(np.uint8) for j in range(args.n_past, pred_seq.shape[1])]
    imageio.mimsave(f'{dir}last.gif', frames, duration=0.1)
    for j in range(pred_seq.shape[1]):
        if j < args.n_past:
            Image.fromarray((validate_seq[-1, j] * 255).astype(np.uint8)).save(f'{dir}last_{j}.png')
        else:
            Image.fromarray((pred_seq[-1, j] * 255).astype(np.uint8)).save(f'{dir}last_{j}.png')

    print(f'test score = {ave_psnr:.5f}')


if __name__ == '__main__':
    main()
        
