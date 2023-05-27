import argparse
import itertools
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from tqdm import tqdm
import imageio

from dataset import iclevr_dataset
from evaluator import evaluation_model
from models.DDPM import Unet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--beta1', default=0.9, type=int, help='momentum term for adam')
    parser.add_argument('--log_dir', default='../logs', help='base directory to save logs')
    parser.add_argument('--model_dir', default='', help='base directory to save logs')
    parser.add_argument('--data_root', default='../iclevr', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--step', type=int, default=1000, help='diffusion step')
    parser.add_argument('--beta_start', type=int, default=1e-4, help='beta start of noise schedule')
    parser.add_argument('--beta_end', type=int, default=0.02, help='beta end of noise schedule')
    parser.add_argument('--noise_schedule', default='cosine', help='noise schedule')
    parser.add_argument('--pred_type', default='simplified_noise', help='prediction type')
    parser.add_argument('--unet_block_type', default='resnet', help='Down/upsampling blocks selection in Unet')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of residual blocks in the Unet')
    parser.add_argument('--model_channel', type=int, default=128, help='channel of the middle layer of the model')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--num_workers', type=int, default=32, help='number of data loading threads')
    parser.add_argument('--cuda', default=False, action='store_true')  
    parser.add_argument('--train_json', default='../dataset/train.json', help='train json file')  
    parser.add_argument('--test_json', default='../dataset/test.json', help='test json file')  
    parser.add_argument('--new_test_json', default='../dataset/new_test.json', help='test json file')  
    parser.add_argument('--object_json', default='../dataset/objects.json', help='object json file')  
    parser.add_argument('--eval_freq', type=int, default=4, help='test frequency')  

    args = parser.parse_args()
    return args


def train(img, cond, ddpm: Unet, args, optimizer, noise_schedule, device):
    '''
        label embedding:
            use one hot vector of condtion. 
    '''
    loss: torch.Tensor = torch.tensor(np.array(0), dtype=torch.float32, device=device)
    x_0 = img.to(device)
    cond = cond.to(device)

    t = torch.randint(1, args.step, (img.shape[0], 1))
    alpha_bar_ts = torch.tensor(np.array([noise_schedule.get_alpha_bar(x.item()) for x in t]), device=device, dtype=torch.float32)
    alpha_ts = torch.tensor(np.array([noise_schedule.get_alpha(x.item()) for x in t]), device=device, dtype=torch.float32)

    eps = torch.randn((img.shape[0], 3, img.shape[2], img.shape[3])).to(device)
    x_t = torch.sqrt(alpha_bar_ts).view(-1, 1, 1, 1) * x_0 + torch.sqrt(1.0 - alpha_bar_ts).view(-1, 1, 1, 1) * eps

    if args.pred_type == 'simplified_noise':
        eps_pred = ddpm(x_t, cond, t.to(device))
        loss = nn.functional.mse_loss(eps_pred, eps)
    elif args.pred_type == 'noise':
        eps_pred = ddpm(x_t, cond, t.to(device))
        for i in range(args.batch_size):
            loss += (1 / (2 * alpha_ts[i] * (1.0 - alpha_bar_ts[i])) * nn.functional.mse_loss(eps_pred[i], eps[i]))
        loss /= args.batch_size
    elif args.pred_type == 'noisy_sample':
        mu_pred = ddpm(x_t, cond, t.to(device))
        beta_ts = torch.tensor(np.array([noise_schedule.get_beta(x.item()) for x in t]), device=device, dtype=torch.float32)
        mu = 1 / torch.sqrt(alpha_ts).view(-1, 1, 1, 1) * (x_t - beta_ts.view(-1, 1, 1, 1) / torch.sqrt(1.0 - alpha_bar_ts).view(-1, 1, 1, 1) * eps)
        for i in range(args.batch_size):
            loss += (1 / (2 * beta_ts[i]) * nn.functional.mse_loss(mu_pred[i], mu[i]))
        loss /= args.batch_size
    else:
        raise ValueError(f'Unknown predicting type: {args.pred_type}')

    ddpm.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy()


def pred(cond, ddpm, args, noise_schedule, device):
    '''
        label embedding:
            I use one hot vector of condtion. 
    '''
    x = torch.randn((cond.shape[0], 3, 64, 64)).to(device)
    cond = cond.to(device)
    with torch.no_grad():
        for t in range(args.step, 0, -1):
            alpha = noise_schedule.get_alpha(t)
            alpha_bar = noise_schedule.get_alpha_bar(t)
            beta = noise_schedule.get_beta(t)
            z = torch.randn(x.shape).to(device)
            if args.pred_type == 'simplified_noise' or args.pred_type == 'noise':
                eps = ddpm(x, cond, torch.tensor(np.array(t), dtype=torch.float32, device=device).expand(cond.shape[0], 1))
                x = 1 / (alpha ** 0.5) * (x - (1.0 - alpha) / ((1.0 - alpha_bar) ** 0.5) * eps) + (beta ** 0.5) * z
            elif args.pred_type == 'noisy_sample':
                mu = ddpm(x, cond, torch.tensor(np.array(t), dtype=torch.float32, device=device).expand(cond.shape[0], 1))
                x = mu + (beta ** 0.5) * z
            else:
                raise ValueError(f'Unknown predicting type: {args.pred_type}')
    return x.detach()

class noise_scheduling():
    def __init__(self, args) -> None:
        if args.noise_schedule == 'linear':
            self.betas = np.linspace(args.beta_start, args.beta_end, args.step, dtype=np.float32)
        elif args.noise_schedule == 'cosine':
            betas = []
            alpha_bar =  lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            max_beta = 0.999
            for i in range(args.step):
                t1 = i / args.step
                t2 = (i + 1) / args.step
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            self.betas = np.array(betas)
        else:
            raise ValueError(f'Unknown noise schedule: {args.noise_schedule}')

        self.alphas = 1.0 - np.array(self.betas)
        self.alpha_bars = np.cumprod(self.alphas, axis=0)
    
    def get_alpha_bar(self, t):
        return self.alpha_bars[t - 1]

    def get_alpha(self, t):
        return self.alphas[t - 1]

    def get_beta(self, t):
        return self.betas[t - 1]


def main ():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        assert False, 'Using CPU.'

    if args.model_dir != '':
        # load model and continue training from checkpoint
        saved_model = torch.load('%s/model.pth' % args.model_dir)
        optimizer = args.optimizer
        model_dir = args.model_dir
        epochs = args.epochs
        args = saved_model['args']
        args.optimizer = optimizer
        args.model_dir = model_dir
        args.log_dir = f'{args.log_dir}/continued'
        start_epoch = saved_model['last_epoch']
    else:
        name = f'lr={args.lr}-batch_size={args.batch_size}-noise_schedule={args.noise_schedule}-pred_type={args.pred_type}-unet_block_type={args.unet_block_type}-num_res_blocks={args.num_res_blocks}-dropout={args.dropout}-model_channel{args.model_channel}'
        args.log_dir = f'{args.log_dir}/{name}'
        epochs = args.epochs
        start_epoch = 1

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('%s/gen/' % args.log_dir, exist_ok=True)

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    if os.path.exists(f'./{args.log_dir}/train_record.txt'):
        os.remove(f'./{args.log_dir}/train_record.txt')
    
    print(args)

    with open(f'./{args.log_dir}/train_record.txt', 'a') as train_record:
        train_record.write('args: {}\n'.format(args))

    # ------------ build the models  --------------
    if args.model_dir != '':
        ddpm = saved_model['ddpm']
    else:
        ddpm = Unet(num_res_blocks=args.num_res_blocks,
                    dropout=args.dropout,
                    model_channels=args.model_channel)
        # DDPM.apply(init_weights)
    eval_model = evaluation_model()

    # --------- transfer to device ------------------------------------
    ddpm.to(device)

    # --------- load a dataset ------------------------------------
    train_loader = DataLoader(iclevr_dataset(args, mode='train'),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              num_workers=args.num_workers)

    test_loader = DataLoader(iclevr_dataset(args, mode='test'),
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True,
                              num_workers=args.num_workers)
    new_test_loader = DataLoader(iclevr_dataset(args, mode='new_test'),
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True,
                              num_workers=args.num_workers)


    # --------- optimizers ------------------------------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    optimizer = args.optimizer(ddpm.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    noise_schedule = noise_scheduling(args)

    # --------- training loop ------------------------------------
    best_test_acc = best_new_test_acc = 0
    for epoch in tqdm(range(start_epoch, epochs + 1)):
        ddpm.train()

        epoch_loss = 0

        for img, cond in train_loader:
            loss = train(img, cond, ddpm, args, optimizer, noise_schedule, device)
            epoch_loss += loss

        with open(f'./{args.log_dir}/train_record.txt', 'a') as f:
            f.write(f'[epoch: {epoch:02d}] loss: {epoch_loss:.5f}\n')

        if epoch % args.eval_freq == 0:
            # --------- testing loop ------------------------------------
            ddpm.eval()
            test_acc = new_test_acc = 0
            for cond in test_loader:
                pred_img = pred(cond, ddpm, args, noise_schedule, device)
                test_acc += eval_model.eval(pred_img, cond)

            for cond in new_test_loader:
                pred_new_img = pred(cond, ddpm, args, noise_schedule, device)
                new_test_acc += eval_model.eval(pred_new_img, cond)
            
            avg_test_acc = test_acc / len(test_loader)
            avg_new_test_acc = test_acc / len(new_test_loader)

            with open(f'./{args.log_dir}/train_record.txt', 'a') as f:
                f.write(f'==================== test acc = {avg_test_acc:.5f} | new test acc = {avg_new_test_acc:.5f} ====================\n')

            if avg_test_acc > best_test_acc:
                best_test_acc = avg_test_acc
                torch.save({
                    'ddpm' : ddpm,
                    'args': args,
                    'las_epoch': epoch},
                    f'{args.log_dir}/test_model.pth'
                )
                pred_img = make_grid(pred_img / 2 + 0.5, nrow=8)
                save_image(pred_img, f'./{args.log_dir}/test.jpg')

            if avg_new_test_acc > best_new_test_acc:
                best_new_test_acc = avg_new_test_acc
                torch.save({
                    'ddpm' : ddpm,
                    'args': args,
                    'las_epoch': epoch},
                    f'{args.log_dir}/new_test_model.pth'
                )
                pred_new_img = make_grid(pred_new_img / 2 + 0.5, nrow=8)
                save_image(pred_new_img, f'./{args.log_dir}/new_test.jpg')


if __name__ == '__main__':
    main()