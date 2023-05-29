'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class ActorNet(nn.Module):
    def __init__(self, action_space, state_dim=8, action_dim=2, hidden_dim=(512, 256)):
        super().__init__()
        ## TODO ##
        self.l = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim[1], action_dim),
            nn.Tanh()
        )
        self.action_space = action_space

    def forward(self, x):
        ## TODO ##
        return self.l(x) * abs(self.action_space.high[0])


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(512, 256)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)


class DDPG:
    def __init__(self, args, num_inputs, action_space):
        # behavior network
        self._actor_net = ActorNet(action_space, state_dim=num_inputs, action_dim=action_space.shape[0]).to(args.device)
        self._critic_net_1 = CriticNet(state_dim=num_inputs, action_dim=action_space.shape[0]).to(args.device)
        self._critic_net_2 = CriticNet(state_dim=num_inputs, action_dim=action_space.shape[0]).to(args.device)
        # target network
        self._target_actor_net = ActorNet(action_space, state_dim=num_inputs, action_dim=action_space.shape[0]).to(args.device)
        self._target_critic_net_1 = CriticNet(state_dim=num_inputs, action_dim=action_space.shape[0]).to(args.device)
        self._target_critic_net_2 = CriticNet(state_dim=num_inputs, action_dim=action_space.shape[0]).to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net_1.load_state_dict(self._critic_net_1.state_dict())
        self._target_critic_net_2.load_state_dict(self._critic_net_2.state_dict())
        ## TODO ##
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt_1 = optim.Adam(self._critic_net_1.parameters(), lr=args.lrc)
        self._critic_opt_2 = optim.Adam(self._critic_net_2.parameters(), lr=args.lrc)
        # action noise
        self._action_noise = GaussianNoise(dim=action_space.shape[0])
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.action_space = action_space
        self.update_actor_freq = args.update_actor_freq

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        with torch.no_grad():
            mu = self._actor_net(torch.tensor(state, dtype=torch.float))

        if noise:
            mu = mu + self._action_noise.sample()

        return torch.clip(mu, torch.tensor(self.action_space.low), torch.tensor(self.action_space.high)).numpy()


    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self, total_steps):
        # update the behavior networks
        self._update_behavior_network(self.gamma, total_steps)
        # update the target networks
        if total_steps % self.update_actor_freq == 0:
            self._update_target_network(self._target_actor_net, self._actor_net,
                                        self.tau)
            self._update_target_network(self._target_critic_net_1, self._critic_net_1,
                                        self.tau)
            self._update_target_network(self._target_critic_net_2, self._critic_net_2,
                                        self.tau)

    def _update_behavior_network(self, gamma, total_steps):
        actor_net, critic_net_1, critic_net_2, target_actor_net, target_critic_net_1, target_critic_net_2 = self._actor_net, self._critic_net_1, self._critic_net_2, self._target_actor_net, self._target_critic_net_1, self._target_critic_net_2
        actor_opt, critic_opt_1, critic_opt_2 = self._actor_opt, self._critic_opt_1, self._critic_opt_2

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        q_value_1 = critic_net_1(state, action)
        q_value_2 = critic_net_2(state, action)
        with torch.no_grad():
            noise = self._action_noise.sample()
            a_next = torch.clip(target_actor_net(next_state) + noise, torch.tensor(self.action_space.low), torch.tensor(self.action_space.high)).to(torch.float)
            q_next = torch.stack([target_critic_net_1(next_state, a_next), target_critic_net_2(next_state, a_next)], dim=-1).min(dim=-1)[0]
            q_target = reward + gamma * q_next * (1 - done)
        criterion = F.mse_loss
        critic_loss_1 = criterion(q_value_1, q_target.detach())
        critic_loss_2 = criterion(q_value_2, q_target.detach())
        # optimize critic
        actor_net.zero_grad()
        critic_net_1.zero_grad()
        critic_net_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        critic_opt_1.step()
        critic_opt_2.step()

        if total_steps % self.update_actor_freq == 0:
            ## update actor ##
            # actor loss
            ## TODO ##
            action = actor_net(state)
            actor_loss = -critic_net_1(state, action).mean()
            # optimize actor
            actor_net.zero_grad()
            critic_net_1.zero_grad()
            actor_loss.backward()
            actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            target.data.copy_(target.data * (1. - tau) + behavior.data * tau)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt': self._critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic_1': self._critic_net_1.state_dict(),
                    'critic_2': self._critic_net_2.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net_1.load_state_dict(model['critic_1'])
        self._critic_net_2.load_state_dict(model['critic_2'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env, agent:DDPG, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    max_ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                writer.add_scalar('action', action[0],
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
        if ewma_reward > max_ewma_reward:
            max_ewma_reward = ewma_reward
            if max_ewma_reward > 250:
                agent.save(args.model)
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1):
            # select action
            action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                # writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f'episode {n_episode + 1}: {total_reward}')
                break
        rewards.append(total_reward)
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-m', '--model', default='pretrained/TD3.pth')
    parser.add_argument('--logdir', default='log/TD3')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=3000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-4, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    parser.add_argument('--update_actor_freq', default=2, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(args, env.observation_space.shape[0], env.action_space)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
