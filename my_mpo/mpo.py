import os
from time import sleep
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import gym
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from my_mpo.actor import Actor
from my_mpo.critic import Critic
from my_mpo.replaybuffer import ReplayBuffer


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)

def bt(m):
    return m.transpose(dim0=-2, dim1=-1)

def gaussian_kl(mu_i, mu, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_mu = KL(f(x|mu_i,sigma_i)||f(x|mu,sigma_i))
    C_sigma = KL(f(x|mu_i,sigma_i)||f(x|mu_i,sigma))
    :param mu_i: (B, n)
    :param mu: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_mu, C_sigma: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of sigma_i, sigma
    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)
    sigma_i = Ai @ bt(Ai)  # (B, n, n)
    sigma = A @ bt(A)  # (B, n, n)
    sigma_i_det = sigma_i.det()  # (B,)
    sigma_det = sigma.det()  # (B,)
    sigma_i_det = torch.clamp_min(sigma_i_det, 1e-6)
    sigma_det = torch.clamp_min(sigma_det, 1e-6)
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)

    inner_mu = ((mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)).squeeze()  # (B,)
    inner_sigma = torch.log(sigma_det / sigma_i_det) - n + btr(sigma_inv @ sigma_i)  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_sigma = 0.5 * torch.mean(inner_sigma)
    return C_mu, C_sigma, torch.mean(sigma_i_det), torch.mean(sigma_det)


class MPO(object):
    def __init__(self, env, args):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.device = args.device
        self.eps_daul = args.dual_constraint
        self.eps_mu = args.kl_mean_constraint
        self.eps_gamma = args.kl_var_constraint
        self.gamma = args.discount_factor
        self.alpha_mu_scale = args.alpha_mean_scale # scale Largrangian multiplier
        self.alpha_sigma_scale = args.alpha_var_scale  # scale Largrangian multiplier
        self.alpha_mu_max = args.alpha_mean_max
        self.alpha_sigma_max = args.alpha_var_max

        self.sample_episode_num = args.sample_episode_num
        self.sample_episode_maxstep = args.sample_episode_maxstep
        self.sample_action_num = args.sample_action_num
        self.batch_size = args.batch_size
        self.episode_rerun_num = args.episode_rerun_num
        self.mstep_iteration_num = args.mstep_iteration_num
        self.evaluate_period = args.evaluate_period
        self.evaluate_episode_num = args.evaluate_episode_num
        self.evaluate_episode_maxstep = args.evaluate_episode_maxstep

        self.actor = Actor(env).to(self.device)
        self.critic = Critic(env).to(self.device)
        self.target_actor = Actor(env).to(self.device)
        self.target_critic = Critic(env).to(self.device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.norm_loss_q = nn.MSELoss() if args.q_loss_type == 'mse' else nn.SmoothL1Loss()

        self.replaybuffer = ReplayBuffer()

        self.eta = np.random.rand()
        self.eta_mu = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.eta_sigma = 0.0  # lagrangian multiplier for continuous action space in the M-step
        self.max_return_eval = -np.inf
        self.start_iteration = 1
        self.render = False

    def __sample_trajectory_worker(self, i):
        buff = []
        state = self.env.reset()
        for steps in range(self.sample_episode_maxstep):
            action = self.target_actor.action( torch.from_numpy(state).type(torch.float32).to(self.device)).cpu().numpy()
            next_state, reward, done, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))
            if self.render and i == 0:
                self.env.render(mode='human')
                sleep(0.01)
            if done:
                break
            else:
                state = next_state
        return buff

    def sample_trajectory(self, sample_episode_num):
        self.replaybuffer.clear()
        episodes = [self.__sample_trajectory_worker(i)
                    for i in tqdm(range(sample_episode_num), desc='sample_trajectory')]
        self.replaybuffer.store_episodes(episodes)

    def train(self, iteration_num=1000, log_dir='log', model_save_period=50, render=False):

        self.render = render

        model_save_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.start_iteration, iteration_num + 1):
            self.sample_trajectory(self.sample_episode_num)
            buffer_size = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            max_kl_mu = []
            max_kl_sigma = []
            max_kl = []
            mean_sigma_det = []

            for r in range(self.episode_rerun_num):
                for indices in tqdm(
                        BatchSampler(SubsetRandomSampler(range(buffer_size)), self.batch_size, drop_last=True),
                        desc='training {}/{}'.format(r+1, self.episode_rerun_num)):
                        
                    K = len(indices)  # the sample number of states
                    N = self.sample_action_num  # the sample number of actions per state

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32).to(self.device)  # (K, state_dim)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32).to(self.device)  # (K, action_dim) or (K,)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32).to(self.device)  # (K, state_dim)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32).to(self.device)  # (K,)

                    # Policy Evaluation
                    loss_q, q = self.critic_update_td( state_batch, action_batch, next_state_batch, reward_batch, self.sample_action_num)
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # E-Step of Policy Improvement
                    with torch.no_grad():
                        # sample N actions per state
                        b_mu, b_A = self.target_actor.forward(state_batch)  # (K,)
                        b = MultivariateNormal(b_mu, scale_tril=b_A)  # (K,)
                        sampled_actions = b.sample((N,))  # (N, K, action_dim)
                        expanded_states = state_batch[None, ...].expand(N, -1, -1)  # (N, K, state_dim)
                        target_q = self.target_critic.forward(
                            expanded_states.reshape(-1, self.state_dim), sampled_actions.reshape(-1, self.action_dim)  # (N * K, action_dim)
                        ).reshape(N, K)
                        target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, N)
                        
                    def dual(eta):
                        ## paper version
                        return eta * self.eps_daul + eta * np.mean(np.log(np.mean(np.exp(target_q_np / eta), axis=1)))

                        ## stabilization version: move out max Q(s, a) to avoid overflow
                        # max_q = np.max(target_q_np, 1)
                        # return eta * self.eps_daul + np.mean(max_q) \
                        #     + eta * np.mean(np.log(np.mean(np.exp((target_q_np - max_q[:, None]) / eta), axis=1)))
                    
                    res = minimize(dual, np.array([self.eta]), method='SLSQP', bounds=[(1e-6, None)])
                    self.eta = res.x[0]

                    # normalize
                    norm_target_q = torch.softmax(target_q / self.eta, dim=0)  # (N, K) or (action_dim, K)

                    # M-Step of Policy Improvement
                    for _ in range(self.mstep_iteration_num):
                        mu, A = self.actor.forward(state_batch)

                        # paper1 version
                        policy = MultivariateNormal(loc=mu, scale_tril=A)  # (K,)
                        loss_p = torch.mean( norm_target_q * policy.expand((N, K)).log_prob(sampled_actions))  # (N, K)
                        C_mu, C_sigma, sigma_i_det, sigma_det = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)

                        # paper2 version normalize
                        # π1 = MultivariateNormal(loc=mu, scale_tril=b_A)  # (K,)
                        # π2 = MultivariateNormal(loc=b_mu, scale_tril=A)  # (K,)
                        # loss_p = torch.mean(
                        #     norm_target_q * (
                        #         π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                        #         + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                        #     )
                        # )
                        # C_mu, C_sigma, sigma_i_det, sigma_det = gaussian_kl( mu_i=b_mu, mu=mu, Ai=b_A, A=A)
                        
                        mean_loss_p.append((-loss_p).item())
                        max_kl_mu.append(C_mu.item())
                        max_kl_sigma.append(C_sigma.item())
                        mean_sigma_det.append(sigma_det.item())

                        # Update lagrange multipliers by gradient descent
                        self.eta_mu -= self.alpha_mu_scale * (self.eps_mu - C_mu).detach().item()
                        self.eta_sigma -= self.alpha_sigma_scale * (self.eps_gamma - C_sigma).detach().item()

                        self.eta_mu = np.clip(0.0, self.eta_mu, self.alpha_mu_max)
                        self.eta_sigma = np.clip(0.0, self.eta_sigma, self.alpha_sigma_max)

                        self.actor_optimizer.zero_grad()
                        loss_l = -( loss_p + self.eta_mu * (self.eps_mu - C_mu) + self.eta_sigma * (self.eps_gamma - C_sigma))
                        mean_loss_l.append(loss_l.item())
                        loss_l.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.1)
                        self.actor_optimizer.step()
                    
            self.update_target_actor_critic()

            
            self.save_model(it, os.path.join(model_save_dir, 'model_latest.pt'))
            if it % model_save_period == 0:
                self.save_model(it, os.path.join(model_save_dir, 'model_{}.pt'.format(it)))

            ################################### evaluate and save logs #########################################
            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)
            max_kl_mu = np.max(max_kl_mu)
            max_kl_sigma = np.max(max_kl_sigma)
            mean_sigma_det = np.mean(mean_sigma_det)

            print('iteration :', it)
            if it % self.evaluate_period == 0:
                self.actor.eval()
                return_eval = self.evaluate()
                self.actor.train()
                self.max_return_eval = max(self.max_return_eval, return_eval)
                print('  max_return_eval :', self.max_return_eval)
                print('  return_eval :', return_eval)
                writer.add_scalar('max_return_eval', self.max_return_eval, it)
                writer.add_scalar('return_eval', return_eval, it)
            print('  mean return :', mean_return)
            print('  mean reward :', mean_reward)
            print('  mean loss_q :', mean_loss_q)
            print('  mean loss_p :', mean_loss_p)
            print('  mean loss_l :', mean_loss_l)
            writer.add_scalar('mean_return', mean_return, it)
            writer.add_scalar('mean_reward', mean_reward, it)
            writer.add_scalar('loss_q', mean_loss_q, it)
            writer.add_scalar('loss_p', mean_loss_p, it)
            writer.add_scalar('loss_l', mean_loss_l, it)
            writer.add_scalar('mean_q', mean_est_q, it)
            writer.add_scalar('eta', self.eta, it)
            writer.add_scalar('max_kl_mu', max_kl_mu, it)
            writer.add_scalar('max_kl_sigma', max_kl_sigma, it)
            writer.add_scalar('mean_sigma_det', mean_sigma_det, it)
            writer.add_scalar('eta_mu', self.eta_mu, it)
            writer.add_scalar('eta_sigma', self.eta_sigma, it)

            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def critic_update_td(self, state_batch, action_batch, next_state_batch, reward_batch, sample_num=64):
        B = state_batch.size(0)
        with torch.no_grad():

            ## get mean, cholesky from target actor --> to sample from Gaussian
            pi_mean, pi_A = self.target_actor.forward(next_state_batch)  # (B,)
            policy = MultivariateNormal(pi_mean, scale_tril=pi_A)  # (B,)
            sampled_next_actions = policy.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, action_dim)
            expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, state_dim)
            
            ## get expected Q value from target critic
            expected_next_q = self.target_critic.forward(
                expanded_next_states.reshape(-1, self.state_dim),  # (B * sample_num, state_dim)
                sampled_next_actions.reshape(-1, self.action_dim)  # (B * sample_num, action_dim)
            ).reshape(B, sample_num).mean(dim=1)  # (B,)
            
            y = reward_batch + self.gamma * expected_next_q
        self.critic_optimizer.zero_grad()
        t = self.critic( state_batch, action_batch).squeeze()
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, y


    def update_target_actor_critic(self):
        # param(target_actor) <-- param(actor)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # param(target_critic) <-- param(critic)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


    def load_model(self, path=None):
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.start_iteration = checkpoint['iteration'] + 1
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()


    def save_model(self, it, path=None):
        data = {
            'iteration': it,
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(data, path)


    def evaluate(self):
        with torch.no_grad():
            total_rewards = []
            for e in tqdm(range(self.evaluate_episode_num), desc='evaluating'):
                total_reward = 0.0
                state = self.env.reset()
                for s in range(self.evaluate_episode_maxstep):
                    action = self.actor.action(torch.from_numpy(state).type(torch.float32).to(self.device)).cpu().numpy()
                    state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
                total_rewards.append(total_reward)
            return np.mean(total_rewards)
