from dm_control import suite, viewer
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
import torch.nn.functional as F

import time
import os


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    @info multilayer perceptron (mlp)
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation[j] if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    - Borrowed from OpenAI Spinning Up -

    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MLPGaussianActor(nn.Module):
    """
    - Borrowed from OpenAI -
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.normal.Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, action=None):
        """
        - Borrowed from https://github.com/JhonPool4/ppo_dm_control/blob/master/ppo_utils/policies.py -
        """
        dist = self._distribution(obs)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(dist, action)
        return dist, logp_a


class MLPCritic(nn.Module):
    """
    - Borrowed from OpenAI -
    """
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):
    """
    - Borrowed from OpenAI -
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()

        # policy builder depends on action space
        self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.vf = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            distribution = self.pi._distribution(obs)
            action = distribution.sample()
            logp_a = self.pi._log_prob_from_distribution(distribution, action)
            vf = self.vf(obs)
        return action.numpy(), vf.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.mean_rews = []

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        self.mean_rews.append(np.mean(rews))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # dictionary with training data
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, mean_rews=self.mean_rews)

        # reset mean rews
        self.mean_rews = []
        # as torcch tensor
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class PPO:
    def __init__(self, env, **hyperparameters):

        # update hyperparameters
        self.set_hyperparameters(hyperparameters)

        # get information from environment
        self.env = env
        self.obs_dim = self.env.obs_dim
        self.act_dim = self.env.act_dim

        # create neural network model
        self.ac_model = MLPActorCritic(self.obs_dim, self.act_dim, self.hidden, self.activation)

        # optimizer for policy and value function
        self.pi_optimizer = Adam(self.ac_model.pi.parameters(), self.pi_lr)
        self.vf_optimizer = Adam(self.ac_model.vf.parameters(), self.vf_lr)

        # buffer of training data
        self.buf = PPOBuffer(self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam)

        # logger to print/save data
        self.logger = {'mean_rew': 0, 'std_rew': 0}

        # create directory to save data and model
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)
            print(f"new directory created: {self.training_path}")
        # save training data
        self.column_names = ['mean', 'std']
        self.df = pd.DataFrame(columns=self.column_names, dtype=object)
        if self.create_new_training_data:
            self.df.to_csv(os.path.join(self.training_path, self.data_filename), mode='w', index=False)
            print(f"new data file created: {self.data_filename}")
            # load model
        if self.load_model:
            self.ac_model.load_state_dict(torch.load(os.path.join(self.training_path, self.model_filename)))
            print(f"model loaded: {self.model_filename}")

    def compute_loss_pi(self, data):
        # get specific training data
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # policy loss
        act_dist, logp = self.ac_model.pi(obs, act)  # eval new policy
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        ent = act_dist.entropy().mean().item()
        loss_pi = -(torch.min(ratio * adv, clip_adv) + self.coef_ent * ent).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_vf(self, data):
        # get specific training data
        obs, ret = data['obs'], data['ret']
        # value function loss
        return ((self.ac_model.vf(obs) - ret) ** 2).mean()

    def update(self):
        # get all training data
        data = self.buf.get()

        # logger reward information
        self.logger['mean_rew'] = data['mean_rews'].mean().item()
        self.logger['std_rew'] = data['mean_rews'].std().item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > self.target_kl_scale * self.target_kl:
                # print(f"Early stop at step {i} due to max kl")
                break
            loss_pi.backward()  # compute grads
            self.pi_optimizer.step()  # update parameters

        # Value function learning
        for i in range(self.train_vf_iters):
            self.vf_optimizer.zero_grad()
            loss_vf = self.compute_loss_vf(data)
            loss_vf.backward()  # compute grads
            self.vf_optimizer.step()  # update parameters

    def rollout(self):
        # reset environment parameters
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        epoch_reward = 0

        # generate training data
        for t in range(self.steps_per_epoch):
            # get action, value function and logprob
            a, v, logp = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1
            epoch_reward += r

            # save and log
            self.buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = copy(next_o)  # should be copy

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = self.ac_model.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                self.buf.finish_path(v)

                # reset environment parameters
                o, ep_ret, ep_len = self.env.reset(), 0, 0

        return epoch_reward

    def learn(self):
        # Define lists to store results at every iteration
        mean_r = []
        # stdev_r = []

        for epoch in tqdm(range(self.epochs)):
            # generate data and get total reward for an epoch
            epoch_reward = self.rollout()

            # call update
            self.update()

            # Append results
            # mean_r.append(self.logger['mean_rew'])
            mean_r.append(epoch_reward)
            # stdev_r.append(self.logger['std_rew'])

            # if (epoch + 1) % 10 == 0:
            #     print("\n")
            #     print(f"epochs: {epoch + 1}")
            #     print(f"mean_rew: {self.logger['mean_rew']}")
            #     # print(f"std_ret: {self.logger['std_rew']}")
            #     # print("\n")

            # Plot result and save model every a few steps
            if (epoch + 1) % self.save_freq == 0:
                # Plot reward
                plt.figure()
                plt.plot(mean_r)
                plt.title("Mean Reward")
                plt.xlabel("Epoch")
                plt.ylabel("Mean Reward")
                plt.savefig(os.path.join(self.training_path, "mean_rewards.png"))
                plt.show()

                # # Plot standard deviation reward
                # plt.figure()
                # plt.plot(stdev_r)
                # plt.title("Standard Deviation")
                # plt.xlabel("Epoch")
                # plt.ylabel("Standard Deviation")
                # plt.savefig(os.path.join(self.training_path, "std_deviation.png"))
                # plt.show()

                torch.save(self.ac_model.state_dict(), os.path.join(self.training_path, self.model_filename))
                print("saving model")

            # reset logger
            self.logger = {'rew_mean': 0, 'rew_std': 0}

            # Adjust lr if necessary
            if (epoch + 1) % self.lr_decay_freq == 0:
                self.pi_lr *= self.lr_gamma
                self.vf_lr *= self.lr_gamma
                self.adjust_lr(self.pi_optimizer, self.pi_lr)
                self.adjust_lr(self.vf_optimizer, self.vf_lr)
                print("New pi_lr = %f" % self.pi_lr)
                print("New vf_lr = %f" % self.vf_lr)

    def adjust_lr(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def set_hyperparameters(self, hyperparameters):
        self.epochs = 5000
        self.steps_per_epoch = 1000
        self.max_ep_len = 1000
        self.gamma = 0.99
        self.lam = 0.98
        self.clip_ratio = 0.06
        self.target_kl = 0.01
        self.target_kl_scale = 2.5
        self.coef_ent = 0.001

        self.train_pi_iters = 50
        self.train_vf_iters = 50
        self.pi_lr = 3e-4 * 0.1
        self.vf_lr = 1e-3 * 0.1
        self.lr_gamma = 0.8
        self.lr_decay_freq = 200

        self.hidden = (128, 128, 128, 128)
        self.activation = [nn.Tanh, nn.ReLU, nn.Tanh, nn.ReLU]

        self.flag_render = False

        self.save_freq = 100

        self.training_path = './training/walker'
        self.data_filename = 'data'
        self.model_filename = 'ppo_walker_model.pth'
        self.create_new_training_data = False
        self.load_model = False

        # change default hyperparameters
        for param, val in hyperparameters.items():
            exec("self." + param + "=" + "val")


class WalkerEnv:
    def __init__(self, env):
        self.env = env
        self.act_space = env.action_spec()
        self.obs_space = env.observation_spec()
        self.act_dim = self.act_space.shape[0]
        # ori_shape = (self.obs_space['orientations']).shape
        # hei_shape = (self.obs_space['height']).shape
        # vel_shape = (self.obs_space['velocity']).shape
        self.obs_dim = 14 + 1 + 9

        self.state = self.env.reset()
        self.done = False
        self.step_count = 0

    def get_obs_array(self, obs):
        parsed = np.concatenate([np.ravel(obs[key]) for key in obs])
        return parsed

    def reset(self):
        self.state = self.env.reset()
        self.step_count = 0
        return self.get_obs_array(self.state.observation)

    def get_reward(self, parsed_obs):
        """
        Unparsed observation has 14 elements in orientations, 1 element in height and 9 elements in velocity
        Default reward
        """
        reward = self.state.reward
        # reward += parsed_obs[14] * 0.05
        return reward

    def step(self, action):
        self.step_count += 1
        self.state = self.env.step(action)
        obs = self.get_obs_array(self.state.observation)
        reward = self.get_reward(obs)
        done = self.state.last()
        return obs, reward, done, {}


if __name__ == '__main__':
    # Setup walker environment
    r0 = np.random.RandomState(42)
    # env = suite.load('walker', 'walk', task_kwargs={'random': r0})
    env = suite.load('walker', 'walk', task_kwargs={'random': r0})
    walker_env = WalkerEnv(env)

    # # Retrieve num of dims of observation and input
    U = env.action_spec()
    udim = U.shape[0]
    X = env.observation_spec()
    xdim = 14 + 1 + 9

    # Visualize a random controller
    RANDOM_CTRL_VISUALIZATION_FLAG = False
    if RANDOM_CTRL_VISUALIZATION_FLAG:
        def u(dt): return np.random.uniform(low=U.minimum, high=U.maximum, size=U.shape)
        viewer.launch(env, policy=u)

    # Train
    agent = PPO(walker_env)
    TRAIN_FLAG = False
    RESUME_TRAINING_FLAG = True
    if TRAIN_FLAG:
        if RESUME_TRAINING_FLAG:
            agent.ac_model.load_state_dict(torch.load('./training/walker/ppo_walker_model.pth'))
        agent.learn()

    #####

    VISUALIZE_FLAG = True
    if VISUALIZE_FLAG:
        # Load trained model
        agent.ac_model.load_state_dict(torch.load('./training/walker/ppo_walker_model.pth'))
        agent.ac_model.eval()

        # Define the policy function for DM Control Viewer
        def policy(time_step):
            if time_step.first():
                action = np.zeros(env.action_spec().shape)
            else:
                observation = walker_env.get_obs_array(time_step.observation)
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
                with torch.no_grad():
                    action_tensor, _, _ = agent.ac_model.step(obs_tensor)
                # Check if action_tensor is already a NumPy array or a PyTorch tensor
                if isinstance(action_tensor, np.ndarray):
                    action = action_tensor
                else:
                    action = action_tensor.numpy()  # Convert only if it's a tensor
            return action

        # Launch the viewer
        viewer.launch(env, policy)
