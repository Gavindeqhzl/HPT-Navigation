import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution
# from crowd_nav.utils.buffer_cql import ReplayBuffer
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.util import ST
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import logging
from tensorboardX import SummaryWriter

# 这段代码是CTDE形式的critic MACQL
TensorBatch = List[torch.Tensor]


class ReparameterizedTanhGaussian(nn.Module):
    def __init__(
        self, log_std_min: float = -20.0, log_std_max: float = 2.0, no_tanh: bool = False
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return action_sample, log_prob


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def init_module_weights(module: torch.nn.Sequential, orthogonal_init: bool = False):
    # Specific orthgonal initialization for inner layers
    # If orthogonal init is off, we do not change default initialization
    if orthogonal_init:
        for submodule in module[:-1]:
            if isinstance(submodule, nn.Linear):
                nn.init.orthogonal_(submodule.weight, gain=np.sqrt(2))
                nn.init.constant_(submodule.bias, 0.0)

    # Lasy layers should be initialzied differently as well
    if orthogonal_init:
        nn.init.orthogonal_(module[-1].weight, gain=1e-2)
    else:
        nn.init.xavier_uniform_(module[-1].weight, gain=1e-2)

    nn.init.constant_(module[-1].bias, 0.0)

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class distributed_policy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        log_std_multiplier: float = 1.0,
        log_std_offset: float = -1.0,
        orthogonal_init: bool = False,
        no_tanh: bool = False,
    ):
        super().__init__()
        self.observation_dim = 91
        self.action_dim = 2
        self.max_action = max_action
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh
        self.kinematics = "holonomic"
        self.multiagent_training = True
        self.attention = ST()
        # hiddendim+statedim
        self.base_network = nn.Sequential(
            nn.Linear(134, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )

        init_module_weights(self.base_network)

        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)
    
    def forward(
        self,
        observations: torch.Tensor,
        #     false表示有噪声
        deterministic: bool = False,
        repeat: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention1 = self.attention(observations)
        self_state = observations[:, 0, :6]
        Attention = torch.mean(attention1, dim=1)
        joint_state = torch.cat((self_state, Attention), dim=1)
        if repeat is not None:
            joint_state = extend_and_repeat(joint_state, 1, repeat)
        base_network_output = self.base_network(joint_state)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)

        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()

        actions, log_probs = self.tanh_gaussian(mean, log_std, deterministic)

        # actions_norm = torch.norm(actions, p=2, dim=-1, keepdim=True)
        # actions = actions / actions_norm.clamp(min=1.0)  # 防止除以 0

        return self.max_action * actions, log_probs

    @torch.no_grad()
    def act(self, state, device):
        # state = torch.as_tensor(state.reshape(1, 7, 13), device=device, dtype=torch.float32)
        with torch.no_grad():
            # 调用forward
            actions, _ = self(state, True)
            actions = actions.squeeze()
        vx = actions[0].detach().cpu().numpy()
        vy = actions[1].detach().cpu().numpy()
        # print(ActionXY(vx, vy))
        return ActionXY(vx, vy), actions.cpu().data.numpy().flatten()

    # @torch.no_grad()
    # def act_cql(self, state, device):
    #     # state = torch.as_tensor(state.reshape(1, 7, 13), device=device, dtype=torch.float32)
    #     with torch.no_grad():
    #         # 调用forward
    #         actions, _ = self(state, True)
    #         actions = actions.squeeze()
    #     vx = actions[0].detach().cpu().numpy()
    #     vy = actions[1].detach().cpu().numpy()
    #     # print(ActionXY(vx, vy))
    #     return ActionXY(vx, vy)

    @torch.no_grad()
    def act_cql(self, state, device):
        with torch.no_grad():
            actions, _ = self(state, True)
            # 将整个Tensor一次性转移到CPU并转换为Numpy
            cpu_actions = actions.squeeze().cpu().numpy()
        # 在CPU上进行索引
        vx = cpu_actions[0]
        vy = cpu_actions[1]
        return ActionXY(vx, vy)

class ctde_critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = 91
        self.action_dim = 2
        self.orthogonal_init = orthogonal_init
        self.attention = ST()
        layers = [
            nn.Linear(152, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)


    def forward(self, observations_, actions_):
        multiple_actions = False
        batch_size = observations_[0].shape[0]
        state = torch.cat([observations_[0], observations_[1], observations_[2]], dim=1)
        attention = self.attention(state)
        Attention = torch.mean(attention, dim=1)

        self_state0 = observations_[0][:, 0, :6]
        self_state1 = observations_[1][:, 0, :6]
        self_state2 = observations_[2][:, 0, :6]

        observations = torch.cat([self_state0, self_state1, self_state2, Attention], dim=1)
        actions = torch.cat([actions_[0], actions_[1], actions_[2]], dim=-1)

        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)
        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        self.observation_dim = 91
        self.action_dim = 2
        self.orthogonal_init = orthogonal_init
        self.attention = ST()
        layers = [
            nn.Linear(136, 256),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))

        self.network = nn.Sequential(*layers)

        init_module_weights(self.network, orthogonal_init)


    def forward(self, observations, actions):
        multiple_actions = False
        batch_size = observations.shape[0]
        # state = observations
        attention = self.attention(observations)
        Attention = torch.mean(attention, dim=1)

        self_state = observations[:, 0, :6]
        
        observations = torch.cat([self_state, Attention], dim=1)

        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        input_tensor = torch.cat([observations, actions], dim=-1)

        q_values = torch.squeeze(self.network(input_tensor), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values




class Traj_MACQL:
    def __init__(
            self,

            critic_1_0,
            critic_1_0_optimizer,
            critic_2_0,
            critic_2_0_optimizer,

            critic_1_1,
            critic_1_1_optimizer,
            critic_2_1,
            critic_2_1_optimizer,

            critic_1_2,
            critic_1_2_optimizer,
            critic_2_2,
            critic_2_2_optimizer,

            actor0,
            actor1,
            actor2,
            actor0_optimizer,
            actor1_optimizer,
            actor2_optimizer,

            human_num: float,

            target_entropy: float,
            discount: float = 0.99,
            alpha_multiplier: float = 1.0,
            use_automatic_entropy_tuning: bool = True,
            backup_entropy: bool = False,
            policy_lr: bool = 3e-4,
            qf_lr: bool = 3e-4,
            soft_target_update_rate: float = 5e-3,
            bc_steps=0,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_importance_sample: bool = True,
            cql_lagrange: bool = False,
            cql_target_action_gap: float = -1.0,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_max_target_backup: bool = False,
            cql_clip_diff_min: float = -np.inf,
            cql_clip_diff_max: float = np.inf,
            device: str = "cpu",
            
    ):
        super().__init__()

        self.discount = discount
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device
        # self.writer = writer


        self.total_it = 0

        self.critic_1_0 = critic_1_0
        self.critic_2_0 = critic_2_0

        self.target_critic_1_0 = deepcopy(self.critic_1_0).to(device)
        self.target_critic_2_0 = deepcopy(self.critic_2_0).to(device)

        self.critic_1_0_optimizer = critic_1_0_optimizer
        self.critic_2_0_optimizer = critic_2_0_optimizer
        self.critic_1_0_schedule = StepLR(self.critic_1_0_optimizer, step_size=100000, gamma=0.5)
        self.critic_2_0_schedule = StepLR(self.critic_2_0_optimizer, step_size=100000, gamma=0.5)

        self.critic_1_1 = critic_1_1
        self.critic_2_1 = critic_2_1

        self.target_critic_1_1 = deepcopy(self.critic_1_1).to(device)
        self.target_critic_2_1 = deepcopy(self.critic_2_1).to(device)
        

        self.critic_1_1_optimizer = critic_1_1_optimizer
        self.critic_2_1_optimizer = critic_2_1_optimizer
        self.critic_1_1_schedule = StepLR(self.critic_1_1_optimizer, step_size=100000, gamma=0.5)
        self.critic_2_1_schedule = StepLR(self.critic_2_1_optimizer, step_size=100000, gamma=0.5)

        self.critic_1_2 = critic_1_2
        self.critic_2_2 = critic_2_2

        self.target_critic_1_2 = deepcopy(self.critic_1_2).to(device)
        self.target_critic_2_2 = deepcopy(self.critic_2_2).to(device)

        self.critic_1_2_optimizer = critic_1_2_optimizer
        self.critic_2_2_optimizer = critic_2_2_optimizer
        self.critic_1_2_schedule = StepLR(self.critic_1_2_optimizer, step_size=100000, gamma=0.5)
        self.critic_2_2_schedule = StepLR(self.critic_2_2_optimizer, step_size=100000, gamma=0.5)

        self.actor0 = actor0
        self.actor1 = actor1
        self.actor2 = actor2

        self.actor0_optimizer = actor0_optimizer
        self.actor1_optimizer = actor1_optimizer
        self.actor2_optimizer = actor2_optimizer

        self.actor0_schedule = StepLR(self.actor0_optimizer, step_size=100000, gamma=0.5)
        self.actor1_schedule = StepLR(self.actor1_optimizer, step_size=100000, gamma=0.5)
        self.actor2_schedule = StepLR(self.actor2_optimizer, step_size=100000, gamma=0.5)
       
        self.log_alpha0 = Scalar(0.0)
        self.alpha_0_optimizer = torch.optim.Adam(
            self.log_alpha0.parameters(),
            lr=self.policy_lr,)

        self.log_alpha1 = Scalar(0.0)
        self.alpha_1_optimizer = torch.optim.Adam(
            self.log_alpha1.parameters(),
            lr=self.policy_lr,)

        self.log_alpha2 = Scalar(0.0)
        self.alpha_2_optimizer = torch.optim.Adam(
            self.log_alpha2.parameters(),
            lr=self.policy_lr,)

        self.human_num = human_num


    @torch.no_grad()
    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1_0, self.critic_1_0, soft_target_update_rate)
        soft_update(self.target_critic_2_0, self.critic_2_0, soft_target_update_rate)
        soft_update(self.target_critic_1_1, self.critic_1_1, soft_target_update_rate)
        soft_update(self.target_critic_2_1, self.critic_2_1, soft_target_update_rate)
        soft_update(self.target_critic_1_2, self.critic_1_2, soft_target_update_rate)
        soft_update(self.target_critic_2_2, self.critic_2_2, soft_target_update_rate)


    def update_critic(self, observations_, actions_, next_observations_, reward, done, mask=None):
        # proxy_reward0, proxy_reward1, proxy_reward2 = 
        # proxy_reward0, proxy_reward1, proxy_reward2 = proxy_reward, proxy_reward, proxy_reward
        # reward = [proxy_reward, proxy_reward, proxy_reward]
        # reward = [proxy_reward[:, 0].unsqueeze(-1), proxy_reward[:, 1].unsqueeze(-1), proxy_reward[:, 2].unsqueeze(-1)]
        q1_predicted_0 = self.critic_1_0(observations_, actions_)
        q2_predicted_0 = self.critic_2_0(observations_, actions_)

        q1_predicted_1 = self.critic_1_1(observations_, actions_)
        q2_predicted_1 = self.critic_2_1(observations_, actions_)

        q1_predicted_2 = self.critic_1_2(observations_, actions_)
        q2_predicted_2 = self.critic_2_2(observations_, actions_)

        new_next_actions00, _ = self.actor0(next_observations_[0])
        with torch.no_grad():
            new_next_actions10, _ = self.actor1(next_observations_[1])
            new_next_actions20, _ = self.actor2(next_observations_[2])

        new_next_actions11, _ = self.actor1(next_observations_[1])
        with torch.no_grad():
            new_next_actions01, _ = self.actor0(next_observations_[0])
            new_next_actions21, _ = self.actor2(next_observations_[2])

        new_next_actions22, _ = self.actor2(next_observations_[2])
        with torch.no_grad():
            new_next_actions02, _ = self.actor0(next_observations_[0])
            new_next_actions12, _ = self.actor1(next_observations_[1])
            

        new_next_actions0 = [new_next_actions00, new_next_actions10, new_next_actions20]
        new_next_actions1 = [new_next_actions01, new_next_actions11, new_next_actions21]
        new_next_actions2 = [new_next_actions02, new_next_actions12, new_next_actions22]

        target_q_values_0 = torch.min(self.target_critic_1_0(next_observations_, new_next_actions0), self.target_critic_2_0(next_observations_, new_next_actions0))
        target_q_values_0 = target_q_values_0.unsqueeze(-1)

        td_target_0 = reward[0] + (1.0 - done) * self.discount * target_q_values_0.detach()
        td_target_0 = td_target_0.squeeze(-1)
        qf1_loss_0 = self.mask_mse_loss(q1_predicted_0, td_target_0.detach(), mask)
        qf2_loss_0 = self.mask_mse_loss(q2_predicted_0, td_target_0.detach(), mask)

        target_q_values_1 = torch.min(self.target_critic_1_1(next_observations_, new_next_actions1), self.target_critic_2_1(next_observations_, new_next_actions1))
        target_q_values_1 = target_q_values_1.unsqueeze(-1)

        td_target_1 = reward[1] + (1.0 - done) * self.discount * target_q_values_1.detach()
        td_target_1 = td_target_1.squeeze(-1)

        qf1_loss_1 = self.mask_mse_loss(q1_predicted_1, td_target_1.detach(), mask)
        qf2_loss_1 = self.mask_mse_loss(q2_predicted_1, td_target_1.detach(), mask)


        target_q_values_2 = torch.min(self.target_critic_1_2(next_observations_, new_next_actions2), self.target_critic_2_2(next_observations_, new_next_actions2))
        target_q_values_2 = target_q_values_2.unsqueeze(-1)

        td_target_2 = reward[2] + (1.0 - done) * self.discount * target_q_values_2.detach()
        td_target_2 = td_target_2.squeeze(-1)
        qf1_loss_2 = self.mask_mse_loss(q1_predicted_2, td_target_2.detach(), mask)
        qf2_loss_2 = self.mask_mse_loss(q2_predicted_2, td_target_2.detach(), mask)

        #CQL

        batch_size = actions_[0].shape[0]
        action_dim = actions_[0].shape[-1]

        cql_random = [
            actions_[0].new_empty((batch_size, self.cql_n_actions, action_dim), requires_grad=False).uniform_(-1, 1)
            for _ in range(3)
        ]           

        with torch.no_grad():

            cql_current_actions0, cql_current_log_pis0 = self.actor0(observations_[0], repeat=self.cql_n_actions)
            cql_current_actions1, cql_current_log_pis1 = self.actor1(observations_[1], repeat=self.cql_n_actions)
            cql_current_actions2, cql_current_log_pis2 = self.actor2(observations_[2], repeat=self.cql_n_actions)

            cql_next_actions0, cql_next_log_pis0 = self.actor0(next_observations_[0], repeat=self.cql_n_actions)
            cql_next_actions1, cql_next_log_pis1 = self.actor1(next_observations_[1], repeat=self.cql_n_actions)
            cql_next_actions2, cql_next_log_pis2 = self.actor2(next_observations_[2], repeat=self.cql_n_actions)

        cql_current_actions = [cql_current_actions0, cql_current_actions1, cql_current_actions2]

        cql_next_actions = [cql_next_actions0, cql_next_actions1, cql_next_actions2]

        cql_q1_rand_0 = self.critic_1_0(observations_, cql_random)
        cql_q2_rand_0 = self.critic_2_0(observations_, cql_random)
        cql_q1_current_actions_0 = self.critic_1_0(observations_, cql_current_actions)
        cql_q2_current_actions_0 = self.critic_2_0(observations_, cql_current_actions)
        cql_q1_next_actions_0 = self.critic_1_0(observations_, cql_next_actions)
        cql_q2_next_actions_0 = self.critic_2_0(observations_, cql_next_actions)

        random_density = np.log(0.5 ** action_dim)
        cql_cat_q1_0 = torch.cat(
            [cql_q1_rand_0 - random_density, cql_q1_next_actions_0 - cql_next_log_pis0, cql_q1_current_actions_0 - cql_current_log_pis0],dim=1,)

        cql_cat_q2_0 = torch.cat(
            [cql_q2_rand_0 - random_density, cql_q2_next_actions_0 - cql_next_log_pis0, cql_q2_current_actions_0 - cql_current_log_pis0],dim=1,)

        cql_qf1_ood_0 = torch.logsumexp(cql_cat_q1_0 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood_0 = torch.logsumexp(cql_cat_q2_0 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        if mask is not None:
            cql_qf1_diff_0 = (torch.clamp(cql_qf1_ood_0 - q1_predicted_0, self.cql_clip_diff_min, self.cql_clip_diff_max) * mask).sum() / mask.sum()
            cql_qf2_diff_0 = (torch.clamp(cql_qf2_ood_0 - q2_predicted_0, self.cql_clip_diff_min, self.cql_clip_diff_max) * mask).sum() / mask.sum()
        else:
            cql_qf1_diff_0 = torch.clamp(cql_qf1_ood_0 - q1_predicted_0, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
            cql_qf2_diff_0 = torch.clamp(cql_qf2_ood_0 - q2_predicted_0, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        cql_min_qf1_loss_0 = cql_qf1_diff_0 * self.cql_alpha
        cql_min_qf2_loss_0 = cql_qf2_diff_0 * self.cql_alpha

        

        cql_q1_rand_1 = self.critic_1_1(observations_, cql_random)
        cql_q2_rand_1 = self.critic_2_1(observations_, cql_random)
        cql_q1_current_actions_1 = self.critic_1_1(observations_, cql_current_actions)
        cql_q2_current_actions_1 = self.critic_2_1(observations_, cql_current_actions)
        cql_q1_next_actions_1 = self.critic_1_1(observations_, cql_next_actions)
        cql_q2_next_actions_1 = self.critic_2_1(observations_, cql_next_actions)


        random_density = np.log(0.5 ** action_dim)
        cql_cat_q1_1 = torch.cat(
            [cql_q1_rand_1 - random_density, cql_q1_next_actions_1 - cql_next_log_pis1, cql_q1_current_actions_1 - cql_current_log_pis1],dim=1,)
        # print(cql_q1_next_actions_1.shape, cql_next_log_pis1.shape)
        

        cql_cat_q2_1 = torch.cat(
            [cql_q2_rand_1 - random_density, cql_q2_next_actions_1 - cql_next_log_pis1, cql_q2_current_actions_1 - cql_current_log_pis1],dim=1,)

        cql_qf1_ood_1 = torch.logsumexp(cql_cat_q1_1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood_1 = torch.logsumexp(cql_cat_q2_1 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        if mask is not None:
            cql_qf1_diff_1 = (torch.clamp(cql_qf1_ood_1 - q1_predicted_1, self.cql_clip_diff_min, self.cql_clip_diff_max) * mask).sum() / mask.sum()
            cql_qf2_diff_1 = (torch.clamp(cql_qf2_ood_1 - q2_predicted_1, self.cql_clip_diff_min, self.cql_clip_diff_max) * mask).sum() / mask.sum()
        else:
            cql_qf1_diff_1 = torch.clamp(cql_qf1_ood_1 - q1_predicted_1, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
            cql_qf2_diff_1 = torch.clamp(cql_qf2_ood_1 - q2_predicted_1, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        cql_min_qf1_loss_1 = cql_qf1_diff_1 * self.cql_alpha
        cql_min_qf2_loss_1 = cql_qf2_diff_1 * self.cql_alpha

        

        cql_q1_rand_2 = self.critic_1_2(observations_, cql_random)
        cql_q2_rand_2 = self.critic_2_2(observations_, cql_random)
        cql_q1_current_actions_2 = self.critic_1_2(observations_, cql_current_actions)
        cql_q2_current_actions_2 = self.critic_2_2(observations_, cql_current_actions)
        cql_q1_next_actions_2 = self.critic_1_2(observations_, cql_next_actions)
        cql_q2_next_actions_2 = self.critic_2_2(observations_, cql_next_actions)


        random_density = np.log(0.5 ** action_dim)
        cql_cat_q1_2 = torch.cat(
            [cql_q1_rand_2 - random_density, cql_q1_next_actions_2 - cql_next_log_pis2, cql_q1_current_actions_2 - cql_current_log_pis2],dim=1,)

        cql_cat_q2_2 = torch.cat(
            [cql_q2_rand_2 - random_density, cql_q2_next_actions_2 - cql_next_log_pis2, cql_q2_current_actions_2 - cql_current_log_pis2],dim=1,)

        cql_qf1_ood_2 = torch.logsumexp(cql_cat_q1_2 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood_2 = torch.logsumexp(cql_cat_q2_2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        if mask is not None:
            cql_qf1_diff_2 = (torch.clamp(cql_qf1_ood_2 - q1_predicted_2, self.cql_clip_diff_min, self.cql_clip_diff_max) * mask).sum() / mask.sum()
            cql_qf2_diff_2 = (torch.clamp(cql_qf2_ood_2 - q2_predicted_2, self.cql_clip_diff_min, self.cql_clip_diff_max) * mask).sum() / mask.sum()
        else:
            cql_qf1_diff_2 = torch.clamp(cql_qf1_ood_2 - q1_predicted_2, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
            cql_qf2_diff_2 = torch.clamp(cql_qf2_ood_2 - q2_predicted_2, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()


        cql_min_qf1_loss_2 = cql_qf1_diff_2 * self.cql_alpha
        cql_min_qf2_loss_2 = cql_qf2_diff_2 * self.cql_alpha


        qf_loss_0 = qf1_loss_0 + qf2_loss_0 + cql_min_qf1_loss_0 + cql_min_qf2_loss_0
        qf_loss_1 = qf1_loss_1 + qf2_loss_1 + cql_min_qf1_loss_1 + cql_min_qf2_loss_1
        qf_loss_2 = qf1_loss_2 + qf2_loss_2 + cql_min_qf1_loss_2 + cql_min_qf2_loss_2

        

        self.critic_1_0_optimizer.zero_grad()
        self.critic_2_0_optimizer.zero_grad()
        qf_loss_0.backward()
        self.critic_1_0_optimizer.step()
        self.critic_2_0_optimizer.step()
        # if self.total_it < 40000:
        self.critic_1_0_schedule.step()
        self.critic_2_0_schedule.step()


        self.critic_1_1_optimizer.zero_grad()
        self.critic_2_1_optimizer.zero_grad()
        qf_loss_1.backward()
        self.critic_1_1_optimizer.step()
        self.critic_2_1_optimizer.step()
        # if self.total_it < 40000:
        self.critic_1_1_schedule.step()
        self.critic_2_1_schedule.step()


        self.critic_1_2_optimizer.zero_grad()
        self.critic_2_2_optimizer.zero_grad()
        qf_loss_2.backward()
        self.critic_1_2_optimizer.step()
        self.critic_2_2_optimizer.step()
        # if self.total_it < 40000:
        self.critic_1_2_schedule.step()
        self.critic_2_2_schedule.step()

        # self.writer.add_scalar('q1', qf_loss_0.detach().item(), self.total_it)
        # self.writer.add_scalar('q2', qf_loss_1.detach().item(), self.total_it)
        # self.writer.add_scalar('q3', qf_loss_2.detach().item(), self.total_it)

   
    def update_actor_alpha(self, observations_ , mask): 

        # torch.autograd.set_detect_anomaly(True)
        state0 = observations_[0]
        state1 = observations_[1]
        state2 = observations_[2]

        new_actions00, log_pi00 = self.actor0(state0)
        with torch.no_grad():
            new_actions10, _ = self.actor1(state1)
            new_actions20, _ = self.actor2(state2) 

        
        new_actions11, log_pi11 = self.actor1(state1)
        with torch.no_grad():
            new_actions01, _ = self.actor0(state0)
            new_actions21, _ = self.actor2(state2) 

      
        new_actions22, log_pi22 = self.actor2(state2)
        with torch.no_grad():
            new_actions02, _ = self.actor0(state0)
            new_actions12, _ = self.actor1(state1)

        new_actions0 = [new_actions00, new_actions10, new_actions20]
        new_actions1 = [new_actions01, new_actions11, new_actions21]
        new_actions2 = [new_actions02, new_actions12, new_actions22]

        alpha_loss0 = -(self.log_alpha0() * (log_pi00 + self.target_entropy).detach()).mean()
        alpha0 = self.log_alpha0().exp() * self.alpha_multiplier
        
        q_new_actions0 = torch.min(self.critic_1_0(observations_, new_actions0), self.critic_2_0(observations_, new_actions0))
        
        
        """ Policy loss1 """
        alpha_loss1 = -(self.log_alpha1() * (log_pi11 + self.target_entropy).detach()).mean()
        alpha1 = self.log_alpha1().exp() * self.alpha_multiplier

        q_new_actions1 = torch.min(self.critic_1_1(observations_, new_actions1), self.critic_2_1(observations_, new_actions1))
        
       
        alpha_loss2 = -(self.log_alpha2() * (log_pi22 + self.target_entropy).detach()).mean()
        alpha2 = self.log_alpha2().exp() * self.alpha_multiplier

        q_new_actions2 = torch.min(self.critic_1_2(observations_, new_actions2), self.critic_2_2(observations_, new_actions2))

        if mask is not None:

            p_loss0 =  (- q_new_actions0) * mask
            policy_loss0 = p_loss0.sum() / mask.sum()

            p_loss1 =  ( - q_new_actions1) * mask
            policy_loss1 = p_loss1.sum() / mask.sum()

            p_loss2 =  ( - q_new_actions2) * mask
            policy_loss2 = p_loss2.sum() / mask.sum()
        else:
            # p_loss0 =  (- q_new_actions0) * mask
            policy_loss0 = (alpha0 * log_pi00 - q_new_actions0).mean()

            # p_loss1 =  ( - q_new_actions1) * mask
            policy_loss1 = (alpha1 * log_pi11 - q_new_actions1).mean()

            # p_loss2 =  ( - q_new_actions2) * mask
            policy_loss2 = (alpha2 * log_pi22 - q_new_actions2).mean()


        self.alpha_0_optimizer.zero_grad()
        alpha_loss0.backward()
        self.alpha_0_optimizer.step()
        self.actor0_optimizer.zero_grad()
        policy_loss0.backward()
        self.actor0_optimizer.step()
        # if self.total_it < 40000:
        self.actor0_schedule.step()
       
        self.alpha_1_optimizer.zero_grad()
        alpha_loss1.backward()
        self.alpha_1_optimizer.step()
        self.actor1_optimizer.zero_grad()
        policy_loss1.backward()
        self.actor1_optimizer.step()
        # if self.total_it < 40000:
        self.actor1_schedule.step()

        self.alpha_2_optimizer.zero_grad()
        alpha_loss2.backward()
        self.alpha_2_optimizer.step()
        self.actor2_optimizer.zero_grad()
        policy_loss2.backward()
        self.actor2_optimizer.step()
        # if self.total_it < 40000:
        self.actor2_schedule.step()

    def mask_mse_loss(self, a, b, mask=None):
        mse_loss = (a - b) ** 2  # 计算平方误差
        if mask is not None:
            masked_mse_loss = mse_loss * mask  # 只计算有效部分
            # Sum over the masked MSE and normalize by the number of valid (non-masked) elements
            valid_mse_loss = masked_mse_loss.sum() / mask.sum()

            return valid_mse_loss
        else:
            return mse_loss.mean()


    def train_episode(self, memorys):
        self.total_it += 1
        # ma_obs      <-> b, max_t, n_agent, s_dim
        # ma_actions  <-> b, max_t, n_agent, a_dim
        # rewards     <-> b, max_t, 1
        # ma_next_obs <-> b, max_t, n_agent, s_dim
        # dones       <-> b, max_t, 1
        # traj_mask   <-> b, max_t, 1
        ma_obs, ma_actions, ma_rewards, ma_next_obs, dones, traj_mask = memorys
        # ma_obs, ma_actions, rewards, ma_next_obs, dones = memorys
        # print(ma_obs.shape)
        # print(rewards.shape)

        ma_obs_list = [ma_obs[:,i,:].reshape(-1, self.human_num, 13) for i in range(3)]

        ma_actions_list = [ma_actions[:,i,:].reshape(-1, 2) for i in range(3)]

        ma_next_obs_list = [ma_next_obs[:,i,:].reshape(-1, self.human_num, 13) for i in range(3)]

        ma_rewards = [ma_rewards[:,i,:].reshape(-1, 1) for i in range(3)]

        # rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)
        # traj_mask = traj_mask.reshape(-1, 1)

        # total_reward = [rewards, rewards, rewards]

        self.update_actor_alpha(ma_obs_list, None)

        self.update_critic(ma_obs_list, ma_actions_list, ma_next_obs_list, ma_rewards, dones, None)

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        
    





    


