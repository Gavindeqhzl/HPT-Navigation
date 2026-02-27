# ma_td3_bc

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
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.util import ST
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import logging

import copy

TensorBatch = List[torch.Tensor]

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, device: None):
        super(Actor, self).__init__()

        self.attention = ST() 

        self.net = nn.Sequential(
            nn.Linear(134, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.device = device
        self.kinematics = "holonomic"
        self.multiagent_training = True

        self.max_action = max_action

    def forward(self, observations):

        attention1 = self.attention(observations)
        self_state = observations[:, 0, :6]
        Attention = torch.mean(attention1, dim=1)
        joint_state = torch.cat((self_state, Attention), dim=1)
        
        return self.max_action * self.net(joint_state)

    @torch.no_grad()
    def act_cql(self, state, device):
        state = torch.as_tensor(state.reshape(1, 7, 13), device=device, dtype=torch.float32)
        actions = self(state)
        actions = actions.squeeze()
        vx = actions[0].detach().cpu().numpy()
        vy = actions[1].detach().cpu().numpy()
        return ActionXY(vx, vy)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.attention = ST()

        self.net = nn.Sequential(
            nn.Linear(152, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


    def forward(self, observations_, actions_):
        state = torch.cat([observations_[0], observations_[1], observations_[2]], dim=1)
        attention = self.attention(state)
        Attention = torch.mean(attention, dim=1)

        self_state0 = observations_[0][:, 0, :6]
        self_state1 = observations_[1][:, 0, :6]
        self_state2 = observations_[2][:, 0, :6]

        observations = torch.cat([self_state0, self_state1, self_state2, Attention], dim=1)
        actions = torch.cat([actions_[0], actions_[1], actions_[2]], dim=-1)

        sa = torch.cat([observations, actions], 1)

        return self.net(sa)


class MATD3BC:
    def __init__(
        self,
        max_action: float,

        actor0: nn.Module,
        actor0_optimizer: torch.optim.Optimizer,
        critic10: nn.Module,
        critic10_optimizer: torch.optim.Optimizer,
        critic20: nn.Module,
        critic20_optimizer: torch.optim.Optimizer,

        actor1: nn.Module,
        actor1_optimizer: torch.optim.Optimizer,
        critic11: nn.Module,
        critic11_optimizer: torch.optim.Optimizer,
        critic21: nn.Module,
        critic21_optimizer: torch.optim.Optimizer,


        actor2: nn.Module,
        actor2_optimizer: torch.optim.Optimizer,
        critic12: nn.Module,
        critic12_optimizer: torch.optim.Optimizer,
        critic22: nn.Module,
        critic22_optimizer: torch.optim.Optimizer,


        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 0.4,
        device: str = "cpu",
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        
    ):
        self.actor0 = actor0
        self.actor0_target = copy.deepcopy(actor0).requires_grad_(False).to(device)
        self.actor0_optimizer = actor0_optimizer
        self.actor0_lr_schedule = StepLR(self.actor0_optimizer, step_size=50000, gamma=0.5)
        # self.actor0_lr_schedule = CosineAnnealingLR(self.actor0_optimizer, 250000)
        # self.scheduler = StepLR(self.reward_optimizer, step_size=100000, gamma=0.9)
        self.critic10 = critic10
        self.critic10_target = copy.deepcopy(critic10).requires_grad_(False).to(device)
        self.critic10_optimizer = critic10_optimizer
        self.critic10_lr_schedule = StepLR(self.critic10_optimizer, step_size=100000, gamma=0.5)
        self.critic20 = critic20
        self.critic20_target = copy.deepcopy(critic20).requires_grad_(False).to(device)
        self.critic20_optimizer = critic20_optimizer
        self.critic20_lr_schedule = StepLR(self.critic20_optimizer, step_size=100000, gamma=0.5)


        self.actor1 = actor1
        self.actor1_target = copy.deepcopy(actor1).requires_grad_(False).to(device)
        self.actor1_optimizer = actor1_optimizer
        self.actor1_lr_schedule = StepLR(self.actor1_optimizer, step_size=50000, gamma=0.5)
        # self.actor1_lr_schedule = CosineAnnealingLR(self.actor1_optimizer, 250000)
        # self.scheduler = StepLR(self.reward_optimizer, step_size=100000, gamma=0.9)
        self.critic11 = critic11
        self.critic11_target = copy.deepcopy(critic11).requires_grad_(False).to(device)
        self.critic11_optimizer = critic11_optimizer
        self.critic11_lr_schedule = StepLR(self.critic11_optimizer, step_size=100000, gamma=0.5)
        self.critic21 = critic21
        self.critic21_target = copy.deepcopy(critic21).requires_grad_(False).to(device)
        self.critic21_optimizer = critic21_optimizer
        self.critic21_lr_schedule = StepLR(self.critic21_optimizer, step_size=100000, gamma=0.5)


        self.actor2 = actor2
        self.actor2_target = copy.deepcopy(actor2).requires_grad_(False).to(device)
        self.actor2_optimizer = actor2_optimizer
        self.actor2_lr_schedule = StepLR(self.actor2_optimizer, step_size=50000, gamma=0.5)
        # self.actor2_lr_schedule = CosineAnnealingLR(self.actor2_optimizer, 250000)
        # self.scheduler = StepLR(self.reward_optimizer, step_size=100000, gamma=0.9)
        self.critic12 = critic12
        self.critic12_target = copy.deepcopy(critic12).requires_grad_(False).to(device)
        self.critic12_optimizer = critic12_optimizer
        self.critic12_lr_schedule = StepLR(self.critic12_optimizer, step_size=100000, gamma=0.5)
        self.critic22= critic22
        self.critic22_target = copy.deepcopy(critic22).requires_grad_(False).to(device)
        self.critic22_optimizer = critic22_optimizer
        self.critic22_lr_schedule = StepLR(self.critic22_optimizer, step_size=100000, gamma=0.5)


        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.grad = False
        

        self.total_it = 0
        self.device = device
       
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max


        self.alpha = 1.0

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    def mask_mse_loss(self, a, b, mask):
        mse_loss = (a - b) ** 2  # 计算平方误差
        masked_mse_loss = mse_loss * mask  # 只计算有效部分
        # Sum over the masked MSE and normalize by the number of valid (non-masked) elements
        valid_mse_loss = masked_mse_loss.sum() / mask.sum()

        return valid_mse_loss


    def update_critic(self, observations_, actions_, next_observations_, reward, global_reward, done, mask):

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise0 = (torch.randn_like(actions_[0]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            noise1 = (torch.randn_like(actions_[1]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            noise2 = (torch.randn_like(actions_[2]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action0 = (self.actor0_target(next_observations_[0]) + noise0).clamp(-self.max_action, self.max_action)
            next_action1 = (self.actor1_target(next_observations_[1]) + noise1).clamp(-self.max_action, self.max_action)
            next_action2 = (self.actor2_target(next_observations_[2]) + noise2).clamp(-self.max_action, self.max_action)
            next_action = [next_action0, next_action1, next_action2]

            # Compute the target Q value
            target_q0 = torch.min(self.critic10_target(next_observations_, next_action), self.critic20_target(next_observations_, next_action))
            target_q0 = reward[0] + (1 - done) * self.discount * target_q0

            target_q1 = torch.min(self.critic11_target(next_observations_, next_action), self.critic21_target(next_observations_, next_action))
            target_q1 = reward[1] + (1 - done) * self.discount * target_q1

            target_q2 = torch.min(self.critic12_target(next_observations_, next_action), self.critic22_target(next_observations_, next_action))
            target_q2 = reward[0] + (1 - done) * self.discount * target_q2


        current_q10 = self.critic10(observations_, actions_)
        current_q20 = self.critic20(observations_, actions_)
        
        current_q11 = self.critic11(observations_, actions_)
        current_q21 = self.critic21(observations_, actions_)
        
        current_q12 = self.critic12(observations_, actions_)
        current_q22 = self.critic22(observations_, actions_)

    
        critic_loss0 = self.mask_mse_loss(current_q10, target_q0, mask) + self.mask_mse_loss(current_q20, target_q0, mask)
        critic_loss1 = self.mask_mse_loss(current_q11, target_q1, mask) + self.mask_mse_loss(current_q21, target_q1, mask)
        critic_loss2 = self.mask_mse_loss(current_q12, target_q2, mask) + self.mask_mse_loss(current_q22, target_q2, mask)

        # Optimize the critic
        self.critic10_optimizer.zero_grad()
        self.critic20_optimizer.zero_grad()
        critic_loss0.backward()
        self.critic10_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.critic10_lr_schedule.step()
        self.critic20_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.critic20_lr_schedule.step()

        self.critic11_optimizer.zero_grad()
        self.critic21_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic11_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.critic11_lr_schedule.step()
        self.critic21_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.critic21_lr_schedule.step()

        self.critic12_optimizer.zero_grad()
        self.critic22_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic12_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.critic12_lr_schedule.step()
        self.critic22_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.critic22_lr_schedule.step()


    def update_actor(self, observations_, actions_, mask):

        pi00 = self.actor0(observations_[0])
        with torch.no_grad():
            pi01 = self.actor1(observations_[1])
            pi02 = self.actor2(observations_[2])
        pi0 = [pi00, pi01, pi02]
        q0 = self.critic10(observations_, pi0)

        pi11 = self.actor1(observations_[1])
        with torch.no_grad():
            pi10 = self.actor0(observations_[0])
            pi12 = self.actor2(observations_[2])
        pi1 = [pi10, pi11, pi12]
        q1 = self.critic11(observations_, pi1)


        pi22 = self.actor2(observations_[2])
        with torch.no_grad():
            pi21 = self.actor1(observations_[1])
            pi20 = self.actor0(observations_[0])
        pi2 = [pi20, pi21, pi22]
        q2 = self.critic12(observations_, pi2) 

        masked_q0 = q0 * mask  # Masking invalid samples in the batch
        masked_q1 = q1 * mask
        masked_q2 = q2 * mask

        # adv = (masked_q0 + masked_q1 + masked_q2 - mc_return).clamp(0, 10)
        # - (adv.detach() * masked_q2).sum() / masked_q2.abs().sum().detach()

        actor_loss0 = -masked_q0.sum() / masked_q0.abs().sum().detach() * 2.5 + self.mask_mse_loss(actions_[0], pi0[0], mask) * 0.4  
        actor_loss1 = -masked_q1.sum() / masked_q1.abs().sum().detach() * 2.5 + self.mask_mse_loss(actions_[1], pi1[1], mask) * 0.4 
        actor_loss2 = -masked_q2.sum() / masked_q2.abs().sum().detach() * 2.5 + self.mask_mse_loss(actions_[2], pi2[2], mask) * 0.4  

        
        # Optimize the actor
        self.actor0_optimizer.zero_grad()
        actor_loss0.backward()
        self.actor0_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.actor0_lr_schedule.step()

        self.actor1_optimizer.zero_grad()
        actor_loss1.backward()
        self.actor1_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.actor1_lr_schedule.step()

        self.actor2_optimizer.zero_grad()
        actor_loss2.backward()
        self.actor2_optimizer.step()
        # if self.grad and self.total_it <= 250000:
        self.actor2_lr_schedule.step()

        self.soft_update(self.critic10_target, self.critic10, self.tau)
        self.soft_update(self.critic20_target, self.critic20, self.tau)
        self.soft_update(self.actor0_target, self.actor0, self.tau)

        self.soft_update(self.critic11_target, self.critic11, self.tau)
        self.soft_update(self.critic21_target, self.critic21, self.tau)
        self.soft_update(self.actor1_target, self.actor1, self.tau)

        self.soft_update(self.critic12_target, self.critic12, self.tau)
        self.soft_update(self.critic22_target, self.critic22, self.tau)
        self.soft_update(self.actor2_target, self.actor2, self.tau)


    def train_episode(self, memorys):
        self.total_it += 1
        # 取轨迹的话才是下面的格式
        # ma_obs      <-> b, max_t, n_agent, s_dim
        # ma_actions  <-> b, max_t, n_agent, a_dim
        # rewards     <-> b, max_t, n_agent, 1
        # ma_next_obs <-> b, max_t, n_agent, s_dim
        # dones       <-> b, max_t, 1
        # traj_mask   <-> b, max_t, 1
        ma_obs, ma_actions, ma_rewards, ma_next_obs, dones, traj_mask = memorys

        ma_obs_list = [ma_obs[:,i,:].reshape(-1, 7, 13) for i in range(3)]

        ma_actions_list = [ma_actions[:,i,:].reshape(-1, 2) for i in range(3)]

        ma_next_obs_list = [ma_next_obs[:,i,:].reshape(-1, 7, 13) for i in range(3)]

        ma_rewards = [ma_rewards[:,i,:].reshape(-1, 1) for i in range(3)]

        dones = dones.reshape(-1, 1)

        traj_mask = traj_mask.reshape(-1, 1)

        total_reward = ma_rewards[0] + ma_rewards[1] + ma_rewards[2]

        pre_rewards = [ma_rewards[0], ma_rewards[1], ma_rewards[2]]

        self.update_critic(ma_obs_list, ma_actions_list, ma_next_obs_list, pre_rewards, total_reward, dones, traj_mask)

        if self.total_it % 2 == 0:
            self.update_actor(ma_obs_list, ma_actions_list, traj_mask)
        






    



