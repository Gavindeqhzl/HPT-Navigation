# ma_iql (multi-agent implicit Q-learning, per-agent)
import copy
import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.util import ST

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
        # observations: [B, 7, 13]
        attention1 = self.attention(observations)
        self_state = observations[:, 0, :6]
        Attention = torch.mean(attention1, dim=1)
        joint_state = torch.cat((self_state, Attention), dim=1)
        return self.max_action * self.net(joint_state)

    @torch.no_grad()
    def act_cql(self, state, device):
        state = torch.as_tensor(state.reshape(1, 7, 13), device=device, dtype=torch.float32)
        actions = self(state).squeeze()
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

    def forward(self, observations_: TensorBatch, actions_: TensorBatch):
        # observations_ is a list of 3 tensors [B,7,13]; actions_ is list of 3 tensors [B,2]
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

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.attention = ST()
        self.net = nn.Sequential(
            nn.Linear(146, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observations_: TensorBatch):
        # centralized state-value: concat self-states and Attention (no actions)
        state = torch.cat([observations_[0], observations_[1], observations_[2]], dim=1)
        attention = self.attention(state)
        Attention = torch.mean(attention, dim=1)

        self_state0 = observations_[0][:, 0, :6]
        self_state1 = observations_[1][:, 0, :6]
        self_state2 = observations_[2][:, 0, :6]

        obs_feat = torch.cat([self_state0, self_state1, self_state2, Attention], dim=1)  # 6+6+6+128=146
        return self.net(obs_feat)

class MAIQL:
    def __init__(
        self,
        max_action: float,

        actor0: nn.Module,
        actor0_optimizer: torch.optim.Optimizer,
        critic10: nn.Module,
        critic10_optimizer: torch.optim.Optimizer,
        critic20: nn.Module,
        critic20_optimizer: torch.optim.Optimizer,
        value0: nn.Module,
        value0_optimizer: torch.optim.Optimizer,

        actor1: nn.Module,
        actor1_optimizer: torch.optim.Optimizer,
        critic11: nn.Module,
        critic11_optimizer: torch.optim.Optimizer,
        critic21: nn.Module,
        critic21_optimizer: torch.optim.Optimizer,
        value1: nn.Module,
        value1_optimizer: torch.optim.Optimizer,

        actor2: nn.Module,
        actor2_optimizer: torch.optim.Optimizer,
        critic12: nn.Module,
        critic12_optimizer: torch.optim.Optimizer,
        critic22: nn.Module,
        critic22_optimizer: torch.optim.Optimizer,
        value2: nn.Module,
        value2_optimizer: torch.optim.Optimizer,

        discount: float = 0.99,
        tau: float = 0.005,            # target update
        expectile: float = 0.8,        # τ for expectile value regression
        beta: float = 3.0,             # temperature for advantage weights
        device: str = "cpu",
    ):
        # --- Agent 0 ---
        self.actor0 = actor0
        self.actor0_target = copy.deepcopy(actor0).requires_grad_(False).to(device)
        self.actor0_optimizer = actor0_optimizer
        self.actor0_lr_schedule = StepLR(self.actor0_optimizer, step_size=50000, gamma=0.5)

        self.critic10 = critic10
        self.critic10_optimizer = critic10_optimizer
        self.critic10_lr_schedule = StepLR(self.critic10_optimizer, step_size=100000, gamma=0.5)

        self.critic20 = critic20
        self.critic20_optimizer = critic20_optimizer
        self.critic20_lr_schedule = StepLR(self.critic20_optimizer, step_size=100000, gamma=0.5)

        self.value0 = value0
        self.value0_target = copy.deepcopy(value0).requires_grad_(False).to(device)
        self.value0_optimizer = value0_optimizer
        self.value0_lr_schedule = StepLR(self.value0_optimizer, step_size=100000, gamma=0.5)

        # --- Agent 1 ---
        self.actor1 = actor1
        self.actor1_target = copy.deepcopy(actor1).requires_grad_(False).to(device)
        self.actor1_optimizer = actor1_optimizer
        self.actor1_lr_schedule = StepLR(self.actor1_optimizer, step_size=50000, gamma=0.5)

        self.critic11 = critic11
        self.critic11_optimizer = critic11_optimizer
        self.critic11_lr_schedule = StepLR(self.critic11_optimizer, step_size=100000, gamma=0.5)

        self.critic21 = critic21
        self.critic21_optimizer = critic21_optimizer
        self.critic21_lr_schedule = StepLR(self.critic21_optimizer, step_size=100000, gamma=0.5)

        self.value1 = value1
        self.value1_target = copy.deepcopy(value1).requires_grad_(False).to(device)
        self.value1_optimizer = value1_optimizer
        self.value1_lr_schedule = StepLR(self.value1_optimizer, step_size=100000, gamma=0.5)

        # --- Agent 2 ---
        self.actor2 = actor2
        self.actor2_target = copy.deepcopy(actor2).requires_grad_(False).to(device)
        self.actor2_optimizer = actor2_optimizer
        self.actor2_lr_schedule = StepLR(self.actor2_optimizer, step_size=50000, gamma=0.5)

        self.critic12 = critic12
        self.critic12_optimizer = critic12_optimizer
        self.critic12_lr_schedule = StepLR(self.critic12_optimizer, step_size=100000, gamma=0.5)

        self.critic22 = critic22
        self.critic22_optimizer = critic22_optimizer
        self.critic22_lr_schedule = StepLR(self.critic22_optimizer, step_size=100000, gamma=0.5)

        self.value2 = value2
        self.value2_target = copy.deepcopy(value2).requires_grad_(False).to(device)
        self.value2_optimizer = value2_optimizer
        self.value2_lr_schedule = StepLR(self.value2_optimizer, step_size=100000, gamma=0.5)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.beta = beta
        self.total_it = 0
        self.device = device

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    def mask_mse_loss(self, a, b, mask):
        mse_loss = (a - b) ** 2
        masked_mse_loss = mse_loss * mask
        return masked_mse_loss.sum() / (mask.sum() + 1e-9)

    def mask_weighted_mse_loss(self, a, b, mask, weights):
        mse_loss = (a - b) ** 2
        w = weights * mask
        return (mse_loss * w).sum() / (w.sum() + 1e-9)

    # ---------- IQL core updates ----------
    def update_value(self, observations_: TensorBatch, actions_: TensorBatch, mask: torch.Tensor):
        with torch.no_grad():
            q10 = self.critic10(observations_, actions_)
            q20 = self.critic20(observations_, actions_)
            qmin0 = torch.min(q10, q20)

            q11 = self.critic11(observations_, actions_)
            q21 = self.critic21(observations_, actions_)
            qmin1 = torch.min(q11, q21)

            q12 = self.critic12(observations_, actions_)
            q22 = self.critic22(observations_, actions_)
            qmin2 = torch.min(q12, q22)

        # Value losses via expectile regression: L = |tau - I[delta<0]| * delta^2
        v0 = self.value0(observations_)
        delta0 = qmin0 - v0
        weight0 = torch.where(delta0 > 0, self.expectile, 1 - self.expectile)
        loss_v0 = self.mask_mse_loss(torch.sqrt(weight0 + 1e-9) * delta0, torch.zeros_like(delta0), mask)

        v1 = self.value1(observations_)
        delta1 = qmin1 - v1
        weight1 = torch.where(delta1 > 0, self.expectile, 1 - self.expectile)
        loss_v1 = self.mask_mse_loss(torch.sqrt(weight1 + 1e-9) * delta1, torch.zeros_like(delta1), mask)

        v2 = self.value2(observations_)
        delta2 = qmin2 - v2
        weight2 = torch.where(delta2 > 0, self.expectile, 1 - self.expectile)
        loss_v2 = self.mask_mse_loss(torch.sqrt(weight2 + 1e-9) * delta2, torch.zeros_like(delta2), mask)

        # optimize
        self.value0_optimizer.zero_grad()
        loss_v0.backward()
        self.value0_optimizer.step()
        self.value0_lr_schedule.step()

        self.value1_optimizer.zero_grad()
        loss_v1.backward()
        self.value1_optimizer.step()
        self.value1_lr_schedule.step()

        self.value2_optimizer.zero_grad()
        loss_v2.backward()
        self.value2_optimizer.step()
        self.value2_lr_schedule.step()

        # target nets
        self.soft_update(self.value0_target, self.value0, self.tau)
        self.soft_update(self.value1_target, self.value1, self.tau)
        self.soft_update(self.value2_target, self.value2, self.tau)

    def update_critic(self, observations_: TensorBatch, actions_: TensorBatch,
                      next_observations_: TensorBatch, reward: TensorBatch,
                      done: torch.Tensor, mask: torch.Tensor):

        with torch.no_grad():
            # V-target bootstrap
            v0n = self.value0_target(next_observations_)
            v1n = self.value1_target(next_observations_)
            v2n = self.value2_target(next_observations_)

            target_q0 = reward[0] + (1 - done) * self.discount * v0n
            target_q1 = reward[1] + (1 - done) * self.discount * v1n
            target_q2 = reward[2] + (1 - done) * self.discount * v2n

        current_q10 = self.critic10(observations_, actions_)
        current_q20 = self.critic20(observations_, actions_)
        loss_q0 = self.mask_mse_loss(current_q10, target_q0, mask) + self.mask_mse_loss(current_q20, target_q0, mask)

        current_q11 = self.critic11(observations_, actions_)
        current_q21 = self.critic21(observations_, actions_)
        loss_q1 = self.mask_mse_loss(current_q11, target_q1, mask) + self.mask_mse_loss(current_q21, target_q1, mask)

        current_q12 = self.critic12(observations_, actions_)
        current_q22 = self.critic22(observations_, actions_)
        loss_q2 = self.mask_mse_loss(current_q12, target_q2, mask) + self.mask_mse_loss(current_q22, target_q2, mask)

        # optimize critics
        self.critic10_optimizer.zero_grad()
        self.critic20_optimizer.zero_grad()
        loss_q0.backward()
        self.critic10_optimizer.step()
        self.critic10_lr_schedule.step()
        self.critic20_optimizer.step()
        self.critic20_lr_schedule.step()

        self.critic11_optimizer.zero_grad()
        self.critic21_optimizer.zero_grad()
        loss_q1.backward()
        self.critic11_optimizer.step()
        self.critic11_lr_schedule.step()
        self.critic21_optimizer.step()
        self.critic21_lr_schedule.step()

        self.critic12_optimizer.zero_grad()
        self.critic22_optimizer.zero_grad()
        loss_q2.backward()
        self.critic12_optimizer.step()
        self.critic12_lr_schedule.step()
        self.critic22_optimizer.step()
        self.critic22_lr_schedule.step()

    def update_actor(self, observations_: TensorBatch, actions_: TensorBatch, mask: torch.Tensor):
        # weights = exp( (Q_min(s,a_dataset) - V(s)) / beta )
        with torch.no_grad():
            qmin0 = torch.min(self.critic10(observations_, actions_), self.critic20(observations_, actions_))
            adv0 = (qmin0 - self.value0(observations_)) / self.beta
            w0 = torch.exp(adv0).clamp(max=100.0)

            qmin1 = torch.min(self.critic11(observations_, actions_), self.critic21(observations_, actions_))
            adv1 = (qmin1 - self.value1(observations_)) / self.beta
            w1 = torch.exp(adv1).clamp(max=100.0)

            qmin2 = torch.min(self.critic12(observations_, actions_), self.critic22(observations_, actions_))
            adv2 = (qmin2 - self.value2(observations_)) / self.beta
            w2 = torch.exp(adv2).clamp(max=100.0)

        pi0 = self.actor0(observations_[0])
        pi1 = self.actor1(observations_[1])
        pi2 = self.actor2(observations_[2])

        actor_loss0 = self.mask_weighted_mse_loss(pi0, actions_[0], mask, w0)
        actor_loss1 = self.mask_weighted_mse_loss(pi1, actions_[1], mask, w1)
        actor_loss2 = self.mask_weighted_mse_loss(pi2, actions_[2], mask, w2)

        self.actor0_optimizer.zero_grad()
        actor_loss0.backward()
        self.actor0_optimizer.step()
        self.actor0_lr_schedule.step()

        self.actor1_optimizer.zero_grad()
        actor_loss1.backward()
        self.actor1_optimizer.step()
        self.actor1_lr_schedule.step()

        self.actor2_optimizer.zero_grad()
        actor_loss2.backward()
        self.actor2_optimizer.step()
        self.actor2_lr_schedule.step()

        # Update actor targets for evaluation-time stability (not strictly required by IQL)
        self.soft_update(self.actor0_target, self.actor0, self.tau)
        self.soft_update(self.actor1_target, self.actor1, self.tau)
        self.soft_update(self.actor2_target, self.actor2, self.tau)

    def train_episode(self, memorys):
        self.total_it += 1
        # ma_obs      <-> b, n_agent, s_dim (here s_dim=13*7)  -> reshape to [B,7,13]
        # ma_actions  <-> b, n_agent, a_dim
        # rewards     <-> b, n_agent, 1
        # ma_next_obs <-> b, n_agent, s_dim
        # dones       <-> b, 1
        # traj_mask   <-> b, 1
        ma_obs, ma_actions, ma_rewards, ma_next_obs, dones, traj_mask = memorys

        ma_obs_list = [ma_obs[:, i, :].reshape(-1, 7, 13) for i in range(3)]
        ma_actions_list = [ma_actions[:, i, :].reshape(-1, 2) for i in range(3)]
        ma_next_obs_list = [ma_next_obs[:, i, :].reshape(-1, 7, 13) for i in range(3)]
        ma_rewards = [ma_rewards[:, i, :].reshape(-1, 1) for i in range(3)]

        dones = dones.reshape(-1, 1)
        traj_mask = traj_mask.reshape(-1, 1)

        # 1) Value update (expectile regression)
        self.update_value(ma_obs_list, ma_actions_list, traj_mask)

        # 2) Q update (TD with V-target)
        self.update_critic(ma_obs_list, ma_actions_list, ma_next_obs_list, ma_rewards, dones, traj_mask)

        # 3) Policy update (advantage-weighted regression)
        if self.total_it % 2 == 0:
            self.update_actor(ma_obs_list, ma_actions_list, traj_mask)
