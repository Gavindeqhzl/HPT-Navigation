import sys
import logging
import argparse
import configparser
import time as T
import os
import torch
import numpy as np
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.state import JointState

import torch
import torch.nn as nn

class FE(nn.Module):
    def __init__(self, input_size):
        super(FE, self).__init__()
        self.FE_layer = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()

    def forward(self, input_state):
        #print('fe', input_state)
        x_fe = self.FE_layer(input_state)
        # print('x_fe', x_fe.shape)
        x = self.relu(x_fe)
        # print('x', x.shape)
        return x

class GT(nn.Module):
    def __init__(self):
        super(GT, self).__init__()
        self.LayerNorm1 = nn.LayerNorm(128)
        self.MutilHeadAtten_layer = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)


        self.LayerNorm2 = nn.LayerNorm(128)
        self.FowardFeed_layer = nn.Sequential(nn.Linear(128, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, 128))

    def forward(self, input_state, batch_first=True):
        # print( 'input_state', input_state.shape)
        x_norm = self.LayerNorm1(input_state)
    
        x_atten, _ = self.MutilHeadAtten_layer(x_norm, x_norm, x_norm)  # TODO
        # print('x_atten', x_atten.shape)
        add = x_atten + input_state
        # print('add', add.shape)
        add_norm = self.LayerNorm2(add)
        # print('add_norm', add_norm.shape)
        add_fowardfeed = self.FowardFeed_layer(add_norm)
        # print('add_fowardfeed', add_fowardfeed.shape)
        return add_fowardfeed + add

class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()
        self.input_size = 13
        self.FE = FE(input_size=self.input_size)

        self.ST = GT()


    def forward(self, input_state):
        state_fe = self.FE(input_state)
        state_st = self.ST(state_fe)
        return state_st


def print_time(begin_time, episode, max_episode):
    slop = T.time()
    time = (slop - begin_time) / episode * (max_episode - episode) / 3600
    print('step', episode / 1000, 'k  about %.2f hours still' % time)

def save_models(output_dir, model_name, model):

    model_path = os.path.join(output_dir, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    actor0_weight_file = os.path.join(model_path, "actor0_weight_{}.pth".format(model_name))
    actor1_weight_file = os.path.join(model_path, "actor1_weight_{}.pth".format(model_name))
    actor2_weight_file = os.path.join(model_path, "actor2_weight_{}.pth".format(model_name))

    torch.save(model.actor0.state_dict(), actor0_weight_file)
    torch.save(model.actor1.state_dict(), actor1_weight_file)
    torch.save(model.actor2.state_dict(), actor2_weight_file)
    

@torch.no_grad()
def run_k_episodes(k, env, phase, device, actor1, actor2, actor3, case = None, episode=None):
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
   
    cumulative_reward0 = []
    cumulative_reward1 = []
    cumulative_reward2 = []
   
    collision_cases = []
    timeout_cases = []
    actor1.eval()
    actor2.eval()
    actor3.eval()
    robot_actor_list = [actor1, actor2, actor3]
    noise = False
    for i in range(k):
        
        if case is not None:
            obs = env.reset(phase, test_case = case * k + i)
        else:
            obs = env.reset(phase)
        done = False
        
        robot_rewards = [[] for _ in range(len(env.robots))]
    
        infos = []

        while not done:
           
            robot_actionXYs = []
           
           
            for i in range(len(env.robots)):
                jointstate = JointState(env.robots[i].get_full_state(), obs[i])
                state = transform(jointstate, device)
                state = state.reshape(1, env.human_num+2, 13)
                robot_actionXY = robot_actor_list[i].act_cql(state, device)

                robot_actionXYs.append(robot_actionXY)


            # 返回一个回合的3个机器人的奖励的列表
            obs, rewards, dones, infos = env.step(robot_actionXYs)

            for i in range(len(env.robots)):
                    robot_rewards[i].append(rewards[i])

            if isinstance(infos[0], ReachGoal):
                done = True
            elif any(isinstance(info, Collision) for info in infos):
                done = True
            elif any(isinstance(info, Timeout) for info in infos):
                done = True

        if isinstance(infos[0], ReachGoal):
            success += 1
            
        elif any(isinstance(info, Collision) for info in infos):
            collision += 1
            collision_cases.append(i)
            collision_times.append(env.global_time)
           
        elif any(isinstance(info, Timeout) for info in infos):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.time_limit)
           
        else:
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.time_limit)

        if isinstance(infos[0], ReachGoal):
            time = env.global_time
        else:
            time = 24.25

        success_times.append(time)
        
        cumulative_reward0.append(
            sum([pow(0.9, t * 0.25) * reward for t, reward in enumerate(robot_rewards[0])]))
        cumulative_reward1.append(
            sum([pow(0.9, t * 0.25) * reward for t, reward in enumerate(robot_rewards[1])]))
        cumulative_reward2.append(
            sum([pow(0.9, t * 0.25) * reward for t, reward in enumerate(robot_rewards[2])]))

    success_rate = success / k
    collision_rate = collision / k
   
    assert success + collision + timeout == k
    avg_nav_time = sum(success_times) / len(success_times)

    extra_info = '' if episode is None else 'in episode {} '.format(episode)
    logging.info(
        '{:<5} {}has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.2f}, leader reward: {:.4f}, f1 reward: {:.4f}, f2 reward: {:.4f}'.
        format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
               average(cumulative_reward0), average(cumulative_reward1), average(cumulative_reward2)))

    actor1.train()
    actor2.train()
    actor3.train()

@torch.no_grad()
def new_run_k_episodes(k, env, phase, device, actor1, actor2, actor3, case=None, episode=None):
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0

    cumulative_reward0 = []
    cumulative_reward1 = []
    cumulative_reward2 = []

    collision_cases = []
    timeout_cases = []
    actor1.eval()
    actor2.eval()
    actor3.eval()
    robot_actor_list = [actor1, actor2, actor3]
    noise = False
    for i in range(k):

        if case is not None:
            obs = env.reset(phase, test_case=case * k + i)
        else:
            obs = env.reset(phase)
        done = False

        robot_rewards = [[] for _ in range(len(env.robots))]

        infos = []

        while not done:

            robot_actionXYs = []

            for i in range(len(env.robots)):
                jointstate = JointState(env.robots[i].get_full_state(), obs[i])
                state = transform(jointstate, device)
                state = state.reshape(1, env.human_num + 2, 13)
                robot_actionXY = robot_actor_list[i].act_cql(state, device)

                robot_actionXYs.append(robot_actionXY)

            # 返回一个回合的3个机器人的奖励的列表
            obs, rewards, dones, infos = env.step_new(robot_actionXYs)

            for i in range(len(env.robots)):
                robot_rewards[i].append(rewards[i])

            if all(isinstance(info, ReachGoal) for info in infos):
                done = True
            elif any(isinstance(info, Collision) for info in infos):
                done = True
            elif any(isinstance(info, Timeout) for info in infos):
                done = True

        if all(isinstance(info, ReachGoal) for info in infos):
            success += 1

        elif any(isinstance(info, Collision) for info in infos):
            collision += 1
            collision_cases.append(i)
            collision_times.append(env.global_time)

        elif any(isinstance(info, Timeout) for info in infos):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.time_limit)

        else:
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(env.time_limit)

        if all(isinstance(info, ReachGoal) for info in infos):
            time = env.global_time
        else:
            time = 24.25

        success_times.append(time)

        cumulative_reward0.append(
            sum([pow(0.9, t * 0.25) * reward for t, reward in enumerate(robot_rewards[0])]))
        cumulative_reward1.append(
            sum([pow(0.9, t * 0.25) * reward for t, reward in enumerate(robot_rewards[1])]))
        cumulative_reward2.append(
            sum([pow(0.9, t * 0.25) * reward for t, reward in enumerate(robot_rewards[2])]))

    success_rate = success / k
    collision_rate = collision / k

    assert success + collision + timeout == k
    avg_nav_time = sum(success_times) / len(success_times)

    extra_info = '' if episode is None else 'in episode {} '.format(episode)
    logging.info(
        '{:<5} {}has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.2f}, leader reward: {:.4f}, f1 reward: {:.4f}, f2 reward: {:.4f}'.
        format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
               average(cumulative_reward0), average(cumulative_reward1), average(cumulative_reward2)))

    actor1.train()
    actor2.train()
    actor3.train()

# def transform(state, device):
#
#     state_tensor = torch.cat([torch.Tensor(np.array([state.self_state + human_state])).to(device)
#                               for human_state in state.human_states], dim=0)
#     state_tensor = rotate(state_tensor)
#     return state_tensor

def transform(state, device):
    """
    Optimized state transformation function.
    It first builds a list of numpy arrays on the CPU, concatenates them into a single
    large numpy array, and then performs a single conversion and transfer to the GPU.
    This significantly reduces CPU-GPU communication overhead.
    """
    # 1. 在CPU上构建一个numpy数组列表
    numpy_states = [state.self_state + human_state for human_state in state.human_states]

    # 2. 将列表堆叠成一个大的Numpy数组
    numpy_batch = np.array(numpy_states, dtype=np.float32)

    # 3. 一次性将Numpy数组转化为GPU Tensor
    state_tensor = torch.from_numpy(numpy_batch).to(device)

    # 4. 执行旋转操作
    state_tensor = rotate(state_tensor)
    return state_tensor

def rotate(state):

    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

    radius = state[:, 4].reshape((batch, -1))
    # set theta to be zero since it's not used
    theta = torch.zeros_like(v_pref)
    vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
    vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
    px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
    px1 = px1.reshape((batch, -1))
    py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
    py1 = py1.reshape((batch, -1))
    radius1 = state[:, 13].reshape((batch, -1))
    radius_sum = radius + radius1
    da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                              reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
    
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum],
                          dim=1)
    return new_state


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0