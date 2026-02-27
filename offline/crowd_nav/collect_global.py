import sys
import argparse
import configparser
import os
import torch
import random
import gym
import h5py
import pickle
import copy
import numpy as np
from numpy.linalg import norm

from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.sac_attention import SACAgent2
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout, Danger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEGMENT_LEN = 15  # 轨迹片段长度

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    parser.add_argument('--output_dir', type=str, default='data/bbb')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 策略权重
    actor0_weight_file = os.path.join(args.output_dir, 'actor0_model.pth')
    actor1_weight_file = os.path.join(args.output_dir, 'actor1_model.pth')
    actor2_weight_file = os.path.join(args.output_dir, 'actor2_model.pth')

    # 创建并加载三个策略
    def create_policy():
        return SACAgent2(
            state_dim=65, action_dim=2, action_range=[-1, 1], device="cpu", critic_cfg=0,
            actor_cfg=0, discount=0.99, init_temperature=0.1, alpha_lr=1e-3, alpha_betas=[0.9, 0.99],
            actor_lr=1e-3, actor_betas=[0.9, 0.99], actor_update_frequency=2, critic_lr=1e-3,
            critic_betas=[0.9, 0.99], critic_tau=0.005, critic_target_update_frequency=1,
            batch_size=128, learnable_temperature=True
        )

    policy0, policy1, policy2 = create_policy(), create_policy(), create_policy()
    policy0.actor.load_state_dict(torch.load(actor0_weight_file, map_location='cpu'))
    policy1.actor.load_state_dict(torch.load(actor1_weight_file, map_location='cpu'))
    policy2.actor.load_state_dict(torch.load(actor2_weight_file, map_location='cpu'))

    # 环境
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot0, robot1, robot2 = Robot(env_config, 'robot'), Robot(env_config, 'robot'), Robot(env_config, 'robot')
    env.set_robot(robot0, robot1, robot2)
    robots = env.robots
    robot0.set_policy(policy0)
    robot1.set_policy(policy1)
    robot2.set_policy(policy2)

    # 数据存储（按机器人分别存，后面再按偏好片段拼接写 HDF5）
    states = [[] for _ in range(3)]       # 每步保存展平后的状态 (H*13,)
    next_states = [[] for _ in range(3)]  # 同上
    actions = [[] for _ in range(3)]      # 每步动作 (2,)
    rewards = [[] for _ in range(3)]
    terminals = [[] for _ in range(3)]

    # 偏好标签需要的索引与标签
    indices1, indices2, human_labels = [], [], []

    # 缓存两段轨迹所需信息
    pre_info = [None] * 3
    pre_dis_list = [[] for _ in range(3)]
    pre_danger_list = [[] for _ in range(3)]
    now_dis_list = [[] for _ in range(3)]
    now_danger_list = [[] for _ in range(3)]

    step = 0          # 全局步计数
    index = -1        # 回合计数
    run = True
    noise = True

    success = 0
    success_times = []
    collision = 0
    timeout = 0

    while run:
        obs = env.reset('train')
        index += 1
        done = False
        j = 0  # 当前回合的步数

        while not done:
            robot_actionXYs = []
            for i in range(3):
                jointstate = JointState(robots[i].get_full_state(), obs[i])

                # 1) 给策略的输入：保持 (H, 13)
                policy_state = transform(jointstate)      # torch.Size([H, 13])

                # 动作
                action_xy, action = robots[i].act_sac(policy_state, noise)

                # 注入少量高斯噪声并裁剪
                vx = clamp_xy(action_xy.vx + np.random.normal(0, 0.1), -1, 1)
                vy = clamp_xy(action_xy.vy + np.random.normal(0, 0.1), -1, 1)
                action_xy = ActionXY(vx, vy)
                robot_actionXYs.append(action_xy)

                # 2) 用于保存：展平状态 (H*13,)
                save_state = policy_state.reshape(-1).cpu().numpy()
                actions[i].append(to_np(action))          # (2,)
                states[i].append(save_state)              # (H*13,)

            # 与环境交互
            obs, reward_list, dones, infos = env.step_new(robot_actionXYs)

            # 记录下一时刻状态与指标
            for i in range(3):
                # 距离目标
                now_dis_list[i].append(norm(np.array(robots[i].get_position()) - np.array(robots[i].get_goal_position())))

                # next_state：同样存展平 (H*13,)
                next_policy_state = transform(JointState(robots[i].get_full_state(), obs[i]))
                next_states[i].append(next_policy_state.reshape(-1).cpu().numpy())

                terminals[i].append(int(dones[i]))
                rewards[i].append(reward_list[i])

                # 危险标记
                now_danger_list[i].append(1 if isinstance(infos[i], Danger) else 0)

            step += 1
            j += 1

            # 回合终止判断
            if all(isinstance(info, ReachGoal) for info in infos):
                done = True
                success += 1
                success_times.append(env.global_time)
            elif any(isinstance(info, Collision) for info in infos):
                done = True
                collision += 1
            elif any(isinstance(info, Timeout) for info in infos):
                done = True
                timeout += 1

            # 总步数上限
            if len(states[0]) >= 300000:
                run = False
                break

        # ===== 偏好采集：两段轨迹片段的比较 =====
        # 第一个回合（index % 4 == 0）缓存 “前一段”
        if len(human_labels) < 2000 and index % 4 == 0:
            pre_info = infos.copy()
            for i in range(3):
                pre_dis_list[i] = copy.deepcopy(now_dis_list[i])
                pre_danger_list[i] = copy.deepcopy(now_danger_list[i])

        # 第二个回合（index % 4 == 1）与前一段比较
        cond_len_ok = all(len(lst) >= SEGMENT_LEN for lst in pre_dis_list + now_dis_list)
        if len(human_labels) < 2000 and index % 4 == 1 and cond_len_ok:
            label = decide_preference(pre_info, infos, pre_dis_list, now_dis_list, pre_danger_list, now_danger_list)
            human_labels.append(label)

            # 对齐到全局时间索引（与保存到 states/actions 的全局下标一致）
            indices1.append(step - j - SEGMENT_LEN)   # 段A末尾发生在上一个回合结束处
            indices2.append(step - SEGMENT_LEN)       # 段B末尾发生在当前回合结束处

            # 清空缓存
            for i in range(3):
                pre_dis_list[i].clear()
                pre_danger_list[i].clear()

        # 清空当前回合统计
        for i in range(3):
            now_dis_list[i].clear()
            now_danger_list[i].clear()

        avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit

        print(f"[回合{index}] 总步: {step} | 成功 {success} | 碰撞 {collision} | 超时 {timeout} | 偏好数 {len(human_labels)} | 导航时间 {avg_nav_time}")

    # ===== 保存标签 =====
    base_path = os.path.join(args.output_dir, 'global')
    os.makedirs(base_path, exist_ok=True)
    save_to_pickle(indices1, "indices1.pkl", base_path)
    save_to_pickle(indices2, "indices2.pkl", base_path)
    save_to_pickle(human_labels, "human_labels.pkl", base_path)

    # ===== 保存多智能体整体 HDF5（按偏好片段组织）=====
    save_multi_agent_hdf5(states, actions, next_states, terminals, rewards,
                          indices1, indices2, human_labels)

    agent_out_dir = os.path.join(args.output_dir, 'per_agent')
    save_per_agent_hdf5(states, actions, next_states, terminals, rewards, agent_out_dir)
    print("Done.")

# ===================== 偏好判定 =====================

def decide_preference(pre_info, infos, pre_dis_list, now_dis_list, pre_danger_list, now_danger_list):
    """
    准则优先级：
      1) 是否到达目标（这里用是否碰撞来区分失败场景；ReachGoal 由环境终止条件涵盖）
      2) 平均距离缩短（“像目标点移动距离 = 编队保持”的综合）
      3) 危险次数
    返回：1 表示“前一段(A) 优于 当前段(B)”，0 表示 B 优于 A
    """
    # 若两段都发生碰撞 -> 比较期末平均距离；若再相等 -> 比较危险次数
    if any(isinstance(info, Collision) for info in pre_info) and any(isinstance(info, Collision) for info in infos):
        pre_avg = avg(pre_dis_list[0][-1], pre_dis_list[1][-1], pre_dis_list[2][-1])
        now_avg = avg(now_dis_list[0][-1], now_dis_list[1][-1], now_dis_list[2][-1])
        if pre_avg < now_avg:
            return 1
        elif pre_avg > now_avg:
            return 0
        else:
            return 1 if compute_danger(pre_danger_list) < compute_danger(now_danger_list) else 0

    # 只有 A 碰撞 -> 选 B
    if any(isinstance(info, Collision) for info in pre_info):
        return 0

    # 只有 B 碰撞 -> 选 A
    if any(isinstance(info, Collision) for info in infos):
        return 1

    # 都未碰撞：比较平均距离缩短；再相等比较危险次数
    pre_reduce = compute_distance_reduction(pre_dis_list)
    now_reduce = compute_distance_reduction(now_dis_list)
    if pre_reduce > now_reduce:
        return 1
    elif pre_reduce < now_reduce:
        return 0
    else:
        return 1 if compute_danger(pre_danger_list) < compute_danger(now_danger_list) else 0


def compute_distance_reduction(dis_list):
    # 使用片段首末差近似“向目标移动距离”，三机器人取平均
    return avg(
        dis_list[0][0] - dis_list[0][-1],
        dis_list[1][0] - dis_list[1][-1],
        dis_list[2][0] - dis_list[2][-1]
    )


def compute_danger(danger_list):
    # 三个机器人的危险计数之和（最近 SEGMENT_LEN 步）
    return sum(danger_list[0][-SEGMENT_LEN:]) + \
           sum(danger_list[1][-SEGMENT_LEN:]) + \
           sum(danger_list[2][-SEGMENT_LEN:])


# ===================== 数据保存 =====================

def save_multi_agent_hdf5(states, actions, next_states, terminals, rewards, indices1, indices2, labels):
    """
    以「偏好样本」为粒度保存：
      每个样本包含 segment_A 与 segment_B 的 [state, action] 序列（长度 SEGMENT_LEN），以及标签 label。
      这里的 state 已经是 (H*13,) 的展平向量；三个机器人在时间维拼接后，再在特征维 concat。
    """
    file_name = 'multi_agent_dataset.hdf5'
    with h5py.File(file_name, 'w') as f:
        for idx, (i1, i2, label) in enumerate(zip(indices1, indices2, labels)):
            grp = f.create_group(f'segment_{idx}')

            # 取出两段的序列并在特征维拼接（3个机器人）
            state_A = np.concatenate([np.stack(states[k][i1:i1 + SEGMENT_LEN], axis=0) for k in range(3)], axis=-1)
            action_A = np.concatenate([np.stack(actions[k][i1:i1 + SEGMENT_LEN], axis=0) for k in range(3)], axis=-1)

            state_B = np.concatenate([np.stack(states[k][i2:i2 + SEGMENT_LEN], axis=0) for k in range(3)], axis=-1)
            action_B = np.concatenate([np.stack(actions[k][i2:i2 + SEGMENT_LEN], axis=0) for k in range(3)], axis=-1)

            grp.create_dataset('state_A', data=state_A)     # shape: (SEGMENT_LEN, 3*H*13)
            grp.create_dataset('action_A', data=action_A)   # shape: (SEGMENT_LEN, 3*2)
            grp.create_dataset('state_B', data=state_B)
            grp.create_dataset('action_B', data=action_B)
            grp.attrs['label'] = float(label)

    print(f'HDF5 文件已保存: {file_name}')


def save_per_agent_hdf5(states, actions, next_states, terminals, rewards, out_dir):
    """
    将运行过程中积累的逐步数据，分别保存成 robot0/1/2 的独立 HDF5，键名满足 buffer.py 的读取：
      - states:       (N, state_dim_flat)
      - actions:      (N, 2)
      - next_states:  (N, state_dim_flat)
      - terminals:    (N,)  或 (N,1)
      - rewards:      (N,)  或 (N,1)  ——先保存环境原始奖励，之后会被奖励模型重写
    """
    os.makedirs(out_dir, exist_ok=True)
    for i in range(3):
        file_path = os.path.join(out_dir, f"robot{i+1}.hdf5")
        with h5py.File(file_path, "w") as f:
            f.create_dataset("states", data=np.asarray(states[i], dtype=np.float32))
            f.create_dataset("actions", data=np.asarray(actions[i], dtype=np.float32))
            f.create_dataset("next_states", data=np.asarray(next_states[i], dtype=np.float32))
            f.create_dataset("terminals", data=np.asarray(terminals[i], dtype=np.int32))
            f.create_dataset("rewards", data=np.asarray(rewards[i], dtype=np.float32))
        print(f"[Per-agent] Saved {file_path}")


def save_to_pickle(data_list, filename, base_dir):
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "wb") as fp:
        pickle.dump(data_list, fp)
    print(f"Saved {filename} to {file_path}")


# ===================== 小工具 =====================

def clamp_xy(num, min_val, max_val):
    return max(min(num, max_val), min_val)


def to_np(t):
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "nelement") and t.nelement() == 0:
        return np.array([])
    return t.detach().cpu().numpy()


def transform(state):
    """
    将 JointState -> (H, 13) 的策略输入张量（不要展平给策略！）
    """
    # 每个人/邻居一行（长度 14），随后 rotate -> 13 维
    rows = []
    for human_state in state.human_states:
        rows.append(torch.tensor(state.self_state + human_state, dtype=torch.float32))
    state_tensor = torch.stack(rows, dim=0)  # (H, 14)
    return rotate(state_tensor)              # (H, 13)


def rotate(state):
    """
    坐标变换函数，将状态从全局坐标系转换为以代理为中心的坐标系。
    Transform the coordinate to agent-centric.
    Input state tensor is of size (batch_size, state_length)

    """
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
    # 将所有转换后的特征拼接在一起形成新的状态张量
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state


def avg(a, b, c):
    return (a + b + c) / 3.0


if __name__ == '__main__':
    main()

