import sys
import argparse
import configparser
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import warnings
import types

# 屏蔽 Matplotlib 警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_nav.policy.MATD3BC import Actor
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout


# ==================== Monkey Patch ====================
def get_center(self):
    return self.px, self.py


setattr(Human, 'center', property(get_center))
setattr(Robot, 'center', property(get_center))


# ======================================================

def transform(state, device):
    numpy_states = [state.self_state + human_state for human_state in state.human_states]
    numpy_batch = np.array(numpy_states, dtype=np.float32)
    state_tensor = torch.from_numpy(numpy_batch).to(device)
    state_tensor = rotate(state_tensor)
    return state_tensor


def rotate(state):
    batch = state.shape[0]
    dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
    dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
    rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])
    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    v_pref = state[:, 7].reshape((batch, -1))
    vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
    vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))
    radius = state[:, 4].reshape((batch, -1))
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
    new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
    return new_state


def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_network'):
            new_key = k.replace('base_network', 'net')
        else:
            new_key = k
        if "log_std_multiplier" in new_key or "log_std_offset" in new_key:
            continue
        new_state_dict[new_key] = v
    return new_state_dict


# ==================== 自定义渲染函数 ====================
def custom_render(self, mode='human', output_file=None):
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    x_offset = 0.11
    y_offset = 0.11
    cmap = plt.cm.get_cmap('hsv', 10)

    # 颜色设置
    robot0_color = 'green'  # Robot1 (Leader)
    robot1_color = 'yellow'  # Robot2 (Follower)
    robot2_color = 'orange'  # Robot3 (Follower)
    goal_color = 'red'

    if mode == 'traj':
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)

        # ---------------- 关键修改开始 ----------------
        # 问题原因：self.states 通常存的是上一刻的状态，step() 结束后 done=True 时，
        # 最新的位置（机器人此时其实已经在此位置）可能还没 append 进去。
        # 解决方法：我们构建临时的位置列表，把当前机器人真实的最新位置加进去。

        # 1. 提取历史轨迹
        robot1_positions = [self.states[i][0].position for i in range(len(self.states))]
        robot2_positions = [self.states[i][1].position for i in range(len(self.states))]
        robot3_positions = [self.states[i][2].position for i in range(len(self.states))]
        human_positions = [[self.states[i][3][j].position for j in range(len(self.humans))]
                           for i in range(len(self.states))]

        # 2. 【核心修复】追加当前时刻（最后一步）的真实位置
        # 只有当历史记录的最后一步跟当前实际位置距离大于微小阈值时才追加，避免重复
        current_pos_0 = self.robots[0].get_position()
        last_pos_0 = robot1_positions[-1]

        # 如果最后记录的位置和当前实际位置不重合（说明缺了最后一步），则补上
        if np.linalg.norm(np.array(current_pos_0) - np.array(last_pos_0)) > 1e-4:
            robot1_positions.append(self.robots[0].get_position())
            robot2_positions.append(self.robots[1].get_position())
            robot3_positions.append(self.robots[2].get_position())

            # 行人也要补最后一步
            last_human_pos = []
            for human in self.humans:
                last_human_pos.append(human.get_position())
            human_positions.append(last_human_pos)

        # 更新总步数长度
        total_steps = len(robot1_positions)
        # ---------------- 关键修改结束 ----------------

        # 绘制终点 (红色星星)
        goal_positions = [r.get_goal_position() for r in self.robots]
        for gp in goal_positions:
            goal_marker = mlines.Line2D([gp[0]], [gp[1]], color=goal_color,
                                        marker='*', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal_marker)

        # === 时间标记逻辑 ===
        target_interval = 4.0
        steps_per_interval = int(target_interval / self.time_step)

        # 使用更新后的 total_steps 进行遍历
        for k in range(total_steps):
            global_time = k * self.time_step

            # 绘制圆圈 (每4步画一次，保持轨迹稀疏度)
            # 另外必须画最后一步 (k == total_steps - 1)
            should_draw_circle = (k % 4 == 0) or (k == total_steps - 1)

            if should_draw_circle:
                r0 = plt.Circle(robot1_positions[k], self.robots[0].radius, fill=True, color=robot0_color, alpha=0.9)
                r1 = plt.Circle(robot2_positions[k], self.robots[1].radius, fill=True, color=robot1_color, alpha=0.9)
                r2 = plt.Circle(robot3_positions[k], self.robots[2].radius, fill=True, color=robot2_color, alpha=0.9)
                ax.add_artist(r0)
                ax.add_artist(r1)
                ax.add_artist(r2)

                humans_circles = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                                  for i in range(len(self.humans))]
                for hc in humans_circles:
                    ax.add_artist(hc)

            # 绘制时间文字
            is_start = (k == 0)
            is_interval = (k % steps_per_interval == 0) and (k != 0)
            is_end = (k == total_steps - 1)

            if is_start or is_interval or is_end:
                current_positions = [robot1_positions[k], robot2_positions[k], robot3_positions[k]]
                for i in range(len(self.humans)):
                    current_positions.append(human_positions[k][i])

                for pos in current_positions:
                    ax.text(pos[0] - x_offset, pos[1] - y_offset,
                            '{:.1f}'.format(global_time),
                            color='black', fontsize=11, fontweight='bold', zorder=100,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            # 绘制连线
            if k != 0:
                # 注意：positions 列表现在包含了补全的点，直接索引即可
                nav_direction1 = plt.Line2D((robot1_positions[k - 1][0], robot1_positions[k][0]),
                                            (robot1_positions[k - 1][1], robot1_positions[k][1]),
                                            color=robot0_color, ls='solid')
                nav_direction2 = plt.Line2D((robot2_positions[k - 1][0], robot2_positions[k][0]),
                                            (robot2_positions[k - 1][1], robot2_positions[k][1]),
                                            color=robot1_color, ls='solid')
                nav_direction3 = plt.Line2D((robot3_positions[k - 1][0], robot3_positions[k][0]),
                                            (robot3_positions[k - 1][1], robot3_positions[k][1]),
                                            color=robot2_color, ls='solid')
                ax.add_artist(nav_direction1)
                ax.add_artist(nav_direction2)
                ax.add_artist(nav_direction3)

                human_directions = [plt.Line2D((human_positions[k - 1][i][0], human_positions[k][i][0]),
                                               (human_positions[k - 1][i][1], human_positions[k][i][1]),
                                               color=cmap(i), ls='solid')
                                    for i in range(self.human_num)]
                for hd in human_directions:
                    ax.add_artist(hd)

        # 图例
        legend_elements = [
            patches.Patch(color=robot0_color, label='Robot1'),
            patches.Patch(color=robot1_color, label='Robot2'),
            patches.Patch(color=robot2_color, label='Robot3'),
            mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                          markersize=15, label='Goal')
        ]

        plt.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)
        plt.show()


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env5.config')
    parser.add_argument('--model_dir', type=str, default='data/MATD3BC5_y')
    # parser.add_argument('--model_epoch', type=str, default='200000')
    parser.add_argument('--model_epoch', type=str, default='finish')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # ================= 🔧 用户配置区域 🔧 =================
    # 在这里调整参数以筛选你想要的轨迹图
    # ----------------------------------------------------

    # 1. [新增] 固定测试种子 (对比实验专用):
    #    None      : 自动循环搜索符合条件的案例 (默认)
    #    整数 (如 5): 强制运行指定的测试用例 ID，忽略下面的筛选条件，直接出图。
    FIXED_TEST_CASE = 169

    # 2. 想要寻找的结果类型 (仅当 FIXED_TEST_CASE 为 None 时生效):
    #    'success'   : 只看成功到达终点的
    #    'collision' : 只看发生碰撞的
    #    'timeout'   : 只看超时的
    #    'any'       : 任何结果都行
    TARGET_RESULT = 'success'

    # 3. 想要的最小导航时间 (秒, 仅当 FIXED_TEST_CASE 为 None 时生效):
    #    设置 > 0 可以过滤掉那些太简单的（比如8秒直线）
    MIN_DURATION = 13.0
    # ======================================================

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    if FIXED_TEST_CASE is not None:
        print(f'Mode: Comparison (Fixed Seed: {FIXED_TEST_CASE})')
    else:
        print(f'Mode: Search -> Target: {TARGET_RESULT}, Min Duration: {MIN_DURATION}s')

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    # 挂载自定义渲染函数
    env.render = types.MethodType(custom_render, env)

    actor0 = Actor(134, 2, 1, device=device).to(device)
    actor1 = Actor(134, 2, 1, device=device).to(device)
    actor2 = Actor(134, 2, 1, device=device).to(device)

    model_path_0 = os.path.join(args.model_dir, args.model_epoch, f'actor0_weight_{args.model_epoch}.pth')
    model_path_1 = os.path.join(args.model_dir, args.model_epoch, f'actor1_weight_{args.model_epoch}.pth')
    model_path_2 = os.path.join(args.model_dir, args.model_epoch, f'actor2_weight_{args.model_epoch}.pth')

    if os.path.exists(model_path_0):
        state_dict_0 = fix_state_dict(torch.load(model_path_0, map_location=device))
        state_dict_1 = fix_state_dict(torch.load(model_path_1, map_location=device))
        state_dict_2 = fix_state_dict(torch.load(model_path_2, map_location=device))
        actor0.load_state_dict(state_dict_0, strict=False)
        actor1.load_state_dict(state_dict_1, strict=False)
        actor2.load_state_dict(state_dict_2, strict=False)
        print(f"Successfully loaded models for epoch: {args.model_epoch}")
    else:
        print(f"Error: Model file not found at {model_path_0}")
        return

    actor0.eval()
    actor1.eval()
    actor2.eval()
    robot_actor_list = [actor0, actor1, actor2]

    robot0 = Robot(env_config, 'robot')
    robot0.set_policy(actor0)
    robot1 = Robot(env_config, 'robot')
    robot1.set_policy(actor1)
    robot2 = Robot(env_config, 'robot')
    robot2.set_policy(actor2)
    env.set_robot(robot0, robot1, robot2)

    # 补全 env 属性
    env.robot1 = env.robots[0]
    env.robot2 = env.robots[1]
    env.robot3 = env.robots[2]

    # ================= 循环逻辑 =================
    if FIXED_TEST_CASE is not None:
        test_case_k = FIXED_TEST_CASE
        print("Starting Fixed Seed Run...")
    else:
        test_case_k = 0
        print("Starting Loop Search...")
        print("Note: Close the plot window to search for the next episode.")

    while True:
        obs = env.reset('test', test_case=test_case_k)
        done = False

        while not done:
            robot_actionXYs = []
            for i in range(len(env.robots)):
                jointstate = JointState(env.robots[i].get_full_state(), obs[i])
                state = transform(jointstate, device)
                state = state.reshape(1, env.human_num + 2, 13)
                robot_actionXY = robot_actor_list[i].act_cql(state, device)
                robot_actionXYs.append(robot_actionXY)

            obs, rewards, dones, infos = env.step_new(robot_actionXYs)

            # 判断当前状态
            is_success = all(isinstance(info, ReachGoal) for info in infos)
            is_collision = any(isinstance(info, Collision) for info in infos)
            is_timeout = any(isinstance(info, Timeout) for info in infos)

            if is_success or is_collision or is_timeout:
                done = True
                final_time = env.global_time

                # 决定是否渲染
                should_render = False
                status_str = "Unknown"

                if is_success:
                    status_str = "Success"
                elif is_collision:
                    status_str = "Collision"
                elif is_timeout:
                    status_str = "Timeout"

                # 逻辑分支：固定种子模式 vs 搜索模式
                if FIXED_TEST_CASE is not None:
                    # 固定种子模式：强制渲染
                    should_render = True
                    print(f"\n[Fixed Seed {test_case_k}] Result: {status_str}, Time: {final_time:.2f}s")
                else:
                    # 搜索模式：根据条件渲染
                    type_match = False
                    if TARGET_RESULT == 'any':
                        type_match = True
                    elif TARGET_RESULT == 'success' and is_success:
                        type_match = True
                    elif TARGET_RESULT == 'collision' and is_collision:
                        type_match = True
                    elif TARGET_RESULT == 'timeout' and is_timeout:
                        type_match = True

                    if type_match:
                        if final_time >= MIN_DURATION:
                            should_render = True
                            print(f"\n[Found!] Case {test_case_k}: {status_str}! Time = {final_time:.2f}s")

                # 执行渲染
                if should_render:
                    print("Rendering... (Close window to continue)")
                    env.render(mode='traj')
                    plt.show()

                    # 如果是固定种子，跑完一次就退出
                    if FIXED_TEST_CASE is not None:
                        print("Done.")
                        sys.exit()
                    else:
                        print("Searching for next case...", end='\r')

        # 如果不是固定种子，增加计数器继续寻找
        if FIXED_TEST_CASE is None:
            test_case_k += 1
        else:
            # 双重保险，防止死循环
            break


if __name__ == '__main__':
    main()