import sys
import logging
import argparse
import configparser
import os
import torch
import gym
import numpy as np
import time as T
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.buffer import ReplayBuffer_episode, ReplayBuffer
from crowd_nav.policy.MACQL import Traj_MACQL, ctde_critic, distributed_policy
from crowd_nav.util import print_time, save_models, new_run_k_episodes
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.state import JointState
from math import pow
from tensorboardX import SummaryWriter

GAMMA = pow(0.9, 0.25)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(env, memory, trainer, output_dir, device):

    t1 = T.time()
    step = 0
    test_eopch = 0
    max_episode = 200000

    episode=0   
    reward_episode = 0
    print("begin")

    # while reward_episode < 200000:

    #     memorys = memory.episode_sample(16)
    #     trainer.train_reward_model(memorys)
    #     reward_episode +=1

    while episode < max_episode:
        
        # test
        if (episode % 50000 == 0 and episode != 0):
            
            save_models(output_dir, str(episode), trainer)
            print_time(t1, episode, max_episode)
            
            new_run_k_episodes(500, env, "test", device, trainer.actor0, trainer.actor1, trainer.actor2, case = None, episode=test_eopch)
            test_eopch += 1

        # val
        if episode % 1000 == 0:
            new_run_k_episodes(50, env, "train", device, trainer.actor0, trainer.actor1, trainer.actor2, case = step + 500, episode=step)
            step+=1

        memorys = memory.sample(256)
        trainer.train_episode(memorys)

        episode+=1

    # 跑完全部后再保存模型和测试
    finish_name = "finish"
    save_models(output_dir, finish_name, trainer)
    new_run_k_episodes(500, env, "test", device, trainer.actor0, trainer.actor1, trainer.actor2, case = None, episode=test_eopch)

def main():

    np.random.seed(10)
    torch.manual_seed(10)

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env5.config')
    parser.add_argument('--output_dir', type=str, default='data/cql5_credit')
    parser.add_argument('--output_file', type=str, default='MACQL5.log')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # log   
    log_file = os.path.join(args.output_dir, args.output_file)
    mode = 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info('Using device: %s', device)

    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot0 = Robot(env_config, 'robot')
    robot1 = Robot(env_config, 'robot')
    robot2 = Robot(env_config, 'robot')
    env.set_robot(robot0, robot1, robot2)
   
    human_num = env_config.getint('sim', 'human_num')
    logging.info('human num: %s', human_num)

    # network
    critic_1_0 = ctde_critic(136, 2, True, 2).to(device)
    critic_2_0 = ctde_critic(134, 2, True, 2).to(device)
    critic_1_0_optimizer = torch.optim.Adam(list(critic_1_0.parameters()), 3e-4)
    critic_2_0_optimizer = torch.optim.Adam(list(critic_2_0.parameters()), 3e-4)

    critic_1_1 = ctde_critic(136, 2, True, 2).to(device)
    critic_2_1 = ctde_critic(134, 2, True, 2).to(device)
    critic_1_1_optimizer = torch.optim.Adam(list(critic_1_1.parameters()), 3e-4)
    critic_2_1_optimizer = torch.optim.Adam(list(critic_2_1.parameters()), 3e-4)

    critic_1_2 = ctde_critic(136, 2, True, 2).to(device)
    critic_2_2 = ctde_critic(134, 2, True, 2).to(device)
    critic_1_2_optimizer = torch.optim.Adam(list(critic_1_2.parameters()), 3e-4)
    critic_2_2_optimizer = torch.optim.Adam(list(critic_2_2.parameters()), 3e-4)

    actor0 = distributed_policy(134, 2, 1, log_std_multiplier=1.0, orthogonal_init=True, ).to(device)
    actor_optimizer_0 = torch.optim.Adam(actor0.parameters(), 3e-4)

    actor1 = distributed_policy(134, 2, 1, log_std_multiplier=1.0, orthogonal_init=True, ).to(device)
    actor_optimizer_1 = torch.optim.Adam(actor1.parameters(), 3e-4)

    actor2 = distributed_policy(134, 2, 1, log_std_multiplier=1.0, orthogonal_init=True, ).to(device)
    actor_optimizer_2 = torch.optim.Adam(actor2.parameters(), 3e-4)

    kwargs = {
        "critic_1_0": critic_1_0,
        "critic_2_0": critic_2_0,
        "critic_1_0_optimizer": critic_1_0_optimizer,
        "critic_2_0_optimizer": critic_2_0_optimizer,

        "critic_1_1": critic_1_1,
        "critic_2_1": critic_2_1,
        "critic_1_1_optimizer": critic_1_1_optimizer,
        "critic_2_1_optimizer": critic_2_1_optimizer,

        "critic_1_2": critic_1_2,
        "critic_2_2": critic_2_2,
        "critic_1_2_optimizer": critic_1_2_optimizer,
        "critic_2_2_optimizer": critic_2_2_optimizer,

        "actor0": actor0,
        "actor1": actor1,
        "actor2": actor2,

        "actor0_optimizer": actor_optimizer_0,
        "actor1_optimizer": actor_optimizer_1,
        "actor2_optimizer": actor_optimizer_2,

        "discount": GAMMA,
        "human_num": human_num+2,
        "soft_target_update_rate": 5e-3,

        "device": device,
        # CQL
        "target_entropy": -np.prod(2).item(),
        "alpha_multiplier": 1.0,
        "use_automatic_entropy_tuning": True,
        "backup_entropy": False,
        "policy_lr": 3e-4,
        "qf_lr": 3e-4,
        "bc_steps": 0,
        "target_update_period": 1,
        "cql_n_actions": 10,
        "cql_importance_sample": True,
        "cql_lagrange": False,
        "cql_target_action_gap": -1.0,
        "cql_temp": 1.0,
        "cql_alpha": 1.0,
        "cql_max_target_backup": False,
        "cql_clip_diff_min": -np.inf,
        "cql_clip_diff_max": np.inf,
    }

    # Initialize actor
    trainer = Traj_MACQL(**kwargs)

    # env
    robot0.set_policy(actor0)
    robot0.print_info()
    robot1.set_policy(actor1)
    robot1.print_info()
    robot2.set_policy(actor2)
    robot2.print_info()


    memory = ReplayBuffer(state_dim=13*(human_num+2), action_dim=2, buffer_size=500000, device = device)
    memory.load_h5py_dataset(human_num=human_num)

    train(env, memory, trainer, args.output_dir, device)
    
if __name__ == '__main__':
    main()