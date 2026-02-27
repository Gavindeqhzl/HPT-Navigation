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
from crowd_nav.utils.buffer import ReplayBuffer
from crowd_nav.policy.MAIQL import MAIQL, Actor, Critic, Value
from crowd_nav.util import print_time, save_models, new_run_k_episodes, run_k_episodes
from math import pow

GAMMA = pow(0.9, 0.25)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(env, memory, trainer, output_dir, device):
    t1 = T.time()
    step = 0
    test_eopch = 0
    max_episode = 100000

    episode = 0
    print("begin")

    while episode < max_episode:
        # test
        if (episode % 50000 == 0 and episode != 0):
            save_models(output_dir, str(episode), trainer)
            print_time(t1, episode, max_episode)
            run_k_episodes(500, env, "test", device, trainer.actor0, trainer.actor1, trainer.actor2, case=None, episode=test_eopch)
            test_eopch += 1

        # val
        if episode % 1000 == 0:
            run_k_episodes(50, env, "train", device, trainer.actor0, trainer.actor1, trainer.actor2, case=step + 500, episode=step)
            step += 1

        memorys = memory.sample(256)
        trainer.train_episode(memorys)

        episode += 1

    # final save + test
    finish_name = "finish"
    save_models(output_dir, finish_name, trainer)
    new_run_k_episodes(500, env, "test", device, trainer.actor0, trainer.actor1, trainer.actor2, case=None, episode=test_eopch)

def main():
    np.random.seed(10)
    torch.manual_seed(10)

    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env5.config')
    parser.add_argument('--output_dir', type=str, default='data/MAIQL5_plot')
    parser.add_argument('--output_file', type=str, default='iql5.log')
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

    logging.info('Using device: %s', device)

    # networks (per-agent)
    actor0 = Actor(134, 2, 1, device=device).to(device)
    critic10 = Critic(134, 2).to(device)
    critic20 = Critic(134, 2).to(device)
    value0 = Value().to(device)

    actor0_optimizer = torch.optim.Adam(actor0.parameters(), lr=3e-4)
    critic10_optimizer = torch.optim.Adam(critic10.parameters(), lr=3e-4)
    critic20_optimizer = torch.optim.Adam(critic20.parameters(), lr=3e-4)
    value0_optimizer = torch.optim.Adam(value0.parameters(), lr=3e-4)

    actor1 = Actor(134, 2, 1, device=device).to(device)
    critic11 = Critic(134, 2).to(device)
    critic21 = Critic(134, 2).to(device)
    value1 = Value().to(device)

    actor1_optimizer = torch.optim.Adam(actor1.parameters(), lr=3e-4)
    critic11_optimizer = torch.optim.Adam(critic11.parameters(), lr=3e-4)
    critic21_optimizer = torch.optim.Adam(critic21.parameters(), lr=3e-4)
    value1_optimizer = torch.optim.Adam(value1.parameters(), lr=3e-4)

    actor2 = Actor(134, 2, 1, device=device).to(device)
    critic12 = Critic(134, 2).to(device)
    critic22 = Critic(134, 2).to(device)
    value2 = Value().to(device)

    actor2_optimizer = torch.optim.Adam(actor2.parameters(), lr=3e-4)
    critic12_optimizer = torch.optim.Adam(critic12.parameters(), lr=3e-4)
    critic22_optimizer = torch.optim.Adam(critic22.parameters(), lr=3e-4)
    value2_optimizer = torch.optim.Adam(value2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": 1,

        "actor0": actor0,
        "actor0_optimizer": actor0_optimizer,
        "critic10": critic10,
        "critic10_optimizer": critic10_optimizer,
        "critic20": critic20,
        "critic20_optimizer": critic20_optimizer,
        "value0": value0,
        "value0_optimizer": value0_optimizer,

        "actor1": actor1,
        "actor1_optimizer": actor1_optimizer,
        "critic11": critic11,
        "critic11_optimizer": critic11_optimizer,
        "critic21": critic21,
        "critic21_optimizer": critic21_optimizer,
        "value1": value1,
        "value1_optimizer": value1_optimizer,

        "actor2": actor2,
        "actor2_optimizer": actor2_optimizer,
        "critic12": critic12,
        "critic12_optimizer": critic12_optimizer,
        "critic22": critic22,
        "critic22_optimizer": critic22_optimizer,
        "value2": value2,
        "value2_optimizer": value2_optimizer,

        "discount": GAMMA,
        "tau": 0.005,
        "device": device,
        "expectile": 0.8,
        "beta": 5.0,
    }

    trainer = MAIQL(**kwargs)

    # env
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot0 = Robot(env_config, 'robot')
    robot1 = Robot(env_config, 'robot')
    robot2 = Robot(env_config, 'robot')
    env.set_robot(robot0, robot1, robot2)

    robot0.set_policy(actor0)
    robot0.print_info()
    robot1.set_policy(actor1)
    robot1.print_info()
    robot2.set_policy(actor2)
    robot2.print_info()

    memory = ReplayBuffer(state_dim=13*7, action_dim=2, buffer_size=500000, device=device)
    memory.load_h5py_dataset(5)

    train(env, memory, trainer, args.output_dir, device)

if __name__ == '__main__':
    main()
