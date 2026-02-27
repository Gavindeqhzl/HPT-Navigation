import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import torch
import math
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from itertools import product


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.flag1 = 0
        self.flag2 = 0
        self.flag3 = 0
        self.flag = [False] * 3

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot0, robot1, robot2):
        self.robots = [robot0, robot1, robot2]

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'sc':
            # assert human_num >= 4
            self.humans = []
            for i in range(human_num):
                pro = np.random.random()
                if pro <= 0.5:
                    self.humans.append(self.generate_square_crossing_human())
                else :
                    self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in self.robots + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in self.robots + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in self.robots + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in self.robots + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human



    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.flag = [False] * len(self.robots)

        for i in range(len(self.robots)):
            if self.robots[i] is None:
                raise AttributeError('self.robots[i] has to be set!')

        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            # 初始化为长度为 self.human_num 的零列表
            self.human_times = [0] * self.human_num
        else:
            for i in range(len(self.robots)):
            #将 self.human_times 初始化为长度为 self.human_num 或 1 的零列表，具体取决于 self.robot.policy.multiagent_training 的值
                self.human_times = [0] * (self.human_num if self.robots[i].policy.multiagent_training else 1)
        for i in range(len(self.robots)):
            if not self.robots[i].policy.multiagent_training:
                self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robots[0].set(0, -4, 0, 4, 0, 0, np.pi / 2)
            self.robots[1].set(-2, -4, -2, 2, 0, 0, np.pi / 2)
            self.robots[2].set(2, -4, 2, 2, 0, 0, np.pi / 2)

            if self.case_counter[phase] >= 0:
                # print(counter_offset[phase] + self.case_counter[phase])
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                # torch.manual_seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    # for i in range(0, len(self.robots) - 1):
                    human_num = self.human_num
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                # assert phase == 'test'
                # if self.case_counter[phase] == -1:
                #     # for debugging purposes
                #     self.human_num = 3
                #     self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                #     self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                #     self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                #     self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                # else:
                raise NotImplementedError

        for agent in self.robots + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        #用循环创建3个列表
        obs = [[] for _ in range(len(self.robots))]
        for i in range(len(self.robots)):

            # if hasattr(self.robots[i].policy, 'action_values'):
            #     self.action_values = list()
            # if hasattr(self.robots[i].policy, 'get_attention_weights'):
            #     self.attention_weights = list()

            # get current observation 环境反馈给每个机器人的状态 123 档智能体位机器人

            if self.robots[i].sensor == 'coordinates':
                # append不能加依托，只能一个一个一个的加  obs的一个例子  5个人+一个机器人视角下的另两个机器人 -> 7
                # -3.6841091174133553 -2.212824017323522 0 0 0.3
                # 2.3524802983191253 -2.6587440900356496 0 0 0.3
                # -1.1756725528226781 4.104431035324694 0 0 0.3
                # 3.4320641683826976 -2.7539789365923735 0 0 0.3
                # -3.5488496570270938 0.6332227669599253 0 0 0.3
                # -4 -4 0 0 0.3
                # 2 -4 0 0 0.3
                for human in self.humans:
                    obs[i].append(human.get_observable_state())
                for j in range(len(self.robots)):
                    if j == i:
                        continue
                    obs[i].append(self.robots[j].get_observable_state())



            elif self.robots[i].sensor == 'RGB':
                raise NotImplementedError
            #print(obs)

        return obs

    # def onestep_lookahead(self, action):
    #     return self.step(action, update=False)


#     def step(self, actions, update=True):
#         """
#         Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
#          行人orca
#         """
#         human_actions = []
#         for human in self.humans:
#             # observation for humans is always coordinates
#             #档智能体位人类是的一个流程图
#             ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
#             if self.robots[0].visible:
#                 ob += [self.robots[0].get_observable_state()]
#             if self.robots[1].visible:
#                 ob += [self.robots[1].get_observable_state()]
#             if self.robots[2].visible:
#                 ob += [self.robots[2].get_observable_state()]
#             human_actions.append(human.act(ob))

#         # collision detection   人与机器人
#         dmins = [float('inf')]*3
#         collisions = [False]*3
#         #                  跟随者减去领航者
#         error1 = norm([self.robots[1].px - self.robots[0].px + 1.5, self.robots[1].py - self.robots[0].py + math.sqrt(1.75)])
#         error2 = norm([self.robots[2].px - self.robots[0].px - 1.5, self.robots[2].py - self.robots[0].py + math.sqrt(1.75)])

#         rewardf1 = 0
#         rewardf2 = 0

#         if error1 >= 0 and error1 <= 0.2:
#             rewardf1 = 1
#         elif error1 > 0.2 and error1 <= 1:
#             rewardf1 = -np.tanh(7.5 * error1 - 3)
#         elif error1 > 1 and error1 <= 2:
#             rewardf1 = -1
#         elif error1 > 2:
#             rewardf1 = error1 * -1

#         if error2 >= 0 and error2 <= 0.2:
#             rewardf2 = 1
#         elif error2 > 0.2 and error2 <= 1:
#             rewardf2 = -np.tanh(7.5 * error2 - 3)
#         elif error2 > 1 and error2 <= 2:
#             rewardf2 = -1
#         elif error2 > 2:
#             rewardf2 = error2 * -1


#         #robot与人类的碰撞
#         for j in range(len(self.robots)):
#         # for i, human in enumerate(self.humans):
#             # 获取列表中的元素索引 i 和相应的人类对象。
#             # for j in range(len(self.robots)):
#             for i, human in enumerate(self.humans):
#                 # print(human.px)
#                 # print(self.robots[j])
#                 px = human.px - self.robots[j].px
#                 py = human.py - self.robots[j].py
#                 if self.robots[j].kinematics == 'holonomic':

#                     vx = human.vx - actions[j].vx
#                     vy = human.vy - actions[j].vy
#                 else:
#                     vx = human.vx - actions[j].v * np.cos(actions[j].r + self.robots[j].theta)
#                     vy = human.vy - actions[j].v * np.sin(actions[j].r + self.robots[j].theta)
#                 ex = px + vx * self.time_step
#                 ey = py + vy * self.time_step

#                 # closest distance between boundaries of two agents
#                 closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robots[j].radius
#                 if closest_dist < 0:
#                     collisions[j] = True
#                     # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
#                     break
#                 elif closest_dist < dmins[j]:
#                     #dim一开始位无穷大
#                     dmins[j] = closest_dist

#         # collision detection between humans  人与人见的碰撞
#         for i, j in product(range(len(self.humans)), range(len(self.humans))):
#             # 避免比较代理与自身的情况，直接跳过
#             if i == j:
#                 continue
#             dx = self.humans[i].px - self.humans[j].px
#             dy = self.humans[i].py - self.humans[j].py
#             dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
#             if dist < 0:
#                 # detect collision but don't take humans' collision into account
#                 logging.debug('Collision happens between humans in step()')
#         #机器人与机器人见的碰撞
#         ##使用 product 函数生成代理数组中所有代理之间的组合（i, j），其中 i 和 j 取值范围是 0 到 num_agent - 1
#         # Rcollisions = [False]*3
#         # for i, j in product(range(len(self.robots)), range(len(self.robots))):
#         #     # 避免比较代理与自身的情况，直接跳过
#         #     if i == j:
#         #         continue
#         #     dx = self.robots[i].px - self.robots[j].px
#         #     dy = self.robots[i].py - self.robots[j].py
#         #     dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[i].radius - self.robots[j].radius
#         #     if dist < 0:
#         #         Rcollisions[i] = True
#         #         # detect collision but don't take humans' collision into account
#         #         logging.debug('Collision happens between robots in step()')

#         R1collisionR2 = False
#         dx = self.robots[0].px - self.robots[1].px
#         dy = self.robots[0].py - self.robots[1].py
#         dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[0].radius - self.robots[1].radius
#         if dist < 0:
#             R1collisionR2 = True
#             # detect collision but don't take robots' collision into account
#             logging.debug('Collision happens between robots in step()')

#         R1collisionR3 = False
#         dx = self.robots[0].px - self.robots[2].px
#         dy = self.robots[0].py - self.robots[2].py
#         dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[0].radius - self.robots[2].radius
#         if dist < 0:
#             R1collisionR3 = True
#             # detect collision but don't take robots' collision into account
#             logging.debug('Collision happens between robots in step()')

#         R2collisionR3 = False
#         dx = self.robots[1].px - self.robots[2].px
#         dy = self.robots[1].py - self.robots[2].py
#         dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[1].radius - self.robots[2].radius
#         if dist < 0:
#             R2collisionR3 = True
#             # detect collision but don't take robots' collision into account
#             logging.debug('Collision happens between robots in step()')

#         Rcollisions = [False] * 3
#         if R1collisionR2 or R2collisionR3:
#             Rcollisions[1] = True
#         if R1collisionR3 or R2collisionR3:
#             Rcollisions[2] = True
#         # check if reaching the goal

#         rewards = [0] * len(self.robots)
#         infos = [None] * len(self.robots)
#         dones = [0] * len(self.robots)
#         for i, action in enumerate(actions):
#             robot = self.robots[i]
#             end_position = np.array(robot.compute_position(action, self.time_step))
#             # 达到目标的布尔值将会被赋值给变量 reaching_goal。
#             reaching_goal = norm(end_position - np.array(robot.get_goal_position())) < robot.radius

#             cur_dist = norm(np.array([robot.px, robot.py]) - np.array(robot.get_goal_position()))
#             end_dist = norm(end_position - np.array(robot.get_goal_position()))
#             #reaching_goal_i = end_dist_i < self.roboti.radius
#             #print(norm(end_position - np.array(self.robot.get_goal_position())))
#             # rewards = np.zeros(len(self.robots))
#             # dones = np.zeros(len(self.robots))
#             # infos = [None] * len(self.robots)
#             if self.global_time >= self.time_limit - 1:
#                 if i == 0:
#                     rewards[i] = 0
#                     dones[i] = True
#                     infos[i] = Timeout()
#                 elif i == 1:
#                     rewards[i] = 0
#                     rewards[i] += rewardf1
#                     dones[i] = True
#                     infos[i] = Timeout()
#                 else:
#                     rewards[i] = 0
#                     rewards[i] += rewardf2
#                     dones[i] = True
#                     infos[i] = Timeout()
#             elif collisions[i] or Rcollisions[i]:
#                 if i == 0:
#                     rewards[i] = self.collision_penalty
#                     dones[i] = True
#                     infos[i] = Collision()
#                 elif i == 1:
#                     rewards[i] = -33.5
#                     rewards[i] += rewardf1
#                     dones[i] = True
#                     infos[i] = Collision()
#                 else:
#                     rewards[i] = -33.5
#                     rewards[i] += rewardf2
#                     dones[i] = True
#                     infos[i] = Collision()
#             elif reaching_goal:
#                 if i == 0:
#                     rewards[i] = self.success_reward
#                     dones[i] = True
#                     infos[i] = ReachGoal()
#                 #     跟随者成功Rf位0
#                 elif i == 1:
#                     rewards[i] = 0
#                     rewards[i] += rewardf1
#                     dones[i] = True
#                     infos[i] = ReachGoal()
#                 else:
#                     rewards[i] = 0
#                     rewards[i] += rewardf2
#                     dones[i] = True
#                     infos[i] = ReachGoal()
#             elif dmins[i] < self.discomfort_dist:
#                 # only penalize agent for getting too close if it's visible
#                 # adjust the reward based on FPS
#                 # rewards[i] = (dmins[i] - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
#                 rewards[i] = (dmins[i] - self.discomfort_dist) * self.discomfort_penalty_factor
#                 dones[i] = False
#                 infos[i] = Danger(dmins[i])
#             else:
#                 if i == 0:
#                     rewards[i] = 2*(cur_dist - end_dist)
#                     # print(rewards[i])
#                     dones[i] = False
#                     infos[i] = Nothing()
#                 elif i == 1:
#                     rewards[i] = 0
#                     rewards[i] += rewardf1
#                     dones[i] = False
#                     infos[i] = Nothing()

#                 else:
#                     rewards[i] = 0
#                     rewards[i] += rewardf2
#                     dones[i] = False
#                     infos[i] = Nothing()

#         # update all agents
#         # for i, action in enumerate(actions):
#         #     self.robots[i].step(action)
#         # self.states.append([robot.get_full_state() for robot in self.robots])
#         # self.global_time += self.time_step
#         # obs = self.compute_observations()
#         # return obs, rewards, dones, infos

#         if update:
#             # store state, action value and attention weights
#             self.states.append([self.robots[0].get_full_state(), self.robots[1].get_full_state() ,self.robots[2].get_full_state(),[human.get_full_state() for human in self.humans]])
#             # update all agents
#             for i, action in enumerate(actions):
#                 self.robots[i].step(action)
#             for i, human_action in enumerate(human_actions):
#                 self.humans[i].step(human_action)
#             self.global_time += self.time_step

#             # compute the observation
#             obs = [[] for _ in range(len(self.robots))]
#             for i in range(len(self.robots)):
#                 if self.robots[i].sensor == 'coordinates':
#                     for human in self.humans:
#                         obs[i].append(human.get_observable_state())
#                     for j in range(len(self.robots)):
#                         if j == i:
#                             continue
#                         obs[i].append(self.robots[j].get_observable_state())

#                 elif self.robots[i].sensor == 'RGB':
#                     raise NotImplementedError

# #done表示碰撞或者到达终点，info信息
#         return obs, rewards, dones, infos
    def step(self, actions, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
         行人orca
        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            human_actions.append(human.act(ob))

        # collision detection   人与机器人
        dmins = [float('inf')]*3
        collisions = [False]*3

        error1 = norm([self.robots[1].px - self.robots[0].px + 1.5, self.robots[1].py - self.robots[0].py + math.sqrt(1.75)])
        error2 = norm([self.robots[2].px - self.robots[0].px - 1.5, self.robots[2].py - self.robots[0].py + math.sqrt(1.75)])

        rewardf1 = 0
        rewardf2 = 0

        if error1 >= 0 and error1 <= 0.2:
            rewardf1 = 1
        elif error1 > 0.2 and error1 <= 1:
            rewardf1 = -np.tanh(7.5 * error1 - 3)
        elif error1 > 1 and error1 <= 2:
            rewardf1 = -1
        elif error1 > 2:
            rewardf1 = error1 * -1

        if error2 >= 0 and error2 <= 0.2:
            rewardf2 = 1
        elif error2 > 0.2 and error2 <= 1:
            rewardf2 = -np.tanh(7.5 * error2 - 3)
        elif error2 > 1 and error2 <= 2:
            rewardf2 = -1
        elif error2 > 2:
            rewardf2 = error2 * -1

        #robot与人类的碰撞
        for j in range(len(self.robots)):
        # for i, human in enumerate(self.humans):
            # 获取列表中的元素索引 i 和相应的人类对象。
            # for j in range(len(self.robots)):
            for i, human in enumerate(self.humans):
                # print(human.px)
                # print(self.robots[j])
                px = human.px - self.robots[j].px
                py = human.py - self.robots[j].py
                if self.robots[j].kinematics == 'holonomic':
                    vx = human.vx - actions[j].vx
                    vy = human.vy - actions[j].vy
                else:
                    vx = human.vx - actions[j].v * np.cos(actions[j].r + self.robots[j].theta)
                    vy = human.vy - actions[j].v * np.sin(actions[j].r + self.robots[j].theta)
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step

                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robots[j].radius
                if closest_dist < 0:
                    collisions[j] = True
                    # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    break
                elif closest_dist < dmins[j]:
                    #dim一开始位无穷大
                    dmins[j] = closest_dist

        # collision detection between robots
        R1collisionR2 = False
        dx = self.robots[0].px - self.robots[1].px
        dy = self.robots[0].py - self.robots[1].py
        dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[0].radius - self.robots[1].radius
        if dist < 0:
            R1collisionR2 = True
            # detect collision but don't take robots' collision into account
            logging.debug('Collision happens between robots in step()')

        R1collisionR3 = False
        dx = self.robots[0].px - self.robots[2].px
        dy = self.robots[0].py - self.robots[2].py
        dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[0].radius - self.robots[2].radius
        if dist < 0:
            R1collisionR3 = True
            # detect collision but don't take robots' collision into account
            logging.debug('Collision happens between robots in step()')

        R2collisionR3 = False
        dx = self.robots[1].px - self.robots[2].px
        dy = self.robots[1].py - self.robots[2].py
        dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[1].radius - self.robots[2].radius
        if dist < 0:
            R2collisionR3 = True
            # detect collision but don't take robots' collision into account
            logging.debug('Collision happens between robots in step()')

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        end_position1 = np.array(self.robots[0].compute_position(actions[0], self.time_step))
        reaching_goal1 = norm(end_position1 - np.array(self.robots[0].get_goal_position())) < self.robots[0].radius
        end_position2 = np.array(self.robots[1].compute_position(actions[1], self.time_step))
        reaching_goal2 = norm(end_position2 - np.array(self.robots[1].get_goal_position())) < self.robots[1].radius
        end_position3 = np.array(self.robots[2].compute_position(actions[2], self.time_step))
        reaching_goal3 = norm(end_position3 - np.array(self.robots[2].get_goal_position())) < self.robots[2].radius

        if self.global_time >= self.time_limit - 1:
            reward1 = 0
        elif collisions[0]:
            reward1 = self.collision_penalty
        elif reaching_goal1:
            if self.flag1:
                reward1 = 0
            else:
                reward1 = self.success_reward
            self.flag1 += 1
        elif dmins[0] < self.discomfort_dist:
            reward1 = (dmins[0] - self.discomfort_dist) * 2.5
        else:
            last1, end1 = self.onestep_distance(self.robots[0], actions[0])
            reward1 = (last1 - end1) * 2

        if self.global_time >= self.time_limit - 1:
            reward2 = 0
            reward2 += rewardf1
        elif collisions[1] or R1collisionR2 or R2collisionR3:
            reward2 = -33.5
            reward2 += rewardf1
        elif reaching_goal2:
            if self.flag2:
                reward2 = 0
            else:
                reward2 = 0
                reward2 += rewardf1
            self.flag2 += 1
        elif dmins[1] < self.discomfort_dist:
            reward2 = (dmins[1] - self.discomfort_dist) * 2.5
        else:
            reward2 = 0
            reward2 += rewardf1

        if self.global_time >= self.time_limit - 1:
            reward3 = 0
            reward3 += rewardf2
        elif collisions[2] or R2collisionR3 or R1collisionR3:
            reward3 = -33.5
            reward3 += rewardf2
        elif reaching_goal3:
            if self.flag3:
                reward3 = 0
            else:
                reward3 = 0
                reward3 += rewardf2
            self.flag3 += 1
        elif dmins[2] < 0.2:
            reward3 = (dmins[2] - self.discomfort_dist) * 2.5
        else:
            reward3 = 0
            reward3 += rewardf2

        if self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collisions[0] or collisions[1] or collisions[2] or R1collisionR2 or R2collisionR3 or R1collisionR3:
            done = True
            info = Collision()
        elif reaching_goal1:
            done = True
            info = ReachGoal()
        else:
            done = False
            info = Nothing()

        rewards = [reward1, reward2, reward3]
        dones= [done,done,done]
        infos=[info,info,info]

        if update:
            # store state, action value and attention weights
            self.states.append([self.robots[0].get_full_state(), self.robots[1].get_full_state(), self.robots[2].get_full_state(), [human.get_full_state() for human in self.humans]])
            # update all agents
            for i, action in enumerate(actions):
                self.robots[i].step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step

            # compute the observation
            obs = [[] for _ in range(len(self.robots))]
            for i in range(len(self.robots)):
                if self.robots[i].sensor == 'coordinates':
                    for human in self.humans:
                        obs[i].append(human.get_observable_state())
                    for j in range(len(self.robots)):
                        if j == i:
                            continue
                        obs[i].append(self.robots[j].get_observable_state())

                elif self.robots[i].sensor == 'RGB':
                    raise NotImplementedError

        return obs, rewards, dones, infos

    def step_new(self, actions, update=True):
        human_actions = []
        for human in self.humans:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robots[0].visible: ob += [self.robots[0].get_observable_state()]
            if self.robots[1].visible: ob += [self.robots[1].get_observable_state()]
            if self.robots[2].visible: ob += [self.robots[2].get_observable_state()]
            human_actions.append(human.act(ob))

        # --- 编队计算：动态统一目标与队形 ---
        goal0 = np.array(self.robots[0].get_goal_position())
        goal1 = np.array(self.robots[1].get_goal_position())
        goal2 = np.array(self.robots[2].get_goal_position())
        desired_rel_pos_10 = goal1 - goal0
        desired_rel_pos_20 = goal2 - goal0
        desired_rel_pos_21 = goal2 - goal1

        pos0 = np.array([self.robots[0].px, self.robots[0].py])
        pos1 = np.array([self.robots[1].px, self.robots[1].py])
        pos2 = np.array([self.robots[2].px, self.robots[2].py])
        error_01 = norm((pos1 - pos0) - desired_rel_pos_10)
        error_02 = norm((pos2 - pos0) - desired_rel_pos_20)
        error_12 = norm((pos2 - pos1) - desired_rel_pos_21)

        # ## 采用您原始Follower模型中被验证有效的编队奖励函数 ##
        def calculate_formation_reward(error):
            if error >= 0 and error <= 0.2:
                return 1.0
            elif error > 0.2 and error <= 1:
                return -np.tanh(7.5 * error - 3)
            elif error > 1 and error <= 2:
                return -1.0
            else:
                return -error

        reward_f_01 = calculate_formation_reward(error_01)
        reward_f_02 = calculate_formation_reward(error_02)
        reward_f_12 = calculate_formation_reward(error_12)

        # --- 碰撞检测部分 (完整代码) ---
        dmins = [float('inf')] * 3
        collisions = [False] * 3
        for j in range(len(self.robots)):
            for i, human in enumerate(self.humans):
                px = human.px - self.robots[j].px
                py = human.py - self.robots[j].py
                if self.robots[j].kinematics == 'holonomic':
                    vx = human.vx - actions[j].vx
                    vy = human.vy - actions[j].vy
                else:
                    vx = human.vx - actions[j].v * np.cos(actions[j].r + self.robots[j].theta)
                    vy = human.vy - actions[j].v * np.sin(actions[j].r + self.robots[j].theta)
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robots[j].radius
                if closest_dist < 0:
                    collisions[j] = True
                    break
                elif closest_dist < dmins[j]:
                    dmins[j] = closest_dist

        for i, j in product(range(len(self.humans)), range(len(self.humans))):
            if i == j: continue
            dx = self.humans[i].px - self.humans[j].px
            dy = self.humans[i].py - self.humans[j].py
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
            if dist < 0: logging.debug('Collision happens between humans in step()')

        Rcollisions = [False] * 3
        robot_pairs = [(0, 1), (0, 2), (1, 2)]
        for r1_idx, r2_idx in robot_pairs:
            dx = self.robots[r1_idx].px - self.robots[r2_idx].px
            dy = self.robots[r1_idx].py - self.robots[r2_idx].py
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robots[r1_idx].radius - self.robots[r2_idx].radius
            if dist < 0:
                Rcollisions[r1_idx] = True
                Rcollisions[r2_idx] = True
                logging.debug(f'Collision happens between robot {r1_idx} and {r2_idx} in step()')

        # --- 奖励计算核心区域 ---
        rewards = [0] * len(self.robots)
        infos = [None] * len(self.robots)
        dones = [0] * len(self.robots)
        for i, action in enumerate(actions):
            robot = self.robots[i]

            # ## 采用您原始Leader模型中被验证有效的导航奖励，不缩放 ##
            end_position = np.array(robot.compute_position(action, self.time_step))
            cur_dist = norm(np.array([robot.px, robot.py]) - np.array(robot.get_goal_position()))
            end_dist = norm(end_position - np.array(robot.get_goal_position()))
            navigation_reward = 6 * (cur_dist - end_dist)

            # ## 采用您原始Follower模型中被验证有效的编队奖励，不缩放 ##
            formation_reward = 0
            formation_weight = 0.5
            if i == 0:
                formation_reward = formation_weight * (reward_f_01 + reward_f_02) / 2.0
            elif i == 1:
                formation_reward = formation_weight * (reward_f_01 + reward_f_12) / 2.0
            else:
                formation_reward = formation_weight * (reward_f_12 + reward_f_02) / 2.0

            reaching_goal = norm(end_position - np.array(robot.get_goal_position())) < robot.radius

            if self.global_time >= self.time_limit - 1:
                # 唯一的关键修正：施加超时惩罚
                rewards[i] = -33.5
                dones[i] = True
                infos[i] = Timeout()
            elif collisions[i] or Rcollisions[i]:
                # 采用Leader的碰撞惩罚
                rewards[i] = self.collision_penalty + formation_reward - 33.5
                dones[i] = True
                infos[i] = Collision()
            elif reaching_goal:
                if not self.flag[i]:
                    # 采用Leader的成功奖励
                    rewards[i] = 35 + formation_reward
                    self.flag[i] = True
                else:
                    rewards[i] = 0.5 + formation_reward
                dones[i] = True
                infos[i] = ReachGoal()
            elif dmins[i] < self.discomfort_dist:
                # 采用统一的危险惩罚
                discomfort_penalty = (dmins[i] - self.discomfort_dist) * self.discomfort_penalty_factor
                rewards[i] = discomfort_penalty + formation_reward
                dones[i] = False
                infos[i] = Danger(dmins[i])
            else:
                # 正常行进: 结合了Leader的导航奖励和Follower的编队奖励
                time_penalty = -0.1
                rewards[i] = navigation_reward + formation_reward + time_penalty
                dones[i] = False
                infos[i] = Nothing()

        if update:
            self.states.append(
                [self.robots[0].get_full_state(), self.robots[1].get_full_state(), self.robots[2].get_full_state(),
                 [human.get_full_state() for human in self.humans]])
            for i, action in enumerate(actions):
                self.robots[i].step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step

            obs = [[] for _ in range(len(self.robots))]
            for i in range(len(self.robots)):
                if self.robots[i].sensor == 'coordinates':
                    for human in self.humans:
                        obs[i].append(human.get_observable_state())
                    for j in range(len(self.robots)):
                        if j == i:
                            continue
                        obs[i].append(self.robots[j].get_observable_state())
                elif self.robots[i].sensor == 'RGB':
                    raise NotImplementedError

        return obs, rewards, dones, infos

    def onestep_distance(self, robot, action):

        robot_state = robot.get_full_state()
        last_position = np.array([robot_state.px, robot_state.py])
        end_position = np.array(robot.compute_position(action, self.time_step))
        last_distance = norm(last_position - np.array(robot.get_goal_position()))
        end_distance = norm(end_position - np.array(robot.get_goal_position()))
        return last_distance, end_distance

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot0_color = 'green'
        robot1_color = 'yellow'
        robot2_color = 'black'
        goal1_color = 'green'
        goal2_color = 'yellow'
        goal3_color = 'black'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            for robot in self.robots:
                ax.add_artist(plt.Circle(robot.get_position(), robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot1_positions = [self.states[i][0].position for i in range(len(self.states))]
            robot2_positions = [self.states[i][1].position for i in range(len(self.states))]
            robot3_positions = [self.states[i][2].position for i in range(len(self.states))]
            human_positions = [[self.states[i][3][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot0 = plt.Circle(robot1_positions[k], self.robots[0].radius, fill=True, color=robot0_color)
                    robot1 = plt.Circle(robot2_positions[k], self.robots[1].radius, fill=True, color=robot1_color)
                    robot2 = plt.Circle(robot3_positions[k], self.robots[2].radius, fill=True, color=robot2_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot0)
                    ax.add_artist(robot1)
                    ax.add_artist(robot2)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = self.humans + self.robots
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 2)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction1 = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot0_color, ls='solid')
                    nav_direction2 = plt.Line2D((self.states[k - 1][1].px, self.states[k][1].px),
                                               (self.states[k - 1][1].py, self.states[k][1].py),
                                               color=robot1_color, ls='solid')
                    nav_direction3 = plt.Line2D((self.states[k - 1][2].px, self.states[k][2].px),
                                                (self.states[k - 1][2].py, self.states[k][2].py),
                                                color=robot2_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][3][i].px, self.states[k][3][i].px),
                                                   (self.states[k - 1][3][i].py, self.states[k][3][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction1)
                    ax.add_artist(nav_direction2)
                    ax.add_artist(nav_direction3)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([self.robot1], ['Robot1'], fontsize=16)
            plt.legend([self.robot2], ['Robot2'], fontsize=16)
            plt.legend([self.robot3], ['Robot3'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-8, 6)
            ax.set_ylim(-8, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal 添加机器人及其目标
            robot0_positions = [state[0].position for state in self.states]
            robot1_positions = [state[1].position for state in self.states]
            robot2_positions = [state[2].position for state in self.states]
            goal0 = mlines.Line2D([0], [4], color=goal1_color, marker='*', linestyle='None', markersize=15, label='Goal1')
            goal1 = mlines.Line2D([-2], [2], color=goal2_color, marker='*', linestyle='None', markersize=15, label='Goal2')
            goal2 = mlines.Line2D([2], [2], color=goal3_color, marker='*', linestyle='None', markersize=15, label='Goal3')
            robot0 = plt.Circle(robot0_positions[0], self.robots[0].radius, fill=True, color=robot0_color)
            robot1 = plt.Circle(robot1_positions[0], self.robots[1].radius, fill=True, color=robot1_color)
            robot2 = plt.Circle(robot2_positions[0], self.robots[2].radius, fill=True, color=robot2_color)
            ax.add_artist(robot0)
            ax.add_artist(robot1)
            ax.add_artist(robot2)
            ax.add_artist(goal0)
            ax.add_artist(goal1)
            ax.add_artist(goal2)
            plt.legend([robot0, robot1, robot2, goal0, goal1, goal2], ['Robot0', 'Robot1', 'Robot2', 'Goal0', 'Goal1',
                                                                       'Goal2'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[3][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction 计算每个步骤的方向，并使用箭头指示方向
            radius = 0.3
            if self.robots[0].kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                # print(self.states[0])
                for i in range(self.human_num + 3):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        elif i == 1:
                            agent_state = state[1]
                        elif i == 2:
                            agent_state = state[2]
                        else:
                            # print(state[3])
                            agent_state = state[3][i - 3]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]

            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot0.center = robot0_positions[frame_num]
                robot1.center = robot1_positions[frame_num]
                robot2.center = robot2_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation1[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation1 in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)

                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robots[0].kinematics == 'holonomic'
                assert self.robots[1].kinematics == 'holonomic'
                assert self.robots[2].kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1] \
                             + self.states[global_step][2] + self.states[global_step][3]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot 当按下任何键时，绘制动作值图
                fig, axis = plt.subplots()
                speeds = [0] + self.robots[0].policy.speeds
                rotations = self.robot1.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][3:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robots[0].policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError