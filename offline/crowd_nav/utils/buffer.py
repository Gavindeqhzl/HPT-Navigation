import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from collections import defaultdict
from tqdm import trange

FILE51 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot1_rm.hdf5"
FILE52 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot2_rm.hdf5"
FILE53 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot3_rm.hdf5"

# FILE51 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot1_rm2.hdf5"
# FILE52 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot2_rm2.hdf5"
# FILE53 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot3_rm2.hdf5"

# FILE51 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot1.hdf5"
# FILE52 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot2.hdf5"
# FILE53 = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent\robot3.hdf5"


FILE71 = "/hd_2t/fuhao2024/lx/Crowd_nav-formation7_robot1.hdf5"
FILE72 = "/hd_2t/fuhao2024/lx/Crowd_nav-formation7_robot2.hdf5"
FILE73 = "/hd_2t/fuhao2024/lx/Crowd_nav-formation7_robot3.hdf5"

REWARD_KEY_PRIMARY = "rewards_rm"
REWARD_KEY_FALLBACK = "rewards"

def _read_rewards(h5file, slc=None):
    """优先读 rewards_rm，不存在则回退到 rewards。"""
    if REWARD_KEY_PRIMARY in h5file:
        arr = h5file[REWARD_KEY_PRIMARY][:]
    else:
        arr = h5file[REWARD_KEY_FALLBACK][:]
    return arr if slc is None else arr[slc]

class ReplayBuffer_episode:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
      
        self._device = device
        self.max_len = 100

        self.start_index = 0
        self.end_index = 0
        self._device = device
        self.state_dim = state_dim

        self.traj, self.traj_len = [], []

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def reset_tensors(self):
        self.e_states = torch.zeros((100, 3, self.state_dim), dtype=torch.float32, device=self._device)
        self.e_actions = torch.zeros((100, 3, 2), dtype=torch.float32, device=self._device)
        self.e_rewards = torch.zeros((100, 3, 1), dtype=torch.float32, device=self._device)
        self.e_next_states = torch.zeros((100, 3, self.state_dim), dtype=torch.float32, device=self._device)
        self.e_dones = torch.ones((100, 1), dtype=torch.float32, device=self._device)
        self.traj_mask = torch.zeros((100, 1), dtype=torch.float32, device=self._device)


    def load_h5py_dataset(self, human_num):

        if human_num == 5:
            f1 = h5py.File(FILE51, "r")
            f2 = h5py.File(FILE52, "r")
            f3 = h5py.File(FILE53, "r")
        else:
            f1 = h5py.File(FILE71, "r")
            f2 = h5py.File(FILE72, "r")
            f3 = h5py.File(FILE73, "r")

        assert f1["states"].shape[0] == f2["states"].shape[0] and f3["states"].shape[0] == f2["states"].shape[0]

        n_transitions = f1["states"].shape[0]
    
        self.ntransitions = n_transitions
        print(n_transitions)

        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        traj, traj_len = [], []

        self.reset_tensors()

        for i in range(n_transitions):

            if f1["terminals"][i]:

                self.end_index = i + 1

                episode_len = self.end_index - self.start_index

                s1 = self._to_tensor(np.array(f1["states"][self.start_index:self.end_index]))
                s2 = self._to_tensor(np.array(f2["states"][self.start_index:self.end_index]))
                s3 = self._to_tensor(np.array(f3["states"][self.start_index:self.end_index]))

                ma_state = torch.stack([s1, s2, s3], dim=1)

                self.e_states[:episode_len] = ma_state

                a1 = self._to_tensor(np.array(f1["actions"][self.start_index:self.end_index]))
                a2 = self._to_tensor(np.array(f2["actions"][self.start_index:self.end_index]))
                a3 = self._to_tensor(np.array(f3["actions"][self.start_index:self.end_index]))

                ma_a = torch.stack([a1, a2, a3], dim=1)
      
                self.e_actions[:episode_len] = ma_a

                r1 = self._to_tensor(np.array(f1["rewards"][self.start_index:self.end_index]).reshape(-1, 1))
                r2 = self._to_tensor(np.array(f2["rewards"][self.start_index:self.end_index]).reshape(-1, 1))
                r3 = self._to_tensor(np.array(f3["rewards"][self.start_index:self.end_index]).reshape(-1, 1))
                print(r1.shape, r2.shape, r3.shape)

                ma_r = torch.stack([r1, r2, r3], dim=1)
                print(ma_r.shape)

                self.e_rewards[:episode_len] = ma_r
                # self.e_rewards[:episode_len] = self._to_tensor(np.array(f1["rewards"][self.start_index:self.end_index]).reshape(-1, 1))

                n_s1 = self._to_tensor(np.array(f1["next_states"][self.start_index:self.end_index]))
                n_s2 = self._to_tensor(np.array(f2["next_states"][self.start_index:self.end_index]))
                n_s3 = self._to_tensor(np.array(f3["next_states"][self.start_index:self.end_index]))

                ma_n_state = torch.stack([n_s1, n_s2, n_s3], dim=1)
            
                self.e_next_states[:episode_len] = ma_n_state
            
                self.e_dones[:episode_len] = self._to_tensor(np.array(f1["terminals"][self.start_index:self.end_index]).reshape(-1, 1))

                self.traj_mask[:episode_len] = 1.0 

                self.start_index = self.end_index

                traj.append(np.array([self.e_states.clone(), self.e_actions.clone(), self.e_rewards.clone(), self.e_next_states.clone(), self.e_dones.clone(), self.traj_mask.clone()]))

                traj_len.append(episode_len)

                # reset 
                self.reset_tensors()

        print("len_traj: " , len(traj_len))
        max_value = max(traj_len)
        # 计算平均值
        average_value = sum(traj_len) / len(traj_len)
        # 输出结果
        print("最大轨迹长度:", max_value)
        print("平均轨迹长度:", average_value)

        self.traj = np.array(traj)
        self.traj_len = np.array(traj_len)

        f1.close()
        f2.close()
        f3.close()


    def episode_sample(self, batch_size):
        indices = np.random.randint(0, len(self.traj_len), size=batch_size)
        max_episode_len = self.traj_len[indices].max()
        # print(max_episode_len)

        sample_traj = self.traj[indices]
        
        # 初始化一个列表用于存储拼接后的张量
        concatenated_tensors = []

        # 遍历每一列
        for col in range(6):
            # 收集当前列的所有张量
            tensors_to_concat = [sample_traj[row, col][:max_episode_len] for row in range(batch_size)]
            
            # 拼接当前列的所有张量
            concatenated_tensor = torch.stack(tensors_to_concat, dim=0)
            
            # 将拼接后的张量添加到结果列表
            concatenated_tensors.append(concatenated_tensor)
                
        # print(states0.shape)
        return concatenated_tensors


    def episode_sample_next(self, batch_size):
        indices = np.random.randint(0, len(self.traj_len), size=batch_size)
        max_episode_len = self.traj_len[indices].max()
        # print(max_episode_len)

        sample_traj = self.traj[indices]
        
        # 初始化一个列表用于存储拼接后的张量
        concatenated_tensors = []
        next_action_list = []

        # 遍历每一列
        for col in range(6):
            # 收集当前列的所有张量
            tensors_to_concat = []
            
            for row in range(batch_size):
                sample = sample_traj[row, col][:max_episode_len]
                tensors_to_concat.append(sample)

                if col == 1:  #action
                    next_action = torch.zeros((max_episode_len, 3, 2), dtype=torch.float32, device=self._device)
                    next_action[:-1] = sample[1:]
                    next_action_list.append(next_action)

            # 拼接当前列的所有张量
            concatenated_tensor = torch.stack(tensors_to_concat, dim=0)
            
            # 将拼接后的张量添加到结果列表
            concatenated_tensors.append(concatenated_tensor)

        next_action_tensor = torch.stack(next_action_list, dim=0)
        concatenated_tensors.append(next_action_tensor)
                
        # print(states0.shape)
        return concatenated_tensors


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self.device = device
        self.state_dim = state_dim

        self._states0 = torch.zeros(
            (buffer_size, self.state_dim), dtype=torch.float32, device=device
        )
        self._states1 = torch.zeros(
            (buffer_size, self.state_dim), dtype=torch.float32, device=device
        )
        self._states2 = torch.zeros(
            (buffer_size, self.state_dim), dtype=torch.float32, device=device
        )
        self._actions0 = torch.zeros(
            (buffer_size, 2), dtype=torch.float32, device=device
        )
        self._actions1 = torch.zeros(
            (buffer_size, 2), dtype=torch.float32, device=device
        )
        self._actions2 = torch.zeros(
            (buffer_size, 2), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._rewards1 = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._rewards2 = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states0 = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._next_states1 = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._next_states2 = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._dones1 = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._dones2 = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    # data = h5py.File('/home/ubuntu/muti-crowdnav-lx-xian/crowd_nav/utils/Crowd_nav-salstm/Crowd_nav-salstm.hdf5', 'r')

    def load_h5py_dataset(self, human_num):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")

        # f = h5py.File(file_name, "r")
        if human_num == 5:
            f1 = h5py.File(FILE51, "r")
            f2 = h5py.File(FILE52, "r")
            f3 = h5py.File(FILE53, "r")
        else:
            f1 = h5py.File(FILE71, "r")
            f2 = h5py.File(FILE72, "r")
            f3 = h5py.File(FILE73, "r")
        n_transitions = f1["states"].shape[0]
        print(n_transitions)
        self.ntransitions = n_transitions
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self._states0[:n_transitions] = self._to_tensor(np.array(f1["states"]))
        self._states1[:n_transitions] = self._to_tensor(np.array(f2["states"]))
        self._states2[:n_transitions] = self._to_tensor(np.array(f3["states"]))

        self._actions0[:n_transitions] = self._to_tensor(np.array(f1["actions"]))
        self._actions1[:n_transitions] = self._to_tensor(np.array(f2["actions"]))
        self._actions2[:n_transitions] = self._to_tensor(np.array(f3["actions"]))

        # ✅ 奖励改为优先读 rewards_rm  ← 改
        self._rewards[:n_transitions] = self._to_tensor(_read_rewards(f1)[:, None])  # ← 改
        self._rewards1[:n_transitions] = self._to_tensor(_read_rewards(f2)[:, None])  # ← 改
        self._rewards2[:n_transitions] = self._to_tensor(_read_rewards(f3)[:, None])  # ← 改

        # self._rewards[:n_transitions] = self._to_tensor(np.array(f1["rewards"])[:, None])
        # self._rewards1[:n_transitions] = self._to_tensor(np.array(f2["rewards"])[:, None])
        # self._rewards2[:n_transitions] = self._to_tensor(np.array(f3["rewards"])[:, None])

        self._next_states0[:n_transitions] = self._to_tensor(np.array(f1["next_states"]))
        self._next_states1[:n_transitions] = self._to_tensor(np.array(f2["next_states"]))
        self._next_states2[:n_transitions] = self._to_tensor(np.array(f3["next_states"]))

        self._dones[:n_transitions] = self._to_tensor(np.array(f1["terminals"])[:, None])
        # self._dones1[:n_transitions] = self._to_tensor(np.array(f["terminal1"])[:, None])
        # self._dones2[:n_transitions] = self._to_tensor(np.array(f["terminal2"])[:, None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample0(self, batch_size: int,index=None):
        if index is None:
            indices = np.random.randint(0, self.ntransitions, size=batch_size)
        else:
            indices = index
        # indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        # indices = np.random.randint(0, self.ntransitions, size=batch_size)
        states0 = self._states0[indices]
        actions0 = self._actions0[indices]
        reward0 = self._rewards[indices]
        next_states0 = self._next_states0[indices]
        dones0 = self._dones[indices]
        # print(states0.shape)
        return [states0, actions0, reward0, next_states0, dones0],indices

    def sample1(self, batch_size: int,index=None):
        if index is None:
            indices = np.random.randint(0, self.ntransitions, size=batch_size)
        else:
            indices = index
        # indices = np.random.randint(0, self.ntransitions, size=batch_size)
        states1 = self._states1[indices]
        actions1 = self._actions1[indices]
        reward1 = self._rewards1[indices]
        next_states1 = self._next_states1[indices]
        # dones1 = self._dones[indices]
        # print(states1.shape)
        return [states1, actions1, reward1, next_states1],indices

    def sample2(self, batch_size: int,index=None):
        if index is None:
            indices = np.random.randint(0, self.ntransitions, size=batch_size)
        else:
            indices = index
        # indices = np.random.randint(0, self.ntransitions, size=batch_size)
        states2 = self._states2[indices]
        actions2 = self._actions2[indices]
        reward2 = self._rewards2[indices]
        next_states2 = self._next_states2[indices]
        # dones2 = self._dones[indices]
        return [states2, actions2, reward2, next_states2],indices


    def sample(self, batch_size):
        indices = np.random.randint(0, self.ntransitions, size=batch_size)
        [states0, actions0, reward0, next_states0, dones0],_ = self.sample0(batch_size, index = indices)
        [states1, actions1, reward1, next_states1],_ = self.sample1(batch_size, index = indices)
        [states2, actions2, reward2, next_states2],_ = self.sample2(batch_size, index = indices)

        state = torch.stack((states0, states1, states2), dim=1) #b, 78
        action = torch.stack((actions0, actions1, actions2), dim=1) #b, 2
        # state = torch.stack((states0, states1, states2), dim=0)
        reward = torch.stack((reward0, reward1, reward2), dim=1) #b, 1
        next_state = torch.stack((next_states0, next_states1, next_states2), dim=1)
        dones = dones0

        mask = torch.ones_like(dones).to(self.device)

        return [state, action, reward, next_state, dones, mask]