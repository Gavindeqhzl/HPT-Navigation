import h5py
import numpy as np
from sklearn.model_selection import train_test_split

def load_hdf5_dataset(dataset_path, test_size=0.1):
    """
    读取 multi_agent_dataset.hdf5 并返回训练集、验证集和环境信息
    dataset_path: HDF5 文件路径
    test_size: 验证集比例
    """
    states_A, actions_A, states_B, actions_B, labels = [], [], [], [], []

    with h5py.File(dataset_path, 'r') as f:
        for grp_name in f.keys():
            grp = f[grp_name]
            states_A.append(grp['state_A'][:])   # shape (SEG_LEN, obs_dim_total)
            actions_A.append(grp['action_A'][:]) # shape (SEG_LEN, act_dim_total)
            states_B.append(grp['state_B'][:])
            actions_B.append(grp['action_B'][:])
            labels.append(grp.attrs['label'])    # 偏好标签 0/1

    states_A = np.array(states_A)  # (N, SEG_LEN, obs_dim_total)
    actions_A = np.array(actions_A)
    states_B = np.array(states_B)
    actions_B = np.array(actions_B)
    labels = np.array(labels, dtype=np.int64)

    # 计算维度信息
    N, SEG_LEN, obs_dim_total = states_A.shape
    act_dim_total = actions_A.shape[2]

    n_agent = 3  # 固定 3 个智能体
    obs_dim = obs_dim_total // n_agent
    act_dim = act_dim_total // n_agent

    # reshape 成 (N, SEG_LEN, n_agent, obs_dim)
    states_A = states_A.reshape(N, SEG_LEN, n_agent, obs_dim)
    states_B = states_B.reshape(N, SEG_LEN, n_agent, obs_dim)
    actions_A = actions_A.reshape(N, SEG_LEN, n_agent, act_dim)
    actions_B = actions_B.reshape(N, SEG_LEN, n_agent, act_dim)

    # 生成时间步 (N, SEG_LEN)，0,1,2,...,SEG_LEN-1
    timesteps = np.tile(np.arange(SEG_LEN), (N, 1))

    dataset = {
        "observations0": states_A,
        "actions0": actions_A,
        "observations1": states_B,
        "actions1": actions_B,
        "timesteps0": timesteps,
        "timesteps1": timesteps,
        "labels": labels
    }

    train_idx, val_idx = train_test_split(np.arange(N), test_size=test_size, random_state=42)
    train_dataset = {k: v[train_idx] for k, v in dataset.items()}
    eval_dataset = {k: v[val_idx] for k, v in dataset.items()}

    env_info = {
        "observation_dim": obs_dim,
        "action_dim": act_dim,
        "max_len": SEG_LEN,
        "n_agent": n_agent
    }

    return train_dataset, eval_dataset, env_info
