# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ========= 路径配置 =========
PER_AGENT_DIR = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent"
OUT_DIR       = r"D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\data\bbb\per_agent"


RM_CKPT_PATH  = r"D:\Python\pycharm\python-learn\transformer\mat\scripts\train_reward\crowd\multi\MultiPrefTransformerDivide\train_optimized\modelsreward_model_15.pt"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 推理/写回选项 =========
SEGMENT_LEN   = 15
BATCH_SIZE    = 256
USE_CENTER    = False       # True 用窗口中点；False 用窗口末步
SMOOTH_AVG    = True        # True 对覆盖该时刻的所有窗口做平均
WRITE_FIELD   = "rewards_rm"

# 写入“每个 agent 奖励”还是“全局奖励”
MODE = "per_agent"          # "per_agent" | "global"
AGG_GLOBAL = "sum"          # 当 MODE="global" 时，"mean" 或 "sum" 聚合 N 个 agent

# ========= 导入 RM 类 =========
from mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel as RMClass
from mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer  # 只用来拿默认配置


# ==================== I/O 辅助 ====================

def load_per_agent_three(h5_dir):
    paths = [os.path.join(h5_dir, f"robot{i}.hdf5") for i in (1, 2, 3)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文件: {p}")
    datas = []
    for p in paths:
        with h5py.File(p, "r") as f:
            datas.append({
                "states":      f["states"][:],      # (Ni, Dsi)
                "actions":     f["actions"][:],     # (Ni, Da)
                "next_states": f["next_states"][:], # (Ni, Dsi)
                "terminals":   f["terminals"][:],   # (Ni,)
                "rewards":     f["rewards"][:],     # (Ni,)
            })
    # 对齐最小长度（按时间对齐）
    N = min(d["states"].shape[0] for d in datas)
    for d in datas:
        for k in d:
            d[k] = d[k][:N]
    return datas, N, paths


def build_tensor_arrays(datas):
    n_agent = len(datas)
    obs_dims = [d["states"].shape[1] for d in datas]
    act_dims = [d["actions"].shape[1] for d in datas]
    assert len(set(obs_dims)) == 1, f"三名智能体的状态维度不一致：{obs_dims}"
    assert len(set(act_dims)) == 1, f"三名智能体的动作维度不一致：{act_dims}"
    obs_dim, act_dim = obs_dims[0], act_dims[0]

    states = np.stack([d["states"] for d in datas], axis=1)   # (N, N_agent, obs_dim)
    actions = np.stack([d["actions"] for d in datas], axis=1) # (N, N_agent, act_dim)
    terminals = datas[0]["terminals"].astype(bool)            # (N,)
    return states, actions, terminals, obs_dim, act_dim, n_agent


def split_episodes_by_done(terminals):
    """根据 done 拆分 episode，包含末尾未终止段。"""
    N = len(terminals)
    idxs = np.where(terminals)[0].tolist()
    spans, last = [], 0
    for t in idxs:
        spans.append((last, t))
        last = t + 1
    if last < N:
        spans.append((last, N - 1))
    if len(spans) == 0:
        spans = [(0, N - 1)]
    return spans


def sliding_windows_episode(X, L):
    """单个 episode 上生成滑窗 (W, L, ...) 与其末索引。"""
    T = X.shape[0]
    if T < L:
        return np.empty((0,) + (L,) + X.shape[1:]), []
    wins, ends = [], []
    for end in range(L - 1, T):
        st = end - (L - 1)
        wins.append(X[st:end + 1])
        ends.append(end)
    return np.stack(wins, axis=0), ends


def windows_over_dataset(states, actions, terminals, L):
    spans = split_episodes_by_done(terminals)
    s_wins, a_wins, end_ids = [], [], []
    for (st, ed) in spans:
        s_epi = states[st:ed+1]   # (T, N, obs)
        a_epi = actions[st:ed+1]  # (T, N, act)
        sw, ends = sliding_windows_episode(s_epi, L)  # (W, L, N, obs)
        aw, _    = sliding_windows_episode(a_epi, L)  # (W, L, N, act)
        if len(ends) > 0:
            s_wins.append(sw); a_wins.append(aw)
            end_ids.extend([st + e for e in ends])
    if len(end_ids) == 0:
        return (np.empty((0, L) + states.shape[1:]),
                np.empty((0, L) + actions.shape[1:]),
                [])
    return np.concatenate(s_wins, axis=0), np.concatenate(a_wins, axis=0), end_ids


# ==================== 奖励模型加载 ====================

def build_rm_from_ckpt(ckpt_path, observation_dim, action_dim, n_agent, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到奖励模型权重：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    seq_len = int(ckpt.get("seq_len", SEGMENT_LEN))

    config = MultiPrefTransformer.get_default_config({
        'embd_dim': 256,
        'action_embd_dim': 64,
        'n_layer': 2,
        'n_head': 8,
        'atten_dropout': 0.2,
        'resid_dropout': 0.2,
        'pref_attn_embd_dim': 256,
        'medium_process_type': 'cat',
        'use_weighted_sum': True,
        'reverse_state_action': False,
        'agent_individual': False,
        'use_dropout': True,
        'use_lstm': False,
        'add_obs_action': False,
        'drop_agent_layer': False,
        'use_highway': False,
        'encoder_mlp': False,
        'decoder_mlp': False,
        'agent_layer_mlp': False,
        'time_layer_mlp': False,
        'layer_norm_epsilon': 1e-5,
    })

    model = RMClass(
        config=config,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_agent=n_agent,
        action_type='Continuous',
        max_episode_steps=seq_len,
        device=device,
    )
    # 兼容不同 key
    state = ckpt.get("reward_model", ckpt.get("model", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ==================== 前向与汇总 ====================

def _forward_step(model, sb, ab):
    """
    sb: (B, L, N, obs_dim)
    ab: (B, L, N, act_dim)
    返回:
      rewards_per_agent: (B, L, N)
    """
    B, L = sb.shape[0], sb.shape[1]
    timesteps = torch.arange(L, device=sb.device).unsqueeze(0).repeat(B, 1)  # (B, L)

    # 直接调用模型（训练时保存的是 trans 的权重）
    out = model(sb, ab, timesteps)  # 预期 (B, L, N, 1) 或 (B, L, N)

    if isinstance(out, dict):
        for k in ("pred", "logits", "rewards", "reward", "out"):
            if k in out:
                out = out[k]
                break

    if out.dim() == 4 and out.size(-1) == 1:
        out = out[..., 0]  # (B, L, N)

    assert out.dim() == 3, f"模型输出形状不符，得到 {tuple(out.shape)}，期望 (B, L, N[ ,1])"
    return out


@torch.no_grad()
def infer_rewards(model, states, actions, terminals, device):
    """
    states:  (N, N_agent, obs_dim)
    actions: (N, N_agent, act_dim)
    terminals: (N,)
    返回:
      per_agent: (N, N_agent)  —— 每时刻每个 agent 的奖励
      global_r : (N,)          —— 全局奖励（sum/mean）
    """
    N = states.shape[0]
    L = SEGMENT_LEN
    n_agent = states.shape[1]

    s_win, a_win, end_ids = windows_over_dataset(states, actions, terminals, L)
    if s_win.shape[0] == 0:
        return np.zeros((N, n_agent), dtype=np.float32), np.zeros(N, dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(s_win).float(),
                       torch.from_numpy(a_win).float())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    r_sum = np.zeros((N, n_agent), dtype=np.float32)
    r_cnt = np.zeros((N, n_agent), dtype=np.int32)

    pos = (L // 2) if USE_CENTER else (L - 1)
    ptr = 0
    model.eval()

    for (sb, ab) in loader:
        sb = sb.to(device)  # (B, L, N, obs)
        ab = ab.to(device)  # (B, L, N, act)

        r_step = _forward_step(model, sb, ab).cpu().numpy()  # (B, L, N)
        B = r_step.shape[0]
        ends = end_ids[ptr: ptr + B]
        ptr += B

        if SMOOTH_AVG:
            for i in range(B):
                e = ends[i]
                for off in range(L):
                    t = e - (L - 1 - off)
                    r_sum[t, :] += r_step[i, off, :]
                    r_cnt[t, :] += 1
        else:
            for i in range(B):
                e = ends[i]
                t = e - (L - 1 - pos)
                r_sum[t, :] += r_step[i, pos, :]
                r_cnt[t, :] += 1

    per_agent = np.zeros((N, n_agent), dtype=np.float32)
    mask = r_cnt > 0
    per_agent[mask] = r_sum[mask] / r_cnt[mask]

    # # 关键消融逻辑 (Average Credit): 强制所有智能体共享平均分
    # # 1. 计算当前时刻所有智能体的平均分: shape (N,)
    # avg_reward = per_agent.mean(axis=1, keepdims=True)
    #
    # # 2. 将平均分广播给所有智能体: shape (N, n_agent)
    # per_agent = np.repeat(avg_reward, n_agent, axis=1)

    if AGG_GLOBAL == "sum":
        global_r = per_agent.sum(axis=1)
    else:
        global_r = per_agent.mean(axis=1)

    return per_agent, global_r


# ==================== 写回（避免“同文件同时读写”） ====================

def safe_out_path(in_path, out_dir, suffix="_rm"):
    base = os.path.basename(in_path)
    name, ext = os.path.splitext(base)
    return os.path.join(out_dir, f"{name}{suffix}{ext}")


# ==================== 主流程 ====================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    datas, N, in_paths = load_per_agent_three(PER_AGENT_DIR)
    states, actions, terminals, obs_dim, act_dim, n_agent = build_tensor_arrays(datas)

    model = build_rm_from_ckpt(
        RM_CKPT_PATH,
        observation_dim=obs_dim,
        action_dim=act_dim,
        n_agent=n_agent,
        device=torch.device(DEVICE),
    )

    per_agent_r, global_r = infer_rewards(model, states, actions, terminals, DEVICE)

    # 写回：为避免 “Unable to create file (unable to truncate a file which is already open)”
    # 统一写到新文件（原名 + _rm 后缀）
    for agent_idx, in_path in enumerate(in_paths):
        base = os.path.basename(in_path)
        out_path = safe_out_path(in_path, OUT_DIR, suffix="_rm")

        with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
            # 复制所有字段
            for k in fin.keys():
                fin.copy(k, fout, name=k)
            # 写奖励
            if WRITE_FIELD in fout:
                del fout[WRITE_FIELD]
            if MODE == "per_agent":
                fout.create_dataset(WRITE_FIELD, data=per_agent_r[:, agent_idx].astype(np.float32))
            else:
                fout.create_dataset(WRITE_FIELD, data=global_r.astype(np.float32))

        print(f"[OK] Wrote {out_path} with {WRITE_FIELD} [{MODE}]")

if __name__ == "__main__":
    main()
