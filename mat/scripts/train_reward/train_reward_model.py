import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import absl.app
import absl.flags
import transformers

sys.path.append("../../")

from mat.algorithms.reward_model.models.MR import MR
from mat.algorithms.reward_model.models.NMR import NMR
from mat.algorithms.reward_model.models.lstm import LSTMRewardModel
from mat.algorithms.reward_model.models.PrefTransformer import PrefTransformer
from mat.algorithms.reward_model.models.trajectory_gpt2 import TransRewardModel
from mat.algorithms.reward_model.models.q_function import FullyConnectedQFunction
from mat.algorithms.reward_model.models.torch_utils import batch_to_torch, index_batch
from mat.algorithms.reward_model.models.MultiPrefTransformer import MultiPrefTransformer
from mat.algorithms.reward_model.models.encoder_decoder_divide import MultiTransRewardDivideModel
from mat.algorithms.reward_model.utils.utils import (
    Timer, define_flags_with_default, set_random_seed, prefix_metrics, WandBLogger
)

from hdf5_loader import load_hdf5_dataset

# ------------------------- FLAGS / CONFIG -------------------------
FLAGS_DEF = define_flags_with_default(
    env='crowd',
    task='multi',
    model_type='MultiPrefTransformerDivide',
    seed=42,
    save_model=True,

    # ====== Training Hyper-params ======
    batch_size=128,
    orthogonal_init=False,
    activations='relu',
    activation_final='none',
    n_epochs=50,
    eval_period=1,
    comment='train_optimized',
    max_traj_length=50,

    # ====== Model configs ======
    multi_transformer=MultiPrefTransformer.get_default_config({
        'trans_lr': 3e-5,
        'optimizer_type': 'adamw',
        'scheduler_type': 'CosineDecay',
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
    }),

    transformer=PrefTransformer.get_default_config(),
    reward=MR.get_default_config(),
    lstm=NMR.get_default_config(),
    logging=WandBLogger.get_default_config(),

    # ====== Runtime ======
    device='cuda',
    dataset_path=r'D:\Python\pycharm\python-learn\muti-crowdnav-lx-xian\crowd_nav\multi_agent_dataset.hdf5',
    model_dir=""
)

# ------------------------- Helpers -------------------------
def output_res(writer, train_infos, step):
    for k, v in train_infos.items():
        writer.add_scalars(k, {k: v}, step)

def compute_pairwise_accuracy(model, eval_dataset, batch_size, device):
    model.trans.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset["observations0"]), batch_size):
            obs0 = torch.tensor(eval_dataset['observations0'][i:i+batch_size], dtype=torch.float32, device=device)
            act0 = torch.tensor(eval_dataset['actions0'][i:i+batch_size], dtype=torch.float32, device=device)
            obs1 = torch.tensor(eval_dataset['observations1'][i:i+batch_size], dtype=torch.float32, device=device)
            act1 = torch.tensor(eval_dataset['actions1'][i:i+batch_size], dtype=torch.float32, device=device)
            labels = torch.tensor(eval_dataset['labels'][i:i+batch_size], dtype=torch.long, device=device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)

            T = obs0.shape[1]
            t0 = torch.arange(T, device=device).unsqueeze(0).repeat(obs0.shape[0], 1)
            t1 = torch.arange(T, device=device).unsqueeze(0).repeat(obs1.shape[0], 1)

            m0 = torch.tensor(eval_dataset.get('attn_mask0', np.ones((obs0.size(0), T, obs0.size(2)))), dtype=torch.float32, device=device)
            m1 = torch.tensor(eval_dataset.get('attn_mask1', np.ones((obs1.size(0), T, obs1.size(2)))), dtype=torch.float32, device=device)

            r0 = model.trans(obs0, act0, t0, attn_mask=m0).squeeze(-1)   # [B, T, N]
            r1 = model.trans(obs1, act1, t1, attn_mask=m1).squeeze(-1)

            s0 = (r0 * m0).sum(dim=(1,2)) / m0.sum(dim=(1,2)).clamp(min=1)
            s1 = (r1 * m1).sum(dim=(1,2)) / m1.sum(dim=(1,2)).clamp(min=1)

            pred = (s1 > s0).long()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"[Eval] Pairwise Accuracy: {acc * 100:.2f}%")
    return acc



# ------------------------- MAIN -------------------------
def main(_):
    FLAGS = absl.flags.FLAGS

    FLAGS.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logger dirs
    save_root = os.path.join(
        r"D:\Python\pycharm\python-learn\transformer\mat\scripts\train_reward",
        FLAGS.env, FLAGS.task, str(FLAGS.model_type), FLAGS.comment
    )
    FLAGS.logging.group = f"{FLAGS.env}_{FLAGS.model_type}_{FLAGS.comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    FLAGS.logging.log_dir = os.path.join(save_root, 'logs')
    FLAGS.logging.model_dir = os.path.join(save_root, 'models')
    os.makedirs(FLAGS.logging.log_dir, exist_ok=True)
    os.makedirs(FLAGS.logging.model_dir, exist_ok=True)
    writer = SummaryWriter(FLAGS.logging.log_dir)

    set_random_seed(FLAGS.seed)

    # Load dataset
    print(f"Loading dataset from {FLAGS.dataset_path}")
    pref_dataset, pref_eval_dataset, env_info = load_hdf5_dataset(FLAGS.dataset_path)

    train_size = pref_dataset["observations0"].shape[0]
    eval_size = pref_eval_dataset["observations0"].shape[0]
    print(f"Dataset loaded: train size {train_size}, eval size {eval_size}")

    observation_dim = env_info['observation_dim']
    action_dim = env_info['action_dim']
    n_agent = env_info['n_agent']
    FLAGS.max_traj_length = int(env_info['max_len'])

    print(f"Observation dim: {observation_dim}, Action dim: {action_dim}, n_agent: {n_agent}")

    interval = int(np.ceil(train_size / FLAGS.batch_size))
    eval_interval = int(np.ceil(eval_size / FLAGS.batch_size))

    # Build model
    if FLAGS.model_type == "MultiPrefTransformerDivide":
        trans = MultiTransRewardDivideModel(
            config=FLAGS.multi_transformer,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_agent=n_agent,
            action_type='Continuous',
            max_episode_steps=FLAGS.max_traj_length,
            device=FLAGS.device,
        )
        reward_model = MultiPrefTransformer(FLAGS.multi_transformer, trans, FLAGS.device)
        eval_loss_key = "reward/eval_trans_loss"
    else:
        raise NotImplementedError(f"Unknown model_type: {FLAGS.model_type}")

    # Training control
    best_eval = float('inf')
    best_acc = 0.0
    patience = 8
    bad_counter = 0

    for epoch in range(FLAGS.n_epochs):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch

        # Train
        shuffled_idx = np.random.permutation(train_size)
        pbar = tqdm(range(interval), desc=f"Epoch {epoch}", ncols=100)

        for i in pbar:
            start_pt = i * FLAGS.batch_size
            end_pt = min((i + 1) * FLAGS.batch_size, train_size)
            if start_pt >= end_pt:
                break

            with Timer():
                batch_np = index_batch(pref_dataset, shuffled_idx[start_pt:end_pt])
                batch = batch_to_torch(batch_np, FLAGS.device)
                step_metrics = reward_model.train(batch)
                for k, v in prefix_metrics(step_metrics, 'reward').items():
                    metrics[k].append(v)
                if 'trans_loss' in step_metrics:
                    pbar.set_postfix(loss=f"{float(step_metrics['trans_loss']):.4f}")

        # Eval
        if epoch % FLAGS.eval_period == 0:
            eval_vals = defaultdict(list)
            for j in range(eval_interval):
                s = j * FLAGS.batch_size
                e = min((j + 1) * FLAGS.batch_size, eval_size)
                if s >= e:
                    break
                batch_eval = batch_to_torch(index_batch(pref_eval_dataset, range(s, e)), FLAGS.device)
                eval_step_metrics = reward_model.evaluation(batch_eval)
                for k, v in prefix_metrics(eval_step_metrics, 'reward').items():
                    eval_vals[k].append(v)

            eval_means = {k: float(np.mean(v)) for k, v in eval_vals.items() if len(v) > 0}
            this_eval = eval_means.get(eval_loss_key, eval_means.get('reward/trans_loss', None))

            # Compute Pairwise Accuracy
            pairwise_acc = compute_pairwise_accuracy(reward_model, pref_eval_dataset, FLAGS.batch_size, FLAGS.device)

            print(f"[Eval] Loss={this_eval:.4f}, PairwiseAcc={pairwise_acc*100:.2f}%")

            if (this_eval < best_eval) or (this_eval == best_eval and pairwise_acc > best_acc):
                best_eval = min(best_eval, this_eval)
                best_acc = max(best_acc, pairwise_acc)
                bad_counter = 0
                print(f"✅ New best @ epoch {epoch} (loss or acc improved). Saving model to {FLAGS.logging.model_dir}")
                reward_model.save_model(FLAGS.logging.model_dir, epoch)
            else:
                bad_counter += 1
                if bad_counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch} (no improvement for {patience} evals)")
                    for key, val in list(metrics.items()):
                        if isinstance(val, list) and len(val) > 0:
                            metrics[key] = float(np.mean(val))
                    output_res(writer, metrics, epoch)
                    return

        # Log
        print(f"\nEpoch {epoch}")
        for key, val in list(metrics.items()):
            if isinstance(val, list) and len(val) > 0:
                metrics[key] = float(np.mean(val))
                print(f"{key}: {metrics[key]:.4f}")
        output_res(writer, metrics, epoch)

    print("Training finished.")

if __name__ == '__main__':
    absl.app.run(main)
