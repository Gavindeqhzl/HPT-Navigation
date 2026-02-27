import torch
import numpy as np
import torch.nn.functional as F
from ml_collections import ConfigDict
from mat.algorithms.reward_model.models.torch_utils import cross_ent_loss


class MultiPrefTransformer(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 3e-5
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.embd_dim = 256  # 确保 embd_dim 能被 n_head 整除
        config.action_embd_dim = 64
        config.n_layer = 2  # 增加层数，提高模型复杂度
        config.n_head = 8  # 增加头数，以便更好地进行多头注意力计算
        config.atten_dropout = 0.2
        config.resid_dropout = 0.2
        config.pref_attn_embd_dim = 256  # 确保一致
        config.weight_decay = 0.01
        config.label_smoothing = 0.05
        config.medium_process_type = 'cat'
        config.use_weighted_sum = True
        config.reverse_state_action = False
        config.agent_individual = False
        config.use_dropout = True
        config.use_lstm = False
        config.add_obs_action = False
        config.drop_agent_layer = False
        config.use_highway = False

        ############ add config for aboration
        config.encoder_mlp = False
        config.decoder_mlp = False

        ############ add config for MPTD aboration
        config.agent_layer_mlp = False
        config.time_layer_mlp = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config, trans, device):
        ####################### config basic info
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim
        self.device = device
        ####################### config optim
        optimizer_class = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
        }[self.config.optimizer_type]
        self.optimizer = optimizer_class(
            self.trans.parameters(),
            lr=self.config.trans_lr,
            weight_decay=getattr(self.config, 'weight_decay', 0.0),
        )
        ####################### config scheduler
        self.scheduler = {
            'CosineDecay': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10,  # not sure if setting of scheduler is correct
            ),
            'StepLR': torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.98,
            ),
            'none': None
        }[self.config.scheduler_type]
        ####################### config other record info
        self._total_steps = 0

    def train(self, batch):
        self.trans.train()
        self._total_steps += 1
        metrics = self._train_pref_step(batch)
        return metrics

    def _train_pref_step(self, batch):
        def loss_fn():
            obs_0 = batch['observations0'];
            act_0 = batch['actions0']
            obs_1 = batch['observations1'];
            act_1 = batch['actions1']
            timestep_0 = batch['timesteps0'];
            timestep_1 = batch['timesteps1']
            labels = torch.as_tensor(batch['labels'], dtype=torch.long, device=self.device)

            # 可能存在 one-hot/二维标签，这里统一成类别 id
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)
            B, T, N, _ = obs_0.shape

            # 可选：对称增强（打乱对）
            if np.random.rand() < 0.5:
                obs_0, obs_1 = obs_1, obs_0
                act_0, act_1 = act_1, act_0
                timestep_0, timestep_1 = timestep_1, timestep_0
                labels = 1 - labels

            # 取 mask（如果数据里没有，就用全 1）
            mask0 = batch.get('attn_mask0', None)
            mask1 = batch.get('attn_mask1', None)
            if mask0 is None: mask0 = torch.ones((B, T, N), device=self.device)
            if mask1 is None: mask1 = torch.ones((B, T, N), device=self.device)

            # 前向：把 mask 传进去！
            trans_pred_0 = self.trans(obs_0, act_0, timestep_0, attn_mask=mask0)
            trans_pred_1 = self.trans(obs_1, act_1, timestep_1, attn_mask=mask1)

            # 先沿 agent 再沿 time 求“有效步平均”分数
            score0 = self._aggregate_with_mask(trans_pred_0, mask0)  # [B]
            score1 = self._aggregate_with_mask(trans_pred_1, mask1)  # [B]
            logits = torch.stack([score0, score1], dim=1)  # [B, 2]

            # CE + label smoothing（默认 0.05，可在 FLAGS 覆盖为 0）
            trans_loss = F.cross_entropy(
                logits, labels, label_smoothing=getattr(self.config, 'label_smoothing', 0.0)
            )

            self.optimizer.zero_grad()
            trans_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trans.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            return {'trans_loss': trans_loss.detach().cpu().numpy()}

        aux = loss_fn()
        return dict(trans_loss=aux['trans_loss'])

    def evaluation(self, batch):
        self.trans.eval()
        metrics = self._eval_pref_step(batch)
        return metrics

    def _eval_pref_step(self, batch):
        def loss_fn():
            obs_0 = batch['observations0'];
            act_0 = batch['actions0']
            obs_1 = batch['observations1'];
            act_1 = batch['actions1']
            timestep_0 = batch['timesteps0'];
            timestep_1 = batch['timesteps1']
            labels = torch.as_tensor(batch['labels'], dtype=torch.long, device=self.device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=-1)
            B, T, N, _ = obs_0.shape

            mask0 = batch.get('attn_mask0', None)
            mask1 = batch.get('attn_mask1', None)
            if mask0 is None: mask0 = torch.ones((B, T, N), device=self.device)
            if mask1 is None: mask1 = torch.ones((B, T, N), device=self.device)

            trans_pred_0 = self.trans(obs_0, act_0, timestep_0, attn_mask=mask0)
            trans_pred_1 = self.trans(obs_1, act_1, timestep_1, attn_mask=mask1)

            score0 = self._aggregate_with_mask(trans_pred_0, mask0)
            score1 = self._aggregate_with_mask(trans_pred_1, mask1)
            logits = torch.stack([score0, score1], dim=1)

            trans_loss = F.cross_entropy(logits, labels)
            return {'trans_loss': trans_loss.detach().cpu().numpy()}

        aux = loss_fn()
        return dict(eval_trans_loss=aux['trans_loss'])

    def get_reward(self, batch):
        self.trans.eval()
        return self._get_reward_step(batch)

    def _get_reward_step(self, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        attn_mask = batch['attn_mask']
        trans_pred = self.trans(obs, act, timestep, attn_mask=attn_mask)

        return trans_pred.squeeze(1)

    ####################### my add method
    def save_model(self, save_path, save_idx):
        torch.save({
            'reward_model': self.trans.state_dict(),
            'seq_len': self.trans.max_episode_steps,
        }, str(save_path) + "reward_model_" + str(save_idx) + ".pt")

    def load_model(self, model_dir):
        model_state_dict = torch.load(model_dir, map_location=torch.device('cpu')) \
            if self.device == torch.device('cpu') else torch.load(model_dir)
        self.trans.load_state_dict(model_state_dict['reward_model'])
        print('--------------- load PrefTransformer -----------------')

    @property
    def total_steps(self):
        return self._total_steps

    def _aggregate_with_mask(self, pred, mask):
        # pred: [B, T, N, 1]  -> [B, T, N]
        pred = pred.squeeze(-1)
        if mask is None:
            # 没 mask 时做均值，避免长度差异引入偏置
            return pred.sum(dim=(1, 2)) / (pred.shape[1] * pred.shape[2])
        # mask: [B, T, N]，只累加有效步
        m = mask.float()
        num = (pred * m).sum(dim=(1, 2))
        den = m.sum(dim=(1, 2)).clamp(min=1.0)
        return num / den  # [B]

