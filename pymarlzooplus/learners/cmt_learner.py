# -*- coding: utf-8 -*-
"""
CMT Learner - Complete Implementation with All Optimizations
包含：QMIX + MOA Loss + AMP + 向量化优化
修复：Target network loss accumulation, FP16 safety
"""

import copy
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.modules.mixers.vdn import VDNMixer
from pymarlzooplus.modules.mixers.qmix import QMixer
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from pymarlzooplus.components.standarize_stream import RunningMeanStd


class CMTLearner:
    """
    CMT Learner with integrated MOA in Agent
    Uses gradient detach for MOA to prevent pollution of main network
    Supports Automatic Mixed Precision (AMP) for 30-50% speedup
    """
    
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        # 主网络参数（包含 MOA，但 MOA 使用 detach()）
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        # Mixer
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        
        # 🚀 AMP (Automatic Mixed Precision) scaler
        self.scaler = GradScaler() if args.use_cuda else None

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 获取数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 标准化奖励
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # 前向传播收集loss
        mac_out = []
        total_moa_loss = 0
        total_sparsity_loss = 0
        total_kl_loss = 0
        
        self.mac.init_hidden(batch.batch_size)
        
        # 🚀 使用 autocast 进行混合精度前向传播
        with autocast(enabled=self.args.use_cuda):
            for t in range(batch.max_seq_length):
                # 🔥 关键：传递actions给Agent用于MOA训练
                agent_outs = self.mac.agent.forward(
                    self.mac._build_inputs(batch, t),
                    self.mac.hidden_states,
                    actions=batch["actions"][:, t:t+1],  # 当前动作
                    next_actions=batch["actions"][:, t+1:t+2] if t < batch.max_seq_length - 1 else None,
                    training=True
                )
                
                # 更新hidden states
                self.mac.hidden_states = agent_outs[1]
                mac_out.append(agent_outs[0])
                
                # 收集各种loss
                if hasattr(self.mac.agent, "moa_loss"):
                    total_moa_loss += self.mac.agent.moa_loss
                if hasattr(self.mac.agent, "sparsity_loss"):
                    total_sparsity_loss += self.mac.agent.sparsity_loss
                if hasattr(self.mac.agent, "kl_loss"):
                    total_kl_loss += self.mac.agent.kl_loss

            mac_out = th.stack(mac_out, dim=1)

            # Q-learning loss
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

            # [修复1] Target Q-values - 重置target_mac的loss防止累积
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            
            # 显式重置target_mac的loss属性，防止显存泄漏
            if hasattr(self.target_mac.agent, 'moa_loss'):
                self.target_mac.agent.moa_loss = th.tensor(0.0)
            if hasattr(self.target_mac.agent, 'sparsity_loss'):
                self.target_mac.agent.sparsity_loss = th.tensor(0.0)
            if hasattr(self.target_mac.agent, 'kl_loss'):
                self.target_mac.agent.kl_loss = th.tensor(0.0)
            
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.agent.forward(
                    self.target_mac._build_inputs(batch, t),
                    self.target_mac.hidden_states,
                    training=False  # 确保training=False跳过MOA计算
                )
                self.target_mac.hidden_states = target_agent_outs[1]
                target_mac_out.append(target_agent_outs[0])
            
            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            if self.args.double_q:
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

            # Mix
            if self.mixer is not None:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

            if self.args.standardise_returns:
                target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

            # Q-learning targets
            targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

            if self.args.standardise_returns:
                self.ret_ms.update(targets)
                targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

            # TD-error
            td_error = (chosen_action_qvals - targets.detach())
            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask

            # 1. Q-learning Loss
            loss = (masked_td_error ** 2).sum() / mask.sum()

            # 2. MOA Loss (使用detach，不污染主网络)
            moa_weight = getattr(self.args, "moa_weight", 0.1)
            loss += moa_weight * (total_moa_loss / batch.max_seq_length)

            # 3. Sparsity Loss (L0正则) - 带warm-up调度
            l0_weight = getattr(self.args, "l0_weight", 0.01)
            # [修复2] Warm-up: 训练初期降低l0_weight，让网络先充分连接
            warmup_steps = 50000  # 前50000步warmup
            if self.training_steps < warmup_steps:
                l0_weight = l0_weight * (self.training_steps / warmup_steps)
            loss += l0_weight * (total_sparsity_loss / batch.max_seq_length)

            # 4. MAGI KL Loss (信息瓶颈)
            beta = getattr(self.args, "beta", 0.001)
            loss += beta * (total_kl_loss / batch.max_seq_length)

        # 🚀 使用 scaler 进行反向传播和更新
        self.optimiser.zero_grad()
        
        if self.args.use_cuda and self.scaler is not None:
            # AMP training path
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()
        else:
            # Standard training path
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        # 更新目标网络
        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1 and
            (self.training_steps - self.last_target_update_step) / 
            self.args.target_update_interval_or_tau >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # 日志
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("moa_loss", (total_moa_loss / batch.max_seq_length).item(), t_env)
            self.logger.log_stat("sparsity_loss", (total_sparsity_loss / batch.max_seq_length).item(), t_env)
            self.logger.log_stat("kl_loss", (total_kl_loss / batch.max_seq_length).item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        if self.args.prioritized_buffer:
            return masked_td_error ** 2

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), 
                                              map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), 
                                               map_location=lambda storage, loc: storage))
