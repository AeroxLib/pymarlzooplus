import copy
from pymarlzooplus.components.episode_buffer import EpisodeBatch
from pymarlzooplus.modules.mixers.vdn import VDNMixer
from pymarlzooplus.modules.mixers.qmix import QMixer
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pymarlzooplus.components.standarize_stream import RunningMeanStd

# ==============================================================================
# 1. MOA 网络定义 (Model of Other Agents)
#    用于预测队友动作，从而计算因果影响力
# ==============================================================================
class MOANet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MOANet, self).__init__()
        # input_dim: Global State + Context (所有人的动作)
        # output_dim: 预测的动作 Logits (通常是 N_Agents * N_Actions)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, inputs):
        return self.net(inputs)


# ==============================================================================
# 2. CMT Learner 核心类
# ==============================================================================
class CMTLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

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

        # ----------------------------------------------------------------------
        # [CMT 新增] 初始化 MOA 网络 (因果点火器)
        # ----------------------------------------------------------------------
        # 输入维度: 全局 State 维度 + (N个智能体 * 每个智能体的动作数)
        # 我们用上一时刻的 Joint Action 和 State 来预测当前的 Joint Action
        moa_input_dim = int(scheme["state"]["vshape"]) + (self.n_agents * self.n_actions)
        
        # 输出维度: 预测所有智能体的动作分布
        moa_output_dim = self.n_agents * self.n_actions
        
        self.moa = MOANet(moa_input_dim, moa_output_dim).to(device)
        self.moa_optim = Adam(self.moa.parameters(), lr=args.lr)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # ----------------------------------------------------------------------
        # [CMT 新增] 计算因果奖励并混合 (The Igniter Logic)
        # ----------------------------------------------------------------------
        # 1. 训练 MOA (监督学习: 预测队友动作)
        moa_loss = self.train_moa(batch)

        # 2. 计算因果影响力 (Intrinsic Reward)
        # 只有在配置文件中开启 lambda_c > 0 时才计算，节省算力
        intrinsic_reward = th.zeros_like(rewards)
        if hasattr(self.args, 'lambda_c') and self.args.lambda_c > 0:
            # 计算反事实影响力
            intrinsic_reward = self.compute_causal_influence(batch)
            
            # 3. 混合奖励: Total = Env_Reward + lambda * Causal_Reward
            # 注意: 这里直接修改了局部变量 rewards，不影响 buffer 中的原始数据
            rewards = rewards + self.args.lambda_c * intrinsic_reward

        # Standardise rewards (原有逻辑)
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # ----------------------------------------------------------------------
        # 正向传播与 Loss 收集
        # ----------------------------------------------------------------------
        mac_out = []
        
        # [CMT 新增] 用于累加 Agent 内部产生的辅助 Loss
        total_sparsity_loss = 0
        total_kl_loss = 0
        
        self.mac.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            
            # [CMT 关键点] 从 mac.agent 中收集每一步的 Loss
            # 注意: mac.agent 是同一个对象，每次 forward 后它的属性会被更新为当前步的 loss
            if hasattr(self.mac.agent, "sparsity_loss"):
                total_sparsity_loss += self.mac.agent.sparsity_loss
            if hasattr(self.mac.agent, "kl_loss"):
                total_kl_loss += self.mac.agent.kl_loss

        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
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

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        # 1. 基础 QMIX Loss
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # ----------------------------------------------------------------------
        # [CMT 新增] 联合 Loss 计算
        # ----------------------------------------------------------------------
        # 2. MOA Loss (因果预测)
        loss += moa_loss 
        
        # 3. Sparsity Loss (L0 正则: 鼓励路由稀疏)
        # 从时间步累加值计算平均值，乘以系数 (需要在 config 中定义 l0_weight，默认 0.01)
        l0_weight = getattr(self.args, "l0_weight", 0.01)
        loss += l0_weight * (total_sparsity_loss / batch.max_seq_length)

        # 4. MAGI KL Loss (信息瓶颈: 鼓励压缩消息)
        # 系数 beta (默认 0.001)
        beta = getattr(self.args, "beta", 0.001)
        loss += beta * (total_kl_loss / batch.max_seq_length)

        # Optimise
        self.optimiser.zero_grad()
        self.moa_optim.zero_grad() # 清零 MOA 梯度
        
        loss.backward()
        
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.moa_optim.step() # 更新 MOA

        # Update Target Networks
        self.training_steps += 1
        if (
                self.args.target_update_interval_or_tau > 1 and
                (
                        (self.training_steps - self.last_target_update_step) /
                        self.args.target_update_interval_or_tau >= 1.0
                )
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("moa_loss", moa_loss.item(), t_env)
            self.logger.log_stat("sparsity_loss", (total_sparsity_loss / batch.max_seq_length).item(), t_env)
            self.logger.log_stat("kl_loss", (total_kl_loss / batch.max_seq_length).item(), t_env)
            self.logger.log_stat("causal_reward_mean", intrinsic_reward.mean().item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        if self.args.prioritized_buffer:
            return masked_td_error ** 2

    # ==========================================================================
    # 辅助方法: 训练 MOA
    # ==========================================================================
    def train_moa(self, batch):
        # 准备数据: [Batch, Time, Features]
        states = batch["state"][:, :-1]
        actions = batch["actions_onehot"][:, :-1] # t 时刻的 Joint Action
        next_actions = batch["actions"][:, 1:]    # t+1 时刻的真实动作 (Target)
        
        # 构造输入: State + Joint Actions
        bs, seq_len, _ = states.shape
        flat_actions = actions.reshape(bs, seq_len, -1)
        inputs = th.cat([states, flat_actions], dim=-1) # [B, T, State+N*Act]
        
        # 预测
        pred_logits = self.moa(inputs) # [B, T, N*Act]
        
        # Reshape 为 [B*T*N, Act] 用于 CrossEntropy
        # 我们这里预测的是所有 Agent 的动作
        pred_logits = pred_logits.view(-1, self.n_actions)
        targets = next_actions.reshape(-1)
        
        loss = F.cross_entropy(pred_logits, targets)
        return loss

    # ==========================================================================
    # 辅助方法: 计算因果影响力 (Counterfactual Reasoning)
    # ==========================================================================
    def compute_causal_influence(self, batch):
        with th.no_grad(): # 不计算梯度，只算 Reward
            states = batch["state"][:, :-1]
            actions = batch["actions_onehot"][:, :-1]
            bs, seq_len, _ = states.shape
            
            # 1. 原始预测 (Factual)
            flat_actions = actions.reshape(bs, seq_len, -1)
            inputs = th.cat([states, flat_actions], dim=-1)
            
            # [B, T, N*Act] -> [B, T, N, Act]
            original_logits = self.moa(inputs).view(bs, seq_len, self.n_agents, self.n_actions)
            p_original = F.softmax(original_logits, dim=-1) 
            
            intrinsic_rewards = th.zeros(bs, seq_len, 1).to(states.device)
            
            # 2. 遍历每个 Agent i，计算它的影响力
            # (为了加速，可以在此处随机采样 Agent，而不是遍历所有)
            for i in range(self.n_agents):
                # 构造反事实动作: 假设 Agent i 什么都不做 (全0 或 Mask)
                cf_actions = actions.clone()
                cf_actions[:, :, i, :] = 0 
                
                flat_cf = cf_actions.reshape(bs, seq_len, -1)
                cf_inputs = th.cat([states, flat_cf], dim=-1)
                
                # 反事实预测 (Counterfactual)
                cf_logits = self.moa(cf_inputs).view(bs, seq_len, self.n_agents, self.n_actions)
                p_cf = F.softmax(cf_logits, dim=-1)
                
                # KL Divergence: 衡量 i 的动作对其他人预测分布的影响
                # KL(P_cf || P_orig)
                # 我们关心的是 i 对 j (j!=i) 的影响
                
                kl = F.kl_div(p_original.log(), p_cf, reduction='none').sum(dim=-1) # [B, T, N]
                # 注意: PyTorch 的 kl_div 顺序是 kl_div(input, target)，计算 sum(target * (log target - input))
                # 这里我们计算 D_KL(P_cf || P_orig) 或者 D_KL(P_orig || P_cf) 都可以作为差异度量
                # 这里用 D_KL(P_cf || P_original) = sum(P_original * (log P_original - log P_cf))
                
                # 排除 i 自己对自己的影响 (我们只奖励改变队友行为的 Agent)
                influence = kl.sum(dim=-1) - kl[:, :, i]
                
                # 累加奖励 (平均分给整个 Team，或者只给 Agent i)
                # 这里采用 QMIX 的合作设定，加到共享的 Team Reward 里
                intrinsic_rewards[:, :, 0] += influence / self.n_agents

            return intrinsic_rewards

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
        self.moa.cuda() # 确保 MOA 在 GPU 上

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))