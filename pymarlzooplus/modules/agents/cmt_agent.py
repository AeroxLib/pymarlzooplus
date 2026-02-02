
"""
在之前的代码里，为了跑通，我们把 mask 简单设为了全 1。
现在需要根据你的 PDF 换成真正的 Hard-Concrete + Top-K。
修改位置: src/modules/agents/cmt_agent.py 里的 HybridRouting 类。
你需要填入的代码逻辑:Hard-Concrete 采样: 引入可学习参数 log_alpha，
使用 Gumbel-Sigmoid 技巧生成 z (0 或 1)。
这对应 PDF 中提到的 "differentiable pruning"。
Gaussian Scoring: 计算每对 Agent 之间的相关性分数。
Perturbed Top-K: 结合 z 和 score，选出前 $K$ 个邻居，生成最终的 mask。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 定义组件 (Routing & GAT) - 放在同一个文件里方便调用
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HybridRouting(nn.Module):
    def __init__(self, n_agents, hidden_dim, k_budget, temp=0.1):
        super(HybridRouting, self).__init__()
        self.n_agents = n_agents
        self.k = k_budget
        self.temp = temp # 温度系数，越小越接近真正的 0/1，但也越难训练
        
        # [A. Hard-Concrete 参数]
        # 初始化为 0.5 的概率 (log_alpha = 0)
        self.log_alpha = nn.Parameter(torch.zeros(n_agents, n_agents))
        
        # [B. Gaussian Scoring 网络]
        # 输入是两个 Agent 的特征拼接 [h_i, h_j]，输出一个分数
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Hard Concrete 的常量参数 (参考 L0 Paper)
        self.gamma = -0.1
        self.zeta = 1.1

    def forward(self, h, training=True):
        # h: [Batch, N_Agents, Dim]
        batch_size = h.shape[0]
        
        # ========================================
        # 1. Hard-Concrete 采样 (结构性剪枝)
        # ========================================
        if training:
            # Gumbel-Softmax 采样技巧
            u = torch.rand_like(self.log_alpha)
            # 加上极小值防止 log(0)
            log_u = torch.log(u + 1e-8) - torch.log(1 - u + 1e-8)
            # Sigmoid 放缩
            s = torch.sigmoid((log_u + self.log_alpha) / self.temp)
            # 拉伸并截断 (Stretch and Rectify) -> 使得部分值真正变为 0
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(s_bar, 0, 1)
        else:
            # 测试时直接由 log_alpha 决定是否连接
            z = (self.log_alpha > 0).float()
            
        # 扩展 z 到 batch 维度 [Batch, N, N]
        z_batch = z.unsqueeze(0).expand(batch_size, -1, -1)

        # ========================================
        # 2. Gaussian Scoring (内容打分)
        # ========================================
        # 构造两两配对特征
        # h_i: [B, N, 1, D] -> [B, N, N, D]
        h_i = h.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        # h_j: [B, 1, N, D] -> [B, N, N, D]
        h_j = h.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        
        pair_feat = torch.cat([h_i, h_j], dim=-1) # [B, N, N, 2*D]
        raw_scores = self.score_net(pair_feat).squeeze(-1) # [B, N, N]
        
        # ========================================
        # 3. 混合逻辑 (Hybrid)
        # ========================================
        # 关键一步：如果 Hard-Concrete 说是 0，那就让分数变成负无穷
        # 这样 Top-K 就绝对不会选中它
        masked_scores = raw_scores + (1 - z_batch) * -1e9
        
        # ========================================
        # 4. Top-K 截断 (带宽限制)
        # ========================================
        # 选出每行最大的 K 个
        # values: [B, N, K], indices: [B, N, K]
        _, topk_indices = torch.topk(masked_scores, k=self.k, dim=-1)
        
        # 生成最终的 Mask
        final_mask = torch.zeros_like(raw_scores)
        final_mask.scatter_(-1, topk_indices, 1.0)
        
        # 自我连接处理 (可选)：通常自己总是连自己
        # eye = torch.eye(self.n_agents).to(h.device).unsqueeze(0)
        # final_mask = torch.max(final_mask, eye)

        return final_mask, z

class MaskedGAT(nn.Module):
    def __init__(self, hidden_dim):
        super(MaskedGAT, self).__init__()
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, mask):
        # 简单的加权聚合演示
        # h: [B, N, D], mask: [B, N, N]
        weights = F.softmax(self.head(h), dim=-1) # 简化版 Attention
        # 实际 GAT 逻辑：利用 mask 过滤邻居
        out = torch.matmul(mask, h) 
        return out

# ==========================================
# 2. 定义你的 Agent (CMT)
# ==========================================
class CMTAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CMTAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        
        # 1. 基础编码
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.gru = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        
        # 2. 路由与通信
        self.routing = HybridRouting(args.n_agents, args.hidden_dim, args.k_budget)
        self.comm = MaskedGAT(args.hidden_dim)
        
        # 🔥 [新增] MAGI (Information Bottleneck)
        # 把 GAT 的输出压缩成 mu 和 sigma
        self.ib_mu = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.ib_std = nn.Linear(args.hidden_dim, args.hidden_dim)
        
        # 3. 输出层
        # 输入是: 原始隐藏状态 + 压缩后的消息
        self.out_net = nn.Linear(args.hidden_dim * 2, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # ... (前面的编码部分不变) ...
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.gru(x, h_in)
        
        # 还原维度 [B, N, D]
        bs = h.shape[0] // self.n_agents
        h_view = h.view(bs, self.n_agents, -1)
        
        # Routing & GAT
        mask, z = self.routing(h_view)
        comm_feat = self.comm(h_view, mask) # [B, N, D]
        
        # 🔥 [新增] IB Compression (MAGI 核心)
        mu = self.ib_mu(comm_feat)
        std = F.softplus(self.ib_std(comm_feat)) + 1e-6 # 保证为正
        dist = torch.distributions.Normal(mu, std)
        message = dist.rsample() # 重参数化采样
        
        # 计算 KL Loss: KL(N(mu, std) || N(0, 1))
        # 这一步是为了让消息尽可能压缩，去除冗余
        kl = torch.distributions.kl_divergence(dist, torch.distributions.Normal(0, 1)).sum(dim=-1)
        
        # 🔥 [保存 Loss] 存到 self 里，让 Learner 来取
        self.kl_loss = kl.mean()       # IB Loss
        self.sparsity_loss = z.sum()   # L0 Loss (希望 z 里的 1 越少越好)
        
        # 变回扁平
        msg_flat = message.view(bs * self.n_agents, -1)
        
        # 决策
        joint = torch.cat([h, msg_flat], dim=-1)
        q = self.out_net(joint)
        
        return q, h