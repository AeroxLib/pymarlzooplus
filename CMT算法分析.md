# CMT 算法全面分析

## 一、核心创新点（论文重点）

**CMT = Communication + MAGI (信息瓶颈) + Top-K 路由**

1. **混合路由机制 (Hybrid Routing)**：结构稀疏 + 内容相关
2. **MAGI (Maximizing Agent-to-Agent Information)**：信息瓶颈压缩
3. **MOA (Model of Other Agents)**：因果影响力计算

---

## 二、算法架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      CMT Agent 架构                              │
├─────────────────────────────────────────────────────────────────┤
│  Input (obs)                                                     │
│    │                                                             │
│    ▼                                                             │
│  FC1 → GRU ──┬─→ h (隐藏状态)                                   │
│              │                                                   │
│              ▼                                                   │
│    ┌──────────────────────┐                                     │
│    │   Hybrid Routing     │  ← 核心创新 1                       │
│    │  ┌────────────────┐  │                                     │
│    │  │ Hard-Concrete  │  │  z ~ Gumbel-Sigmoid(log_alpha)      │
│    │  │ (结构性剪枝)    │  │  → 学习"该不该连"                   │
│    │  └────────────────┘  │                                     │
│    │          ↓           │                                     │
│    │  ┌────────────────┐  │                                     │
│    │  │ Gaussian Scoring│  │  score = MLP([h_i, h_j])           │
│    │  │ (内容相关性)    │  │  → 学习"连得有多重要"               │
│    │  └────────────────┘  │                                     │
│    │          ↓           │                                     │
│    │  ┌────────────────┐  │                                     │
│    │  │    Top-K       │  │  mask = TopK(z * score)            │
│    │  │ (带宽限制)      │  │  → 只保留K个连接                    │
│    │  └────────────────┘  │                                     │
│    └──────────────────────┘                                     │
│              │                                                   │
│              ▼                                                   │
│    MaskedGAT(comm_feat, mask)  ← 带掩码的图注意力               │
│              │                                                   │
│              ▼                                                   │
│    ┌──────────────────────┐                                     │
│    │      MAGI (IB)       │  ← 核心创新 2                       │
│    │  ┌────────────────┐  │                                     │
│    │  │   μ = MLP(f)   │  │                                     │
│    │  │   σ = MLP(f)   │  │                                     │
│    │  │   m ~ N(μ, σ)  │  │  重参数化采样                        │
│    │  └────────────────┘  │                                     │
│    │  KL(N(μ,σ) || N(0,1))│  → 信息瓶颈损失 (β * KL)            │
│    └──────────────────────┘                                     │
│              │                                                   │
│              ▼                                                   │
│    concat([h, m]) → FC → Q-values                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、关键组件详解

### 3.1 Hybrid Routing（混合路由）

```python
# Hard-Concrete 采样（可微分剪枝）
u = torch.rand_like(log_alpha)
s = sigmoid((log(u/(1-u)) + log_alpha) / temp)
z = clamp(s * (zeta - gamma) + gamma, 0, 1)  # 接近 0/1

# Gaussian 评分（内容相关性）
pair_feat = concat([h_i, h_j])
score = MLP(pair_feat)

# 混合：结构 * 内容
final_score = score + (1 - z) * -1e9  # z=0 则分数为负无穷
mask = TopK(final_score, k=K)
```

**论文表述**：
- **Hard-Concrete**：学习"是否应该连接"（结构性先验）
- **Gaussian Scoring**：学习"连接的重要性"（内容相关性）
- **Top-K**：限制带宽，强制稀疏通信图

### 3.2 MAGI（信息瓶颈）

```python
# 编码器输出 μ 和 σ
mu = ib_mu(comm_feat)
std = softplus(ib_std(comm_feat)) + 1e-6

# 重参数化采样
message = mu + std * epsilon  # epsilon ~ N(0,1)

# KL 散度约束
kl_loss = KL(N(mu, std) || N(0, 1))
```

**论文表述**：
> 通过变分信息瓶颈，迫使智能体只传递与任务最相关的最小信息，抑制冗余噪声。

### 3.3 MOA（因果影响力计算）

```python
# 1. 训练 MOA（监督学习）
input = concat([state_t, actions_t])
pred_actions_t1 = MOA(input)
loss = CrossEntropy(pred_actions_t1, real_actions_t1)

# 2. 计算因果影响力（反事实推理）
for each agent i:
    # 反事实：假设 agent i 不采取行动
    cf_actions = actions.clone()
    cf_actions[i] = 0 
    
    # 比较预测分布差异
    p_orig = softmax(MOA(state, actions))
    p_cf = softmax(MOA(state, cf_actions))
    
    # KL 散度作为影响力度量
    influence_i = KL(p_orig || p_cf)
    
    # 奖励 = 对队友的影响
    intrinsic_reward[i] = influence_i
```

**论文表述**：
> 通过反事实推理，量化每个智能体对其他智能体决策的因果影响，给予"改变队友行为"的奖励。

---

## 四、训练流程（多损失联合优化）

```
┌─────────────────────────────────────────────────────────────────┐
│                     CMT Learner 训练流程                         │
├─────────────────────────────────────────────────────────────────┤
│  1. 采样 batch (s, a, r, s')                                    │
│                                                                  │
│  2. 计算因果奖励（可选）                                         │
│     r_causal = MOA.compute_causal_influence(batch)              │
│     r_total = r_env + λ_c * r_causal                            │
│                                                                  │
│  3. 前向传播（收集各步损失）                                     │
│     for t in range(T):                                          │
│         q_t, sparsity_loss_t, kl_loss_t = Agent.forward()       │
│                                                                  │
│  4. 计算总损失                                                   │
│     Loss = L_QMIX                                               │
│          + L_MOA                                                │
│          + λ_L0 * mean(sparsity_loss)  ← 鼓励稀疏               │
│          + β * mean(kl_loss)           ← MAGI压缩               │
│                                                                  │
│  5. 反向传播更新                                                 │
│     - Agent 参数                                                 │
│     - MOA 参数                                                   │
│     - Target 网络（软/硬更新）                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、超参数说明（配置文件）

```yaml
# CMT 特有参数
beta: 0.001           # MAGI KL 损失系数，控制压缩程度
l0_weight: 0.01       # L0 稀疏正则系数，控制路由稀疏度
lambda_c: ?           # 因果奖励系数（可选，代码中有但未在配置中）

# 基础 QMIX 参数
mixer: "vdn"          # 或 "qmix"，值函数混合方式
target_update_interval_or_tau: 200
double_q: True
```

---

## 六、与其他算法的对比

| 特性 | QMIX | CommNet | TarMAC | CMT (本算法) |
|------|------|---------|--------|-------------|
| 通信结构 | 无 | 全连接 | 注意力 | **Top-K 稀疏** |
| 通信内容 | - | 原始向量 | 注意力加权 | **信息瓶颈压缩** |
| 结构学习 | - | 固定 | 固定 | **Hard-Concrete** |
| 因果推理 | - | 无 | 无 | **MOA 反事实** |

---

## 七、论文写作建议

**标题**：CMT: Sparse Communication via Maximizing Agent-to-Agent Information

**核心卖点**：
1. **高效**：Top-K 稀疏路由减少 90% 通信量
2. **有效**：MAGI 确保只传递关键信息
3. **可解释**：Hard-Concrete 学到清晰的通信拓扑

**实验设计**：
- 消融实验：分别去掉 Top-K / MAGI / MOA
- 可视化：展示学到的通信图（mask 热力图）
- 效率对比：通信带宽 vs 性能曲线

---

## 八、关键代码文件说明

| 文件 | 作用 |
|------|------|
| `cmt_agent.py` | CMT Agent 实现，包含 Hybrid Routing 和 MAGI |
| `cmt_learner.py` | 训练逻辑，包含 MOA 和联合损失计算 |
| `cmt.yaml` | 算法配置参数 |
| `basic_controller.py` | 基础 MAC 控制器 |
| `run.py` | 训练主流程 |

---

*分析完成时间：2026/2/2*
