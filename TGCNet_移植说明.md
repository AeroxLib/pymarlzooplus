# TGCNet 移植说明

## 移植完成情况

TGCNet (AAAI 2025 Oral) 已成功移植到 pymarlzooplus 框架！

## 移植文件列表

### 1. 核心代码文件
| 源文件 | 目标路径 | 说明 |
|--------|---------|------|
| `tgcnet_controller.py` | `pymarlzooplus/controllers/tgcnet_controller.py` | MAC 控制器 |
| `tgcnet.py` | `pymarlzooplus/modules/agents/tgcnet_agent.py` | Agent 网络 |
| `tgc_learner.py` | `pymarlzooplus/learners/tgc_learner.py` | 训练器 |
| `graph_coarsening.py` | `pymarlzooplus/modules/mixers/graph_coarsening.py` | 图粗化混合器 |

### 2. 网络层组件
全部复制到 `pymarlzooplus/modules/layers/`：
- `add_norm.py` - 添加归一化
- `graph_convolution.py` - 图卷积
- `multi_head_attention.py` - 多头注意力
- `multi_head_hard_attention.py` - 多头硬注意力
- `position_wise_ffn.py` - 位置前馈网络
- `sag_pool.py` - 自注意力图池化
- `self_attention_pooling.py` - 自注意力池化
- `transformer_decoder.py` - Transformer 解码器
- `utils.py` - 工具函数

### 3. 配置文件
- `pymarlzooplus/config/algs/tgcnet.yaml` - TGCNet 算法配置

## 注册情况

已在以下 `__init__.py` 中注册 TGCNet：

✅ `controllers/__init__.py` - 注册为 `tgcnet_mac`
✅ `modules/agents/__init__.py` - 注册为 `tgcnet`
✅ `learners/__init__.py` - 注册为 `tgc_learner`
✅ `modules/mixers/__init__.py` - 注册为 `coarsen`

## 运行命令

### 1. 运行 TGCNet
```bash
python3 -m pymarlzooplus.main --config=tgcnet --env-config=hallway with t_max=10000
```

### 2. 运行 TGCNet on SMAC
```bash
python3 -m pymarlzooplus.main --config=tgcnet --env-config=sc2 with env_args.map_name=3m t_max=100000
```

### 3. 运行 TGCNet on LBF
```bash
python3 -m pymarlzooplus.main --config=tgcnet --env-config=gymma with env_args.key="lbforaging:Foraging-11x11-6p-4f-v2"
```

## 关键配置参数

```yaml
mac: "tgcnet_mac"          # 控制器
agent: "tgcnet"            # Agent 类型
learner: "tgc_learner"     # 训练器
mixer: "coarsen"           # 混合器

# 图粗化参数
coarsening_embed_dim: 32
mixing_embed_dim: 32
hypernet_layers: 2

# Transformer 参数
enc_att_heads: 2
dec_att_heads: 4
att_enc_dim: 128
att_dec_dim: 32
num_layers: 2
dropout: 0
```

## 与 CMT 的对比

现在你可以直接比较 CMT 和 TGCNet：

```bash
# 运行 CMT
python3 -m pymarlzooplus.main --config=cmt --env-config=hallway with t_max=10000

# 运行 TGCNet
python3 -m pymarlzooplus.main --config=tgcnet --env-config=hallway with t_max=10000
```

## 算法对比

| 特性 | CMT | TGCNet |
|------|-----|--------|
| 路由学习 | Hard-Concrete (可微分) | Hard Attention (离散化) |
| 消息压缩 | MAGI (信息瓶颈) | Transformer Decoder |
| 图粗化 | ❌ 无 | ✅ SAGPool |
| 因果推理 | ✅ MOA | ❌ 无 |
| 价值混合 | VDN/QMIX | Graph Coarsening Mixer |

## 注意事项

1. **路径修改**: 所有 `from modules.` 和 `from components.` 导入已修改为 `from pymarlzooplus.modules.` 和 `from pymarlzooplus.components.`

2. **依赖检查**: TGCNet 依赖的 layers 已全部复制到 `pymarlzooplus/modules/layers/`

3. **配置兼容**: TGCNet 配置与 pymarlzooplus 框架兼容

---

*移植完成时间：2026/2/2*
