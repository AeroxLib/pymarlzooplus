#!/bin/bash
# CMT 训练脚本 - 优化配置
# 针对 hallway_join1 环境的最佳实践设置
# 配置与 pymarlzooplus/config/algs/cmt.yaml 保持一致
# 注意：只覆盖命令行特有的参数，cmt.yaml中已有的参数不要再传

# 关键优化：
# 1. batch_size=32: 与配置文件一致，sweet spot
# 2. test_interval=100000: 只在最后10轮测试 (t_max/10)
# 3. save_model_interval=1000000: 只保留最后模型
# 4. epsilon_anneal_time=200000: 总步数的20%，充分探索
# 5. label=CMT_hallway_join1: TensorBoard标识名

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m pymarlzooplus.main \
  --config=cmt \
  --env-config=hallway_join1 \
  with \
  t_max=1000000 \
  batch_size=32 \
  batch_size_run=1 \
  epsilon_anneal_time=200000 \
  epsilon_start=1.0 \
  epsilon_finish=0.05 \
  test_interval=100000 \
  test_nepisode=32 \
  save_model_interval=1000000 \
  save_model=True \
  target_update_interval_or_tau=200 \
  lr=0.0005 \
  use_cuda=True \
  use_tensorboard=True \
  standardise_rewards=True \
  label="CMT_hallway_join1" \
  seed=42 \
  "$@"
