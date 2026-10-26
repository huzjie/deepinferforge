# 自动调优

统一编译器不能给所有算子写死同一个 tile size，必须「按算子 × 按后端」调优。

## 成本模型

`autotune/cost.py` 的 `op_cost(op, tile, layout, unroll)` 是一个「真值可验证」的模型：
gold 配置恒为最小值，任何偏离按维度加固定惩罚。真实引擎只需把这里的解析式换成
测量值或学习到的预测器，接口不变。

## 搜索策略

- `grid_search`：穷举候选空间，**确定性找到全局最优**（`test_autotune` 断言它等于
  `gold_config`）。
- `anneal`：模拟退火，`max_trials` 有界，用于大空间近似。

## 复用要点

调优质量即 `skill`：`bench/autotune.py` 建模「skill 的调优器以概率 skill 选中 gold」，
准确率严格单调（0.5 → 1.0）。真实工程里这个 skill 来自历史测量 / 迁移学习。
