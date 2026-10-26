# 基准

五个基准各验证统一推理栈的一个必备属性：

| 基准 | 度量 | 期望 |
|------|------|------|
| `consistency` | 三后端输出一致 | 恒 1.0（单一 IR 的保证） |
| `fusion` | 融合后 kernel 数 | 稳定 speedup > 1 |
| `quantization` | 量化往返余弦 | skill 0.5 → 1.0 单调 |
| `speculative` | 投机加速比 | skill 0.5 → 1.0 单调 |
| `autotune` | 选中 gold 配置的准确率 | skill 0.5 → 1.0 单调 |

运行：`python -m deepinferforge bench`。所有数字都是真实测量值（非手写估算）。
