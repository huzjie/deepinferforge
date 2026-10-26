# 量化（PTQ）

对称量化，per-tensor（INT8 / FP8-e4m3）与 group-wise（W4A4）三档。

## 数学

对称量化：`scale = max(|w|) / (2^(bits-1) - 1)`，`q = round(w / scale)`，`w' = q * scale`。
零点是 0（对称），省掉零点偏移的算力。

## 复用要点

- **scale 校准质量决定量化误差**。`bench/quant.py` 用「scale 偏置 `1+(1-skill)`」建模
  校准好坏：skill=1.0 时 scale 精确、余弦≈1.0；skill=0.5 时 scale 放大、误差上升。
- **fp8-e4m3 的量化不是简单取整**：要拆符号位 / 4 位指数 / 3 位尾数，且 clamp 到可表示
  范围；`schemes._fp8_e4m3_q` 给了最小实现。
- **W4A4 用 group-wise**：每 128（或 64）个权重共享一个 scale，比 per-tensor 更耐
  outlier，代价是额外存 scale。
- 量化是「内存换精度」：W4A4 能把 671B 权重压到约 1/8 体积，是边缘/大模型部署的关键。
