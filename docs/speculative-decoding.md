# 投机解码

用小而快的 draft 模型先猜 `g` 个 token，target 模型一次验证。期望每轮产出 token 数
（Leviathan 闭式）：

```
expected = (1 - a^(g+1)) / (1 - a)      # a → 1 时极限为 g+1
speedup  = expected / (1 + g * c)       # c = draft 单 token 成本 / target 单 token 成本
```

## 复用要点

- **公式方向别写反**：`speedup = expected / (1 + g*c)`，分母是「草稿成本 + 验证成本」，
  不是「云/端延迟比」。接受率 `a` 越高、草稿越长，加速比越大，但 `c` 太大时草稿反而拖累。
- **接受率是核心变量**：`a = base + (1-base)*skill`，随引擎能力单调上升。
  `test_speculative.test_speedup_monotonic` 断言了单调性。
- 配套 **paged KV cache**（`kvcache/paged.py`）：固定块 + 块表 + LRU 驱逐，长上下文不
  碎片化、可共享、可逐序列回收。
