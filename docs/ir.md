# 中间表示（IR）

IR 是一个**算子级数据流图**：节点是算子（`matmul` / `softmax` / `layernorm` / ...），
边是具名数据流。它有意保持在「算子」这一层——比框架级图（`nn.Module`）低、比指令级
（SASS / 昇腾指令）高，恰好是「平台无关优化」能发挥作用的层。

## 结构

- `ir/dtypes.py` — 统一 dtype 枚举（`float32` / `fp8_e4m3` / `int4` ...）。NVIDIA 与
  昇腾对低精度类型的命名与实现不同，IR 用统一枚举，后端各自决定如何落地。
- `ir/node.py` — 单个节点：`op`、`inputs`、`attrs`、`shape`、`dtype`。
- `ir/graph.py` — 图 + 拓扑排序（Kahn 算法，检测环）。
- `ir/builder.py` — 便捷构建器，把张量算子调用编译成节点。
- `ir/dsl.py` — 声明式 DSL，模型可写成 `(op, inputs, attrs)` 列表再 `lower`。

## 为什么 IR 是跨平台的关键

如果每个后端各自读一个框架特定的图，那么「NVIDIA 上是对的、昇腾上差一丢丢」这类漂移
会分散在每个后端里，无法根治。把「唯一的表示」下沉到 IR，一致性就成了表示的属性，
而不是每个后端的负担。这也是为什么 `test_backends.test_consistency` 能断言三后端输出
逐元素相等。
