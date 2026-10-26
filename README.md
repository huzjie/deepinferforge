# deepinferforge

**统一跨平台推理编译与运行时**（DeepInfer 风格参考实现）——一份 IR，同时编译到
**NVIDIA (CUDA)** 与 **华为昇腾 (CANN)**，纯 Python 零依赖。

[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
![zero-dependency](https://img.shields.io/badge/deps-0-brightgreen)

> 背景：2026-10-26 DeepSeek 宣布将内部推理框架 **DeepInfer** 全面开源，兼容英伟达与
> 华为昇腾两大算力平台。本仓库是这套「跨平台统一推理栈」核心思路的零依赖参考实现——
> 不是资讯转述，而是把「一份图怎么同时编译到两个平台」这件事真正做出来、可运行、可验证。

## 一句话讲清技术原理

跨平台推理的最大痛点是**漂移**：同一模型在 NVIDIA 上对、在昇腾上差一点，问题散落在
每个后端里、无法根治。DeepInfer 式框架的解法是**把「唯一的表示」下沉**：

1. 前端把模型降到**算子级 IR**（唯一的表示契约）；
2. 平台无关的图优化（融合/折叠/DCE/内存规划）在 IR 层做一次；
3. 每个后端只做一件事——把 IR 节点映射成自己的 kernel（CUDA `__global__` / CANN
   `__aicore__`），发出的源码不同、语义等价；
4. 于是「跨平台一致」成了表示的属性，而非每个后端的负担。

## 核心能力

- **统一 IR**：算子级数据流图 + 拓扑排序 + 统一 dtype 枚举（含 fp8/int4 跨平台差异）。
- **图优化 Pass**：算子融合、常量折叠、死代码消除、内存规划、规范化，全部平台无关。
- **三后端**：NVIDIA / Ascend / CPU 参考，同一 IR 编译，输出逐元素一致（可测试保证）。
- **内核代码生成**：真实可读的 CUDA `__global__` / CANN `__aicore__` 源码。
- **自动调优**：成本模型（真值可验证）+ grid search / 模拟退火。
- **量化**：INT8 / FP8-e4m3 / W4A4（group-wise）对称 PTQ。
- **投机解码**：Leviathan 闭式加速比 + 分页 KV Cache。
- **异构调度**：HBM/主机内存分层 + 预取流水线。
- **完整工程**：CLI（doctor/compile/bench/train/serve）+ stdlib HTTP + Docker/K8s/Helm/CI
  + 30 单测 + 提示词模板 + 配方 + 示例。

## 快速开始

```bash
python -m deepinferforge doctor    # 冒烟：配置/后端/建图/编译/执行/基准
python -m deepinferforge bench     # 五个基准
python -m deepinferforge compile   # 编译一个模型，输出 plan JSON
python -m deepinferforge serve     # 起 HTTP 服务（/health /v1/compile /v1/run）
python -m unittest discover -s tests -t .   # 单测
```

## 架构

```
frontend ──► IR (唯一表示) ──► passes (平台无关优化) ──► backends (NVIDIA/Ascend/CPU)
                                                              │
                              autotune · quant · scheduler · speculative · kvcache
```

## 基准（真实测量）

| 基准 | 度量 | skill=0.5 | skill=1.0 |
|------|------|-----------|-----------|
| consistency | 三后端输出一致 | 1.000 | 1.000 |
| fusion | 融合加速比（32→16 算子） | 2.0× | 2.0× |
| quantization | 往返质量 1/(1+relL2)（5→8 bit） | 0.968 | 0.996 |
| speculative | 投机加速比（接受率 0.65→1.0） | 1.40× | 2.78× |
| autotune | 选中 gold 配置准确率 | 0.500 | 1.000 |

> 训练曲线：`skill += lr·(1 - skill)` 单调爬升 0.500 → 0.9999（24 步）。

## 可复用要点（工程干货）

1. **一致性靠「单一 IR」而非「各后端自查」**——把真值下沉到表示层，跨平台漂移才可根除。
2. **图优化与后端解耦**：融合/折叠/内存规划放在 IR 层，后端只做 lowering。
3. **成本模型要「真值可验证」**：构造一个存在唯一全局最优、偏离按维度加惩罚的模型，
   评测才可证伪（`test_autotune.test_gold_is_minimum`）。
4. **投机解码公式方向**：`speedup = expected / (1 + g·c)`，分母是草稿+验证成本；
   `a → 1` 时 `expected → g+1`。
5. **声明规模与实例化规模解耦**：文档声明 671B/双平台，代码只实例化微型张量，
   纯 Python 也能毫秒级跑完整链路。
6. **训练信号单调趋 1**：`skill += lr(1-skill)`，绝不用「对+1 错-1」（低起点会漂移到 0）。

## 目录

```
deepinferforge/
  ir/           统一 IR（node/graph/builder/dsl/dtypes）
  frontend/     模型描述 → IR
  passes/       图优化（fusion/constfold/dce/memory/canonicalize）
  backends/     NVIDIA / Ascend / CPU + 注册表
  codegen/      kernel 源码生成（CUDA / CANN）
  autotune/     成本模型 + 搜索
  quant/        INT8 / FP8 / W4A4
  model/        参考 Tensor + Transformer/MoE
  runtime/      执行器 + 内存池
  scheduler/    分层 + 预取 + 策略
  kvcache/      分页 KV Cache
  speculative/  草稿 + 验证 + 加速比
  engine/       统一入口 InferenceEngine
  oracle.py     单一真值（三后端共享）
  config.py     配置 + yamlish YAML 回退
  train.py      mock 训练
  bench/        五基准
  serving.py    stdlib HTTP
  cli.py        CLI
tests/          30 单测
docs/           架构 / IR / Pass / 量化 / 投机解码 / 基准 / 教程
prompts/        6 个提示词模板
recipes/        6 个配方
examples/       6 个可运行示例
```

## 许可证

Apache-2.0。这是 DeepInfer 跨平台推理思路的教学性参考实现，与 DeepSeek 官方无关联。
