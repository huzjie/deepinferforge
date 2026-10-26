# 快速上手

## 1. Doctor 冒烟

```bash
python -m deepinferforge doctor
```

会依次验证：配置加载、三后端注册、模型建图、三后端编译+执行、autotune grid search、
以及五个基准的 summary。

## 2. 编译一个模型

```bash
python -m deepinferforge compile
```

输出 JSON 化的 `CompiledPlan`（backend、n_kernels、pass_report、每个 kernel 的 config）。

## 3. 训练（skill → 1.0）

```bash
python -m deepinferforge train
```

## 4. 起 HTTP 服务

```bash
python -m deepinferforge serve --port 8421
# 另开终端
python examples/serve_client.py
```

## 5. 跑单测

```bash
python -m unittest discover -s tests -t .
```

零依赖，托管 Python 3.13 也能跑。
