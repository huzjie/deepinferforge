# Contributing

欢迎贡献。提交前请：

1. 跑单测：`python -m unittest discover -s tests -t .`
2. 跑 doctor 冒烟：`python -m deepinferforge doctor`
3. 保持零依赖：不引入 numpy / torch / 第三方运行时依赖。
4. 训练/基准信号遵循「单调趋 1」铁律：`skill += lr(1-skill)`，boost=(skill-0.5)·2。
