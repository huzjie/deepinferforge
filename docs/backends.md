# 后端

三个参考后端都消费同一份 IR：

| 后端 | 内核源码 | 说明 |
|------|---------|------|
| `cpu` | 无（直接执行） | golden 参考，其余后端对照它验证 |
| `nvidia` | CUDA `__global__` | 目标平台之一 |
| `ascend` | CANN `__aicore__` | 目标平台之一 |

## 一致性如何保证

参考实现里，所有后端的**执行**都走共享 oracle（`oracle.reference_output`），因此三后端
数值必然一致；而**编译**（`build_kernels`）则按各自后端发出不同的 kernel 源码。于是
`test_backends.test_kernel_source_differs` 能同时断言两点：内核数相同、源码不同，
而 `test_consistency` 断言输出逐元素相等——「一份 IR、两种源码、同一结果」。

## 接入真实后端的落点

要把参考实现接到真机，只需替换 `Backend.compile` 里的 `build_kernels`：把 IR 节点映射
成真实的 CUDA / CANN kernel，再在 `run` 里真正下发到设备队列。IR、Pass、调度、量化、
投机解码全部不用动。
