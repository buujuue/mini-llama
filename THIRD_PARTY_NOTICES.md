# 第三方代码许可

本项目的 tokenizer、RadixTree 移植实现与相关测试、RequestContext 参考实现、前向输入检查，以及 KV cache、batch 和 RequestContext 的实例移植测试，源自本地参考项目 `references/mini-llama-cpp/`，原作者为 yus3nable，采用 MIT 许可。

`include/sampler.h` 与 `src/sampler.cpp` 的采样器构造、统一 `Sample` 入口、贪心、Top-k 和温度采样及输入校验适配自上游 `include/mini_llama/sampler.h` 与 `src/sampler.cc`；候选项改为本项目的 `Candidate`，并通过 `SampleTemperature(const std::vector<Candidate>&, float)` 重载复用温度采样逻辑。`test/test_sampler.cpp` 适配 `tests/gtest/test_sampler.cc` 的统一入口及 Top-k 测试，并补充温度权重、数值稳定性、随机状态和输入边界检查。

`src/forward.cpp` 中的 `CheckForwardInputs` 适配自上游 `src/forward.cc` 的 `ValidateForwardInputs`；`ComputeHead` 与 CPU 注意力逐头汇总适配自同文件 `AttentionForward` 中的 `compute_head` lambda 及调用循环。

`src/forward.cpp` 中的 `ForwardAdd` 与注意力输出投影的一维变形、残差相加适配自上游 `src/forward.cc`，使用本项目的 `Tensor::Reshape`、`AssertShape` 和显式后端参数。

`src/forward.cpp` 中的 `ForwardSwiGlu` 与 `FfnForward` 的 CPU 路径适配自上游 `src/forward.cc`，复用本项目的 F32 算子，使用显式后端参数，非 CPU 后端抛出尚未实现异常。

`include/tensor.h` 和 `src/tensor.cpp` 中的 `Tensor::AssertShape` 适配自上游同名接口与实现，形状类型改为 `std::vector<size_t>`，复用本项目的形状格式化函数；`test/test_tensor.cpp` 移植 `tests/gtest/test_tensor.cc` 中的两项 AssertShape 测试，并补充错误上下文、维数、轴顺序、空形状和空调用名称的检查。

本次新增测试的对应来源：

- `test/test_batch.cpp`：适配 `tests/gtest/test_batch.cc` 中的四项批次构造测试，另补充默认位置与输入副本测试。
- `test/test_request_context.cpp`：适配 `tests/test_request_context.cc` 中的三项请求跟踪测试，另补充 trace ID、事件顺序、计时与错误更新测试。
- `test/test_context.cpp`：为本项目已有的 Context 构造与请求隔离行为新增测试。
- `test/test_tensor.cpp`：另适配 `tests/gtest/test_tensor.cc` 的两项 ReshapeChecked 测试，接口使用本项目的 `Reshape`，并补充零维度与元素数溢出检查。
- `test/test_forward.cpp`：适配 `tests/gtest/test_forward.cc` 的五项 F32 单 token 前向测试，以及 `tests/gtest/test_batch.cc` 的五项批次前向测试；使用固定内存权重替代模型加载，另补充 CPU 注意力、GQA 与 NeoX 模式传递检查。为测试内部输入检查函数，测试文件直接包含现有 `src/forward.cpp`，未改动生产接口。

上游原始许可全文如下：

```text
MIT License

Copyright (c) 2026 yus3nable

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
