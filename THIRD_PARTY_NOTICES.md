# 第三方代码许可

本项目的 tokenizer、RadixTree 移植实现与相关测试、RequestContext 参考实现、前向输入检查，以及 KV cache、batch 和 RequestContext 的实例移植测试，源自本地参考项目 `references/mini-llama-cpp/`，原作者为 yus3nable，采用 MIT 许可。

`src/forward.cpp` 中的 `CheckForwardInputs` 适配自上游 `src/forward.cc` 的 `ValidateForwardInputs`。

本次新增测试的对应来源：

- `test/test_batch.cpp`：适配 `tests/gtest/test_batch.cc` 中的四项批次构造测试，另补充默认位置与输入副本测试。
- `test/test_request_context.cpp`：适配 `tests/test_request_context.cc` 中的三项请求跟踪测试，另补充 trace ID、事件顺序、计时与错误更新测试。
- `test/test_context.cpp`：为本项目已有的 Context 构造与请求隔离行为新增测试。

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
