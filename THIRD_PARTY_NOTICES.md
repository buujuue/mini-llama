# 第三方许可

本项目从 `mini-llama-cpp` 移植的 tokenizer、RadixTree、RequestContext、
前向输入检查、采样器、会话、PromptBuilder、Terminal、模型描述解析器、加载失败处理及相关测试
遵循以下 MIT 许可。模型描述解析与加载失败处理涉及 `include/loader.h`、`src/loader.cpp` 和
`test/test_loader.cpp`，原始实现位于课程参考源码的 `include/mini_llama/loader.h`、
`src/loader.cc` 和 `tests/gtest/test_loader.cc`。

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
