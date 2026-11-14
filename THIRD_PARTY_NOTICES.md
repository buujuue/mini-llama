# 第三方许可

本项目从 `mini-llama-cpp` 移植的 tokenizer、RadixTree、RequestContext、
前向输入检查、采样器、会话、PromptBuilder、Terminal、模型描述解析器、加载失败处理、GGUF 值读取、GGUF 张量名称映射、GGUF 配置构建、GGUF 模型加载及相关测试
遵循以下 MIT 许可。模型描述解析与加载失败处理涉及 `include/loader.h`、`src/loader.cpp` 和
`test/test_loader.cpp`，原始实现位于课程参考源码的 `include/mini_llama/loader.h`、
`src/loader.cc` 和 `tests/gtest/test_loader.cc`。
`src/gguf.cpp` 中的值读取、类型名及类型化元数据读取辅助函数适配自课程参考源码的 `src/gguf.cc`。
`src/gguf_loader.cpp` 中的 `MapTensorName`、`BuildConfig`、`LoadGgufModel` 及权重形状校验适配自课程参考源码的 `src/gguf_loader.cc`。
`test/test_gguf.cpp` 的基础场景移植自 `tests/gtest/test_gguf.cc`，
`test/test_gguf_loader.cpp` 的加载失败场景适配自 `tests/gtest/test_gguf_loader.cc`；
`test/gguf_fixture.h` 的字节构造辅助函数与微型 GGUF 布局适配自
`tests/gtest/test_gguf.cc` 和 `scripts/make_test_gguf.py`。
`scripts/export_gguf_tokenizer.py` 移植自课程参考源码的同名脚本，
增加 GGUF 文件头、类型、截断和重复词表检查，输出格式保持兼容。

`include/gguf_tokenizer.h` 和 `src/tokenizer/gguf_tokenizer.cpp` 适配自课程的
`include/mini_llama/gguf_tokenizer.h` 和 `src/gguf_tokenizer.cc`；
`test/test_gguf_tokenizer.cpp` 的基础场景及微型词表适配自
`tests/gtest/test_gguf_tokenizer.cc` 和 `tests/gguf_tokenizer_fixture.h`。

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
