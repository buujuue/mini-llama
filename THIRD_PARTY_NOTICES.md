# 第三方代码声明

本项目的 `include/tokenizer.h`、`src/tokenizer/` 和 `test/test_tokenizer.cpp` 包含从课程配套项目 `mini-llama-cpp` 移植、适配的分词器接口、实现和基础测试。

`test/test_kv_cache.cpp` 包含从同一项目 `tests/gtest/test_kv_cache.cc` 移植、适配的 KV cache 测试。

参考文件：`include/mini_llama/tokenizer.h`、`src/tokenizer.cc`、`src/json_vocab_tokenizer.cc`、`src/bpe_tokenizer.cc`、`tests/gtest/test_tokenizer.cc`、`tests/gtest/test_kv_cache.cc`。以下保留原项目的完整 MIT 许可；本声明适用于上述移植部分。

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
