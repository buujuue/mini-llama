# mini_llama

## 项目介绍

mini_llama 是一个面向 Llama 大语言模型推理的 C++17 学习与实践项目。项目参考 `mini-llama-cpp` 和 `llama.cpp`，通过手写张量与基础算子，理解 Transformer 的核心计算过程，为后续实现完整推理流程打下基础。

目前项目以 CPU 上的 FP32 计算为主，构建产物为静态库，已实现：

- **张量容器**：`Tensor` 提供连续数据存储、形状查询、通用及 1D–4D 索引访问，并检查形状和索引边界。
- **基础算子**：矩阵乘法（Matmul）、线性变换（Linear）、均方根归一化（RmsNorm）、Softmax、SwiGLU 和旋转位置编码（RoPE，支持 Normal 与 NeoX 两种布局）。
- **分词器**：ASCII、JSON 词表最长匹配和字节级 BPE（课程简化实现），以及 JSON/ASCII 回退和 BPE 创建接口。保留原有 `JsonVocabTokenizor` 名称，并提供与实例一致的 `JsonVocabTokenizer` 别名。
- **单元测试**：使用 GoogleTest 和 CTest 验证张量访问、算子数值结果、形状约束及分词器行为；分词器测试自行生成小词表，无需下载模型。

当前已实现张量、基础算子与分词器，尚未接入模型权重加载和完整文本生成流程。项目目录与重要文件见 [GUIDE.md](GUIDE.md)；学习文档和上游参考源码保存在本地，不提交到 Git。移植代码的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

提交代码请遵循 [项目提交规范](CONTRIBUTING.md)。

## 技术栈

- C++17（GCC / MinGW 或 MSVC）
- CMake 3.16+、Ninja
- GoogleTest 1.18.0、CTest

## 运行命令

在项目根目录执行，首次构建测试需要 Git 和网络连接。

**Linux / WSL**

```bash
cmake -S . -B build/linux -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build/linux --parallel
ctest --test-dir build/linux --output-on-failure
```

**Windows（已配置 C++ 编译器的终端）**

```powershell
cmake -S . -B build/windows -DBUILD_TESTING=ON
cmake --build build/windows --config Debug --parallel
ctest --test-dir build/windows -C Debug --output-on-failure
```

只运行分词器测试时，使用 `ctest --test-dir build/windows -C Debug -L tokenizer --output-on-failure`；Linux / WSL 将构建目录改为 `build/linux`。

JSON 词表使用 `[{"id": 0, "content": "<unk>", "special": true}, ...]` 格式。BPE 使用 `{"token": id, ...}` 词表、每行一对符号的合并规则和可选的 `bos_id` / `eos_id` / `unk_id` 配置。ASCII 与 JSON 编码添加 BOS；BPE 不自动添加 BOS/EOS，`DecodeToken` 返回词表映射符号，`Decode` 还原文本字节。
