# mini_llama

## 项目介绍

mini_llama 是一个面向 Llama 大语言模型推理的 C++17 学习与实践项目。项目参考 `mini-llama-cpp` 和 `llama.cpp`，通过手写张量与基础算子，理解 Transformer 的核心计算过程，为后续实现完整推理流程打下基础。

目前项目以 CPU 上的 FP32 计算为主，构建产物为静态库，已实现：

- **张量容器**：`Tensor` 提供连续数据存储、形状查询、通用及 1D–4D 索引访问，并检查形状和索引边界。
- **基础算子**：矩阵乘法（Matmul）、线性变换（Linear）、均方根归一化（RmsNorm）、Softmax、SwiGLU 和旋转位置编码（RoPE，支持 Normal 与 NeoX 两种布局）。
- **分词器**：ASCII、JSON 词表最长匹配和字节级 BPE（课程简化实现），以及 JSON/ASCII 回退和 BPE 创建接口。保留原有 `JsonVocabTokenizor` 名称，并提供与实例一致的 `JsonVocabTokenizer` 别名。
- **前缀树**：`RadixTree` 支持 token 序列插入、压缩节点分裂、最长公共前缀长度查询、去重计数和清空，用于前缀缓存索引；不持有 KV cache 数据。
- **批处理与上下文**：`batch` 构造单 token 或序列批次并生成连续位置；`Context` 绑定模型、按配置初始化独立 KV cache 与请求状态。
- **请求跟踪**：`RequestContext` 提供 trace ID、请求计时、事件记录、错误状态与只读摘要/事件格式化。
- **单元测试**：使用 GoogleTest 和 CTest 验证张量访问、算子数值结果、形状约束、分词器、KV cache、前缀树、批次构造、上下文初始化与请求跟踪；分词器测试自行生成小词表，无需下载模型。

张量、基础算子、分词器与前缀树已接入静态库；KV cache、batch、Context 与 RequestContext 当前通过独立测试目标编译验证，尚未接入模型权重加载和完整文本生成流程。项目目录与重要文件见 [GUIDE.md](GUIDE.md)；学习文档和上游参考源码保存在本地，不提交到 Git。移植代码的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

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

MinGW 构建会把编译器配套的运行库复制到测试程序目录，避免系统 `PATH` 中另一版本的 MinGW 导致“无法定位程序输入点”。已有构建目录重新配置一次即可部署运行库，无需修改系统 `PATH`。

只运行分词器测试时，使用 `ctest --test-dir build/windows -C Debug -L tokenizer --output-on-failure`；Linux / WSL 将构建目录改为 `build/linux`。

只运行前缀树测试时，使用 `ctest --test-dir build/linux -L radix_tree --output-on-failure`；Windows 使用 `build/windows -C Debug`。`RadixTree::size()` 统计不同的完整序列（包含空序列），`LongestPrefix()` 返回与任一已插入序列的最长公共前缀长度，允许查询在压缩节点中间结束。

只运行新增的批次、上下文和请求跟踪测试时，使用：

```powershell
ctest --test-dir build/windows -C Debug -L "^(batch|context|request_context)$" --output-on-failure
```

Linux / WSL 将构建目录改为 `build/linux`。本次移植覆盖已实现的构造、事件和格式化行为；依赖模型加载与 Forward 的实例测试待相应功能实现后接入。`RequestContext` 的各阶段汇总统计字段尚无公开更新接口，目前只验收其初始零值，事件中的非零耗时和 token 数单独验证。

JSON 词表使用 `[{"id": 0, "content": "<unk>", "special": true}, ...]` 格式。BPE 使用 `{"token": id, ...}` 词表、每行一对符号的合并规则和可选的 `bos_id` / `eos_id` / `unk_id` 配置。ASCII 与 JSON 编码添加 BOS；BPE 不自动添加 BOS/EOS，`DecodeToken` 返回词表映射符号，`Decode` 还原文本字节。
