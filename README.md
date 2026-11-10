# mini_llama

## 项目介绍

mini_llama 是一个面向 Llama 大语言模型推理的 C++17 学习与实践项目。项目参考 `mini-llama-cpp` 和 `llama.cpp`，通过手写张量与基础算子，理解 Transformer 的核心计算过程，为后续实现完整推理流程打下基础。

目前项目以 CPU 上的 FP32 计算为主，构建产物为静态库，已实现：

- **张量容器**：`Tensor` 提供连续数据存储、形状查询、通用及 1D–4D 索引访问，并检查形状和索引边界。
- **基础算子**：矩阵乘法（Matmul）、线性变换（Linear）、均方根归一化（RmsNorm）、Softmax、SwiGLU 和旋转位置编码（RoPE，支持 Normal 与 NeoX 两种布局）。
- **分词器**：ASCII、JSON 词表最长匹配和字节级 BPE（课程简化实现），以及 JSON/ASCII 回退和 BPE 创建接口。保留原有 `JsonVocabTokenizor` 名称，并提供与实例一致的 `JsonVocabTokenizer` 别名。
- **前缀树**：`RadixTree` 支持 token 序列插入、压缩节点分裂、最长公共前缀长度查询、去重计数和清空，用于前缀缓存索引；不持有 KV cache 数据。
- **批处理与上下文**：`Batch` 构造单 token 或序列批次，通过 `GetPos(int i)` 按起始位置与下标计算连续位置；`Context` 绑定模型、按配置初始化独立 KV cache 与请求状态。
- **请求跟踪**：`RequestContext` 提供 trace ID、请求计时、事件记录、错误状态与只读摘要/事件格式化。
- **采样器**：`Sampler::Sample` 根据 `SamplingParams` 选择贪心、全词表温度或 Top-k 采样，支持固定种子复现，并校验 logits 与参数。
- **会话状态**：`ChatSession` 管理消息、token 历史、前缀缓存和累计统计；`ChatMessage` 使用 `ChatRole` 枚举保存角色。
- **提示词构建**：`PromptBuilder` 将消息历史构造成纯文本、内置 Qwen2 或课程简化 Jinja 模板文本，角色名适配 `ChatRole`。
- **终端交互**：`Terminal` 提供提示符、逐行输入、流式 token 输出，以及帮助、会话统计和采样参数展示。
- **单元测试**：使用 GoogleTest 和 CTest 验证张量访问、算子数值结果、形状约束、分词器、KV cache、前缀树、批次构造、上下文初始化与请求跟踪；分词器测试自行生成小词表，无需下载模型。

张量、基础算子、分词器、前缀树、采样器、会话状态、提示词构建与终端输入输出已接入静态库；KV cache、batch、Context 与 RequestContext 当前通过独立测试目标编译验证，尚未接入模型权重加载和完整文本生成流程。项目目录与重要文件见 [GUIDE.md](GUIDE.md)；学习文档和上游参考源码保存在本地，不提交到 Git。移植代码的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

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

Linux / WSL 将构建目录改为 `build/linux`。`RequestContext` 的各阶段汇总统计字段尚无公开更新接口，目前只验收其初始零值，事件中的非零耗时和 token 数单独验证。

采样器测试使用 `ctest --test-dir build/linux -L sampler --output-on-failure`。`temperature < 1e-6` 或 `top_k=1` 时贪心，`top_k=0` 时禁用 Top-k，`top_k>1` 时仅从前 k 个候选项中按温度采样。`seed` 仅在构造 `Sampler` 时使用，0 表示使用随机设备；温度采样会推进实例的随机状态，贪心采样不消耗随机数。

会话测试使用 `ctest --test-dir build/linux -L chat --output-on-failure`，包含 7 项课程移植测试和 5 项补充测试。`AddMessage` 直接接受 `ChatRole` 枚举；`Clear` 清空消息、token 历史、前缀缓存和累计统计，保留采样参数。Windows 使用 `build/windows -C Debug`。

提示词测试使用 `ctest --test-dir build/linux -L prompt_builder --output-on-failure`。默认按 `System: / User: / Assistant: ` 格式输出历史，并追加 `Assistant:`；`SetChatTemplate("qwen2")` 使用 ChatML 格式，在历史中没有 system 消息时注入默认系统提示。`Build(session.messages)` 返回供 tokenizer 编码的字符串，不修改会话状态。

`SetChatTemplate` 还接受以 `{%` 或 `{{` 开头的课程简化 Jinja 模板，支持单引号字符串拼接、消息字段与固定下标访问、`loop.first`、`if/endif`、`for message in messages/endfor`、`and/or`、`==/!=`，`add_generation_prompt` 固定为 true。接受标签内的 `-` 标记，但不实现完整 Jinja 的空白裁剪、`else` 或过滤器；未知模板名称回退到纯文本，空模板恢复默认模式。当前项目尚无 GGUF 元数据读取模块，因此本次仅移植 `PromptBuilder` 类，模板由调用方传入，未引入 `LoadChatTemplateFromGguf`。

终端测试使用 `ctest --test-dir build/linux -L terminal --output-on-failure`，包含 7 项课程移植测试和 7 项补充测试，自动重定向标准输入输出，无需人工交互。Windows 使用 `build/windows -C Debug`。`PrintUserPrompt` 和 `PrintAssistantPrefix` 输出后刷新，`PrintTokenText` 不换行、不刷新，调用方可用 `Flush` 刷新流式输出；`ReadLine` 保留行内空白，空行与 EOF 均返回空字符串，可用 `std::cin.eof()` 区分。统计中的耗时和吞吐保留两位小数，零耗时显示 `n/a`，统计和参数展示不改变 `std::cout` 的浮点格式。帮助列出 `/help`、`/clear`、`/stats`、`/params` 和 `/exit`，命令解析由调用方负责；当前尚无可运行的聊天主程序。

前向测试使用固定的内存 F32 权重，无需模型加载器或外部模型文件。已移植单 token 输出与 KV 写入、连续位置、非法输入，以及批次与逐 token 计算等价、历史更新和空批次拒绝测试；尚未实现的量化测试暂不接入。单独运行前向测试：

```powershell
ctest --test-dir build/windows -C Debug -L forward --output-on-failure
```

2026-11-03 最新 Windows / MinGW Debug 验证：编译成功，183 项测试全部通过，包含 50 项算子测试及此前失败的 Batch、GQA 和 NeoX 测试。本轮测试适配为 7 处 `Rope` 调用显式传入 `RopeType::Normal`，未修改功能代码；完整输出位于 `build/windows/ctest-run.log`。

JSON 词表使用 `[{"id": 0, "content": "<unk>", "special": true}, ...]` 格式。BPE 使用 `{"token": id, ...}` 词表、每行一对符号的合并规则和可选的 `bos_id` / `eos_id` / `unk_id` 配置。ASCII 与 JSON 编码添加 BOS；BPE 不自动添加 BOS/EOS，`DecodeToken` 返回词表映射符号，`Decode` 还原文本字节。
