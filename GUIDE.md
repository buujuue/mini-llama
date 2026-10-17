# mini_llama 项目目录索引



```text
mini_llama/
├── AGENTS.md                      # 项目约定，索引 GUIDE.md
├── GUIDE.md                       # 本文件：本地目录导航
├── README.md                      # 项目简介
├── CONTRIBUTING.md                # 项目提交规范、验证要求与 PR 审核流程
├── THIRD_PARTY_NOTICES.md          # 移植课程 tokenizer 源码与测试的 MIT 许可
├── CMakeLists.txt                 # C++17 静态库与 CTest 测试入口
├── .gitignore                     # 本地资料、配置和产物的忽略规则
├── .vscode/                       # Git 忽略：本地 VSCode 配置
│   ├── settings.json              # CMake 启用测试、按系统构建并同步编译数据库给 clangd
│   └── c_cpp_properties.json      # C/C++ 扩展使用 CMake Tools 和根目录编译数据库
├── .agents/                       # 项目级技能
│   └── skills/
│       └── complete-error-context/ # 用户显式调用：补全 IDE 或指定文件的报错
│           ├── SKILL.md            # 技能说明与诊断工作流
│           └── agents/openai.yaml  # 禁止自动调用的策略
├── include/                       # 自己实现的头文件
│   ├── tensor.h                    # Tensor：构造、元信息/数据查询、通用/1D–4D 访问和行指针
│   ├── ops.h                       # 六个 F32 基础算子的接口、RoPE 辅助函数、形状约定与 RopeType
│   └── tokenizer.h                 # ITokenizer、ASCII/JSON/BPE tokenizer 与创建接口
├── src/                           # 自己实现的源文件
│   ├── tensor.cpp                  # Tensor 实现与形状、索引检查
│   ├── ops.cpp                     # Matmul/Linear Naive 实现，以及 RmsNorm、Softmax、SwiGlu、Rope
│   └── tokenizer/                 # 已接入静态库的 tokenizer 实现
│       ├── ascii_tokenizer.cpp     # ASCII 编码与解码
│       ├── json_vocab_tokenizer.cpp # JSON 词表加载、最长匹配编码、解码与 ASCII 回退工厂
│       └── bpe_tokenizer.cpp       # BPE 加载、字节映射、合并、特殊 token、解码与工厂
├── test/                          # GoogleTest 单元测试
│   ├── CMakeLists.txt              # 下载 GoogleTest、构建三个测试程序并注册 CTest
│   ├── test_tensor.cpp             # Tensor 正常行为、const 访问与边界测试
│   ├── test_ops.cpp                # 六个已实现基础算子的数值、形状与边界验收，CTest 标签 ops
│   └── test_tokenizer.cpp          # ASCII/JSON/BPE 与工厂测试，自建词表，CTest 标签 tokenizer
├── docs/                          # Git 忽略：本地文档
│   ├── agents/                    # 工程技能使用的本地约定
│   └── source/
├── references/                    # Git 忽略：上游参考源码
│   ├── mini-llama-cpp/             # 课程配套推理引擎
│   └── llama.cpp/                 # 工业级实现参考
├── build/                         # Git 忽略：CMake/Ninja 构建产物
│   ├── vscode-linux/               # VSCode WSL/Linux 构建与测试编译数据库
│   ├── vscode-win32/               # VSCode Windows 构建与编译数据库
│   ├── windows/                    # Windows 本机验证使用的独立构建目录
│   │   ├── _deps/                  # CMake 下载的 GoogleTest 源码和构建产物
│   │   ├── test/tensor_tests.exe   # Tensor 测试程序
│   │   ├── test/ops_tests.exe      # 基础算子测试程序
│   │   └── test/tokenizer_tests.exe # tokenizer 测试程序
│   ├── windows-no-tests/           # 验证 BUILD_TESTING=OFF 的纯库构建
│   └── .npm-cache/                # 文档检索工具的本地缓存
├── compile_commands.json          # Git 忽略：编译命令数据库
├── .cache/                        # Git 忽略：工具缓存
└── .git/                          # 本仓库 Git 元数据
```
