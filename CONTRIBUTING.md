# mini_llama 项目提交规范

本规范依据《代码提交流程》整理，适用于 mini_llama 的课程作业开发、代码提交和导师审核。提交顺序为：准备个人仓库 → 创建开发分支 → 开发与验证 → 暂存检查 → 本地提交 → 推送 → PR 审核 → 合并与同步。

## 1. 仓库准备

- 作业仓库放在自己的 GitHub 账号下；包含私有课程源码时必须使用 Private。
- 按课程流程邀请 `yus3nable` 为 Collaborator，待邀请被接受后再请求审核。
- `origin` 指向个人作业仓库；需要跟进课程起始代码时，`upstream` 指向课程仓库。不要向课程仓库推送作业。
- 新仓库先建立主分支基线，再开发功能；已有仓库直接沿用现有历史，不重复初始化或替换远程配置。
- 主分支默认使用 `main`；已有仓库使用 `master` 时，将下文命令中的 `main` 替换为 `master`。

首次操作前检查仓库和远程地址：

```bash
git status
git branch --show-current
git remote -v
```

尚未配置 `origin` 时，使用自己的仓库地址配置；执行前替换用户名占位符，并确认远程仓库已创建：

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/mini_llama.git
git push -u origin main
```

首次上传前先完成第 3～5 节的内容检查、验证和本地提交。HTTPS 推送需配置凭据管理器、Token 或 GitHub CLI；也可使用已配置 SSH key 的 SSH 地址。

## 2. 开发分支

每项独立任务创建一个开发分支。切换前先确认已有改动已妥善保存，避免把其他任务的修改带入新分支。

```bash
git checkout main
git pull --ff-only
git checkout -b feature/tensor-indexing
```

分支名使用英文小写，单词以连字符连接：

| 类型 | 用途 | 示例 |
| --- | --- | --- |
| `feature/` | 新功能 | `feature/rope-neox` |
| `fix/` | 缺陷修复 | `fix/softmax-stability` |
| `refactor/` | 保持行为的重构 | `refactor/tensor-access` |
| `test/` | 测试完善 | `test/ops-boundaries` |
| `docs/` | 文档修改 | `docs/contributing` |

同一个 PR 聚焦一个明确任务；无关功能、批量格式调整和其他任务修改应单独提交。

## 3. 提交范围

- 源码位于 `include/`、`src/`，单元测试位于 `test/`；构建入口为根目录和测试目录的 `CMakeLists.txt`。
- 功能或行为修改应补充能覆盖变化的测试；仅修改文档时检查内容和链接即可。
- 修改公共接口、形状约定、边界行为或数值算法时，说明影响范围，并覆盖对应的正常、边界或异常情况。
- 新增、删除或移动目录及重要入口文件，或修改忽略规则时，同步更新 `GUIDE.md`；运行方式变化时同步更新 `README.md`。
- 遵循 `.gitignore`：不提交 `build/`、`docs/`、`references/`、`.vscode/`、`.agents/`、`.cache/`、`compile_commands.json` 等本地资料、配置与产物。
- 不提交模型权重、临时日志、个人凭据或无关文件；新增此类本地文件时检查忽略规则。
- 已跟踪文件不会因新增忽略规则自动退出版本控制，必须检查实际差异。

## 4. 编译与验证

源码、测试或构建配置修改后，在项目根目录配置测试构建、编译并运行 CTest。首次测试配置通过 Git 获取 GoogleTest，需要网络连接。

Linux / WSL（与 README 的运行方式一致）：

```bash
cmake -S . -B build/linux -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build/linux --parallel
ctest --test-dir build/linux --output-on-failure
```

Windows：先进入已配置 C++ 编译器的终端；首次配置使用该环境支持的默认生成器，随后复用同一构建目录：

```powershell
cmake -S . -B build/windows -DBUILD_TESTING=ON
cmake --build build/windows --config Debug --parallel
ctest --test-dir build/windows -C Debug --output-on-failure
```

已有构建目录时沿用其生成器；切换编译器或生成器应使用新的构建目录。单配置生成器需要 Debug 时，在首次配置中加入 `-DCMAKE_BUILD_TYPE=Debug`。

提交前确保相关测试通过。若环境限制导致无法执行，PR 中明确说明未执行的检查及原因，不得把未验证写成通过。涉及性能或数值改动时，补充输入、运行环境、比较方法和结果。

## 5. 暂存与提交说明

先检查差异，再按本次任务明确选择文件：

```bash
git status
git diff
git add include/tensor.h src/tensor.cpp test/test_tensor.cpp
git diff --cached
git diff --cached --check
```

上述文件仅为张量任务示例，按实际改动替换。`git add -u` 不包含新文件；使用 `git add .` 前须确认根目录全部改动都属于本次任务。最后检查暂存内容，避免混入其他未完成工作。

提交信息格式：

```text
<type>(<scope>): <简要说明>
```

| type | 含义 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | 缺陷修复 |
| `refactor` | 不改变外部行为的重构 |
| `test` | 测试新增或调整 |
| `docs` | 文档变更 |
| `build` | 构建或依赖配置变更 |
| `perf` | 性能优化 |
| `chore` | 其他维护工作 |

scope 使用受影响模块，例如 `tensor`、`ops`、`cmake`、`repo`。说明可以使用中文，需写明实际变化，避免仅写“修改代码”“更新”或“修复问题”。

```bash
git commit -m "feat(tensor): 增加四维索引访问与边界检查"
git commit -m "fix(ops): 修复 Softmax 大输入下的溢出"
git commit -m "docs(repo): 补充项目提交规范"
```

每个 commit 表达一组完整、可解释的修改。一个功能允许多个 commit；审核期间通过新 commit 补充修复。`git commit --amend` 仅用于尚未推送的最后一次提交，默认保留已推送和审核中的历史。

## 6. 推送与 PR

首次推送开发分支并建立跟踪关系：

```bash
git push -u origin feature/tensor-indexing
```

后续在同一分支提交后执行 `git push`。推送前再次确认分支名称和 `origin` 地址。

PR 在自己的作业仓库内创建：base 为 `main`，compare 为本次开发分支。创建前检查 Files changed，确认只包含本次任务内容。标题说明本次功能或修复，描述按以下模板填写：

```markdown
## 本次目标
说明要实现的功能或要修复的问题。

## 主要改动
列出核心修改及影响范围；接口或行为变化需明确说明。

## 验证
- 环境：操作系统、编译器和构建配置。
- 执行方式：实际使用的编译、运行和测试命令。
- 结果：通过的检查、失败项和未执行项及原因。

## 希望重点审核
列出需要导师重点检查的实现、疑问或已知限制。
```

选择 `yus3nable` 为 Reviewer；无法选择时先检查 Collaborator 邀请是否被接受。按课程要求将 PR 链接发到服务群并 @导师。

## 7. 审核与合并

- 根据评论修改代码，完成相关验证，追加 commit 并推送到原分支；PR 会自动更新。
- 回复修改情况，修复后再次请求 review；有分歧时解释原因，确认处理方案。
- 完成审核意见并获得导师认可后再合并。合并方式按导师或仓库约定选择。
- 合并后确认本地改动已保存，再同步主分支：

```bash
git checkout main
git pull --ff-only
git fetch --prune
```

确认主分支已包含本次修改后，再创建下一项任务的开发分支。普通 merge 后可使用 `git branch -d` 清理已合并分支；squash 或 rebase 合并后，应先核对实际内容，不直接强制删除。

## 8. 提交前检查清单

- [ ] 当前分支和远程仓库正确，改动仅属于本次任务。
- [ ] 功能、接口、形状与边界行为已说明，相关测试已补充。
- [ ] 已完成适用的编译与测试，并如实记录结果。
- [ ] 已检查暂存差异及空白错误，没有混入本地资料或构建产物。
- [ ] 提交信息符合格式，`GUIDE.md` 和相关说明已同步更新。
- [ ] PR 的 base、compare、描述和 Reviewer 正确。
- [ ] 审核意见已处理，获得认可后再合并。
