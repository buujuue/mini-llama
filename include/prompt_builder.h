

// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

#include "chat.h"


// 将消息历史构造成纯文本、内置 Qwen2 或课程简化 Jinja 提示词。
class PromptBuilder {
public:
    PromptBuilder() = default;

    void SetChatTemplate(const std::string& template_str);

    std::string Build(const std::vector<ChatMessage>& messages) const;

private:
    std::string chat_template_;

    std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
    std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

// 从 GGUF v2/v3 读取原始 tokenizer.chat_template，不读取张量权重。
// 模板缺失或类型不符时，仅 qwen2 架构回退为 "qwen2"；其余返回空字符串。
// 显式存储的空模板保持为空；文件无法读取或元数据损坏时返回空字符串。
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);
