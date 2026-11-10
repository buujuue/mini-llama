// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "chat.h"

// 聊天模式的终端输入输出；命令解析由调用方负责。
class Terminal {
public:
    Terminal() = default;

    // 输出提示符并刷新 stdout。
    void PrintUserPrompt() const;
    void PrintAssistantPrefix() const;

    // 从 stdin 读取一行；EOF 或读取失败时返回空字符串。
    std::string ReadLine() const;

    // 输出已解码的 token 文本，不换行、不刷新。
    void PrintTokenText(const std::string& text) const;
    void Flush() const;
    void NewLine() const;

    void PrintHelp() const;
    void PrintStats(const ChatSession& session) const;
    void PrintParams(const SamplingParams& params) const;
    void PrintMessage(const std::string& msg) const;
};
