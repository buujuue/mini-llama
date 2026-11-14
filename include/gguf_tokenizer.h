// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
#pragma once

#include "tokenizer.h"

// 从 GGUF 元数据加载课程版 GPT-2 字节级 BPE，不读取张量权重。
// Encode 不自动添加 BOS/EOS；DecodeToken 返回词表映射符号，Decode 还原字节。
class GgufTokenizer : public ITokenizer {
public:
    GgufTokenizer() = default;
    ~GgufTokenizer() override = default;

    // 缺少词表或文件无效时返回 false；重新加载失败后清空旧状态。
    bool Load(const std::string& gguf_path);

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override {
        return static_cast<int>(id_to_token_.size());
    }
    int bos_id() const override {
        return bos_id_;
    }
    int eos_id() const override {
        return eos_id_;
    }
    int unk_id() const override {
        return unk_id_;
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> merge_ranks_;
    std::vector<std::pair<std::string, int>> special_tokens_;  // 按长度降序匹配。

    std::vector<std::string> b2u_;
    std::unordered_map<std::string, uint8_t> u2b_;

    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    void BuildByteMappings();
};

// 加载失败时返回 nullptr，不回退到其他分词器。
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);
