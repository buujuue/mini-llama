// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "tokenizer.h"

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.push_back(BosId);
    for (unsigned char c : text) {
        int id = static_cast<int>(c);

        if (id >= VocabSize) {
            tokens.push_back(UnkId);
        } else {
            tokens.push_back(id);
        }
    }
    return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
    switch (token) {
    case UnkId:
        return "<unk>";
    case BosId:
        return "<bos>";
    case EosId:
        return "<eos>";
    default:
        if (token < 32 || token == 127) {
            return "";
        }
        if (token >= VocabSize) {
            return "<unk>";
        }
        return std::string(1, static_cast<char>(token));
    }
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += DecodeToken(token);
    }
    return result;
}
