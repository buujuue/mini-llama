// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tokenizer.h"


BpeTokenizer::BpeTokenizer() {
    B2U_.resize(256);
    U2B_.clear();

    std::vector<int> bs;
    for (int c = '!'; c <= '~'; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xA1; c <= 0xAC; ++c) {
        bs.push_back(c);
    }
    for (int c = 0xAE; c <= 0xFF; ++c) {
        bs.push_back(c);
    }

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        B2U_[bs[i]] = CodePointToUtf8(static_cast<char32_t>(cs[i]));
    }

    for (int b = 0; b < 256; ++b) {
        U2B_[B2U_[b]] = static_cast<uint8_t>(b);
    }
}

std::string BpeTokenizer::CodePointToUtf8(char32_t cp) {
    std::string result;

    if (cp <= 0x7F) {

        result += static_cast<char>(cp);
    } else if (cp <= 0x7FF) {

        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {

        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {

        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}


bool BpeTokenizer::ParseJsonInt(const std::string& content, std::size_t& pos, int& out) {

    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos >= content.size()) {
        return false;
    }
    const std::size_t start = pos;
    if (content[pos] == '-') {
        ++pos;
    }
    while (pos < content.size() && std::isdigit(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (start == pos) {
        return false;
    }
    try {

        out = std::stoi(content.substr(start, pos - start));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool BpeTokenizer::ParseJsonString(const std::string& content, std::size_t& pos, std::string& out) {
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos >= content.size() || content[pos] != '"') {
        return false;
    }

    ++pos;
    out.clear();
    while (pos < content.size() && content[pos] != '"') {
        if (content[pos] == '\\' && pos + 1 < content.size()) {

            const char esc = content[pos + 1];
            switch (esc) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case '/':
                    out += '/';
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {

                    if (pos + 5 >= content.size()) {
                        return false;
                    }
                    const std::string hex = content.substr(pos + 2, 4);
                    try {
                        const char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
                        out += CodePointToUtf8(cp);
                    } catch (const std::exception&) {

                        out += '?';
                    }

                    pos += 4;
                    break;
                }

                default:
                    out += esc;
                    break;
            }

            pos += 2;
        } else {

            out += content[pos];
            ++pos;
        }
    }
    if (pos >= content.size()) {
        return false;
    }

    ++pos;
    return true;
}


bool BpeTokenizer::Load(const std::string& vocab_path, const std::string& merges_path,
                        const std::string& special_path) {

    vocab_.clear();
    id_to_token_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    bos_id_ = eos_id_ = unk_id_ = -1;

    {
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            std::cerr << "无法打开词表文件：" << vocab_path << std::endl;
            return false;
        }

        std::stringstream buf;
        buf << f.rdbuf();
        const std::string content = buf.str();

        std::size_t pos = 0;
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos < content.size() && content[pos] == '{') {
            ++pos;
        }
        while (pos < content.size()) {
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }
            if (pos < content.size() && content[pos] == '}') {
                break;
            }

            std::string key;
            if (!ParseJsonString(content, pos, key)) {
                break;
            }
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }
            if (pos >= content.size() || content[pos] != ':') {
                break;
            }
            ++pos;
            int val = 0;
            if (!ParseJsonInt(content, pos, val)) {
                break;
            }

            if (val < 0 || val == std::numeric_limits<int>::max()) {
                std::cerr << "词表 ID 无效：token =" << key << "，ID=" << val << std::endl;
                return false;
            }

            vocab_[key] = val;
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }
            if (pos < content.size() && content[pos] == ',') {
                ++pos;
            }
        }

        if (vocab_.empty()) {
            std::cerr << "词表为空" << std::endl;
            return false;
        }

        int max_id = 0;
        for (const auto& kv : vocab_) {
            max_id = std::max(max_id, kv.second);
        }
        id_to_token_.resize(max_id + 1);
        for (const auto& kv : vocab_) {
            id_to_token_[kv.second] = kv.first;
        }

        for (const auto& kv : vocab_) {
            if (kv.first.size() >= 4 && kv.first.front() == '<' && kv.first[1] == '|' &&
                kv.first[kv.first.size() - 2] == '|' && kv.first.back() == '>') {
                special_tokens_.push_back({kv.first, kv.second});
            }
        }

        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.content.size() > b.content.size(); });
    }

    {
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            std::cerr << "无法打开合并规则文件：" << merges_path << std::endl;
            return false;
        }
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            MergePair symbols;

            if (ss >> symbols.left >> symbols.right) {
                
                merge_ranks_[symbols] = rank++;
            }
        }
    }


    {
        std::ifstream f(special_path);
        if (f.is_open()) {
            std::stringstream buf;
            buf << f.rdbuf();
            const std::string content = buf.str();

            auto find_int = [&](const std::string& key) -> int {
                std::size_t pos = content.find('"' + key + '"');
                if (pos == std::string::npos) {
                    return -1;
                }
                pos = content.find(':', pos);
                if (pos == std::string::npos) {
                    return -1;
                }
                ++pos;
                int value = -1;
                return ParseJsonInt(content, pos, value) ? value : -1;
            };
            bos_id_ = find_int("bos_id");
            eos_id_ = find_int("eos_id");
            unk_id_ = find_int("unk_id");
        }
    }

   
    std::cout << "BPE 分词器加载完成：词表项数=" << vocab_.size() << "，合并规则数=" << merge_ranks_.size()
              << "，BOS ID=" << bos_id_ << "，EOS ID=" << eos_id_ << "，未知token ID=" << unk_id_ << std::endl;
    return true;
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    std::vector<int> result;
    size_t pos = 0;
    
    while (pos < text.size()) {
        bool match = false;
        for (const auto& st : special_tokens_) {
            if (pos + st.content.size() <= text.size() && text.compare(pos, st.content.size(), st.content) == 0){
                result.push_back(st.id);
                pos += st.content.size();
                match = true;
                break;
            }
        }
        if (match) {
            continue;
        }

        size_t end = text.size();
        for (const auto& st : special_tokens_){
            size_t p = text.find(st.content, pos);
            if (p != std::string::npos && p < end){
                end = p;
            }
        }

        std::string segment = text.substr(pos, end - pos);
        pos = end;
        
        std::vector<std::string> word;
        for (unsigned char c : segment) {
            if (c >= B2U_.size()) {
                std::cerr << "ERROR: byte " << static_cast<int>(c) << " >= B2U size " << B2U_.size() << "\n";
                return {};
            }
            word.push_back(B2U_[c]);
        }


        // bpe匹配
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_idx = word.size();

            for (size_t i = 0; i + 1 < word.size(); ++i){
                auto it = merge_ranks_.find({word[i], word[i + 1]});
                if (it != merge_ranks_.end() && it->second < best_rank){
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            if (best_idx >= word.size()) {
                break;
            }

            word[best_idx] += word[best_idx + 1];
            word.erase(word.begin() + best_idx + 1);
        }

        for (const auto& w : word){
            auto it = vocab_.find(w);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(unk_id_);
            }
        }
    }
    return result;
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const{
    std::string text;
    for (int id : tokens) {
        if (id >= 0 && id < static_cast<int>(id_to_token_.size())){
            text += id_to_token_[id];
        }
    }

    std::string result = UnicodeToBytes(text);
    return result;
}
std::string BpeTokenizer::UnicodeToBytes(const std::string& text) const{
    std::string result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        std::string cp = NextUtf8CodePoint(text, pos);
        if (cp.empty()) {
            if (pos == start) {
                ++pos;
            }
            continue;
        }
        auto it = U2B_.find(cp);
        if (it != U2B_.end()){
            result += static_cast<char>(it->second);
        } else {
            result += cp;
        }
    }
    return result;
}
std::string BpeTokenizer::NextUtf8CodePoint(const std::string& s, size_t& pos) const{
    if(pos >= s.size()) {
        return "";
    }
    unsigned char first = s[pos];
    size_t len = 1;
    if ((first & 0x80) == 0) {
        len = 1;
    } else if ((first & 0xE0) == 0xC0) {
        len = 2;
    } else if ((first & 0xF0) == 0xE0) {
        len = 3;
    } else if ((first & 0xF8) == 0xF0) {
        len = 4;
    } else {
        ++pos;
        return "";
    }
    if (pos + len > s.size()) {
        ++pos;
        return "";
    }
    std::string result = s.substr(pos, len);
    pos += len;
    return result;
}

std::string BpeTokenizer::DecodeToken(int token) const {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[token];
    }
    return "";
}

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                           const std::string& merges_path,
                                           const std::string& special_path) {
    auto tokenizer = std::make_unique<BpeTokenizer>();
    if (tokenizer->Load(vocab_path, merges_path, special_path)) {
        return tokenizer;
    }
    return nullptr;
}
