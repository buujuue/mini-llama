// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include "tokenizer.h"

namespace {

std::string JsonErrorContext(const std::string& content, std::size_t pos) {
    const std::size_t offset = std::min(pos, content.size());
    const std::size_t start = offset > 16 ? offset - 16 : 0;
    const std::size_t end = start + std::min<std::size_t>(32, content.size() - start);
    std::string snippet;
    for (std::size_t i = start; i < end; ++i) {
        const unsigned char c = static_cast<unsigned char>(content[i]);
        switch (c) {
        case '\n': snippet += "\\n"; break;
        case '\r': snippet += "\\r"; break;
        case '\t': snippet += "\\t"; break;
        case '\\': snippet += "\\\\"; break;
        case '"': snippet += "\\\""; break;
        default:
            if (c < 0x20 || c == 0x7F) {
                constexpr char hex[] = "0123456789ABCDEF";
                snippet += "\\x";
                snippet += hex[c >> 4];
                snippet += hex[c & 0x0F];
            } else {
                snippet += static_cast<char>(c);
            }
        }
    }
    return "字节偏移: " + std::to_string(pos) + "，内容字节数: " + std::to_string(content.size()) +
           "，片段起始偏移: " + std::to_string(start) + "，附近内容: \"" + snippet +
           (end < content.size() ? "..." : "") + "\"";
}

}  // namespace

JsonVocabTokenizor::JsonVocabTokenizor(const std::string& vocab_path) try {
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        throw std::runtime_error("无法打开词表文件：要求文件存在且可读。");
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    if (buffer.fail()) {
        f.peek();
    }
    if (f.bad() || buffer.bad()) {
        throw std::runtime_error("读取词表文件失败：已读取字节数: " +
                                 std::to_string(buffer.str().size()) + "。");
    }

    std::string content = buffer.str();

    size_t pos = content.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("词表 JSON 缺少数组起始符：预期 '['，" +
                                 JsonErrorContext(content, 0) + "。");
    }
    ++pos;

    int max_id = -1;

    while (pos < content.size()) {
   
        while (pos < content.size() && content[pos] != '{') {
            if (content[pos] == ']') {
                break;
            }
            ++pos;
        }
        if (pos >= content.size() || content[pos] != '{') {
            break;
        }
        const std::size_t entry_start = pos;
        ++pos;

        VocabEntry entry;
        bool has_id = false;
        bool has_content = false;

        
        while (pos < content.size() && content[pos] != '}') {
            
            while (pos < content.size() &&
                   (std::isspace(static_cast<unsigned char>(content[pos])) || content[pos] == ',')) {
                ++pos;
            }
            if (pos >= content.size() || content[pos] == '}') {
                break;
            }

            
            std::string key = ParseJsonString(content, pos);

            
            while (pos < content.size() && content[pos] != ':') {
                ++pos;
            }
            if (pos < content.size()) {
                ++pos;
            }
            
            while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
                ++pos;
            }

            if (key == "id") {
                entry.id = ParseJsonInt(content, pos);
                has_id = true;
            } else if (key == "content") {
                entry.content = ParseJsonString(content, pos);
                has_content = true;
            } else if (key == "special") {
                entry.special = ParseJsonBool(content, pos);
            } else {
                
                if (pos < content.size() && content[pos] == '"') {
                    ParseJsonString(content, pos);
                } else {
                    while (pos < content.size() && content[pos] != ',' && content[pos] != '}') {
                        ++pos;
                    }
                }
            }
        }

        if (pos < content.size() && content[pos] == '}') {
            ++pos;
        }

        if (!has_id || !has_content) {
            const std::string missing_fields = !has_id ? (!has_content ? "id, content" : "id") : "content";
            throw std::runtime_error("词条缺少必需字段：缺失字段: " + missing_fields +
                                     "，词条索引: " + std::to_string(content_to_id_.size()) + "，" +
                                     JsonErrorContext(content, entry_start) + "。");
        }

        if (entry.id > max_id) {
            max_id = entry.id;
        }
        if (entry.id < 0) {
            throw std::runtime_error("词条 id 必须非负：id: " + std::to_string(entry.id) +
                                     "，词条索引: " + std::to_string(content_to_id_.size()) + "，" +
                                     JsonErrorContext(content, entry_start) + "。");
        }
        if (entry.id == std::numeric_limits<int>::max()) {
            throw std::runtime_error("词条 id 导致词表大小溢出：要求 id + 1 不超过 int 最大值: " +
                                     std::to_string(std::numeric_limits<int>::max()) +
                                     "，id: " + std::to_string(entry.id) +
                                     "，词条索引: " + std::to_string(content_to_id_.size()) + "，" +
                                     JsonErrorContext(content, entry_start) + "。");
        }

        if (entry.content == "<bos>") {
            bos_id_ = entry.id;
        } else if (entry.content == "<eos>") {
            eos_id_ = entry.id;
        } else if (entry.content == "<unk>") {
            unk_id_ = entry.id;
        }

        if (static_cast<size_t>(entry.id) >= id_to_content_.size()) {
            id_to_content_.resize(entry.id + 1);
        }
        id_to_content_[entry.id] = entry;
        content_to_id_.push_back(entry);
    }

    if (max_id < 0) {
        throw std::runtime_error("词表不包含任何词条：预期词条数: > 0，实际词条数: " +
                                 std::to_string(content_to_id_.size()) + "，" +
                                 JsonErrorContext(content, pos) + "。");
    }

    vocab_size_ = max_id + 1;

    
    std::sort(content_to_id_.begin(), content_to_id_.end(),
              [](const auto& a, const auto& b) { return a.content.size() > b.content.size(); });
} catch (const std::runtime_error& error) {
    throw std::runtime_error("[JsonVocabTokenizor] 加载词表失败：vocab_path: \"" + vocab_path +
                             "\"，" + error.what());
}

std::vector<int> JsonVocabTokenizor::Encode(const std::string& text) const {
    std::vector<int> tokens{bos_id_};
    std::size_t pos = 0;
    while (pos < text.size()) {
        bool matched = false;
        // 加载时已按内容长度降序排列；跳过空词条，保证游标总能前进。
        for (const auto& entry : content_to_id_) {
            if (!entry.content.empty() &&
                text.compare(pos, entry.content.size(), entry.content) == 0) {
                tokens.push_back(entry.id);
                pos += entry.content.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            // 与课程实例一致：无匹配时按字节消耗输入，每字节产生一个 UNK。
            tokens.push_back(unk_id_);
            ++pos;
        }
    }
    return tokens;
}

std::string JsonVocabTokenizor::DecodeToken(int token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= id_to_content_.size()) {
        return "<unk>";
    }
    return id_to_content_[token].content;
}

std::string JsonVocabTokenizor::Decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int token : tokens) {
        result += DecodeToken(token);
    }
    return result;
}

std::string JsonVocabTokenizor::ParseJsonString(const std::string& content, std::size_t& pos) {
    const std::size_t start = pos;
    while (pos < content.size() && content[pos] != '"') {
        ++pos;
    }
    if (pos >= content.size() || content[pos] != '"') {
        throw std::runtime_error("ParseJsonString 缺少字符串起始引号：预期 '\"'，搜索起始偏移: " +
                                 std::to_string(start) + "，" + JsonErrorContext(content, pos) + "。");
    }
    const std::size_t string_start = pos;
    ++pos;
    std::string result;
    while (pos < content.size() && content[pos] != '"') {
        if (content[pos] == '\\' && pos + 1 < content.size()) {
            char next = content[pos + 1];
            if (next == 'n') {
                result += '\n';
            } else if (next == 't') {
                result += '\t';
            } else if (next == 'r') {
                result += '\r';
            } else if (next == '\\') {
                result += '\\';
            } else if (next == '"') {
                result += '"';
            } else if (next == 'u' && pos + 5 < content.size()) {
                std::string hex = content.substr(pos + 2, 4);
                unsigned int codepoint;
                try {
                    codepoint = std::stoul(hex, nullptr, 16);
                } catch (const std::invalid_argument&) {
                    throw std::runtime_error("ParseJsonString 的 Unicode 转义无法解析为十六进制整数："
                                             "预期 \\uXXXX，" + JsonErrorContext(content, pos) + "。");
                } catch (const std::out_of_range&) {
                    throw std::runtime_error("ParseJsonString 的 Unicode 转义数值超出范围："
                                             "预期 \\uXXXX，" + JsonErrorContext(content, pos) + "。");
                }
                if (codepoint <= 0x7F) {
                    result += static_cast<char>(codepoint);
                } else {
                    result += '?';
                }
                pos += 6;
                continue;
            } else {
                result += next;
            }
            pos += 2;
        } else {
            result += content[pos];
            ++pos;
        }
    }
    if (pos >= content.size()) {
        throw std::runtime_error("ParseJsonString 的字符串未闭合：预期结束引号 '\"'，字符串起始偏移: " +
                                 std::to_string(string_start) + "，" + JsonErrorContext(content, pos) + "。");
    }
    ++pos;
    return result;
}

std::string JsonVocabTokenizor::Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

int JsonVocabTokenizor::ParseJsonInt(const std::string& content, std::size_t& pos) {
    while (pos < content.size() && !std::isdigit(static_cast<unsigned char>(content[pos])) && content[pos] != '-') {
        ++pos;
    }
    const std::size_t start = pos;
    if (pos >= content.size()) {
        throw std::runtime_error("ParseJsonInt 未找到整数：预期十进制整数，" +
                                 JsonErrorContext(content, pos) + "。");
    }
    if (content[pos] == '-') {
        ++pos;
    }
    while (pos < content.size() && std::isdigit(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    try {
        return std::stoi(content.substr(start, pos - start));
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("ParseJsonInt 的整数格式无效：预期可选负号后至少一位十进制数字，" +
                                 JsonErrorContext(content, start) + "。");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("ParseJsonInt 的整数超出 int 范围：有效范围为 [" +
                                 std::to_string(std::numeric_limits<int>::min()) + ", " +
                                 std::to_string(std::numeric_limits<int>::max()) + "]，" +
                                 JsonErrorContext(content, start) + "。");
    }
}

bool JsonVocabTokenizor::ParseJsonBool(const std::string& content, std::size_t& pos) {
    if (content.substr(pos, 4) == "true") {
        pos += 4;
        return true;
    }
    if (content.substr(pos, 5) == "false") {
        pos += 5;
        return false;
    }
    throw std::runtime_error("ParseJsonBool 的布尔值无效：预期 true 或 false，" +
                             JsonErrorContext(content, pos) + "。");
}

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
    if (!vocab_path.empty()) {
        std::ifstream file(vocab_path);
        if (file.good()) {
            try {
                return std::make_unique<JsonVocabTokenizer>(vocab_path);
            } catch (const std::exception& error) {
                std::cerr << "加载 JSON 词表失败，回退到 ASCII tokenizer："
                          << error.what() << std::endl;
            }
        }
    }
    return std::make_unique<AsciiTokenizer>();
}
