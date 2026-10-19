#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <map>
#include <memory>
#include <vector>

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int> Encode(const std::string& text) const = 0;

    virtual std::string DecodeToken(int token) const = 0;

    virtual std::string Decode(const std::vector<int>& tokens) const = 0;

    virtual int vocab_size() const = 0;
    virtual int bos_id() const = 0;
    virtual int eos_id() const = 0;
    virtual int unk_id() const = 0;
};

// ASCII 和 JSON 词表编码自动添加 BOS，不自动添加 EOS。
class AsciiTokenizer : public ITokenizer {
public:
    static constexpr int VocabSize = 128;
    static constexpr int BosId = 1;
    static constexpr int EosId = 2;
    static constexpr int UnkId = 0;

    AsciiTokenizer() = default;

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;

    int vocab_size() const override {
        return VocabSize;
    }
    int bos_id() const override {
        return BosId;
    }
    int eos_id() const override {
        return EosId;
    }
    int unk_id() const override {
        return UnkId;
    }
};

class JsonVocabTokenizor : public ITokenizer{
public:
    explicit JsonVocabTokenizor(const std::string& vocab_path);

    std::vector<int> Encode(const std::string& text) const override;
    std::string DecodeToken(int token) const override;
    std::string Decode(const std::vector<int>& tokens) const override;
    
    int vocab_size() const override {
        return vocab_size_;
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
    std::string ParseJsonString(const std::string& content, std::size_t& pos);
    std::string Trim(const std::string& s);
    int ParseJsonInt(const std::string& content, std::size_t& pos);
    bool ParseJsonBool(const std::string& content, std::size_t& pos);

    struct VocabEntry{
        int id = 0;
        std::string content;
        bool special = false;
    };

    std::vector<VocabEntry> id_to_content_;
    std::vector<VocabEntry> content_to_id_;

    int vocab_size_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;
    int unk_id_ = 0;
};

// 保留现有 Tokenizor 拼写，同时兼容课程实例中的 Tokenizer 名称。
using JsonVocabTokenizer = JsonVocabTokenizor;

// 字节级 BPE 不自动添加 BOS/EOS；DecodeToken 返回词表中的映射符号，
// Decode 才将整段映射符号还原为原始字节（包括 UTF-8 文本）。
class BpeTokenizer : public ITokenizer {
public:
    BpeTokenizer();
    ~BpeTokenizer() override = default;

    bool Load(const std::string& vocab_path, const std::string& merges_path, const std::string& special_path);

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
    struct MergePair {
        std::string left;
        std::string right;

        bool operator<(const MergePair& other) const {
            return left < other.left || (left == other.left && right < other.right);
        }
    };

    struct SpecialToken {
        std::string content;
        int id;
    };

    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::map<MergePair, int> merge_ranks_;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;

    std::vector<SpecialToken> special_tokens_;

    std::vector<std::string> B2U_;                  
    std::unordered_map<std::string, uint8_t> U2B_;  

    std::string CodePointToUtf8(char32_t cp);
    
    std::string NextUtf8CodePoint(const std::string& s, size_t& pos) const;
    std::string UnicodeToBytes(const std::string& text) const;
    bool ParseJsonString(const std::string& content, std::size_t& pos, std::string& out);
    bool ParseJsonInt(const std::string& content, std::size_t& pos, int& out);
};


std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);


std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                           const std::string& merges_path,
                                           const std::string& special_path);
