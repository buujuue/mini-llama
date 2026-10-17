// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 基础用例移植自 mini-llama-cpp/tests/gtest/test_tokenizer.cc。

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "tokenizer.h"

namespace {

static_assert(std::is_same_v<JsonVocabTokenizer, JsonVocabTokenizor>);

// 每个测试在当前测试工作目录创建独立输入文件，不依赖 models/ 或 references/。
class TokenizerFileTest : public testing::Test {
protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::current_path() /
            ("tokenizer_test_" + std::string(testing::UnitTest::GetInstance()
                                              ->current_test_info()->name()) +
             "_" + std::to_string(stamp));
        owns_directory_ = std::filesystem::create_directory(directory_);
        ASSERT_TRUE(owns_directory_);
    }

    void TearDown() override {
        if (owns_directory_) {
            std::error_code error;
            std::filesystem::remove_all(directory_, error);
            EXPECT_FALSE(error) << error.message();
        }
    }

    std::string FilePath(const std::string& name) const {
        return (directory_ / name).string();
    }

    std::string WriteFile(const std::string& name, const std::string& content) const {
        const std::string file_path = FilePath(name);
        std::ofstream file(file_path, std::ios::binary);
        file << content;
        file.close();
        if (!file) {
            throw std::runtime_error("无法写入测试文件: " + file_path);
        }
        return file_path;
    }

private:
    std::filesystem::path directory_;
    bool owns_directory_ = false;
};

class JsonVocabTokenizerTest : public TokenizerFileTest {
protected:
    void SetUp() override {
        TokenizerFileTest::SetUp();
        if (HasFatalFailure()) {
            return;
        }
        // 课程 tiny 词表，加上 world 的字符，使比较用例覆盖完整短句。
        vocab_path_ = WriteFile("vocab.json", R"([
            {"id": 0, "content": "<unk>", "special": true},
            {"id": 1, "content": "<bos>", "special": true},
            {"id": 2, "content": "<eos>", "special": true},
            {"id": 32, "content": " ", "special": false},
            {"id": 33, "content": "!", "special": false},
            {"id": 97, "content": "a", "special": false},
            {"id": 100, "content": "d", "special": false},
            {"id": 101, "content": "e", "special": false},
            {"id": 104, "content": "h", "special": false},
            {"id": 105, "content": "i", "special": false},
            {"id": 108, "content": "l", "special": false},
            {"id": 109, "content": "m", "special": false},
            {"id": 110, "content": "n", "special": false},
            {"id": 111, "content": "o", "special": false},
            {"id": 114, "content": "r", "special": false},
            {"id": 119, "content": "w", "special": false},
            {"id": 127, "content": "", "special": false}
        ])");
    }

    std::string vocab_path_;
};

class BpeTokenizerTest : public TokenizerFileTest {
protected:
    void SetUp() override {
        TokenizerFileTest::SetUp();
        if (HasFatalFailure()) {
            return;
        }
        vocab_path_ = WriteFile("vocab.json", R"({
            "h": 0, "e": 1, "l": 2, "o": 3,
            "he": 4, "hel": 5, "hell": 6, "hello": 7,
            "<|im_start|>": 8, "<unk>": 9, "<bos>": 10, "<eos>": 11,
            "\u0120": 12, "\u010a": 13, "\u0109": 14, "\u0100": 15,
            "\u00e4": 16, "\u00bd": 17, "\u0142": 18,
            "\u00e5": 19, "\u00a5": 20
        })");
        merges_path_ = WriteFile("merges.txt", "h e\nhe l\nhel l\nhell o\n");
        special_path_ = WriteFile("special_tokens.json",
            R"({"bos_id": 10, "eos_id": 11, "unk_id": 9})");
    }

    std::string vocab_path_;
    std::string merges_path_;
    std::string special_path_;
};

TEST(AsciiTokenizerTest, EncodesBasicTextWithBos) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(tokenizer.Encode("hi"), (std::vector<int>{1, 104, 105}));
}

TEST(AsciiTokenizerTest, EmptyTextStillProducesBos) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(tokenizer.Encode(""), (std::vector<int>{1}));
}

TEST(AsciiTokenizerTest, DecodesSpecialPrintableAndControlTokens) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(tokenizer.DecodeToken(0), "<unk>");
    EXPECT_EQ(tokenizer.DecodeToken(1), "<bos>");
    EXPECT_EQ(tokenizer.DecodeToken(2), "<eos>");
    EXPECT_EQ(tokenizer.DecodeToken(97), "a");
    EXPECT_EQ(tokenizer.DecodeToken(32), " ");
    EXPECT_EQ(tokenizer.DecodeToken(10), "");
    EXPECT_EQ(tokenizer.DecodeToken(127), "");
    EXPECT_EQ(tokenizer.DecodeToken(128), "<unk>");
    EXPECT_EQ(tokenizer.DecodeToken(std::numeric_limits<int>::max()), "<unk>");
}

TEST(AsciiTokenizerTest, DecodesSequenceAndEmptyInput) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(tokenizer.Decode({1, 104, 105, 2}), "<bos>hi<eos>");
    EXPECT_EQ(tokenizer.Decode({}), "");
}

TEST(AsciiTokenizerTest, ReportsVocabularyAndSpecialIds) {
    const AsciiTokenizer tokenizer;
    EXPECT_EQ(tokenizer.vocab_size(), 128);
    EXPECT_EQ(tokenizer.bos_id(), 1);
    EXPECT_EQ(tokenizer.eos_id(), 2);
    EXPECT_EQ(tokenizer.unk_id(), 0);
}

TEST(AsciiTokenizerTest, AllHighBytesBecomeUnkInsteadOfNegativeIds) {
    const AsciiTokenizer tokenizer;
    std::string input;
    for (int byte = 128; byte <= 255; ++byte) {
        input += static_cast<char>(byte);
    }
    std::vector<int> expected(129, 0);
    expected[0] = 1;
    EXPECT_EQ(tokenizer.Encode(input), expected);
    EXPECT_EQ(tokenizer.Encode(u8"你好"), (std::vector<int>{1, 0, 0, 0, 0, 0, 0}));
}

TEST_F(JsonVocabTokenizerTest, LoadsSparseVocabularyAndSpecialIds) {
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.vocab_size(), 128);
    EXPECT_EQ(tokenizer.bos_id(), 1);
    EXPECT_EQ(tokenizer.eos_id(), 2);
    EXPECT_EQ(tokenizer.unk_id(), 0);
}

TEST_F(JsonVocabTokenizerTest, EncodesBasicTextWithBos) {
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.Encode("hi"), (std::vector<int>{1, 104, 105}));
}

TEST_F(JsonVocabTokenizerTest, DecodesSpecialTokensSparseSlotsAndInvalidIds) {
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.DecodeToken(0), "<unk>");
    EXPECT_EQ(tokenizer.DecodeToken(1), "<bos>");
    EXPECT_EQ(tokenizer.DecodeToken(2), "<eos>");
    EXPECT_EQ(tokenizer.DecodeToken(97), "a");
    EXPECT_EQ(tokenizer.DecodeToken(32), " ");
    EXPECT_EQ(tokenizer.DecodeToken(10), "");
    EXPECT_EQ(tokenizer.DecodeToken(127), "");
    EXPECT_EQ(tokenizer.DecodeToken(-1), "<unk>");
    EXPECT_EQ(tokenizer.DecodeToken(128), "<unk>");
    EXPECT_EQ(tokenizer.DecodeToken(std::numeric_limits<int>::max()), "<unk>");
}

TEST_F(JsonVocabTokenizerTest, DecodesSequenceAndEmptyInput) {
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.Decode({1, 104, 105, 2}), "<bos>hi<eos>");
    EXPECT_EQ(tokenizer.Decode({}), "");
}

TEST_F(JsonVocabTokenizerTest, MatchesAsciiForKnownText) {
    const AsciiTokenizer ascii;
    const JsonVocabTokenizer tokenizer(vocab_path_);
    const std::string prompt = "hello world!";
    EXPECT_EQ(tokenizer.Encode(prompt), ascii.Encode(prompt));
    EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(prompt)), "<bos>" + prompt);
}

TEST_F(JsonVocabTokenizerTest, MissingFileReportsPathAndReason) {
    const std::string missing = FilePath("missing.json");
    try {
        const JsonVocabTokenizer tokenizer(missing);
        FAIL() << "缺失词表应抛出异常";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(missing), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("无法打开词表文件"), std::string::npos);
    }
}

TEST_F(JsonVocabTokenizerTest, EmptyTextProducesOnlyBos) {
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.Encode(""), (std::vector<int>{1}));
}

TEST_F(JsonVocabTokenizerTest, UsesLongestMatchSkipsEmptyEntriesAndReadsCustomIds) {
    WriteFile("vocab.json", R"([
        {"id": 8, "content": "<bos>", "special": true},
        {"id": 9, "content": "<eos>", "special": true},
        {"id": 7, "content": "<unk>", "special": true},
        {"id": 0, "content": ""}, {"id": 1, "content": "h"},
        {"id": 2, "content": "he"}, {"id": 3, "content": "hello"}
    ])");
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.vocab_size(), 10);
    EXPECT_EQ(tokenizer.bos_id(), 8);
    EXPECT_EQ(tokenizer.eos_id(), 9);
    EXPECT_EQ(tokenizer.unk_id(), 7);
    EXPECT_EQ(tokenizer.Encode("helloheh?"), (std::vector<int>{8, 3, 2, 1, 7}));
    EXPECT_EQ(tokenizer.Decode({8, 3, 9}), "<bos>hello<eos>");
}

TEST_F(JsonVocabTokenizerTest, UnknownUtf8TextFallsBackOneByteAtATime) {
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.Encode(u8"h你?"), (std::vector<int>{1, 104, 0, 0, 0, 0}));
}

TEST_F(JsonVocabTokenizerTest, MatchesUtf8VocabularyEntries) {
    WriteFile("vocab.json", u8R"([
        {"id": 0, "content": "<unk>"}, {"id": 1, "content": "<bos>"},
        {"id": 3, "content": "你"}, {"id": 4, "content": "你好"}
    ])");
    const JsonVocabTokenizer tokenizer(vocab_path_);
    EXPECT_EQ(tokenizer.Encode(u8"你好你"), (std::vector<int>{1, 4, 3}));
    EXPECT_EQ(tokenizer.Decode({4, 3}), u8"你好你");
}

TEST_F(JsonVocabTokenizerTest, ParsesEscapedAsciiAndIgnoresUnknownFields) {
    WriteFile("vocab.json", R"([
        {"id": 0, "content": "<unk>"}, {"id": 1, "content": "<bos>"},
        {"id": 4, "content": "\u0041\n\t\r\\\"", "note": "ignored", "score": 42}
    ])");
    const JsonVocabTokenizer tokenizer(vocab_path_);
    const std::string text = "A\n\t\r\\\"";
    EXPECT_EQ(tokenizer.Encode(text), (std::vector<int>{1, 4}));
    EXPECT_EQ(tokenizer.DecodeToken(4), text);
}

TEST_F(JsonVocabTokenizerTest, RejectsInvalidEntriesWithFileAndOffsetContext) {
    const std::vector<std::string> invalid = {
        "", "{}", "[]", R"([{"content":"x"}])", R"([{"id":0}])",
        R"([{"id":-1,"content":"x"}])",
        R"([{"id":2147483647,"content":"x"}])",
        R"([{"id":2147483648,"content":"x"}])",
        R"([{"id":0,"content":"x","special":null}])",
        R"([{"id":0,"content":"\uZZZZ"}])",
        R"([{"id":0,"content":"unterminated])"
    };
    for (const auto& content : invalid) {
        SCOPED_TRACE(content);
        WriteFile("vocab.json", content);
        try {
            const JsonVocabTokenizer tokenizer(vocab_path_);
            FAIL() << "无效词表应抛出异常";
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(vocab_path_), std::string::npos);
            EXPECT_NE(message.find("字节偏移"), std::string::npos);
        }
    }
}

TEST_F(JsonVocabTokenizerTest, FactoryUsesJsonWhenAvailable) {
    const auto tokenizer = CreateTokenizer(vocab_path_);
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_NE(dynamic_cast<JsonVocabTokenizer*>(tokenizer.get()), nullptr);
    EXPECT_EQ(tokenizer->Encode("hi"), (std::vector<int>{1, 104, 105}));
}

TEST_F(JsonVocabTokenizerTest, FactoryFallsBackForEmptyMissingAndInvalidVocab) {
    const std::string invalid = WriteFile("invalid.json", "[]");
    for (const std::string& file : {std::string{}, FilePath("missing.json"), invalid}) {
        SCOPED_TRACE(file);
        const auto tokenizer = CreateTokenizer(file);
        ASSERT_NE(tokenizer, nullptr);
        EXPECT_NE(dynamic_cast<AsciiTokenizer*>(tokenizer.get()), nullptr);
        EXPECT_EQ(tokenizer->vocab_size(), 128);
        EXPECT_EQ(tokenizer->Encode("hi"), (std::vector<int>{1, 104, 105}));
    }
}

TEST_F(BpeTokenizerTest, FactoryEncodesDecodesAndPreservesSpecialToken) {
    const auto tokenizer = CreateBpeTokenizer(vocab_path_, merges_path_, special_path_);
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->bos_id(), 10);
    EXPECT_EQ(tokenizer->eos_id(), 11);
    EXPECT_EQ(tokenizer->unk_id(), 9);
    EXPECT_EQ(tokenizer->vocab_size(), 21);
    EXPECT_EQ(tokenizer->Encode("hello"), (std::vector<int>{7}));
    EXPECT_EQ(tokenizer->Decode({7}), "hello");
    EXPECT_EQ(tokenizer->Encode("<|im_start|>"), (std::vector<int>{8}));
    EXPECT_EQ(tokenizer->Decode({8}), "<|im_start|>");
}

TEST_F(BpeTokenizerTest, EmptyInputDoesNotAddBosOrEos) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    EXPECT_TRUE(tokenizer.Encode("").empty());
    EXPECT_EQ(tokenizer.Decode({}), "");
    EXPECT_EQ(tokenizer.Encode("h"), (std::vector<int>{0}));
}

TEST_F(BpeTokenizerTest, AlternatesPlainTextAndAdjacentSpecialTokens) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    const std::string text = "hello<|im_start|><|im_start|>hello";
    EXPECT_EQ(tokenizer.Encode(text), (std::vector<int>{7, 8, 8, 7}));
    EXPECT_EQ(tokenizer.Decode({7, 8, 8, 7}), text);
}

TEST_F(BpeTokenizerTest, SpecialTokensUseLongestMatch) {
    WriteFile("vocab.json", R"({"<|a|>":0,"<|a|>b|>":1,"<unk>":9})");
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    EXPECT_EQ(tokenizer.Encode("<|a|>b|><|a|>"), (std::vector<int>{1, 0}));
    EXPECT_EQ(tokenizer.Decode({1, 0}), "<|a|>b|><|a|>");
}

TEST_F(BpeTokenizerTest, MergeRankTakesPriorityOverLeftmostPair) {
    WriteFile("vocab.json", R"({"a":0,"b":1,"c":2,"ab":3,"bc":4})");
    WriteFile("merges.txt", "b c\na b\n");
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, ""));
    EXPECT_EQ(tokenizer.Encode("abc"), (std::vector<int>{0, 4}));
    EXPECT_EQ(tokenizer.Decode({0, 4}), "abc");
}

TEST_F(BpeTokenizerTest, EqualRankPairsMergeFromTheLeft) {
    WriteFile("vocab.json", R"({"a":0,"aa":1})");
    WriteFile("merges.txt", "a a\n");
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, ""));
    EXPECT_EQ(tokenizer.Encode("aaa"), (std::vector<int>{1, 0}));
}

TEST_F(BpeTokenizerTest, EmptyMergesLeaveSeparateBytesAndUseUnk) {
    WriteFile("merges.txt", "");
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    EXPECT_EQ(tokenizer.Encode("hello?"), (std::vector<int>{0, 1, 2, 2, 3, 9}));
    EXPECT_EQ(tokenizer.Decode({0, 1, 2, 2, 3, 9}), "hello<unk>");
}

TEST_F(BpeTokenizerTest, RestoresSpacesControlsAndEmbeddedNullBytes) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    const std::string text(" hello\n\t\0", 9);
    EXPECT_EQ(tokenizer.Encode(text), (std::vector<int>{12, 7, 13, 14, 15}));
    EXPECT_EQ(tokenizer.Decode({12, 7, 13, 14, 15}), text);
    // 单 token 接口保留课程实例的原始词表符号语义。
    EXPECT_EQ(tokenizer.DecodeToken(12), u8"Ġ");
    EXPECT_EQ(tokenizer.Decode({12}), " ");
}

TEST_F(BpeTokenizerTest, ChineseUtf8BytesRoundTripAcrossTokenBoundaries) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    const std::vector<int> expected{16, 17, 18, 19, 20, 17};
    EXPECT_EQ(tokenizer.Encode(u8"你好"), expected);
    EXPECT_EQ(tokenizer.Decode(expected), u8"你好");
}

TEST_F(BpeTokenizerTest, InvalidTokenIdsAndUnusedSlotsDecodeToEmpty) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    EXPECT_EQ(tokenizer.DecodeToken(-1), "");
    EXPECT_EQ(tokenizer.DecodeToken(21), "");
    EXPECT_EQ(tokenizer.DecodeToken(std::numeric_limits<int>::max()), "");
    EXPECT_EQ(tokenizer.Decode({-1, 7, 21}), "hello");
    WriteFile("vocab.json", R"({"h":0,"o":4})");
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, ""));
    EXPECT_EQ(tokenizer.vocab_size(), 5);
    EXPECT_EQ(tokenizer.DecodeToken(2), "");
}

TEST_F(BpeTokenizerTest, MissingSpecialConfigKeepsIdsUnset) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, FilePath("missing.json")));
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), -1);
    EXPECT_EQ(tokenizer.unk_id(), -1);
    EXPECT_EQ(tokenizer.Encode("?"), (std::vector<int>{-1}));
}

TEST_F(BpeTokenizerTest, ReloadClearsOldVocabularyMergesSpecialsAndIds) {
    BpeTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    WriteFile("vocab.json", R"({"h":0,"e":1,"l":2,"o":3})");
    WriteFile("merges.txt", "");
    ASSERT_TRUE(tokenizer.Load(vocab_path_, merges_path_, ""));
    EXPECT_EQ(tokenizer.vocab_size(), 4);
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), -1);
    EXPECT_EQ(tokenizer.unk_id(), -1);
    EXPECT_EQ(tokenizer.Encode("hello"), (std::vector<int>{0, 1, 2, 2, 3}));
    const auto special = tokenizer.Encode("<|im_start|>");
    EXPECT_EQ(special, std::vector<int>(12, -1));
    EXPECT_EQ(tokenizer.DecodeToken(7), "");
}

TEST_F(BpeTokenizerTest, LoadAndFactoryRejectMissingRequiredFiles) {
    BpeTokenizer tokenizer;
    const std::string missing = FilePath("missing");
    EXPECT_FALSE(tokenizer.Load(missing, merges_path_, special_path_));
    EXPECT_FALSE(tokenizer.Load(vocab_path_, missing, special_path_));
    EXPECT_EQ(CreateBpeTokenizer(missing, merges_path_, special_path_), nullptr);
    EXPECT_EQ(CreateBpeTokenizer(vocab_path_, missing, special_path_), nullptr);
}

TEST_F(BpeTokenizerTest, RejectsEmptyMalformedAndOutOfRangeVocabularies) {
    for (const std::string& content : {
             std::string{}, std::string{"{}"}, std::string{"not json"},
             std::string{R"({"h":-1})"}, std::string{R"({"h":2147483647})"},
             std::string{R"({"h":2147483648})"},
             std::string{R"({"unterminated:0})"}}) {
        SCOPED_TRACE(content);
        WriteFile("vocab.json", content);
        BpeTokenizer tokenizer;
        EXPECT_FALSE(tokenizer.Load(vocab_path_, merges_path_, special_path_));
    }
}

} // namespace
