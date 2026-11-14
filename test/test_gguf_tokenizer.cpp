// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 基础场景及微型词表适配自 mini-llama-cpp/tests/gtest/test_gguf_tokenizer.cc
// 和 tests/gguf_tokenizer_fixture.h；补充异常输入与字节级 BPE 验证。

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "gguf_fixture.h"
#include "gguf_tokenizer.h"
#include "gguf_tokenizer.h"  // 公开头文件可重复包含。

namespace {

using namespace mini_llama::test;

std::vector<MetadataEntry> TinyTokenizerMetadata() {
    return {
        {"tokenizer.ggml.model", GgufTypeString, StringValue("gpt2")},
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({
            "a", "b", "ab", "<s>", "</s>", "<|im_start|>", "<|im_end|>",
            "u", "s", "e", "r", "system", "assistant", "\n"})},
        {"tokenizer.ggml.token_type", GgufTypeArray,
         IntArrayValue({1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1})},
        {"tokenizer.ggml.merges", GgufTypeArray, StringArrayValue({"a b"})},
        {"tokenizer.ggml.bos_token_id", GgufTypeUint32, IntegerValue(3)},
        {"tokenizer.ggml.eos_token_id", GgufTypeUint32, IntegerValue(4)},
    };
}

class GgufTokenizerTest : public GgufFileTest {};

TEST_F(GgufTokenizerTest, LoadsFromTinyFixture) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf(TinyTokenizerMetadata(), {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->vocab_size(), 14);
    EXPECT_EQ(tokenizer->bos_id(), 3);
    EXPECT_EQ(tokenizer->eos_id(), 4);
    EXPECT_EQ(tokenizer->unk_id(), -1);
}

TEST_F(GgufTokenizerTest, EncodeDecodeRoundtrip) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf(TinyTokenizerMetadata(), {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("ab"), (std::vector<int>{2}));
    EXPECT_EQ(tokenizer->Encode("abab"), (std::vector<int>{2, 2}));
    EXPECT_EQ(tokenizer->Decode({2, 2}), "abab");
    EXPECT_TRUE(tokenizer->Encode("").empty());
}

TEST_F(GgufTokenizerTest, SpecialTokens) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf(TinyTokenizerMetadata(), {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("<|im_start|>"), (std::vector<int>{5}));
    EXPECT_EQ(tokenizer->Encode("<|im_end|>"), (std::vector<int>{6}));
    EXPECT_EQ(tokenizer->Encode("a<|im_start|>ab<|im_end|><s></s>"),
              (std::vector<int>{0, 5, 2, 6, 3, 4}));
    EXPECT_EQ(tokenizer->DecodeToken(4), "</s>");
    EXPECT_EQ(tokenizer->Decode({0, 5, 2, 6, 3, 4}), "a<|im_start|>ab<|im_end|><s></s>");
}

TEST_F(GgufTokenizerTest, SpecialTokensUseTypesAndLongestMatch) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray,
         StringArrayValue({"<custom>", "<custom>more", "<|ordinary|>"})},
        {"tokenizer.ggml.token_type", GgufTypeArray, IntArrayValue({3, 3, 1})},
    }, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("<custom>more<custom>"), (std::vector<int>{1, 0}));
    EXPECT_EQ(tokenizer->Encode("<|ordinary|>"), std::vector<int>(12, -1));
}

TEST_F(GgufTokenizerTest, OptionalMetadataMayBeAbsent) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"a", "b", "ab"})},
    }, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->bos_id(), -1);
    EXPECT_EQ(tokenizer->eos_id(), -1);
    EXPECT_EQ(tokenizer->unk_id(), -1);
    EXPECT_EQ(tokenizer->Encode("ab!"), (std::vector<int>{0, 1, -1}));
}

TEST_F(GgufTokenizerTest, MismatchedTokenTypesAreIgnored) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"a", "<s>"})},
        {"tokenizer.ggml.token_type", GgufTypeArray, IntArrayValue({3})},
    }, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("<s>"), (std::vector<int>{-1, -1, -1}));
}

TEST_F(GgufTokenizerTest, UsesUnknownTokenId) {
    auto metadata = TinyTokenizerMetadata();
    metadata[1].value = StringArrayValue({"a", "b", "ab", "<unk>"});
    metadata[2].value = IntArrayValue({1, 1, 1, 2});
    metadata[4].value = IntegerValue(0);
    metadata[5].value = IntegerValue(1);
    metadata.push_back({"tokenizer.ggml.unknown_token_id", GgufTypeInt32, IntegerValue(3)});
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf(metadata, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->unk_id(), 3);
    EXPECT_EQ(tokenizer->Encode("ab!?"), (std::vector<int>{2, 3, 3}));
}

TEST_F(GgufTokenizerTest, DecodesUtf8WhitespaceAndBinaryBytes) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray,
         StringArrayValue({"Ġ", "Ċ", "ĉ", "Ā", "ä", "½", "ł", "å", "¥"})},
    }, {})));
    ASSERT_NE(tokenizer, nullptr);
    const std::string text = std::string(" \n\t\0", 4) + "你好";
    const std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 5};
    EXPECT_EQ(tokenizer->Encode(text), expected);
    EXPECT_EQ(tokenizer->Decode(expected), text);
    EXPECT_EQ(tokenizer->DecodeToken(0), "Ġ");
    EXPECT_EQ(tokenizer->Decode({-1, 0, 999}), " ");
    EXPECT_TRUE(tokenizer->DecodeToken(-1).empty());
    EXPECT_TRUE(tokenizer->DecodeToken(9).empty());
    EXPECT_TRUE(tokenizer->Decode({}).empty());
}

TEST_F(GgufTokenizerTest, MergeRankOverridesLeftToRightOrder) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"a", "b", "c", "ab", "bc"})},
        {"tokenizer.ggml.merges", GgufTypeArray, StringArrayValue({"b c", "a b"})},
    }, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("abc"), (std::vector<int>{0, 4}));
}

TEST_F(GgufTokenizerTest, EqualMergeRanksChooseLeftmostPair) {
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"a", "aa"})},
        {"tokenizer.ggml.merges", GgufTypeArray, StringArrayValue({"a a"})},
    }, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("aaa"), (std::vector<int>{1, 0}));
}

TEST_F(GgufTokenizerTest, MalformedMergeLinesAreSkipped) {
    auto metadata = TinyTokenizerMetadata();
    metadata[3].value = StringArrayValue({"", "ab", " a", "b ", "a b"});
    const auto tokenizer = CreateGgufTokenizer(Write(MakeGguf(metadata, {})));
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->Encode("ab"), (std::vector<int>{2}));
}

TEST_F(GgufTokenizerTest, UnloadedInstanceIsEmpty) {
    const GgufTokenizer tokenizer;
    EXPECT_EQ(tokenizer.vocab_size(), 0);
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), -1);
    EXPECT_EQ(tokenizer.unk_id(), -1);
    EXPECT_TRUE(tokenizer.Encode("ab").empty());
    EXPECT_TRUE(tokenizer.Decode({0}).empty());
}

TEST_F(GgufTokenizerTest, ReloadReplacesVocabMergesAndSpecialIds) {
    GgufTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.Load(Write(MakeGguf(TinyTokenizerMetadata(), {}))));
    ASSERT_TRUE(tokenizer.Load(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"b", "a"})},
    }, {}))));
    EXPECT_EQ(tokenizer.vocab_size(), 2);
    EXPECT_EQ(tokenizer.Encode("ab"), (std::vector<int>{1, 0}));
    EXPECT_EQ(tokenizer.Encode("<s>"), (std::vector<int>{-1, -1, -1}));
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), -1);
    EXPECT_EQ(tokenizer.unk_id(), -1);
}

TEST_F(GgufTokenizerTest, FailedReloadClearsOldStateAndCanRecover) {
    GgufTokenizer tokenizer;
    const std::string path = Write(MakeGguf(TinyTokenizerMetadata(), {}));
    ASSERT_TRUE(tokenizer.Load(path));
    EXPECT_FALSE(tokenizer.Load(missing_path_.string()));
    EXPECT_EQ(tokenizer.vocab_size(), 0);
    EXPECT_EQ(tokenizer.bos_id(), -1);
    EXPECT_EQ(tokenizer.eos_id(), -1);
    EXPECT_TRUE(tokenizer.Encode("ab").empty());
    EXPECT_TRUE(tokenizer.Decode({2}).empty());
    ASSERT_TRUE(tokenizer.Load(path));
    auto metadata = TinyTokenizerMetadata();
    metadata[1].value = StringArrayValue({"a", "a"});
    EXPECT_FALSE(tokenizer.Load(Write(MakeGguf(metadata, {}))));
    EXPECT_EQ(tokenizer.vocab_size(), 0);
    ASSERT_TRUE(tokenizer.Load(Write(MakeGguf(TinyTokenizerMetadata(), {}))));
    EXPECT_EQ(tokenizer.Encode("ab"), (std::vector<int>{2}));
}

TEST_F(GgufTokenizerTest, RejectsMissingFile) {
    EXPECT_EQ(CreateGgufTokenizer(missing_path_.string()), nullptr);
}

TEST_F(GgufTokenizerTest, RejectsMissingEmptyOrDuplicateVocab) {
    EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf({}, {}))), nullptr);
    for (const auto& tokens : std::vector<std::vector<std::string>>{
             {}, {""}, {"a", "a"}}) {
        EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf({
            {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue(tokens)},
        }, {}))), nullptr);
    }
    EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf({
        {"tokenizer.ggml.tokens", GgufTypeString, StringValue("a")},
    }, {}))), nullptr);
}

TEST_F(GgufTokenizerTest, RejectsUnsupportedTokenizerModel) {
    auto metadata = TinyTokenizerMetadata();
    metadata[0].value = StringValue("llama");
    EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf(metadata, {}))), nullptr);
}

TEST_F(GgufTokenizerTest, RejectsOutOfRangeSpecialIds) {
    for (const char* key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                            "tokenizer.ggml.unknown_token_id"}) {
        for (int64_t value : {-2LL, 1LL, 4294967296LL}) {
            SCOPED_TRACE(key);
            SCOPED_TRACE(value);
            EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf({
                {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"a"})},
                {key, GgufTypeInt64, IntegerValue(static_cast<uint64_t>(value), 8)},
            }, {}))), nullptr);
        }
    }
}

TEST_F(GgufTokenizerTest, ReadsV2AndV3MetadataWithoutTensorPayload) {
    Bytes data = MakeGguf(TinyTokenizerMetadata(), {{"unused.weight", {1}, 999}});
    for (uint32_t version : {2u, 3u}) {
        WriteLe(data, 4, version, 4);
        const auto tokenizer = CreateGgufTokenizer(Write(data));
        ASSERT_NE(tokenizer, nullptr);
        EXPECT_EQ(tokenizer->Encode("ab"), (std::vector<int>{2}));
    }
}

TEST_F(GgufTokenizerTest, RejectsMalformedMetadata) {
    Bytes data = MakeGguf(TinyTokenizerMetadata(), {});
    data[0] = '!';
    EXPECT_EQ(CreateGgufTokenizer(Write(data)), nullptr);
    data[0] = 'G';
    WriteLe(data, 4, 99, 4);
    EXPECT_EQ(CreateGgufTokenizer(Write(data)), nullptr);
    EXPECT_EQ(CreateGgufTokenizer(Write({'G', 'G', 'U', 'F'})), nullptr);

    auto metadata = TinyTokenizerMetadata();
    metadata.push_back(metadata[1]);
    EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf(metadata, {}))), nullptr);
    metadata = TinyTokenizerMetadata();
    metadata.push_back({"truncated", GgufTypeString, IntegerValue(100, 8)});
    EXPECT_EQ(CreateGgufTokenizer(Write(MakeGguf(metadata, {}))), nullptr);
}

TEST(GgufTokenizerLocalModelTest, LoadsRealModelWhenAvailable) {
    const auto path = std::filesystem::path(MINI_LLAMA_SOURCE_DIR) /
        "references/mini-llama-cpp/models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "local demo GGUF is not available";
    const auto tokenizer = CreateGgufTokenizer(path.string());
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->vocab_size(), 151936);
    EXPECT_EQ(tokenizer->bos_id(), 151643);
    EXPECT_EQ(tokenizer->eos_id(), 151645);
    EXPECT_EQ(tokenizer->Encode("<|im_start|>"), (std::vector<int>{151644}));
    const std::string text = "<|im_start|>user\nHello, 你好！<|im_end|>\n";
    EXPECT_EQ(tokenizer->Decode(tokenizer->Encode(text)), text);
}

TEST(GgufTokenizerLocalModelTest, MatchesExportedBpeWhenAvailable) {
    const std::filesystem::path root(MINI_LLAMA_SOURCE_DIR);
    const auto path = root / "references/mini-llama-cpp/models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
    const auto vocab = root / "models/chat/vocab.json";
    const auto merges = root / "models/chat/merges.txt";
    if (!std::filesystem::exists(path) || !std::filesystem::exists(vocab) ||
        !std::filesystem::exists(merges)) GTEST_SKIP() << "local GGUF/BPE exports are not available";
    const auto gguf = CreateGgufTokenizer(path.string());
    const auto bpe = CreateBpeTokenizer(vocab.string(), merges.string(),
                                      (root / "models/chat/special_tokens.json").string());
    ASSERT_NE(gguf, nullptr);
    ASSERT_NE(bpe, nullptr);
    for (const std::string text : {"Hello", "Hello world!", "你好", " \n\t",
                                  "<|im_start|>user\nHello<|im_end|>\n",
                                  "The quick brown fox jumps over the lazy dog."}) {
        SCOPED_TRACE(text);
        EXPECT_EQ(gguf->Encode(text), bpe->Encode(text));
        EXPECT_EQ(gguf->Decode(gguf->Encode(text)), text);
    }
}

}  // namespace
