// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 加载失败场景适配自 mini-llama-cpp/tests/gtest/test_gguf_loader.cc。

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "gguf_fixture.h"
#include "gguf_loader.h"

namespace {

using namespace mini_llama::test;

TEST(GgufTensorNameTest, GlobalWeights) {
    EXPECT_EQ(MapTensorName("token_embd.weight"), "token_embedding");
    EXPECT_EQ(MapTensorName("output_norm.weight"), "final_norm");
    EXPECT_EQ(MapTensorName("output.weight"), "lm_head");
}

TEST(GgufTensorNameTest, LayerWeights) {
    const std::vector<std::pair<std::string, std::string>> names = {
        {"attn_norm", "attention_norm"}, {"attn_q", "wq"},
        {"attn_k", "wk"}, {"attn_v", "wv"}, {"attn_output", "wo"},
        {"ffn_norm", "ffn_norm"}, {"ffn_gate", "w_gate"},
        {"ffn_up", "w_up"}, {"ffn_down", "w_down"},
    };
    for (const std::string layer : {"0", "1", "23"}) {
        for (const auto& name : names) {
            const std::string input = "blk." + layer + "." + name.first + ".weight";
            SCOPED_TRACE(input);
            EXPECT_EQ(MapTensorName(input), "layers." + layer + "." + name.second);
        }
    }
}

TEST(GgufTensorNameTest, AttentionBiases) {
    EXPECT_EQ(MapTensorName("blk.0.attn_q.bias"), "layers.0.bq");
    EXPECT_EQ(MapTensorName("blk.1.attn_k.bias"), "layers.1.bk");
    EXPECT_EQ(MapTensorName("blk.23.attn_v.bias"), "layers.23.bv");
}

TEST(GgufTensorNameTest, UnknownOrIncompleteNamesIgnored) {
    for (const char* name : {"", "unknown.weight", "token_embd.bias", "blk.",
                             "blk.0", "blk.0.attn_q", "blk.0.unknown.weight",
                             "blk.0.attn_output.bias", "blk.0.ffn_gate.bias",
                             "blk.0.attn_q.scales", "blk.0.attn_q.weight.extra"}) {
        SCOPED_TRACE(name);
        EXPECT_TRUE(MapTensorName(name).empty());
    }
}

class GgufLoaderTest : public GgufFileTest {};

TEST_F(GgufLoaderTest, MissingFile) {
    const Model model = LoadGgufModel(missing_path_.string());
    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error.find("GGUF parse error: "), 0u);
    EXPECT_NE(model.load_error.find(missing_path_.string()), std::string::npos);
    EXPECT_TRUE(model.layers.empty());
    EXPECT_TRUE(model.token_embedding.data.empty());
}

TEST_F(GgufLoaderTest, InvalidFile) {
    const Model model = LoadGgufModel(Write({'{', '}', '\n', ' '}));
    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error.find("GGUF parse error: "), 0u);
    EXPECT_NE(model.load_error.find("魔数不匹配"), std::string::npos);
}

TEST_F(GgufLoaderTest, ConfigFailureReturned) {
    // 上游的 RejectsIncompleteConfig 场景：先证明解析成功，再检查配置异常被返回。
    // 类型化读取已接入；此夹具缺少模型维度，配置完整性检查应拒绝加载。
    const std::string path = Write(TinyGguf());
    GgufReader reader;
    ASSERT_TRUE(reader.load(path)) << reader.error;
    const Model model = LoadGgufModel(path);
    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error.find("Failed to build config: "), 0u);
    EXPECT_GT(model.load_error.size(), std::string("Failed to build config: ").size());
    EXPECT_TRUE(model.layers.empty());
    EXPECT_TRUE(model.token_embedding.data.empty());
}

TEST_F(GgufLoaderTest, TensorBoundsErrorReturned) {
    const Model model = LoadGgufModel(Write(MakeGguf({}, {{"weight", {32}}})));
    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error,
              "GGUF parse error: tensor data outside file bounds: weight");
}

}  // namespace
