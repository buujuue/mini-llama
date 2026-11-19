// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 加载与前向场景适配自 mini-llama-cpp/tests/gtest/test_gguf_loader.cc。

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gguf_fixture.h"
#include "gguf_loader.h"
#include "forward.h"

namespace {

using namespace mini_llama::test;

std::vector<MetadataEntry> TinyModelMetadata(const std::string& arch = "llama") {
    const std::string prefix = arch + ".";
    return {
        {"general.architecture", GgufTypeString, StringValue(arch)},
        {prefix + "block_count", GgufTypeUint32, IntegerValue(2)},
        {prefix + "embedding_length", GgufTypeUint32, IntegerValue(32)},
        {prefix + "feed_forward_length", GgufTypeUint32, IntegerValue(64)},
        {prefix + "attention.head_count", GgufTypeUint32, IntegerValue(4)},
        {prefix + "attention.head_count_kv", GgufTypeUint32, IntegerValue(2)},
        {prefix + "context_length", GgufTypeUint32, IntegerValue(8)},
        {prefix + "rope.freq_base", GgufTypeFloat32, IntegerValue(0x461c4000)},
        {prefix + "attention.layer_norm_rms_epsilon", GgufTypeFloat32, IntegerValue(0x3727c5ac)},
        {"tokenizer.ggml.tokens", GgufTypeArray, StringArrayValue({"a", "b", "c", "d"})},
    };
}

std::vector<TensorEntry> TinyModelTensors(uint32_t type = GgmlTypeF32, bool biases = false) {
    std::vector<TensorEntry> tensors = {
        {"token_embd.weight", {32, 4}, type},
        {"output_norm.weight", {32}, type},
        {"output.weight", {32, 4}, type},
    };
    for (size_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const std::vector<TensorEntry> weights = {
            {prefix + "attn_norm.weight", {32}, type},
            {prefix + "attn_q.weight", {32, 32}, type},
            {prefix + "attn_k.weight", {32, 16}, type},
            {prefix + "attn_v.weight", {32, 16}, type},
            {prefix + "attn_output.weight", {32, 32}, type},
            {prefix + "ffn_norm.weight", {32}, type},
            {prefix + "ffn_gate.weight", {32, 64}, type},
            {prefix + "ffn_up.weight", {32, 64}, type},
            {prefix + "ffn_down.weight", {64, 32}, type},
        };
        tensors.insert(tensors.end(), weights.begin(), weights.end());
        if (biases) {
            tensors.push_back({prefix + "attn_q.bias", {32}});
            tensors.push_back({prefix + "attn_k.bias", {16}});
            tensors.push_back({prefix + "attn_v.bias", {16}});
        }
    }
    return tensors;
}

Bytes MakeModelGguf(std::vector<TensorEntry> tensors = TinyModelTensors(),
                   const std::vector<MetadataEntry>& metadata = TinyModelMetadata()) {
    Bytes payload;
    for (size_t index = 0; index < tensors.size(); ++index) {
        TensorEntry& tensor = tensors[index];
        payload.resize((payload.size() + 31) / 32 * 32, 0);
        tensor.offset = payload.size();
        size_t count = 1;
        for (uint64_t dim : tensor.shape) {
            count *= static_cast<size_t>(dim);
        }
        if (tensor.type == GgmlTypeF32) {
            // 每个 F32 张量具有独立的已知值，验证全局和逐层权重的赋值。
            const float value = static_cast<float>(index + 1) / 64.0f;
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (size_t i = 0; i < count; ++i) {
                AppendLe(payload, bits, 4);
            }
        } else if (tensor.type == GgmlTypeF16) {
            for (size_t i = 0; i < count; ++i) {
                AppendLe(payload, 0x3800, 2);
            }
        } else {
            // 三种量化格式均独立编码为 0.5，Q4_1 使用 0.5 * 3 - 1。
            for (size_t block = 0; block < count / 32; ++block) {
                AppendLe(payload, 0x3800, 2);
                if (tensor.type == GgmlTypeQ41) {
                    AppendLe(payload, 0xbc00, 2);
                }
                if (tensor.type == GgmlTypeQ80) {
                    payload.insert(payload.end(), 32, 1);
                } else if (tensor.type == GgmlTypeQ40) {
                    payload.insert(payload.end(), 16, static_cast<char>(0x99));
                } else {
                    payload.insert(payload.end(), 16, 0x33);
                }
            }
        }
    }
    Bytes data = MakeGguf(metadata, tensors, payload.size());
    std::copy(payload.begin(), payload.end(), data.end() - payload.size());
    return data;
}

void ExpectFilledTensor(const Tensor& tensor, const std::vector<size_t>& shape, float value) {
    ASSERT_EQ(tensor.GetShape(), shape);
    ASSERT_FALSE(tensor.data.empty());
    for (float actual : tensor.data) {
        EXPECT_FLOAT_EQ(actual, value);
    }
}

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

TEST_F(GgufLoaderTest, LoadsF32WeightsWithGqaAndBiases) {
    const Model model = LoadGgufModel(Write(MakeModelGguf(TinyModelTensors(GgmlTypeF32, true))));
    ASSERT_TRUE(model.loaded) << model.load_error;
    EXPECT_TRUE(model.load_error.empty());
    EXPECT_EQ(model.config.vocab_size, 4u);
    EXPECT_EQ(model.config.dim, 32u);
    EXPECT_EQ(model.config.hidden_dim, 64u);
    EXPECT_EQ(model.config.layers_num, 2u);
    EXPECT_EQ(model.config.heads_num, 4u);
    EXPECT_EQ(model.config.kv_heads_num, 2u);
    EXPECT_EQ(model.config.head_dim, 8u);
    EXPECT_EQ(model.config.max_seq_len, 8u);
    EXPECT_EQ(model.config.rope_type, RopeType::Normal);
    EXPECT_FLOAT_EQ(model.config.rope_theta, 10000.0f);
    EXPECT_FLOAT_EQ(model.config.rms_norm_eps, 1e-5f);
    ExpectFilledTensor(model.token_embedding, {4, 32}, 1.0f / 64);
    ExpectFilledTensor(model.final_norm, {32}, 2.0f / 64);
    ExpectFilledTensor(model.lm_head, {4, 32}, 3.0f / 64);
    ASSERT_EQ(model.layers.size(), 2u);
    for (size_t layer = 0; layer < model.layers.size(); ++layer) {
        SCOPED_TRACE(layer);
        const LayerWeights& weights = model.layers[layer];
        const float first = static_cast<float>(4 + layer * 12);
        ExpectFilledTensor(weights.attention_norm, {32}, first / 64);
        ExpectFilledTensor(weights.wq, {32, 32}, (first + 1) / 64);
        ExpectFilledTensor(weights.wk, {16, 32}, (first + 2) / 64);
        ExpectFilledTensor(weights.wv, {16, 32}, (first + 3) / 64);
        ExpectFilledTensor(weights.wo, {32, 32}, (first + 4) / 64);
        ExpectFilledTensor(weights.ffn_norm, {32}, (first + 5) / 64);
        ExpectFilledTensor(weights.w_gate, {64, 32}, (first + 6) / 64);
        ExpectFilledTensor(weights.w_up, {64, 32}, (first + 7) / 64);
        ExpectFilledTensor(weights.w_down, {32, 64}, (first + 8) / 64);
        ExpectFilledTensor(weights.bq, {32}, (first + 9) / 64);
        ExpectFilledTensor(weights.bk, {16}, (first + 10) / 64);
        ExpectFilledTensor(weights.bv, {16}, (first + 11) / 64);
    }
}

TEST_F(GgufLoaderTest, LoadsF16Qwen2Weights) {
    const Model model = LoadGgufModel(Write(MakeModelGguf(
        TinyModelTensors(GgmlTypeF16), TinyModelMetadata("qwen2"))));
    ASSERT_TRUE(model.loaded) << model.load_error;
    EXPECT_EQ(model.config.rope_type, RopeType::NeoX);
    ExpectFilledTensor(model.token_embedding, {4, 32}, 0.5f);
    ExpectFilledTensor(model.final_norm, {32}, 0.5f);
    ExpectFilledTensor(model.lm_head, {4, 32}, 0.5f);
    ASSERT_EQ(model.layers.size(), 2u);
    for (const auto& weights : model.layers) {
        ExpectFilledTensor(weights.wq, {32, 32}, 0.5f);
        ExpectFilledTensor(weights.w_down, {32, 64}, 0.5f);
        EXPECT_TRUE(weights.bq.data.empty());
        EXPECT_TRUE(weights.bk.data.empty());
        EXPECT_TRUE(weights.bv.data.empty());
    }
}

TEST_F(GgufLoaderTest, DequantizesModelWeightsToF32) {
    for (uint32_t type : {GgmlTypeQ80, GgmlTypeQ40, GgmlTypeQ41}) {
        SCOPED_TRACE(type);
        const Model model = LoadGgufModel(Write(MakeModelGguf(TinyModelTensors(type))));
        ASSERT_TRUE(model.loaded) << model.load_error;
        ExpectFilledTensor(model.token_embedding, {4, 32}, 0.5f);
        ExpectFilledTensor(model.final_norm, {32}, 0.5f);
        ExpectFilledTensor(model.lm_head, {4, 32}, 0.5f);
        ASSERT_EQ(model.layers.size(), 2u);
        for (const auto& weights : model.layers) {
            ExpectFilledTensor(weights.attention_norm, {32}, 0.5f);
            ExpectFilledTensor(weights.wk, {16, 32}, 0.5f);
            ExpectFilledTensor(weights.w_gate, {64, 32}, 0.5f);
        }
    }
}

TEST_F(GgufLoaderTest, TiesMissingOutputWeightsToEmbedding) {
    auto tensors = TinyModelTensors();
    tensors.erase(tensors.begin() + 2);
    Model model = LoadGgufModel(Write(MakeModelGguf(tensors)));
    ASSERT_TRUE(model.loaded) << model.load_error;
    EXPECT_EQ(model.lm_head.GetShape(), model.token_embedding.GetShape());
    EXPECT_EQ(model.lm_head.data, model.token_embedding.data);
    model.lm_head.data[0] = 42.0f;
    EXPECT_FLOAT_EQ(model.token_embedding.data[0], 1.0f / 64);
}

TEST_F(GgufLoaderTest, LoadedModelsEnterForwardBatchPath) {
    for (uint32_t type : {GgmlTypeF32, GgmlTypeF16, GgmlTypeQ80, GgmlTypeQ40, GgmlTypeQ41}) {
        SCOPED_TRACE(type);
        const Model model = LoadGgufModel(Write(MakeModelGguf(TinyModelTensors(type))));
        ASSERT_TRUE(model.loaded) << model.load_error;
        Context context(model);
        const Tensor logits = ForwardBatch(context, model, Batch(std::vector<int>{1, 2}, 0));
        EXPECT_EQ(logits.GetShape(), (std::vector<size_t>{4}));
        for (float value : logits.data) {
            EXPECT_TRUE(std::isfinite(value));
        }
        EXPECT_EQ(context.pos, 1u);
        EXPECT_EQ(context.token_history, (std::vector<int>{1, 2}));
    }
}

TEST_F(GgufLoaderTest, MissingRequiredWeightReturnsError) {
    auto tensors = TinyModelTensors();
    tensors.erase(tensors.begin() + 5);  // blk.0.attn_k.weight。
    const Model model = LoadGgufModel(Write(MakeModelGguf(tensors)));
    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find("Missing tensor: layers.0.wk"), std::string::npos);
}

TEST_F(GgufLoaderTest, IncorrectWeightShapesReturnError) {
    auto tensors = TinyModelTensors();
    tensors[5].shape = {16, 32};
    const Model model = LoadGgufModel(Write(MakeModelGguf(tensors)));
    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find("GGUF layers.0.wk"), std::string::npos);
}

TEST_F(GgufLoaderTest, IncorrectOptionalBiasShapeReturnsError) {
    auto tensors = TinyModelTensors(GgmlTypeF32, true);
    tensors.back().shape = {32};
    const Model model = LoadGgufModel(Write(MakeModelGguf(tensors)));
    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find("GGUF layers.1.bv"), std::string::npos);
}

TEST_F(GgufLoaderTest, IncompleteQuantizedRowReturnsError) {
    auto tensors = TinyModelTensors(GgmlTypeQ80);
    tensors[0].shape = {16, 8};
    const Model model = LoadGgufModel(Write(MakeModelGguf(tensors)));
    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find("row size must be a multiple of 32"), std::string::npos);
}

TEST_F(GgufLoaderTest, MissingVocabularyReturnsConfigError) {
    auto metadata = TinyModelMetadata();
    metadata.pop_back();
    const Model model = LoadGgufModel(Write(MakeModelGguf(TinyModelTensors(), metadata)));
    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error.find("Failed to build config: "), 0u);
}

TEST_F(GgufLoaderTest, InvalidDimensionsReturnConfigError) {
    auto metadata = TinyModelMetadata();
    metadata[2].type = GgufTypeInt32;
    metadata[2].value = IntegerValue(0xffffffff);
    const Model negative = LoadGgufModel(Write(MakeModelGguf(TinyModelTensors(), metadata)));
    EXPECT_FALSE(negative.loaded);
    EXPECT_NE(negative.load_error.find("llama.embedding_length"), std::string::npos);
    metadata = TinyModelMetadata();
    metadata[4].value = IntegerValue(5);
    const Model indivisible = LoadGgufModel(Write(MakeModelGguf(TinyModelTensors(), metadata)));
    EXPECT_FALSE(indivisible.loaded);
    EXPECT_NE(indivisible.load_error.find("dim not divisible by heads_num"), std::string::npos);
}

TEST_F(GgufLoaderTest, UnsupportedTensorTypeReturnsParseError) {
    const Model model = LoadGgufModel(Write(MakeGguf(
        TinyModelMetadata(), {{"token_embd.weight", {32, 4}, 999}}, 512)));
    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error.find("GGUF parse error: "), 0u);
    EXPECT_NE(model.load_error.find("999"), std::string::npos);
}

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
