// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 与 test_forward.cpp 一致，直接包含实现来测试内部形状映射、解码与校验。
#include "../src/gguf_loader.cpp"

namespace {

Model MakeGqaModel() {
    Model model;
    model.config.vocab_size = 8;
    model.config.dim = 4;
    model.config.hidden_dim = 6;
    model.config.layers_num = 2;
    model.config.heads_num = 2;
    model.config.kv_heads_num = 1;
    model.config.head_dim = 2;
    model.token_embedding = Tensor({8, 4});
    model.final_norm = Tensor({4});
    model.lm_head = Tensor({8, 4});
    model.layers.resize(2);
    for (auto& layer : model.layers) {
        layer.attention_norm = Tensor({4});
        layer.wq = Tensor({4, 4});
        layer.wk = Tensor({2, 4});
        layer.wv = Tensor({2, 4});
        layer.wo = Tensor({4, 4});
        layer.ffn_norm = Tensor({4});
        layer.w_gate = Tensor({6, 4});
        layer.w_up = Tensor({6, 4});
        layer.w_down = Tensor({4, 6});
    }
    return model;
}

void ExpectShapeFailure(const Model& model, const std::string& name) {
    try {
        ValidateLoadedShapes(model);
        FAIL() << "invalid model shape was accepted: " << name;
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(name), std::string::npos) << error.what();
    }
}

struct LayerShapeCase {
    Tensor LayerWeights::* member;
    const char* name;
    std::vector<size_t> wrong_shape;
};

const std::vector<LayerShapeCase> kRequiredLayerWeights = {
    {&LayerWeights::attention_norm, "attention_norm", {1, 4}},
    {&LayerWeights::wq, "wq", {2, 8}},
    {&LayerWeights::wk, "wk", {4, 2}},
    {&LayerWeights::wv, "wv", {4, 2}},
    {&LayerWeights::wo, "wo", {2, 8}},
    {&LayerWeights::ffn_norm, "ffn_norm", {1, 4}},
    {&LayerWeights::w_gate, "w_gate", {4, 6}},
    {&LayerWeights::w_up, "w_up", {4, 6}},
    {&LayerWeights::w_down, "w_down", {6, 4}},
};

TEST(GgufShapeTest, AcceptsGqaWeightsWithoutBiases) {
    EXPECT_NO_THROW(ValidateLoadedShapes(MakeGqaModel()));
}

TEST(GgufShapeTest, AcceptsOptionalAttentionBiases) {
    Model model = MakeGqaModel();
    for (auto& layer : model.layers) {
        layer.bq = Tensor({4});
        layer.bk = Tensor({2});
        layer.bv = Tensor({2});
    }
    EXPECT_NO_THROW(ValidateLoadedShapes(model));
}

TEST(GgufShapeTest, RejectsGlobalWeightShapes) {
    for (Tensor Model::* member : {&Model::token_embedding, &Model::final_norm,
                                   &Model::lm_head}) {
        Model model = MakeGqaModel();
        model.*member = Tensor({1});
        const char* name = member == &Model::token_embedding ? "GGUF token_embedding" :
                           member == &Model::final_norm ? "GGUF final_norm" : "GGUF lm_head";
        SCOPED_TRACE(name);
        ExpectShapeFailure(model, name);
    }
}

TEST(GgufShapeTest, RejectsLayerCountMismatch) {
    for (size_t count : {1u, 3u}) {
        SCOPED_TRACE(count);
        Model model = MakeGqaModel();
        model.layers.resize(count);
        ExpectShapeFailure(model, "GGUF layer count mismatch: expected 2, got " +
                                  std::to_string(count));
    }
}

TEST(GgufShapeTest, RejectsWrongLayerShapesWithSameElementCount) {
    for (const auto& weight : kRequiredLayerWeights) {
        Model model = MakeGqaModel();
        model.layers[1].*(weight.member) = Tensor(weight.wrong_shape);
        const std::string name = std::string("GGUF layers.1.") + weight.name;
        SCOPED_TRACE(name);
        ExpectShapeFailure(model, name);
    }
}

TEST(GgufShapeTest, RejectsMissingRequiredLayerWeights) {
    for (const auto& weight : kRequiredLayerWeights) {
        Model model = MakeGqaModel();
        model.layers[0].*(weight.member) = Tensor();
        const std::string name = std::string("GGUF layers.0.") + weight.name;
        SCOPED_TRACE(name);
        ExpectShapeFailure(model, name);
    }
}

TEST(GgufShapeTest, RejectsWrongOptionalBiasShapes) {
    const std::vector<std::pair<Tensor LayerWeights::*, std::string>> biases = {
        {&LayerWeights::bq, "bq"}, {&LayerWeights::bk, "bk"}, {&LayerWeights::bv, "bv"},
    };
    for (const auto& bias : biases) {
        Model model = MakeGqaModel();
        model.layers[1].*(bias.first) = Tensor({3});
        SCOPED_TRACE(bias.second);
        ExpectShapeFailure(model, "GGUF layers.1." + bias.second);
    }
}

TEST(GgufShapeTest, MapsGgufDimensionsToTensorShape) {
    EXPECT_EQ(MapShape({32}), (std::vector<size_t>{32}));
    EXPECT_EQ(MapShape({32, 64}), (std::vector<size_t>{64, 32}));
    EXPECT_EQ(MapShape({2, 3, 4}), (std::vector<size_t>{2, 3, 4}));
    EXPECT_EQ(MapShape(std::vector<uint64_t>(8, 1)), (std::vector<size_t>(8, 1)));
}

TEST(GgufShapeTest, RejectsInvalidGgufDimensions) {
    EXPECT_THROW(MapShape({}), std::runtime_error);
    EXPECT_THROW(MapShape({0, 32}), std::runtime_error);
    EXPECT_THROW(MapShape(std::vector<uint64_t>(9, 1)), std::runtime_error);
    if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
        EXPECT_THROW(MapShape({std::numeric_limits<uint64_t>::max()}), std::runtime_error);
    }
}

TEST(GgufDecodeTest, ReadsF32ValuesWithoutReordering) {
    const Tensor tensor = LoadQuantizedTensor(
        {0, 0, 0x80, 0x3f, 0, 0, 0x20, 0xc0}, {1, 2}, GgmlTypeF32);
    EXPECT_EQ(tensor.GetShape(), (std::vector<size_t>{1, 2}));
    EXPECT_EQ(tensor.data, (std::vector<float>{1.0f, -2.5f}));
}

TEST(GgufDecodeTest, ReadsF16NormalSubnormalAndSpecialValues) {
    const Tensor tensor = LoadQuantizedTensor(
        {0, 0, 0, 0x80, 0, 0x3c, 0, 0xc1, 1, 0, 0xff, 3,
         0, 4, 0xff, 0x7b, 0, 0x7c, 0, 0xfc, 0, 0x7e}, {11}, GgmlTypeF16);
    ASSERT_EQ(tensor.size(), 11u);
    EXPECT_FLOAT_EQ(tensor.data[0], 0.0f);
    EXPECT_FALSE(std::signbit(tensor.data[0]));
    EXPECT_FLOAT_EQ(tensor.data[1], 0.0f);
    EXPECT_TRUE(std::signbit(tensor.data[1]));
    EXPECT_FLOAT_EQ(tensor.data[2], 1.0f);
    EXPECT_FLOAT_EQ(tensor.data[3], -2.5f);
    EXPECT_FLOAT_EQ(tensor.data[4], std::ldexp(1.0f, -24));
    EXPECT_FLOAT_EQ(tensor.data[5], std::ldexp(1023.0f, -24));
    EXPECT_FLOAT_EQ(tensor.data[6], std::ldexp(1.0f, -14));
    EXPECT_FLOAT_EQ(tensor.data[7], 65504.0f);
    EXPECT_EQ(tensor.data[8], std::numeric_limits<float>::infinity());
    EXPECT_EQ(tensor.data[9], -std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isnan(tensor.data[10]));
}

TEST(GgufDecodeTest, DecodesQ80SignedValuesAndMultipleBlocks) {
    std::vector<uint8_t> raw(68, 0);
    raw[1] = 0x38;   // 第一块的 scale 为 0.5。
    raw[35] = 0xc0;  // 第二块的 scale 为 -2。
    const std::vector<int> values{-128, -127, -2, -1, 0, 1, 2, 127};
    for (size_t i = 0; i < 32; ++i) {
        raw[2 + i] = static_cast<uint8_t>(values[i % values.size()]);
        raw[36 + i] = raw[2 + i];
    }
    const Tensor tensor = LoadQuantizedTensor(raw, {2, 32}, GgmlTypeQ80);
    ASSERT_EQ(tensor.size(), 64u);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(tensor.data[i], values[i % values.size()] * 0.5f);
        EXPECT_FLOAT_EQ(tensor.data[32 + i], values[i % values.size()] * -2.0f);
    }
}

TEST(GgufDecodeTest, DecodesQ40NibbleOrderAndSignedScales) {
    std::vector<uint8_t> raw(36, 0);
    raw[1] = 0x38;
    raw[19] = 0xc0;
    for (size_t i = 0; i < 16; ++i) {
        raw[2 + i] = static_cast<uint8_t>(i | ((15 - i) << 4));
        raw[20 + i] = raw[2 + i];
    }
    const Tensor tensor = LoadQuantizedTensor(raw, {2, 32}, GgmlTypeQ40);
    ASSERT_EQ(tensor.size(), 64u);
    for (size_t i = 0; i < 16; ++i) {
        const int low = static_cast<int>(i) - 8;
        const int high = 7 - static_cast<int>(i);
        EXPECT_FLOAT_EQ(tensor.data[i], low * 0.5f);
        EXPECT_FLOAT_EQ(tensor.data[16 + i], high * 0.5f);
        EXPECT_FLOAT_EQ(tensor.data[32 + i], low * -2.0f);
        EXPECT_FLOAT_EQ(tensor.data[48 + i], high * -2.0f);
    }
}

TEST(GgufDecodeTest, DecodesQ41ScaleMinimumAndNibbleOrder) {
    std::vector<uint8_t> raw(40, 0);
    raw[1] = 0x38;
    raw[3] = 0xc0;   // 第一块为 0.5 * q - 2。
    raw[21] = 0xbc;
    raw[23] = 0x3d;  // 第二块为 -q + 1.25。
    for (size_t i = 0; i < 16; ++i) {
        raw[4 + i] = static_cast<uint8_t>(i | ((15 - i) << 4));
        raw[24 + i] = raw[4 + i];
    }
    const Tensor tensor = LoadQuantizedTensor(raw, {2, 32}, GgmlTypeQ41);
    ASSERT_EQ(tensor.size(), 64u);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(tensor.data[i], static_cast<float>(i) * 0.5f - 2.0f);
        EXPECT_FLOAT_EQ(tensor.data[16 + i], static_cast<float>(15 - i) * 0.5f - 2.0f);
        EXPECT_FLOAT_EQ(tensor.data[32 + i], -static_cast<float>(i) + 1.25f);
        EXPECT_FLOAT_EQ(tensor.data[48 + i], -static_cast<float>(15 - i) + 1.25f);
    }
}

TEST(GgufDecodeTest, RejectsMissingTruncatedAndTrailingBytes) {
    for (const auto& storage : std::vector<std::pair<uint32_t, size_t>>{
             {GgmlTypeF32, 128}, {GgmlTypeF16, 64}, {GgmlTypeQ80, 34},
             {GgmlTypeQ40, 18}, {GgmlTypeQ41, 20}}) {
        SCOPED_TRACE(storage.first);
        for (size_t size : {size_t{0}, storage.second - 1, storage.second + 1}) {
            SCOPED_TRACE(size);
            EXPECT_THROW(LoadQuantizedTensor(std::vector<uint8_t>(size), {32}, storage.first),
                         std::runtime_error);
        }
    }
}

TEST(GgufDecodeTest, RejectsInvalidShapesOverflowAndUnsupportedTypes) {
    const size_t max = std::numeric_limits<size_t>::max();
    for (const auto& shape : std::vector<std::vector<size_t>>{
             {}, {0}, {32, 0}, std::vector<size_t>(9, 1), {max, 2}, {max / 4 + 1}}) {
        SCOPED_TRACE(testing::PrintToString(shape));
        EXPECT_THROW(LoadQuantizedTensor({}, shape, GgmlTypeF32), std::runtime_error);
    }
    EXPECT_THROW(LoadQuantizedTensor({}, {max / 32 * 32}, GgmlTypeQ80), std::runtime_error);
    EXPECT_THROW(LoadQuantizedTensor({}, {32}, 999), std::runtime_error);
    for (uint32_t type : {GgmlTypeQ80, GgmlTypeQ40, GgmlTypeQ41}) {
        SCOPED_TRACE(type);
        EXPECT_THROW(LoadQuantizedTensor({}, {31}, type), std::runtime_error);
        EXPECT_THROW(LoadQuantizedTensor({}, {2, 16}, type), std::runtime_error);
    }
}

}  // namespace
