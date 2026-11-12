// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 与 test_forward.cpp 一致，直接包含实现来测试内部的纯形状校验。
// 本文件提供加载器符号，测试目标不再单独编译 gguf_loader.cpp。
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

}  // namespace
