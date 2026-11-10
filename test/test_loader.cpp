// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 基础场景移植自 mini-llama-cpp/tests/gtest/test_loader.cc。

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "loader.h"

namespace {

struct TensorLayout {
    const char* name;
    TensorKind kind;
    int layer;
    std::vector<size_t> shape;
};

// 两层 GQA 模型：独立列出形状，测试输入不依赖 references/ 或模型权重。
const std::vector<TensorLayout> kLayouts = {
    {"token_embedding", TensorKind::TokenEmbedding, -1, {8, 4}},
    {"layers.0.attention_norm", TensorKind::AttentionNorm, 0, {4}},
    {"layers.0.wq", TensorKind::Wq, 0, {4, 4}},
    {"layers.0.wk", TensorKind::Wk, 0, {2, 4}},
    {"layers.0.wv", TensorKind::Wv, 0, {2, 4}},
    {"layers.0.wo", TensorKind::Wo, 0, {4, 4}},
    {"layers.0.ffn_norm", TensorKind::FfnNorm, 0, {4}},
    {"layers.0.w_gate", TensorKind::WGate, 0, {6, 4}},
    {"layers.0.w_up", TensorKind::WUp, 0, {6, 4}},
    {"layers.0.w_down", TensorKind::WDown, 0, {4, 6}},
    {"layers.1.attention_norm", TensorKind::AttentionNorm, 1, {4}},
    {"layers.1.wq", TensorKind::Wq, 1, {4, 4}},
    {"layers.1.wk", TensorKind::Wk, 1, {2, 4}},
    {"layers.1.wv", TensorKind::Wv, 1, {2, 4}},
    {"layers.1.wo", TensorKind::Wo, 1, {4, 4}},
    {"layers.1.ffn_norm", TensorKind::FfnNorm, 1, {4}},
    {"layers.1.w_gate", TensorKind::WGate, 1, {6, 4}},
    {"layers.1.w_up", TensorKind::WUp, 1, {6, 4}},
    {"layers.1.w_down", TensorKind::WDown, 1, {4, 6}},
    {"final_norm", TensorKind::FinalNorm, -1, {4}},
    {"lm_head", TensorKind::LmHead, -1, {8, 4}},
};

std::string MakeDocument(const std::string& tokenizer =
    R"({"type":"json_vocab","path":"vocab.json","bos_id":1,"eos_id":2,"unk_id":0})") {
    std::ostringstream json;
    json << R"({"config":{"vocab_size":8,"dim":4,"hidden_dim":6,"n_layers":2,)"
         << R"("n_heads":2,"n_kv_heads":1,"max_seq_len":16,)"
         << R"("rope_theta":10000.0,"rms_norm_eps":0.00001},)";
    if (!tokenizer.empty()) {
        json << "\"tokenizer\":" << tokenizer << ',';
    }
    json << "\"tensors\":[";
    size_t offset = 0;
    for (size_t i = 0; i < kLayouts.size(); ++i) {
        const auto& layout = kLayouts[i];
        if (i > 0) json << ',';
        json << "{\"name\":\"" << layout.name << "\",\"shape\":[";
        size_t elements = 1;
        for (size_t axis = 0; axis < layout.shape.size(); ++axis) {
            if (axis > 0) json << ',';
            json << layout.shape[axis];
            elements *= layout.shape[axis];
        }
        const size_t bytes = elements * sizeof(float);
        json << "],\"dtype\":\"float32\",\"offset\":" << offset
             << ",\"byte_size\":" << bytes << '}';
        offset += bytes;
    }
    json << "]}";
    return json.str();
}

std::string ReplaceOnce(std::string json, const std::string& from,
                         const std::string& to) {
    const size_t pos = json.find(from);
    if (pos == std::string::npos) {
        throw std::runtime_error("test replacement not found: " + from);
    }
    json.replace(pos, from.size(), to);
    return json;
}

class LoaderTest : public testing::Test {
protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        file_path_ = std::filesystem::current_path() /
            ("loader_test_" + std::to_string(stamp) + ".json");
        weights_path_ = file_path_;
        weights_path_.replace_extension(".bin");
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(file_path_, error);
        EXPECT_FALSE(error) << error.message();
        std::filesystem::remove(weights_path_, error);
        EXPECT_FALSE(error) << error.message();
    }

    ModelDescriptor Parse(const std::string& json) const {
        std::ofstream file(file_path_, std::ios::binary);
        file << json;
        file.close();
        if (!file) throw std::runtime_error("failed to write test config");
        return ParseDescriptor(file_path_.string());
    }

    void ExpectFailure(const std::string& json, const std::string& message) const {
        try {
            Parse(json);
            FAIL() << "invalid descriptor was accepted";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
                << error.what();
        }
    }

    void WriteWeights(const ModelDescriptor& descriptor) const {
        std::ofstream file(weights_path_, std::ios::binary);
        for (size_t i = 0; i < descriptor.tensors.size(); ++i) {
            const auto& info = descriptor.tensors[i];
            std::vector<float> values(info.byte_size / sizeof(float));
            for (size_t j = 0; j < values.size(); ++j) {
                values[j] = static_cast<float>(100 * i + j) + 0.5f;
            }
            file.seekp(static_cast<std::streamoff>(info.offset));
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(info.byte_size));
        }
        file.close();
        if (!file) throw std::runtime_error("failed to write test weights");
    }

    void ExpectLoadedWeights(const Model& model) const {
        ASSERT_TRUE(model.loaded) << model.load_error;
        EXPECT_TRUE(model.load_error.empty());
        ASSERT_EQ(model.layers.size(), 2u);
        std::vector<const Tensor*> tensors = {&model.token_embedding};
        for (const auto& layer : model.layers) {
            tensors.insert(tensors.end(), {
                &layer.attention_norm, &layer.wq, &layer.wk, &layer.wv, &layer.wo,
                &layer.ffn_norm, &layer.w_gate, &layer.w_up, &layer.w_down});
            EXPECT_EQ(layer.bq.size(), 0u);
            EXPECT_EQ(layer.bk.size(), 0u);
            EXPECT_EQ(layer.bv.size(), 0u);
        }
        tensors.push_back(&model.final_norm);
        tensors.push_back(&model.lm_head);
        ASSERT_EQ(tensors.size(), kLayouts.size());
        for (size_t i = 0; i < tensors.size(); ++i) {
            SCOPED_TRACE(kLayouts[i].name);
            const Tensor& tensor = *tensors[i];
            EXPECT_EQ(tensor.GetShape(), kLayouts[i].shape);
            size_t elements = 1;
            for (size_t dim : kLayouts[i].shape) elements *= dim;
            ASSERT_EQ(tensor.size(), elements);
            for (size_t j = 0; j < tensor.size(); ++j) {
                EXPECT_FLOAT_EQ(tensor.data[j], static_cast<float>(100 * i + j) + 0.5f);
            }
        }
    }

    std::filesystem::path file_path_;
    std::filesystem::path weights_path_;
};

TEST_F(LoaderTest, ParsesTypedMetadataWithoutWeights) {
    const ModelDescriptor descriptor = Parse(MakeDocument());
    EXPECT_EQ(descriptor.config.vocab_size, 8u);
    EXPECT_EQ(descriptor.config.dim, 4u);
    EXPECT_EQ(descriptor.config.hidden_dim, 6u);
    EXPECT_EQ(descriptor.config.layers_num, 2u);
    EXPECT_EQ(descriptor.config.heads_num, 2u);
    EXPECT_EQ(descriptor.config.kv_heads_num, 1u);
    EXPECT_EQ(descriptor.config.head_dim, 2u);
    EXPECT_EQ(descriptor.config.max_seq_len, 16u);
    EXPECT_FLOAT_EQ(descriptor.config.rope_theta, 10000.0f);
    EXPECT_FLOAT_EQ(descriptor.config.rms_norm_eps, 1e-5f);
    EXPECT_EQ(descriptor.config.rope_type, RopeType::Normal);
    EXPECT_EQ(descriptor.tokenizer.type, TokenizerType::JsonVocab);
    EXPECT_EQ(descriptor.tokenizer.path, "vocab.json");
    EXPECT_EQ(descriptor.tokenizer.bos_id, 1);
    EXPECT_EQ(descriptor.tokenizer.eos_id, 2);
    EXPECT_EQ(descriptor.tokenizer.unk_id, 0);
    ASSERT_EQ(descriptor.tensors.size(), kLayouts.size());
    size_t offset = 0;
    for (size_t i = 0; i < kLayouts.size(); ++i) {
        SCOPED_TRACE(kLayouts[i].name);
        const auto& tensor = descriptor.tensors[i];
        EXPECT_EQ(tensor.id.kind, kLayouts[i].kind);
        EXPECT_EQ(tensor.id.layer, kLayouts[i].layer);
        EXPECT_EQ(tensor.shape, kLayouts[i].shape);
        EXPECT_EQ(tensor.dataType, DataType::Float32);
        EXPECT_EQ(tensor.offset, offset);
        size_t elements = 1;
        for (size_t dim : kLayouts[i].shape) elements *= dim;
        EXPECT_EQ(tensor.byte_size, elements * sizeof(float));
        offset += tensor.byte_size;
    }
}

TEST_F(LoaderTest, DefaultsAsciiWithoutTokenizerBlock) {
    const auto descriptor = Parse(MakeDocument(""));
    EXPECT_EQ(descriptor.tokenizer.type, TokenizerType::Ascii);
    EXPECT_TRUE(descriptor.tokenizer.path.empty());
    EXPECT_EQ(descriptor.tokenizer.bos_id, 1);
    EXPECT_EQ(descriptor.tokenizer.eos_id, 2);
    EXPECT_EQ(descriptor.tokenizer.unk_id, 0);
}

TEST_F(LoaderTest, DefaultsAsciiWithEmptyTokenizerObject) {
    const auto descriptor = Parse(MakeDocument("{}"));
    EXPECT_EQ(descriptor.tokenizer.type, TokenizerType::Ascii);
    EXPECT_TRUE(descriptor.tokenizer.path.empty());
}

TEST_F(LoaderTest, ParsesCustomTokenizerIdsAndEscapedPath) {
    const auto descriptor = Parse(MakeDocument(
        R"({"type":"json_vocab","path":"dir\\vocab\"{test}.json","bos_id":3,"eos_id":4,"unk_id":5})"));
    EXPECT_EQ(descriptor.tokenizer.path, "dir\\vocab\"{test}.json");
    EXPECT_EQ(descriptor.tokenizer.bos_id, 3);
    EXPECT_EQ(descriptor.tokenizer.eos_id, 4);
    EXPECT_EQ(descriptor.tokenizer.unk_id, 5);
}

TEST_F(LoaderTest, AcceptsUnsortedNonOverlappingRangesWithGaps) {
    const auto descriptor = Parse(ReplaceOnce(
        MakeDocument(), "\"offset\":0", "\"offset\":4096"));
    EXPECT_EQ(descriptor.tensors.front().offset, 4096u);
    EXPECT_EQ(descriptor.tensors[1].offset, 128u);
}

TEST_F(LoaderTest, RejectsMissingConfigFile) {
    EXPECT_THROW(ParseDescriptor(file_path_.string()), std::runtime_error);
}

TEST_F(LoaderTest, RejectsEmptyConfigFile) {
    ExpectFailure("", "缺少必需字段");
}

TEST_F(LoaderTest, RejectsMissingRequiredTensor) {
    std::string json = MakeDocument();
    const size_t start = json.find("{\"name\":\"token_embedding\"");
    ASSERT_NE(start, std::string::npos);
    const size_t end = json.find("},", start);
    ASSERT_NE(end, std::string::npos);
    json.erase(start, end - start + 2);
    ExpectFailure(json, "缺少必需张量");
}

TEST_F(LoaderTest, RejectsEmptyTensorArray) {
    std::string json = MakeDocument();
    const size_t start = json.find("\"tensors\":[");
    ASSERT_NE(start, std::string::npos);
    json.erase(start);
    json += "\"tensors\":[]}";
    ExpectFailure(json, "模型描述中没有张量");
}

TEST_F(LoaderTest, RejectsOffsetOverflow) {
    ExpectFailure(ReplaceOnce(MakeDocument(), "\"offset\":0",
        "\"offset\":" + std::to_string(std::numeric_limits<size_t>::max())),
        "偏移量溢出");
}

TEST_F(LoaderTest, RejectsTensorSizeOverflow) {
    ExpectFailure(ReplaceOnce(MakeDocument(), "[8,4]",
        "[2147483647,2147483647,2147483647]"), "溢出");
}

TEST_F(LoaderTest, RejectsUnclosedTensorArray) {
    std::string json = MakeDocument();
    json.erase(json.size() - 2, 1);
    ExpectFailure(json, "JSON 容器未闭合");
}

TEST_F(LoaderTest, LoadsAllLayerAndGlobalWeightsById) {
    const auto descriptor = Parse(MakeDocument());
    WriteWeights(descriptor);

    const Model model = LoadModel(file_path_.string(), weights_path_.string());

    ExpectLoadedWeights(model);
    EXPECT_EQ(model.config.dim, 4u);
    EXPECT_EQ(model.config.hidden_dim, 6u);
    EXPECT_EQ(model.config.layers_num, 2u);
    EXPECT_EQ(model.config.heads_num, 2u);
    EXPECT_EQ(model.config.kv_heads_num, 1u);
    EXPECT_EQ(model.config.head_dim, 2u);
}

TEST_F(LoaderTest, LoadsWeightsAtUnsortedOffsetsWithGaps) {
    const auto descriptor = Parse(ReplaceOnce(
        MakeDocument(), "\"offset\":0", "\"offset\":4096"));
    WriteWeights(descriptor);

    const Model model = LoadModel(file_path_.string(), weights_path_.string());

    ExpectLoadedWeights(model);
}

class LoadModelFailureTest : public LoaderTest {
protected:
    void SetUp() override {
        LoaderTest::SetUp();
        saved_stderr_ = std::cerr.rdbuf(stderr_.rdbuf());
    }

    void TearDown() override {
        std::cerr.rdbuf(saved_stderr_);
        LoaderTest::TearDown();
    }

    std::ostringstream stderr_;

private:
    std::streambuf* saved_stderr_ = nullptr;
};

TEST_F(LoadModelFailureTest, ReturnsUnloadedModelForMissingConfig) {
    const Model model = LoadModel(file_path_.string(), "unused_weights.bin");

    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find("无法打开配置文件"), std::string::npos);
    EXPECT_NE(model.load_error.find(file_path_.string()), std::string::npos);
    EXPECT_EQ(stderr_.str(), "Failed to load model: " + model.load_error + '\n');
    EXPECT_TRUE(model.layers.empty());
    EXPECT_EQ(model.token_embedding.size(), 0u);
    EXPECT_EQ(model.final_norm.size(), 0u);
    EXPECT_EQ(model.lm_head.size(), 0u);
}

TEST_F(LoadModelFailureTest, PreservesDescriptorErrorAndLogsItOnce) {
    std::string expected_error;
    try {
        Parse(ReplaceOnce(MakeDocument(), "\"dim\":4,", ""));
    } catch (const std::runtime_error& error) {
        expected_error = error.what();
    }
    ASSERT_FALSE(expected_error.empty());

    const Model model = LoadModel(file_path_.string(), "unused_weights.bin");

    EXPECT_FALSE(model.loaded);
    EXPECT_EQ(model.load_error, expected_error);
    EXPECT_EQ(stderr_.str(), "Failed to load model: " + expected_error + '\n');
}

TEST_F(LoadModelFailureTest, ReturnsUnloadedModelForMissingWeights) {
    Parse(MakeDocument());

    const Model model = LoadModel(file_path_.string(), weights_path_.string());

    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find(weights_path_.string()), std::string::npos);
    EXPECT_EQ(stderr_.str(), "Failed to load model: " + model.load_error + '\n');
    EXPECT_TRUE(model.layers.empty());
}

TEST_F(LoadModelFailureTest, RejectsTruncatedLayerWeightsAndDiscardsPartialModel) {
    const auto descriptor = Parse(MakeDocument());
    WriteWeights(descriptor);
    const auto& info = descriptor.tensors[16]; // layers.1.w_gate
    std::filesystem::resize_file(weights_path_, info.offset + 2 * sizeof(float));

    const Model model = LoadModel(file_path_.string(), weights_path_.string());
    SCOPED_TRACE(model.load_error);

    EXPECT_FALSE(model.loaded);
    EXPECT_NE(model.load_error.find("layers.1.w_gate"), std::string::npos);
    EXPECT_NE(model.load_error.find("预期字节数: 96"), std::string::npos);
    EXPECT_NE(model.load_error.find("实际读取字节数: 8"), std::string::npos);
    EXPECT_EQ(stderr_.str(), "Failed to load model: " + model.load_error + '\n');
    EXPECT_TRUE(model.layers.empty());
    EXPECT_EQ(model.token_embedding.size(), 0u);
    EXPECT_EQ(model.final_norm.size(), 0u);
    EXPECT_EQ(model.lm_head.size(), 0u);
}

struct InvalidCase {
    const char* name;
    const char* from;
    const char* to;
    const char* message;
};

class InvalidDescriptorTest : public LoaderTest,
                              public testing::WithParamInterface<InvalidCase> {};

TEST_P(InvalidDescriptorTest, RejectsInvalidMetadata) {
    const auto& invalid = GetParam();
    ExpectFailure(ReplaceOnce(MakeDocument(), invalid.from, invalid.to), invalid.message);
}

INSTANTIATE_TEST_SUITE_P(Boundaries, InvalidDescriptorTest, testing::Values(
    InvalidCase{"MissingConfigField", "\"dim\":4,", "", "缺少必需字段: dim"},
    InvalidCase{"ZeroHeads", "\"n_heads\":2", "\"n_heads\":0", "必须为正数"},
    InvalidCase{"NegativeLayers", "\"n_layers\":2", "\"n_layers\":-1", "必须为正数"},
    InvalidCase{"FractionalHeads", "\"n_heads\":2", "\"n_heads\":2.5", "数值格式无效"},
    InvalidCase{"NonDivisibleHeads", "\"n_heads\":2", "\"n_heads\":3", "dim 必须能被 n_heads 整除"},
    InvalidCase{"NonDivisibleKvHeads", "\"n_kv_heads\":1", "\"n_kv_heads\":3", "n_heads 必须能被 n_kv_heads 整除"},
    InvalidCase{"OddHeadDim", "\"n_heads\":2", "\"n_heads\":4", "必须为正偶数"},
    InvalidCase{"NonfiniteTheta", "10000.0", "NaN", "rope_theta 必须为正数"},
    InvalidCase{"ZeroEpsilon", "0.00001", "0", "rms_norm_eps 必须为正数"},
    InvalidCase{"ExcessiveLayers", "\"n_layers\":2", "\"n_layers\":2147483647", "缺少必需张量"},
    InvalidCase{"UnsupportedTokenizer", "json_vocab", "bpe", "不支持的分词器类型"},
    InvalidCase{"MissingTokenizerPath", "\"path\":\"vocab.json\",", "", "必须提供路径"},
    InvalidCase{"NegativeSpecialId", "\"bos_id\":1", "\"bos_id\":-1", "必须为非负数"},
    InvalidCase{"DuplicateSpecialId", "\"bos_id\":1", "\"bos_id\":2", "必须互不相同"},
    InvalidCase{"FractionalSpecialId", "\"bos_id\":1", "\"bos_id\":1.5", "数值格式无效"},
    InvalidCase{"DuplicateTensor", "\"final_norm\"", "\"token_embedding\"", "张量名称重复"},
    InvalidCase{"RequiredShapeMismatch", "[8,4]", "[4,8]", "张量形状不匹配"},
    InvalidCase{"OverlappingRanges", "\"offset\":0", "\"offset\":4", "张量字节区间重叠"},
    InvalidCase{"MissingDtype", "\"dtype\":\"float32\",", "", "缺少必需字段: dtype"},
    InvalidCase{"UnsupportedDtype", "float32", "float16", "不支持的 dtype"},
    InvalidCase{"MissingName", "\"name\":\"token_embedding\",", "", "缺少必需字段: name"},
    InvalidCase{"UnknownTensor", "token_embedding", "unknown_tensor", "不支持的张量名称"},
    InvalidCase{"OutOfRangeLayer", "layers.0.wq", "layers.2.wq", "张量层号越界"},
    InvalidCase{"NegativeLayer", "layers.0.wq", "layers.-1.wq", "不支持的张量名称"},
    InvalidCase{"FractionalShape", "[8,4]", "[8.5,4]", "整数值无效"},
    InvalidCase{"ZeroShape", "[8,4]", "[0,4]", "张量形状无效"},
    InvalidCase{"NegativeShape", "[8,4]", "[-8,4]", "张量形状无效"},
    InvalidCase{"EmptyShape", "[8,4]", "[]", "张量形状不能为空"},
    InvalidCase{"MissingShape", "\"shape\":[8,4],", "", "缺少必需字段: shape"},
    InvalidCase{"TrailingShapeComma", "[8,4]", "[8,4,]", "张量形状无效"},
    InvalidCase{"NegativeOffset", "\"offset\":0", "\"offset\":-1", "非负整数值无效"},
    InvalidCase{"FractionalOffset", "\"offset\":0", "\"offset\":0.5", "数值格式无效"},
    InvalidCase{"MissingOffset", "\"offset\":0,", "", "缺少必需字段: offset"},
    InvalidCase{"ByteSizeMismatch", "\"byte_size\":128", "\"byte_size\":124", "byte_size 不匹配"},
    InvalidCase{"NegativeByteSize", "\"byte_size\":128", "\"byte_size\":-1", "非负整数值无效"},
    InvalidCase{"MissingTensors", "\"tensors\"", "\"weights\"", "缺少必需字段: tensors"}
), [](const testing::TestParamInfo<InvalidCase>& info) { return info.param.name; });

}  // namespace
