// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// CheckForwardInputs 已改为内部函数，直接包含实现以保持功能代码不变。
// 本测试提供前向符号，静态库中的 forward.cpp 对象不会被重复链接。
#include "../src/forward.cpp"

namespace {

class ForwardInputsTest : public ::testing::Test {
protected:
    Model model;

    void SetUp() override {
        model.config.vocab_size = 4;
        model.config.dim = 2;
        model.config.layers_num = 1;
        model.config.max_seq_len = 3;
        model.config.kv_heads_num = 1;
        model.config.head_dim = 2;
        model.loaded = true;
        model.layers.resize(model.config.layers_num);
        model.token_embedding = Tensor({4, 2}, 1.0f);
    }
};

TEST_F(ForwardInputsTest, AcceptsBoundaryTokensAndPositionsWithoutChangingState) {
    Context ctx(model);
    EXPECT_NO_THROW(CheckForwardInputs(ctx, model, 0));

    ctx.pos = model.config.max_seq_len - 1;
    ctx.prefill_len = 2;
    ctx.decode_len = 1;
    ctx.token_history = {1, 2};
    Tensor keys({1, 2}, 7.0f);
    Tensor values({1, 2}, 9.0f);
    ctx.kv_cache.Write(0, 2, keys, values);

    EXPECT_NO_THROW(CheckForwardInputs(ctx, model, 3));

    EXPECT_EQ(ctx.model_ptr, &model);
    EXPECT_EQ(ctx.pos, 2u);
    EXPECT_EQ(ctx.prefill_len, 2u);
    EXPECT_EQ(ctx.decode_len, 1u);
    EXPECT_EQ(ctx.token_history, (std::vector<int>{1, 2}));
    EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(0, 2, 0)[0], 7.0f);
    EXPECT_FLOAT_EQ(ctx.kv_cache.ValuePtr(0, 2, 0)[0], 9.0f);
    EXPECT_FLOAT_EQ(model.token_embedding.At2D(3, 1), 1.0f);
}

TEST_F(ForwardInputsTest, RejectsUnloadedModel) {
    const Context ctx(model);
    model.loaded = false;
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::runtime_error);
}

TEST_F(ForwardInputsTest, RejectsTokensOutsideVocabulary) {
    const Context ctx(model);
    EXPECT_THROW(CheckForwardInputs(ctx, model, -1), std::out_of_range);
    EXPECT_THROW(CheckForwardInputs(ctx, model, 4), std::out_of_range);
    EXPECT_THROW(CheckForwardInputs(ctx, model, std::numeric_limits<int>::max()),
                 std::out_of_range);

    model.config.vocab_size = 0;
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::out_of_range);
}

TEST_F(ForwardInputsTest, RejectsPositionsOutsideSequenceLimit) {
    Context ctx(model);
    ctx.pos = model.config.max_seq_len;
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::out_of_range);
    ctx.pos = std::numeric_limits<size_t>::max();
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::out_of_range);

    ctx.pos = 0;
    model.config.max_seq_len = 0;
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::out_of_range);
}

TEST_F(ForwardInputsTest, RejectsTooFewOrTooManyLayers) {
    const Context ctx(model);
    model.layers.clear();
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::runtime_error);
    model.layers.resize(2);
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::runtime_error);
}

TEST_F(ForwardInputsTest, RejectsEmptyOrUndersizedEmbedding) {
    const Context ctx(model);
    model.token_embedding = Tensor();
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::runtime_error);
    model.token_embedding = Tensor({7});
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::runtime_error);
}

TEST_F(ForwardInputsTest, AcceptsEmbeddingLargerThanRequired) {
    const Context ctx(model);
    model.token_embedding = Tensor({5, 2});
    EXPECT_NO_THROW(CheckForwardInputs(ctx, model, 3));
}

TEST_F(ForwardInputsTest, RejectsEmbeddingSizeOverflow) {
    const Context ctx(model);
    // 4 * (SIZE_MAX / 4 + 1) 会回绕为 0，必须在乘法之前拒绝。
    model.config.dim = std::numeric_limits<size_t>::max() / 4 + 1;
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::runtime_error);
}

// 固定且非零的 F32 权重，替代尚未实现的 LoadModel，测试不依赖外部模型文件。
Tensor FixedWeights(const std::vector<size_t>& shape, int seed) {
    Tensor tensor(shape);
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor.data[i] = static_cast<float>(static_cast<int>((i + seed) % 11) - 5)
                           * 0.05f;
    }
    return tensor;
}

Model MakeForwardModel(size_t heads = 1, size_t kv_heads = 1) {
    Model model;
    auto& config = model.config;
    config.vocab_size = 8;
    config.dim = 4;
    config.hidden_dim = 6;
    config.layers_num = 2;
    config.heads_num = heads;
    config.kv_heads_num = kv_heads;
    config.head_dim = config.dim / heads;
    config.max_seq_len = 8;
    model.token_embedding = FixedWeights({config.vocab_size, config.dim}, 1);
    model.final_norm = Tensor({config.dim}, 1.0f);
    model.lm_head = FixedWeights({config.vocab_size, config.dim}, 2);
    for (size_t layer = 0; layer < config.layers_num; ++layer) {
        LayerWeights weights;
        weights.attention_norm = Tensor({config.dim}, 1.0f);
        weights.wq = FixedWeights({heads * config.head_dim, config.dim}, 3);
        weights.wk = FixedWeights({kv_heads * config.head_dim, config.dim}, 4);
        weights.wv = FixedWeights({kv_heads * config.head_dim, config.dim}, 5);
        weights.wo = FixedWeights({config.dim, heads * config.head_dim}, 6);
        weights.ffn_norm = Tensor({config.dim}, 1.0f);
        weights.w_gate = FixedWeights({config.hidden_dim, config.dim}, 7);
        weights.w_up = FixedWeights({config.hidden_dim, config.dim}, 8);
        weights.w_down = FixedWeights({config.dim, config.hidden_dim}, 9);
        model.layers.push_back(weights);
    }
    model.loaded = true;
    return model;
}

void ExpectFiniteLogits(const Tensor& logits, size_t vocab_size) {
    ASSERT_EQ(logits.GetShape(), (std::vector<size_t>{vocab_size}));
    ASSERT_EQ(logits.size(), vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        EXPECT_TRUE(std::isfinite(logits.data[i])) << "logit=" << i;
    }
}

void ExpectSameLogits(const Tensor& actual, const Tensor& expected) {
    ASSERT_EQ(actual.GetShape(), expected.GetShape());
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual.data[i], expected.data[i], 1e-6f) << "logit=" << i;
    }
}

// 以下五项适配 tests/gtest/test_forward.cc；量化、采样器尚未实现，暂不移植。
TEST(ForwardTest, ForwardTokenProducesLogitShape) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ExpectFiniteLogits(ForwardToken(ctx, model, 1), model.config.vocab_size);
}

TEST(ForwardTest, ForwardTokenChangesKvCache) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(0, 0, 0)[0], 0.0f);
    EXPECT_FLOAT_EQ(ctx.kv_cache.ValuePtr(0, 0, 0)[0], 0.0f);
    ForwardToken(ctx, model, 1);
    for (int layer = 0; layer < 2; ++layer) {
        bool key_changed = false;
        bool value_changed = false;
        for (size_t d = 0; d < model.config.head_dim; ++d) {
            const float key = ctx.kv_cache.KeyPtr(layer, 0, 0)[d];
            const float value = ctx.kv_cache.ValuePtr(layer, 0, 0)[d];
            EXPECT_TRUE(std::isfinite(key));
            EXPECT_TRUE(std::isfinite(value));
            key_changed |= key != 0.0f;
            value_changed |= value != 0.0f;
            EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(layer, 1, 0)[d], 0.0f);
            EXPECT_FLOAT_EQ(ctx.kv_cache.ValuePtr(layer, 1, 0)[d], 0.0f);
        }
        EXPECT_TRUE(key_changed) << "layer=" << layer;
        EXPECT_TRUE(value_changed) << "layer=" << layer;
    }
}

TEST(ForwardTest, PositionIncrementsCorrectly) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ForwardToken(ctx, model, 1);
    const float first_key = ctx.kv_cache.KeyPtr(0, 0, 0)[0];
    ctx.pos = 1;
    ExpectFiniteLogits(ForwardToken(ctx, model, 2), model.config.vocab_size);
    EXPECT_EQ(ctx.pos, 1u);
    EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(0, 0, 0)[0], first_key);
}

TEST(ForwardTest, ForwardRejectsInvalidToken) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    EXPECT_THROW(ForwardToken(ctx, model, -1), std::out_of_range);
    EXPECT_THROW(ForwardToken(ctx, model, static_cast<int>(model.config.vocab_size)),
                 std::out_of_range);
}

TEST(ForwardTest, ForwardRejectsInvalidPosition) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ctx.pos = model.config.max_seq_len;
    EXPECT_THROW(ForwardToken(ctx, model, 1), std::out_of_range);
}

// 适配 tests/gtest/test_batch.cc 的五项 ForwardBatch 测试。
// Batch 自动生成连续位置，没有上游 tokens/positions 长度不一致的状态。
TEST(ForwardBatchTest, PrefillMatchesIndividual) {
    const Model model = MakeForwardModel();
    const std::vector<int> tokens{1, 4, 5, 2, 2, 7};
    Context individual(model);
    Tensor expected;
    for (size_t i = 0; i < tokens.size(); ++i) {
        individual.pos = i;
        expected = ForwardToken(individual, model, tokens[i]);
    }
    Context batched(model);
    const Tensor actual = ForwardBatch(batched, model, Batch(tokens, 0));
    ExpectFiniteLogits(actual, model.config.vocab_size);
    ExpectSameLogits(actual, expected);
}

TEST(ForwardBatchTest, SingleMatchesIndividual) {
    const Model model = MakeForwardModel();
    Context individual(model);
    individual.pos = 3;
    const Tensor expected = ForwardToken(individual, model, 2);
    Context batched(model);
    const Tensor actual = ForwardBatch(batched, model, Batch(2, 3));
    ExpectFiniteLogits(actual, model.config.vocab_size);
    ExpectSameLogits(actual, expected);
}

TEST(ForwardBatchTest, UpdatesContext) {
    const Model model = MakeForwardModel();
    const std::vector<int> tokens{1, 4, 5};
    Context ctx(model);
    ForwardBatch(ctx, model, Batch(tokens, 0));
    EXPECT_EQ(ctx.token_history, tokens);
    EXPECT_EQ(ctx.pos, tokens.size() - 1);
}

TEST(ForwardBatchTest, AppendsDecodeHistory) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ForwardBatch(ctx, model, Batch(std::vector<int>{1, 4}, 0));
    const Tensor logits = ForwardBatch(ctx, model, Batch(5, 2));
    ExpectFiniteLogits(logits, model.config.vocab_size);
    EXPECT_EQ(ctx.token_history, (std::vector<int>{1, 4, 5}));
    EXPECT_EQ(ctx.pos, 2u);
}

TEST(ForwardBatchTest, EmptyRejected) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(std::vector<int>{}, 0)),
                 std::runtime_error);
    EXPECT_TRUE(ctx.token_history.empty());
    EXPECT_EQ(ctx.pos, 0u);
}

// 补充已实现的 CPU 注意力、GQA 配置和 RoPE 模式传递检查。
TEST(ForwardTest, AttentionSharesKvHeadAndAveragesEqualScores) {
    const Model model = MakeForwardModel(2, 1);
    KvCache cache(1, 3, 1, 2);
    Tensor key({1, 2}, 0.0f);
    Tensor previous_value({1, 2});
    previous_value.data[0] = 2.0f;
    previous_value.data[1] = 4.0f;
    cache.Write(0, 0, key, previous_value);
    Tensor value({1, 2});
    value.data[0] = 6.0f;
    value.data[1] = 8.0f;
    const Tensor query({2, 2}, 0.0f);
    const Tensor actual = AttentionForward(model, query, key, value, 1, 0, cache,
                                           2, 1, 2, BackendKind::Cpu);
    ASSERT_EQ(actual.GetShape(), (std::vector<size_t>{2, 2}));
    for (size_t head = 0; head < 2; ++head) {
        EXPECT_NEAR(actual.At2D(head, 0), 4.0f, 1e-6f);
        EXPECT_NEAR(actual.At2D(head, 1), 6.0f, 1e-6f);
    }
    EXPECT_FLOAT_EQ(cache.ValuePtr(0, 2, 0)[0], 0.0f);
}

TEST(ForwardTest, ForwardTokenSupportsGroupedQueryAttention) {
    const Model model = MakeForwardModel(2, 1);
    Context ctx(model);
    ExpectFiniteLogits(ForwardToken(ctx, model, 1), model.config.vocab_size);
}

TEST(ForwardTest, ForwardRopeHonorsNeoXMode) {
    const Model model = MakeForwardModel();
    Tensor q({1, 4}, 0.0f);
    Tensor k({1, 4}, 0.0f);
    q.data[0] = 1.0f;
    k.data[3] = 1.0f;
    ForwardRope(model, q, k, 1, 10000.0f, RopeType::NeoX, BackendKind::Cpu);
    // NeoX 配对为 (0,2)、(1,3)，角度分别为 1 和 0.01。
    EXPECT_NEAR(q.At2D(0, 0), std::cos(1.0f), 1e-6f);
    EXPECT_NEAR(q.At2D(0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(q.At2D(0, 2), std::sin(1.0f), 1e-6f);
    EXPECT_NEAR(k.At2D(0, 1), -std::sin(0.01f), 1e-6f);
    EXPECT_NEAR(k.At2D(0, 3), std::cos(0.01f), 1e-6f);
}

}  // namespace
