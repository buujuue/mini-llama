// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// 直接包含实现，以保留 CPU 注意力与 RoPE 辅助函数的数值测试。
// 前向计算与输入校验均通过公开的 ForwardBatch 入口验证。
// 本测试提供前向符号，静态库中的 forward.cpp 对象不会被重复链接。
#include "../src/forward.cpp"

namespace {

// 固定且非零的 F32 权重，测试不依赖外部模型文件。
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

void ExpectSameKvCache(const KvCache& actual, const KvCache& expected,
                       const ModelConfig& config) {
    for (size_t layer = 0; layer < config.layers_num; ++layer) {
        for (size_t pos = 0; pos < config.max_seq_len; ++pos) {
            for (size_t head = 0; head < config.kv_heads_num; ++head) {
                for (size_t d = 0; d < config.head_dim; ++d) {
                    EXPECT_FLOAT_EQ(actual.KeyPtr(layer, pos, head)[d],
                                    expected.KeyPtr(layer, pos, head)[d]);
                    EXPECT_FLOAT_EQ(actual.ValuePtr(layer, pos, head)[d],
                                    expected.ValuePtr(layer, pos, head)[d]);
                }
            }
        }
    }
}

class ForwardBatchInputsTest : public ::testing::Test {
protected:
    Model model = MakeForwardModel();
};

TEST_F(ForwardBatchInputsTest, AcceptsBoundaryTokensAndPositions) {
    Context ctx(model);
    ExpectFiniteLogits(ForwardBatch(ctx, model, Batch(0, 0)), model.config.vocab_size);
    const float first_key = ctx.kv_cache.KeyPtr(0, 0, 0)[0];
    const float first_value = ctx.kv_cache.ValuePtr(0, 0, 0)[0];
    const int last_token = static_cast<int>(model.config.vocab_size - 1);
    const size_t last_pos = model.config.max_seq_len - 1;
    const float embedding_value = model.token_embedding.At2D(last_token, 0);
    ctx.prefill_len = 2;
    ctx.decode_len = 1;

    ExpectFiniteLogits(ForwardBatch(ctx, model, Batch(last_token, last_pos)),
                      model.config.vocab_size);

    EXPECT_EQ(ctx.model_ptr, &model);
    EXPECT_EQ(ctx.pos, last_pos);
    EXPECT_EQ(ctx.prefill_len, 2u);
    EXPECT_EQ(ctx.decode_len, 1u);
    EXPECT_EQ(ctx.token_history, (std::vector<int>{0, last_token}));
    EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(0, 0, 0)[0], first_key);
    EXPECT_FLOAT_EQ(ctx.kv_cache.ValuePtr(0, 0, 0)[0], first_value);
    EXPECT_FLOAT_EQ(model.token_embedding.At2D(last_token, 0), embedding_value);
}

TEST_F(ForwardBatchInputsTest, RejectsUnloadedModel) {
    Context ctx(model);
    model.loaded = false;
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::runtime_error);
}

TEST_F(ForwardBatchInputsTest, RejectsTokensOutsideVocabulary) {
    Context ctx(model);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(-1)), std::out_of_range);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(static_cast<int>(model.config.vocab_size))),
                 std::out_of_range);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(std::numeric_limits<int>::max())),
                 std::out_of_range);

    model.config.vocab_size = 0;
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::out_of_range);
}

TEST_F(ForwardBatchInputsTest, RejectsPositionsOutsideSequenceLimit) {
    Context ctx(model);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0, model.config.max_seq_len)),
                 std::out_of_range);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0, std::numeric_limits<size_t>::max())),
                 std::out_of_range);

    model.config.max_seq_len = 0;
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::out_of_range);
}

TEST_F(ForwardBatchInputsTest, RejectsTooFewOrTooManyLayers) {
    Context ctx(model);
    model.layers.clear();
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::runtime_error);
    model.layers.resize(model.config.layers_num + 1);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::runtime_error);
}

TEST_F(ForwardBatchInputsTest, RejectsEmptyOrUndersizedEmbedding) {
    Context ctx(model);
    model.token_embedding = Tensor();
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::runtime_error);
    model.token_embedding = Tensor({model.config.vocab_size * model.config.dim - 1});
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::runtime_error);
}

TEST_F(ForwardBatchInputsTest, AcceptsEmbeddingLargerThanRequired) {
    Context ctx(model);
    model.token_embedding = FixedWeights({model.config.vocab_size + 1, model.config.dim}, 1);
    ExpectFiniteLogits(ForwardBatch(ctx, model, Batch(static_cast<int>(model.config.vocab_size - 1))),
                      model.config.vocab_size);
}

TEST_F(ForwardBatchInputsTest, RejectsEmbeddingSizeOverflow) {
    Context ctx(model);
    // 必须在计算 vocab_size * dim 之前拒绝元素数溢出。
    model.config.dim = std::numeric_limits<size_t>::max() / model.config.vocab_size + 1;
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(0)), std::runtime_error);
}

// 以下五项适配 tests/gtest/test_forward.cc，通过单 token 批次验证公开入口。
TEST(ForwardTest, SingleTokenBatchProducesLogitShape) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ExpectFiniteLogits(ForwardBatch(ctx, model, Batch(1)), model.config.vocab_size);
}

TEST(ForwardTest, SingleTokenBatchChangesKvCache) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(0, 0, 0)[0], 0.0f);
    EXPECT_FLOAT_EQ(ctx.kv_cache.ValuePtr(0, 0, 0)[0], 0.0f);
    ForwardBatch(ctx, model, Batch(1));
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

TEST(ForwardTest, BatchPositionUpdatesContext) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ForwardBatch(ctx, model, Batch(1, 0));
    const float first_key = ctx.kv_cache.KeyPtr(0, 0, 0)[0];
    ExpectFiniteLogits(ForwardBatch(ctx, model, Batch(2, 1)), model.config.vocab_size);
    EXPECT_EQ(ctx.pos, 1u);
    EXPECT_EQ(ctx.token_history, (std::vector<int>{1, 2}));
    EXPECT_FLOAT_EQ(ctx.kv_cache.KeyPtr(0, 0, 0)[0], first_key);
}

TEST(ForwardTest, ForwardRejectsInvalidToken) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(-1)), std::out_of_range);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(static_cast<int>(model.config.vocab_size))),
                 std::out_of_range);
}

TEST(ForwardTest, ForwardRejectsInvalidPosition) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    EXPECT_THROW(ForwardBatch(ctx, model, Batch(1, model.config.max_seq_len)),
                 std::out_of_range);
}

// 适配 tests/gtest/test_batch.cc 的五项 ForwardBatch 测试。
// Batch 自动生成连续位置，没有上游 tokens/positions 长度不一致的状态。
TEST(ForwardBatchTest, PrefillMatchesIndividual) {
    const Model model = MakeForwardModel();
    const std::vector<int> tokens{1, 4, 5, 2, 2, 7};
    Context individual(model);
    Tensor expected;
    for (size_t i = 0; i < tokens.size(); ++i) {
        expected = ForwardBatch(individual, model, Batch(tokens[i], i));
    }
    Context batched(model);
    const Tensor actual = ForwardBatch(batched, model, Batch(tokens, 0));
    ExpectFiniteLogits(actual, model.config.vocab_size);
    ExpectSameLogits(actual, expected);
    EXPECT_EQ(batched.token_history, individual.token_history);
    EXPECT_EQ(batched.pos, individual.pos);
    ExpectSameKvCache(batched.kv_cache, individual.kv_cache, model.config);
}

TEST(ForwardBatchTest, SingleTokenConstructorMatchesSequence) {
    const Model model = MakeForwardModel();
    Context sequence(model);
    const Tensor expected = ForwardBatch(sequence, model, Batch(std::vector<int>{2}, 3));
    Context batched(model);
    const Tensor actual = ForwardBatch(batched, model, Batch(2, 3));
    ExpectFiniteLogits(actual, model.config.vocab_size);
    ExpectSameLogits(actual, expected);
    EXPECT_EQ(batched.token_history, sequence.token_history);
    EXPECT_EQ(batched.pos, sequence.pos);
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

TEST(ForwardBatchTest, AcceptsBatchEndingAtSequenceLimit) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    const std::vector<int> tokens{1, 2};
    const Tensor logits = ForwardBatch(ctx, model,
                                      Batch(tokens, model.config.max_seq_len - tokens.size()));
    ExpectFiniteLogits(logits, model.config.vocab_size);
    EXPECT_EQ(ctx.pos, model.config.max_seq_len - 1);
    EXPECT_EQ(ctx.token_history, tokens);
}

TEST(ForwardBatchTest, RejectsBatchEndingPastSequenceLimit) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ForwardBatch(ctx, model, Batch(1));
    const Context before = ctx;
    const std::vector<int> tokens{2, 3};
    for (const size_t start_pos : {model.config.max_seq_len - 1,
                                   std::numeric_limits<size_t>::max() - 1}) {
        EXPECT_THROW(ForwardBatch(ctx, model, Batch(tokens, start_pos)), std::out_of_range);
        EXPECT_EQ(ctx.pos, before.pos);
        EXPECT_EQ(ctx.token_history, before.token_history);
        ExpectSameKvCache(ctx.kv_cache, before.kv_cache, model.config);
    }
}

TEST(ForwardBatchTest, RejectsInvalidTokenBeforeChangingState) {
    const Model model = MakeForwardModel();
    Context ctx(model);
    ForwardBatch(ctx, model, Batch(1));
    ctx.prefill_len = 1;
    ctx.decode_len = 2;
    const Context before = ctx;
    for (const int invalid_token : {-1, static_cast<int>(model.config.vocab_size)}) {
        const Batch batch(std::vector<int>{2, invalid_token}, 1);
        EXPECT_THROW(ForwardBatch(ctx, model, batch), std::out_of_range);
        EXPECT_EQ(ctx.model_ptr, before.model_ptr);
        EXPECT_EQ(ctx.pos, before.pos);
        EXPECT_EQ(ctx.prefill_len, before.prefill_len);
        EXPECT_EQ(ctx.decode_len, before.decode_len);
        EXPECT_EQ(ctx.token_history, before.token_history);
        ExpectSameKvCache(ctx.kv_cache, before.kv_cache, model.config);
    }
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

TEST(ForwardTest, SingleTokenBatchSupportsGroupedQueryAttention) {
    const Model model = MakeForwardModel(2, 1);
    Context ctx(model);
    ExpectFiniteLogits(ForwardBatch(ctx, model, Batch(1)), model.config.vocab_size);
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
