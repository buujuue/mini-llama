#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "forward.h"

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

    ctx.pos_ = model.config.max_seq_len - 1;
    ctx.prefill_len_ = 2;
    ctx.decode_len_ = 1;
    ctx.token_history_ = {1, 2};
    Tensor keys({1, 2}, 7.0f);
    Tensor values({1, 2}, 9.0f);
    ctx.kv_cache_.Write(0, 2, keys, values);

    EXPECT_NO_THROW(CheckForwardInputs(ctx, model, 3));

    EXPECT_EQ(ctx.model_ptr_, &model);
    EXPECT_EQ(ctx.pos_, 2u);
    EXPECT_EQ(ctx.prefill_len_, 2u);
    EXPECT_EQ(ctx.decode_len_, 1u);
    EXPECT_EQ(ctx.token_history_, (std::vector<int>{1, 2}));
    EXPECT_FLOAT_EQ(ctx.kv_cache_.KeyPtr(0, 2, 0)[0], 7.0f);
    EXPECT_FLOAT_EQ(ctx.kv_cache_.ValuePtr(0, 2, 0)[0], 9.0f);
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
    ctx.pos_ = model.config.max_seq_len;
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::out_of_range);
    ctx.pos_ = std::numeric_limits<size_t>::max();
    EXPECT_THROW(CheckForwardInputs(ctx, model, 0), std::out_of_range);

    ctx.pos_ = 0;
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

}  // namespace
