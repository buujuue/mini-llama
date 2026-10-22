#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "context.h"
#include "tensor.h"

// 补充 Context 已实现的构造与请求隔离验收，不依赖尚未实现的模型加载和 Forward。
TEST(ContextTest, InitializesStateAndCacheFromModelConfig) {
    Model model;
    model.config.layers_num = 2;
    model.config.max_seq_len = 3;
    model.config.kv_heads_num = 2;
    model.config.head_dim = 4;
    const Context context(model);

    EXPECT_EQ(context.model_ptr_, &model);
    EXPECT_EQ(context.pos_, 0u);
    EXPECT_EQ(context.prefill_len_, 0u);
    EXPECT_EQ(context.decode_len_, 0u);
    EXPECT_TRUE(context.token_history_.empty());
    const float* keys = context.kv_cache_.KeyPtr(1, 2, 1);
    const float* values = context.kv_cache_.ValuePtr(1, 2, 1);
    for (int dim = 0; dim < 4; ++dim) {
        EXPECT_FLOAT_EQ(keys[dim], 0.0f);
        EXPECT_FLOAT_EQ(values[dim], 0.0f);
    }
    EXPECT_THROW(context.kv_cache_.KeyPtr(2, 0, 0), std::out_of_range);
    EXPECT_THROW(context.kv_cache_.KeyPtr(0, 3, 0), std::out_of_range);
    EXPECT_THROW(context.kv_cache_.ValuePtr(0, 0, 2), std::out_of_range);
}

TEST(ContextTest, RequestsOwnIndependentStateAndCaches) {
    Model model;
    model.config.layers_num = 1;
    model.config.max_seq_len = 2;
    model.config.kv_heads_num = 1;
    model.config.head_dim = 2;
    Context first(model);
    const Context second(model);
    Tensor keys({1, 2}, 7.0f);
    Tensor values({1, 2}, 9.0f);
    first.kv_cache_.Write(0, 1, keys, values);
    first.pos_ = 1;
    first.prefill_len_ = 2;
    first.decode_len_ = 1;
    first.token_history_ = {10, 20};

    EXPECT_EQ(first.model_ptr_, second.model_ptr_);
    EXPECT_NE(first.kv_cache_.KeyPtr(0, 1, 0), second.kv_cache_.KeyPtr(0, 1, 0));
    EXPECT_NE(first.kv_cache_.ValuePtr(0, 1, 0), second.kv_cache_.ValuePtr(0, 1, 0));
    for (int dim = 0; dim < 2; ++dim) {
        EXPECT_FLOAT_EQ(first.kv_cache_.KeyPtr(0, 1, 0)[dim], 7.0f);
        EXPECT_FLOAT_EQ(first.kv_cache_.ValuePtr(0, 1, 0)[dim], 9.0f);
        EXPECT_FLOAT_EQ(second.kv_cache_.KeyPtr(0, 1, 0)[dim], 0.0f);
        EXPECT_FLOAT_EQ(second.kv_cache_.ValuePtr(0, 1, 0)[dim], 0.0f);
    }
    EXPECT_EQ(second.pos_, 0u);
    EXPECT_EQ(second.prefill_len_, 0u);
    EXPECT_EQ(second.decode_len_, 0u);
    EXPECT_TRUE(second.token_history_.empty());
}
