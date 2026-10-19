// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

#include "kv_cache.h"

// 移植自 references/mini-llama-cpp/tests/gtest/test_kv_cache.cc。
// 通过公开指针检查私有缓存的布局与初始值，保留参考测试要求的异常类型。
TEST(KvCacheTest, ConstructsExpectedStorageShape) {
    const KvCache cache(2, 8, 3, 4);
    const float* keys_begin = cache.KeyPtr(0, 0, 0);
    const float* values_begin = cache.ValuePtr(0, 0, 0);
    ASSERT_NE(keys_begin, nullptr);
    ASSERT_NE(values_begin, nullptr);

    for (int layer = 0; layer < 2; ++layer) {
        for (int pos = 0; pos < 8; ++pos) {
            for (int head = 0; head < 3; ++head) {
                SCOPED_TRACE(testing::Message()
                             << "layer=" << layer << ", pos=" << pos
                             << ", head=" << head);
                const std::size_t offset = ((layer * 8 + pos) * 3 + head) * 4;
                const float* keys = cache.KeyPtr(layer, pos, head);
                const float* values = cache.ValuePtr(layer, pos, head);
                ASSERT_EQ(keys, keys_begin + offset);
                ASSERT_EQ(values, values_begin + offset);
                for (int dim = 0; dim < 4; ++dim) {
                    EXPECT_NEAR(keys[dim], 0.0f, 1e-6f);
                    EXPECT_NEAR(values[dim], 0.0f, 1e-6f);
                }
            }
        }
    }

    EXPECT_THROW(cache.KeyPtr(2, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 8, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 0, 3), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(2, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(0, 8, 0), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(0, 0, 3), std::out_of_range);
}

TEST(KvCacheTest, WriteStoresKeyAndValueByLayerPositionAndHead) {
    KvCache cache(2, 4, 2, 3);
    Tensor k({2, 3}, 0.0f);
    Tensor v({2, 3}, 0.0f);

    for (int i = 0; i < 6; ++i) {
        k.Data()[i] = static_cast<float>(i + 1);
        v.Data()[i] = static_cast<float>(100 + i);
    }

    cache.Write(1, 2, k, v);

    const float* key_head_0 = cache.KeyPtr(1, 2, 0);
    const float* key_head_1 = cache.KeyPtr(1, 2, 1);
    const float* value_head_0 = cache.ValuePtr(1, 2, 0);
    const float* value_head_1 = cache.ValuePtr(1, 2, 1);

    EXPECT_NEAR(key_head_0[0], 1.0f, 1e-6f);
    EXPECT_NEAR(key_head_0[1], 2.0f, 1e-6f);
    EXPECT_NEAR(key_head_0[2], 3.0f, 1e-6f);
    EXPECT_NEAR(key_head_1[0], 4.0f, 1e-6f);
    EXPECT_NEAR(key_head_1[1], 5.0f, 1e-6f);
    EXPECT_NEAR(key_head_1[2], 6.0f, 1e-6f);

    EXPECT_NEAR(value_head_0[0], 100.0f, 1e-6f);
    EXPECT_NEAR(value_head_0[1], 101.0f, 1e-6f);
    EXPECT_NEAR(value_head_0[2], 102.0f, 1e-6f);
    EXPECT_NEAR(value_head_1[0], 103.0f, 1e-6f);
    EXPECT_NEAR(value_head_1[1], 104.0f, 1e-6f);
    EXPECT_NEAR(value_head_1[2], 105.0f, 1e-6f);
}

TEST(KvCacheTest, WritePreservesOtherPositions) {
    KvCache cache(1, 3, 1, 2);
    Tensor k({1, 2}, 0.0f);
    Tensor v({1, 2}, 0.0f);
    k.Data()[0] = 7.0f;
    k.Data()[1] = 8.0f;
    v.Data()[0] = 9.0f;
    v.Data()[1] = 10.0f;

    cache.Write(0, 1, k, v);

    const float* untouched_key = cache.KeyPtr(0, 0, 0);
    const float* untouched_value = cache.ValuePtr(0, 2, 0);
    EXPECT_NEAR(untouched_key[0], 0.0f, 1e-6f);
    EXPECT_NEAR(untouched_key[1], 0.0f, 1e-6f);
    EXPECT_NEAR(untouched_value[0], 0.0f, 1e-6f);
    EXPECT_NEAR(untouched_value[1], 0.0f, 1e-6f);
}

TEST(KvCacheTest, WriteRejectsBadIndices) {
    KvCache cache(1, 2, 1, 2);
    Tensor k({1, 2}, 0.0f);
    Tensor v({1, 2}, 0.0f);

    EXPECT_THROW(cache.Write(-1, 0, k, v), std::out_of_range);
    EXPECT_THROW(cache.Write(1, 0, k, v), std::out_of_range);
    EXPECT_THROW(cache.Write(0, -1, k, v), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 2, k, v), std::out_of_range);
}

TEST(KvCacheTest, WriteRejectsBadTensorShapes) {
    KvCache cache(1, 2, 1, 2);
    Tensor good({1, 2}, 0.0f);
    Tensor rank_3({1, 1, 2}, 0.0f);
    Tensor wrong_heads({2, 2}, 0.0f);
    Tensor wrong_dim({1, 3}, 0.0f);

    EXPECT_THROW(cache.Write(0, 0, rank_3, good), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, good, wrong_dim), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, wrong_heads, wrong_heads), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, wrong_dim, wrong_dim), std::out_of_range);
}

TEST(KvCacheTest, PtrAccessRejectsBadIndices) {
    KvCache cache(1, 2, 1, 2);

    EXPECT_THROW(cache.KeyPtr(-1, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, -1, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 0, -1), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(1, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 2, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 0, 1), std::out_of_range);

    EXPECT_THROW(cache.ValuePtr(-1, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(0, -1, 0), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(0, 0, -1), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(1, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(0, 2, 0), std::out_of_range);
    EXPECT_THROW(cache.ValuePtr(0, 0, 1), std::out_of_range);
}
