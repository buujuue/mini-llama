// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "batch.h"

// 前四项移植自 references/mini-llama-cpp/tests/gtest/test_batch.cc。
// 适配现有构造函数与只读接口；空批次使用空 vector 构造。
TEST(BatchTest, Single) {
    const Batch input(42, 5);
    EXPECT_EQ(input.size(), 1u);
    EXPECT_EQ(input.GetToken(0), 42);
    EXPECT_EQ(input.GetPos(0), 5u);
    EXPECT_NO_THROW(input.check());
}

TEST(BatchTest, FromTokens) {
    const std::vector<int> tokens{10, 20, 30};
    const Batch input(tokens, 0);
    EXPECT_EQ(input.size(), 3u);
    EXPECT_EQ(input.GetToken(0), 10);
    EXPECT_EQ(input.GetToken(1), 20);
    EXPECT_EQ(input.GetToken(2), 30);
    EXPECT_EQ(input.GetPos(0), 0u);
    EXPECT_EQ(input.GetPos(1), 1u);
    EXPECT_EQ(input.GetPos(2), 2u);
}

TEST(BatchTest, FromTokensWithOffset) {
    const Batch input(std::vector<int>{5, 6}, 10);
    EXPECT_EQ(input.size(), 2u);
    EXPECT_EQ(input.GetToken(0), 5);
    EXPECT_EQ(input.GetToken(1), 6);
    EXPECT_EQ(input.GetPos(0), 10u);
    EXPECT_EQ(input.GetPos(1), 11u);
    EXPECT_NO_THROW(input.check());
}

TEST(BatchTest, Empty) {
    const Batch input(std::vector<int>{}, 10);
    EXPECT_EQ(input.size(), 0u);
    EXPECT_THROW(input.GetToken(0), std::out_of_range);
    EXPECT_THROW(input.GetPos(0), std::out_of_range);
    EXPECT_THROW(input.check(), std::runtime_error);
}

TEST(BatchTest, SingleDefaultsToPositionZero) {
    const Batch input(7);
    EXPECT_EQ(input.GetToken(0), 7);
    EXPECT_EQ(input.GetPos(0), 0u);
}

TEST(BatchTest, OwnsCopyOfInputTokens) {
    std::vector<int> tokens{1, 2, 3};
    const Batch input(tokens, 4);
    tokens[0] = 99;
    tokens.clear();
    EXPECT_EQ(input.size(), 3u);
    EXPECT_EQ(input.GetToken(0), 1);
    EXPECT_EQ(input.GetToken(1), 2);
    EXPECT_EQ(input.GetToken(2), 3);
    EXPECT_EQ(input.GetPos(0), 4u);
    EXPECT_EQ(input.GetPos(1), 5u);
    EXPECT_EQ(input.GetPos(2), 6u);
}

TEST(BatchTest, AccessorsRejectOutOfRange) {
    const Batch input(std::vector<int>{10, 20}, 5);
    EXPECT_THROW(input.GetToken(-1), std::out_of_range);
    EXPECT_THROW(input.GetToken(2), std::out_of_range);
    EXPECT_THROW(input.GetPos(-1), std::out_of_range);
    EXPECT_THROW(input.GetPos(2), std::out_of_range);
}
