// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "batch.h"

// 前四项移植自 references/mini-llama-cpp/tests/gtest/test_batch.cc。
// 适配现有构造函数与只读接口；空批次使用空 vector 构造。
TEST(BatchTest, Single) {
    const batch input(42, 5);
    EXPECT_EQ(input.size(), 1u);
    EXPECT_EQ(input.GetTokens(), (std::vector<int>{42}));
    EXPECT_EQ(input.GetPositions(), (std::vector<size_t>{5}));
}

TEST(BatchTest, FromTokens) {
    const std::vector<int> tokens{10, 20, 30};
    const batch input(tokens, 0);
    EXPECT_EQ(input.size(), 3u);
    EXPECT_EQ(input.GetTokens(), tokens);
    EXPECT_EQ(input.GetPositions(), (std::vector<size_t>{0, 1, 2}));
}

TEST(BatchTest, FromTokensWithOffset) {
    const batch input(std::vector<int>{5, 6}, 10);
    EXPECT_EQ(input.size(), 2u);
    EXPECT_EQ(input.GetTokens(), (std::vector<int>{5, 6}));
    EXPECT_EQ(input.GetPositions(), (std::vector<size_t>{10, 11}));
}

TEST(BatchTest, Empty) {
    const batch input(std::vector<int>{}, 10);
    EXPECT_EQ(input.size(), 0u);
    EXPECT_TRUE(input.GetTokens().empty());
    EXPECT_TRUE(input.GetPositions().empty());
}

TEST(BatchTest, SingleDefaultsToPositionZero) {
    const batch input(7);
    EXPECT_EQ(input.GetTokens(), (std::vector<int>{7}));
    EXPECT_EQ(input.GetPositions(), (std::vector<size_t>{0}));
}

TEST(BatchTest, OwnsCopyOfInputTokens) {
    std::vector<int> tokens{1, 2, 3};
    const batch input(tokens, 4);
    tokens[0] = 99;
    tokens.clear();
    EXPECT_EQ(input.size(), 3u);
    EXPECT_EQ(input.GetTokens(), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(input.GetPositions(), (std::vector<size_t>{4, 5, 6}));
}
