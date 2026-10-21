// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>
#include <utility>

#include "radix_tree.h"

// 以下 8 项基础测试移植自 references/mini-llama-cpp/tests/test_radix_tree.cc。
TEST(RadixTreeTest, EmptyMiss) {
    const RadixTree tree;
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 0u);
}

TEST(RadixTreeTest, ExactMatch) {
    RadixTree tree;
    tree.Insert({1, 2, 3});
    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
}

TEST(RadixTreeTest, QueryIsStoredPrefix) {
    RadixTree tree;
    tree.Insert({1, 2, 3, 4});
    EXPECT_EQ(tree.LongestPrefix({1, 2}), 2u);
}

TEST(RadixTreeTest, StoredSequenceIsQueryPrefix) {
    RadixTree tree;
    tree.Insert({1, 2});
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4}), 2u);
}

TEST(RadixTreeTest, SplitOnBranch) {
    RadixTree tree;
    tree.Insert({1, 2, 3, 4});
    tree.Insert({1, 2, 9});
    EXPECT_EQ(tree.size(), 2u);
    EXPECT_EQ(tree.LongestPrefix({1, 8}), 1u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 9, 8}), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 8}), 2u);
}

TEST(RadixTreeTest, SplitWithInsertedMiddlePrefix) {
    RadixTree tree;
    tree.Insert({1, 2, 3, 4, 5});
    tree.Insert({1, 2, 3});
    tree.Insert({1, 2, 3, 9});
    EXPECT_EQ(tree.size(), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4}), 4u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 9, 8}), 4u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 8}), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 8}), 2u);
}

TEST(RadixTreeTest, DuplicateInsertStable) {
    RadixTree tree;
    tree.Insert({4, 5, 6});
    tree.Insert({4, 5, 6});
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.LongestPrefix({4, 5, 6, 7}), 3u);
}

TEST(RadixTreeTest, Clear) {
    RadixTree tree;
    tree.Insert({1, 2, 3});
    tree.Clear();
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 0u);

    tree.Clear();
    tree.Insert({1, 9});
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 1u);
    EXPECT_EQ(tree.LongestPrefix({1, 9}), 2u);
}

TEST(RadixTreeTest, EmptyQuery) {
    RadixTree tree;
    EXPECT_EQ(tree.LongestPrefix({}), 0u);
    tree.Insert({1, 2});
    EXPECT_EQ(tree.LongestPrefix({}), 0u);
    EXPECT_EQ(tree.size(), 1u);
}

TEST(RadixTreeTest, EmptySequenceCountsOnce) {
    RadixTree tree;
    tree.Insert({});
    tree.Insert({});
    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.size(), 1u);
    EXPECT_EQ(tree.LongestPrefix({}), 0u);
    EXPECT_EQ(tree.LongestPrefix({1}), 0u);

    tree.Insert({1, 2});
    tree.Insert({});
    EXPECT_EQ(tree.size(), 2u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 2u);

    tree.Clear();
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);
    tree.Insert({});
    EXPECT_EQ(tree.size(), 1u);
}

TEST(RadixTreeTest, IndependentBranchesAndSignedTokens) {
    RadixTree tree;
    const int min_token = std::numeric_limits<int>::min();
    const int max_token = std::numeric_limits<int>::max();
    tree.Insert({0, 1, 2});
    tree.Insert({-1, 2, 3});
    tree.Insert({min_token, max_token});
    tree.Insert({max_token, 0});

    EXPECT_EQ(tree.size(), 4u);
    EXPECT_EQ(tree.LongestPrefix({0, 1, 9}), 2u);
    EXPECT_EQ(tree.LongestPrefix({-1, 2, 3, 4}), 3u);
    EXPECT_EQ(tree.LongestPrefix({min_token, max_token}), 2u);
    EXPECT_EQ(tree.LongestPrefix({max_token, 0}), 2u);
    EXPECT_EQ(tree.LongestPrefix({42}), 0u);
}

TEST(RadixTreeTest, InsertExistingBranchPrefixCountsOnce) {
    RadixTree tree;
    tree.Insert({1, 2, 3});
    tree.Insert({1, 2, 4});
    tree.Insert({1, 2});
    tree.Insert({1, 2});
    tree.Insert({1, 2, 3});

    EXPECT_EQ(tree.size(), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 4}), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 5}), 2u);
}

TEST(RadixTreeTest, InsertExtensionOfExistingSequence) {
    RadixTree tree;
    tree.Insert({1, 2});
    tree.Insert({1, 2, 3, 4});
    tree.Insert({1, 2, 3, 5});
    tree.Insert({1, 2});

    EXPECT_EQ(tree.size(), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4, 6}), 4u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 5}), 4u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 9}), 2u);
}

TEST(RadixTreeTest, MoveConstructionTransfersSequences) {
    RadixTree source;
    source.Insert({});
    source.Insert({1, 2, 3});
    source.Insert({1, 2, 4});

    RadixTree tree(std::move(source));
    EXPECT_EQ(tree.size(), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 5}), 3u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 4}), 3u);
    tree.Insert({});
    EXPECT_EQ(tree.size(), 3u);

    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.size(), 0u);
    EXPECT_EQ(source.LongestPrefix({1, 2, 3}), 0u);
    source.Insert({});
    source.Insert({9});
    EXPECT_EQ(source.size(), 2u);
    EXPECT_EQ(source.LongestPrefix({9}), 1u);
    EXPECT_EQ(tree.LongestPrefix({9}), 0u);
}

TEST(RadixTreeTest, MoveAssignmentReplacesSequences) {
    RadixTree source;
    source.Insert({});
    source.Insert({4, 5});
    RadixTree tree;
    tree.Insert({1, 2, 3});

    tree = std::move(source);
    EXPECT_EQ(tree.size(), 2u);
    EXPECT_EQ(tree.LongestPrefix({4, 5, 6}), 2u);
    EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 0u);
    tree.Insert({});
    EXPECT_EQ(tree.size(), 2u);

    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.size(), 0u);
    EXPECT_EQ(source.LongestPrefix({4, 5}), 0u);
    source.Insert({});
    EXPECT_EQ(source.size(), 1u);

    RadixTree empty;
    tree = std::move(empty);
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.size(), 0u);
    EXPECT_EQ(tree.LongestPrefix({4, 5}), 0u);
}
