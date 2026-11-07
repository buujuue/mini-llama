// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "chat.h"

// 前七项适配 references/mini-llama-cpp/tests/gtest/test_chat.cc 与
// tests/test_chat.cc 的 ChatMessage / ChatSession 示例；角色使用本项目的枚举。
TEST(ChatTest, ChatMessageStoresRoleAndContent) {
    const ChatMessage message{ChatRole::user, "hello"};
    EXPECT_EQ(message.role, ChatRole::user);
    EXPECT_EQ(message.content, "hello");
    EXPECT_EQ(message.RoleToString(), "user");
}

TEST(ChatTest, ChatSessionStartsEmpty) {
    const ChatSession session;
    EXPECT_TRUE(session.messages.empty());
    EXPECT_TRUE(session.token_history.empty());
    EXPECT_TRUE(session.prefix_cache.empty());
    EXPECT_EQ(session.total_prompt_tokens, 0);
    EXPECT_EQ(session.total_generated_tokens, 0);
    EXPECT_DOUBLE_EQ(session.total_time_ms, 0.0);
}

TEST(ChatTest, ChatSessionAddMessageAppends) {
    ChatSession session;
    session.AddMessage(ChatRole::system, "You are helpful.");
    session.AddMessage(ChatRole::user, "hi");
    session.AddMessage(ChatRole::assistant, "Hello!");
    ASSERT_EQ(session.messages.size(), 3u);
    EXPECT_EQ(session.messages[0].role, ChatRole::system);
    EXPECT_EQ(session.messages[0].content, "You are helpful.");
    EXPECT_EQ(session.messages[1].role, ChatRole::user);
    EXPECT_EQ(session.messages[1].content, "hi");
    EXPECT_EQ(session.messages[2].role, ChatRole::assistant);
    EXPECT_EQ(session.messages[2].content, "Hello!");
}

TEST(ChatTest, ChatSessionClearResets) {
    ChatSession session;
    session.AddMessage(ChatRole::user, "test");
    session.SetTokenHistory({1, 2, 3});
    session.RecordPrefix({1, 2, 3});
    session.RecordTurn(10, 5, 100.0);
    session.Clear();
    EXPECT_TRUE(session.messages.empty());
    EXPECT_TRUE(session.token_history.empty());
    EXPECT_TRUE(session.prefix_cache.empty());
    EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0u);
    EXPECT_EQ(session.total_prompt_tokens, 0);
    EXPECT_EQ(session.total_generated_tokens, 0);
    EXPECT_DOUBLE_EQ(session.total_time_ms, 0.0);
}

TEST(ChatTest, ChatSessionTracksTokenHistory) {
    ChatSession session;
    session.SetTokenHistory({1, 10, 11});
    session.AppendToken(12);
    EXPECT_EQ(session.token_history, (std::vector<int>{1, 10, 11, 12}));
}

TEST(ChatTest, ChatSessionRecordsPrefixCache) {
    ChatSession session;
    session.RecordPrefix({1, 2, 3, 4});
    session.RecordPrefix({1, 2, 9});
    const ChatSession& view = session;
    EXPECT_EQ(view.LongestCachedPrefix({1, 2, 3}), 3u);
    EXPECT_EQ(view.LongestCachedPrefix({1, 2, 9, 8}), 3u);
    EXPECT_EQ(view.LongestCachedPrefix({1, 8}), 1u);
    EXPECT_EQ(view.LongestCachedPrefix({8}), 0u);
    EXPECT_EQ(view.LongestCachedPrefix({}), 0u);
}

TEST(ChatTest, ChatSessionRecordTurnAccumulates) {
    ChatSession session;
    session.RecordTurn(10, 5, 100.0);
    session.RecordTurn(20, 8, 200.0);
    EXPECT_EQ(session.total_prompt_tokens, 30);
    EXPECT_EQ(session.total_generated_tokens, 13);
    EXPECT_DOUBLE_EQ(session.total_time_ms, 300.0);
}

TEST(ChatTest, ChatMessageFormatsAllRoles) {
    EXPECT_EQ((ChatMessage{ChatRole::system, ""}).RoleToString(), "system");
    EXPECT_EQ((ChatMessage{ChatRole::user, ""}).RoleToString(), "user");
    EXPECT_EQ((ChatMessage{ChatRole::assistant, ""}).RoleToString(), "assistant");
    EXPECT_EQ((ChatMessage{static_cast<ChatRole>(-1), ""}).RoleToString(), "unknown role");
}

TEST(ChatTest, ChatSessionStoresEnumRoleDirectly) {
    ChatSession session;
    session.AddMessage(ChatRole::user, "hi");
    const auto unknown_role = static_cast<ChatRole>(-1);
    session.AddMessage(unknown_role, "unknown");
    ASSERT_EQ(session.messages.size(), 2u);
    EXPECT_EQ(session.messages[0].role, ChatRole::user);
    EXPECT_EQ(session.messages[0].content, "hi");
    EXPECT_EQ(session.messages[1].role, unknown_role);
    EXPECT_EQ(session.messages[1].content, "unknown");
}

TEST(ChatTest, ChatSessionClearKeepsSamplingParamsAndAllowsReuse) {
    ChatSession session;
    session.sampling_params.temperature = 0.7f;
    session.sampling_params.top_k = 40;
    session.sampling_params.seed = 42;
    session.AddMessage(ChatRole::user, "old");
    session.RecordPrefix({1, 2, 3});
    session.Clear();
    session.Clear();
    EXPECT_FLOAT_EQ(session.sampling_params.temperature, 0.7f);
    EXPECT_EQ(session.sampling_params.top_k, 40);
    EXPECT_EQ(session.sampling_params.seed, 42u);

    session.AddMessage(ChatRole::user, "new");
    session.AppendToken(9);
    session.RecordPrefix({9});
    session.RecordTurn(1, 2, 3.5);
    ASSERT_EQ(session.messages.size(), 1u);
    EXPECT_EQ(session.messages[0].content, "new");
    EXPECT_EQ(session.token_history, (std::vector<int>{9}));
    EXPECT_EQ(session.LongestCachedPrefix({9, 10}), 1u);
    EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0u);
    EXPECT_EQ(session.total_prompt_tokens, 1);
    EXPECT_EQ(session.total_generated_tokens, 2);
    EXPECT_DOUBLE_EQ(session.total_time_ms, 3.5);
}

TEST(ChatTest, ChatSessionReplacesTokenHistoryWithOwnedCopy) {
    ChatSession session;
    session.SetTokenHistory({1, 2, 3});
    std::vector<int> replacement{4, 5};
    session.SetTokenHistory(replacement);
    replacement[0] = 99;
    EXPECT_EQ(session.token_history, (std::vector<int>{4, 5}));
    session.SetTokenHistory({});
    EXPECT_TRUE(session.token_history.empty());
    session.AppendToken(6);
    EXPECT_EQ(session.token_history, (std::vector<int>{6}));
}

TEST(ChatTest, ChatSessionKeepsSessionsIndependent) {
    ChatSession first;
    ChatSession second;
    first.AddMessage(ChatRole::user, "first");
    first.SetTokenHistory({1, 2});
    first.RecordPrefix({1, 2});
    first.RecordTurn(2, 1, 3.0);
    first.sampling_params.top_k = 5;
    EXPECT_TRUE(second.messages.empty());
    EXPECT_TRUE(second.token_history.empty());
    EXPECT_EQ(second.LongestCachedPrefix({1, 2}), 0u);
    EXPECT_EQ(second.total_prompt_tokens, 0);
    EXPECT_EQ(second.total_generated_tokens, 0);
    EXPECT_DOUBLE_EQ(second.total_time_ms, 0.0);
    EXPECT_EQ(second.sampling_params.top_k, 0);

    second.RecordPrefix({1, 2, 3});
    first.Clear();
    EXPECT_EQ(second.LongestCachedPrefix({1, 2, 3}), 3u);
}
