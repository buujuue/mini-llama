// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "prompt_builder.h"
#include "tokenizer.h"

namespace {

// 与课程 tests/gguf_tokenizer_fixture.h 中的 TinyQwen2Template 相同，
// 直接传入模板文本，不依赖尚未移植的 GGUF 读取模块。
const std::string kQwen2Template =
    "{% for message in messages %}{% if loop.first and "
    "messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a "
    "helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + "
    "message['role'] + '\n' + message['content'] + '<|im_end|>' + "
    "'\n'}}{% endfor %}{% if add_generation_prompt %}{{ "
    "'<|im_start|>assistant\n' }}{% endif %}";

}  // namespace

// 以下五项适配课程 tests/gtest/test_chat.cc 的 PromptBuilder 测试。
TEST(PromptBuilderTest, EmptyHistory) {
    const PromptBuilder builder;
    EXPECT_EQ(builder.Build({}), "Assistant:");
}

TEST(PromptBuilderTest, MultiTurn) {
    const PromptBuilder builder;
    const std::vector<ChatMessage> messages{
        {ChatRole::system, "You are helpful."},
        {ChatRole::user, "hi"},
        {ChatRole::assistant, "Hello!"},
        {ChatRole::user, "bye"},
    };
    EXPECT_EQ(builder.Build(messages),
              "System: You are helpful.\n"
              "User: hi\n"
              "Assistant: Hello!\n"
              "User: bye\n"
              "Assistant:");
}

TEST(PromptBuilderTest, IgnoresUnknownRoles) {
    const PromptBuilder builder;
    EXPECT_EQ(builder.Build({{static_cast<ChatRole>(-1), "ignored"},
                             {ChatRole::user, "hello"}}),
              "User: hello\nAssistant:");
}

TEST(PromptBuilderTest, Qwen2InjectsSystemAndPrefix) {
    PromptBuilder builder;
    builder.SetChatTemplate("qwen2");
    EXPECT_EQ(builder.Build({{ChatRole::user, "你好"}}),
              "<|im_start|>system\n"
              "You are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n"
              "你好<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, Qwen2KeepsExistingSystem) {
    PromptBuilder builder;
    builder.SetChatTemplate("qwen2");
    EXPECT_EQ(builder.Build({{ChatRole::system, "Be concise."},
                             {ChatRole::user, "hi"}}),
              "<|im_start|>system\n"
              "Be concise.<|im_end|>\n"
              "<|im_start|>user\n"
              "hi<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, Qwen2EmptyHistoryAndUnknownRoles) {
    PromptBuilder builder;
    builder.SetChatTemplate("qwen2");
    const std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>assistant\n";
    EXPECT_EQ(builder.Build({}), expected);
    EXPECT_EQ(builder.Build({{static_cast<ChatRole>(99), "ignored"}}), expected);
}

TEST(PromptBuilderTest, Qwen2PreservesMessageOrderAndContent) {
    PromptBuilder builder;
    builder.SetChatTemplate("qwen2");
    EXPECT_EQ(builder.Build({{ChatRole::user, "你好\n世界"},
                             {static_cast<ChatRole>(-1), "ignored"},
                             {ChatRole::system, ""},
                             {ChatRole::assistant, "答复"},
                             {ChatRole::user, ""}}),
              "<|im_start|>user\n你好\n世界<|im_end|>\n"
              "<|im_start|>system\n<|im_end|>\n"
              "<|im_start|>assistant\n答复<|im_end|>\n"
              "<|im_start|>user\n<|im_end|>\n"
              "<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, TemplateCanBeChangedAndReset) {
    PromptBuilder builder;
    const std::vector<ChatMessage> messages{{ChatRole::user, "hello"}};
    builder.SetChatTemplate("qwen2");
    EXPECT_NE(builder.Build(messages).find("<|im_start|>"), std::string::npos);
    builder.SetChatTemplate("{{ 'custom' }}");
    EXPECT_EQ(builder.Build(messages), "custom");
    builder.SetChatTemplate("");
    EXPECT_EQ(builder.Build(messages), "User: hello\nAssistant:");
    builder.SetChatTemplate("unknown-template");
    EXPECT_EQ(builder.Build(messages), "User: hello\nAssistant:");
}

// 适配课程 tests/gtest/test_gguf_tokenizer.cc 中的模板渲染测试。
TEST(PromptBuilderTest, JinjaRendersQwen2WithAndWithoutSystem) {
    PromptBuilder builder;
    builder.SetChatTemplate(kQwen2Template);
    PromptBuilder builtin;
    builtin.SetChatTemplate("qwen2");
    const std::vector<std::vector<ChatMessage>> histories{
        {{ChatRole::user, "Hello"}},
        {{ChatRole::system, "You are a coding assistant."},
         {ChatRole::user, "Write a hello world program."}},
        {{ChatRole::user, "你好"}, {ChatRole::assistant, "Hello!"},
         {ChatRole::user, "bye"}},
    };
    for (const auto& messages : histories) {
        EXPECT_EQ(builder.Build(messages), builtin.Build(messages));
    }
    EXPECT_EQ(builder.Build({}), "<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, JinjaAcceptsTrimMarkers) {
    PromptBuilder builder;
    builder.SetChatTemplate(
        "{%- for message in messages -%}{{- '<|im_start|>' + message['role'] + "
        "'\n' + message['content'] -}}{%- endfor -%}{{- "
        "'<|im_start|>assistant\n' -}}");
    EXPECT_EQ(builder.Build({{ChatRole::user, "Hello"}}),
              "<|im_start|>user\nHello<|im_start|>assistant\n");
}

TEST(PromptBuilderTest, JinjaMalformedTemplatesAreSafe) {
    PromptBuilder builder;
    const std::vector<std::pair<std::string, std::string>> cases{
        {"{% endif %}{{ 'ok' }}", "ok"},
        {"{% endfor %}{{ 'ok' }}", "ok"},
        {"{% if missing %}bad", ""},
        {"{% for message in messages %}bad", ""},
        {"{{ 'unterminated'", ""},
        {"{% if missing", ""},
        {"{{ messages[0][] }}", ""},
        {"{{ messages[-1]['role'] }}", ""},
        {"{{ messages[99999999999999999999999]['role'] }}", ""},
        {"{{ messages[invalid]['role'] }}", ""},
    };
    for (const auto& entry : cases) {
        SCOPED_TRACE(entry.first);
        builder.SetChatTemplate(entry.first);
        EXPECT_EQ(builder.Build({}), entry.second);
    }
    builder.SetChatTemplate("{{ messages[0][] }}");
    EXPECT_EQ(builder.Build({{ChatRole::user, "hi"}}), "");
}

TEST(PromptBuilderTest, JinjaFieldsConditionsAndLiteralContent) {
    PromptBuilder builder;
    builder.SetChatTemplate(
        "{{ messages[0][\"role\"] + ':' + messages[0]['content'] }}|"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' or message['role'] == 'user' %}"
        "{{ message[\"content\"] }}{% endif %}"
        "{% if message['role'] != 'system' and loop.first %}{{ 'first' }}"
        "{% endif %}{% endfor %}"
        "{% if missing %}bad{% endif %}"
        "{% if add_generation_prompt %}{{ '|generate' }}{% endif %}");
    EXPECT_EQ(builder.Build({{ChatRole::system, "规则"},
                             {ChatRole::user, "{{ literal }}"},
                             {ChatRole::assistant, "skipped"}}),
              "system:规则|规则{{ literal }}|generate");
}

TEST(PromptBuilderTest, JinjaAndHasHigherPrecedenceThanOr) {
    PromptBuilder builder;
    builder.SetChatTemplate(
        "{% if missing and add_generation_prompt or add_generation_prompt %}"
        "{{ 'ok' }}{% endif %}");
    EXPECT_EQ(builder.Build({}), "ok");
}

TEST(PromptBuilderTest, JinjaNestedLoopsRestoreOuterMessage) {
    PromptBuilder builder;
    builder.SetChatTemplate(
        "{% for message in messages %}{{ message['role'] + ':' }}"
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{{ ':' + message['role'] + ';' }}{% endfor %}");
    EXPECT_EQ(builder.Build({{ChatRole::user, "a"}, {ChatRole::assistant, "b"}}),
              "user:ab:user;assistant:ab:assistant;");
}

TEST(PromptBuilderTest, SessionPromptCanBeTokenizedWithoutChangingState) {
    ChatSession session;
    session.AddMessage(ChatRole::system, "Be concise.");
    session.AddMessage(ChatRole::user, "hi\nthere");
    session.SetTokenHistory({1, 2});
    session.RecordPrefix({1, 2});
    session.RecordTurn(2, 1, 3.0);
    const PromptBuilder builder;
    const std::string expected = "System: Be concise.\nUser: hi\nthere\nAssistant:";
    EXPECT_EQ(builder.Build(session.messages), expected);
    EXPECT_EQ(builder.Build(session.messages), expected);
    const AsciiTokenizer tokenizer;
    const auto tokens = tokenizer.Encode(builder.Build(session.messages));
    ASSERT_EQ(tokens.size(), expected.size() + 1);
    EXPECT_EQ(tokens.front(), tokenizer.bos_id());
    EXPECT_EQ(tokens[1], static_cast<int>('S'));
    EXPECT_EQ(tokens.back(), static_cast<int>(':'));
    ASSERT_EQ(session.messages.size(), 2u);
    EXPECT_EQ(session.messages[1].content, "hi\nthere");
    EXPECT_EQ(session.token_history, (std::vector<int>{1, 2}));
    EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3}), 2u);
    EXPECT_EQ(session.total_prompt_tokens, 2);
    EXPECT_EQ(session.total_generated_tokens, 1);
    EXPECT_DOUBLE_EQ(session.total_time_ms, 3.0);
}
