// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>

#include "terminal.h"

namespace {

class CountingBuffer : public std::stringbuf {
public:
    int sync_count = 0;

protected:
    int sync() override {
        ++sync_count;
        return std::stringbuf::sync();
    }
};

// 重定向标准流，验证交互行为且不等待真实终端输入。
class TerminalTest : public ::testing::Test {
protected:
    void SetUp() override {
        saved_format_.copyfmt(std::cout);
        saved_input_state_ = std::cin.rdstate();
        saved_output_state_ = std::cout.rdstate();
        saved_input_buffer_ = std::cin.rdbuf(input_.rdbuf());
        saved_output_buffer_ = std::cout.rdbuf(&output_);
        std::cout.copyfmt(input_);
        std::cout.imbue(std::locale::classic());
    }

    void TearDown() override {
        std::cin.rdbuf(saved_input_buffer_);
        std::cin.clear(saved_input_state_);
        std::cout.rdbuf(saved_output_buffer_);
        std::cout.copyfmt(saved_format_);
        std::cout.clear(saved_output_state_);
    }

    Terminal terminal_;
    std::istringstream input_;
    CountingBuffer output_;

private:
    std::ios saved_format_{nullptr};
    std::ios::iostate saved_input_state_{};
    std::ios::iostate saved_output_state_{};
    std::streambuf* saved_input_buffer_ = nullptr;
    std::streambuf* saved_output_buffer_ = nullptr;
};

// 前七项适配上游 tests/gtest/test_chat.cc 的 Terminal 示例，补充输出断言。
TEST_F(TerminalTest, PrintUserPrompt) {
    terminal_.PrintUserPrompt();
    EXPECT_EQ(output_.str(), "mini-llama > ");
    EXPECT_EQ(output_.sync_count, 1);
}

TEST_F(TerminalTest, PrintAssistantPrefix) {
    terminal_.PrintAssistantPrefix();
    EXPECT_EQ(output_.str(), "assistant  > ");
    EXPECT_EQ(output_.sync_count, 1);
}

TEST_F(TerminalTest, PrintHelp) {
    terminal_.PrintHelp();
    EXPECT_EQ(output_.str(),
              "Commands:\n"
              "  /help    Show this help message\n"
              "  /clear   Clear chat history and context\n"
              "  /stats   Show session statistics\n"
              "  /params  Show current sampling parameters\n"
              "  /exit    Exit chat\n");
}

TEST_F(TerminalTest, PrintStats) {
    ChatSession session;
    session.AddMessage(ChatRole::user, "hello");
    session.AddMessage(ChatRole::assistant, "hi");
    session.SetTokenHistory({1, 10, 11, 12});
    session.RecordTurn(10, 5, 250.0);
    terminal_.PrintStats(session);
    EXPECT_EQ(output_.str(),
              "Session stats:\n"
              "  messages:           2\n"
              "  context tokens:     4\n"
              "  total prompt tokens: 10\n"
              "  total generated:     5\n"
              "  total time:          250.00 ms\n"
              "  tokens/s:            20.00\n");
    EXPECT_EQ(session.messages.size(), 2u);
    EXPECT_EQ(session.total_generated_tokens, 5);
    EXPECT_DOUBLE_EQ(session.total_time_ms, 250.0);
}

TEST_F(TerminalTest, PrintParams) {
    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 40;
    params.seed = 42;
    terminal_.PrintParams(params);
    EXPECT_EQ(output_.str(),
              "Sampling params:\n"
              "  temperature: 0.7\n"
              "  top_k:       40\n"
              "  seed:        42\n");
}

TEST_F(TerminalTest, PrintMessage) {
    terminal_.PrintMessage("test message / 测试消息");
    terminal_.PrintMessage("");
    EXPECT_EQ(output_.str(), "test message / 测试消息\n\n");
}

TEST_F(TerminalTest, NewLine) {
    terminal_.NewLine();
    EXPECT_EQ(output_.str(), "\n");
}

TEST_F(TerminalTest, PrintTokenTextDoesNotAddNewlineOrFlush) {
    terminal_.PrintTokenText("你");
    terminal_.PrintTokenText("");
    terminal_.PrintTokenText("好");
    EXPECT_EQ(output_.str(), "你好");
    EXPECT_EQ(output_.sync_count, 0);
}

TEST_F(TerminalTest, FlushSyncsOutput) {
    terminal_.PrintTokenText("token");
    terminal_.Flush();
    EXPECT_EQ(output_.str(), "token");
    EXPECT_EQ(output_.sync_count, 1);
}

TEST_F(TerminalTest, ReadLinePreservesWhitespaceAndUtf8) {
    input_.str("  hello world  \n你好\n");
    EXPECT_EQ(terminal_.ReadLine(), "  hello world  ");
    EXPECT_EQ(terminal_.ReadLine(), "你好");
    EXPECT_EQ(terminal_.ReadLine(), "");
    EXPECT_TRUE(std::cin.eof());
}

TEST_F(TerminalTest, ReadLineContinuesAfterBlankLine) {
    input_.str("\nnext\n");
    EXPECT_EQ(terminal_.ReadLine(), "");
    EXPECT_FALSE(std::cin.eof());
    EXPECT_EQ(terminal_.ReadLine(), "next");
}

TEST_F(TerminalTest, ReadLineHandlesFinalLineAndRepeatedEof) {
    input_.str("last line");
    EXPECT_EQ(terminal_.ReadLine(), "last line");
    EXPECT_EQ(terminal_.ReadLine(), "");
    EXPECT_EQ(terminal_.ReadLine(), "");
    EXPECT_TRUE(std::cin.eof());
}

TEST_F(TerminalTest, PrintStatsWithoutTimeShowsUnavailableRate) {
    const ChatSession session;
    terminal_.PrintStats(session);
    EXPECT_EQ(output_.str(),
              "Session stats:\n"
              "  messages:           0\n"
              "  context tokens:     0\n"
              "  total prompt tokens: 0\n"
              "  total generated:     0\n"
              "  total time:          0.00 ms\n"
              "  tokens/s:            n/a\n");
}

TEST_F(TerminalTest, NumericReportsPreserveOutputFormatting) {
    std::cout << std::scientific << std::setprecision(9);
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();
    terminal_.PrintStats(ChatSession{});
    EXPECT_EQ(std::cout.flags(), flags);
    EXPECT_EQ(std::cout.precision(), precision);

    SamplingParams params;
    params.temperature = 0.7654321f;
    params.top_k = 40;
    params.seed = 42;
    terminal_.PrintParams(params);
    EXPECT_NE(output_.str().find("  temperature: 0.765432\n"), std::string::npos);
    EXPECT_NE(output_.str().find("  top_k:       40\n"), std::string::npos);
    EXPECT_NE(output_.str().find("  seed:        42\n"), std::string::npos);
    EXPECT_EQ(std::cout.flags(), flags);
    EXPECT_EQ(std::cout.precision(), precision);
}

}  // namespace
