#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include "radix_tree.h"
#include "sampler.h"

enum class ChatRole{
    system,
    user,
    assistant,
};

struct ChatMessage {
    ChatRole role;
    std::string content;
    std::string RoleToString() const {
        switch (role) {
            case ChatRole::system: return "system";
            case ChatRole::user: return "user";
            case ChatRole::assistant: return "assistant";
        }
        return "unknown role";
    }
};


class ChatSession {
public:
    std::vector<ChatMessage> messages;
    std::vector<int> token_history;
    RadixTree prefix_cache;
    SamplingParams sampling_params;


    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;
    double total_time_ms = 0.0;

    ChatSession() = default;


    
    void Clear();

    void AddMessage(ChatRole chat_role, const std::string& content);

    void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);

    void SetTokenHistory(const std::vector<int>& tokens);


    void AppendToken(int token);


    void RecordPrefix(const std::vector<int>& tokens);


    std::size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};
