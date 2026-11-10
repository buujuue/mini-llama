

#pragma once

#include <string>
#include <vector>

#include "chat.h"


class PromptBuilder {
public:
    PromptBuilder() = default;

    void SetChatTemplate(const std::string& template_str);

    std::string Build(const std::vector<ChatMessage>& messages) const;

private:
    std::string chat_template_;

    std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
    std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};
