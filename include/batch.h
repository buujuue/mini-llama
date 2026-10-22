#pragma once

#include <cstddef>
#include <vector>

class batch{
public:
    batch() = delete;
    batch(int token, size_t pos = 0);
    batch(const std::vector<int>& tokens, size_t start_pos);
    size_t size() const {return tokens_.size();}
    const std::vector<int>& GetTokens() const {return tokens_;}
    const std::vector<size_t>& GetPositions() const {return positions_;}
private:
    std::vector<int> tokens_;
    std::vector<size_t> positions_;
};
