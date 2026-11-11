#pragma once

#include <cstddef>
#include <vector>

class Batch{
public:
    Batch() = delete;
    Batch(int token, size_t pos = 0);
    Batch(const std::vector<int>& tokens, size_t start_pos);
    void check() const;
    size_t size() const {return tokens_.size();}
    int GetToken(int i) const;
    size_t GetPos(int i) const;
    std::vector<int> GetTokens() const {return tokens_;}
private:
    std::vector<int> tokens_;
    size_t start_pos_;
};
