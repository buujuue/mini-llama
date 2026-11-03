#include "batch.h"
#include <cstddef>
#include <stdexcept>
#include <string>

Batch::Batch(int token, size_t pos)
    : tokens_{token}, start_pos_(pos) {}

Batch::Batch(const std::vector<int>& tokens, size_t start_pos)
    : tokens_(tokens), start_pos_(start_pos) {}

int Batch::GetToken(int i) const{
    if (i < 0 || static_cast<size_t>(i) >= tokens_.size()) {
        throw std::out_of_range(
            "Batch::GetToken 下标越界：i: " + std::to_string(i)
            + "，tokens 数量: " + std::to_string(tokens_.size()) + "。");
    }
    return tokens_[static_cast<size_t>(i)];
}

size_t Batch::GetPos(int i) const{
    if (i < 0 || static_cast<size_t>(i) >= tokens_.size()) {
        throw std::out_of_range(
            "Batch::GetPos 下标越界：i: " + std::to_string(i)
            + "，tokens 数量: " + std::to_string(tokens_.size()) + "。");
    }
    return start_pos_ + static_cast<size_t>(i);
}

void Batch::check() const{
    if (tokens_.empty()) {
        throw std::runtime_error(
            "Batch::check 批次不能为空：tokens 数量: " + std::to_string(tokens_.size())
            + "，start_pos: " + std::to_string(start_pos_) + "。");
    }
}
