#include "batch.h"
#include <cstddef>

batch::batch(int token, size_t pos){
    tokens_.push_back(token);
    positions_.push_back(pos);
}

batch::batch(const std::vector<int>& tokens, size_t start_pos){
    tokens_ = tokens;
    positions_.reserve(tokens_.size());
    for(size_t i = 0; i < tokens_.size(); ++i){
        positions_[i] = i + start_pos;
    }
}