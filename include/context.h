#pragma once

#include <cstddef>
#include <vector>
#include "model.h"
#include "kv_cache.h"

class Context{
public:
    explicit Context(const Model& model) {
        model_ptr_ = &model;
        const auto& c = model_ptr_->config;
        kv_cache_ = KvCache(c.layers_num, c.max_seq_len, c.kv_heads_num, c.head_dim);
    }
    const Model* model_ptr_ = nullptr;
    KvCache kv_cache_;
    size_t pos_ = 0;
    size_t prefill_len_ = 0;
    size_t decode_len_ = 0;
    std::vector<int> token_history_;
};
