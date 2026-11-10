#pragma once

#include <cstddef>
#include <vector>
#include "model.h"
#include "kv_cache.h"
#include "tool.h"

struct Context{
    explicit Context(const Model& model, BackendKind backend_kind = BackendKind::Cpu)
        : model_ptr(&model), backend_kind(backend_kind) {
        const auto& c = model_ptr->config;
        kv_cache = KvCache(c.layers_num, c.max_seq_len, c.kv_heads_num, c.head_dim);
    }
    const Model* model_ptr = nullptr;
    BackendKind backend_kind = BackendKind::Cpu;
    KvCache kv_cache;
    size_t pos = 0;
    size_t prefill_len = 0;
    size_t decode_len = 0;
    std::vector<int> token_history;
};
