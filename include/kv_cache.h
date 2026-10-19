#pragma once

#include <cstddef>
#include "tensor.h"


class KvCache {
public:
    KvCache() = delete;
    // layer: 层数, seq_len_max: 最大token数, kv_heads_num: KV头数, head_dim: 头的向量数
    KvCache(size_t layers_num, size_t seq_len_max, size_t kv_heads_num, size_t head_dim);
    
    void Write(size_t layer, size_t pos, Tensor& k, Tensor& v);

    const float* KeyPtr(int layer, int pos, int kv_head) const;
    const float* ValuePtr(int layer, int pos, int kv_head) const;

private:
    Tensor keys_;
    Tensor valves_;
    bool is_start_ = false;

};