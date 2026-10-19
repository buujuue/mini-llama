#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tensor.h"
#include "tool.h"

// Forward 阶段的结构占位；权重初始化与加载留到“模型格式与 Loader”章节。
struct ModelConfig {
    size_t vocab_size = 128;
    size_t dim = 32;
    size_t hidden_dim = 86;
    size_t layers_num = 2;
    size_t head_num = 4;
    size_t kv_heads_num = 4;
    size_t head_dim = 8;
    size_t max_seq_len = 128;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    RopeType rope_type = RopeType::Normal;
};

struct LayerWeights {
    Tensor attention_norm; // [dim]
    Tensor wq;             // [head_num * head_dim, dim]
    Tensor wk;             // [kv_heads_num * head_dim, dim]
    Tensor wv;             // [kv_heads_num * head_dim, dim]
    Tensor bq;             // 可选 [head_num * head_dim]
    Tensor bk;             // 可选 [kv_heads_num * head_dim]
    Tensor bv;             // 可选 [kv_heads_num * head_dim]
    Tensor wo;             // [dim, head_num * head_dim]

    Tensor ffn_norm;       // [dim]
    Tensor w_gate;         // [hidden_dim, dim]
    Tensor w_up;           // [hidden_dim, dim]
    Tensor w_down;         // [dim, hidden_dim]
};

struct MiniLlamaModel {
    ModelConfig config;
    bool loaded = false;
    std::string load_error;

    // 形状注释表示后续需要填入的权重，当前 Tensor 和 layers 均保持为空。
    Tensor token_embedding;           // [vocab_size, dim]
    std::vector<LayerWeights> layers; // 数量为 layers_num
    Tensor final_norm;                // [dim]
    Tensor lm_head;                   // [vocab_size, dim]
};
