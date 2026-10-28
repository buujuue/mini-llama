// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "forward.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include "context.h"
#include "model.h"
#include "tensor.h"
#include "tool.h"

void CheckForwardInputs(const Context& ctx, const Model& model, int token) {
    const ModelConfig& c = model.config;
    if (!model.loaded) {
        throw std::runtime_error("CheckForwardInputs 要求模型已加载：loaded: false。");
    }
    if (token < 0 || static_cast<size_t>(token) >= c.vocab_size) {
        throw std::out_of_range(
            "CheckForwardInputs token 越界：token: " + std::to_string(token)
            + "，有效范围为 [0, " + std::to_string(c.vocab_size) + ")。");
    }
    if (ctx.pos >= c.max_seq_len) {
        throw std::out_of_range(
            "CheckForwardInputs 位置越界：pos_: " + std::to_string(ctx.pos)
            + "，有效范围为 [0, " + std::to_string(c.max_seq_len) + ")。");
    }
    if (model.layers.size() != c.layers_num) {
        throw std::runtime_error(
            "CheckForwardInputs 模型层数与配置不匹配：实际层数: "
            + std::to_string(model.layers.size())
            + "，配置 layers_num: " + std::to_string(c.layers_num) + "。");
    }
    
    if (c.vocab_size != 0
        && c.dim > std::numeric_limits<size_t>::max() / c.vocab_size) {
        throw std::runtime_error(
            "CheckForwardInputs token embedding 元素数溢出：vocab_size: "
            + std::to_string(c.vocab_size) + "，dim: " + std::to_string(c.dim) + "。");
    }
    const size_t expected_size = c.vocab_size * c.dim;
    if (model.token_embedding.GetDataSize() < expected_size) {
        throw std::runtime_error(
            "CheckForwardInputs token embedding 数据不足：实际元素数: "
            + std::to_string(model.token_embedding.GetDataSize())
            + "，至少需要 vocab_size * dim: " + std::to_string(expected_size) + "。");
    }
}

Tensor EmbedToken(const Model& model, int token_id) {
    size_t dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    std::memcpy(x.Data(), model.token_embedding.Data() + token_id * dim, dim * sizeof(float));
    return x;
}

Tensor ForwardToken(Context& ctx, const Model& model, int token){
    CheckForwardInputs(ctx, model, token);
    const ModelConfig& config = model.config;
    size_t layers_num = config.layers_num;
    const BackendKind backend_kind = ctx.backend_kind;

    if (backend_kind == BackendKind::Cpu){
         Tensor x = EmbedToken(model, token);

    } else {
        //报尚未实现的错
    }

}

