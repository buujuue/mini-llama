#include "forward.h"


static void CheckForwardInputs(const Context& ctx, const Model& model, int token) {
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
    if (model.token_embedding.size() < expected_size) {
        throw std::runtime_error(
            "CheckForwardInputs token embedding 数据不足：实际元素数: "
            + std::to_string(model.token_embedding.size())
            + "，至少需要 vocab_size * dim: " + std::to_string(expected_size) + "。");
    }
}

static size_t MapQHeadToKvHead(size_t q_head, size_t n_heads, size_t kv_heads_num) {
    return q_head / (n_heads / kv_heads_num);
}

// 计算单个查询头的注意力，将结果写入 attn_out 对应的行。
static void ComputeHead(const Tensor& q, size_t h, size_t pos, size_t layer,
                        const KvCache& kv_cache, size_t heads_num,
                        size_t kv_heads_num, size_t head_dim, float scale,
                        Tensor& attn_out) {
    const size_t kv_head = MapQHeadToKvHead(h, heads_num, kv_heads_num);

    Tensor scores({pos + 1}, 0.0f);
    for (size_t t = 0; t <= pos; ++t) {
        const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            dot += q.data[h * head_dim + d] * k_ptr[d];
        }
        scores.data[t] = dot * scale;
    }

    const Tensor probs = Softmax(scores);

    for (size_t d = 0; d < head_dim; ++d) {
        float val = 0.0f;
        for (size_t t = 0; t <= pos; ++t) {
            const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
            val += probs.data[t] * v_ptr[d];
        }
        attn_out.data[h * head_dim + d] = val;
    }
}

static Tensor AttentionForward(const Model& model, const Tensor& q, const Tensor& k, const Tensor& v, size_t pos, size_t layer, KvCache& kv_cache, size_t heads_num, size_t kv_heads_num, size_t head_dim, BackendKind backend){
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "AttentionForward 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "。");
    } else {
        kv_cache.Write(layer, pos, k, v);
        Tensor attn_out({heads_num, head_dim}, 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (size_t h = 0; h < heads_num; ++h) {
            ComputeHead(q, h, pos, layer, kv_cache, heads_num, kv_heads_num,
                        head_dim, scale, attn_out);
        }
        return attn_out;
    }
}

static Tensor EmbedToken(const Model& model, int token_id) {
    size_t dim = model.config.dim;
    Tensor x({dim}, 0.0f);
    std::memcpy(x.data.data(), model.token_embedding.data.data() + token_id * dim, dim * sizeof(float));
    return x;
}

static void ForwardRope(const Model& model, Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type, BackendKind backend){
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardRope 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "。");
    } else {
        Rope(q, k, pos, theta, rope_type);
    }
}

static Tensor
    ForwardRmsNorm(const Model& model, const Tensor& x, const Tensor& weight, const float eps, BackendKind backend) {
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardRmsNorm 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "。");
    } else {
        return RmsNorm(x, weight, eps);
    }
}

// gate、up 形状相同，返回 SiLU(gate) * up。
static Tensor ForwardSwiGlu(const Model& model, const Tensor& gate, const Tensor& up,
                            BackendKind backend) {
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardSwiGlu 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "。");
    } else {
        return SwiGlu(gate, up);
    }
}

static Tensor ForwardAdd(const Model& model, const Tensor& a, const Tensor& b,
                         BackendKind backend) {
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardAdd 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "。");
    }

    b.AssertShape(a.GetShape(), "ForwardAdd");
    Tensor y = a;
    for (size_t i = 0; i < a.size(); ++i) {
        y.data[i] = a.data[i] + b.data[i];
    }
    return y;
}

static void AddOptionalBias(Tensor& x, const Tensor& bias, const char* caller) {
    if (bias.size() == 0) {
        return ;
    }
    if (x.GetNumDims() != 1 || bias.GetNumDims() != 1 || x.GetShape()[0] != bias.GetShape()[0]) {
        throw std::runtime_error(
            std::string(caller) + " AddOptionalBias 要求 x 和 bias 均为长度相同的一维张量："
            "x 维数: " + std::to_string(x.GetNumDims())
            + "，x 元素数: " + std::to_string(x.size())
            + "，bias 维数: " + std::to_string(bias.GetNumDims())
            + "，bias 元素数: " + std::to_string(bias.size()) + "。");
    }

    for (int i = 0; i < x.GetShape()[0]; ++i) {
        x.data[i] += bias.data[i];
    }
}

static Tensor
    ForwardLinear(const Model& model, const std::string& name, const Tensor& x, const Tensor& weight,
                  const BackendKind backend) {
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardLinear 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "，weight: " + name + "。");
    } else {
        return Linear(x, weight);
    }
}

static void ForwardQkvProjection(const Model& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& lw, Tensor&q_flat, Tensor& k_flat, Tensor&v_flat, const BackendKind backend) {
    const std::string q_name = layer_prefix + "wq";
    const std::string k_name = layer_prefix + "wk";
    const std::string v_name = layer_prefix + "wv";
    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardQkvProjection 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend))
            + "，layer_prefix: " + layer_prefix + "。");
    } else {
        q_flat = ForwardLinear(model, q_name, h, lw.wq, backend);
        AddOptionalBias(q_flat, lw.bq, "forward_layer q");

        k_flat = ForwardLinear(model, k_name, h, lw.wk, backend);
        AddOptionalBias(k_flat, lw.bk, "forward_layer k");

        v_flat = ForwardLinear(model, v_name, h, lw.wv, backend);
        AddOptionalBias(v_flat, lw.bv, "forward_layer v");
        return;
    }
}

static Tensor FfnForward(const Model& model, const std::string& layer_prefix, const Tensor& h, const LayerWeights& lw, BackendKind backend){
    const std::string gate_name = layer_prefix + "w_gate";
    const std::string up_name = layer_prefix + "w_up";
    const std::string down_name = layer_prefix + "w_down";

    if (backend == BackendKind::Cuda) {
        throw std::logic_error(
            "FfnForward 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend)) + "。");
    }

    Tensor gate = ForwardLinear(model, gate_name, h, lw.w_gate, backend);
    Tensor up = ForwardLinear(model, up_name, h, lw.w_up, backend);
    Tensor ffn_hidden = ForwardSwiGlu(model, gate, up, backend);
    return ForwardLinear(model, down_name, ffn_hidden, lw.w_down, backend);
}

Tensor ForwardLayer(Context& ctx, const Model& model, const Tensor& x, int layer, const LayerWeights& lw,
                        const ModelConfig& config) {
    size_t dim = config.dim;
    size_t heads_num = config.heads_num;
    size_t kv_heads_num = config.kv_heads_num;
    size_t head_dim = config.head_dim;
    size_t pos = ctx.pos;
    BackendKind backend = ctx.backend_kind;

    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

    Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, config.rms_norm_eps, backend);

    Tensor q_flat;
    Tensor k_flat;
    Tensor v_flat;

    ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat, backend);

    Tensor q = q_flat.Reshape({heads_num, head_dim}, "forward_layer q");
    Tensor k = k_flat.Reshape({kv_heads_num, head_dim}, "forward_layer k");
    Tensor v = v_flat.Reshape({kv_heads_num, head_dim}, "forward_layer v");

    ForwardRope(model, q, k, pos, config.rope_theta, config.rope_type, backend);

    Tensor attn_out = AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache, heads_num,  kv_heads_num, head_dim, backend);

    Tensor attn_out_flat = attn_out.Reshape({1, heads_num * head_dim}, "forward_layer attn_out");

    Tensor attn_proj = ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo, BackendKind::Cpu);
    attn_proj.AssertShape({1, dim}, "forward_layer attn_proj");

    Tensor attn_proj_1d = attn_proj.Reshape({dim}, "forward_layer attn_proj 1d");
    Tensor x_attn = ForwardAdd(model, x, attn_proj_1d, backend);

    Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, config.rms_norm_eps, BackendKind::Cpu);
    
    Tensor ff = FfnForward(model, layer_prefix, h2, lw, BackendKind::Cpu);

    ff.AssertShape({dim}, "forward_layer ffn");

    return ForwardAdd(model, x_attn, ff, BackendKind::Cpu);
}

static Tensor ComputeLogits(const Tensor& x, const Model& model, const ModelConfig& config, BackendKind backend) {
    Tensor normed = ForwardRmsNorm(model, x, model.final_norm, config.rms_norm_eps, backend);

    Tensor logits_flat = ForwardLinear(model, "lm_head", normed, model.lm_head, backend);

    return logits_flat.Reshape({config.vocab_size}, "compute_logits");
}

Tensor ForwardToken(Context& ctx, const Model& model, int token) {
    CheckForwardInputs(ctx, model, token);
    const ModelConfig& config = model.config;
    size_t layers_num = config.layers_num;
    const BackendKind backend_kind = ctx.backend_kind;

    if (backend_kind == BackendKind::Cuda) {
        throw std::logic_error(
            "ForwardToken 非 CPU 后端尚未实现：backend: "
            + std::to_string(static_cast<int>(backend_kind)) + "。");
    } else {
        Tensor x = EmbedToken(model, token);
        for(int layer = 0; layer < layers_num; ++layer) {
            x = ForwardLayer(ctx, model, x, layer, model.layers[layer], config);
        }

        Tensor logits = ComputeLogits(x, model, config, BackendKind::Cpu);
        return logits;
    }
}

Tensor ForwardBatch(Context& ctx, const Model model, const Batch& batch){
    batch.check();
    Tensor logits;
    for (int i = 0; i < batch.size(); ++i){
        ctx.pos = batch.GetPos(i);
        logits = ForwardToken(ctx, model, batch.GetToken(i));
        ctx.token_history.push_back(batch.GetToken(i));
    }
    return logits;
}
