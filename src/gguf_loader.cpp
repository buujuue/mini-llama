
#include "gguf_loader.h"
#include "gguf.h"
#include "model.h"
#include "tensor.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

std::string MapTensorName(const std::string& gguf_name) {
    if (gguf_name == "token_embd.weight") {
        return "token_embedding";
    }
    if (gguf_name == "output_norm.weight") {
        return "final_norm";
    }
    if (gguf_name == "output.weight") {
        return "lm_head";
    }

    // 解析 blk.<层号>.<操作名>.weight 或 blk.<层号>.<操作名>.bias。
    if (gguf_name.rfind("blk.", 0) == 0) {
        size_t dot1 = gguf_name.find('.', 4);
        if (dot1 == std::string::npos) {
            return "";
        }
        std::string layer_str = gguf_name.substr(4, dot1 - 4);
        size_t dot2 = gguf_name.find('.', dot1 + 1);
        if (dot2 == std::string::npos) {
            return "";
        }
        std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
        std::string suffix = gguf_name.substr(dot2 + 1);
        std::string mapped_op;
        if (suffix == "bias") {
            if (op == "attn_q") {
                mapped_op = "bq";
            } else if (op == "attn_k") {
                mapped_op = "bk";
            } else if (op == "attn_v") {
                mapped_op = "bv";
            } else {
                return "";
            }
        } else if (suffix != "weight") {
            return "";
        } else if (op == "attn_norm") {
            mapped_op = "attention_norm";
        } else if (op == "attn_q") {
            mapped_op = "wq";
        } else if (op == "attn_k") {
            mapped_op = "wk";
        } else if (op == "attn_v") {
            mapped_op = "wv";
        } else if (op == "attn_output") {
            mapped_op = "wo";
        } else if (op == "ffn_norm") {
            mapped_op = "ffn_norm";
        } else if (op == "ffn_gate") {
            mapped_op = "w_gate";
        } else if (op == "ffn_up") {
            mapped_op = "w_up";
        } else if (op == "ffn_down") {
            mapped_op = "w_down";
        } else {
            return "";
        }

        return "layers." + layer_str + "." + mapped_op;
    }

    return "";
}

ModelConfig BuildConfig(const std::string& path) {
    ModelConfig config;

    // 当前项目使用 size_t 保存维度，转换前拒绝负数和超出范围的值。
    auto get_size = [&](const std::string& key, size_t fallback) -> size_t {
        int64_t value = 0;
        if (GgufGetMetadataInt(path, key, value)) {
            if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("Invalid model config from GGUF: " + key
                                         + " is outside size_t range");
            }
            return static_cast<size_t>(value);
        }
        return fallback;
    };

    auto get_f32 = [&](const std::string& key, float fallback) -> float {
        double value = 0.0;
        if (GgufGetMetadataFloat(path, key, value)) {
            return static_cast<float>(value);
        }
        return fallback;
    };

    std::string arch;
    if (GgufGetMetadataString(path, "general.architecture", arch) && arch == "qwen2") {
        config.rope_type = RopeType::NeoX;
    }

    // 优先使用 qwen2.*，缺失时回退到 llama.* 和示例中的默认值。
    config.layers_num = get_size("qwen2.block_count", get_size("llama.block_count", 0));
    config.dim = get_size("qwen2.embedding_length", get_size("llama.embedding_length", 0));
    config.hidden_dim = get_size("qwen2.feed_forward_length",
                                 get_size("llama.feed_forward_length", 0));
    config.heads_num = get_size("qwen2.attention.head_count",
                                get_size("llama.attention.head_count", 0));
    config.kv_heads_num = get_size("qwen2.attention.head_count_kv",
                                   get_size("llama.attention.head_count_kv", 0));
    config.max_seq_len = get_size("qwen2.context_length", get_size("llama.context_length", 0));
    config.rope_theta = get_f32("qwen2.rope.freq_base",
                                get_f32("llama.rope.freq_base", 10000.0f));
    config.rms_norm_eps = get_f32("qwen2.attention.layer_norm_rms_epsilon",
                                  get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

    std::vector<std::string> tokens;
    if (GgufGetMetadataStringArray(path, "tokenizer.ggml.tokens", tokens)) {
        config.vocab_size = tokens.size();
    }

    if (config.layers_num == 0 || config.dim == 0 || config.hidden_dim == 0 ||
        config.heads_num == 0 || config.kv_heads_num == 0 ||
        config.max_seq_len == 0 || config.vocab_size == 0) {
        throw std::runtime_error(
            "Invalid model config from GGUF: dim=" + std::to_string(config.dim)
            + " layers=" + std::to_string(config.layers_num)
            + " heads=" + std::to_string(config.heads_num)
            + " vocab=" + std::to_string(config.vocab_size));
    }
    if (config.dim % config.heads_num != 0) {
        throw std::runtime_error("dim not divisible by heads_num");
    }
    if (config.heads_num % config.kv_heads_num != 0) {
        throw std::runtime_error("heads_num not divisible by kv_heads_num");
    }

    config.head_dim = config.dim / config.heads_num;

    std::cout << "GGUF config: vocab=" << config.vocab_size
              << " dim=" << config.dim << " layers=" << config.layers_num
              << " heads=" << config.heads_num << " kv_heads=" << config.kv_heads_num
              << " head_dim=" << config.head_dim << " hidden_dim=" << config.hidden_dim
              << " max_seq_len=" << config.max_seq_len << " rope_theta=" << config.rope_theta
              << " rms_norm_eps=" << config.rms_norm_eps << std::endl;

    return config;
}

// 后续移植 GGUF 形状映射时替换，返回当前 Tensor 使用的 size_t 形状。
static std::vector<size_t> MapShape(const std::vector<uint64_t>&) {
    throw std::runtime_error("未完成：MapShape GGUF 张量形状映射");
}

// 当前模型使用 F32 Tensor；量化存储及原始权重解码尚未移植。
static Tensor LoadQuantizedTensor(const std::vector<uint8_t>&,
                                  const std::vector<size_t>&, uint32_t ggml_type) {
    throw std::runtime_error("未完成：LoadQuantizedTensor GGUF 权重解码及量化支持，类型: "
                             + std::to_string(ggml_type));
}

static void RequireShape(const Tensor& tensor, const std::vector<size_t>& expected,
                         const std::string& name) {
    tensor.AssertShape(expected, name.c_str());
}

static void RequireOptionalShape(const Tensor& tensor, const std::vector<size_t>& expected,
                                 const std::string& name) {
    if (tensor.data.empty()) {
        return;
    }
    RequireShape(tensor, expected, name);
}

static void ValidateLoadedShapes(const Model& model) {
    const ModelConfig& config = model.config;
    const size_t q_dim = config.heads_num * config.head_dim;
    const size_t kv_dim = config.kv_heads_num * config.head_dim;

    RequireShape(model.token_embedding, {config.vocab_size, config.dim}, "GGUF token_embedding");
    RequireShape(model.final_norm, {config.dim}, "GGUF final_norm");
    RequireShape(model.lm_head, {config.vocab_size, config.dim}, "GGUF lm_head");

    if (model.layers.size() != config.layers_num) {
        throw std::runtime_error("GGUF layer count mismatch: expected "
                                 + std::to_string(config.layers_num) + ", got "
                                 + std::to_string(model.layers.size()));
    }

    for (size_t layer = 0; layer < config.layers_num; ++layer) {
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
        RequireShape(lw.attention_norm, {config.dim}, prefix + "attention_norm");
        RequireShape(lw.wq, {q_dim, config.dim}, prefix + "wq");
        RequireShape(lw.wk, {kv_dim, config.dim}, prefix + "wk");
        RequireShape(lw.wv, {kv_dim, config.dim}, prefix + "wv");
        RequireOptionalShape(lw.bq, {q_dim}, prefix + "bq");
        RequireOptionalShape(lw.bk, {kv_dim}, prefix + "bk");
        RequireOptionalShape(lw.bv, {kv_dim}, prefix + "bv");
        RequireShape(lw.wo, {config.dim, q_dim}, prefix + "wo");
        RequireShape(lw.ffn_norm, {config.dim}, prefix + "ffn_norm");
        RequireShape(lw.w_gate, {config.hidden_dim, config.dim}, prefix + "w_gate");
        RequireShape(lw.w_up, {config.hidden_dim, config.dim}, prefix + "w_up");
        RequireShape(lw.w_down, {config.dim, config.hidden_dim}, prefix + "w_down");
    }
}

Model LoadGgufModel(const std::string& gguf_path) {
    Model model;
    GgufReader reader;
    if (!reader.load(gguf_path)) {
        model.load_error = "GGUF parse error: " + reader.error;
        return model;
    }

    try {
        model.config = BuildConfig(gguf_path);
    } catch (const std::exception& error) {
        model.load_error = std::string("Failed to build config: ") + error.what();
        return model;
    }

    const ModelConfig& config = model.config;
    std::cout << "GGUF config: vocab=" << config.vocab_size << " dim=" << config.dim << " layers=" << config.layers_num
              << " heads=" << config.heads_num << " kv_heads=" << config.kv_heads_num << " head_dim=" << config.head_dim
              << " hidden_dim=" << config.hidden_dim << " max_seq_len=" << config.max_seq_len << std::endl;
    
    std::map<std::string, const GgufTensorInfo*> tensor_map;
    for (const auto& t : reader.tensors) {
        std::string mapped = MapTensorName(t.name);
        if (mapped.empty()) {
            continue;
        }
        tensor_map[mapped] = &t;
    }


    auto load_weight = [&](const std::string& mapped_name) -> Tensor {
        auto it = tensor_map.find(mapped_name);
        if (it == tensor_map.end()) {
            throw std::runtime_error("Missing tensor: " + mapped_name);
        }
        const GgufTensorInfo* info = it->second;
        std::vector<size_t> shape = MapShape(info->shape);

        std::vector<uint8_t> raw = ReadGgufTensorRaw(gguf_path, *info, reader.data_offset);
        if (raw.empty() && info->type != GgmlTypeF32) {
            throw std::runtime_error("Failed to read tensor: " + mapped_name);
        }

        return LoadQuantizedTensor(raw, shape, info->type);
    };

    [[maybe_unused]] auto load_optional_weight = [&](const std::string& mapped_name) -> Tensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return Tensor();
        }
        return load_weight(mapped_name);
    };

    
    auto load_f32 = [&](const std::string& mapped_name) -> Tensor {
        auto it = tensor_map.find(mapped_name);
        if (it == tensor_map.end()) {
            throw std::runtime_error("Missing tensor: " + mapped_name);
        }
        switch (it->second->type) {
            case GgmlTypeF32:
                return load_weight(mapped_name);
            case GgmlTypeQ80:
                throw std::runtime_error("未完成：DequantizeFromQ80，张量: " + mapped_name);
            case GgmlTypeQ40:
                throw std::runtime_error("未完成：DequantizeFromQ40，张量: " + mapped_name);
            case GgmlTypeQ41:
                throw std::runtime_error("未完成：DequantizeFromQ41，张量: " + mapped_name);
            default:
                throw std::runtime_error("Cannot convert tensor to F32: " + mapped_name);
        }
    };

    auto load_optional_f32 = [&](const std::string& mapped_name) -> Tensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return Tensor();
        }
        return load_f32(mapped_name);
    };

    try {
        model.token_embedding = load_f32("token_embedding");
        model.layers.resize(config.layers_num);
        for (size_t layer = 0; layer < config.layers_num; ++layer) {
            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            lw.attention_norm = load_f32(prefix + "attention_norm");
            lw.wq = load_weight(prefix + "wq");
            lw.wk = load_weight(prefix + "wk");
            lw.wv = load_weight(prefix + "wv");
            lw.bq = load_optional_f32(prefix + "bq");
            lw.bk = load_optional_f32(prefix + "bk");
            lw.bv = load_optional_f32(prefix + "bv");
            lw.wo = load_weight(prefix + "wo");
            lw.ffn_norm = load_f32(prefix + "ffn_norm");
            lw.w_gate = load_weight(prefix + "w_gate");
            lw.w_up = load_weight(prefix + "w_up");
            lw.w_down = load_weight(prefix + "w_down");
        }

        model.final_norm = load_f32("final_norm");

        // 有独立输出权重时读取，否则沿用 embedding 的权重值。
        auto it = tensor_map.find("lm_head");
        if (it != tensor_map.end()) {
            model.lm_head = load_weight("lm_head");
        } else {
            // 当前两者均为 F32 Tensor，直接复制形状和数据。
            model.lm_head = model.token_embedding;
        }

        ValidateLoadedShapes(model);
    } catch (const std::exception& error) {
        model.load_error = std::string("Failed to load GGUF weights: ") + error.what();
        return model;
    }

    model.loaded = true;
    std::cout << "GGUF model loaded successfully from " << gguf_path << std::endl;
    return model;
}
