// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// GGUF 加载适配自 mini-llama-cpp/src/gguf_loader.cc，解码公式适配自 src/quant.cc。

#include "gguf_loader.h"
#include "gguf.h"
#include "model.h"
#include "tensor.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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
    config.vocab_size = 0;

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

static std::vector<size_t> MapShape(const std::vector<uint64_t>& gguf_shape) {
    if (gguf_shape.empty() || gguf_shape.size() > MaxTensorDims) {
        throw std::runtime_error("Invalid GGUF tensor rank");
    }
    std::vector<size_t> shape;
    shape.reserve(gguf_shape.size());
    for (uint64_t dim : gguf_shape) {
        if (dim == 0 || dim > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("GGUF tensor dimension is outside size_t range");
        }
        shape.push_back(static_cast<size_t>(dim));
    }
    if (shape.size() == 2) {
        std::swap(shape[0], shape[1]);
    }
    return shape;
}

static float ReadFp16(const uint8_t* bytes) {
    const uint32_t value = static_cast<uint32_t>(bytes[0]) |
                           (static_cast<uint32_t>(bytes[1]) << 8);
    const uint32_t sign = (value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = sign;
    if (exponent == 0) {
        if (mantissa != 0) {
            uint32_t exponent32 = 113;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent32;
            }
            bits |= (exponent32 << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31) {
        bits |= 0x7f800000u | (mantissa << 13);
    } else {
        bits |= ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// 模型统一使用 F32 Tensor，加载时完成 F16 转换或量化块解码。
static Tensor LoadQuantizedTensor(const std::vector<uint8_t>& raw,
                                  const std::vector<size_t>& shape, uint32_t ggml_type) {
    if (shape.empty() || shape.size() > MaxTensorDims) {
        throw std::runtime_error("Invalid GGUF tensor rank");
    }
    size_t num_elements = 1;
    for (size_t dim : shape) {
        if (dim == 0 || num_elements > std::numeric_limits<size_t>::max() / dim) {
            throw std::runtime_error("Invalid GGUF tensor element count");
        }
        num_elements *= dim;
    }

    size_t block_size = 1;
    size_t block_bytes = 0;
    switch (ggml_type) {
        case GgmlTypeF32:
            block_bytes = 4;
            break;
        case GgmlTypeF16:
            block_bytes = 2;
            break;
        case GgmlTypeQ80:
            block_size = QuantBlockSize;
            block_bytes = 34;
            break;
        case GgmlTypeQ40:
            block_size = QuantBlockSize;
            block_bytes = 18;
            break;
        case GgmlTypeQ41:
            block_size = QuantBlockSize;
            block_bytes = 20;
            break;
        default:
            throw std::runtime_error("Unsupported GGUF tensor type: " + std::to_string(ggml_type));
    }
    if (shape.back() % block_size != 0) {
        throw std::runtime_error("Quantized GGUF tensor row size must be a multiple of 32");
    }
    const size_t num_blocks = num_elements / block_size;
    if (num_blocks > std::numeric_limits<size_t>::max() / block_bytes) {
        throw std::runtime_error("GGUF tensor byte size overflow");
    }
    const size_t expected_bytes = num_blocks * block_bytes;
    if (raw.size() != expected_bytes) {
        throw std::runtime_error(std::string(GgmlTypeName(ggml_type))
                                 + " tensor size mismatch: expected " + std::to_string(expected_bytes)
                                 + " bytes, got " + std::to_string(raw.size()));
    }

    Tensor tensor(shape);
    if (ggml_type == GgmlTypeF32) {
        std::memcpy(tensor.data.data(), raw.data(), expected_bytes);
        return tensor;
    }
    if (ggml_type == GgmlTypeF16) {
        for (size_t i = 0; i < num_elements; ++i) {
            tensor.data[i] = ReadFp16(raw.data() + i * 2);
        }
        return tensor;
    }

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        const uint8_t* block = raw.data() + block_index * block_bytes;
        const float scale = ReadFp16(block);
        const size_t base = block_index * QuantBlockSize;
        if (ggml_type == GgmlTypeQ80) {
            for (size_t i = 0; i < QuantBlockSize; ++i) {
                int value = block[2 + i];
                if (value >= 128) {
                    value -= 256;
                }
                tensor.data[base + i] = scale * static_cast<float>(value);
            }
            continue;
        }

        float minimum = 0.0f;
        int zero_point = 8;
        const uint8_t* values = block + 2;
        if (ggml_type == GgmlTypeQ41) {
            minimum = ReadFp16(block + 2);
            zero_point = 0;
            values += 2;
        }
        // 每字节的低四位对应前 16 个元素，高四位对应后 16 个元素。
        for (size_t i = 0; i < QuantBlockSize / 2; ++i) {
            const int low = static_cast<int>(values[i] & 0x0f) - zero_point;
            const int high = static_cast<int>(values[i] >> 4) - zero_point;
            tensor.data[base + i] = scale * static_cast<float>(low) + minimum;
            tensor.data[base + i + QuantBlockSize / 2] = scale * static_cast<float>(high) + minimum;
        }
    }
    return tensor;
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
    try {
        if (!reader.load(gguf_path)) {
            model.load_error = "GGUF parse error: " + reader.error;
            return model;
        }
    } catch (const std::exception& error) {
        model.load_error = std::string("GGUF parse error: ") + error.what();
        return model;
    }

    try {
        model.config = BuildConfig(gguf_path);
    } catch (const std::exception& error) {
        model.load_error = std::string("Failed to build config: ") + error.what();
        return model;
    }

    const ModelConfig& config = model.config;
    
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
        if (raw.empty()) {
            throw std::runtime_error("Failed to read tensor: " + mapped_name);
        }

        return LoadQuantizedTensor(raw, shape, info->type);
    };

    auto load_optional_weight = [&](const std::string& mapped_name) -> Tensor {
        if (tensor_map.find(mapped_name) == tensor_map.end()) {
            return Tensor();
        }
        return load_weight(mapped_name);
    };

    try {
        model.token_embedding = load_weight("token_embedding");
        model.layers.resize(config.layers_num);
        for (size_t layer = 0; layer < config.layers_num; ++layer) {
            LayerWeights& lw = model.layers[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            lw.attention_norm = load_weight(prefix + "attention_norm");
            lw.wq = load_weight(prefix + "wq");
            lw.wk = load_weight(prefix + "wk");
            lw.wv = load_weight(prefix + "wv");
            lw.bq = load_optional_weight(prefix + "bq");
            lw.bk = load_optional_weight(prefix + "bk");
            lw.bv = load_optional_weight(prefix + "bv");
            lw.wo = load_weight(prefix + "wo");
            lw.ffn_norm = load_weight(prefix + "ffn_norm");
            lw.w_gate = load_weight(prefix + "w_gate");
            lw.w_up = load_weight(prefix + "w_up");
            lw.w_down = load_weight(prefix + "w_down");
        }

        model.final_norm = load_weight("final_norm");

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
