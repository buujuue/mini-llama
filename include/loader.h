#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include "model.h"

enum class TensorKind{
    TokenEmbedding,
    AttentionNorm,
    Wq, Wk, Wv, Wo,
    FfnNorm,
    WGate, WUp, WDown,
    FinalNorm,
    LmHead
};

enum class DataType{
    Float32,
    Float16,
    Int8,
};

struct TensorId{
    TensorKind kind;
    int layer = -1;

    bool operator<(const TensorId& other) const {
        if (kind != other.kind) return kind < other.kind;
        return layer < other.layer;
    }
};

struct TensorInfo {
    TensorId id;
    std::vector<size_t> shape;
    DataType dataType;
    size_t offset = 0;
    size_t byte_size = 0;
};

enum class TokenizerType {
    Ascii,
    JsonVocab,
};

struct TokenizerInfo {
    TokenizerType type = TokenizerType::Ascii;
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};

struct ModelDescriptor {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
};

ModelDescriptor ParseDescriptor(const std::string& config_path);

Model LoadModel(const std::string& config_path, const std::string& weights_path);