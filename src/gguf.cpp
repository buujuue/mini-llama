// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// GGUF 值读取、类型名、加载校验与信息打印适配自 mini-llama-cpp/src/gguf.cc。

#include "gguf.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
template <typename T>
static bool ReadLe(std::ifstream& f, T& out) {
    char buf[sizeof(out)];
    if (!f.read(buf, sizeof(T))) {
        return false;
    }
    std::memcpy(&out, buf, sizeof(T));
    return true;
}

bool ReadU32(std::ifstream& f, uint32_t& out) {
    return ReadLe(f, out);
}

bool ReadU64(std::ifstream& f, uint64_t& out) {
    return ReadLe(f, out);
}

bool ReadString(std::ifstream& f, std::string& out) {
    uint64_t len = 0;
    if (!ReadU64(f, len)) {
        return false;
    }
    if (len > MaxStringBytes || len > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    out.resize(len);
    if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
        return false;
    }
    return true;
}

void GgufReader::clear(){
    version = 0;
    tensors_num = 0;
    metadata_num = 0;
    data_offset = 0;
    metadata.clear();
    tensors.clear();
    error.clear();
}

const char* GgmlTypeName(uint32_t type) {
    switch (type) {
        case GgmlTypeF32:
            return "F32";
        case GgmlTypeF16:
            return "F16";
        case GgmlTypeQ40:
            return "Q4_0";
        case GgmlTypeQ41:
            return "Q4_1";
        case GgmlTypeQ80:
            return "Q8_0";
        default:
            return "UNKNOWN";
    }
}

static const char* GgufValueTypeName(uint32_t type) {
    switch (type) {
        case GgufTypeUint8:
            return "uint8";
        case GgufTypeInt8:
            return "int8";
        case GgufTypeUint16:
            return "uint16";
        case GgufTypeInt16:
            return "int16";
        case GgufTypeUint32:
            return "uint32";
        case GgufTypeInt32:
            return "int32";
        case GgufTypeFloat32:
            return "float32";
        case GgufTypeBool:
            return "bool";
        case GgufTypeString:
            return "string";
        case GgufTypeArray:
            return "array";
        case GgufTypeUint64:
            return "uint64";
        case GgufTypeInt64:
            return "int64";
        case GgufTypeFloat64:
            return "float64";
        default:
            return "unknown";
    }
}

static bool SkipValue(std::ifstream& f, uint32_t type) {
    switch (type) {
        case GgufTypeUint8:
        case GgufTypeInt8: {
            char dummy;
            return f.read(&dummy, 1).good();
        }
        case GgufTypeUint16:
        case GgufTypeInt16: {
            char dummy[2];
            return f.read(dummy, 2).good();
        }
        case GgufTypeUint32:
        case GgufTypeInt32:
        case GgufTypeFloat32: {
            char dummy[4];
            return f.read(dummy, 4).good();
        }
        case GgufTypeBool: {
            char dummy;
            return f.read(&dummy, 1).good();
        }
        case GgufTypeString: {
            std::string dummy;
            return ReadString(f, dummy);
        }
        case GgufTypeUint64:
        case GgufTypeInt64:
        case GgufTypeFloat64: {
            char dummy[8];
            return f.read(dummy, 8).good();
        }
        case GgufTypeArray: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > MaxArrayElements) {
                return false;
            }
            for (uint64_t i = 0; i < len; ++i) {
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
    std::ostringstream oss;
    switch (type) {
        case GgufTypeUint8: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << static_cast<unsigned int>(v);
            break;
        }
        case GgufTypeInt8: {
            int8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << static_cast<int>(v);
            break;
        }
        case GgufTypeUint16: {
            uint16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeInt16: {
            int16_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeUint32: {
            uint32_t v = 0;
            if (!ReadU32(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeInt32: {
            int32_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeFloat32: {
            float v = 0.0f;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeBool: {
            uint8_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << (v ? "true" : "false");
            break;
        }
        case GgufTypeString: {
            if (!ReadString(f, out)) {
                return false;
            }
            return true;
        }
        case GgufTypeUint64: {
            uint64_t v = 0;
            if (!ReadU64(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeInt64: {
            int64_t v = 0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeFloat64: {
            double v = 0.0;
            if (!ReadLe(f, v)) {
                return false;
            }
            oss << v;
            break;
        }
        case GgufTypeArray: {
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
                return false;
            }
            if (len > MaxArrayElements) {
                return false;
            }
            oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
            for (uint64_t i = 0; i < len; ++i) {
                if (!SkipValue(f, elem_type)) {
                    return false;
                }
            }
            break;
        }
        default:
            return false;
    }
    out = oss.str();
    return true;
}

static bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

static bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
    if (tensor.shape.empty() || tensor.shape.size() > MaxTensorDims) {
        return false;
    }
    out = 1;
    for (uint64_t dim : tensor.shape) {
        if (dim == 0 || !MultiplyChecked(out, dim, out)) {
            return false;
        }
    }
    return true;
}

// 计算存储字节数；量化格式每 32 个元素为一块，包含缩放值等块头数据。
static bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
    uint64_t num_elements = 0;
    if (!TensorNumel(tensor, num_elements)) {
        return false;
    }

    switch (tensor.type) {
        case GgmlTypeF32:
            return MultiplyChecked(num_elements, 4, out);
        case GgmlTypeF16:
            return MultiplyChecked(num_elements, 2, out);
        case GgmlTypeQ40: {
            if (num_elements % QuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = num_elements / QuantBlockSize;
            return MultiplyChecked(n_blocks, 18, out);
        }
        case GgmlTypeQ41: {
            if (num_elements % QuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = num_elements / QuantBlockSize;
            return MultiplyChecked(n_blocks, 20, out);
        }
        case GgmlTypeQ80: {
            if (num_elements % QuantBlockSize != 0) {
                return false;
            }
            uint64_t n_blocks = num_elements / QuantBlockSize;
            return MultiplyChecked(n_blocks, 34, out);
        }
        default:
            throw std::runtime_error("未完成：GGML 张量类型 " + std::to_string(tensor.type)
                                     + " 的数据大小计算，张量: " + tensor.name);
    }
}

static bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

static uint64_t ParseAlignment(const std::vector<GgufMetadataKV>& metadata) {
    uint64_t alignment = DefaultAlignment;
    for (const auto& kv : metadata) {
        if (kv.key != "general.alignment") {
            continue;
        }
        try {
            size_t parsed = 0;
            alignment = std::stoull(kv.value_str, &parsed);
            if (parsed != kv.value_str.size()) {
                return 0;
            }
        } catch (const std::exception&) {
            return 0;
        }
    }
    if (alignment == 0 || alignment > MaxAlignment) {
        return 0;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return 0;
    }
    return alignment;
}

bool GgufReader::load(const std::string& path) {
    clear();
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        error = "[GgufReader] 调用 load 时无法打开 GGUF 文件，路径: '" + path + "'。";
        return false;
    }

    char magic[4];
    if (!f.read(magic, 4)) {
        error = "[GgufReader] 调用 load 时读取 GGUF 魔数失败，路径: '" + path
            + "'，偏移: 0，预期字节数: 4，实际读取字节数: "
            + std::to_string(f.gcount()) + "。";
        return false;
    }

    if (std::memcmp(magic, "GGUF", 4) != 0) {
        error = "[GgufReader] 调用 load 时 GGUF 魔数不匹配，路径: '" + path + "读取magic: " + std::string(magic, sizeof(magic)) + ".";
        return false;
    }

    if (!ReadU32(f, version)){
        error = "failed to read version";
        return false;
    }

    if (version != 3) {
        error = "unsupported GGUF version: " + std::to_string(version);
        return false;
    }

    if (!ReadU64(f, tensors_num) || !ReadU64(f, metadata_num)) {
        error = "failed to read header counts";
        return false;
    }

    if (tensors_num > MaxTensorEntries) {
        error = "too many tensors: " + std::to_string(tensors_num);
        return false;
    }
    if (metadata_num > MaxMetadataEntries) {
        error = "too many metadata entries: " + std::to_string(metadata_num);
        return false;
    }

    metadata.reserve(metadata_num);
    std::unordered_set<std::string> metadata_keys;
    for (uint64_t i = 0; i < metadata_num; ++i) {
        GgufMetadataKV kv;
        if (!ReadString(f, kv.key)) {
            error = "failed to read metadata key at FlatIndex " + std::to_string(i);
            return false;
        }

        if (!metadata_keys.insert(kv.key).second) {
            error = "duplicate metadata key: " + kv.key;
            return false;
        }

        if (!ReadU32(f, kv.type)) {
            error = "failed to read metadata type for key: " + kv.key;
            return false;
        }

        if (!ReadValueAsString(f, kv.type, kv.value_str)) {
            error = "failed to read metadata value for key: " + kv.key;
            return false;
        }
        metadata.push_back(std::move(kv));
    }
    tensors.reserve(tensors_num);
    for (uint64_t i = 0; i < tensors_num; ++i) {
        GgufTensorInfo info;
        if (!ReadString(f, info.name)) {
            error = "failed to read tensor name at FlatIndex " + std::to_string(i);
            return false;
        }
        uint32_t n_dims = 0;
        if (!ReadU32(f, n_dims)) {
            error = "failed to read tensor n_dims for: " + info.name;
            return false;
        }
        if (n_dims == 0 || n_dims > MaxTensorDims) {
            error = "invalid tensor n_dims for: " + info.name;
            return false;
        }
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!ReadU64(f, info.shape[d])) {
                error = "failed to read tensor shape for: " + info.name;
                return false;
            }
            if (info.shape[d] == 0) {
                error = "invalid zero tensor dimension for: " + info.name;
                return false;
            }
        }
        if (!ReadU32(f, info.type)) {
            error = "failed to read tensor type for: " + info.name;
            return false;
        }
        if (!ReadU64(f, info.offset)) {
            error = "failed to read tensor offset for: " + info.name;
            return false;
        }
        tensors.push_back(std::move(info));
    }

    std::unordered_set<std::string> names;
    for (const auto& tensor : tensors) {
        if (!names.insert(tensor.name).second) {
            error = "duplicate tensor name: " + tensor.name;
            return false;
        }
    }

    std::streampos current = f.tellg();
    if (current < 0) {
        error = "failed to determine GGUF data offset";
        return false;
    }
    data_offset = static_cast<uint64_t>(current);
    uint64_t alignment = ParseAlignment(metadata);
    if (alignment == 0) {
        error = "invalid general.alignment";
        return false;
    }
    uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
    data_offset += padding;

    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
        error = "GGUF file ended before tensor data section";
        return false;
    }
    uint64_t file_size = static_cast<uint64_t>(end);

    for (const auto& tensor : tensors) {
        uint64_t tensor_size = 0;
        if (!TensorDataSize(tensor, tensor_size)) {
            error = "invalid tensor data size for: " + tensor.name;
            return false;
        }
        if (tensor_size == 0) {
            continue;
        }

        uint64_t tensor_start = 0;
        if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
            error = "tensor offset overflow for: " + tensor.name;
            return false;
        }

        uint64_t tensor_end = 0;
        if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
            error = "tensor data range overflow for: " + tensor.name;
            return false;
        }

        if (tensor_end > file_size) {
            error = "tensor data outside file bounds: " + tensor.name;
            return false;
        }
    }

    return true;
}

void InspectGguf(const GgufReader& reader) {
    std::cout << "=== GGUF Info ===" << std::endl;
    std::cout << "  version:    " << reader.version << std::endl;
    std::cout << "  n_tensors:  " << reader.tensors_num << std::endl;
    std::cout << "  n_metadata: " << reader.metadata_num << std::endl;
    std::cout << "  data_offset: " << reader.data_offset << std::endl;

    if (!reader.metadata.empty()) {
        std::cout << std::endl;
        std::cout << "=== Metadata ===" << std::endl;
        for (const auto& kv : reader.metadata) {
            std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
                      << "): " << kv.value_str << std::endl;
        }
    }

    if (!reader.tensors.empty()) {
        std::cout << std::endl;
        std::cout << "=== Tensors ===" << std::endl;
        for (const auto& tensor : reader.tensors) {
            std::cout << "  " << tensor.name << std::endl;
            std::cout << "    shape:  [";
            for (size_t i = 0; i < tensor.shape.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << tensor.shape[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "    dtype:  " << GgmlTypeName(tensor.type) << " (" << tensor.type
                      << ")" << std::endl;
            std::cout << "    offset: " << tensor.offset << std::endl;
        }
    }
}

// BuildConfig 的读取依赖占位，后续移植类型化元数据读取时替换。
bool GgufGetMetadataInt(const std::string& path, const std::string& key, int64_t&) {
    throw std::runtime_error("未完成：GgufGetMetadataInt，路径: " + path + "，键: " + key);
}

bool GgufGetMetadataFloat(const std::string& path, const std::string& key, double&) {
    throw std::runtime_error("未完成：GgufGetMetadataFloat，路径: " + path + "，键: " + key);
}

bool GgufGetMetadataString(const std::string& path, const std::string& key, std::string&) {
    throw std::runtime_error("未完成：GgufGetMetadataString，路径: " + path + "，键: " + key);
}

bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>&) {
    throw std::runtime_error("未完成：GgufGetMetadataStringArray，路径: " + path + "，键: " + key);
}

std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                     const GgufTensorInfo& info, uint64_t) {
    throw std::runtime_error("未完成：ReadGgufTensorRaw 原始张量读取，路径: " + path
                             + "，张量: " + info.name);
}
