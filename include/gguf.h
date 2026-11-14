#pragma once

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

constexpr uint64_t DefaultAlignment = 32;
constexpr uint64_t MaxAlignment = 1024 * 1024;
constexpr uint64_t MaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t MaxMetadataEntries = 1000000;
constexpr uint64_t MaxTensorEntries = 1000000;
constexpr uint64_t MaxArrayElements = 10000000;
constexpr uint32_t MaxTensorDims = 8;
constexpr uint64_t QuantBlockSize = 32;

// GGUF 张量的存储类型，与下方的元数据值类型不同。
enum GgmlType : uint32_t {
    GgmlTypeF32 = 0,
    GgmlTypeF16 = 1,
    GgmlTypeQ40 = 2,
    GgmlTypeQ41 = 3,
    GgmlTypeQ80 = 8,
};

const char* GgmlTypeName(uint32_t type);

enum GgufValueType : uint32_t {
    GgufTypeUint8 = 0,
    GgufTypeInt8 = 1,
    GgufTypeUint16 = 2,
    GgufTypeInt16 = 3,
    GgufTypeUint32 = 4,
    GgufTypeInt32 = 5,
    GgufTypeFloat32 = 6,
    GgufTypeBool = 7,
    GgufTypeString = 8,
    GgufTypeArray = 9,
    GgufTypeUint64 = 10,
    GgufTypeInt64 = 11,
    GgufTypeFloat64 = 12,
};

struct GgufMetadataKV {
    std::string key;
    std::string value_str;
    uint32_t type = 0;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
};

struct GgufReader {
    uint32_t version = 0;
    uint64_t tensors_num = 0;
    uint64_t metadata_num = 0;
    std::vector<GgufMetadataKV> metadata;
    std::vector<GgufTensorInfo> tensors;
    uint64_t data_offset = 0;
    std::string error;

    bool load(const std::string& path);
private:
    void clear();
};

void InspectGguf(const GgufReader& reader);

// 从小端 GGUF v2/v3 读取元数据，不读取或解码张量；缺失、类型不符或损坏时
// 返回 false，保持 out 不变。整数数组要求 int32 元素。
bool GgufGetMetadataInt(const std::string& path, const std::string& key, int64_t& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key, double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key, std::string& out);
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out);

// 原始张量读取尚未移植，当前调用会抛出“未完成”异常。
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                     const GgufTensorInfo& info, uint64_t data_offset);
