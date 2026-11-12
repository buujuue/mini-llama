// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 字节序列构造适配自 mini-llama-cpp/tests/gtest/test_gguf.cc
// 和 scripts/make_test_gguf.py；测试不依赖 references/ 中的模型文件。
#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "gguf.h"

namespace mini_llama::test {

using Bytes = std::vector<char>;

inline void AppendLe(Bytes& data, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
    }
}

inline void AppendString(Bytes& data, const std::string& value) {
    AppendLe(data, value.size(), 8);
    data.insert(data.end(), value.begin(), value.end());
}

inline void WriteLe(Bytes& data, size_t offset, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) {
        data.at(offset + i) = static_cast<char>((value >> (i * 8)) & 0xff);
    }
}

inline Bytes IntegerValue(uint64_t value, size_t width = 4) {
    Bytes data;
    AppendLe(data, value, width);
    return data;
}

inline Bytes StringValue(const std::string& value) {
    Bytes data;
    AppendString(data, value);
    return data;
}

struct MetadataEntry {
    std::string key;
    uint32_t type;
    Bytes value;
};

struct TensorEntry {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = GgmlTypeF32;
    uint64_t offset = 0;
};

inline Bytes MakeGguf(const std::vector<MetadataEntry>& metadata,
                      const std::vector<TensorEntry>& tensors,
                      size_t payload_bytes = 0, size_t alignment = 32) {
    Bytes data{'G', 'G', 'U', 'F'};
    AppendLe(data, 3, 4);
    AppendLe(data, tensors.size(), 8);
    AppendLe(data, metadata.size(), 8);
    for (const auto& kv : metadata) {
        AppendString(data, kv.key);
        AppendLe(data, kv.type, 4);
        data.insert(data.end(), kv.value.begin(), kv.value.end());
    }
    for (const auto& tensor : tensors) {
        AppendString(data, tensor.name);
        AppendLe(data, tensor.shape.size(), 4);
        for (uint64_t dim : tensor.shape) AppendLe(data, dim, 8);
        AppendLe(data, tensor.type, 4);
        AppendLe(data, tensor.offset, 8);
    }
    const size_t padding = (alignment - data.size() % alignment) % alignment;
    data.resize(data.size() + padding + payload_bytes, 0);
    return data;
}

inline Bytes TinyGguf() {
    return MakeGguf({{"general.architecture", GgufTypeString, StringValue("llama")},
                     {"general.name", GgufTypeString, StringValue("tiny-test")},
                     {"general.alignment", GgufTypeUint32, IntegerValue(64)}},
                    {{"token_embd.weight", {128, 32}}}, 128 * 32 * 4, 64);
}

class GgufFileTest : public testing::Test {
protected:
    void SetUp() override {
        const auto* test = testing::UnitTest::GetInstance()->current_test_info();
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::current_path() /
            (std::string(test->test_suite_name()) + "_" + test->name() + "_" +
             std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directory(directory_));
        path_ = directory_ / "fixture.gguf";
        missing_path_ = directory_ / "missing.gguf";
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(directory_, error);
        EXPECT_FALSE(error) << error.message();
    }

    std::string Write(const Bytes& data) const {
        std::ofstream file(path_, std::ios::binary);
        if (!data.empty()) {
            file.write(data.data(), static_cast<std::streamsize>(data.size()));
        }
        file.close();
        if (!file) throw std::runtime_error("failed to write GGUF fixture");
        return path_.string();
    }

    std::filesystem::path directory_;
    std::filesystem::path path_;
    std::filesystem::path missing_path_;
};

}  // namespace mini_llama::test
