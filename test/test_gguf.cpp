// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT
// 基础场景移植自 mini-llama-cpp/tests/gtest/test_gguf.cc。

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gguf_fixture.h"

namespace {

using namespace mini_llama::test;

class GgufTest : public GgufFileTest {
protected:
    void ExpectRejected(const Bytes& data, const std::string& message) const {
        GgufReader reader;
        EXPECT_FALSE(reader.load(Write(data)));
        EXPECT_EQ(reader.error, message);
    }
};

// 上游的 15 项测试：适配 load/计数字段/中文错误，并自行生成 test.gguf。
TEST_F(GgufTest, LoadTestFile) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    EXPECT_EQ(reader.version, 3u);
    EXPECT_EQ(reader.tensors_num, 1u);
    EXPECT_EQ(reader.metadata_num, 3u);
    EXPECT_TRUE(reader.error.empty());
}

TEST_F(GgufTest, Metadata) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    ASSERT_EQ(reader.metadata.size(), 3u);
    EXPECT_EQ(reader.metadata[0].key, "general.architecture");
    EXPECT_EQ(reader.metadata[0].value_str, "llama");
    EXPECT_EQ(reader.metadata[0].type, GgufTypeString);
    EXPECT_EQ(reader.metadata[1].key, "general.name");
    EXPECT_EQ(reader.metadata[1].value_str, "tiny-test");
    EXPECT_EQ(reader.metadata[2].key, "general.alignment");
    EXPECT_EQ(reader.metadata[2].value_str, "64");
    EXPECT_EQ(reader.metadata[2].type, GgufTypeUint32);
}

TEST_F(GgufTest, TensorInfo) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    ASSERT_EQ(reader.tensors.size(), 1u);
    EXPECT_EQ(reader.tensors[0].name, "token_embd.weight");
    EXPECT_EQ(reader.tensors[0].shape, (std::vector<uint64_t>{128, 32}));
    EXPECT_EQ(reader.tensors[0].type, GgmlTypeF32);
    EXPECT_EQ(reader.tensors[0].offset, 0u);
}

TEST_F(GgufTest, DataOffsetComputed) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    // 独立已知布局：200 字节的信息段补齐到 256 字节。
    EXPECT_EQ(reader.data_offset, 256u);
    EXPECT_EQ(reader.data_offset % 64, 0u);
}

TEST_F(GgufTest, RepeatedLoadResetsState) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    EXPECT_FALSE(reader.load(Write({'{', '}', '\n', ' '})));
    EXPECT_EQ(reader.version, 0u);
    EXPECT_EQ(reader.tensors_num, 0u);
    EXPECT_EQ(reader.metadata_num, 0u);
    EXPECT_TRUE(reader.metadata.empty());
    EXPECT_TRUE(reader.tensors.empty());
    EXPECT_EQ(reader.data_offset, 0u);
    EXPECT_NE(reader.error.find("魔数不匹配"), std::string::npos);
}

TEST_F(GgufTest, UnsupportedVersionRejected) {
    Bytes data = TinyGguf();
    WriteLe(data, 4, 2, 4);
    ExpectRejected(data, "unsupported GGUF version: 2");
}

TEST_F(GgufTest, TruncatedFileRejected) {
    Bytes data = TinyGguf();
    data.resize(12);
    ExpectRejected(data, "failed to read header counts");
}

TEST_F(GgufTest, MissingFile) {
    GgufReader reader;
    EXPECT_FALSE(reader.load(missing_path_.string()));
    EXPECT_NE(reader.error.find("无法打开 GGUF 文件"), std::string::npos);
    EXPECT_NE(reader.error.find(missing_path_.string()), std::string::npos);
}

TEST_F(GgufTest, InvalidMagic) {
    GgufReader reader;
    EXPECT_FALSE(reader.load(Write({'{', '}', '\n', ' '})));
    EXPECT_NE(reader.error.find("魔数不匹配"), std::string::npos);
    EXPECT_NE(reader.error.find(path_.string()), std::string::npos);
}

TEST_F(GgufTest, InvalidAlignmentRejected) {
    for (uint32_t alignment : {0u, 48u, 2097152u}) {
        SCOPED_TRACE(alignment);
        ExpectRejected(MakeGguf({{"general.alignment", GgufTypeUint32,
                                  IntegerValue(alignment)}}, {}),
                       "invalid general.alignment");
    }
}

TEST_F(GgufTest, DuplicateMetadataKeyRejected) {
    ExpectRejected(MakeGguf({{"general.name", GgufTypeString, StringValue("first")},
                             {"general.name", GgufTypeString, StringValue("second")}}, {}),
                   "duplicate metadata key: general.name");
}

TEST_F(GgufTest, ZeroTensorDimensionRejected) {
    ExpectRejected(MakeGguf({}, {{"token_embd.weight", {0, 32}}}),
                   "invalid zero tensor dimension for: token_embd.weight");
}

TEST_F(GgufTest, TensorDataBoundsChecked) {
    ExpectRejected(MakeGguf({}, {{"token_embd.weight", {128, 32}, GgmlTypeF32, 4096}}),
                   "tensor data outside file bounds: token_embd.weight");
}

TEST(GgufTypeTest, GgmlTypeName) {
    EXPECT_STREQ(GgmlTypeName(GgmlTypeF32), "F32");
    EXPECT_STREQ(GgmlTypeName(GgmlTypeF16), "F16");
    EXPECT_STREQ(GgmlTypeName(GgmlTypeQ40), "Q4_0");
    EXPECT_STREQ(GgmlTypeName(GgmlTypeQ41), "Q4_1");
    EXPECT_STREQ(GgmlTypeName(GgmlTypeQ80), "Q8_0");
    EXPECT_STREQ(GgmlTypeName(999), "UNKNOWN");
}

TEST(GgufTypeTest, ValueTypeNamesThroughInspect) {
    // GgufValueTypeName 是内部函数，通过公开的 InspectGguf 验证全部类型名。
    const std::vector<std::pair<uint32_t, std::string>> types = {
        {GgufTypeUint8, "uint8"}, {GgufTypeInt8, "int8"},
        {GgufTypeUint16, "uint16"}, {GgufTypeInt16, "int16"},
        {GgufTypeUint32, "uint32"}, {GgufTypeInt32, "int32"},
        {GgufTypeFloat32, "float32"}, {GgufTypeBool, "bool"},
        {GgufTypeString, "string"}, {GgufTypeArray, "array"},
        {GgufTypeUint64, "uint64"}, {GgufTypeInt64, "int64"},
        {GgufTypeFloat64, "float64"}, {999, "unknown"},
    };
    GgufReader reader;
    for (const auto& type : types) {
        reader.metadata.push_back({"key_" + type.second, "value", type.first});
    }
    std::ostringstream output;
    auto* saved = std::cout.rdbuf(output.rdbuf());
    InspectGguf(reader);
    std::cout.rdbuf(saved);
    for (const auto& type : types) {
        EXPECT_NE(output.str().find("key_" + type.second + " (" + type.second +
                                    "): value"), std::string::npos);
    }
}

// 补充正常边界与畸形输入，覆盖当前已完成的 GGUF 解析和信息打印。
TEST_F(GgufTest, InspectIncludesHeaderMetadataAndTensor) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    std::ostringstream output;
    auto* saved = std::cout.rdbuf(output.rdbuf());
    InspectGguf(reader);
    std::cout.rdbuf(saved);
    for (const char* text : {"version:    3", "n_tensors:  1", "n_metadata: 3",
                             "data_offset: 256", "general.name (string): tiny-test",
                             "token_embd.weight", "shape:  [128, 32]",
                             "dtype:  F32 (0)", "offset: 0"}) {
        EXPECT_NE(output.str().find(text), std::string::npos) << text;
    }
}

TEST_F(GgufTest, EmptyFileReportsShortMagic) {
    GgufReader reader;
    EXPECT_FALSE(reader.load(Write({})));
    EXPECT_NE(reader.error.find("读取 GGUF 魔数失败"), std::string::npos);
    EXPECT_NE(reader.error.find("实际读取字节数: 0"), std::string::npos);
}

TEST_F(GgufTest, TruncatedVersionRejected) {
    ExpectRejected({'G', 'G', 'U', 'F', 3}, "failed to read version");
}

TEST_F(GgufTest, HeaderCountsLimited) {
    Bytes data = MakeGguf({}, {});
    WriteLe(data, 8, 1000001, 8);
    ExpectRejected(data, "too many tensors: 1000001");
    WriteLe(data, 8, 0, 8);
    WriteLe(data, 16, 1000001, 8);
    ExpectRejected(data, "too many metadata entries: 1000001");
}

TEST_F(GgufTest, DefaultAlignmentAndEmptySections) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(MakeGguf({}, {})))) << reader.error;
    EXPECT_EQ(reader.data_offset, 32u);
    EXPECT_EQ(reader.tensors_num, 0u);
    EXPECT_EQ(reader.metadata_num, 0u);
    EXPECT_TRUE(reader.metadata.empty());
    EXPECT_TRUE(reader.tensors.empty());
}

TEST_F(GgufTest, RepeatedSuccessfulLoadReplacesEntriesAndClearsError) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    ASSERT_TRUE(reader.load(Write(MakeGguf({}, {})))) << reader.error;
    EXPECT_TRUE(reader.metadata.empty());
    EXPECT_TRUE(reader.tensors.empty());
    EXPECT_EQ(reader.data_offset, 32u);
    EXPECT_FALSE(reader.load(missing_path_.string()));
    ASSERT_TRUE(reader.load(Write(TinyGguf()))) << reader.error;
    EXPECT_TRUE(reader.error.empty());
    EXPECT_EQ(reader.metadata.size(), 3u);
    EXPECT_EQ(reader.tensors.size(), 1u);
}

TEST_F(GgufTest, ScalarMetadataDecoded) {
    // IEEE-754 的 1.5f 和 -2.25 直接写入已知位模式，独立于读取实现。
    const std::vector<MetadataEntry> entries = {
        {"u8", GgufTypeUint8, IntegerValue(255, 1)},
        {"i8", GgufTypeInt8, IntegerValue(0x80, 1)},
        {"u16", GgufTypeUint16, IntegerValue(65535, 2)},
        {"i16", GgufTypeInt16, IntegerValue(0x8000, 2)},
        {"u32", GgufTypeUint32, IntegerValue(4294967295ULL)},
        {"i32", GgufTypeInt32, IntegerValue(0x80000000)},
        {"f32", GgufTypeFloat32, IntegerValue(0x3fc00000)},
        {"false", GgufTypeBool, IntegerValue(0, 1)},
        {"true", GgufTypeBool, IntegerValue(1, 1)},
        {"string", GgufTypeString, StringValue("模型")},
        {"empty", GgufTypeString, StringValue("")},
        {"u64", GgufTypeUint64, IntegerValue(18446744073709551615ULL, 8)},
        {"i64", GgufTypeInt64, IntegerValue(0x8000000000000000ULL, 8)},
        {"f64", GgufTypeFloat64, IntegerValue(0xc002000000000000ULL, 8)},
    };
    const std::vector<std::string> expected = {
        "255", "-128", "65535", "-32768", "4294967295", "-2147483648",
        "1.5", "false", "true", "模型", "", "18446744073709551615",
        "-9223372036854775808", "-2.25",
    };
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(MakeGguf(entries, {})))) << reader.error;
    ASSERT_EQ(reader.metadata.size(), entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        SCOPED_TRACE(entries[i].key);
        EXPECT_EQ(reader.metadata[i].key, entries[i].key);
        EXPECT_EQ(reader.metadata[i].type, entries[i].type);
        EXPECT_EQ(reader.metadata[i].value_str, expected[i]);
    }
}

TEST_F(GgufTest, ArraysSkippedWithoutLosingFollowingMetadata) {
    std::vector<MetadataEntry> entries;
    const std::vector<std::pair<uint32_t, size_t>> types = {
        {GgufTypeUint8, 1}, {GgufTypeInt8, 1}, {GgufTypeUint16, 2},
        {GgufTypeInt16, 2}, {GgufTypeUint32, 4}, {GgufTypeInt32, 4},
        {GgufTypeFloat32, 4}, {GgufTypeBool, 1}, {GgufTypeUint64, 8},
        {GgufTypeInt64, 8}, {GgufTypeFloat64, 8}, {GgufTypeString, 0},
    };
    for (const auto& type : types) {
        Bytes array = IntegerValue(type.first);
        AppendLe(array, 2, 8);
        if (type.first == GgufTypeString) {
            AppendString(array, "first");
            AppendString(array, "模型");
        } else {
            array.resize(array.size() + 2 * type.second, 0);
        }
        entries.push_back({"array_" + std::to_string(type.first), GgufTypeArray, array});
    }
    Bytes empty = IntegerValue(GgufTypeString);
    AppendLe(empty, 0, 8);
    entries.push_back({"empty", GgufTypeArray, empty});
    entries.push_back({"after", GgufTypeString, StringValue("still readable")});
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(MakeGguf(entries, {})))) << reader.error;
    ASSERT_EQ(reader.metadata.size(), entries.size());
    EXPECT_EQ(reader.metadata[0].value_str, "[2 x uint8]");
    EXPECT_EQ(reader.metadata[11].value_str, "[2 x string]");
    EXPECT_EQ(reader.metadata[12].value_str, "[0 x string]");
    EXPECT_EQ(reader.metadata.back().key, "after");
    EXPECT_EQ(reader.metadata.back().value_str, "still readable");
}

TEST_F(GgufTest, InvalidMetadataValuesRejected) {
    ExpectRejected(MakeGguf({{"unknown", 999, {}}}, {}),
                   "failed to read metadata value for key: unknown");
    ExpectRejected(MakeGguf({{"long_string", GgufTypeString,
                              IntegerValue(16777217, 8)}}, {}),
                   "failed to read metadata value for key: long_string");
    Bytes array = IntegerValue(GgufTypeUint8);
    AppendLe(array, 10000001, 8);
    ExpectRejected(MakeGguf({{"large_array", GgufTypeArray, array}}, {}),
                   "failed to read metadata value for key: large_array");
    array = IntegerValue(999);
    AppendLe(array, 1, 8);
    ExpectRejected(MakeGguf({{"bad_array", GgufTypeArray, array}}, {}),
                   "failed to read metadata value for key: bad_array");
}

TEST_F(GgufTest, DuplicateTensorNameRejected) {
    ExpectRejected(MakeGguf({}, {{"weight", {1}}, {"weight", {1}}}, 8),
                   "duplicate tensor name: weight");
}

TEST_F(GgufTest, InvalidTensorRankRejected) {
    ExpectRejected(MakeGguf({}, {{"weight", {}}}), "invalid tensor n_dims for: weight");
    ExpectRejected(MakeGguf({}, {{"weight", std::vector<uint64_t>(9, 1)}}),
                   "invalid tensor n_dims for: weight");
}

TEST_F(GgufTest, MissingAlignmentPaddingRejected) {
    Bytes data = MakeGguf({}, {});
    data.resize(24);  // 完整 GGUF 头，但缺少到默认 32 字节数据段的 padding。
    ExpectRejected(data, "GGUF file ended before tensor data section");
}

TEST_F(GgufTest, SupportedTensorStorageSizes) {
    // 独立列出 32 元素的存储字节数，确认正好到 EOF 有效，少一字节被拒绝。
    for (const auto& storage : std::vector<std::pair<uint32_t, size_t>>{
             {GgmlTypeF32, 128}, {GgmlTypeF16, 64}, {GgmlTypeQ40, 18},
             {GgmlTypeQ41, 20}, {GgmlTypeQ80, 34}}) {
        SCOPED_TRACE(storage.first);
        Bytes data = MakeGguf({}, {{"weight", {32}, storage.first}}, storage.second);
        GgufReader reader;
        ASSERT_TRUE(reader.load(Write(data))) << reader.error;
        data.pop_back();
        ExpectRejected(data, "tensor data outside file bounds: weight");
    }
}

TEST_F(GgufTest, QuantizedTensorNeedsCompleteBlocks) {
    for (uint32_t type : {GgmlTypeQ40, GgmlTypeQ41, GgmlTypeQ80}) {
        SCOPED_TRACE(type);
        ExpectRejected(MakeGguf({}, {{"weight", {31}, type}}, 128),
                       "invalid tensor data size for: weight");
    }
}

TEST_F(GgufTest, TensorSizeAndOffsetOverflowRejected) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    ExpectRejected(MakeGguf({}, {{"weight", {max, 2}}}),
                   "invalid tensor data size for: weight");
    ExpectRejected(MakeGguf({}, {{"weight", {max / 4 + 1}}}),
                   "invalid tensor data size for: weight");
    ExpectRejected(MakeGguf({}, {{"weight", {1}, GgmlTypeF32, max}}),
                   "tensor offset overflow for: weight");
    // 单个名为 weight 的一维张量：数据段位于 64，起点未溢出而终点溢出。
    ExpectRejected(MakeGguf({}, {{"weight", {1}, GgmlTypeF32, max - 64}}),
                   "tensor data range overflow for: weight");
}

TEST_F(GgufTest, MultipleTensorsKeepRelativeOffsets) {
    GgufReader reader;
    ASSERT_TRUE(reader.load(Write(MakeGguf({}, {{"first", {2}, GgmlTypeF32, 0},
                                               {"second", {4}, GgmlTypeF16, 32}}, 40))))
        << reader.error;
    ASSERT_EQ(reader.tensors.size(), 2u);
    EXPECT_EQ(reader.tensors[0].offset, 0u);
    EXPECT_EQ(reader.tensors[1].name, "second");
    EXPECT_EQ(reader.tensors[1].offset, 32u);
    EXPECT_EQ(reader.tensors[1].shape, (std::vector<uint64_t>{4}));
}

TEST_F(GgufTest, ReadsTypedIntegerMetadata) {
    struct IntegerCase {
        uint32_t type;
        size_t width;
        int64_t value;
    };
    for (const auto& sample : std::vector<IntegerCase>{
             {GgufTypeUint8, 1, 255}, {GgufTypeInt8, 1, -128},
             {GgufTypeUint16, 2, 65535}, {GgufTypeInt16, 2, -32768},
             {GgufTypeUint32, 4, 4294967295LL}, {GgufTypeInt32, 4, -2147483648LL},
             {GgufTypeUint64, 8, std::numeric_limits<int64_t>::max()},
             {GgufTypeInt64, 8, std::numeric_limits<int64_t>::min()}}) {
        SCOPED_TRACE(sample.type);
        const std::string path = Write(MakeGguf({
            {"value", sample.type, IntegerValue(static_cast<uint64_t>(sample.value), sample.width)},
        }, {}));
        int64_t value = 0;
        ASSERT_TRUE(GgufGetMetadataInt(path, "value", value));
        EXPECT_EQ(value, sample.value);
    }
    const auto path = Write(MakeGguf({
        {"too_large", GgufTypeUint64, IntegerValue(std::numeric_limits<uint64_t>::max(), 8)},
        {"wrong_type", GgufTypeBool, IntegerValue(1, 1)},
    }, {}));
    int64_t value = 42;
    EXPECT_FALSE(GgufGetMetadataInt(path, "too_large", value));
    EXPECT_FALSE(GgufGetMetadataInt(path, "wrong_type", value));
    EXPECT_FALSE(GgufGetMetadataInt(path, "absent", value));
    EXPECT_EQ(value, 42);
}

TEST_F(GgufTest, ReadsTypedStringAndFloatMetadata) {
    const auto path = Write(MakeGguf({
        {"text", GgufTypeString, StringValue("你好\n\"quoted\"")},
        {"f32", GgufTypeFloat32, IntegerValue(0x3f000000)},
        {"f64", GgufTypeFloat64, IntegerValue(0x3fe0000000000000ULL, 8)},
    }, {}));
    std::string text = "previous";
    ASSERT_TRUE(GgufGetMetadataString(path, "text", text));
    EXPECT_EQ(text, "你好\n\"quoted\"");
    EXPECT_FALSE(GgufGetMetadataString(path, "f32", text));
    EXPECT_FALSE(GgufGetMetadataString(path, "absent", text));
    EXPECT_EQ(text, "你好\n\"quoted\"");
    double value = 1.0;
    ASSERT_TRUE(GgufGetMetadataFloat(path, "f32", value));
    EXPECT_DOUBLE_EQ(value, 0.5);
    ASSERT_TRUE(GgufGetMetadataFloat(path, "f64", value));
    EXPECT_DOUBLE_EQ(value, 0.5);
    EXPECT_FALSE(GgufGetMetadataFloat(path, "text", value));
    EXPECT_FALSE(GgufGetMetadataFloat(path, "absent", value));
    EXPECT_DOUBLE_EQ(value, 0.5);
}

TEST_F(GgufTest, ReadsTypedArraysAndSkipsUnrelatedMetadata) {
    const std::vector<std::string> tokens{"a", "Ġ", "你好", "\n\"\\", ""};
    const std::vector<int32_t> types{1, 3, -1, std::numeric_limits<int32_t>::min()};
    const auto path = Write(MakeGguf({
        {"floats", GgufTypeArray, [] {
            Bytes data = IntegerValue(GgufTypeFloat32);
            AppendLe(data, 2, 8);
            data.resize(data.size() + 8, 0);
            return data;
        }()},
        {"tokens", GgufTypeArray, StringArrayValue(tokens)},
        {"types", GgufTypeArray, IntArrayValue(types)},
        {"empty_strings", GgufTypeArray, StringArrayValue({})},
        {"empty_ints", GgufTypeArray, IntArrayValue({})},
    }, {}));
    std::vector<std::string> strings;
    std::vector<int32_t> integers;
    ASSERT_TRUE(GgufGetMetadataStringArray(path, "tokens", strings));
    ASSERT_TRUE(GgufGetMetadataIntArray(path, "types", integers));
    EXPECT_EQ(strings, tokens);
    EXPECT_EQ(integers, types);
    EXPECT_FALSE(GgufGetMetadataStringArray(path, "types", strings));
    EXPECT_FALSE(GgufGetMetadataIntArray(path, "tokens", integers));
    EXPECT_FALSE(GgufGetMetadataStringArray(path, "absent", strings));
    EXPECT_FALSE(GgufGetMetadataIntArray(path, "absent", integers));
    EXPECT_EQ(strings, tokens);
    EXPECT_EQ(integers, types);
    ASSERT_TRUE(GgufGetMetadataStringArray(path, "empty_strings", strings));
    ASSERT_TRUE(GgufGetMetadataIntArray(path, "empty_ints", integers));
    EXPECT_TRUE(strings.empty());
    EXPECT_TRUE(integers.empty());
}

TEST_F(GgufTest, TypedReadsRejectInvalidHeadersAndCounts) {
    int64_t value = 42;
    EXPECT_FALSE(GgufGetMetadataInt(missing_path_.string(), "value", value));
    Bytes valid = MakeGguf({{"value", GgufTypeInt32, IntegerValue(1)}}, {});
    for (size_t length : {size_t{0}, size_t{4}, size_t{8}, size_t{16}, size_t{23}}) {
        EXPECT_FALSE(GgufGetMetadataInt(Write(Bytes(valid.begin(), valid.begin() + length)),
                                        "value", value));
    }
    Bytes invalid = valid;
    invalid[0] = '!';
    EXPECT_FALSE(GgufGetMetadataInt(Write(invalid), "value", value));
    invalid = valid;
    WriteLe(invalid, 4, 1, 4);
    EXPECT_FALSE(GgufGetMetadataInt(Write(invalid), "value", value));
    for (size_t count_offset : {size_t{8}, size_t{16}}) {
        invalid = valid;
        WriteLe(invalid, count_offset, 1000001, 8);
        EXPECT_FALSE(GgufGetMetadataInt(Write(invalid), "value", value));
    }
    EXPECT_EQ(value, 42);
}

TEST_F(GgufTest, TypedReadsRejectTruncationAndDuplicateKeysAfterTarget) {
    int64_t value = 42;
    const MetadataEntry target{"value", GgufTypeInt32, IntegerValue(1)};
    EXPECT_FALSE(GgufGetMetadataInt(Write(MakeGguf({target, target}, {})), "value", value));
    EXPECT_FALSE(GgufGetMetadataInt(Write(MakeGguf({target,
        {"truncated", GgufTypeString, IntegerValue(100, 8)},
    }, {})), "value", value));
    EXPECT_EQ(value, 42);
}

TEST_F(GgufTest, TypedArraysRejectInvalidLengthsTypesAndTruncation) {
    std::vector<std::string> strings{"previous"};
    std::vector<int32_t> integers{42};
    for (const auto& sample : std::vector<std::pair<uint32_t, uint64_t>>{
             {GgufTypeString, 10000001}, {GgufTypeInt32, 10000001},
             {GgufTypeString, 100}, {GgufTypeInt32, 100},
             {999, 0}, {GgufTypeArray, 0}}) {
        SCOPED_TRACE(sample.first);
        SCOPED_TRACE(sample.second);
        Bytes array = IntegerValue(sample.first);
        AppendLe(array, sample.second, 8);
        const auto path = Write(MakeGguf({{"array", GgufTypeArray, array}}, {}));
        EXPECT_FALSE(GgufGetMetadataStringArray(path, "array", strings));
        EXPECT_FALSE(GgufGetMetadataIntArray(path, "array", integers));
    }
    EXPECT_EQ(strings, (std::vector<std::string>{"previous"}));
    EXPECT_EQ(integers, (std::vector<int32_t>{42}));
}

}  // namespace
