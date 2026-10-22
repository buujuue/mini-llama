// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "request_context.h"

namespace {

std::string SummaryValue(const std::string& summary, const std::string& key) {
    const std::string marker = " " + key + "=";
    const auto field = summary.find(marker);
    if (field == std::string::npos) {
        return {};
    }
    const auto begin = field + marker.size();
    return summary.substr(begin, summary.find(' ', begin) - begin);
}

// 前三项移植自 references/mini-llama-cpp/tests/test_request_context.cc。
// 通过公开格式化结果验收私有状态，不直接访问或修改内部字段。
TEST(RequestContextTest, StartsWithTraceMetadata) {
    const RequestContext request(RequestMode::Generate, BackendKind::Cpu, "models/tiny");
    const std::string summary = request.FormatRequestTraceSummary();
    EXPECT_EQ(summary.find("trace summary: trace_id=req_"), 0u);
    EXPECT_EQ(SummaryValue(summary, "mode"), "generate");
    EXPECT_EQ(SummaryValue(summary, "backend"), "cpu");
    EXPECT_EQ(SummaryValue(summary, "model"), "models/tiny");
    EXPECT_EQ(SummaryValue(summary, "status"), "ok");
    EXPECT_FALSE(request.IsError());
    EXPECT_TRUE(request.FormatRequestTraceEvents().empty());
}

TEST(RequestContextTest, RecordsEventsAndError) {
    RequestContext request(RequestMode::Run, BackendKind::Cuda, "models/chat");
    request.RecordEvent(TraceEventType::tokenize, 1.25, 4, "prompt");
    request.SetError("boom");
    request.Finish();

    const std::string summary = request.FormatRequestTraceSummary();
    const std::string trace_id = SummaryValue(summary, "trace_id");
    EXPECT_TRUE(request.IsError());
    EXPECT_EQ(SummaryValue(summary, "mode"), "run");
    EXPECT_EQ(SummaryValue(summary, "backend"), "cuda");
    EXPECT_EQ(SummaryValue(summary, "model"), "models/chat");
    EXPECT_EQ(SummaryValue(summary, "status"), "error");
    EXPECT_NE(summary.find(" error=\"boom\""), std::string::npos);
    ASSERT_FALSE(SummaryValue(summary, "total_ms").empty());
    EXPECT_GE(std::stod(SummaryValue(summary, "total_ms")), 0.0);
    const std::vector<std::string> events = request.FormatRequestTraceEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "trace event: trace_id=" + trace_id
                        + " stage=tokenize tokens=4 elapsed_ms=1.250 detail=\"prompt\"");
}

TEST(RequestContextTest, TraceFormattingIncludesStableFields) {
    RequestContext request(RequestMode::Generate, BackendKind::Cpu, "models/tiny");
    request.RecordEvent(TraceEventType::prefill, 3.0, 3, "batch");
    const std::string summary = request.FormatRequestTraceSummary();
    const std::string trace_id = SummaryValue(summary, "trace_id");

    // 统计字段尚无公开写入接口，只验收当前已实现的初始值与事件输出。
    EXPECT_EQ(summary, "trace summary: trace_id=" + trace_id
                       + " mode=generate status=ok backend=cpu model=models/tiny"
                         " prompt_tokens=0 prefill_tokens=0 decode_tokens=0 generated_tokens=0"
                         " model_load_ms=0.000 tokenize_ms=0.000 prefill_ms=0.000"
                         " decode_ms=0.000 sample_ms=0.000 total_ms=0.000");
    const std::vector<std::string> events = request.FormatRequestTraceEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "trace event: trace_id=" + trace_id
                        + " stage=prefill tokens=3 elapsed_ms=3.000 detail=\"batch\"");
}

TEST(RequestContextTest, SequentialRequestsHaveDistinctTraceIds) {
    const RequestContext first(RequestMode::Generate, BackendKind::Cpu, "model");
    const RequestContext second(RequestMode::Generate, BackendKind::Cpu, "model");
    const auto first_id = SummaryValue(first.FormatRequestTraceSummary(), "trace_id");
    const auto second_id = SummaryValue(second.FormatRequestTraceSummary(), "trace_id");
    EXPECT_NE(first_id, second_id);
    ASSERT_NE(first_id.find_last_of('_'), std::string::npos);
    ASSERT_NE(second_id.find_last_of('_'), std::string::npos);
    EXPECT_EQ(std::stoull(second_id.substr(second_id.find_last_of('_') + 1)),
              std::stoull(first_id.substr(first_id.find_last_of('_') + 1)) + 1);
}

TEST(RequestContextTest, EventsPreserveStageOrderAndOmitEmptyDetail) {
    RequestContext request(RequestMode::Generate, BackendKind::Cpu, "model");
    const std::vector<TraceEventType> types{
        TraceEventType::model_load, TraceEventType::quantize, TraceEventType::prompt_build,
        TraceEventType::tokenize, TraceEventType::prefill, TraceEventType::decode,
        TraceEventType::sample};
    const std::vector<std::string> names{
        "model_load", "quantize", "prompt_build", "tokenize", "prefill", "decode", "sample"};
    for (const auto type : types) {
        request.RecordEvent(type, 1.23456, 0, "");
    }
    const auto trace_id = SummaryValue(request.FormatRequestTraceSummary(), "trace_id");
    const auto events = request.FormatRequestTraceEvents();
    ASSERT_EQ(events.size(), names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        SCOPED_TRACE(names[i]);
        EXPECT_EQ(events[i], "trace event: trace_id=" + trace_id + " stage=" + names[i]
                            + " tokens=0 elapsed_ms=1.235");
    }
}

TEST(RequestContextTest, FinishMeasuresElapsedTimeFromConstruction) {
    const auto begin = RequestClock::now();
    RequestContext request(RequestMode::Run, BackendKind::Cpu, "model");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    request.Finish();
    const double upper_ms =
        std::chrono::duration<double, std::milli>(RequestClock::now() - begin).count();
    const auto total = SummaryValue(request.FormatRequestTraceSummary(), "total_ms");
    ASSERT_FALSE(total.empty());
    const double total_ms = std::stod(total);
    EXPECT_GE(total_ms, 2.0);
    EXPECT_LE(total_ms, upper_ms + 0.001);
}

TEST(RequestContextTest, ErrorCanBeReplacedAndCleared) {
    RequestContext request(RequestMode::Run, BackendKind::Cpu, "model");
    const RequestContext& view = request;
    request.SetError("first error");
    request.SetError("latest error");
    EXPECT_TRUE(view.IsError());
    const auto summary = view.FormatRequestTraceSummary();
    EXPECT_NE(summary.find(" error=\"latest error\""), std::string::npos);
    EXPECT_EQ(summary.find("first error"), std::string::npos);

    request.SetError("");
    EXPECT_FALSE(view.IsError());
    const auto cleared = view.FormatRequestTraceSummary();
    EXPECT_EQ(SummaryValue(cleared, "status"), "ok");
    EXPECT_EQ(cleared.find(" error="), std::string::npos);
}

} // namespace
