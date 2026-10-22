// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "request_context.h"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <ratio>
#include <sstream>

std::size_t RequestContext::request_counter_ = 0;

const char* RequestContext::ToString(RequestMode mode) {
    switch (mode) {
        case RequestMode::Generate: return "generate";
        case RequestMode::Run: return "run";
    }
    return "unknown";
}

const char* RequestContext::ToString(BackendKind backend) {
    switch (backend) {
        case BackendKind::Cpu: return "cpu";
        case BackendKind::Cuda: return "cuda";
    }
    return "unknown";
}

const char* RequestContext::ToString(TraceEventType type) {
    switch (type) {
        case TraceEventType::model_load: return "model_load";
        case TraceEventType::quantize: return "quantize";
        case TraceEventType::prompt_build: return "prompt_build";
        case TraceEventType::tokenize: return "tokenize";
        case TraceEventType::prefill: return "prefill";
        case TraceEventType::decode: return "decode";
        case TraceEventType::sample: return "sample";
    }
    return "unknown";
}

void RequestContext::Start() {
    start_time_ = RequestClock::now();
}

double RequestContext::ElapsedMS(RequestClock::time_point begin) {
    return std::chrono::duration<double, std::milli>(RequestClock::now() - begin).count();
}

void RequestContext::Finish(){
    total_ms_ = ElapsedMS(start_time_);
}

void RequestContext::RecordEvent(TraceEventType type, double elapsed_ms, size_t tokens, const std::string& detail){
    events_.push_back({type, elapsed_ms, tokens, detail});
}

std::string RequestContext::NewTraceId(){
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    const unsigned long long seq = ++request_counter_;
    std::tm tm_snapshot;
#if defined (_WIN32)
    localtime_s(&tm_snapshot, &time);
#else 
    localtime_r(&time, &tm_snapshot);
#endif
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &tm_snapshot);

    std::string sequence = std::to_string(seq);
    if (sequence.size() < 6) {
        sequence.insert(0, 6 - sequence.size(), '0');
    }
    return std::string("req_") + timestamp + "_" + sequence;
}
RequestContext::RequestContext(RequestMode mode, BackendKind backend, const std::string& model_path){
    trace_id_ = NewTraceId();
    mode_ = mode;
    backend_ = backend;
    model_path_ = model_path;
    Start();
}

std::string RequestContext::FormatRequestTraceSummary() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "trace summary: trace_id=" << trace_id_
        << " mode=" << ToString(mode_)
        << " status=" << (error_.empty() ? "ok" : "error")
        << " backend=" << ToString(backend_)
        << " model=" << model_path_
        << " prompt_tokens=" << prompt_tokens_
        << " prefill_tokens=" << prefill_tokens_
        << " decode_tokens=" << decode_tokens_
        << " generated_tokens=" << generated_tokens_
        << " model_load_ms=" << model_load_ms_
        << " tokenize_ms=" << tokenize_ms_
        << " prefill_ms=" << prefill_ms_
        << " decode_ms=" << decode_ms_
        << " sample_ms=" << sample_ms_
        << " total_ms=" << total_ms_;
    if (!error_.empty()) {
        out << " error=\"" << error_ << "\"";
    }
    return out.str();
}

std::vector<std::string> RequestContext::FormatRequestTraceEvents() const {
    std::vector<std::string> lines;
    lines.reserve(events_.size());
    for (const TraceEvent& event : events_) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3);
        out << "trace event: trace_id=" << trace_id_
            << " stage=" << ToString(event.type)
            << " tokens=" << event.tokens
            << " elapsed_ms=" << event.elapsed_ms;
        if (!event.datail.empty()) {
            out << " detail=\"" << event.datail << "\"";
        }
        lines.push_back(out.str());
    }
    return lines;
}
