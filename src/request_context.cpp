// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "request_context.h"
#include <chrono>
#include <cstdio>
#include <ctime>
#include <ratio>

namespace {

std::string FormatMilliseconds(double milliseconds) {
    const int length = std::snprintf(nullptr, 0, "%.3f", milliseconds);
    std::string buffer(static_cast<std::size_t>(length), '\0');
    std::snprintf(buffer.data(), buffer.size() + 1, "%.3f", milliseconds);
    return buffer;
}

} // namespace

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
    std::string out = "trace summary: trace_id=" + trace_id_;
    out += std::string("\nmode=") + ToString(mode_);
    out += std::string("\nstatus=") + (error_.empty() ? "ok" : "error");
    out += std::string("\nbackend=") + ToString(backend_);
    out += "\nmodel=" + model_path_;
    out += "\nprompt_tokens=" + std::to_string(prompt_tokens_);
    out += "\nprefill_tokens=" + std::to_string(prefill_tokens_);
    out += "\ndecode_tokens=" + std::to_string(decode_tokens_);
    out += "\ngenerated_tokens=" + std::to_string(generated_tokens_);
    out += "\nmodel_load_ms=" + FormatMilliseconds(model_load_ms_);
    out += "\ntokenize_ms=" + FormatMilliseconds(tokenize_ms_);
    out += "\nprefill_ms=" + FormatMilliseconds(prefill_ms_);
    out += "\ndecode_ms=" + FormatMilliseconds(decode_ms_);
    out += "\nsample_ms=" + FormatMilliseconds(sample_ms_);
    out += "\ntotal_ms=" + FormatMilliseconds(total_ms_);
    if (!error_.empty()) {
        out += "\nerror=\"" + error_ + "\"";
    }
    return out;
}

std::vector<std::string> RequestContext::FormatRequestTraceEvents() const {
    std::vector<std::string> lines;
    lines.reserve(events_.size());
    for (const TraceEvent& event : events_) {
        std::string out = "trace event: trace_id=" + trace_id_;
        out += std::string("\nstage=") + ToString(event.type);
        out += "\ntokens=" + std::to_string(event.tokens);
        out += "\nelapsed_ms=" + FormatMilliseconds(event.elapsed_ms);
        if (!event.datail.empty()) {
            out += "\ndetail=\"" + event.datail + "\"";
        }
        lines.push_back(out);
    }
    return lines;
}
