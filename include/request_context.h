#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>


enum class RequestMode {
    Generate,
    Run,
};

enum class BackendKind {
    Cpu,
    Cuda,
};

enum class TraceEventType{
    model_load,
    quantize,
    prompt_build,
    tokenize,
    prefill,
    decode,
    sample,
};

struct TraceEvent{
    TraceEventType type;
    double elapsed_ms = 0.0;
    size_t tokens = 0;
    std::string datail;
};

using RequestClock = std::chrono::steady_clock;

class RequestContext {
public:
    RequestContext(RequestMode mode, BackendKind backend, const std::string& model_path);
    void Finish();
    void RecordEvent(TraceEventType type, double elapsed_ms, size_t tokens,const std::string& detail); 
    void SetError(const std::string& message){error_ = message;}
    bool IsError() const {return !error_.empty();}
    std::string FormatRequestTraceSummary() const;
    std::vector<std::string> FormatRequestTraceEvents() const;
private: 
    static std::size_t request_counter_;

    std::string trace_id_;
    RequestMode mode_;
    BackendKind backend_;
    std::string model_path_;

    RequestClock::time_point start_time_;
    double total_ms_ = 0.0;

    size_t prompt_tokens_ = 0;
    size_t prefill_tokens_ = 0;
    size_t decode_tokens_ = 0;
    size_t generated_tokens_ = 0;

    double model_load_ms_ = 0.0;
    double tokenize_ms_ = 0.0;
    double prefill_ms_ = 0.0;
    double decode_ms_ = 0.0;
    double sample_ms_ = 0.0;

    std::string error_;
    std::vector<TraceEvent> events_;

    static const char* ToString(RequestMode mode);
    static const char* ToString(BackendKind backend);
    static const char* ToString(TraceEventType type);

    double ElapsedMS(RequestClock::time_point begin);
    std::string NewTraceId();
    void Start();
};
