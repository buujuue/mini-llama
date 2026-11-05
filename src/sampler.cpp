// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "tensor.h"

Sampler::Sampler(unsigned int seed) {
    if (seed != 0) {
        rng_.seed(seed);
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

Sampler::Sampler(const SamplingParams& params) : Sampler(params.seed) {}

void Sampler::CheckLogits(const Tensor& logits) const {
    if (logits.size() == 0) {
        throw std::runtime_error(
            "[Sampler] CheckLogits 不能从空 logits 采样：元素数: 0。");
    }
    for (size_t i = 0; i < logits.size(); ++i){
        if (!std::isfinite(logits.data[i])){
            throw std::runtime_error(
                "[Sampler] CheckLogits 要求 logits 元素为有限值：idx: "
                + std::to_string(i)
                + "，value: " + std::to_string(logits.data[i])
                + "，元素数: " + std::to_string(logits.size()) + "。");
        }
    }
}

int Sampler::ArgMax(const Tensor& x) const {
    int best = 0;
    float best_val = x.data[0];
    for (int i = 1; i < x.size(); ++i) {
        if (x.data[i] > best_val) {
            best_val = x.data[i];
            best = i;
        }
    }
    return best;
}

int Sampler::ArgMax(const std::vector<Candidate>& candidates) const {
    if (candidates.empty()) {
        throw std::runtime_error("[Sampler] ArgMax 不能从空 candidates 中取最大值。");
    }

    int best = candidates[0].idx;
    float best_val = candidates[0].value;
    for (size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].value > best_val) {
            best_val = candidates[i].value;
            best = candidates[i].idx;
        }
    }
    return best;
}

int Sampler::SampleGreedy(const Tensor& logits) const{
    return ArgMax(logits);
}

void Sampler::CheckTemperature(float temperature) const{
    if (!std::isfinite(temperature) || temperature < 0.0f) {
        throw std::runtime_error(
            "[Sampler] CheckTemperature 要求 temperature 为有限非负数：temperature: "
            + std::to_string(temperature) + "。");
    }

}

void Sampler::CheckTopK(int top_k) const{
    if (top_k < 0){
        throw std::runtime_error(
            "[Sampler] CheckTopK 要求 top_k 为非负整数：top_k: "
            + std::to_string(top_k) + "。");
    }
}

int Sampler::SampleTemperature(const Tensor& logits, float temperature){
    if (temperature < GreedyTemperature) {
        return SampleGreedy(logits);
    }

    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        candidates.push_back({static_cast<int>(i), logits.data[i]});
    }
    return SampleTemperature(candidates, temperature);
}

int Sampler::SampleTemperature(const std::vector<Candidate>& candidates,
                               float temperature) {
    if (temperature < GreedyTemperature){
        return ArgMax(candidates);
    }

    float max_val = candidates[0].value;
    for (size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].value > max_val) {
            max_val = candidates[i].value;
        }
    }

    std::vector<float> probs(candidates.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp((candidates[i].value - max_val) / temperature);
        sum += probs[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i] / sum;
        if (r <= cumulative) {
            return candidates[i].idx;
        }
    }
    return candidates.back().idx;
}

int Sampler::SampleTopK(const Tensor& logits, float temperature, int top_k) {
    if (top_k == 0) {
        throw std::runtime_error(
            "[Sampler] SampleTopK 要求 top_k 为正整数：top_k: 0。");
    }
    if (top_k == 1) {
        return SampleGreedy(logits);
    }

    const size_t candidate_size = std::min(static_cast<size_t>(top_k), logits.size());
    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        candidates.push_back({static_cast<int>(i), logits.data[i]});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.value == b.value) {
                      return a.idx < b.idx;
                  }
                  return a.value > b.value;
              });
    candidates.resize(candidate_size);
    return SampleTemperature(candidates, temperature);
}

int Sampler::Sample(const Tensor& logits, const SamplingParams& params) {
    CheckLogits(logits);
    CheckTemperature(params.temperature);
    CheckTopK(params.top_k);
    if (params.temperature < GreedyTemperature || params.top_k == 1) {
        return SampleGreedy(logits);
    }
    if (params.top_k > 1) {
        return SampleTopK(logits, params.temperature, params.top_k);
    }
    return SampleTemperature(logits, params.temperature);
}
