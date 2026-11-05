#pragma once

#include <random>
#include <vector>
#include "tensor.h"

struct SamplingParams {
    float temperature = 0.0f;
    int top_k = 0;  // 0 表示禁用 Top-k。
    unsigned int seed = 0;  // 仅构造 Sampler 时使用，0 表示使用随机设备。
};

struct Candidate {
    int idx;
    float value;
};

const float GreedyTemperature = 1e-6f;

class Sampler {
public:
    explicit Sampler(unsigned int seed = 0);
    explicit Sampler(const SamplingParams& params);

    int Sample(const Tensor& logits, const SamplingParams& params);

private:

    void CheckLogits(const Tensor& logits) const;
    void CheckTemperature(float temperature) const;
    void CheckTopK(int top_k) const;

    int ArgMax(const Tensor& x) const;
    // 返回 value 最大的候选项的 idx。
    int ArgMax(const std::vector<Candidate>& candidates) const;
    int SampleGreedy(const Tensor& logits) const;
    int SampleTemperature(const Tensor& logits, float temperature);
    int SampleTemperature(const std::vector<Candidate>& candidates, float temperature);
    int SampleTopK(const Tensor& logits, float temperature, int top_k);

    std::mt19937 rng_;
};
