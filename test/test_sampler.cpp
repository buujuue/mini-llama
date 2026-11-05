// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "sampler.h"

namespace {

// 适配上游统一入口测试，保留本项目的 Sampler 名称及私有采样辅助函数。
TEST(SamplerTest, SampleDefaultParamsAreGreedy) {
    Tensor logits({5});
    logits.data = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f};
    Sampler sampler(42);
    EXPECT_EQ(sampler.Sample(logits, SamplingParams{}), 1);
}

TEST(SamplerTest, SampleZeroAndTinyTemperaturesAreGreedy) {
    Tensor logits({5});
    logits.data = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f};
    Sampler sampler(42);
    SamplingParams params;
    for (float temperature : {0.0f, 1e-7f}) {
        params.temperature = temperature;
        for (int top_k : {0, 1, 3, 100}) {
            params.top_k = top_k;
            for (int i = 0; i < 20; ++i) {
                EXPECT_EQ(sampler.Sample(logits, params), 1);
            }
        }
    }
}

TEST(SamplerTest, SampleTopKOneIsGreedyAndPreservesRandomState) {
    Tensor logits({4});
    logits.data = {0.0f, 1.0f, 3.0f, 2.0f};
    Sampler sampler(42);
    Sampler untouched(42);
    SamplingParams params;
    params.temperature = 100.0f;
    params.top_k = 1;
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(sampler.Sample(logits, params), 2);
    }
    params.top_k = 0;
    for (int i = 0; i < 40; ++i) {
        EXPECT_EQ(sampler.Sample(logits, params), untouched.Sample(logits, params));
    }
}

TEST(SamplerTest, SampleTemperatureUsesWholeVocabAndReproducesWeightedDraws) {
    Tensor logits({4});
    logits.data = {0.0f, std::log(3.0f), std::log(2.0f), std::log(4.0f)};
    const auto original = logits.data;
    constexpr int draws = 4096;
    for (float temperature : {1.0f, 0.5f}) {
        SamplingParams params;
        params.temperature = temperature;
        params.seed = 123;
        Sampler first(params);
        Sampler second(params.seed);
        std::vector<int> counts(4, 0);
        for (int i = 0; i < draws; ++i) {
            const int idx = first.Sample(logits, params);
            ASSERT_GE(idx, 0);
            ASSERT_LT(idx, 4);
            EXPECT_EQ(idx, second.Sample(logits, params));
            ++counts[idx];
        }
        const std::vector<float> expected = temperature == 1.0f
            ? std::vector<float>{0.1f, 0.3f, 0.2f, 0.4f}
            : std::vector<float>{1.0f / 30, 9.0f / 30, 4.0f / 30, 16.0f / 30};
        for (size_t i = 0; i < counts.size(); ++i) {
            EXPECT_NEAR(static_cast<float>(counts[i]) / draws, expected[i], 0.03f);
        }
    }
    EXPECT_EQ(logits.data, original);
}

TEST(SamplerTest, SampleTopKIsReproducibleAndKeepsExactlyK) {
    Tensor logits({5});
    logits.data = {0.0f, 5.0f, 1.0f, 5.0f, 5.0f};
    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;
    params.seed = 321;
    Sampler sampler(params);
    Sampler repeated(params.seed);
    bool saw_first = false;
    bool saw_second = false;
    for (int i = 0; i < 100; ++i) {
        const int idx = sampler.Sample(logits, params);
        EXPECT_EQ(idx, repeated.Sample(logits, params));
        EXPECT_TRUE(idx == 1 || idx == 3) << idx;
        saw_first |= idx == 1;
        saw_second |= idx == 3;
    }
    EXPECT_TRUE(saw_first);
    EXPECT_TRUE(saw_second);
}

TEST(SamplerTest, SampleRejectsInvalidTemperatureBeforeGreedyShortcut) {
    const Tensor logits({3});
    Sampler sampler(42);
    SamplingParams params;
    for (float temperature : {-0.1f, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        params.temperature = temperature;
        for (int top_k : {0, 1, 2}) {
            params.top_k = top_k;
            EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
        }
    }
}

TEST(SamplerTest, SampleRejectsNegativeTopKBeforeGreedyShortcut) {
    const Tensor logits({3});
    Sampler sampler(42);
    SamplingParams params;
    for (int top_k : {-1, std::numeric_limits<int>::min()}) {
        params.top_k = top_k;
        for (float temperature : {0.0f, 1.0f}) {
            params.temperature = temperature;
            EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
        }
    }
}

TEST(SamplerTest, SampleRejectsEmptyLogitsInEveryMode) {
    const Tensor logits;
    Sampler sampler(42);
    SamplingParams params;
    for (float temperature : {0.0f, 1.0f}) {
        params.temperature = temperature;
        for (int top_k : {0, 1, 2}) {
            params.top_k = top_k;
            EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
        }
    }
}

TEST(SamplerTest, SampleRejectsNonfiniteLogitsInEveryMode) {
    Tensor logits({3});
    Sampler sampler(42);
    SamplingParams params;
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        logits.data = {1.0f, value, 2.0f};
        for (float temperature : {0.0f, 1.0f}) {
            params.temperature = temperature;
            for (int top_k : {0, 1, 2}) {
                params.top_k = top_k;
                EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
            }
        }
    }
}

// 适配上游 tests/gtest/test_sampler.cc 的 Top-k 测试，通过统一 Sample 入口验证。
TEST(SamplerTest, TopKOneIsGreedy) {
    Tensor logits({5});
    logits.data = {1.0f, 5.0f, 3.0f, 5.0f, 2.0f};
    Sampler sampler(42);
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(sampler.Sample(logits, SamplingParams{100.0f, 1}), 1);
    }
}

TEST(SamplerTest, TopKLimitsCandidatesAndPreservesLogits) {
    Tensor logits({5});
    logits.data = {0.0f, 3.0f, 1.0f, 4.0f, 2.0f};
    const auto original = logits.data;
    Sampler sampler(789);
    for (int i = 0; i < 100; ++i) {
        const int idx = sampler.Sample(logits, SamplingParams{1.0f, 2});
        EXPECT_TRUE(idx == 1 || idx == 3) << idx;
    }
    EXPECT_EQ(logits.data, original);
}

TEST(SamplerTest, ZeroAndTinyTemperaturesAreGreedy) {
    Tensor logits({5});
    logits.data = {1.0f, 2.0f, 5.0f, 5.0f, 0.0f};
    Sampler sampler(42);
    for (float temperature : {0.0f, 1e-7f}) {
        for (int i = 0; i < 20; ++i) {
            EXPECT_EQ(sampler.Sample(logits, SamplingParams{temperature, 3}), 2);
        }
    }
}

TEST(SamplerTest, TopKLargerThanVocabIsClamped) {
    Tensor logits({4});
    logits.data = {0.0f, 1.0f, 2.0f, 3.0f};
    Sampler oversized(42);
    Sampler exact(42);
    for (int i = 0; i < 40; ++i) {
        EXPECT_EQ(oversized.Sample(logits, SamplingParams{1.0f, 100}),
                  exact.Sample(logits, SamplingParams{1.0f, 4}));
    }

    const Tensor single({1}, 3.0f);
    EXPECT_EQ(oversized.Sample(single, SamplingParams{1.0f, 100}), 0);
}

TEST(SamplerTest, TiedLogitsKeepExactlyKCandidatesByIndex) {
    Tensor logits({5});
    logits.data = {0.0f, 5.0f, 1.0f, 5.0f, 5.0f};
    Sampler sampler(321);
    bool saw_first = false;
    bool saw_second = false;
    for (int i = 0; i < 100; ++i) {
        const int idx = sampler.Sample(logits, SamplingParams{1.0f, 2});
        EXPECT_TRUE(idx == 1 || idx == 3) << idx;
        saw_first |= idx == 1;
        saw_second |= idx == 3;
    }
    EXPECT_TRUE(saw_first);
    EXPECT_TRUE(saw_second);
}

TEST(SamplerTest, FixedSeedReproducesTopKWithEitherConstructor) {
    Tensor logits({10});
    for (size_t i = 0; i < logits.size(); ++i) {
        logits.data[i] = static_cast<float>(i) * 0.5f;
    }
    SamplingParams params;
    params.seed = 456;
    Sampler first(params);
    Sampler second(params.seed);
    for (int i = 0; i < 40; ++i) {
        EXPECT_EQ(first.Sample(logits, SamplingParams{0.8f, 3}),
                  second.Sample(logits, SamplingParams{0.8f, 3}));
    }
}

// 补充候选项原始 token 下标、温度权重、数值稳定性与异常边界。
TEST(SamplerTest, TemperatureControlsCandidateProbabilities) {
    Tensor logits({4}, -100.0f);
    logits.data[1] = 0.0f;
    logits.data[3] = std::log(3.0f);
    constexpr int draws = 4096;
    for (float temperature : {1.0f, 0.5f}) {
        Sampler sampler(123);
        int high_count = 0;
        for (int i = 0; i < draws; ++i) {
            const int idx = sampler.Sample(logits, SamplingParams{temperature, 2});
            ASSERT_TRUE(idx == 1 || idx == 3) << idx;
            high_count += idx == 3;
        }
        const float expected = temperature == 1.0f ? 0.75f : 0.9f;
        EXPECT_NEAR(static_cast<float>(high_count) / draws, expected, 0.03f);
    }
}

TEST(SamplerTest, LargeLogitsUseStableSoftmax) {
    Tensor logits({4});
    logits.data = {-10000.0f, 9999.0f, -10000.0f, 10000.0f};
    Sampler sampler(42);
    for (int i = 0; i < 40; ++i) {
        EXPECT_EQ(sampler.Sample(logits, SamplingParams{0.01f, 2}), 3);
    }
}

TEST(SamplerTest, RejectsNegativeTopK) {
    const Tensor logits({3});
    Sampler sampler(42);
    for (int top_k : {-1, std::numeric_limits<int>::min()}) {
        EXPECT_THROW(sampler.Sample(logits, SamplingParams{1.0f, top_k}), std::runtime_error);
        EXPECT_THROW(sampler.Sample(logits, SamplingParams{0.0f, top_k}), std::runtime_error);
    }
}

TEST(SamplerTest, RejectsInvalidTemperatureBeforeGreedyShortcut) {
    const Tensor logits({3});
    Sampler sampler(42);
    for (float temperature : {-0.1f, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        EXPECT_THROW(sampler.Sample(logits, SamplingParams{temperature, 1}), std::runtime_error);
        EXPECT_THROW(sampler.Sample(logits, SamplingParams{temperature, 2}), std::runtime_error);
    }
}

TEST(SamplerTest, RejectsEmptyLogits) {
    const Tensor logits;
    Sampler sampler(42);
    EXPECT_THROW(sampler.Sample(logits, SamplingParams{1.0f, 2}), std::runtime_error);
    EXPECT_THROW(sampler.Sample(logits, SamplingParams{0.0f, 1}), std::runtime_error);
}

TEST(SamplerTest, RejectsNonfiniteLogitsBeforeFiltering) {
    Tensor logits({3});
    Sampler sampler(42);
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        logits.data = {1.0f, value, 2.0f};
        EXPECT_THROW(sampler.Sample(logits, SamplingParams{1.0f, 2}), std::runtime_error);
        EXPECT_THROW(sampler.Sample(logits, SamplingParams{0.0f, 1}), std::runtime_error);
    }
}

}  // namespace
