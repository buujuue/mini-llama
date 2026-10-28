#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"

enum class Mode {
    Naive,
    Threaded,
};

constexpr Mode MODE = Mode::Naive;

static std::string ShapeToString(const Tensor& tensor) {
    std::string shape = "[";
    for (size_t axis = 0; axis < tensor.GetNumDims(); ++axis) {
        if (axis > 0) {
            shape += ", ";
        }
        shape += std::to_string(tensor.GetShape()[axis]);
    }
    return shape + "]";
}

static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
    size_t m = a.GetShape()[0];
    size_t inner_dim = a.GetShape()[1];
    size_t n = b.GetShape()[1];
    Tensor c({m, n}, 0.0f);
    const float* a_data = a.Data();
    const float* b_data = b.Data();
    float* c_data = c.Data();
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < inner_dim; ++k) {
                sum += a_data[i * inner_dim + k] * b_data[k * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    return c;
}

Tensor Matmul(const Tensor& a, const Tensor& b) {
    if (a.GetNumDims() != 2 || b.GetNumDims() != 2) {
        throw std::runtime_error(
            "Ops::Matmul 要求两个输入均为二维张量：预期维数: 2，a 维数: " + std::to_string(a.GetNumDims()) +
            "，a shape: " + ShapeToString(a) + "，b 维数: " + std::to_string(b.GetNumDims()) +
            "，b shape: " + ShapeToString(b) + "。");
    }

    if (a.GetShape()[1] != b.GetShape()[0]) {
        throw std::runtime_error("Ops::Matmul 的内部维度不匹配：要求 a.shape[1] == b.shape[0]，a.shape[1]: " +
                                 std::to_string(a.GetShape()[1]) + "，b.shape[0]: " + std::to_string(b.GetShape()[0]) +
                                 "，a shape: " + ShapeToString(a) + "，b shape: " + ShapeToString(b) + "。");
    }
    switch (MODE) {
        case Mode::Naive:
            return MatmulNaive(a, b);
        case Mode::Threaded:
            throw std::logic_error("Ops::MatmulThreaded 尚未实现：mode: Threaded，a shape: " + ShapeToString(a) +
                                   "，b shape: " + ShapeToString(b) + "。");
        default:
            throw std::logic_error("Ops::未知的 Matmul 模式：mode: " + std::to_string(static_cast<int>(MODE)) +
                                   "，a shape: " + ShapeToString(a) + "，b shape: " + ShapeToString(b) + "。");
    }
}

static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {

    const size_t in_features = weight.GetShape()[1];
    const size_t out_features = weight.GetShape()[0];
    Tensor result(x.GetNumDims() == 1 ? std::vector<size_t>{out_features} : std::vector<size_t>{1, out_features}, 0.0f);
    const float* x_data = x.Data();
    const float* weight_data = weight.Data();
    float* result_data = result.Data();
    for (size_t j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (size_t k = 0; k < in_features; ++k) {
            sum += x_data[k] * weight_data[j * in_features + k];
        }
        result_data[j] = sum;
    }
    return result;
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
    if (weight.GetNumDims() != 2) {
        throw std::runtime_error(
            "Ops::Linear 要求权重为二维张量 [out_features, in_features]：预期维数: 2，weight 维数: " +
            std::to_string(weight.GetNumDims()) + "，weight shape: " + ShapeToString(weight) +
            "，x shape: " + ShapeToString(x) + "。");
    }

    size_t in_features;
    if (x.GetNumDims() == 1) {
        in_features = x.GetShape()[0];
    } else if (x.GetNumDims() == 2 && x.GetShape()[0] == 1) {
        in_features = x.GetShape()[1];
    } else {
        throw std::runtime_error(
            "Ops::Linear 要求 x 形状为 [in_features] 或 [1, in_features]：x 维数: " + std::to_string(x.GetNumDims()) +
            "，x shape: " + ShapeToString(x) + "，weight shape: " + ShapeToString(weight) + "。");
    }

    if (weight.GetShape()[1] != in_features) {
        throw std::runtime_error(
            "Ops::Linear 的输入特征数不匹配：要求 weight.shape[1] == in_features，weight.shape[1]: " +
            std::to_string(weight.GetShape()[1]) + "，in_features: " + std::to_string(in_features) +
            "，x shape: " + ShapeToString(x) + "，weight shape: " + ShapeToString(weight) + "。");
    }

    switch (MODE) {
        case Mode::Naive:
            return LinearNaive(x, weight);
        case Mode::Threaded:
            throw std::logic_error("Ops::LinearThreaded 尚未实现：mode: Threaded，x shape: " + ShapeToString(x) +
                                   "，weight shape: " + ShapeToString(weight) + "。");
        default:
            throw std::logic_error("Ops::未知的 Linear 模式：mode: " + std::to_string(static_cast<int>(MODE)) +
                                   "，x shape: " + ShapeToString(x) + "，weight shape: " + ShapeToString(weight) +
                                   "。");
    }
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    if (x.GetNumDims() != 1 || weight.GetNumDims() != 1) {
        throw std::runtime_error(
            "Ops::RmsNorm 要求 x 和 weight 均为一维张量：预期维数: 1，x 维数: " + std::to_string(x.GetNumDims()) +
            "，x shape: " + ShapeToString(x) + "，weight 维数: " + std::to_string(weight.GetNumDims()) +
            "，weight shape: " + ShapeToString(weight) + "。");
    }

    size_t dim = x.GetShape()[0];

    if (weight.GetShape()[0] != dim) {
        throw std::runtime_error(
            "Ops::RmsNorm 的输入与权重长度不匹配：要求 x.shape[0] == weight.shape[0]，x.shape[0]: " +
            std::to_string(dim) + "，weight.shape[0]: " + std::to_string(weight.GetShape()[0]) +
            "，x shape: " + ShapeToString(x) + "，weight shape: " + ShapeToString(weight) + "。");
    }

    const float* x_data = x.Data();
    const float* weight_data = weight.Data();
    float tmp = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        tmp += x_data[i] * x_data[i];
    }

    float scale = 1.0f / std::sqrt(tmp / static_cast<float>(dim) + eps);

    Tensor y({dim}, 0.0f);
    float* y_data = y.Data();
    for (size_t i = 0; i < dim; i++) {
        y_data[i] = x_data[i] * scale * weight_data[i];
    }
    return y;
}

Tensor Softmax(const Tensor& x) {
    if (x.GetNumDims() != 1) {
        throw std::runtime_error("Ops::Softmax 要求输入为一维张量：预期维数: 1，实际维数: " +
                                 std::to_string(x.GetNumDims()) + "，x shape: " + ShapeToString(x) + "。");
    }
    if (x.GetDataSize() == 0) {
        throw std::runtime_error("Ops::Softmax 要求输入非空：预期元素数: > 0，实际元素数: " +
                                 std::to_string(x.GetDataSize()) + "，x shape: " + ShapeToString(x) + "。");
    }

    size_t n = x.GetShape()[0];
    const float* x_data = x.Data();
    float max = x_data[0];

    for (size_t i = 0; i < n; ++i) {
        max = std::max(x_data[i], max);
    }

    float sum = 0.0f;

    Tensor y({n}, 0.0f);
    float* y_data = y.Data();

    for (size_t i = 0; i < n; ++i) {
        y_data[i] = std::exp(x_data[i] - max);
        sum += y_data[i];
    }

    for (size_t i = 0; i < n; ++i) {
        y_data[i] /= sum;
    }

    return y;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
    if (gate.GetShape() != up.GetShape()) {
        throw std::runtime_error("Ops::SwiGlu 要求 gate 和 up 形状相同：gate shape: " + ShapeToString(gate) +
                                 "，up shape: " + ShapeToString(up) + "。");
    }
    Tensor y(gate.GetShape(), 0.0f);
    const float* gate_data = gate.Data();
    const float* up_data = up.Data();
    float* y_data = y.Data();
    for (size_t i = 0; i < gate.GetDataSize(); i++) {
        float val = gate_data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        y_data[i] = sigmoid * val * up_data[i];
    }
    return y;
}

void NormalRope(Tensor& x, size_t pos, float theta) {
    size_t heads = x.GetShape()[0];
    size_t head_dim = x.GetShape()[1];
    float* x_data = x.Data();
    for (size_t i = 0; i < heads; ++i) {
        float* head_data = x_data + i * head_dim;
        for (size_t j = 0; j < head_dim; j += 2) {
            float freq = std::pow(theta, -static_cast<float>(j) / static_cast<float>(head_dim));

            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);

            float x0 = head_data[j];
            float x1 = head_data[j + 1];

            head_data[j] = x0 * cos_val - x1 * sin_val;
            head_data[j + 1] = x1 * cos_val + x0 * sin_val;
        }
    }
}

void NormalNeoX(Tensor& x, size_t pos, float theta) {
    size_t heads = x.GetShape()[0];
    size_t head_dim = x.GetShape()[1];
    size_t half_dim = head_dim / 2;
    float* x_data = x.Data();
    for (size_t i = 0; i < heads; ++i) {
        float* head_data = x_data + i * head_dim;
        for (size_t j = 0; j < half_dim; ++j) {
            float freq = std::pow(theta, -2 * static_cast<float>(j) / static_cast<float>(head_dim));

            float cos_val = std::cos(pos * freq);
            float sin_val = std::sin(pos * freq);

            float x0 = head_data[j];
            float x1 = head_data[j + half_dim];

            head_data[j] = x0 * cos_val - x1 * sin_val;
            head_data[j + half_dim] = x1 * cos_val + x0 * sin_val;
        }
    }
}

void Rope(Tensor& q, Tensor& k, size_t pos, float theta, RopeType rope_type) {
    if (q.GetNumDims() != 2 || k.GetNumDims() != 2) {
        throw std::runtime_error(
            "Ops::Rope 要求 q 和 k 均为二维张量：预期维数: 2，q 维数: " + std::to_string(q.GetNumDims()) +
            "，q shape: " + ShapeToString(q) + "，k 维数: " + std::to_string(k.GetNumDims()) +
            "，k shape: " + ShapeToString(k) + "。");
    }


    if (pos == std::numeric_limits<size_t>::max()) {
        throw std::out_of_range("Ops::Rope 要求 pos 非负：pos: " + std::to_string(pos) +
                                "，q shape: " + ShapeToString(q) + "，k shape: " + ShapeToString(k) + "。");
    }

    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("Ops::Rope 要求 theta 为有限正数：theta: " + std::to_string(theta) +
                                 "，q shape: " + ShapeToString(q) + "，k shape: " + ShapeToString(k) + "。");
    }

    size_t q_heads = q.GetShape()[0];
    size_t q_head_dim = q.GetShape()[1];

    size_t k_heads = k.GetShape()[0];
    size_t k_head_dim = k.GetShape()[1];

    if (q_head_dim != k_head_dim) {
        throw std::runtime_error("Ops::Rope 要求 q 和 k 的 head_dim 相同：q.shape[1]: " + std::to_string(q_head_dim) +
                                 "，k.shape[1]: " + std::to_string(k_head_dim) + "，q shape: " + ShapeToString(q) +
                                 "，k shape: " + ShapeToString(k) + "。");
    }

    if (q_head_dim == 0 || q_head_dim % 2 != 0) {
        throw std::runtime_error("Ops::Rope 要求 head_dim 为正偶数：head_dim: " + std::to_string(q_head_dim) +
                                 "，q shape: " + ShapeToString(q) + "，k shape: " + ShapeToString(k) + "。");
    }

    switch (rope_type) {
        case RopeType::Normal:
            NormalRope(q, pos, theta);
            NormalRope(k, pos, theta);
            break;
        case RopeType::NeoX:
            NormalNeoX(q, pos, theta);
            NormalNeoX(k, pos, theta);
            break;
        default:
            throw std::runtime_error(
                "Ops::Rope 收到未知的 rope_type：rope_type: " + std::to_string(static_cast<int>(rope_type)) +
                "，q shape: " + ShapeToString(q) + "，k shape: " + ShapeToString(k) + "。");
    }
}
