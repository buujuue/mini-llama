#pragma once

#include <cstddef>
#include "tensor.h"
#include "tool.h"  //RopeType

// a: [M, K]，b: [K, N]，返回 [M, N]，计算 a * b。
Tensor Matmul(const Tensor& a, const Tensor& b);
// x: [K] 或 [1, K]，weight: [N, K]，计算 x * weight^T。
// 返回形状保持 x 的维数：[N] 或 [1, N]。
Tensor Linear(const Tensor& x, const Tensor& weight);
// x、weight: [dim]，返回 [dim]。
// y = x / sqrt(mean(x^2) + eps) * weight。
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);
// 对非空一维张量计算 Softmax，返回形状与 x 相同。
Tensor Softmax(const Tensor& x);
// gate、up 形状相同，返回 Silu(gate) * up。
Tensor SwiGlu(const Tensor& gate, const Tensor& up);
// 对当前 token 的 q、k 原地施加旋转位置编码（RoPE)
void Rope(Tensor& q, Tensor& k, size_t pos, float theta, RopeType rope_type = RopeType::Normal);
