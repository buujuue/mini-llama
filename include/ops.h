#pragma once

#include <cstddef>
#include "tensor.h"
#include "tool.h"  


Tensor Matmul(const Tensor& a, const Tensor& b);

Tensor Linear(const Tensor& x, const Tensor& weight);


Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

Tensor Softmax(const Tensor& x);

Tensor SwiGlu(const Tensor& gate, const Tensor& up);

void Rope(Tensor& q, Tensor& k, size_t pos, float theta, RopeType rope_type);
