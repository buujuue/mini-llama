#pragma once
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "batch.h"
#include "context.h"
#include "model.h"
#include "ops.h"
#include "tensor.h"
#include "tool.h"

Tensor ForwardBatch(Context& ctx, const Model model, const Batch& batch);
