#pragma once

#include <string>
#include "model.h"


std::string MapTensorName(const std::string& gguf_name);

// 从 GGUF 元数据构建模型配置；缺少必需维度或配置无效时抛出异常。
ModelConfig BuildConfig(const std::string& gguf_path);

// 加载 GGUF v3 的 F32/F16/Q8_0/Q4_0/Q4_1 权重，统一解码为 F32 Tensor。
// 解析/配置/权重加载失败时通过 load_error 返回原因，全部权重形状校验通过后设置 loaded。
Model LoadGgufModel(const std::string& gguf_path);
