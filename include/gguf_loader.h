#pragma once

#include <string>
#include "model.h"


std::string MapTensorName(const std::string& gguf_name);

// 从 GGUF 元数据构建模型配置；读取依赖未实现时抛出“未完成”异常。
ModelConfig BuildConfig(const std::string& gguf_path);

// 配置/权重加载失败时通过 load_error 返回原因；全部权重形状校验通过后设置 loaded。
Model LoadGgufModel(const std::string& gguf_path);
