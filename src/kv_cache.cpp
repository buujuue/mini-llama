#include "kv_cache.h"
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include "tensor.h"

static std::string ShapeToString(const Tensor& tensor) {
    const auto& shape = tensor.GetShape();
    std::string result = "[";
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (axis > 0) {
            result += ", ";
        }
        result += std::to_string(shape[axis]);
    }
    return result + "]";
}

KvCache::KvCache(size_t layers_num, size_t seq_len_max, size_t kv_heads_num, size_t head_dim) {
    keys_ = Tensor({layers_num, seq_len_max, kv_heads_num, head_dim}, 0.0f);
    valves_ = Tensor({layers_num, seq_len_max, kv_heads_num, head_dim}, 0.0f);
    is_start_ = true;
}

void KvCache::Write(size_t layer, size_t pos, Tensor& k, Tensor& v){
    if (!is_start_) {
        throw std::runtime_error(
            "KvCache::Write 无法写入未初始化的缓存：is_start_: false"
            "，layer: " + std::to_string(layer)
            + "，pos: " + std::to_string(pos)
            + "，keys_ shape: " + ShapeToString(keys_)
            + "，valves_ shape: " + ShapeToString(valves_) + "。");
    }
    const auto& shape = keys_.GetShape();
    const auto& k_shape = k.GetShape();
    const auto& v_shape = v.GetShape();
    if (k.GetNumDims() != 2 || v.GetNumDims() != 2 || k_shape != v_shape){
        throw std::out_of_range(
            "KvCache::Write 要求 k 和 v 均为形状相同的二维张量 [kv_heads_num, head_dim]："
            "预期维数: 2，k 维数: " + std::to_string(k.GetNumDims())
            + "，k shape: " + ShapeToString(k)
            + "，v 维数: " + std::to_string(v.GetNumDims())
            + "，v shape: " + ShapeToString(v) + "。");
    }

    size_t kv_heads_num = k_shape[0];
    size_t head_dim = k_shape[1];

    if (layer < 0 || layer >= shape[0]) {
        throw std::out_of_range(
            "KvCache::Write 层索引越界：layer: " + std::to_string(layer)
            + "，有效范围为 [0, " + std::to_string(shape[0])
            + ")，keys_ shape: " + ShapeToString(keys_) + "。");
    }
    if (pos < 0 || pos >= shape[1]){
        throw std::out_of_range(
            "KvCache::Write 位置索引越界：pos: " + std::to_string(pos)
            + "，有效范围为 [0, " + std::to_string(shape[1])
            + ")，keys_ shape: " + ShapeToString(keys_) + "。");
    }  
    if (kv_heads_num != shape[2] || head_dim != shape[3]){
        throw std::out_of_range(
            "KvCache::Write 的输入形状与缓存不匹配：预期 [kv_heads_num, head_dim]: ["
            + std::to_string(shape[2]) + ", " + std::to_string(shape[3])
            + "]，实际 kv_heads_num: " + std::to_string(kv_heads_num)
            + "，实际 head_dim: " + std::to_string(head_dim)
            + "，k shape: " + ShapeToString(k) + "，v shape: " + ShapeToString(v)
            + "，keys_ shape: " + ShapeToString(keys_) + "。");
    }

    size_t layer_stride = shape[1] * shape[2] * shape[3];
    size_t pos_stride = shape[2] * shape[3];
    size_t head_stride = shape[3];

    size_t base = layer * layer_stride + pos * pos_stride;

    for (size_t h = 0; h < kv_heads_num; ++h) {
        size_t offset = h * head_stride;

        std::memcpy(&keys_.Data()[base + offset], &k.Data()[offset], head_dim * sizeof(float));
        std::memcpy(&valves_.Data()[base + offset], &v.Data()[offset], head_dim * sizeof(float));
    }
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
    if (!is_start_) {
        throw std::runtime_error(
            "KvCache::KeyPtr 无法访问未初始化的缓存：is_start_: false。");
    }
    const auto& shape = keys_.GetShape();
    if (layer < 0 || static_cast<size_t>(layer) >= shape[0]
        || pos < 0 || static_cast<size_t>(pos) >= shape[1]
        || kv_head < 0 || static_cast<size_t>(kv_head) >= shape[2]) {
        throw std::out_of_range(
            "KvCache::KeyPtr 缓存索引越界：layer: " + std::to_string(layer)
            + "，pos: " + std::to_string(pos)
            + "，kv_head: " + std::to_string(kv_head)
            + "，keys_ shape: " + ShapeToString(keys_) + "。");
    }
    const size_t layer_stride = shape[1] * shape[2] * shape[3];
    const size_t pos_stride = shape[2] * shape[3];
    const size_t head_stride = shape[3];
    return keys_.Data() + static_cast<size_t>(layer) * layer_stride
        + static_cast<size_t>(pos) * pos_stride + static_cast<size_t>(kv_head) * head_stride;
}

const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
    if (!is_start_) {
        throw std::runtime_error(
            "KvCache::ValuePtr 无法访问未初始化的缓存：is_start_: false。");
    }
    const auto& shape = valves_.GetShape();
    if (layer < 0 || static_cast<size_t>(layer) >= shape[0]
        || pos < 0 || static_cast<size_t>(pos) >= shape[1]
        || kv_head < 0 || static_cast<size_t>(kv_head) >= shape[2]) {
        throw std::out_of_range(
            "KvCache::ValuePtr 缓存索引越界：layer: " + std::to_string(layer)
            + "，pos: " + std::to_string(pos)
            + "，kv_head: " + std::to_string(kv_head)
            + "，valves_ shape: " + ShapeToString(valves_) + "。");
    }
    const size_t layer_stride = shape[1] * shape[2] * shape[3];
    const size_t pos_stride = shape[2] * shape[3];
    const size_t head_stride = shape[3];
    return valves_.Data() + static_cast<size_t>(layer) * layer_stride
        + static_cast<size_t>(pos) * pos_stride + static_cast<size_t>(kv_head) * head_stride;
}
