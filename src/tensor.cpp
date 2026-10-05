#include "tensor.h"

#include <limits>
#include <stdexcept>

Tensor::Tensor(const std::vector<size_t>& shape, float fill) : shape_(shape) {
    data_.resize(CheckNumel(), fill);
}

size_t Tensor::CheckNumel() {
    size_t count = 1;
    for (size_t axis = 0; axis < shape_.size(); ++axis){
        const size_t dim = shape_[axis];
        if (dim == 0) {
            throw std::runtime_error("[Tensor] shape 包含空维度：axis: " + std::to_string(axis)
                + "，dim: 0，shape: " + ShapeToString() + "。");
        }
        if (count > std::numeric_limits<size_t>::max() / dim){
            throw std::runtime_error("[Tensor] shape 元素总数溢出：axis: " + std::to_string(axis)
                + "，已累计元素数: " + std::to_string(count)
                + "，当前维度: " + std::to_string(dim)
                + "，shape: " + ShapeToString() + "。");
        }
        count *= dim;
    }
    return count;
}

std::string Tensor::ShapeToString() const {
    std::string s = "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += std::to_string(shape_[i]);
    }
    s += "]";
    return s;
}

void Tensor::CheckAxisIndex(size_t axis, size_t index, const std::string& caller) const {
    if (axis >= shape_.size()) {
        throw std::runtime_error("[Tensor] 调用 " + caller + " 时轴超出范围：axis: " + std::to_string(axis)
            + "，有效范围为 [0, " + std::to_string(shape_.size())
            + ")，shape: " + ShapeToString() + "。");
    }

    const size_t dim = shape_[axis];
    if (index >= dim){
        throw std::runtime_error("[Tensor] 调用 " + caller + " 时发生下标越界：axis: " + std::to_string(axis)
            + " idx: " + std::to_string(index)
            + "，有效范围为 [0, " + std::to_string(dim - 1)
            + "]，shape: " + ShapeToString() + "。");
    }
}

size_t Tensor::FlatIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()){
        throw std::runtime_error("[Tensor] 索引数量与张量维数不匹配 索引数量: " + std::to_string(indices.size())
            + "，shape: " + ShapeToString()
            + "，张量维数: " + std::to_string(shape_.size()) + "。");
    }

    if (data_.empty()) {
        throw std::runtime_error("[Tensor] 空张量不能访问元素：shape: " + ShapeToString() + "。");
    }

    for (size_t axis = 0; axis < indices.size(); ++axis) {
        CheckAxisIndex(axis, indices[axis], "FlatIndex");
    }
    size_t flat = 0;
    size_t stride = 1;
    for (size_t axis = shape_.size(); axis > 0; --axis){
        const size_t i = axis - 1;
        flat += indices[i] * stride;
        stride *= shape_[i];
    }
    return  flat;
}

void Tensor::CheckRunk(size_t expected, const std::string& caller) const {
    if (expected != shape_.size()) {
        throw std::runtime_error("[Tensor] 调用 " + caller + " 时张量维数不匹配：预期维数: " + std::to_string(expected)
            + "，实际维数: " + std::to_string(shape_.size())
            + "，shape: " + ShapeToString() + "。");
    }
}

float& Tensor::At(const std::vector<size_t>& indices){
    return data_[FlatIndex(indices)];
}

float Tensor::At(const std::vector<size_t>& indices) const {
    return data_[FlatIndex(indices)];
}
float& Tensor::At1D(size_t i){
    CheckRunk(1, "At1D");
    CheckAxisIndex(0, i, "At1D");
    return data_[i];
}

float Tensor::At1D(size_t i) const {
    CheckRunk(1, "At1D");
    CheckAxisIndex(0, i, "At1D");
    return data_[i];
}

float& Tensor::At2D(size_t i, size_t j) {
    CheckRunk(2, "At2D");
    CheckAxisIndex(0, i, "At2D");
    CheckAxisIndex(1, j, "At2D");
    return data_[i * shape_[1] + j];
}

float Tensor::At2D(size_t i, size_t j) const {
    CheckRunk(2, "At2D");
    CheckAxisIndex(0, i, "At2D");
    CheckAxisIndex(1, j, "At2D");
    return data_[i * shape_[1] + j];
}

float* Tensor::RowPtr(int row) {
    CheckRunk(2, "row_ptr");
    CheckAxisIndex(0, row, "row_ptr");
    return data_.data() + static_cast<size_t>(row) * shape_[1];
}

const float* Tensor::RowPtr(int row) const {
    CheckRunk(2, "row_ptr");
    CheckAxisIndex(0, row, "row_ptr");
    return data_.data() + static_cast<size_t>(row) * shape_[1];
}

float& Tensor::At3D(size_t i, size_t j, size_t k) {
    CheckRunk(3, "At3D");
    CheckAxisIndex(0, i, "At3D");
    CheckAxisIndex(1, j, "At3D");
    CheckAxisIndex(2, k, "At3D");
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

float Tensor::At3D(size_t i, size_t j, size_t k) const {
    CheckRunk(3, "At3D");
    CheckAxisIndex(0, i, "At3D");
    CheckAxisIndex(1, j, "At3D");
    CheckAxisIndex(2, k, "At3D");
    return data_[(i * shape_[1] + j) * shape_[2] + k];
}

float& Tensor::At4D(size_t i, size_t j, size_t k, size_t l) {
    CheckRunk(4, "At4D");
    CheckAxisIndex(0, i, "At4D");
    CheckAxisIndex(1, j, "At4D");
    CheckAxisIndex(2, k, "At4D");
    CheckAxisIndex(3, l, "At4D");
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

float Tensor::At4D(size_t i, size_t j, size_t k, size_t l) const {
    CheckRunk(4, "At4D");
    CheckAxisIndex(0, i, "At4D");
    CheckAxisIndex(1, j, "At4D");
    CheckAxisIndex(2, k, "At4D");
    CheckAxisIndex(3, l, "At4D");
    return data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}
