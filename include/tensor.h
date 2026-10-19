#pragma once

#include <cstddef>
#include <string>
#include <vector>

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape, float fill = 0.0f);

    const std::vector<size_t>& GetShape() const noexcept { return shape_; }
    size_t GetNumDims() const noexcept { return shape_.size(); }
    size_t GetDataSize() const noexcept { return data_.size(); }

    float* Data() noexcept { return data_.data(); }
    const float* Data() const noexcept { return data_.data(); }

    float& At(const std::vector<size_t>& indices);
    float At(const std::vector<size_t>& indices) const;
    float& At1D(size_t i);
    float At1D(size_t i) const;
    float& At2D(size_t i, size_t j);
    float At2D(size_t i, size_t j) const;
    float* RowPtr(int row);
    const float* RowPtr(int row) const;
    float& At3D(size_t i, size_t j, size_t k);
    float At3D(size_t i, size_t j, size_t k) const;
    float& At4D(size_t i, size_t j, size_t k, size_t l);
    float At4D(size_t i, size_t j, size_t k, size_t l) const;

private:
    void CheckRunk(size_t expected, const std::string& caller) const;
    size_t CheckNumel();
    std::string ShapeToString() const;
    void CheckAxisIndex(size_t axis, size_t index, const std::string& caller) const;
    size_t FlatIndex(const std::vector<size_t>& indices) const;
    std::vector<float> data_;
    std::vector<size_t> shape_;
};
