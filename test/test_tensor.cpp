#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"

namespace {

// 默认构造的张量没有数据，所有元素访问都应报错。
TEST(TensorTest, DefaultConstructionRejectsElementAccess) {
    Tensor tensor;
    const Tensor& view = tensor;

    EXPECT_THROW(tensor.At({}), std::runtime_error);
    EXPECT_THROW(view.At({}), std::runtime_error);
    EXPECT_THROW(tensor.At1D(0), std::runtime_error);
    EXPECT_THROW(view.At1D(0), std::runtime_error);
    EXPECT_THROW(tensor.At2D(0, 0), std::runtime_error);
    EXPECT_THROW(view.At2D(0, 0), std::runtime_error);
    EXPECT_THROW(tensor.At3D(0, 0, 0), std::runtime_error);
    EXPECT_THROW(view.At3D(0, 0, 0), std::runtime_error);
    EXPECT_THROW(tensor.At4D(0, 0, 0, 0), std::runtime_error);
    EXPECT_THROW(view.At4D(0, 0, 0, 0), std::runtime_error);
    EXPECT_THROW(tensor.RowPtr(0), std::runtime_error);
    EXPECT_THROW(view.RowPtr(0), std::runtime_error);
}

// 与参考实现一致：显式空 shape 表示包含一个元素的零维标量。
TEST(TensorTest, EmptyShapeRepresentsScalar) {
    Tensor tensor(std::vector<size_t>{}, 2.5f);
    const Tensor& view = tensor;

    EXPECT_FLOAT_EQ(view.At({}), 2.5f);
    tensor.At({}) = -4.0f;
    EXPECT_FLOAT_EQ(view.At({}), -4.0f);
    EXPECT_THROW(tensor.At({0}), std::runtime_error);
    EXPECT_THROW(view.At({0}), std::runtime_error);
}

// 未指定初始值时，所有元素默认为 0。
TEST(TensorTest, DefaultFillInitializesAllElements) {
    Tensor tensor({2, 3});
    const Tensor& view = tensor;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(tensor.At2D(i, j), 0.0f);
            EXPECT_FLOAT_EQ(view.At({i, j}), 0.0f);
        }
    }
}

// 指定初始值时，每个元素都应填入该值。
TEST(TensorTest, CustomFillInitializesAllElements) {
    Tensor tensor({2, 3, 4, 5}, -1.25f);
    const Tensor& view = tensor;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                for (size_t l = 0; l < 5; ++l) {
                    EXPECT_FLOAT_EQ(tensor.At4D(i, j, k, l), -1.25f);
                    EXPECT_FLOAT_EQ(view.At({i, j, k, l}), -1.25f);
                }
            }
        }
    }
}

// 一维专用接口与通用索引应访问同一元素，const 接口也应读到相同值。
TEST(TensorTest, OneDimensionalAccessMatchesGenericIndexing) {
    Tensor tensor({5});
    const Tensor& view = tensor;

    for (size_t i = 0; i < 5; ++i) {
        const float value = static_cast<float>(i + 1);
        tensor.At({i}) = value;
        EXPECT_FLOAT_EQ(tensor.At1D(i), value);
        EXPECT_FLOAT_EQ(view.At1D(i), value);
        EXPECT_FLOAT_EQ(view.At({i}), value);
        EXPECT_EQ(&tensor.At1D(i), &tensor.At({i}));
    }
}

// 二维专用接口与通用索引应访问同一元素，const 接口也应读到相同值。
TEST(TensorTest, TwoDimensionalAccessMatchesGenericIndexing) {
    Tensor tensor({2, 3});
    const Tensor& view = tensor;
    float value = 1.0f;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tensor.At2D(i, j) = value;
            EXPECT_FLOAT_EQ(tensor.At({i, j}), value);
            EXPECT_FLOAT_EQ(view.At({i, j}), value);
            EXPECT_FLOAT_EQ(view.At2D(i, j), value);
            EXPECT_EQ(&tensor.At({i, j}), &tensor.At2D(i, j));
            value += 1.0f;
        }
    }
}

// 三维专用接口与通用索引应访问同一元素，const 接口也应读到相同值。
TEST(TensorTest, ThreeDimensionalAccessMatchesGenericIndexing) {
    Tensor tensor({2, 3, 4});
    const Tensor& view = tensor;
    float value = 1.0f;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                tensor.At3D(i, j, k) = value;
                EXPECT_FLOAT_EQ(tensor.At({i, j, k}), value);
                EXPECT_FLOAT_EQ(view.At({i, j, k}), value);
                EXPECT_FLOAT_EQ(view.At3D(i, j, k), value);
                EXPECT_EQ(&tensor.At({i, j, k}), &tensor.At3D(i, j, k));
                value += 1.0f;
            }
        }
    }
}

// 四维专用接口与通用索引应访问同一元素，const 接口也应读到相同值。
TEST(TensorTest, FourDimensionalAccessMatchesGenericIndexing) {
    Tensor tensor({2, 3, 4, 5});
    const Tensor& view = tensor;
    float value = 1.0f;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                for (size_t l = 0; l < 5; ++l) {
                    tensor.At4D(i, j, k, l) = value;
                    EXPECT_FLOAT_EQ(tensor.At({i, j, k, l}), value);
                    EXPECT_FLOAT_EQ(view.At({i, j, k, l}), value);
                    EXPECT_FLOAT_EQ(view.At4D(i, j, k, l), value);
                    EXPECT_EQ(&tensor.At({i, j, k, l}), &tensor.At4D(i, j, k, l));
                    value += 1.0f;
                }
            }
        }
    }
}

// 高维张量按行优先存储，最后一维变化最快。
TEST(TensorTest, GenericIndexingSupportsHigherRanksInRowMajorOrder) {
    Tensor tensor({2, 3, 2, 3, 2});
    const Tensor& view = tensor;
    float* first = &tensor.At({0, 0, 0, 0, 0});

    EXPECT_EQ(&tensor.At({0, 1, 1, 1, 0}), first + 20);
    EXPECT_EQ(&tensor.At({1, 0, 0, 0, 1}), first + 37);
    EXPECT_EQ(&tensor.At({1, 2, 1, 2, 1}), first + 71);
    tensor.At({1, 2, 1, 2, 1}) = 7.5f;
    EXPECT_FLOAT_EQ(view.At({1, 2, 1, 2, 1}), 7.5f);
    EXPECT_FLOAT_EQ(view.At({0, 0, 0, 0, 0}), 0.0f);
}

// 行指针可直接读写元素，各行在内存中连续排列。
TEST(TensorTest, RowPointersExposeContiguousRows) {
    Tensor tensor({3, 4});
    const Tensor& view = tensor;

    for (int row = 0; row < 3; ++row) {
        float* data = tensor.RowPtr(row);
        EXPECT_EQ(view.RowPtr(row), data);
        for (size_t column = 0; column < 4; ++column) {
            const float value = static_cast<float>(row * 10) + static_cast<float>(column);
            data[column] = value;
            EXPECT_FLOAT_EQ(view.At2D(static_cast<size_t>(row), column), value);
            EXPECT_FLOAT_EQ(view.RowPtr(row)[column], value);
        }
    }

    EXPECT_EQ(tensor.RowPtr(1), tensor.RowPtr(0) + 4);
    EXPECT_EQ(tensor.RowPtr(2), tensor.RowPtr(0) + 8);
    EXPECT_EQ(&tensor.At2D(2, 3), tensor.RowPtr(0) + 11);
}

// 行指针只适用于二维张量，其他维度应报错。
TEST(TensorTest, RowPointerRejectsInvalidRank) {
    Tensor vector({5});
    Tensor volume({2, 3, 4});
    const Tensor& vector_view = vector;
    const Tensor& volume_view = volume;

    EXPECT_THROW(vector.RowPtr(0), std::runtime_error);
    EXPECT_THROW(vector_view.RowPtr(0), std::runtime_error);
    EXPECT_THROW(volume.RowPtr(0), std::runtime_error);
    EXPECT_THROW(volume_view.RowPtr(0), std::runtime_error);
}

// 负数行号和超出行数的行号都应报错。
TEST(TensorTest, RowPointerRejectsOutOfRangeRows) {
    Tensor tensor({3, 4});
    const Tensor& view = tensor;

    for (const int row : {-1, 3, std::numeric_limits<int>::max()}) {
        SCOPED_TRACE(row);
        EXPECT_THROW(tensor.RowPtr(row), std::runtime_error);
        EXPECT_THROW(view.RowPtr(row), std::runtime_error);
    }
}

// 专用访问接口要求张量维数匹配，否则应报错。
TEST(TensorTest, SpecializedAccessorsRejectIncorrectRanks) {
    Tensor vector({5});
    Tensor matrix({2, 3});
    const Tensor& vector_view = vector;
    const Tensor& matrix_view = matrix;

    EXPECT_THROW(matrix.At1D(0), std::runtime_error);
    EXPECT_THROW(matrix_view.At1D(0), std::runtime_error);
    EXPECT_THROW(vector.At2D(0, 0), std::runtime_error);
    EXPECT_THROW(vector_view.At2D(0, 0), std::runtime_error);
    EXPECT_THROW(matrix.At3D(0, 0, 0), std::runtime_error);
    EXPECT_THROW(matrix_view.At3D(0, 0, 0), std::runtime_error);
    EXPECT_THROW(matrix.At4D(0, 0, 0, 0), std::runtime_error);
    EXPECT_THROW(matrix_view.At4D(0, 0, 0, 0), std::runtime_error);
}

// 逐个检查专用接口的每个轴，索引达到该轴长度时应报错。
TEST(TensorTest, SpecializedAccessorsRejectOutOfRangeIndices) {
    Tensor vector({5});
    Tensor matrix({2, 3});
    Tensor volume({2, 3, 4});
    Tensor batch({2, 3, 4, 5});
    const Tensor& vector_view = vector;
    const Tensor& matrix_view = matrix;
    const Tensor& volume_view = volume;
    const Tensor& batch_view = batch;

    EXPECT_THROW(vector.At1D(5), std::runtime_error);
    EXPECT_THROW(vector_view.At1D(5), std::runtime_error);
    EXPECT_THROW(matrix.At2D(2, 0), std::runtime_error);
    EXPECT_THROW(matrix_view.At2D(2, 0), std::runtime_error);
    EXPECT_THROW(matrix.At2D(0, 3), std::runtime_error);
    EXPECT_THROW(matrix_view.At2D(0, 3), std::runtime_error);
    EXPECT_THROW(volume.At3D(2, 0, 0), std::runtime_error);
    EXPECT_THROW(volume_view.At3D(2, 0, 0), std::runtime_error);
    EXPECT_THROW(volume.At3D(0, 3, 0), std::runtime_error);
    EXPECT_THROW(volume_view.At3D(0, 3, 0), std::runtime_error);
    EXPECT_THROW(volume.At3D(0, 0, 4), std::runtime_error);
    EXPECT_THROW(volume_view.At3D(0, 0, 4), std::runtime_error);
    EXPECT_THROW(batch.At4D(2, 0, 0, 0), std::runtime_error);
    EXPECT_THROW(batch_view.At4D(2, 0, 0, 0), std::runtime_error);
    EXPECT_THROW(batch.At4D(0, 3, 0, 0), std::runtime_error);
    EXPECT_THROW(batch_view.At4D(0, 3, 0, 0), std::runtime_error);
    EXPECT_THROW(batch.At4D(0, 0, 4, 0), std::runtime_error);
    EXPECT_THROW(batch_view.At4D(0, 0, 4, 0), std::runtime_error);
    EXPECT_THROW(batch.At4D(0, 0, 0, 5), std::runtime_error);
    EXPECT_THROW(batch_view.At4D(0, 0, 0, 5), std::runtime_error);
}

// 拷贝构造和拷贝赋值都应独立保存数据，修改副本不影响原张量。
TEST(TensorTest, CopiesOwnIndependentStorage) {
    Tensor original({2, 3}, 1.5f);
    Tensor copied = original;
    Tensor assigned;
    assigned = original;

    copied.At2D(0, 1) = 8.0f;
    assigned.At2D(1, 2) = -2.0f;
    EXPECT_FLOAT_EQ(original.At2D(0, 1), 1.5f);
    EXPECT_FLOAT_EQ(original.At2D(1, 2), 1.5f);
    EXPECT_FLOAT_EQ(copied.At2D(0, 1), 8.0f);
    EXPECT_FLOAT_EQ(assigned.At2D(1, 2), -2.0f);
}

// 报错信息应包含类名、接口名、越界轴、索引和形状。
TEST(TensorTest, ErrorsIncludeTensorAndShapeContext) {
    Tensor tensor({1, 2});

    try {
        tensor.At2D(1, 0);
        FAIL() << "Expected an out-of-range exception";
    } catch (const std::runtime_error& exception) {
        const std::string message = exception.what();
        EXPECT_EQ(message.find("[Tensor] "), 0u);
        EXPECT_NE(message.find("At2D"), std::string::npos);
        EXPECT_NE(message.find("axis: 0"), std::string::npos);
        EXPECT_NE(message.find("idx: 1"), std::string::npos);
        EXPECT_NE(message.find("shape: [1, 2]"), std::string::npos);
    }
}

// 参数化测试会对下方提供的每一种形状重复执行测试。
class InvalidTensorShapeTest : public testing::TestWithParam<std::vector<size_t>> {};

// 形状中任一维度长度为 0 时，构造应失败。
TEST_P(InvalidTensorShapeTest, RejectsZeroDimensions) {
    EXPECT_THROW((void)Tensor(GetParam()), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    ZeroOnEachAxis,
    InvalidTensorShapeTest,
    testing::Values(
        std::vector<size_t>{0},
        std::vector<size_t>{2, 0},
        std::vector<size_t>{2, 3, 0},
        std::vector<size_t>{2, 3, 4, 0}
    )
);

class TensorOverflowTest : public testing::TestWithParam<std::vector<size_t>> {};

// 各维度长度的乘积超出 size_t 范围时，构造应失败。
TEST_P(TensorOverflowTest, RejectsElementCountOverflow) {
    EXPECT_THROW((void)Tensor(GetParam()), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    OverflowOnDifferentAxes,
    TensorOverflowTest,
    testing::Values(
        std::vector<size_t>{std::numeric_limits<size_t>::max(), 2},
        std::vector<size_t>{2, std::numeric_limits<size_t>::max()},
        std::vector<size_t>{std::numeric_limits<size_t>::max() / 4 + 1, 2, 2}
    )
);

class TensorIndexTest : public testing::TestWithParam<std::vector<size_t>> {};

// 通用索引的数量必须等于张量维数，少一个或多一个都应报错。
TEST_P(TensorIndexTest, RejectsIncorrectIndexCount) {
    Tensor tensor(GetParam());
    const Tensor& view = tensor;
    const std::vector<size_t> fewer(GetParam().size() - 1, 0);
    const std::vector<size_t> more(GetParam().size() + 1, 0);

    EXPECT_THROW(tensor.At(fewer), std::runtime_error);
    EXPECT_THROW(view.At(fewer), std::runtime_error);
    EXPECT_THROW(tensor.At(more), std::runtime_error);
    EXPECT_THROW(view.At(more), std::runtime_error);
}

// 逐个检查每个轴，普通越界索引和 size_t 最大值都应报错。
TEST_P(TensorIndexTest, RejectsOutOfRangeIndicesOnEveryAxis) {
    Tensor tensor(GetParam());
    const Tensor& view = tensor;

    for (size_t axis = 0; axis < GetParam().size(); ++axis) {
        SCOPED_TRACE(axis);
        std::vector<size_t> indices(GetParam().size(), 0);
        indices[axis] = GetParam()[axis];
        EXPECT_THROW(tensor.At(indices), std::runtime_error);
        EXPECT_THROW(view.At(indices), std::runtime_error);
        indices[axis] = std::numeric_limits<size_t>::max();
        EXPECT_THROW(tensor.At(indices), std::runtime_error);
        EXPECT_THROW(view.At(indices), std::runtime_error);
    }
}

// 使用一维到五维形状，复用上述两项通用索引检查。
INSTANTIATE_TEST_SUITE_P(
    RanksOneThroughFive,
    TensorIndexTest,
    testing::Values(
        std::vector<size_t>{5},
        std::vector<size_t>{2, 3},
        std::vector<size_t>{2, 3, 4},
        std::vector<size_t>{2, 3, 4, 5},
        std::vector<size_t>{2, 3, 2, 3, 2}
    )
);

} // namespace
