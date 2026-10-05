#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ops.h"

namespace {

// 只通过公开接口测试算子；预期数值使用手算或已知数学结果的常量。
// 与课程参考实现一致：非法形状/空输入抛 runtime_error，RoPE 负位置抛 out_of_range。
// 仅测试 src/ops.cpp 已实现的 Matmul、Linear、RmsNorm、Softmax、SwiGlu 和 Rope。

Tensor TensorWithValues(const std::vector<size_t>& shape,
                        std::initializer_list<float> values) {
    Tensor tensor(shape);
    if (tensor.GetDataSize() != values.size()) {
        throw std::invalid_argument("测试数据的元素数与形状不一致");
    }
    std::copy(values.begin(), values.end(), tensor.Data());
    return tensor;
}

void ExpectTensorNear(const Tensor& actual, const std::vector<size_t>& shape,
                      std::initializer_list<float> expected,
                      float tolerance = 1e-5f) {
    ASSERT_EQ(actual.GetShape(), shape);
    ASSERT_EQ(actual.GetDataSize(), expected.size());

    size_t index = 0;
    for (const float value : expected) {
        SCOPED_TRACE(index);
        EXPECT_TRUE(std::isfinite(actual.Data()[index]));
        EXPECT_NEAR(actual.Data()[index], value, tolerance);
        ++index;
    }
}

void ExpectMatmulError(const Tensor& a, const Tensor& b,
                       std::initializer_list<const char*> expected_context) {
    try {
        (void)Matmul(a, b);
    } catch (const std::runtime_error& exception) {
        const std::string message = exception.what();
        SCOPED_TRACE(message);
        EXPECT_EQ(message.find("Ops::Matmul"), 0u);
        for (const char* context : expected_context) {
            EXPECT_NE(message.find(context), std::string::npos)
                << "缺少报错上下文: " << context;
        }
        return;
    }
    FAIL() << "Matmul 应拒绝非法输入并抛出 std::runtime_error";
}

// Matmul：M、K、N 各不相同，预期值独立手算，并检查错误中的输入上下文。
TEST(OpsMatmulTest, MultipliesRectangularMatrices) {
    const Tensor a = TensorWithValues({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = TensorWithValues({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    ExpectTensorNear(Matmul(a, b), {2, 4}, {38, 44, 50, 56, 83, 98, 113, 128});
}

TEST(OpsMatmulTest, IdentityPreservesInputValuesAndShape) {
    const Tensor a = TensorWithValues({2, 3}, {-1, 2, 3, 4, -5, 6});
    const Tensor identity = TensorWithValues({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});

    ExpectTensorNear(Matmul(a, identity), {2, 3}, {-1, 2, 3, 4, -5, 6});
}

TEST(OpsMatmulTest, MultipliesColumnVectorWithSignedValues) {
    const Tensor a = TensorWithValues({2, 3}, {1, -2, 3, -4, 5, -6});
    const Tensor b = TensorWithValues({3, 1}, {2, -1, 0.5f});

    ExpectTensorNear(Matmul(a, b), {2, 1}, {5.5f, -16});
}

TEST(OpsMatmulTest, RowTimesColumnProducesOneByOneMatrix) {
    const Tensor a = TensorWithValues({1, 3}, {1, -2, 0.5f});
    const Tensor b = TensorWithValues({3, 1}, {2, 3, -4});

    ExpectTensorNear(Matmul(a, b), {1, 1}, {-6});
}

TEST(OpsMatmulTest, SingleInnerDimensionProducesOuterProduct) {
    const Tensor a = TensorWithValues({2, 1}, {-2, 3});
    const Tensor b = TensorWithValues({1, 3}, {4, -0.5f, 2});

    ExpectTensorNear(Matmul(a, b), {2, 3}, {-8, 1, -4, 12, -1.5f, 6});
}

TEST(OpsMatmulTest, SingleElementResultKeepsMatrixRank) {
    const Tensor a = TensorWithValues({1, 1}, {-2});
    const Tensor b = TensorWithValues({1, 1}, {3});

    ExpectTensorNear(Matmul(a, b), {1, 1}, {-6});
}

TEST(OpsMatmulTest, EitherZeroOperandProducesZeroOutput) {
    const Tensor a = TensorWithValues({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = TensorWithValues({3, 2}, {1, -2, 3, 4, -5, 6});

    ExpectTensorNear(Matmul(Tensor({2, 3}, 0.0f), b), {2, 2}, {0, 0, 0, 0});
    ExpectTensorNear(Matmul(a, Tensor({3, 2}, 0.0f)), {2, 2}, {0, 0, 0, 0});
}

TEST(OpsMatmulTest, LeavesInputsUnchangedAndReturnsIndependentStorage) {
    const Tensor a = TensorWithValues({2, 2}, {1, 2, 3, 4});
    const Tensor b = TensorWithValues({2, 2}, {5, 6, 7, 8});

    Tensor result = Matmul(a, b);
    ExpectTensorNear(result, {2, 2}, {19, 22, 43, 50});
    EXPECT_NE(result.Data(), a.Data());
    EXPECT_NE(result.Data(), b.Data());
    result.At2D(0, 0) = -100.0f;

    ExpectTensorNear(a, {2, 2}, {1, 2, 3, 4});
    ExpectTensorNear(b, {2, 2}, {5, 6, 7, 8});
}

TEST(OpsMatmulTest, RejectsNonMatrixLeftOperand) {
    const Tensor a({3});
    const Tensor b({3, 2});

    ExpectMatmulError(a, b, {"预期维数: 2", "a 维数: 1", "a shape: [3]",
                             "b 维数: 2", "b shape: [3, 2]"});
}

TEST(OpsMatmulTest, RejectsNonMatrixRightOperand) {
    const Tensor a({2, 3});
    const Tensor b({3});

    ExpectMatmulError(a, b, {"预期维数: 2", "a 维数: 2", "a shape: [2, 3]",
                             "b 维数: 1", "b shape: [3]"});
}

TEST(OpsMatmulTest, RejectsBothNonMatrixOperands) {
    const Tensor a({3});
    const Tensor b({1, 3, 1});

    ExpectMatmulError(a, b, {"预期维数: 2", "a 维数: 1", "a shape: [3]",
                             "b 维数: 3", "b shape: [1, 3, 1]"});
}

TEST(OpsMatmulTest, RejectsEmptyOperandsBeforeReadingDimensions) {
    const Tensor empty;

    ExpectMatmulError(empty, Tensor({3, 2}),
                      {"预期维数: 2", "a 维数: 0", "a shape: []",
                       "b 维数: 2", "b shape: [3, 2]"});
    ExpectMatmulError(Tensor({2, 3}), empty,
                      {"预期维数: 2", "a 维数: 2", "a shape: [2, 3]",
                       "b 维数: 0", "b shape: []"});
}

TEST(OpsMatmulTest, RejectsScalarOperands) {
    const Tensor scalar = TensorWithValues({}, {1});

    ExpectMatmulError(scalar, Tensor({1, 2}),
                      {"预期维数: 2", "a 维数: 0", "a shape: []",
                       "b 维数: 2", "b shape: [1, 2]"});
    ExpectMatmulError(Tensor({2, 1}), scalar,
                      {"预期维数: 2", "a 维数: 2", "a shape: [2, 1]",
                       "b 维数: 0", "b shape: []"});
}

TEST(OpsMatmulTest, RejectsHigherRankOperands) {
    const Tensor matrix({3, 3});
    for (const auto& shape : {std::vector<size_t>{1, 3, 3},
                              std::vector<size_t>{1, 1, 3, 3}}) {
        SCOPED_TRACE(shape.size());
        const Tensor higher_rank(shape);
        EXPECT_THROW(Matmul(higher_rank, matrix), std::runtime_error);
        EXPECT_THROW(Matmul(matrix, higher_rank), std::runtime_error);
    }
}

TEST(OpsMatmulTest, RejectsMismatchedInnerDimension) {
    const Tensor a({2, 3});
    const Tensor b({4, 2});

    ExpectMatmulError(a, b, {"内部维度不匹配", "a.shape[1] == b.shape[0]",
                             "a.shape[1]: 3", "b.shape[0]: 4",
                             "a shape: [2, 3]", "b shape: [4, 2]"});
}

TEST(OpsMatmulTest, RejectsMismatchedInnerDimensionDespiteEqualElementCounts) {
    const Tensor a({2, 3});
    const Tensor b({2, 3});
    ASSERT_EQ(a.GetDataSize(), b.GetDataSize());

    ExpectMatmulError(a, b, {"内部维度不匹配", "a.shape[1]: 3", "b.shape[0]: 2",
                             "a shape: [2, 3]", "b shape: [2, 3]"});
}

// Linear：权重按 [out_features, in_features] 存放，不要求方阵。
TEST(OpsLinearTest, VectorInputProducesVectorProjection) {
    const Tensor x = TensorWithValues({3}, {1, 2, 3});
    const Tensor weight = TensorWithValues({2, 3}, {4, 5, 6, 7, 8, 9});

    ExpectTensorNear(Linear(x, weight), {2}, {32, 50});
}

TEST(OpsLinearTest, RowInputKeepsTwoDimensionalOutput) {
    const Tensor x = TensorWithValues({1, 3}, {1, 2, 3});
    const Tensor weight = TensorWithValues({2, 3}, {4, 5, 6, 7, 8, 9});

    ExpectTensorNear(Linear(x, weight), {1, 2}, {32, 50});
}

TEST(OpsLinearTest, SupportsSingleInputFeature) {
    const Tensor x = TensorWithValues({1}, {-2});
    const Tensor weight = TensorWithValues({2, 1}, {3, -4});

    ExpectTensorNear(Linear(x, weight), {2}, {-6, 8});
}

TEST(OpsLinearTest, SupportsSignedValuesAndIndependentStorageForBothInputRanks) {
    const Tensor weight = TensorWithValues({2, 3}, {2, 3, -4, -1, 2, 6});
    for (const auto& shape : {std::vector<size_t>{3}, std::vector<size_t>{1, 3}}) {
        SCOPED_TRACE(shape.size());
        const Tensor x = TensorWithValues(shape, {1, -2, 0.5f});
        const std::vector<size_t> output_shape =
            shape.size() == 1 ? std::vector<size_t>{2} : std::vector<size_t>{1, 2};

        Tensor result = Linear(x, weight);
        ExpectTensorNear(result, output_shape, {-6, -2});
        EXPECT_NE(result.Data(), x.Data());
        EXPECT_NE(result.Data(), weight.Data());
        result.Data()[0] = 100.0f;

        ExpectTensorNear(x, shape, {1, -2, 0.5f});
        ExpectTensorNear(weight, {2, 3}, {2, 3, -4, -1, 2, 6});
    }
}

TEST(OpsLinearTest, RejectsEmptyAndScalarOperandsBeforeReadingShapes) {
    const Tensor empty;
    const Tensor scalar = TensorWithValues({}, {1});
    const Tensor x({3});
    const Tensor weight({2, 3});

    for (const Tensor* invalid : {&empty, &scalar}) {
        EXPECT_THROW(Linear(*invalid, weight), std::runtime_error);
        EXPECT_THROW(Linear(x, *invalid), std::runtime_error);
    }
}

TEST(OpsLinearTest, RejectsNonMatrixWeight) {
    const Tensor x({3});
    const Tensor weight({3});

    EXPECT_THROW(Linear(x, weight), std::runtime_error);
}

TEST(OpsLinearTest, RejectsUnsupportedInputRank) {
    const Tensor x({1, 1, 3});
    const Tensor weight({2, 3});

    EXPECT_THROW(Linear(x, weight), std::runtime_error);
}

TEST(OpsLinearTest, RejectsMoreThanOneInputRow) {
    const Tensor x({2, 3});
    const Tensor weight({2, 3});

    EXPECT_THROW(Linear(x, weight), std::runtime_error);
}

TEST(OpsLinearTest, RejectsMismatchedInputFeatures) {
    const Tensor x({3});
    const Tensor row({1, 3});
    const Tensor weight({2, 4});

    EXPECT_THROW(Linear(x, weight), std::runtime_error);
    EXPECT_THROW(Linear(row, weight), std::runtime_error);
}

// RMSNorm：用能精确手算的 eps 区分 sqrt(mean(x^2) + eps) 与其他写法。
TEST(OpsRmsNormTest, AppliesWeightAndEpsilonInsideSquareRoot) {
    const Tensor x = TensorWithValues({2}, {-3, 4});
    const Tensor weight = TensorWithValues({2}, {2, 0.5f});

    // mean(x^2) = 12.5；加 eps=3.5 后平方根为 4；输出为 [-1.5, 0.5]。
    ExpectTensorNear(RmsNorm(x, weight, 3.5f), {2}, {-1.5f, 0.5f});
}

TEST(OpsRmsNormTest, DoesNotSubtractMeanFromInput) {
    const Tensor x({2}, 2.0f);
    const Tensor weight({2}, 1.0f);

    ExpectTensorNear(RmsNorm(x, weight, 1e-5f), {2},
                     {0.99999875f, 0.99999875f}, 1e-6f);
}

TEST(OpsRmsNormTest, ZeroInputProducesFiniteZeros) {
    const Tensor x({3}, 0.0f);
    const Tensor weight = TensorWithValues({3}, {1, 2, -3});

    ExpectTensorNear(RmsNorm(x, weight, 1e-5f), {3}, {0, 0, 0});
}

TEST(OpsRmsNormTest, RejectsNonVectorInput) {
    const Tensor x({1, 3});
    const Tensor weight({3});

    EXPECT_THROW(RmsNorm(x, weight, 1e-5f), std::runtime_error);
}

TEST(OpsRmsNormTest, RejectsNonVectorWeight) {
    const Tensor x({3});
    const Tensor weight({1, 3});

    EXPECT_THROW(RmsNorm(x, weight, 1e-5f), std::runtime_error);
}

TEST(OpsRmsNormTest, RejectsMismatchedWeightLength) {
    const Tensor x({3});
    const Tensor weight({2});

    EXPECT_THROW(RmsNorm(x, weight, 1e-5f), std::runtime_error);
}

// Softmax：小数据概率常量与大幅平移后的同一分布。
TEST(OpsSoftmaxTest, ProducesKnownProbabilityDistribution) {
    const Tensor x = TensorWithValues({3}, {1, 2, 3});
    const Tensor result = Softmax(x);

    ExpectTensorNear(result, {3}, {0.09003057f, 0.24472847f, 0.66524096f}, 1e-6f);
    float sum = 0.0f;
    for (size_t i = 0; i < result.GetDataSize(); ++i) {
        EXPECT_GE(result.Data()[i], 0.0f);
        EXPECT_LE(result.Data()[i], 1.0f);
        sum += result.Data()[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(OpsSoftmaxTest, LargePositiveValuesStayFinite) {
    const Tensor x = TensorWithValues({3}, {1000, 1001, 1002});

    ExpectTensorNear(Softmax(x), {3}, {0.09003057f, 0.24472847f, 0.66524096f}, 1e-6f);
}

TEST(OpsSoftmaxTest, LargeNegativeValuesStayFinite) {
    const Tensor x = TensorWithValues({3}, {-1000, -999, -998});

    ExpectTensorNear(Softmax(x), {3}, {0.09003057f, 0.24472847f, 0.66524096f}, 1e-6f);
}

TEST(OpsSoftmaxTest, EqualScoresProduceUniformProbabilities) {
    const Tensor x({3}, 7.0f);

    ExpectTensorNear(Softmax(x), {3}, {0.33333333f, 0.33333333f, 0.33333333f}, 1e-6f);
}

TEST(OpsSoftmaxTest, SingleScoreHasProbabilityOne) {
    const Tensor x = TensorWithValues({1}, {-1000});

    ExpectTensorNear(Softmax(x), {1}, {1});
}

TEST(OpsSoftmaxTest, RejectsNonVectorInput) {
    const Tensor x({2, 3});

    EXPECT_THROW(Softmax(x), std::runtime_error);
}

TEST(OpsSoftmaxTest, RejectsEmptyInput) {
    const Tensor x;

    EXPECT_THROW(Softmax(x), std::runtime_error);
}

TEST(OpsSwiGluTest, ActivatesGateBeforeMultiplyingUpBranch) {
    const Tensor gate = TensorWithValues({2, 2}, {-1, 0, 1, 2});
    const Tensor up = TensorWithValues({2, 2}, {2, 3, -2, 0.5f});

    ExpectTensorNear(SwiGlu(gate, up), {2, 2},
                     {-0.53788284f, 0, -1.46211716f, 0.88079708f});
}

TEST(OpsSwiGluTest, RejectsMismatchedBranchLengths) {
    const Tensor gate({3});
    const Tensor up({4});

    EXPECT_THROW(SwiGlu(gate, up), std::runtime_error);
}

TEST(OpsSwiGluTest, RejectsDifferentShapesWithSameElementCount) {
    const Tensor gate({2, 2});
    const Tensor up({4});

    EXPECT_THROW(SwiGlu(gate, up), std::runtime_error);
}

// RoPE 的 head_dim=4、pos=1、theta=10000 时，两对旋转角分别为 1 和 0.01。
// 使用固定数值检查结对、方向和频率，并允许 Q 与 K 的 head 数不同。
TEST(OpsRopeTest, DefaultNormalModeRotatesAdjacentPairsInPlace) {
    Tensor q = TensorWithValues({1, 4}, {1, 0, 0, 1});
    Tensor k = TensorWithValues({2, 4}, {0, 1, 1, 0, 1, 1, -1, -1});

    Rope(q, k, 1, 10000.0f);

    ExpectTensorNear(q, {1, 4}, {0.54030231f, 0.84147098f, -0.00999983f, 0.99995000f});
    ExpectTensorNear(k, {2, 4},
                     {-0.84147098f, 0.54030231f, 0.99995000f, 0.00999983f,
                      -0.30116868f, 1.38177329f, -0.98995017f, -1.00994983f});
}

TEST(OpsRopeTest, NeoXModeRotatesSplitHalfPairsInPlace) {
    Tensor q = TensorWithValues({1, 4}, {1, 0, 0, 1});
    Tensor k = TensorWithValues({2, 4}, {0, 1, 1, 0, 1, 1, -1, -1});

    Rope(q, k, 1, 10000.0f, RopeType::NeoX);

    ExpectTensorNear(q, {1, 4}, {0.54030231f, -0.00999983f, 0.84147098f, 0.99995000f});
    ExpectTensorNear(k, {2, 4},
                     {-0.84147098f, 0.99995000f, 0.54030231f, 0.00999983f,
                      1.38177329f, 1.00994983f, 0.30116868f, -0.98995017f});
}

TEST(OpsRopeTest, ZeroPositionPreservesAllHeadValues) {
    for (const RopeType type : {RopeType::Normal, RopeType::NeoX}) {
        SCOPED_TRACE(static_cast<int>(type));
        Tensor q = TensorWithValues({1, 4}, {1, 2, 3, 4});
        Tensor k = TensorWithValues({2, 4}, {-1, -2, -3, -4, 5, 6, 7, 8});

        Rope(q, k, 0, 10000.0f, type);

        ExpectTensorNear(q, {1, 4}, {1, 2, 3, 4});
        ExpectTensorNear(k, {2, 4}, {-1, -2, -3, -4, 5, 6, 7, 8});
    }
}

TEST(OpsRopeTest, RejectsNonMatrixQ) {
    Tensor q({4});
    Tensor k({2, 4});

    EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(OpsRopeTest, RejectsNonMatrixK) {
    Tensor q({2, 4});
    Tensor k({4});

    EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(OpsRopeTest, RejectsMismatchedHeadDimension) {
    Tensor q({2, 4});
    Tensor k({1, 6});

    EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(OpsRopeTest, RejectsOddHeadDimension) {
    Tensor q({2, 3});
    Tensor k({1, 3});

    EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(OpsRopeTest, RejectsNegativePosition) {
    Tensor q({2, 4});
    Tensor k({1, 4});

    EXPECT_THROW(Rope(q, k, -1, 10000.0f), std::out_of_range);
}

TEST(OpsRopeTest, RejectsNonpositiveOrNonfiniteTheta) {
    for (const float theta : {0.0f, -1.0f,
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
        SCOPED_TRACE(theta);
        Tensor q({2, 4});
        Tensor k({1, 4});

        EXPECT_THROW(Rope(q, k, 0, theta), std::runtime_error);
    }
}

} // namespace
