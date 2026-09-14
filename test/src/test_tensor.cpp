#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "Tensor.hpp"

TEST_CASE("Tensor Default Construction") {
    st::Tensor<int> t;
    REQUIRE(t.size() == 1);
    REQUIRE(t.ndim() == 0);
    REQUIRE(t.shape().empty());
}

TEST_CASE("Tensor Scalar (empty shape) Construction") {
    st::Tensor<int> t(std::vector<std::size_t>{});
    REQUIRE(t.size() == 1);
    REQUIRE(t.ndim() == 0);
    REQUIRE(t.shape().empty());
}

TEST_CASE("Tensor Shape Construction") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> t(shape);
    REQUIRE(t.size() == 6);
    REQUIRE(t.ndim() == 2);
    REQUIRE(t.shape() == shape);
}

TEST_CASE("Tensor MAX_NDIM Boundary") {
    const std::vector<std::size_t> ok(32, 1);
    const std::vector<std::size_t> over(33, 1);

    st::Tensor<int> t(ok);
    REQUIRE(t.ndim() == 32);
    REQUIRE(t.size() == 1);

    CHECK_THROWS_AS(st::Tensor<int>{over}, std::invalid_argument);
}

TEST_CASE("Tensor Fill Construction") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> t(shape, 7);
    REQUIRE(t.size() == 6);
    REQUIRE(t.shape() == shape);
    for (std::size_t row = 0; row < shape[0]; row++)
        for (std::size_t col = 0; col < shape[1]; col++)
            REQUIRE(t(row, col) == 7);
}

TEST_CASE("Tensor Copy Construction") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> a(shape);
    for (std::size_t i = 0; i < a.size(); i++)
        a(i / shape[1], i % shape[1]) = static_cast<int>(i);

    st::Tensor<int> b(a);
    REQUIRE(a == b);
    REQUIRE(b(1, 1) == 3);
    REQUIRE(b.shape() == shape);
    REQUIRE(b.size() == 6);
}

TEST_CASE("Tensor Copy Is Deep") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> a(shape, 1);
    st::Tensor<int> b(a);

    b(0, 0) = 99;
    CHECK(a(0, 0) == 1);

    a(1, 0) = -5;
    CHECK(b(1, 0) == 1);
}

TEST_CASE("Tensor Copy Assignment") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> a(shape);
    for (std::size_t i = 0; i < a.size(); i++)
        a(i / shape[1], i % shape[1]) = static_cast<int>(i);

    st::Tensor<int> b({2, 2});
    b = a;
    REQUIRE(b == a);
    REQUIRE(b.shape() == shape);

    b(0, 0) = 42;
    CHECK(a(0, 0) == 0);
    CHECK_FALSE(b == a);
}

TEST_CASE("Tensor Copy Assignment Self") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> a(shape, 5);
    st::Tensor<int> b(a);

    a = a;
    REQUIRE(a == b);
    REQUIRE(a.shape() == shape);
}

TEST_CASE("Tensor Move Constructor") {
    const std::vector<std::size_t> shape = {3};

    st::Tensor<int> a(shape, 1);
    int* og_ptr = &a(0);

    st::Tensor<int> b(std::move(a));
    CHECK(&b(0) == og_ptr);
    REQUIRE(b.size() == 3);
    REQUIRE(b.shape() == shape);

    // Moved-from tensor is reset to a default scalar state.
    CHECK(a.size() == 1);
    CHECK(a.ndim() == 0);
    CHECK(a.shape().empty());
}

TEST_CASE("Tensor Move Assignment") {
    const std::vector<std::size_t> shape = {3};

    st::Tensor<int> a(shape, 1);
    int* og_ptr = &a(0);

    st::Tensor<int> b({2, 2}, 0);
    REQUIRE(b.size() == 4);

    b = std::move(a);
    CHECK(&b(0) == og_ptr);
    REQUIRE(b.size() == 3);
    REQUIRE(b.shape() == shape);

    // Moved-from tensor is reset to a default scalar state.
    CHECK(a.size() == 1);
    CHECK(a.ndim() == 0);
    CHECK(a.shape().empty());
}

TEST_CASE("Tensor Equality") {
    const std::vector<std::size_t> shape = {2, 3};

    SUBCASE("equal tensors") {
        st::Tensor<int> a(shape, 1);
        st::Tensor<int> b(shape, 1);
        CHECK(a == b);
        CHECK(b == a);
    }

    SUBCASE("different shapes") {
        st::Tensor<int> a(shape, 1);
        st::Tensor<int> b({3, 2}, 1);
        CHECK_FALSE(a == b);
    }

    SUBCASE("same shape, different values") {
        st::Tensor<int> a(shape, 1);
        st::Tensor<int> b(shape, 1);
        // Difference at flat index 4 (>= ndim() == 2): catches a loop bound
        // bug that only compared the first ndim() elements.
        b(1, 1) = 2;
        CHECK_FALSE(a == b);
    }

    SUBCASE("same shape, different leading element") {
        st::Tensor<int> a(shape, 1);
        st::Tensor<int> b(shape, 1);
        b(0, 0) = 2;
        CHECK_FALSE(a == b);
    }

    SUBCASE("scalars") {
        st::Tensor<int> a(std::vector<std::size_t>{});
        st::Tensor<int> b(std::vector<std::size_t>{});
        CHECK(a == b);
        CHECK(a.size() == 1);
    }

    SUBCASE("const tensors") {
        const st::Tensor<int> a(shape, 3);
        st::Tensor<int> b(shape, 3);
        CHECK(a == b);
    }
}

TEST_CASE("Tensor Multi-Index Access (Data Mapping)") {
    const std::vector<std::size_t> shape = {3, 2};
    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];

    st::Tensor<int> t(shape, 0);
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            t(row, col) = static_cast<int>(row * cols + col);

    // Row-major: t(row, col) must alias flat index (row * cols + col).
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            REQUIRE(t(row, col) == static_cast<int>(row * cols + col));
}

TEST_CASE("Tensor Indexing Bounds Checks") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> t(shape, 0);

    SUBCASE("row out of range") {
        CHECK_THROWS_AS(t(3, 0), std::out_of_range);
        CHECK_THROWS_AS(t(4, 0), std::out_of_range);
    }

    SUBCASE("column out of range") {
        CHECK_THROWS_AS(t(0, 2), std::out_of_range);
        CHECK_THROWS_AS(t(0, 5), std::out_of_range);
    }

    SUBCASE("both out of range") {
        CHECK_THROWS_AS(t(3, 2), std::out_of_range);
    }

    SUBCASE("wrong number of indices") {
        CHECK_THROWS_AS(t(1), std::invalid_argument);
        CHECK_THROWS_AS(t(1, 0, 0), std::invalid_argument);
    }
}

TEST_CASE("Tensor Fill Method") {
    const std::vector<std::size_t> shape = {3, 3};

    st::Tensor<float> t(shape);
    t.fill(10.0f);

    for (std::size_t row = 0; row < shape[0]; row++)
        for (std::size_t col = 0; col < shape[1]; col++)
            REQUIRE(t(row, col) == 10.0f);

    t.fill(2.5f);
    for (std::size_t row = 0; row < shape[0]; row++)
        for (std::size_t col = 0; col < shape[1]; col++)
            REQUIRE(t(row, col) == 2.5f);
}

TEST_CASE("Tensor View Shares Storage") {
    const std::vector<std::size_t> shape = {3, 2};

    st::Tensor<int> t(shape, 1);
    auto v = t.view();

    REQUIRE(v.shape() == shape);
    REQUIRE(v.size() == 6);
    REQUIRE(v.ndim() == 2);

    CHECK(v(1, 1) == t(1, 1));

    v(0, 0) = 42;
    CHECK(t(0, 0) == 42);

    t(2, 1) = 7;
    CHECK(v(2, 1) == 7);
}

TEST_CASE("Tensor Const View Reads") {
    st::Tensor<int> t({3, 2}, 4);
    const auto cv = t.view();

    CHECK(cv(0, 0) == 4);
    CHECK(cv(2, 1) == 4);
}

TEST_CASE("Tensor Reshape Reinterprets Data") {
    const std::vector<std::size_t> shape = {2, 3};

    st::Tensor<int> t(shape);
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / shape[1], i % shape[1]) = static_cast<int>(i);

    const std::vector<std::size_t> newShape = {3, 2};
    auto v = t.reshape(newShape);

    REQUIRE(v.shape() == newShape);
    REQUIRE(v.size() == t.size());

    // Unchanged data reinterpreted under the new layout (row-major flat order).
    for (std::size_t row = 0; row < newShape[0]; row++)
        for (std::size_t col = 0; col < newShape[1]; col++)
            REQUIRE(v(row, col) == static_cast<int>(row * newShape[1] + col));

    // The view aliases the original storage.
    v(0, 1) = 77;
    CHECK(t(0, 1) == 77);
}

TEST_CASE("Tensor Reshape To Scalar") {
    st::Tensor<int> t({1}, 9);
    auto v = t.reshape({});

    REQUIRE(v.shape().empty());
    REQUIRE(v.size() == 1);
    REQUIRE(v.ndim() == 0);
}

TEST_CASE("Tensor Reshape Errors") {
    st::Tensor<int> t({2, 3});

    SUBCASE("element count mismatch") {
        const std::vector<std::size_t> wrong = {4, 2};
        CHECK_THROWS_AS(t.reshape(wrong), std::invalid_argument);
    }

    SUBCASE("non-scalar cannot become scalar") {
        CHECK_THROWS_AS(t.reshape({}), std::invalid_argument);
    }

    SUBCASE("shape product overflows size_t") {
        const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
        const std::vector<std::size_t> overflow = {huge, 3};
        CHECK_THROWS_AS(t.reshape(overflow), std::length_error);
    }
}

TEST_CASE("Tensor Transpose Reverses Axes (2-D Data)") {
    const std::vector<std::size_t> shape = {2, 3};

    st::Tensor<int> t(shape);
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / shape[1], i % shape[1]) = static_cast<int>(i);

    const std::vector<std::size_t> expected = {3, 2};
    auto v = t.transpose();

    REQUIRE(v.shape() == expected);
    for (std::size_t i = 0; i < expected[0]; i++)
        for (std::size_t j = 0; j < expected[1]; j++)
            REQUIRE(v(i, j) == t(j, i));

    // Transposed view aliases the original storage.
    v(2, 1) = 99;
    CHECK(t(1, 2) == 99);
}

TEST_CASE("Tensor Transpose Reverses Axes (3-D Data)") {
    const std::vector<std::size_t> shape = {2, 3, 4};

    st::Tensor<int> t(shape);
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / (shape[1] * shape[2]), (i / shape[2]) % shape[1], i % shape[2])
            = static_cast<int>(i);

    const std::vector<std::size_t> expected = {4, 3, 2};
    auto v = t.transpose();

    REQUIRE(v.shape() == expected);
    for (std::size_t i = 0; i < expected[0]; i++)
        for (std::size_t j = 0; j < expected[1]; j++)
            for (std::size_t k = 0; k < expected[2]; k++)
                REQUIRE(v(i, j, k) == t(k, j, i));
}

TEST_CASE("Tensor Transpose 1-D Passes Through") {
    st::Tensor<int> t({5});
    for (std::size_t i = 0; i < 5; i++)
        t(i) = static_cast<int>(i);

    auto v = t.transpose();
    REQUIRE(v.shape() == t.shape());

    v(0) = 42;
    CHECK(t(0) == 42);
}

TEST_CASE("Tensor Transpose With Axes (Data)") {
    const std::vector<std::size_t> shape = {2, 3, 4};

    st::Tensor<int> t(shape);
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / (shape[1] * shape[2]), (i / shape[2]) % shape[1], i % shape[2])
            = static_cast<int>(i);

    const std::vector<std::size_t> axes = {1, 2, 0};
    const std::vector<std::size_t> expected = {3, 4, 2};
    auto v = t.transpose(axes);

    REQUIRE(v.shape() == expected);

    // out[i][j][k] == in[k][i][j] for axes (1, 2, 0).
    for (std::size_t i = 0; i < expected[0]; i++)
        for (std::size_t j = 0; j < expected[1]; j++)
            for (std::size_t k = 0; k < expected[2]; k++)
                REQUIRE(v(i, j, k) == t(k, i, j));
}

TEST_CASE("Tensor Transpose With Axes Errors") {
    st::Tensor<int> t({2, 3});

    SUBCASE("wrong number of axes") {
        const std::vector<std::size_t> axes = {1, 0, 0};
        CHECK_THROWS_AS(t.transpose(axes), std::invalid_argument);
    }

    SUBCASE("axis index out of range") {
        const std::vector<std::size_t> axes = {2, 0};
        CHECK_THROWS_AS(t.transpose(axes), std::invalid_argument);
    }
}

TEST_CASE("Tensor Accessors") {
    const std::vector<std::size_t> shape = {2, 3, 4};

    st::Tensor<int> t(shape);
    REQUIRE(t.size() == 24);
    REQUIRE(t.ndim() == 3);
    REQUIRE(t.shape() == shape);
    const std::vector<std::size_t> stride234 = {12, 4, 1};
    REQUIRE(t.stride() == stride234);
}

TEST_CASE("Random Test") {
    st::Tensor<int> t({3, 2, 3}, 1);
    try {
        auto v = t.slice(1);
        v.info();
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
    }
}
