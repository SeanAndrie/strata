#include "doctest.h"
#include "Tensor.hpp"

#include <limits>
#include <vector>

namespace {

st::Tensor<int> make23() {
    st::Tensor<int> t({2, 3});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 3, i % 3) = static_cast<int>(i);
    return t;
}

}

TEST_CASE("View Transpose Reverses Axes") {
    auto t = make23();
    auto v = t.view().transpose();

    REQUIRE(v.shape() == std::vector<std::size_t>{3, 2});
    REQUIRE(v.size() == 6);
    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 2; j++)
            CHECK(v(i, j) == t(j, i));

    v(0, 1) = 99;
    CHECK(t(1, 0) == 99);
}

TEST_CASE("View Transpose 3D") {
    st::Tensor<int> t({2, 3, 4});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 12, (i / 4) % 3, i % 4) = static_cast<int>(i);

    auto v = t.view().transpose();

    REQUIRE(v.shape() == std::vector<std::size_t>{4, 3, 2});
    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 3; j++)
            for (std::size_t k = 0; k < 2; k++)
                CHECK(v(i, j, k) == t(k, j, i));
}

TEST_CASE("View Permute With Axes") {
    st::Tensor<int> t({2, 3, 4});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 12, (i / 4) % 3, i % 4) = static_cast<int>(i);

    auto v = t.view().permute({1, 2, 0});

    REQUIRE(v.shape() == std::vector<std::size_t>{3, 4, 2});
    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 4; j++)
            for (std::size_t k = 0; k < 2; k++)
                CHECK(v(i, j, k) == t(k, i, j));
}

TEST_CASE("View Transpose Pair") {
    st::Tensor<int> t({2, 3, 4});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 12, (i / 4) % 3, i % 4) = static_cast<int>(i);

    auto v = t.view().transpose(0, 2);

    REQUIRE(v.shape() == std::vector<std::size_t>{4, 3, 2});
    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 3; j++)
            for (std::size_t k = 0; k < 2; k++)
                CHECK(v(i, j, k) == t(k, j, i));
}

TEST_CASE("View Permute Axes Errors") {
    auto t = make23();
    auto v = t.view();

    CHECK_THROWS_AS(v.permute(std::vector<std::size_t>{0}), std::invalid_argument);
    CHECK_THROWS_AS(v.permute(std::vector<std::size_t>{0, 2}), std::invalid_argument);
    CHECK_THROWS_AS(v.permute(std::vector<std::size_t>{1, 1}), std::invalid_argument);
    CHECK_THROWS_AS(v.transpose(0, 2), std::invalid_argument);
}

TEST_CASE("View Reshape Reinterprets") {
    auto t = make23();
    auto v = t.view().reshape({3, 2});

    REQUIRE(v.shape() == std::vector<std::size_t>{3, 2});
    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 2; j++)
            CHECK(v(i, j) == static_cast<int>(i * 2 + j));

    v(0, 1) = 77;
    CHECK(t(0, 1) == 77);
}

TEST_CASE("View Reshape Scalar") {
    st::Tensor<int> t({1}, 9);
    auto v = t.view().reshape({});

    REQUIRE(v.shape().empty());
    REQUIRE(v.ndim() == 0);
    REQUIRE(v.size() == 1);
    CHECK(*v.begin() == 9);
}

TEST_CASE("View Reshape Errors") {
    auto t = make23();
    auto v = t.view();

    CHECK_THROWS_AS(v.reshape({4, 2}), std::invalid_argument);
    CHECK_THROWS_AS(v.reshape({}), std::invalid_argument);

    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2 + 1;
    CHECK_THROWS_AS(v.reshape({huge, 3}), std::length_error);
}

TEST_CASE("View Reshape Non-Contiguous Errors") {
    auto t = make23();

    CHECK_THROWS_AS(t.view().transpose().reshape({2, 3}), std::runtime_error);
    CHECK_THROWS_AS(t.view().permute({1, 0}).reshape({2, 3}), std::runtime_error);
    CHECK_THROWS_AS(t.slice(st::all, st::Slice(0, 3, 2)).reshape({4}), std::runtime_error);
}
