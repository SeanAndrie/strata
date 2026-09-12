#include "doctest.h"
#include "Tensor.hpp"

#include <iostream>

TEST_CASE("Tensor Default Construction") {
    st::Tensor<int> t;
    REQUIRE(t.size() == 1);
}

TEST_CASE("Tensor Copy Construction") {
    st::Tensor<int> a({5}, 1);
    st::Tensor<int> b(a);

    REQUIRE(a == b);
}

TEST_CASE("Tensor Shape Construction") {
    st::Tensor<int> t1({3, 2});
    REQUIRE(t1.size() == 6);

    std::vector<std::size_t> v;
    for (std::size_t i = 0; i < 34; i++)
        v.push_back(1);

    CHECK_THROWS_AS(st::Tensor<int> t2(v), std::invalid_argument);
}

TEST_CASE("Tensor Move Constructor") {
    st::Tensor<float> a({3}, 1.0f);
    float *og_ptr = &a(0);

    st::Tensor<float> b(std::move(a));
    CHECK(&b(0) == og_ptr);
}

TEST_CASE("Tensor Fill Construction") {
    st::Tensor<int> t({3}, 1);
    for (std::size_t i = 0; i < t.size(); i++)
        REQUIRE(t(i) == 1);
}

TEST_CASE("Tensor Indexing") {
    st::Tensor<int> t({3, 2}, 0);
    auto shape = t.shape();

    for (std::size_t i = 0; i < t.size(); i++) {
        std::size_t row = i / shape[0];
        std::size_t col = i % shape[1];
        REQUIRE(t(row, col) == 0);
    }

    CHECK_THROWS_AS(t(0, 3), std::out_of_range);
}

TEST_CASE("Tensor Fill Method") {
    st::Tensor<float> t({3, 3});
    auto shape = t.shape();

    t.fill(10.0f);

    for (std::size_t i = 0; i < t.size(); i++) {
        std::size_t row = i / shape[0];
        std::size_t col = i % shape[0];
        REQUIRE(t(row, col) == 10.0f);
    }
}

TEST_CASE("Tensor View") {
    st::Tensor<float> t({3,2}, 1.0f);
    auto v = t.view();
    v(0, 0) = 42.0f;
    CHECK(t(0, 0) == 42.0f);
    CHECK(v.shape() == t.shape());
}
