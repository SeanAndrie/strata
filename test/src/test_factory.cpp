#include "doctest.h"
#include "Tensor.hpp"

#include <vector>

TEST_CASE("arange Single Argument") {
    auto t = st::Tensor<int>::arange(5);

    REQUIRE(t.shape() == std::vector<std::size_t>{5});
    REQUIRE(t.size() == 5);
    for (std::size_t i = 0; i < 5; i++)
        CHECK(t(i) == static_cast<int>(i));
}

TEST_CASE("arange Zero Stop") {
    auto t = st::Tensor<int>::arange(0);

    CHECK(t.ndim() == 1);
    CHECK(t.shape() == std::vector<std::size_t>{0});
    CHECK(t.size() == 0);
}

TEST_CASE("arange Start Stop Step") {
    SUBCASE("aligned") {
        auto t = st::Tensor<int>::arange(0, 10, 2);
        REQUIRE(t.size() == 5);
        for (std::size_t i = 0; i < 5; i++)
            CHECK(t(i) == static_cast<int>(2 * i));
    }

    SUBCASE("non-zero start") {
        auto t = st::Tensor<int>::arange(2, 10, 2);
        REQUIRE(t.size() == 4);
        CHECK(t(0) == 2);
        CHECK(t(1) == 4);
        CHECK(t(2) == 6);
        CHECK(t(3) == 8);
    }

    SUBCASE("non-divisible range") {
        auto t = st::Tensor<int>::arange(0, 10, 3);
        REQUIRE(t.size() == 4);
        CHECK(t(0) == 0);
        CHECK(t(1) == 3);
        CHECK(t(2) == 6);
        CHECK(t(3) == 9);
    }

    SUBCASE("default step of one") {
        auto t = st::Tensor<int>::arange(1, 5);
        REQUIRE(t.size() == 4);
        for (std::size_t i = 0; i < 4; i++)
            CHECK(t(i) == static_cast<int>(1 + i));
    }
}

TEST_CASE("arange Errors") {
    CHECK_THROWS_AS(st::Tensor<int>::arange(0, 10, 0), std::invalid_argument);
    CHECK_THROWS_AS(st::Tensor<int>::arange(5, 5), std::invalid_argument);
    CHECK_THROWS_AS(st::Tensor<int>::arange(6, 2), std::invalid_argument);
}

TEST_CASE("linspace Evenly Spaced") {
    auto t = st::Tensor<double>::linspace(0.0, 1.0, 5);

    REQUIRE(t.size() == 5);
    CHECK(t(0) == doctest::Approx(0.0));
    CHECK(t(1) == doctest::Approx(0.25));
    CHECK(t(2) == doctest::Approx(0.5));
    CHECK(t(3) == doctest::Approx(0.75));
    CHECK(t(4) == doctest::Approx(1.0));
}

TEST_CASE("linspace Two Points") {
    auto t = st::Tensor<double>::linspace(-1.0, 1.0, 2);

    REQUIRE(t.size() == 2);
    CHECK(t(0) == doctest::Approx(-1.0));
    CHECK(t(1) == doctest::Approx(1.0));
}

TEST_CASE("linspace Single Point") {
    auto t = st::Tensor<double>::linspace(3.0, 9.0, 1);

    REQUIRE(t.size() == 1);
    CHECK(t(0) == doctest::Approx(3.0));
}

TEST_CASE("linspace Default Count") {
    auto t = st::Tensor<double>::linspace(0.0, 1.0);

    CHECK(t.size() == 50);
    CHECK(t(0) == doctest::Approx(0.0));
    CHECK(t(49) == doctest::Approx(1.0));
}

TEST_CASE("linspace Zero Count Errors") {
    CHECK_THROWS_AS(st::Tensor<double>::linspace(0.0, 1.0, 0), std::invalid_argument);
}

TEST_CASE("ones zeros full") {
    SUBCASE("ones") {
        auto t = st::Tensor<int>::ones({2, 3});
        REQUIRE(t.shape() == std::vector<std::size_t>{2, 3});
        REQUIRE(t.size() == 6);
        for (std::size_t i = 0; i < t.size(); i++)
            CHECK(t(i / 3, i % 3) == 1);
    }

    SUBCASE("zeros") {
        auto t = st::Tensor<int>::zeros({3});
        REQUIRE(t.size() == 3);
        for (std::size_t i = 0; i < 3; i++)
            CHECK(t(i) == 0);
    }

    SUBCASE("full") {
        auto t = st::Tensor<double>::full({4}, 2.5);
        REQUIRE(t.size() == 4);
        for (std::size_t i = 0; i < 4; i++)
            CHECK(t(i) == doctest::Approx(2.5));
    }
}
