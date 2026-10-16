#include "doctest.h"
#include "Tensor.hpp"

#include <vector>

TEST_CASE("Slice Defaults And all") {
    st::Slice s;
    CHECK(s.start == 0);
    CHECK(s.stop == st::Slice::npos);
    CHECK(s.step == 1);

    CHECK(st::all.start == 0);
    CHECK(st::all.stop == st::Slice::npos);
    CHECK(st::all.step == 1);

    st::Slice custom(1, 4, 2);
    CHECK(custom.start == 1);
    CHECK(custom.stop == 4);
    CHECK(custom.step == 2);
}

TEST_CASE("Layout Slice Full") {
    st::Layout base({2, 3});
    auto [layout, offset] = st::Layout::sliceLayout(base, {st::all, st::all});

    REQUIRE(layout.shape() == base.shape());
    REQUIRE(layout.stride() == base.stride());
    CHECK(offset == 0);
}

TEST_CASE("Layout Slice Partial Range") {
    st::Layout base({2, 3});
    auto [layout, offset] = st::Layout::sliceLayout(base, {st::all, st::Slice(1, 3)});

    REQUIRE(layout.shape() == std::vector<std::size_t>{2, 2});
    REQUIRE(layout.stride() == std::vector<std::size_t>{3, 1});
    CHECK(offset == 1);
}

TEST_CASE("Layout Slice Stepped") {
    st::Layout base({2, 4});
    auto [layout, offset] = st::Layout::sliceLayout(base, {st::all, st::Slice(0, 4, 2)});

    REQUIRE(layout.shape() == std::vector<std::size_t>{2, 2});
    REQUIRE(layout.stride() == std::vector<std::size_t>{4, 2});
    CHECK(offset == 0);
}

TEST_CASE("Layout Slice Open Start And Stop") {
    // Slice(1) means [1, ndim) with step 1.
    st::Layout base({5});
    auto [layout, offset] = st::Layout::sliceLayout(base, {st::Slice(1)});

    REQUIRE(layout.shape() == std::vector<std::size_t>{4});
    REQUIRE(layout.stride() == std::vector<std::size_t>{1});
    CHECK(offset == 1);
}

TEST_CASE("Layout Slice Errors") {
    st::Layout base({2, 3});

    SUBCASE("zero step") {
        CHECK_THROWS_AS(st::Layout::sliceLayout(base, {st::Slice(0, 2, 0), st::all}),
                        std::invalid_argument);
    }

    SUBCASE("stop exceeds axis size") {
        CHECK_THROWS_AS(st::Layout::sliceLayout(base, {st::Slice(0, 5), st::all}),
                        std::out_of_range);
    }

    SUBCASE("start beyond resolved stop") {
        CHECK_THROWS_AS(st::Layout::sliceLayout(base, {st::Slice(5), st::all}),
                        std::invalid_argument);
    }

    SUBCASE("start greater than explicit stop") {
        CHECK_THROWS_AS(st::Layout::sliceLayout(base, {st::Slice(2, 1), st::all}),
                        std::invalid_argument);
    }

    SUBCASE("slice count mismatch") {
        CHECK_THROWS_AS(st::Layout::sliceLayout(base, {st::all}), std::invalid_argument);
        CHECK_THROWS_AS(st::Layout::sliceLayout(base, {st::all, st::all, st::all}),
                        std::invalid_argument);
    }
}

TEST_CASE("Tensor Slice Shares Storage") {
    st::Tensor<int> t({2, 3});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 3, i % 3) = static_cast<int>(i);

    auto v = t.slice(st::all, st::Slice(1, 3));

    REQUIRE(v.shape() == std::vector<std::size_t>{2, 2});
    REQUIRE(v.size() == 4);
    CHECK(v(0, 0) == t(0, 1));
    CHECK(v(0, 1) == t(0, 2));
    CHECK(v(1, 0) == t(1, 1));
    CHECK(v(1, 1) == t(1, 2));

    v(0, 0) = 99;
    CHECK(t(0, 1) == 99);

    t(1, 1) = 42;
    CHECK(v(1, 0) == 42);
}

TEST_CASE("Tensor Slice 1D") {
    st::Tensor<int> t({5});
    for (std::size_t i = 0; i < 5; i++)
        t(i) = static_cast<int>(i);

    auto v = t.slice(st::Slice(1, 4));

    REQUIRE(v.shape() == std::vector<std::size_t>{3});
    CHECK(v(0) == 1);
    CHECK(v(1) == 2);
    CHECK(v(2) == 3);

    v(0) = 10;
    CHECK(t(1) == 10);
}

TEST_CASE("Tensor Slice 3D Offset") {
    const std::vector<std::size_t> shape = {2, 3, 4};
    st::Tensor<int> t(shape);
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 12, (i / 4) % 3, i % 4) = static_cast<int>(i);

    auto v = t.slice(st::Slice(1, 2), st::all, st::Slice(0, 4, 2));

    REQUIRE(v.shape() == std::vector<std::size_t>{1, 3, 2});
    for (std::size_t j = 0; j < 3; j++)
        for (std::size_t k = 0; k < 2; k++)
            CHECK(v(0, j, k) == t(1, j, k * 2));

    v(0, 1, 0) = 777;
    CHECK(t(1, 1, 0) == 777);
}

TEST_CASE("Tensor Slice Empty Range") {
    st::Tensor<int> t({2, 3}, 1);
    auto v = t.slice(st::all, st::Slice(1, 1));

    REQUIRE(v.shape() == std::vector<std::size_t>{2, 0});
    CHECK(v.size() == 0);
}

TEST_CASE("TensorView Slice Chains") {
    st::Tensor<int> t({2, 3});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 3, i % 3) = static_cast<int>(i);

    auto v = t.view().slice(st::all, st::Slice(1, 3)).transpose();

    REQUIRE(v.shape() == std::vector<std::size_t>{2, 2});
    CHECK(v(0, 0) == t(0, 1));
    CHECK(v(1, 0) == t(0, 2));
    CHECK(v(0, 1) == t(1, 1));
    CHECK(v(1, 1) == t(1, 2));
}

TEST_CASE("Tensor Slice Errors") {
    st::Tensor<int> t({2, 3});

    CHECK_THROWS_AS(t.slice(st::all), std::invalid_argument);
    CHECK_THROWS_AS(t.slice(st::all, st::Slice(0, 2, 0)), std::invalid_argument);
    CHECK_THROWS_AS(t.slice(st::Slice(0, 5), st::all), std::out_of_range);
    CHECK_THROWS_AS(t.slice(st::Slice(5), st::all), std::invalid_argument);
}
