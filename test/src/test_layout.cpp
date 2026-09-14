#include "doctest.h"
#include "Layout.hpp"

namespace {

const std::vector<std::size_t> shape234 = {2, 3, 4};
const std::vector<std::size_t> stride234 = {12, 4, 1};
const std::vector<std::size_t> shape23 = {2, 3};
const std::vector<std::size_t> stride23 = {3, 1};
const std::vector<std::size_t> stride13 = {1, 3};
const std::vector<std::size_t> shape11 = {1, 1};

}

TEST_CASE("Layout Size") {
    CHECK(strata::Layout({}).size() == 1);
    CHECK(strata::Layout({3}).size() == 3);
    CHECK(strata::Layout({2, 3}).size() == 6);
    CHECK(strata::Layout({2, 3, 4}).size() == 24);
}

TEST_CASE("Layout Shape And Stride Accessors") {
    strata::Layout l(shape234);

    REQUIRE(l.ndim() == 3);
    REQUIRE(l.shape() == shape234);
    REQUIRE(l.stride() == stride234);

    CHECK(l.shape(0) == 2);
    CHECK(l.shape(1) == 3);
    CHECK(l.shape(2) == 4);
    CHECK(l.stride(0) == 12);
    CHECK(l.stride(1) == 4);
    CHECK(l.stride(2) == 1);
}

TEST_CASE("Layout Row-Major Offset") {
    strata::Layout l({3, 2});

    for (std::size_t row = 0; row < 3; row++)
        for (std::size_t col = 0; col < 2; col++)
            REQUIRE(l.offset(row, col) == row * 2 + col);
}

TEST_CASE("Layout Custom Stride Offset") {
    // Transposed 3x2 shape: element (row, col) lives at row * 1 + col * 3.
    strata::Layout l({3, 2}, stride13);

    REQUIRE(l.stride() == stride13);
    CHECK(l.offset(0, 1) == 3);
    CHECK(l.offset(1, 0) == 1);
    CHECK(l.offset(2, 1) == 5);
}

TEST_CASE("Layout Offset Bounds Checks") {
    strata::Layout l({3, 2});

    SUBCASE("first axis out of range") {
        CHECK_THROWS_AS(l.offset(3, 0), std::out_of_range);
        CHECK_THROWS_AS(l.offset(4, 1), std::out_of_range);
    }

    SUBCASE("second axis out of range") {
        CHECK_THROWS_AS(l.offset(0, 2), std::out_of_range);
        CHECK_THROWS_AS(l.offset(2, 9), std::out_of_range);
    }

    SUBCASE("multiple axes out of range") {
        CHECK_THROWS_AS(l.offset(3, 2), std::out_of_range);
    }
}

TEST_CASE("Layout Offset Wrong Argument Count") {
    strata::Layout l({3, 2});

    CHECK_THROWS_AS(l.offset(1), std::invalid_argument);
    CHECK_THROWS_AS(l.offset(1, 0, 0), std::invalid_argument);
}

TEST_CASE("Layout MAX_NDIM Boundary") {
    const std::vector<std::size_t> ok(32, 1);
    const std::vector<std::size_t> over(33, 1);

    strata::Layout l(ok);
    REQUIRE(l.ndim() == 32);
    REQUIRE(l.size() == 1);

    CHECK_THROWS_AS(strata::Layout{over}, std::invalid_argument);

    // The shape+stride constructor applies the same limit.
    CHECK_THROWS_AS(strata::Layout(over, ok), std::invalid_argument);
}

TEST_CASE("Layout Shape And Stride Constructor Validation") {
    const std::vector<std::size_t> shortStride = {1};

    CHECK_THROWS_AS(strata::Layout(shape23, shortStride), std::invalid_argument);

    strata::Layout l(shape23, stride13);
    REQUIRE(l.shape() == shape23);
    REQUIRE(l.stride() == stride13);
}

TEST_CASE("Layout Contiguity") {
    CHECK(strata::Layout({}).isContiguous());
    CHECK(strata::Layout({3}).isContiguous());
    CHECK(strata::Layout({2, 3, 4}).isContiguous());

    CHECK_FALSE(strata::Layout({2, 3}, stride13).isContiguous());
    CHECK_FALSE(strata::Layout({3, 2}, stride13).isContiguous());
}

TEST_CASE("Layout Copy Semantics") {
    strata::Layout a(shape23);

    strata::Layout b(a);
    REQUIRE(b.shape() == shape23);
    REQUIRE(b.stride() == stride23);

    strata::Layout c(shape11);
    c = a;
    REQUIRE(c.shape() == shape23);
    REQUIRE(c.stride() == stride23);

    c = c;
    REQUIRE(c.shape() == shape23);
    REQUIRE(c.stride() == stride23);
}
