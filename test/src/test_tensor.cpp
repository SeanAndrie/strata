#include "doctest.h"

#include "Tensor.hpp"

TEST_CASE("Tensor default construction") {
    st::Tensor<int> t;
    REQUIRE(t.size() == 1);
}

TEST_CASE("Tensor shape construction") {
    st::Tensor<int> t({3, 2});
    REQUIRE(t.size() == 6);
}
