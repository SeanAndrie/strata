#include "doctest.h"
#include "Tensor.hpp"

#include <iterator>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace {

st::Tensor<int> make23() {
    st::Tensor<int> t({2, 3});
    for (std::size_t i = 0; i < t.size(); i++)
        t(i / 3, i % 3) = static_cast<int>(i);
    return t;
}

}

TEST_CASE("Tensor Iteration Order") {
    auto t = make23();

    std::vector<int> got;
    for (int x : t)
        got.push_back(x);

    CHECK(got == std::vector<int>{0, 1, 2, 3, 4, 5});
}

TEST_CASE("Tensor Iteration Distance") {
    auto t = make23();

    CHECK(std::distance(t.begin(), t.end()) == 6);
    CHECK(t.begin() == t.begin());
    CHECK(t.begin() != t.end());
}

TEST_CASE("Tensor Iteration Sum") {
    auto t = make23();

    int sum = 0;
    for (int x : t)
        sum += x;

    CHECK(sum == 15);
}

TEST_CASE("Tensor Iteration Scalar") {
    st::Tensor<int> s(std::vector<std::size_t>{}, 9);

    std::vector<int> got;
    for (int x : s)
        got.push_back(x);

    CHECK(got == std::vector<int>{9});
}

TEST_CASE("Transposed View Iteration Order") {
    auto t = make23();
    auto v = t.view().transpose();

    std::vector<int> got;
    for (int x : v)
        got.push_back(x);

    CHECK(got == std::vector<int>{0, 3, 1, 4, 2, 5});
}

TEST_CASE("Sliced View Iteration Order") {
    auto t = make23();
    auto v = t.slice(st::all, st::Slice(1, 3));

    std::vector<int> got;
    for (int x : v)
        got.push_back(x);

    CHECK(got == std::vector<int>{1, 2, 4, 5});
}

TEST_CASE("Iterator Arithmetic") {
    st::Tensor<int> t({5});
    for (std::size_t i = 0; i < 5; i++)
        t(i) = static_cast<int>(i);

    auto it = t.begin();
    CHECK(*it == 0);
    CHECK(*(it + 2) == 2);

    auto shifted = t.begin();
    shifted += 3;
    CHECK(*shifted == 3);

    auto old = it++;
    CHECK(*old == 0);
    CHECK(*it == 1);

    auto at3 = t.begin() + 3;
    auto back = at3 - 2;
    CHECK(*back == 1);
    CHECK(*at3 == 3);

    auto at3b = t.begin() + 3;
    --at3b;
    CHECK(*at3b == 2);

    auto at2 = t.begin() + 2;
    auto old2 = at2--;
    CHECK(*old2 == 2);
    CHECK(*at2 == 1);
}

TEST_CASE("Mutable Iterator Via Range-For") {
    auto t = make23();

    for (int& x : t)
        x += 100;

    std::vector<int> got;
    for (int x : t)
        got.push_back(x);

    CHECK(got == std::vector<int>{100, 101, 102, 103, 104, 105});
}

TEST_CASE("Mutable Iterator Dereference And Fill") {
    auto t = make23();

    *t.begin() = 42;
    CHECK(t(0, 0) == 42);

    std::fill(t.begin(), t.end(), 7);
    for (std::size_t i = 0; i < t.size(); i++)
        CHECK(t(i / 3, i % 3) == 7);
}

TEST_CASE("Iterator Reference Types") {
    auto t = make23();
    static_assert(std::is_same_v<decltype(*t.begin()), int&>);

    const auto& c = t;
    static_assert(std::is_same_v<decltype(*c.begin()), const int&>);
    static_assert(std::is_same_v<decltype(*c.cbegin()), const int&>);
}

TEST_CASE("Const Tensor Iteration") {
    const auto t = make23();

    std::vector<int> got;
    for (const int& x : t)
        got.push_back(x);

    CHECK(got == std::vector<int>{0, 1, 2, 3, 4, 5});

    std::vector<int> cgot;
    for (auto it = t.cbegin(); it != t.cend(); ++it)
        cgot.push_back(*it);

    CHECK(cgot == got);
}

TEST_CASE("Mutable View Iteration Aliases Owner") {
    auto t = make23();

    for (int& x : t.view().transpose())
        x = 0;

    for (std::size_t i = 0; i < t.size(); i++)
        CHECK(t(i / 3, i % 3) == 0);
}
