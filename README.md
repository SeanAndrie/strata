# strata

A tensor library written from scratch in C++17.

## Features

- **`Layout`** — shape/stride descriptor with bounds-checked multi-index
  offsetting, contiguity testing, and stride-preserving slicing.
- **`Tensor<T>`** — owning, heap-allocated (`std::unique_ptr<T[]>`) N-dimensional
  array with copy/move semantics, equality, and multi-index access.
- **`TensorView<T>`** — non-owning, stride-aware view supporting reshape,
  transpose, permute, and slicing without copying data.
- **`TensorIterator<T>`** — forward iterator (mutable and const) that traverses
  logical row-major order while honoring arbitrary strides.
- **Slicing** — `Slice(start, stop, step)` and `all`, with per-axis validation.
- **Shape manipulation** — `reshape`, `transpose` (full reverse and axis pair),
  and `permute` with duplicate-axis rejection.
- **Factories** — `arange`, `linspace`, `ones`, `zeros`, `full`.
- **Consistent errors** — `std::invalid_argument`, `std::out_of_range`,
  `std::length_error`, and `std::runtime_error` for the relevant failure modes.

## Requirements

- A C++17 compiler (`c++`, g++, or clang++)
- GNU Make
- doctest is vendored in `test/` — no external dependencies

## Repository layout

    .
    ├── includes/
    │   ├── Layout.hpp          # shape/stride descriptor, Slice, sliceLayout
    │   ├── Tensor.hpp          # owning tensor + factories
    │   ├── TensorView.hpp      # non-owning view
    │   └── TensorIterator.hpp  # forward iterator
    ├── test/
    │   ├── src/                # doctest suites
    │   ├── doctest.h
    │   └── Makefile
    ├── Makefile                # top-level: `make test`
    └── README.md

## Build & test

    make test

Compiles every suite under `test/src/` with
`-Wall -Wextra -Werror -std=c++17` and runs doctest.

    make clean      # remove the test binary
    make fclean     # alias for clean

Optional AddressSanitizer + UndefinedBehaviorSanitizer build:

    c++ -Wall -Wextra -Itest -Iincludes -std=c++17 -g \
        -fsanitize=address,undefined test/src/*.cpp -o /tmp/strata_tests \
        && /tmp/strata_tests

## Design overview

- **Ownership.** `Tensor` owns its buffer; `TensorView` borrows a raw pointer
  plus a `Layout`. Every view-returning `Tensor` method is lvalue-qualified
  (`&`) and has a deleted rvalue overload (`&& = delete`), so a view can never
  outlive the temporary it would point into.
- **Strides.** Row-major (C-order) strides are stored explicitly, so views may
  be non-contiguous (transposed, strided slices, etc.).
- **Fixed rank cap.** `MAX_NDIM = 32`; shape and stride are `std::array`-backed.
- **Scalars.** Zero-dimensional tensors/views represent scalars (size 1).
- **Contiguity.** `reshape` on a view requires a contiguous layout; otherwise it
  throws `std::runtime_error` rather than silently mis-mapping memory.
- **Iteration.** Iterators visit elements in logical row-major order and honor
  arbitrary strides; both mutable (`T&`) and const (`const T&`) forms are
  provided, with `begin/end` and `cbegin/cend` overloads.

## Roadmap

| Priority | Item | Notes |
|:--------:|------|-------|
| 0 | Broadcasting logic | prerequisite for shape-mismatched elementwise operations |
| 1 | Tensor operations | elementwise add/sub/mul/div, reductions (sum/mean/min/max), matrix multiplication |
| 2 | `fill(value)` method | today only the shape+value constructor |
| 3 | Random-access iterator | `operator<`, iterator subtraction, `operator[]`, `difference_type` arithmetic |
| 4 | Stream output | `operator<<` for printing; today only `info()` |
