#ifndef TENSOR_HPP
# define TENSOR_HPP

# include "Layout.hpp"
# include "TensorView.hpp"

# include <memory>

namespace strata {

template <typename T>
class Tensor {

public:

    // Default Constructor
    Tensor( void )
        : _layout(std::vector<std::size_t>{}),
          _data(nullptr) {}

    // Shape Constructor
    explicit Tensor( const std::vector<std::size_t>& shape )
        : _layout(shape),
          _data(std::make_unique<T[]>(_layout.size())) {}

    // Shape + Fill Value Constructor
    Tensor( const std::vector<std::size_t>& shape, T value )
        : _layout(shape),
          _data(std::make_unique<T[]>(_layout.size())) {

        for (std::size_t i = 0; i < _layout.size(); i++)
            _data[i] = value;
    }

    // Copy Constructor
    Tensor( const Tensor& other )
        : _layout(other._layout),
          _data(std::make_unique<T[]>(other.size())) {

        for (std::size_t i = 0; i < other.size(); i++)
            _data[i] = other._data[i];
    }

    // Copy-assignment Operator
    Tensor& operator=( const Tensor& other ) {
        if (this != &other) {
            _layout = other._layout;
            _data = std::make_unique<T[]>(other.size());

            for (std::size_t i = 0; i < other.size(); i++)
                _data[i] = other._data[i];
        }
        return *this;
    }

    /*
     * Move Semantics
     * --------------
     * Move gets used whenever the compiler can prove, or you explicitly
     * assert with `std::move`, that the source is a temporary or otherwise
     * about to be discarded.
     *
     * `T&&` - rvalue reference
     * ------------------------
     * A normal reference (`T&`) binds to an lvalue, something with a name,
     * something that persists. `T&&` binds to an rvalue instead, a temporary, something
     * about to be destroyed, or something explicitly cast with `std::move`.
    */

    // Move Operator
    // Equivalent to Tensor( Tensor&& other ) noexcept = default
    Tensor( Tensor&& other ) noexcept
        : _layout(other._layout),
          _data(std::move(other._data)) {
        other._layout = Layout(std::vector<std::size_t>{});
    }

    // Move-assignment Operator
    // Equivalent to Tensor& operator=( Tensor&& other ) noexcept = default
    Tensor& operator=( Tensor&& other ) noexcept {
        if (this != &other) {
            _layout = other._layout;
            _data = std::move(other._data);
            other._layout = Layout(std::vector<std::size_t>{});
        }
        return *this;
    }

    // Generic Equality Operator
    bool operator==( const Tensor& other ) const {
        if (shape() != other.shape())
            return false;
        for (std::size_t i = 0; i < ndim(); i++) {
            if (_data[i] != other._data[i])
                return false;
        }
        return true;
    }

    // Multi-Index Operator
    template <typename... Indices>
    T& operator()( Indices... indices ) { return _data[_layout.offset(indices...)]; }

    // Const Multi-Index Operator
    template <typename... Indices>
    const T& operator()( Indices... indices) const { return _data[_layout.offset(indices...)]; }

    // Attributes
    std::size_t size( void ) const { return _layout.size(); }
    std::size_t ndim( void ) const { return _layout.ndim(); }
    std::vector<std::size_t> shape( void ) const { return _layout.shape(); }
    std::size_t shape( const std::size_t index ) { return _layout.shape(index); }

    // Views
    TensorView<T> view( void ) & { return TensorView<T>(_data.get(), _layout); }
    TensorView<T> view( void ) && = delete;

    void fill(T value) {
        for (std::size_t i = 0; i < this->size(); i++)
            _data[i] = value;
    }

private:
    Layout _layout;
    std::unique_ptr<T[]> _data;

};

}

namespace st = strata;

#endif
