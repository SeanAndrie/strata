#ifndef TENSOR_HPP
# define TENSOR_HPP

# include "Layout.hpp"
# include "TensorView.hpp"

# include <limits>
# include <memory>
# include <string>
# include <algorithm>

// TODO: [P2] TensorIterator Class (multi-index traversal + fast path for contiguous)
//            Define implementation in separate header

// TODO: [P3] Tensor Creation (Zeros, Ones, Linearly-spaced)
//            Define implementations in separate `.cpp` file (?)

// TODO: [P4] Broadcasting Logic

namespace strata {

template <typename T>
class Tensor {

private:
    Layout _layout;
    std::unique_ptr<T[]> _data;

    std::size_t shape( const std::size_t index ) { return _layout.shape(index); }
    std::size_t stride( const std::size_t index ) { return _layout.stride(index); }

    std::string vecToString( const std::vector<std::size_t>& v ) const {
        std::string res = "(";
        for (std::size_t i = 0; i < v.size(); i++) {
            res += v[i];
            if (i != v.size() - 1)
                res += " ";
        }
        res += ")";
        return res;
    }

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
     * lvalue (&)
     * ----------
     * These are objects that occupy a identifiable location in memory.
     * (it has a name and an address you can take).
     *
     * rvalue (&&)
     * -----------
     * Temporary, disposable values that do not have a persistent memory
     * address.
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
        for (std::size_t i = 0; i < size(); i++) {
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
    std::vector<std::size_t> stride( void ) const { return _layout.stride(); }

    /*
     * TensorView
     * ----------
     * A TensorView is an object that shares the same data as a Tensor object.
     * This allows us to perform operations such as slicing, transposing, reshapes, etc.
     * without overwriting the original data pointer or allocating memory for an entirely new tensor.
     *
     * For TensorView operations, we restrict their use on primarily lvalue references (&) and return
     * a compile error on rvalue references (&&). Since a TensorView borrows its data rather than owning it,
     * and calling this on a temporary Tensor would leave the view pointing at memory that's already been freed.
    */

    TensorView<T> view( void ) & { return TensorView<T>(_data.get(), _layout); }

    TensorView<T> transpose( void ) & {
        if (ndim() <= 1)
            return view();

        std::vector<std::size_t> newShape = shape(), newStride = stride();

        std::reverse(newShape.begin(), newShape.end());
        std::reverse(newStride.begin(), newStride.end());

        return TensorView<T>(_data.get(), Layout(newShape, newStride));
    }

    TensorView<T> transpose( const std::vector<std::size_t>& axes ) & {
        if (axes.size() != ndim())
            throw std::invalid_argument("Tensor::transpose: number of axes ("
                    + std::to_string(axes.size()) + ") don't match tensor dimensions ("
                    + std::to_string(ndim()) + ")");

        std::vector<std::size_t> newShape, newStride;
        for (const std::size_t& ax : axes) {
            if (ax >= ndim())
                throw std::invalid_argument("Tensor::transpose: axis ("
                        + std::to_string(ax) + ") is not valid for tensor with "
                        + std::to_string(ndim()) + "dimensions");
            newShape.push_back(shape(ax));
            newStride.push_back(stride(ax));
        }

        return TensorView<T>(_data.get(), Layout(newShape, newStride));
    }

    TensorView<T> reshape( const std::vector<std::size_t> &shape ) & {
        if (shape.empty()) {
            if (size() != 1)
                throw std::invalid_argument("Tensor::reshape: cannot reshape tensor of size "
                        + std::to_string(size()) + " to a scalar");
            return TensorView<T>(_data.get(), Layout(std::vector<std::size_t>{}));
        }

        std::size_t total = 1;
        for (std::size_t dim : shape) {
            if (dim != 0 && total > std::numeric_limits<std::size_t>::max() / dim)
                throw std::length_error("Tensor::reshape: requested shape " 
                        + vecToString(shape) + " exceeds representable size");
            total *= dim;
        }

        if (total != size())
            throw std::invalid_argument("Tensor::reshape: cannot reshape tensor of shape "
                    + vecToString(this->shape()) + " into shape " + vecToString(shape));

        Layout new_layout(shape);
        return TensorView<T>(_data.get(), new_layout);
    }

    // TODO: [P1] Slice Operation (single-slice & flexible-slice w/ <start, end, step>)

    TensorView<T> view( void ) && = delete;
    TensorView<T> transpose( void ) && = delete;
    TensorView<T> transpose( const std::vector<std::size_t>& shape ) && = delete;
    TensorView<T> reshape( const std::vector<std::size_t>& shape ) && = delete;

    // TODO: [P5] Tensor Operations

    void fill(T value) {
        for (std::size_t i = 0; i < this->size(); i++)
            _data[i] = value;
    }

    void info( void ) const { _layout.info(); }

};

}

namespace st = strata;

#endif
