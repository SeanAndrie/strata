#ifndef TENSOR_HPP
# define TENSOR_HPP

# include "Layout.hpp"
# include "TensorView.hpp"
# include "TensorIterator.hpp"

# include <limits>
# include <memory>
# include <string>
# include <numeric>
# include <algorithm>

// TODO: [P4] Broadcasting Logic

namespace strata {

template <typename T>
class Tensor {

private:
    Layout _layout;
    std::unique_ptr<T[]> _data;

    static std::string vecToString( const std::vector<std::size_t>& v ) {
        std::string res = "(";
        for (std::size_t i = 0; i < v.size(); i++) {
            res += std::to_string(v[i]);
            if (i < v.size() - 1)
                res += " ";
        }
        res += ")";

        return res;
    }

public:

    // Default Constructor
    Tensor( void )
        : _layout(std::vector<std::size_t>{})
        , _data(nullptr) {}

    // Shape Constructor
    explicit Tensor( const std::vector<std::size_t>& shape )
        : _layout(shape), _data(std::make_unique<T[]>(_layout.size())) {
    }

    // Shape + Fill Value Constructor
    Tensor( const std::vector<std::size_t>& shape, T value )
        : _layout(shape)
        , _data(std::make_unique<T[]>(_layout.size())) {

        for (std::size_t i = 0; i < _layout.size(); i++)
            _data[i] = value;
    }

    // Copy Constructor
    Tensor( const Tensor& other )
        : _layout(other._layout)
        , _data(std::make_unique<T[]>(other.size())) {

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
        : _layout(other._layout)
        , _data(std::move(other._data)) {
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

    bool operator!=( const Tensor& other ) const {
        return !(*this == other);
    }

    // Multi-Index Operator
    template <typename... Indices>
    T& operator()( Indices... indices ) { return _data[_layout.offset(indices...)]; }

    // Const Multi-Index Operator
    template <typename... Indices>
    const T& operator()( Indices... indices) const { return _data[_layout.offset(indices...)]; }

    std::vector<std::size_t> shape( void ) const { return _layout.shape(); }
    std::size_t shape( const std::size_t index ) const { return _layout.shape(index); }

    std::vector<std::size_t> stride( void ) const { return _layout.stride(); }
    std::size_t stride( const std::size_t index ) const {return _layout.stride(index); }

    // Attributes
    std::size_t size( void ) const { return _layout.size(); }
    std::size_t ndim( void ) const { return _layout.ndim(); }

    // Prints out information about the tensor i.e., shape, stride, ndim
    void info( void ) const { _layout.info(); }

    /*
     * TensorIterator
     * --------------
    */

    TensorIterator<T> begin( void ) { return TensorIterator<T>(_data.get(), shape(), stride(), false); }
    TensorIterator<T> end( void ) { return TensorIterator<T>(_data.get(), shape(), stride(), true); }

    TensorIterator<const T> cbegin( void ) const { return TensorIterator<const T>(_data.get(), shape(), stride(), false); }
    TensorIterator<const T> cend( void ) const { return TensorIterator<const T>(_data.get(), shape(), stride(), true); }

    TensorIterator<const T> begin( void ) const { return cbegin(); }
    TensorIterator<const T> end( void ) const { return cend(); }

    /*
     * TensorView
     * ----------
     * A TensorView is an object that shares the same data as a Tensor object.
     * This allows us to perform operations such as slicing, transposing, reshapes, etc.
     * without overwriting the original data pointer or allocating memory for an entirely new tensor.
     *
     * For TensorView operations, we restrict their use on primarily lvalue references (&) and return
     * a compile error on rvalue references (&&). Since a TensorView borrows its data rather than owning it,
     * calling a TensorView operation on a temporary Tensor would leave the view pointing at memory
     * that's already been freed.
    */

    TensorView<T> view( void ) & { return TensorView<T>(_data.get(), _layout); }

    TensorView<T> permute( const std::vector<std::size_t>& axes ) & {
        if (axes.size() != ndim())
            throw std::invalid_argument("Tensor::permute: number of axes ("
                    + std::to_string(axes.size()) + ") don't match tensor dimensions ("
                    + std::to_string(ndim()) + ")");

        std::vector<std::size_t> newShape;
        std::vector<std::size_t> newStride;

        newShape.reserve(ndim());
        newStride.reserve(ndim());

        std::vector<bool> seen(ndim(), false);

        for (const std::size_t& axis : axes) {
            if (axis >= ndim())
                throw std::invalid_argument("Tensor::permute: axis ("
                        + std::to_string(axis) + ") is not valid for tensor with "
                        + std::to_string(ndim()) + "dimensions");
            if (seen[axis])
                throw std::invalid_argument("Tensor::permute: axis ("
                        + std::to_string(axis) + ") appears more than once");

            seen[axis] = true;
            newShape.push_back(shape(axis));
            newStride.push_back(stride(axis));
        }

        return TensorView<T>(_data.get(), Layout(newShape, newStride));
    }

    TensorView<T> transpose( void ) & {
        if (ndim() <= 1)
            return view();

        std::vector<std::size_t> axes(ndim());

        // Fills a range with consecutive increasing values given
        // `std::iota(start iter., end iter., T value)`
        std::iota(axes.begin(), axes.end(), 0);
        std::reverse(axes.begin(), axes.end());

        return permute(axes);
    }

    TensorView<T> transpose( const std::size_t a, const std::size_t b ) & {
        if (a >= ndim() || b >= ndim())
            throw std::invalid_argument("Tensor::transpose: axis out of range");

        std::vector<std::size_t> axes(ndim());
        std::iota(axes.begin(), axes.end(), 0);
        std::swap(axes[a], axes[b]);

        return permute(axes);
    }

    TensorView<T> reshape( const std::vector<std::size_t> &newShape ) & {
        if (newShape.empty()) {
            if (size() != 1)
                throw std::invalid_argument("Tensor::reshape: cannot reshape tensor of size "
                        + std::to_string(size()) + " to a scalar");
            return TensorView<T>(_data.get(), Layout(std::vector<std::size_t>{}));
        }

        std::size_t total = 1;
        for (std::size_t dim : newShape) {
            if (dim != 0 && total > std::numeric_limits<std::size_t>::max() / dim)
                throw std::length_error("Tensor::reshape: requested shape "
                        + vecToString(newShape) + " exceeds representable size");
            total *= dim;
        }

        if (total != size())
            throw std::invalid_argument("Tensor::reshape: cannot reshape tensor of shape "
                    + vecToString(shape()) + " into shape " + vecToString(newShape));

        return TensorView<T>(_data.get(), Layout(newShape));
    }

    template <typename... Slices>
    TensorView<T> slice( Slices... slices ) & {
        static_assert(sizeof...(Slices) > 0, "TensorView::slice: must have at least one slice");
        static_assert(sizeof...(Slices) <= MAX_NDIM, "TensorView::slice: number of slices exceeds MAX_NDIM");

        // Unpack Slice pack to Vector of Slice objects
        std::vector<Slice> sliceArr = { slices... };
        auto [layout, offset] = Layout::sliceLayout(_layout, sliceArr);

        return TensorView<T>(_data.get() + offset, layout);
    }

    TensorView<T> view( void ) && = delete;
    TensorView<T> transpose( void ) && = delete;
    TensorView<T> transpose( const std::size_t a, const std::size_t b ) && = delete;
    TensorView<T> permute( const std::vector<std::size_t>& axes ) && = delete;
    TensorView<T> reshape( const std::vector<std::size_t>& shape ) && = delete;

    template <typename... Slices>
    TensorView<T> slice( Slices... slices ) && = delete;

    // TODO: [P5] Tensor Operations

    static Tensor<T> arange( const std::size_t stop ) {
        Tensor<T> t({stop});

        for (std::size_t i = 0; i < stop; i++)
            t(i) = static_cast<T>(i);

        return t;
    }

    static Tensor<T> arange( const std::size_t start, const std::size_t stop, const std::size_t step = 1 ) {
        if (step == 0)
            throw std::invalid_argument("Tensor::arange: division by zero");
        if (start >= stop)
            throw std::invalid_argument("Tensor::arange: invalid range given");

        std::size_t total = (stop - start + step - 1) / step;

        Tensor<T> t({total});
        for (std::size_t i = 0; i < total; i++)
            t(i) = static_cast<T>(start + i * step);

        return t;
    }

    static Tensor<T> linspace( const double start, const double end, const std::size_t n = 50 ) {
        if (n == 0)
            throw std::invalid_argument("Tensor::linspace: n parameter cannot be zero");

        Tensor<T> t({n}, 0);

        if (n == 1) {
            t(0) = static_cast<T>(start);
            return t;
        }

        double step = (end - start) / (static_cast<double>(n) - 1);
        for (std::size_t i = 0; i < n; i++)
            t(i) = start + static_cast<double>(i) * step;

        return t;
    }

    static Tensor<T> ones( const std::vector<std::size_t>& shape ) { return Tensor<T>(shape, 1); }
    static Tensor<T> zeros( const std::vector<std::size_t>& shape ) { return Tensor<T>(shape, 0); }
    static Tensor<T> full( const std::vector<std::size_t>& shape, T value ) { return Tensor<T>(shape, value); }

};

}

namespace st = strata;

#endif
