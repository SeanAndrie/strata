#ifndef TENSOR_VIEW_HPP
# define TENSOR_VIEW_HPP

# include <vector>
# include <string>
# include <limits>
# include <numeric>
# include <utility>
# include <algorithm>

# include "Layout.hpp"
# include "TensorIterator.hpp"

namespace strata {

template <typename T>
class TensorView {

private:
    T* _data;
    Layout _layout;

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
    TensorView( T* data, const Layout& layout )
        : _data(data), _layout(layout) {}

    template <typename... Indices>
    T& operator()( Indices... indices ) { return _data[_layout.offset(indices...)]; }

    template <typename... Indices>
    const T& operator()( Indices... indices ) const { return _data[_layout.offset(indices...)]; }

    /*
     * TensorIterator
     * --------------
    */

    TensorIterator<T> begin( void ) { return TensorIterator<T>(_data, shape(), stride(), false); }
    TensorIterator<T> end( void ) { return TensorIterator<T>(_data, shape(), stride(), true); }

    TensorIterator<const T> cbegin( void ) const { return TensorIterator<const T>(_data, shape(), stride(), false); }
    TensorIterator<const T> cend( void ) const { return TensorIterator<const T>(_data, shape(), stride(), true); }

    TensorIterator<const T> begin( void ) const { return cbegin(); }
    TensorIterator<const T> end( void ) const { return cend(); }

    /*
     * Attributes
     * ----------
    */

    std::size_t size( void ) const { return _layout.size(); }
    std::size_t ndim( void ) const { return _layout.ndim(); }

    std::vector<std::size_t> shape( void ) const { return _layout.shape(); }
    std::size_t shape( const std::size_t index ) const { return _layout.shape(index); }

    std::vector<std::size_t> stride( void ) const { return _layout.stride(); }
    std::size_t stride( const std::size_t index ) const { return _layout.stride(index); }

    void info( void ) const { _layout.info(); }

    template<typename... Slices>
    TensorView<T> slice( Slices... slices ) {
        static_assert(sizeof...(Slices) > 0, "TensorView::slice: must have at least one slice");
        static_assert(sizeof...(Slices) <= MAX_NDIM, "TensorView::slice: number of slices exceeds MAX_NDIM");

        std::vector<Slice> sliceArr = { slices... };
        auto [layout, offset] = Layout::sliceLayout(_layout, sliceArr);
        return TensorView<T>(_data + offset, layout);
    }

    TensorView<T> transpose( void ) {
        if (ndim() <= 1)
            return TensorView(_data, _layout);

        std::vector<std::size_t> axes(ndim());
        std::iota(axes.begin(), axes.end(), 0);
        std::reverse(axes.begin(), axes.end());

        return permute(axes);
    }

    TensorView<T> permute( const std::vector<std::size_t>& axes ) {
        if (axes.size() != ndim())
            throw std::invalid_argument("TensorView::permute: number of axes ("
                    + std::to_string(axes.size()) + ") don't match tensor dimensions ("
                    + std::to_string(ndim()) + ")");

        std::vector<std::size_t> newShape;
        std::vector<std::size_t> newStride;

        newShape.reserve(ndim());
        newStride.reserve(ndim());

        std::vector<bool> seen(ndim(), false);

        for (const std::size_t& axis : axes) {
            if (axis >= ndim())
                throw std::invalid_argument("TensorView::permute: axis ("
                        + std::to_string(axis) + ") is not valid for tensor with "
                        + std::to_string(ndim()) + "dimensions");
            if (seen[axis])
                throw std::invalid_argument("TensorView::permute: axis ("
                        + std::to_string(axis) + ") appears more than once");

            seen[axis] = true;
            newShape.push_back(shape(axis));
            newStride.push_back(stride(axis));
        }

        return TensorView<T>(_data, Layout(newShape, newStride));
    }

    TensorView<T> transpose( const std::size_t a, const std::size_t b ) {
        if (a >= ndim() || b >= ndim())
            throw std::invalid_argument("TensorView::transpose: axis out of range");

        std::vector<std::size_t> axes(ndim());
        std::iota(axes.begin(), axes.end(), 0);
        std::swap(axes[a], axes[b]);

        return permute(axes);
    }

    TensorView<T> reshape( const std::vector<std::size_t> &shape ) {
        if (shape.empty()) {
            if (size() != 1)
                throw std::invalid_argument("Tensor::reshape: cannot reshape tensor of size "
                        + std::to_string(size()) + " to a scalar");
            return TensorView<T>(_data, Layout(std::vector<std::size_t>{}));
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

        return TensorView<T>(_data, Layout(shape));
    }

};

}

#endif
