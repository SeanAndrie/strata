#ifndef LAYOUT_HPP
# define LAYOUT_HPP

# include <array>
# include <vector>
# include <utility>
# include <sstream>
# include <iostream>
# include <stdexcept>

namespace strata {

// Maximum Number of Dimensions
constexpr std::size_t MAX_NDIM = 32;

struct Slice {
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    std::size_t start;
    std::size_t stop;
    std::size_t step;

    constexpr Slice( const std::size_t start = 0, const std::size_t stop = npos, const std::size_t step = 1 )
        : start(start), stop(stop), step(step) {}
};

inline constexpr Slice all{};

class Layout {

private:
    std::size_t _ndim;
    std::array<std::size_t, MAX_NDIM> _shape;
    std::array<std::size_t, MAX_NDIM> _stride;

public:
    explicit Layout( const std::vector<std::size_t>& shape )
        : _ndim(shape.size()), _shape{}, _stride{} {

        if (_ndim > MAX_NDIM)
            throw std::invalid_argument("Layout: shape exceeds MAX_NDIM");
        if (_ndim == 0)
            return ;

        for (std::size_t i = 0; i < _ndim; i++)
            _shape[i] = shape[i];

        _stride[_ndim - 1] = 1;
        for (std::size_t i = _ndim - 1; i-- > 0;)
            _stride[i] = _stride[i + 1] * _shape[i + 1];
    }

    Layout( const std::vector<std::size_t>& shape, const std::vector<std::size_t>& stride )
        : _ndim(shape.size()), _shape{}, _stride{} {

        if (_ndim != stride.size())
            throw std::invalid_argument("Layout: shape and stride size mismatch");
        if (_ndim > MAX_NDIM)
            throw std::invalid_argument("Layout: shape exceeds MAX_NDIM");

        for (std::size_t i = 0; i < _ndim; i++) {
            _shape[i] = shape[i];
            _stride[i] = stride[i];
        }
    }

    Layout( const Layout& other )
        : _ndim(other._ndim),
          _shape(other._shape),
          _stride(other._stride) {};

    Layout& operator=( const Layout& other ) {
        if (this != &other) {
            _ndim = other._ndim;
            _shape = other._shape;
            _stride = other._stride;
        }
        return *this;
    }

    std::size_t size( void ) const {
        std::size_t total = 1;
        for (std::size_t i = 0; i < _ndim; i++)
            total *= _shape[i];
        return total;
    }

    bool isContiguous( void ) const {
        if (_ndim == 0)
            return true;

        std::size_t expStride = 1;
        for (std::size_t i = _ndim; i-- > 0;) {
            if (_stride[i] != expStride)
                return false;
            expStride *= _shape[i];
        }

        return true;
    }

    /*
     * Pack
     * ----
     * A C++ entity that defines one of the following
     *  - a parameter pack
     *    - template parameter pack
     *    - function parameter pack
    */

    template <typename... Indices>
    std::size_t offset( Indices... indices ) const {
        static_assert(sizeof...(Indices) > 0, "Layout::offset: at least one index required");

        if (sizeof...(Indices) != _ndim)
            throw std::invalid_argument("Layout::offset: incorrect number of arguments");

        std::size_t offset = 0;
        std::size_t idxArr[] = { static_cast<std::size_t>(indices)... };
        for (std::size_t i = 0; i < _ndim; i++) {
            if (idxArr[i] >= _shape[i])
                throw std::out_of_range("Layout::offset: index out of bounds");
            offset += idxArr[i] * _stride[i];
        }

        return offset;
    }

    std::size_t ndim( void ) const { return _ndim; };

    // Shape Getter (index & vector)
    std::size_t shape( const std::size_t index ) const { return _shape[index]; }
    std::vector<std::size_t> shape( void ) const {
        return std::vector<std::size_t>(_shape.begin(), _shape.begin() + _ndim);
    }

    // Stride Getter (index & vector)
    std::vector<std::size_t> stride( void ) const {
        return std::vector<std::size_t>(_stride.begin(), _stride.begin() + _ndim);
    }
    std::size_t stride( const std::size_t index ) const { return _stride[index]; }

    void info( void ) const {
        std::ostringstream oss;

        std::string horiRule;
        for (std::size_t i = 0; i < 11 + _ndim; i++)
            horiRule += "-";

        oss << horiRule;
        oss << "\n";

        oss << "ndim: " << _ndim << "\n";
        oss << "size: " << size() << "\n";

        oss << "shape: [";
        for (auto it = _shape.begin(); it != _shape.begin() + _ndim; it++) {
            oss << *it;
            if (it != _shape.begin() + _ndim - 1)
                oss << " ";
        }
        oss << "]\n";

        oss << "stride: [";
        for (auto it = _stride.begin(); it != _stride.begin() + _ndim; it++) {
            oss << *it;
            if (it != _stride.begin() + _ndim - 1)
                oss << " ";
        }
        oss << "]\n" << horiRule;

        std::cout << oss.str() << std::endl;
    }

    static inline std::pair<Layout, std::size_t>
    sliceLayout( const Layout& base, const std::vector<Slice>& slices ) {
        if (slices.size() != base.ndim())
            throw std::invalid_argument("Layout::sliceLayout: number of slices ("
                    + std::to_string(slices.size()) + ") must match tensor dimensions ("
                    + std::to_string(base.ndim()) + ")");

        std::vector<std::size_t> newShape;
        std::vector<std::size_t> newStride;
        newShape.reserve(slices.size());
        newStride.reserve(slices.size());

        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < slices.size(); axis++) {
            const Slice& s_i = slices[axis];

            if (s_i.step == 0)
                throw std::invalid_argument("Layout::sliceLayout: step cannot be zero");

            std::size_t stop = (s_i.stop == Slice::npos) ? base.shape(axis) : s_i.stop;

            if (stop > base.shape(axis))
                throw std::out_of_range("Layout::sliceLayout: stop exceeds axis size");
            if (s_i.start > stop)
                throw std::invalid_argument("Layout::sliceLayout: start cannot be greater than stop");

            newShape.push_back((stop - s_i.start + s_i.step - 1) / s_i.step);
            offset += s_i.start * base.stride(axis);
            newStride.push_back(base.stride(axis) * s_i.step);
        }

        return {Layout(newShape, newStride), offset};
    }

};

}

#endif
