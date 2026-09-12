#ifndef LAYOUT_HPP
# define LAYOUT_HPP

# include <array>
# include <vector>
# include <stdexcept>

namespace strata {

// Maximum Number of Dimensions
constexpr std::size_t MAX_NDIM = 32;

class Layout {

public:
    explicit Layout( const std::vector<std::size_t>& shape )
        : _ndim(shape.size()) {

        if (_ndim > MAX_NDIM)
            throw std::invalid_argument("shape exceeds MAX_NDIM");

        for (std::size_t i = 0; i < _ndim; i++)
            _shape[i] = shape[i];

        if (_ndim == 0)
            return ;

        // Calculate stride
        _stride[_ndim - 1] = 1;
        for (std::size_t i = _ndim - 1; i > 0; i--)
            _stride[i] = _stride[i + 1] * _shape[i + 1];

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

    template <typename... Indices>
    std::size_t offset( Indices... indices ) const {
        static_assert(sizeof...(Indices) > 0, "at least one index required");

        if (sizeof...(Indices) != _ndim)
            throw std::invalid_argument("wrong number of arguments");

        std::size_t offset = 0;
        std::size_t idx_arr[] = { static_cast<std::size_t>(indices)... };
        for (std::size_t i = 0; i < _ndim; i++) {
            if (idx_arr[i] >= _shape[i])
                throw std::out_of_range("index out of bounds");
            offset += idx_arr[i] * _stride[i];
        }

        return offset;
    }

    std::size_t ndim( void ) const { return _ndim; };

    std::vector<std::size_t> shape( void ) const {
        return std::vector<std::size_t>(_shape.begin(), _shape.begin() + _ndim);
    }

    std::size_t shape( const std::size_t index ) const { return _shape[index]; }

private:
    std::size_t _ndim;
    std::array<std::size_t, MAX_NDIM> _shape;
    std::array<std::size_t, MAX_NDIM> _stride;

};

}

#endif
