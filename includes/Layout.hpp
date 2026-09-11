#ifndef LAYOUT_HPP
# define LAYOUT_HPP

# include <array>
# include <vector>
# include <stdexcept>

namespace strata {

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

        _stride[_ndim - 1] = 1;
        for (std::size_t i = _ndim - 2; i-- > 0;)
            _stride[i] = _stride[i + 1] * _shape[i + 1];

    }

    std::size_t ndim( void ) const { return _ndim; };

    std::size_t size( void ) const {
        std::size_t total = 1;
        for (std::size_t i = 0; i < _ndim; i++)
            total *= _shape[i];
        return total;
    }


private:
    std::size_t _ndim;
    std::array<std::size_t, MAX_NDIM> _shape;
    std::array<std::size_t, MAX_NDIM> _stride;

};

}

#endif
