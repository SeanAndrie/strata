#ifndef TENSOR_HPP
# define TENSOR_HPP

# include "Layout.hpp"

# include <memory>

namespace strata {

template <typename T>
class Tensor {

public:
    Tensor( void )
        : _layout(std::vector<std::size_t>{}), _data(nullptr) {}

    explicit Tensor( const std::vector<std::size_t>& shape )
        : _layout(shape), _data(std::make_unique<T[]>(_layout.size())) {};

    std::size_t size( void ) const { return _layout.size(); }

private:
    Layout _layout;
    std::unique_ptr<T[]> _data;

};

}

namespace st = strata;

#endif
