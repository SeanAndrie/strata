#ifndef TENSOR_VIEW_HPP
# define TENSOR_VIEW_HPP

# include "Layout.hpp"

namespace strata {

template <typename T>
class TensorView {

private:
    T* _data;
    Layout _layout;

public:
    TensorView( T* data, const Layout& layout )
        : _data(data), _layout(layout) {}

    template <typename... Indices>
    T& operator()( Indices... indices ) { return _data[_layout.offset(indices...)]; }

    template <typename... Indices>
    const T& operator()( Indices... indices ) const { return _data[_layout.offset(indices...)]; }

    std::size_t size( void ) const { return _layout.size(); }
    std::size_t ndim( void ) const { return _layout.ndim(); }
    std::vector<std::size_t> shape( void ) const { return _layout.shape(); }
    std::vector<std::size_t> stride( void ) const { return _layout.stride(); }

    void info( void ) const { _layout.info(); }

};

}

#endif
