# ifndef TENSOR_ITERATOR_HPP
# define TENSOR_ITERATOR_HPP

# include <vector>
# include <utility>
# include <iterator>
# include <type_traits>

namespace strata {

template <typename T>
class TensorIterator {

private:
    T* _data;

    std::vector<std::size_t> _index;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride;

    bool _atEnd;

    std::size_t flatOffset( void ) const {
        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < _shape.size(); axis++)
            offset += _index[axis] * _stride[axis];
        return offset;
    }

public:

    /*
     * Iterator Traits
     * ---------------
     * Generic algorithms like `std::distance`, `std::for_each`, or range-based
     * for loops don't know anything about `TensorIterator` specifically. They work by
     * looking up `std::iterator_traits<YourIterator>`, which in turn looks for these
     * five type names inside your class.
    */

    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_const_t<T>;
    using reference = T&;
    using pointer = std::remove_reference_t<reference>*;

    TensorIterator(T* data, std::vector<std::size_t> shape, std::vector<std::size_t> stride, bool atEnd)
        : _data(data)
        , _index(shape.size(), 0)
        , _shape(std::move(shape))
        , _stride(std::move(stride))
        , _atEnd(atEnd) {

        if (_atEnd) {
            for (std::size_t i = 0; i < _index.size(); i++)
                _index[i] = _shape[i] - 1;
        }
    }

    reference operator*( void ) const { return _data[flatOffset()]; }
    pointer operator->( void ) const { return _data + flatOffset(); }

    TensorIterator& operator++( void ) {
        for (int axis = static_cast<int>(_shape.size()) - 1; axis >= 0; axis--) {
            _index[axis]++;
            if (_index[axis] < _shape[axis])
                return *this;
            _index[axis] = 0;
        }
        _atEnd = true;
        return *this;
    }

    TensorIterator operator++( int ) {
        TensorIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    TensorIterator operator+( const std::size_t n ) {
        TensorIterator tmp(*this);
        for (std::size_t i = 0; i < n; i++)
            ++tmp;
        return (tmp);
    }

    TensorIterator& operator+=( const std::size_t n ) {
        for (std::size_t i = 0; i < n; i++)
            ++(*this);
        return *this;
    }

    TensorIterator& operator--( void ) {
        for (int axis = static_cast<int>(_shape.size()) - 1; axis >= 0; axis--) {
            _index[axis]--;
            if (_index[axis] > 0)
                return *this;
        }
        _atEnd = false;
        return *this;
    }

    TensorIterator operator--( int ) {
        TensorIterator tmp(*this);
        --(*this);
        return tmp;
    }

    TensorIterator operator-( const std::size_t n ) {
        TensorIterator tmp(*this);
        for (std::size_t i = n; i > 0; i--)
            --tmp;
        return tmp;
    }

    bool operator==( const TensorIterator& other ) const {
        return _atEnd == other._atEnd && (_atEnd || _index == other._index);
    }

    bool operator!=( const TensorIterator& other ) const {
        return !(other == *this);
    }

};

}

#endif
