#ifndef _ARRAYS_H_
#define _ARRAYS_H_
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <tuple>
#include <memory>
#include <iostream>
#include <cassert>

template <class T>
struct Array3D
{
    // T* data;
    std::unique_ptr<T[]> data;
    size_t size;
    size_t x;
    size_t y;
    size_t z;
    bool allocated = false;

    Array3D(size_t x, size_t y, size_t z) : x(x), y(y), z(z)
    {
        if (x * y * z > 0)
        {
            this->data = std::unique_ptr<T[]>(new T[x * y * z]);
            std::fill(&this->data[0], &this->data[x * y * z], 0);
            allocated = true;
        }
        else
        {
            allocated = false;
        }
    }

    T& operator[] (const std::tuple<size_t, size_t, size_t>& index)
    {
        size_t i = std::get<0>(index);
        size_t j = std::get<1>(index);
        size_t k = std::get<2>(index);

        assert(i < x);
        assert(j < y);
        assert(k < z);

        return this->data[
            i * y * z + j * z + k
        ];
    }

    T* begin()
    {
        return &this->data[0];
    }

    T* end()
    {
        return &this->data[x * y * z];
    }
};

#define accessArray(array, i, j, k) array[std::make_tuple(i, j, k)]

#endif