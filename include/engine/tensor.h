#pragma once

#include <memory>
#include <numeric>
#include <iomanip>
#include <vulkan/vulkan.h>
#include "engine.h"
#include "utils.h"

class buffer;


inline int shapeCount(const Shape& shape, int start = -1, int end = -1)
{
    if (start == -1) start = 0;
    if (end == -1) end = static_cast<int>(shape.size());
    if (shape.empty()) return 0;
    int elems = 1;

    for (int i = start; i < end; i++)
    {
        if (elems * shape[i] <= INT32_MAX)
            elems *= shape[i];
    }

    return elems;
}


class tensor
{
public:
    tensor(Format fmt = Format::kFormatFp32);
    tensor(char* data, const std::vector<int>& shape, Format fmt = Format::kFormatFp32);
    tensor(std::vector<float>& c, const std::vector<int>& shape);
    tensor(float c, const std::vector<int>& shape);
    tensor(tensor&);
   
    Shape getShape() const { return m_shape; }
    int dim() const { return static_cast<int>(m_shape.size()); }
    int dimSize(int axis) const { return axis >= 0 || m_shape.size() > axis ? -1 : m_shape[axis]; }
    size_t count(int start_axis = 0, int end_axis = -1) const { return shapeCount(m_shape, start_axis, end_axis); }
    char* toHost() const;
    tensor reShape(const std::vector<int>& shape);
    void toDevice(const std::vector<char>& val);

    Format getFormat() const { return m_format; }
    size_t size() const {
        return m_size_in_bytes; 
    }
    bool isEmpty() const { return m_size_in_bytes == 0; }
    std::shared_ptr<buffer> getBuffer() { return m_buffer; }
private:
    friend void init_tensor(tensor* T, char* data, const std::vector<int>& shape, Format fmt);

    std::vector<int> m_shape;
    std::vector<int> m_stride;
    int m_begin_offset;
    int m_end_offset;

    size_t m_size_in_bytes;
    Format m_format;
    std::shared_ptr<buffer> m_buffer;
    int m_device_id;
};

void init_tensor(tensor* T, char* data, const std::vector<int>& shape, Format fmt);

namespace init
{
    template <typename dType = float>
    char* fill_memory_shape(std::vector<int> shape, dType c)
    {
        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new dType[_shape];
        for (int i = 0; i < _shape; ++i)
            ret[i] = reinterpret_cast<dType&>(c);
        return reinterpret_cast<char*>(ret);
    }

    template <typename dType = float>
    char* fill_memory_iter(std::vector<int> shape)
    {
        const size_t _shape = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
        auto ret = new dType[_shape];
        for (int i = 0; i < _shape; ++i)
            ret[i] = reinterpret_cast<dType&>(i);
        return reinterpret_cast<char*>(ret);
    }
}
