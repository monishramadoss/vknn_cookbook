#include <algorithm>

#include "common.h"
#include "utils.h"

void* map(int device_id, const VkDeviceMemory& mem, int size) {
    void* p=nullptr;
    VK_CHECK_RESULT(vkMapMemory(kDevices[device_id], mem, 0, size, 0, (void**)&p));
    return p;
}

tensor::tensor(tensor& t) {
    m_begin_offset = t.m_begin_offset;
    m_end_offset = t.m_end_offset;
    m_device_id = t.m_device_id;
    m_size_in_bytes = t.m_size_in_bytes;
    m_shape = t.m_shape;
    m_stride = t.m_stride;
    m_buffer = t.m_buffer;
}




void init_tensor(tensor* T, char* data, const std::vector<int>& shape, Format fmt) {
    createContext();
    T->m_device_id = 0;
    auto data_size = elementSize(fmt);
    T->m_size_in_bytes = std::accumulate(shape.begin(), shape.end(), 0) * data_size;
    T->m_begin_offset = 0;
    T->m_end_offset = T->m_size_in_bytes;
    T->m_shape = shape;
    T->m_stride.resize(shape.size());
    T->m_stride[0] = 1;
    for (size_t i = 1; i < shape.size(); ++i)
        T->m_stride[i] = shape[i - 1] * T->m_stride[i - 1];
    T->m_buffer.reset(new buffer(T->m_device_id, T->m_size_in_bytes, data));
}


tensor::tensor(Format fmt) : m_format(fmt)
{
    createContext();
    m_shape = { 0 };
    m_stride = { 0 };
    m_buffer = nullptr;
}

tensor::tensor(char* data, const std::vector<int>& shape, Format fmt) : m_format(fmt), m_device_id(0), m_begin_offset(0), m_end_offset(0), m_size_in_bytes(0)
{
    init_tensor(this, data, shape, fmt);
}

tensor::tensor(std::vector<float>& c, const std::vector<int>& shape) : m_format(Format::kFormatFp32), m_device_id(0), m_begin_offset(0), m_end_offset(0), m_size_in_bytes(0)
{
    init_tensor(this, (char*)c.data(), shape, Format::kFormatFp32);
}

tensor::tensor(float c, const std::vector<int>& shape) : m_format(Format::kFormatFp32), m_device_id(0), m_begin_offset(0), m_end_offset(0), m_size_in_bytes(0)
{
    char* c_arr = init::fill_memory_shape<float>(shape, c);
    init_tensor(this, c_arr, shape, m_format);
}


tensor tensor::reShape(const std::vector<int>& shape)
{
    const size_t _size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
    if (count() != _size)
        std::cerr << "SHAPE ERROR" << std::endl;
    if (m_shape != shape) m_shape = shape;
    return *this;
}


void tensor::toDevice(const std::vector<char>& data)
{

}


char* tensor::toHost() const
{
    char* dst = new char[size()];
    void* p = map(m_device_id, m_buffer->getVkMemory(), m_size_in_bytes);
    memcpy(dst, p, size());
    return  dst;
}