#include <algorithm>

#include "common.h"
#include "utils.h"

void* map(int device_id, const VkDeviceMemory& mem, int size,  int offset=0) {
    void* p=nullptr;
    VK_CHECK_RESULT(vkMapMemory(kDevices[device_id], mem, offset, size, 0, (void**)&p));
    return p;
}

inline void rewrite_stride_size(std::vector<int>& shape, std::vector<int>& stride, std::vector<int>& size) {
    stride.resize(shape.size());
    size.resize(shape.size() + 1);

    stride[0] = 1;
    for (size_t i = 1; i < shape.size(); ++i)
        stride[i] = shape[i - 1] * stride[i - 1];

    size[shape.size()] = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
        size[i] = shape[i] * size[i + 1];
}

void init_tensor(tensor* T, char* data, const std::vector<int>& shape, Format fmt) {
    createContext();
    T->m_device_id = 0;
    auto data_size = elementSize(fmt);
    T->m_begin_offset = 0;
    T->m_end_offset = T->m_size_in_bytes;
    T->m_shape = shape;
    rewrite_stride_size(T->m_shape, T->m_stride, T->m_size);
    T->m_size_in_bytes = T->m_size[0] * data_size;
    T->m_buffer.reset(new buffer(T->m_device_id, T->m_size_in_bytes, data));
    T->host_data = std::unique_ptr<char>(new char[T->m_size_in_bytes]);
    memcpy(T->host_data.get(), data, T->m_size_in_bytes);
}


tensor::tensor(Format fmt) : m_format(fmt)
{
    createContext();
    m_shape = { 0 };
    m_stride = { 0 };
    m_size = { 0 };
    m_buffer = nullptr;
    host_data = nullptr;
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

tensor::tensor(int c, const std::vector<int>& shape) : m_format(Format::kFormatInt32), m_device_id(0), m_begin_offset(0), m_end_offset(0), m_size_in_bytes(0)
{
    char* c_arr = init::fill_memory_shape<int>(shape, c);
    init_tensor(this, c_arr, shape, m_format);
}

tensor::tensor(const tensor& t) {
    m_begin_offset = t.m_begin_offset;
    m_end_offset = t.m_end_offset;
    m_device_id = t.m_device_id;
    m_size_in_bytes = t.m_size_in_bytes;
    m_shape = t.m_shape;
    m_size = t.m_size;
    m_stride = t.m_stride;
    m_buffer = t.m_buffer;
    host_data.reset(t.download());
}

tensor& tensor::operator=(const tensor& rhs){
    m_begin_offset = rhs.m_begin_offset;
    m_end_offset = rhs.m_end_offset;
    m_device_id = rhs.m_device_id;
    m_size_in_bytes = rhs.m_size_in_bytes;
    m_shape = rhs.m_shape;
    m_size = rhs.m_size;
    m_stride = rhs.m_stride;
    m_buffer = rhs.m_buffer;
    host_data.reset(rhs.download());
    upload(host_data);
    return *this;
}

tensor tensor::reShape(const std::vector<int>& shape)
{
    const size_t _size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<int>());
    if (m_size[0] != _size)
        std::cerr << "SHAPE ERROR" << std::endl;
    if (m_shape != shape) m_shape = shape;
    rewrite_stride_size(m_shape, m_stride, m_size);
    return *this;
}


void tensor::upload(const std::vector<char>& data, uint32_t offset)
{
    void* dst = map(m_device_id, m_buffer->getVkMemory(), data.size(), offset);    
    memcpy(dst, data.data(), data.size());
    memcpy(host_data.get() + offset, data.data(), data.size());
}

void tensor::upload(std::unique_ptr<char>& ptr) {
    void* dst = map(m_device_id, m_buffer->getVkMemory(), m_size_in_bytes, m_begin_offset);
    memcpy(dst, ptr.get(), m_size_in_bytes);
}

char* tensor::download(uint32_t offset) const
{
    void* src = map(m_device_id, m_buffer->getVkMemory(), m_size_in_bytes, offset);
    memcpy(host_data.get() + offset, src, m_size_in_bytes - offset);
    return host_data.get();
}

tensor& tensor::slice(int split_size, int axes = 0) {
    int shape = m_shape[axes];
    int splits = std::ceil(shape / split_size);
    int size = m_size[axes];
    shard_set.resize(splits);
    for (int i = 0; i < splits; ++i) {
        shard_set[i] = tensor();
    }
    return shard_set[0];
}

tensor& tensor::slice(std::vector<int> split_shape, int axes = 0) {
    int shape = m_shape[axes];
    int stride = m_stride[axes];
    int size = m_size[axes];
        
}


std::vector<tensor> tensor::shard(std::vector<int>& axes) {
    for (size_t i = 0; i < axes.size(); ++i) {
       int shape = m_shape[axes[i]];
       int size = m_size[axes[i]];
       int stride = m_stride[axes[i]];
    }
    
    return shard_set;
}

