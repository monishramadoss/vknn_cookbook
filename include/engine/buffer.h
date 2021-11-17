#pragma once

#include <vulkan/vulkan.h>

class buffer
{
public:
    buffer();
    buffer(int device_id, size_t size_in_bytes, const char* data);
    ~buffer();
    VkDeviceMemory getVkMemory() const {  return m_memory; }
    VkBuffer getVkBuffer() const { return m_buffer; }

private:
    bool init(size_t size_in_bytes, const char* data);
    
    int m_device_id;
    VkBuffer m_buffer;
    VkDeviceMemory m_memory;
};
