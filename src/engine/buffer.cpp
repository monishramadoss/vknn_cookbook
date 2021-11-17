#include "common.h"
#include "utils.h"
#include "buffer.h"
#include "vk_mem_alloc.h"




bool buffer::init(size_t size_in_bytes, const char* data)
{
    if (m_buffer != nullptr)
    {
        printf("Warn: Buffer object already initiated\n");
        return false;
    }

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size_in_bytes;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(kDevices[m_device_id], &bufferCreateInfo, nullptr, &m_buffer));

    

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(kDevices[m_device_id], m_buffer, &memoryRequirements);
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(m_device_id, memoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(kDevices[m_device_id], &allocateInfo, nullptr, &m_memory));

    if (data)
    {
        char* dst = nullptr;
        VK_CHECK_RESULT(vkMapMemory(kDevices[m_device_id], m_memory, 0, 1024, 0, (void**)&dst));
        memcpy(dst, data, size_in_bytes);
        vkUnmapMemory(kDevices[m_device_id], m_memory);
    }

    VK_CHECK_RESULT(vkBindBufferMemory(kDevices[m_device_id], m_buffer, m_memory, 0));
    return true;
}


buffer::buffer(int device_id, size_t size_in_bytes, const char* data) {
    m_device_id = device_id;
    m_buffer = nullptr;
    m_memory = nullptr;
    init(size_in_bytes, data);
}

buffer::~buffer()
{
    vkFreeMemory(kDevices[m_device_id], m_memory, nullptr);
    vkDestroyBuffer(kDevices[m_device_id], m_buffer, nullptr);
}