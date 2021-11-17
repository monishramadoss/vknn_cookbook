#pragma once
#include <layer.h>
#include <tensor.h>

#include "spv_shader.hpp"


template<typename param>
class masked_operator : public layer {
    param m_param = {};
    tensor mask;
    
public:
    binary_operator(const uint32_t* shaderCode, const size_t shaderSize) {
        initVulkanThing(4);
        _shader = shaderCode;
        _shader_size = shaderSize;
    }

    tensor& forward(tensor& x, tensor& w, tensor& y) {
        if (m_pipeline == nullptr) {

            m_param.upper_bound = 0;
            m_param.lower_bound = 0;
            m_group_x = static_cast<int>(alignSize(m_param.upper_bound, binary_op_local_sz_x)) / binary_op_local_sz_x;
            if (m_group_x > max_compute_work_group_count)
                m_group_x = max_compute_work_group_count - 1;
            if (m_group_y > max_compute_work_group_count)
                m_group_y = max_compute_work_group_count - 1;
            createShaderModule(_shader, _shader_size);
            createPipeline(sizeof(param));
        }
        if (y == nullptr)
            y = tensor(0, x.getShape());
        if (mask == nullptr)
            )

        bindtensor(x, 0);
        bindtensor(w, 1);
        bindtensor(maksed, 2)
        bindtensor(y, 3);
        recordCommandBuffer(static_cast<void*>(&m_param), sizeof(m_param));
    }
    binary_operator(binary_operator<param>& B) {
        *this = B;
    }
};