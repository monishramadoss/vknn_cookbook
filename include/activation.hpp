#pragma once

#include <vector>

#include <engine.h>
#include <layer.h>
#include <tensor.h>
#include <utils.h>

#include "spv_shader.hpp"

struct activation_param {
    int total;
};

template<typename param>
class activation : public layer {
    param m_param = {};
    tensor act_field;
    tensor y;
    bool init;
public:
    activation(const uint32_t* shaderCode, const size_t codeSize) : init(false){
        initVulkanThing(3);
        _shader = shaderCode;   
        _shader_size = codeSize;
    }

    tensor& forward(tensor& x) {
        if (!init) {
            std::vector<int> tmp_data(x.count());
            act_field = tensor((char*)&tmp_data[0], x.getShape(), Format::kFormatInt32);
            init = true;
            y = tensor(0.0, x.getShape());
        }   
               
        forward(x, y, act_field);
        return y;
    }

    tensor& forward(tensor& x, tensor& y, tensor& activation_field) {
        if (m_pipeline == nullptr) {
            m_param.total = x.count();
            m_group_x = static_cast<int>(alignSize(m_param.total, local_sz)) / local_sz;
            if (m_group_x > max_compute_work_group_count)
                m_group_x = max_compute_work_group_count - 1;
            createShaderModule(_shader, _shader_size);
            createPipeline(sizeof(param));
        }

        bindtensor(x, 0);
        bindtensor(activation_field, 1);
        bindtensor(y, 2);
        recordCommandBuffer(static_cast<void*>(&m_param), sizeof(param));
        runCommandBuffer();
        return y;
    }

    activation(const activation<param>& A) {
        *this = A;
    }
};



class relu : public activation<activation_param> {
public:
    relu() : activation<activation_param>(relu_spv, sizeof(relu_spv)) {}
};