#pragma once

#include <vector>

#include <engine.h>
#include <layer.h>
#include <tensor.h>

#include "spv_shader.h"

struct activation_param {
    int total;
};

template<typename param>
class activation : public layer {
    param m_param = { 0 };
    tensor act_field ;
public:
    activation(const uint32_t* shaderCode){
        initVulkanThing(3);
        _shader = shaderCode;
    }

    tensor& forward(tensor& x, tensor& y = nullptr, tensor& activation_field = nullptr) {
        if (m_pipeline == nullptr) {
            m_param.total = x.count();
            m_group_x = static_cast<int>(alignSize(m_param.total, local_sz)) / local_sz;
            if (m_group_x > max_compute_work_group_count)
                m_group_x = max_compute_work_group_count - 1;
            createShaderModule(_shader, sizeof(_shader));
            createPipeline(sizeof(param));
        }

        if (activation_field == nullptr && act_field == nullptr) {
            std::vector<bool> tmp_data;
            tmp_data.resize(x.count());
            std::fill_n<bool>(tmp_data, x.count(), 0);
            act_field = tensor(static_cast<char*>(tmp_data.data), x.getShape(), Format::kFormatBool);
            activation_field = act_field;
        }
        else if (activation_field != nullptr) {
            act_field = activation_field;
        }

        if (y == nullptr)
            y = tensor(0, x.getShape());

        bindtensor(x, 0);
        bindtensor(act_field, 1);
        bindtensor(y, 2);
        recordCommandBuffer(static_cast<void*>(&m_param), sizeof(param));
        return y
    }
};


class relu : public activation<activation_param> {
public:
    relu() : activation<activation_param>(relu_spv){}
};