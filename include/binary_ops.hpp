#pragma once
#include <layer.h>
#include <tensor.h>

#include "spv_shader.hpp"

constexpr int binary_op_local_sz_x = 64;
constexpr int binary_op_local_sz_y = 16;

struct binary_op_params {
    int upper_bound;
    int lower_bound;
};


template<typename param>
class binary_operator : public layer {
     param m_param = {};

public:
    binary_operator(const uint32_t* shaderCode, const size_t shaderSize){
        initVulkanThing(3);
        _shader = shaderCode;
        _shader_size = shaderSize;
    }

    tensor& forward(tensor& x, tensor& w, tensor& y = nullptr){
        if(m_pipeline == nullptr){
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
        
        bindtensor(x, 0);
        bindtensor(w, 1);
        bindtensor(y, 2);
        recordCommandBuffer(static_cast<void*>(&m_param), sizeof(m_param));
    }
    binary_operator(binary_operator<param>& B) {
        *this = B;
    }
};

// arithmetic

class add : public binary_operator<binary_op_params>{
public:
    add() : binary_operator<binary_op_params>(add_spv, sizeof(add_spv)){};
};

class sub : public binary_operator<binary_op_params> {
public:
    sub();
};

class mul : public binary_operator<binary_op_params> {
public:
    mul();
};

class div : public binary_operator<binary_op_params> {
public:
    div();    
};

class max : public binary_operator<binary_op_params> {
public:
    max();
};

class min : public binary_operator<binary_op_params> {
public:
    min();
};

class pow : public binary_operator<binary_op_params> {
public:
    pow();
};

class binary_operator_logical : public layer {
    explicit binary_operator_logical();
    tensor& forward(tensor& x, tensor& w);
    binary_op_params m_param;
};

// mixed logical

class eq : public binary_operator_logical {
public:
    eq();
};

class ne : public binary_operator_logical {
public:
    ne();
};

class ge : public binary_operator_logical {
public:
    ge();    
};

class gt : public binary_operator_logical {
public:
    gt();
};

class le : public binary_operator_logical {
public:
    le();
};

class lt : public binary_operator_logical {
public:
    explicit lt();
};


// pure logical

class logical_or : public binary_operator_logical {
    explicit logical_or();
    tensor& forward(tensor& x, tensor& w);
};

class logical_and : public binary_operator_logical {
    explicit logical_and();
    tensor& forward(tensor& x, tensor& w);
};

class logical_xor : public binary_operator_logical {
    explicit logical_xor();
    tensor& forward(tensor& x, tensor& w);
};