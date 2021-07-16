#pragma once
#include <layer.h>
#include <tensor.h>

// constexpr int local_sz_x = 64;
// constexpr int local_sz_y = 16;

struct binary_op_params {
    int upper_bound;
    int lower_bound;
};

class binary_operator : public layer {
    explicit binary_operator();
    tensor& forward(tensor& x, tensor& w);
    binary_op_params m_param;
};

// arithmetic

class add : public binary_operator {
    explicit add();
};

class sub : public binary_operator {
    explicit sub();
};

class mul : public binary_operator {
    explicit mul();
};

class div : public binary_operator {
    explicit div();    
};

class max : public binary_operator {
    explicit max();
};

class min : public binary_operator {
    explicit min();
};

class pow : public binary_operator {
    explicit pow();
};

class binary_operator_logical : public layer {
    explicit binary_operator_logical();
    tensor& forward(tensor& x, tensor& w);
    binary_op_params m_param;
};

// mixed logical

class eq : public binary_operator_logical {
    explicit eq();
};

class ne : public binary_operator_logical {
    explicit ne();
};

class ge : public binary_operator_logical {
    explicit ge();    
};

class gt : public binary_operator_logical {
    explicit gt();
};

class le : public binary_operator_logical {
    explicit le();
};

class lt : public binary_operator_logical {
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