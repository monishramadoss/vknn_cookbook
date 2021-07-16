#pragma once
#include <layer.h>
#include <tensor.h>

// constexpr int local_sz_x = 1024;

struct unary_param {
    int total;
};

class unary_operator : public layer {
    explicit unary_operator();
    tensor& forward(tensor& x);
    unary_param m_param;
};

// trig

class cos : public unary_operator {
    explicit cos();
};

class cosh : public unary_operator {
    explicit cosh();
};

class sin : public unary_operator {
    explicit sin();
};

class sinh : public unary_operator {
    explicit sinh();
};

class tan : public unary_operator {
    explicit tan();
};

class tanh : public unary_operator {
    explicit tanh();
};

// arithmetic

class abs : public unary_operator {
    explicit abs();
};

class log : public unary_operator {
    explicit log();
};

class neg : public unary_operator {
    explicit neg();
};

class sqrt : public unary_operator {
    explicit sqrt();
};

class inverse : public unary_operator {
    explicit inverse();
};

// logical

class logical_not : public unary_operator {
    explicit logical_not();
};

