#pragma once
#include <layer.h>
#include <tensor.h>

// level1 axpy (y = ax + y) dot product, vector norms
class axpy : public layer {
    explicit axpy();
    tensor& forward(tensor& x, tensor& w);
};

// level2 gemv (y = aAx + by) matrix vector multiplications
class gemv : public layer {
    explicit gemv();
    tensor& forward(tensor& x, tensor& w);
};

// level3 gemm (y = aA^T B + bC) 

class gemm : public layer {
    explicit gemm();
    tensor& forward(tensor& x, tensor& w);
};