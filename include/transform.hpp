#pragma once
#include <vector>
#include <layer.h>
#include <tensor.h>


class tensor2col : public layer {
    explicit tensor2col();
    tensor& forward(tensor& x);
};

class col2tensor : public layer {
    explicit col2tensor();
    tensor& forward(tensor& x);
};

class transpose : public layer {
    explicit transpose(std::vector<int> axes);
    tensor& forward(tensor& x);
};

class concat : public layer {
    explicit concat();
    tensor& forward(std::vector<tensor&>& xS);
};