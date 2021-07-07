#pragma once
#include <vector>
#include <layer.h>
#include <tensor.h>

class activation : public layer {
    explicit activation();
    tensor& forward(tensor& x);
};