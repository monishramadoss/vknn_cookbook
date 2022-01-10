#pragma once
#include <vector>
#include <layer.h>
#include <tensor.h>

int32_t Offset(int32_t in_offset, const int32_t* out_stride, const int32_t* out_shape, const int32_t n) {
    int32_t remaining = 0;
    int32_t out_offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int32_t dim = n; dim >= 0; --dim) {
        remaining = in_offset % out_shape[dim];
        out_offset += remaining * out_stride[dim];
        in_offset = in_offset / out_shape[dim];
    }
    return out_offset;
}

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