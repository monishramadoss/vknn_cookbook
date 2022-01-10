#include "cookbook.hpp"

int main() {

    tensor t1 = tensor(1.0f, { 2,12 });
    tensor t2 = tensor(1.0f, { 2,12 });
    
    auto mod = add();
    tensor t3 = mod.forward(t1, t2);
    mod.runCommandBuffer();

    float* tmp2 = (float*)t2.download();
    float* tmp = (float*)t3.download();
    std::cout << tmp[0] << " " << tmp[1] << std::endl;
}