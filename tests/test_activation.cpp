
#include "cookbook.hpp"
#include <iostream>


int main() {

    tensor t1 = tensor(-1.0, { 2,12 });
    tensor t2 = tensor(1.0, { 2,12 });
    
    auto mod = relu();
    auto t3 = mod.forward(t1);
    mod.runCommandBuffer();
    auto t = (float*)t3.toHost();
    bool issue = false;
    
    for (int i = 0; i < t3.count(); ++i) {
        if (t[i] != 0)
            issue = true;
    }

    return (int)issue;
}