#include "cookbook.hpp"

int main() {

    tensor t1 = tensor(-1.0, { 2,12 });
    tensor t2 = tensor(1.0, { 2,12 });
    
    auto mod = add();
    mod.forward(t1, t2);
    
}