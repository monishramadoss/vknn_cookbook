
#include "cookbook.hpp"
#include <iostream>
#include <chrono>

#define MS std::chrono::milliseconds

template<typename layer_type>
bool test_activation(tensor& t1=nullptr) {
    auto init_mod = std::chrono::high_resolution_clock::now();
    auto mod = layer_type();
    auto mod_forward = std::chrono::high_resolution_clock::now();
    tensor t3 = mod.forward(t1);
    auto mod_execution = std::chrono::high_resolution_clock::now();
    mod.runCommandBuffer();
    auto end_execution = std::chrono::high_resolution_clock::now();
    auto init_time = -std::chrono::duration_cast<MS> (init_mod - mod_forward).count();
    auto forward_time = -std::chrono::duration_cast<MS>(mod_forward - mod_execution).count();
    auto exec_time = -std::chrono::duration_cast<MS>(mod_execution - end_execution).count();

    auto t = (float*)t3.download();
    bool issue = false;
    for (int i = 0; i < t3.count(); ++i) {
        if (t[i] == 0)
            issue = true;
    }

    if (!issue)
        std::cout << "ACTIVATION FN " << init_time << " " << forward_time << " " << exec_time << std::endl;
    else
        std::cout << "ACTIVATION KERNEL FAILURE" << std::endl;
    return issue;
}

int main() {

    tensor t1 = tensor(1.0f, { 2,12 });
    bool issue = test_activation<relu>(t1);
    


    return (int)issue;
}