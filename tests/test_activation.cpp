
#include "cookbook.hpp"
#include <iostream>
#include <chrono>

#define MS std::chrono::milliseconds


int main() {

    tensor t1 = tensor(-1.0, { 2,12 });
    auto init_mod = std::chrono::high_resolution_clock::now();
    auto mod = relu();
    auto mod_forward = std::chrono::high_resolution_clock::now();
    auto t3 = mod.forward(t1);

    auto mod_execution = std::chrono::high_resolution_clock::now();
    mod.runCommandBuffer();
    auto end_execution = std::chrono::high_resolution_clock::now();

    auto init_time = std::chrono::duration_cast<MS> (init_mod - mod_forward).count();
    auto forward_time = std::chrono::duration_cast<MS>(mod_forward - mod_execution).count();
    auto exec_time = std::chrono::duration_cast<MS>(mod_execution - end_execution).count();

    auto t = (float*)t3.toHost();
    bool issue = false;
    
    for (int i = 0; i < t3.count(); ++i) {
        if (t[i] != 0)
            issue = true;
    }

    std::cout << init_time << " " << forward_time << " " << exec_time << std::endl;

    return (int)issue;
}