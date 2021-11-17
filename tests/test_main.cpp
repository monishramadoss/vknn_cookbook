#include <stdio.h>
#include <string>
#include <iostream>
#include <engine/engine.h>
#include <engine/utils.h>
#include <spv_shader.hpp>

std::string shader = (R"(
        #version 460
        layout(push_constant) uniform pushBlock {
              int total;      
        };

        layout(binding = 0) readonly buffer buf1 { float X[]; };

        layout(binding = 1) writeonly buffer buf2 {  float Y[]; };

        layout(local_size_x = %s, local_size_y = %s, local_size_z = 1) in;

        void main()
        {
            for (int i = int(gl_GlobalInvocationID.x); i < total; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x))
            {
                Y[i] = abs(X[i]);
            }
        }
    )");

int main() {
    int devices = number_devices();
    for (int i = 0; i < devices; ++i) {
        std::cout << "Device ID " << i << " Avalible Memory: " << avalible_memory(i) << std::endl;
    }

};

//std::cout << "size of compiled source" << x.count() << " " << size << std::endl;