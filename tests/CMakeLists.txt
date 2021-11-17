

# target_include_directories(activation_test_cpp PUBLIC "${cookbook_include}")
add_executable(activation_test_cpp "${vulkan_ai_cookbook_SOURCE_DIR}/tests/test_activation.cpp" "${vulkan_ai_cookbook_SOURCE_DIR}/include/activation.hpp")
target_link_libraries(activation_test_cpp cookbook)
add_test(activation_fns activation_test_cpp)


add_executable(binary_ops_test_cpp "${vulkan_ai_cookbook_SOURCE_DIR}/tests/test_binary_ops.cpp" "${vulkan_ai_cookbook_SOURCE_DIR}/include/binary_ops.hpp")
target_link_libraries(binary_ops_test_cpp cookbook)
add_test(binary_ops_fns binary_ops_test_cpp)

add_executable(test_base_ops_cpp "${vulkan_ai_cookbook_SOURCE_DIR}/tests/test_main.cpp")
target_link_libraries(test_base_ops_cpp cookbook)
add_test(base_ops_fns test_base_ops_cpp)