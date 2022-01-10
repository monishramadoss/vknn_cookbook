cmake_minimum_required(VERSION 3.20)
project(vulkan_ai_cookbook)


add_subdirectory(external/vma)
add_subdirectory(src)

	
include(CTest)
enable_testing()
add_subdirectory(tests)