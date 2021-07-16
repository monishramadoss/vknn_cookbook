find_package(Vulkan REQUIRED)
include_directories("${vulkan_ai_cookbook_SOURCE_DIR}/include")
set(cookbook_include "${vulkan_ai_cookbook_SOURCE_DIR}/include")

file(GLOB engine_CPP
    "${vulkan_ai_cookbook_SOURCE_DIR}/src/engine/*.cpp"
)

file(GLOB cookbook_CPP
    "${vulkan_ai_cookbook_SOURCE_DIR}/src/*.cpp"
)

add_library(engine STATIC ${engine_CPP})
target_include_directories(engine PUBLIC $<BUILD_INTERFACE:${Vulkan_INCLUDE_DIR}>)
target_include_directories(engine PUBLIC $<BUILD_INTERFACE:${vulkan_ai_cookbook_SOURCE_DIR}/include/engine>)
target_link_libraries(engine ${Vulkan_LIBRARY})

add_library(cookbook STATIC ${cookbook_CPP})
target_include_directories(cookbook PUBLIC $<BUILD_INTERFACE:${Vulkan_INCLUDE_DIR}>)
target_include_directories(cookbook PUBLIC $<BUILD_INTERFACE:${vulkan_ai_cookbook_SOURCE_DIR}/include/engine>)
target_include_directories(cookbook PUBLIC $<BUILD_INTERFACE:${vulkan_ai_cookbook_SOURCE_DIR}/include>)
target_link_libraries(cookbook ${Vulkan_LIBRARY})
target_link_libraries(cookbook engine)