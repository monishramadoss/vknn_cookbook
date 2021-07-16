#pragma once

#ifdef USE_SHADERC
#include <shaderc/shaderc.hpp>
#else
typedef int shaderc_shader_kind;
#define SHADERC_COMPUTE_SHADER 0
#endif
#include <string>

#include "engine.h"
#include "context.h"

inline size_t alignSize(size_t sz, int n) { return (sz + n - 1) & -n; }

inline std::vector<uint32_t> compile(const std::string& name, const std::string& data)
{
    std::vector<uint32_t> result;
#ifdef USE_SHADERC

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetGenerateDebugInfo();
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(data.c_str(), data.size(), shaderc_glsl_compute_shader, name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cerr << module.GetErrorMessage();
    }
    result.assign(module.cbegin(), module.cend());
    return result;
#else
    return result;
#endif
}

inline bool checkFormat(Format fmt) { return fmt > Format::kFormatInvalid && fmt < Format::kFormatNone; }

inline size_t elementSize(Format fmt)
{
    if (fmt == Format::kFormatFp32 || fmt == Format::kFormatInt32 || fmt == Format::kFormatBool)
    {
        return 4;
    }
    if (fmt == Format::kFormatFp64 || fmt == Format::kFormatInt64)
    {
        return 8;
    }
    if (fmt == Format::kFormatFp16 || fmt == Format::kFormatInt16)
    {
        return 2;
    }
    if (fmt == Format::kFormatInt8 || fmt == Format::kFormatUInt8)
    {
        return 1;
    }
    if (fmt >= Format::kFormatFp16 && fmt < Format::kFormatNone)
    {
        printf("Unsupported format %d", fmt);
    }
    else
    {
        printf("Invalid format %d", fmt);
    }
    return 0;
}

inline int shapeCount(const Shape& shape, int start = -1, int end = -1)
{
    if (start == -1) start = 0;
    if (end == -1) end = static_cast<int>(shape.size());
    if (shape.empty()) return 0;
    int elems = 1;

    for (int i = start; i < end; i++)
    {
        if (elems * shape[i] <= INT32_MAX)
            elems *= shape[i];
    }

    return elems;
}

inline bool is_arithmetic(Format fmt) {
    return !(fmt == Format::kFormatBool || fmt == Format(-1));
}