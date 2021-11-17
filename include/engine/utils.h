#pragma once

#ifdef USE_SHADERC
#include <shaderc/shaderc.hpp>
#else
typedef int shaderc_shader_kind;
#define SHADERC_COMPUTE_SHADER 0
#endif
#include <string>
#include <fstream>
#include <iostream>
#include "engine.h"
#include "context.h"

inline size_t alignSize(size_t sz, int n) { return (sz + n - 1) & -n; }


template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}


static std::vector<uint32_t> compile(const std::string& source, char* filename=nullptr)
{
    std::string tmp_filename_in = tmpnam(nullptr);
    std::string tmp_filename_out = tmpnam(nullptr);
    FILE* tmp_file = nullptr;
    tmp_file = fopen(tmp_filename_in.c_str(), "wb+");
    fputs(source.c_str(), tmp_file);
    fclose(tmp_file);

    tmp_file = fopen(tmp_filename_out.c_str(), "wb+");
    fclose(tmp_file);


    std::string cmd_str = std::string("glslangValidator -V " + tmp_filename_in + " -S comp -o " + tmp_filename_out);
    
    
    std::cout << cmd_str << std::endl;
    auto system_return = system(cmd_str.c_str());
    if (system_return)
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream(tmp_filename_out, std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return { (uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size()) };
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


inline bool is_arithmetic(Format fmt) {
    return !(fmt == Format::kFormatBool || fmt == Format(-1));
}