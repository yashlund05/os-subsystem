cmake_minimum_required(VERSION 3.20)
project(neuroos-lite VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler Warnings and Flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# SIMD & Fast-path compilation flags (conditional)
option(ENABLE_AVX2 "Enable AVX2 integer instructions for micro-inference" OFF)
if(ENABLE_AVX2)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2)
    endif()
endif()

# Core include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/allocators/include
    ${CMAKE_CURRENT_SOURCE_DIR}/telemetry/ring_buffer/include
)

# Placeholder/Scaffold Library for Kernel Common Headers & Interfaces
add_library(neuroos_core_scaffold INTERFACE)
target_include_directories(neuroos_core_scaffold INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/schedulers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/allocators/include
    ${CMAKE_CURRENT_SOURCE_DIR}/telemetry/ring_buffer/include
)

# Scaffolding Test: Interface and Header Validation
add_executable(test_headers_scaffold
    tests/kernel/test_headers_scaffold.c
)
target_link_libraries(test_headers_scaffold PRIVATE neuroos_core_scaffold)

# Testing support
enable_testing()
add_test(NAME HeaderInterfaceSanity COMMAND test_headers_scaffold)

message(STATUS "NeuroOS-Lite Foundation build configured successfully.")
message(STATUS "Build target: test_headers_scaffold")
