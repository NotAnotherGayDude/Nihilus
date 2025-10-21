# Copyright (c) 2025 RealTimeChris (Chris M.)
# 
# This file is part of software offered under a restricted-use license to a designated Licensee,
# whose identity is confirmed in writing by the Author.
# 
# License Terms (Summary):
# - Exclusive, non-transferable license for internal use only.
# - Redistribution, sublicensing, or public disclosure is prohibited without written consent.
# - Full ownership remains with the Author.
# - License may terminate if unused for [X months], if materially breached, or by mutual agreement.
# - No warranty is provided, express or implied.
# 
# Full license terms are provided in the LICENSE file distributed with this software.
# 
# Signed,
# RealTimeChris (Chris M.)
# 2025
cmake_minimum_required(VERSION 3.28)

if(WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows-static CACHE STRING "" FORCE)
elseif(UNIX AND NOT APPLE)
    set(VCPKG_TARGET_TRIPLET x64-linux CACHE STRING "" FORCE)
elseif(APPLE)
    set(VCPKG_TARGET_TRIPLET x64-osx CACHE STRING "" FORCE)
endif()

if(MSVC OR NIHILUS_CUDA)
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(NIHILUS_DETECT_CPU_PROPERTIES "Override cpu-cache-size selection" OFF)
option(NIHILUS_DETECT_GPU_PROPERTIES "Override gpu-cache-size selection" OFF)
option(NIHILUS_CUDA "Enable CUDA support" OFF)
option(NIHILUS_DEV "Development mode" OFF)
option(NIHILUS_DEPLOY "Deployment mode" OFF)
option(NIHILUS_CPU "CPU build" OFF)
option(VS_LLAMA_CLEAN "Build vs-llama-clean tests" OFF)
option(VS_LLAMA_DEV "Build vs-llama-dev tests" OFF)
option(VS_LLAMA_BENCHMARK "Build vs-llama-benchmark tests" OFF)
option(VS_LLAMA_CUDA "Build vs-llama-cuda tests" OFF)

if(NIHILUS_CUDA)
    set(NIHILUS_LANGUAGE CUDA)
    project(nihilus
        VERSION 1.0.0
        DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
        LANGUAGES CXX CUDA
    )
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
    include(cmake/detection/nihilus_detect_gpu_properties.cmake)
    set(CMAKE_CUDA_ARCHITECTURES ${NIHILUS_MAJOR_COMPUTE_CAPABILITY}${NIHILUS_MINOR_COMPUTE_CAPABILITY} CACHE STRING "Cuda architectures." FORCE)
    set(NIHILUS_MAIN_FILE_EXTENSION .cu)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(NIHILUS_LANGUAGE CXX)
    project(nihilus
        VERSION 1.0.0
        DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
        LANGUAGES CXX
    )
    set(NIHILUS_MAIN_FILE_EXTENSION .cpp)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(NIHILUS_COMPILER_ID ${CMAKE_${NIHILUS_LANGUAGE}_COMPILER_ID})

include(cmake/detection/nihilus_detect_cpu_properties.cmake)

file(GLOB_RECURSE NIHILUS_HEADERS CONFIGURE_DEPENDS include/*.hpp include/*.cuh)

add_library(${PROJECT_NAME} INTERFACE)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME ${PROJECT_NAME}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        INTERPROCEDURAL_OPTIMIZATION ON
        POSITION_INDEPENDENT_CODE OFF
)

include(cmake/flags_and_options.cmake)

if(NIHILUS_DEV)
    include(FetchContent)
    
    FetchContent_Declare(Jsonifier
        GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
        GIT_TAG dev-new
        FIND_PACKAGE_ARGS NAMES Jsonifier
    )
    
    FetchContent_MakeAvailable(Jsonifier)
endif()

if(NIHILUS_DEPLOY)
    find_package(gRPC CONFIG REQUIRED)
    find_package(Protobuf CONFIG REQUIRED)
    
    string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_LOWER)
    set(NIHILUS_NETWORK_INSTALL_PATH C:/Users/Chris/source/repos/Nihilus-Network/out/install/${BUILD_TYPE_LOWER}-test)
    
    message(STATUS "Looking for nihilus-network at: ${NIHILUS_NETWORK_INSTALL_PATH}")
    
    find_package(nihilus-network CONFIG REQUIRED
        PATHS ${NIHILUS_NETWORK_INSTALL_PATH}
        NO_DEFAULT_PATH
    )
endif()

target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${NIHILUS_HEADERS}
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
        $<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>
        $<$<TARGET_EXISTS:Jsonifier::Jsonifier>:Jsonifier::Jsonifier>
        $<$<TARGET_EXISTS:nihilus-network::nihilus-network>:nihilus-network::nihilus-network>
)

target_compile_options(${PROJECT_NAME}
    INTERFACE
        $<$<CONFIG:Release>:${NIHILUS_COMPILE_OPTIONS}>
)

target_link_options(${PROJECT_NAME}
    INTERFACE
        $<$<CONFIG:Release>:${NIHILUS_LINK_OPTIONS}>
)

target_compile_definitions(${PROJECT_NAME}
    INTERFACE
        ${NIHILUS_COMPILE_DEFINITIONS}
)

message(STATUS "Nihilus Configuration Summary:")
message(STATUS "=============================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Max Thread Count: ${NIHILUS_THREAD_COUNT}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${NIHILUS_COMPILER_ID}")
message(STATUS "CPU Variants: ${NIHILUS_INSTRUCTION_SET_NAME}")

if(NIHILUS_CUDA)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(NIHILUS_DEPLOY OR NIHILUS_CPU OR NIHILUS_CUDA)
    add_subdirectory(src)
elseif(VS_LLAMA_CLEAN)
    add_subdirectory(tests/vs-llama-clean)
elseif(VS_LLAMA_DEV)
    add_subdirectory(tests/vs-llama-dev)
elseif(VS_LLAMA_BENCHMARK)
    add_subdirectory(tests/vs-llama-benchmark)
elseif(VS_LLAMA_CUDA)
    add_subdirectory(tests/vs-llama-cuda)
endif()