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

option(NIHILUS_DETECT_CPU_PROPERTIES "Override cpu-cache-size selection" OFF)
option(NIHILUS_DETECT_GPU_PROPERTIES "Override gpu-cache-size selection" OFF)
option(NIHILUS_COMPILER_CUDA "Enable CUDA support" OFF)

if(NIHILUS_COMPILER_CUDA)
    set(NIHILUS_LANGUAGE "CUDA")    
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
    include("cmake/detection/nihilus_detect_gpu_properties.cmake")
    set(CMAKE_CUDA_ARCHITECTURES "${NIHILUS_MAJOR_COMPUTE_CAPABILITY}${NIHILUS_MINOR_COMPUTE_CAPABILITY}" CACHE STRING "Cuda architectures." FORCE)
    set(NIHILUS_MAIN_FILE_EXTENSION ".cu")
else()
    set(NIHILUS_LANGUAGE "CXX")
    set(NIHILUS_MAIN_FILE_EXTENSION ".cpp")
endif()

project(nihilus
    VERSION "1.0.0"
    DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
    LANGUAGES "${NIHILUS_LANGUAGE}"
)

set(CMAKE_${NIHILUS_LANGUAGE}_STANDARD 20)
set(CMAKE_${NIHILUS_LANGUAGE}_STANDARD_REQUIRED ON)

set(NIHILUS_COMPILER_ID "${CMAKE_${NIHILUS_LANGUAGE}_COMPILER_ID}")

include("cmake/detection/nihilus_detect_cpu_properties.cmake")

file(GLOB_RECURSE NIHILUS_HEADERS CONFIGURE_DEPENDS "include/*.hpp" "include/*.cuh")

add_library("${PROJECT_NAME}" INTERFACE)
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

set_target_properties("${PROJECT_NAME}" PROPERTIES
    OUTPUT_NAME "${PROJECT_NAME}"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    INTERPROCEDURAL_OPTIMIZATION ON
    POSITION_INDEPENDENT_CODE OFF
)

include("cmake/flags_and_options.cmake")

if("${NIHILUS_DEV}")
    include(FetchContent)
    
    FetchContent_Declare(Jsonifier
        GIT_REPOSITORY "https://github.com/realtimechris/jsonifier.git"
        GIT_TAG "dev-new"
        FIND_PACKAGE_ARGS NAMES Jsonifier
    )
    
    FetchContent_MakeAvailable(Jsonifier)
endif()

target_include_directories("${PROJECT_NAME}" INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_sources("${PROJECT_NAME}" INTERFACE
    FILE_SET HEADERS 
    BASE_DIRS "include"
    FILES "${NIHILUS_HEADERS}"
)

target_link_libraries(
    "${PROJECT_NAME}" INTERFACE
    "$<$<TARGET_EXISTS:CUDA::cudart>:CUDA::cudart>"
    "$<$<TARGET_EXISTS:Jsonifier::Jsonifier>:Jsonifier::Jsonifier>"
)

target_compile_options("${PROJECT_NAME}" INTERFACE
    "$<$<CONFIG:Release>:${NIHILUS_COMPILE_OPTIONS}>"
)

target_link_options("${PROJECT_NAME}" INTERFACE
    "$<$<CONFIG:Release>:${NIHILUS_LINK_OPTIONS}>"
)

target_compile_definitions("${PROJECT_NAME}" INTERFACE
    "${NIHILUS_COMPILE_DEFINITIONS}" 
)

message(STATUS "Nihilus Configuration Summary:")
message(STATUS "=============================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Max Thread Count: ${NIHILUS_THREAD_COUNT}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${NIHILUS_COMPILER_ID}")
message(STATUS "CPU Variants: ${NIHILUS_INSTRUCTION_SET_NAME}")
if("${NIHILUS_COMPILER_CUDA}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if("${NIHILUS_CPU}" OR "${NIHILUS_COMPILER_CUDA}")
    add_subdirectory("src")
elseif("${VS_LLAMA_CLEAN}")
    add_subdirectory("tests/vs-llama-clean")
elseif("${VS_LLAMA_DEV}")
    add_subdirectory("tests/vs-llama-dev")
elseif("${VS_LLAMA_BENCHMARK}")
    add_subdirectory("tests/vs-llama-benchmark")
elseif("${VS_LLAMA_CUDA}")
    add_subdirectory("tests/vs-llama-cuda")
endif()