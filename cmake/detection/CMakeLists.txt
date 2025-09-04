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
# */

cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NIHILUS_DETECT_GPU_ARCH)
    project(
        feature_detector 
        VERSION "1.0.0"
        LANGUAGES CXX CUDA
    )
    find_package(CUDAToolkit REQUIRED)
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES native)
else()
    project(
        feature_detector 
        VERSION "1.0.0"
        LANGUAGES CXX 
    )
endif()

set(NIHILUS_PLATFORM_WINDOWS "NIHILUS_PLATFORM_WINDOWS=1;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=0")
set(NIHILUS_PLATFORM_LINUX "NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=1")
set(NIHILUS_PLATFORM_MAC "NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=1;NIHILUS_PLATFORM_LINUX=0")
set(NIHILUS_PLATFORM_DEFS "$<IF:$<PLATFORM_ID:Windows>,${NIHILUS_PLATFORM_WINDOWS},$<IF:$<PLATFORM_ID:Linux>,${NIHILUS_PLATFORM_LINUX},${NIHILUS_PLATFORM_MAC}>>")

add_executable(feature_detector main.cpp)

if(NIHILUS_DETECT_CPU_ARCH)
    target_compile_definitions(feature_detector PUBLIC "NIHILUS_DETECT_CPU_ARCH;${NIHILUS_PLATFORM_DEFS}")
elseif(NIHILUS_DETECT_GPU_ARCH)
    set_property(TARGET feature_detector PROPERTY CUDA_STANDARD 20)
    set_property(TARGET feature_detector PROPERTY CUDA_STANDARD_REQUIRED ON)    
    target_link_libraries(feature_detector PUBLIC CUDA::cudart)
    target_compile_definitions(feature_detector PUBLIC "NIHILUS_DETECT_GPU_ARCH;${NIHILUS_PLATFORM_DEFS}")
elseif(NIHILUS_DETECT_CPU_CACHE_SIZES)
    target_compile_definitions(feature_detector PUBLIC "NIHILUS_DETECT_CPU_CACHE_SIZES;${NIHILUS_PLATFORM_DEFS}")
endif()