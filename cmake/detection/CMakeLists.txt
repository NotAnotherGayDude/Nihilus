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

project(feature_detector
    VERSION 1.0.0
    LANGUAGES CXX
)

if(NIHILUS_DETECT_GPU_PROPERTIES)
    find_package(CUDAToolkit REQUIRED)
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES native)
endif()

add_executable(feature_detector main.cpp)

include(${CMAKE_SOURCE_DIR}/../flags_and_options.cmake)

if(NIHILUS_DETECT_CPU_PROPERTIES)
    target_compile_definitions(feature_detector
        PUBLIC
            NIHILUS_DETECT_CPU_PROPERTIES
            ${NIHILUS_COMPILE_DEFINITIONS}
    )
    
elseif(NIHILUS_DETECT_GPU_PROPERTIES)
    target_link_libraries(feature_detector
        PUBLIC
            CUDA::cudart
    )
    
    target_compile_definitions(feature_detector
        PUBLIC
            NIHILUS_DETECT_GPU_PROPERTIES
            ${NIHILUS_COMPILE_DEFINITIONS}
    )
endif()