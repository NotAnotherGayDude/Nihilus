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

find_package(CUDAToolkit REQUIRED)

include("${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_detect_gpu_properties.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/flags_and_options.cmake")

message(STATUS "CUDA Architecture: ${NIHILUS_GPU_NAME}")

file(GLOB_RECURSE NIHILUS_CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/cuda/*.cu")

add_library("${PROJECT_NAME}-cuda" STATIC "${NIHILUS_CUDA_SOURCES}")
add_library("${PROJECT_NAME}::${PROJECT_NAME}-cuda" ALIAS "${PROJECT_NAME}-cuda")

set_property(TARGET "${PROJECT_NAME}-cuda" PROPERTY CUDA_STANDARD 20)

target_compile_definitions("${PROJECT_NAME}-cuda" PUBLIC "${NIHILUS_CUDA_COMPILE_DEFINITIONS}")
target_compile_options("${PROJECT_NAME}-cuda" PUBLIC "${NIHILUS_CUDA_COMPILE_OPTIONS}")
target_link_options("${PROJECT_NAME}-cuda" PUBLIC "${NIHILUS_CUDA_LINK_OPTIONS}")

target_include_directories(
    "${PROJECT_NAME}-cuda" PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

target_link_libraries("${PROJECT_NAME}-cuda" PUBLIC
    CUDA::cudart
)
