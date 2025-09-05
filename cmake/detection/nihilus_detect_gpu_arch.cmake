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

if (UNIX OR APPLE)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArchCuda.sh" "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch-Cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNIHILUS_DETECT_GPU_ARCH=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Arch-Cuda --config=Release"
    )
    execute_process(
        COMMAND chmod +x "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArchCuda.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterArchCuda.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArchCuda.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Arch-Cuda/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArchCuda.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch-Cuda  -DCMAKE_BUILD_TYPE=Release  -DNIHILUS_DETECT_GPU_ARCH=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Arch-Cuda --config=Release"
    )
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArchCuda.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Arch-Cuda/Release/feature_detector.exe")
endif()

set(NIHILUS_CUDA_9_NAME "cuda_9")
set(NIHILUS_CUDA_10_NAME "cuda_10")
set(NIHILUS_CUDA_11_NAME "cuda_11")
set(NIHILUS_CUDA_12_NAME "cuda_12")

if (NOT NIHILUS_GPU_ARCHITECTURE)

    execute_process(
        COMMAND "${FEATURE_TESTER_FILE}"
        RESULT_VARIABLE NIHILUS_GPU_ARCHITECTURE_NEW
    )

    set(NIHILUS_GPU_NAME 0)

    math(EXPR NIHILUS_GPU_ARCHITECTURE_NUMERIC "${NIHILUS_GPU_ARCHITECTURE_NEW}")
    math(EXPR NIHILUS_GPU_ARCHITECTURE 0)
    math(EXPR INSTRUCTION_PRESENT_CUDA_12 "(${NIHILUS_GPU_ARCHITECTURE_NUMERIC} & 0x8)")
    math(EXPR INSTRUCTION_PRESENT_CUDA_11 "(${NIHILUS_GPU_ARCHITECTURE_NUMERIC} & 0x4)")
    math(EXPR INSTRUCTION_PRESENT_CUDA_10 "(${NIHILUS_GPU_ARCHITECTURE_NUMERIC} & 0x2)")
    math(EXPR INSTRUCTION_PRESENT_CUDA_9 "(${NIHILUS_GPU_ARCHITECTURE_NUMERIC} & 0x1)")

    if(INSTRUCTION_PRESENT_CUDA_9)
        set(NIHILUS_GPU_ARCHITECTURE 1)
        set(NIHILUS_GPU_NAME "${NIHILUS_CUDA_9_NAME}")
    elseif(INSTRUCTION_PRESENT_CUDA_10)
        set(NIHILUS_GPU_ARCHITECTURE 2)
        set(NIHILUS_GPU_NAME "${NIHILUS_CUDA_10_NAME}")
    elseif(INSTRUCTION_PRESENT_CUDA_11)
        set(NIHILUS_GPU_ARCHITECTURE 3)
        set(NIHILUS_GPU_NAME "${NIHILUS_CUDA_11_NAME}")
    elseif(INSTRUCTION_PRESENT_CUDA_12)
        set(NIHILUS_GPU_ARCHITECTURE 4)
        set(NIHILUS_GPU_NAME "${NIHILUS_CUDA_12_NAME}")
    else()
        set(NIHILUS_GPU_ARCHITECTURE 0)
    endif()

    set(NIHILUS_GPU_NAME "${NIHILUS_GPU_NAME}" CACHE STRING "GPU flags" FORCE)
    set(NIHILUS_GPU_ARCHITECTURE "${NIHILUS_GPU_ARCHITECTURE}" CACHE STRING "GPU Instruction Sets" FORCE)

endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_gpu_arch.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cuda/nihilus_gpu_arch.hpp"
    @ONLY
)
