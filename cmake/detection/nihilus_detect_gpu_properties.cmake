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
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterGpuProperties.sh" "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Gpu-Properties -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNIHILUS_DETECT_GPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Gpu-Properties --config=Release"
    )
    execute_process(
        COMMAND chmod +x "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterGpuProperties.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterGpuProperties.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterGpuProperties.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Gpu-Properties/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterGpuProperties.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Gpu-Properties -DCMAKE_BUILD_TYPE=Release  -DNIHILUS_DETECT_GPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Gpu-Properties --config=Release"
    )
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterGpuProperties.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Gpu-Properties/Release/feature_detector.exe")
endif()
    
if(NOT DEFINED NIHILUS_SM_COUNT OR 
   NOT DEFINED NIHILUS_MAX_THREADS_PER_SM OR 
   NOT DEFINED NIHILUS_L2_CACHE_SIZE OR
   NOT NIHILUS_DETECT_GPU_PROPERTIES)

    execute_process(
        COMMAND "${FEATURE_TESTER_FILE}"
        RESULT_VARIABLE FEATURE_TESTER_EXIT_CODE
        OUTPUT_VARIABLE GPU_PROPERTIES_OUTPUT
        ERROR_VARIABLE FEATURE_TESTER_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
    
if(FEATURE_TESTER_EXIT_CODE EQUAL 0 AND GPU_PROPERTIES_OUTPUT MATCHES "GPU_SUCCESS=1")
    string(REGEX MATCH "SM_COUNT=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_SM_COUNT)
        set(NIHILUS_SM_COUNT "${CMAKE_MATCH_1}" CACHE STRING "GPU SM count" FORCE)
    endif()
        
    string(REGEX MATCH "MAX_THREADS_PER_SM=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MAX_THREADS_PER_SM)
        set(NIHILUS_MAX_THREADS_PER_SM "${CMAKE_MATCH_1}" CACHE STRING "GPU max threads per SM" FORCE)
    endif()
        
    string(REGEX MATCH "MAX_THREADS_PER_BLOCK=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MAX_THREADS_PER_BLOCK)
        set(NIHILUS_MAX_THREADS_PER_BLOCK "${CMAKE_MATCH_1}" CACHE STRING "GPU max threads per block" FORCE)
    endif()
        
    string(REGEX MATCH "WARP_SIZE=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_WARP_SIZE)
        set(NIHILUS_WARP_SIZE "${CMAKE_MATCH_1}" CACHE STRING "GPU warp size" FORCE)
    endif()
        
    string(REGEX MATCH "L2_CACHE_SIZE=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_L2_CACHE_SIZE)
        set(NIHILUS_L2_CACHE_SIZE "${CMAKE_MATCH_1}" CACHE STRING "GPU L2 cache size" FORCE)
    endif()
        
    string(REGEX MATCH "SHARED_MEM_PER_BLOCK=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_SHARED_MEM_PER_BLOCK)
        set(NIHILUS_SHARED_MEM_PER_BLOCK "${CMAKE_MATCH_1}" CACHE STRING "GPU shared memory per block" FORCE)
    endif()
        
    string(REGEX MATCH "MEMORY_BUS_WIDTH=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MEMORY_BUS_WIDTH)
        set(NIHILUS_MEMORY_BUS_WIDTH "${CMAKE_MATCH_1}" CACHE STRING "GPU memory bus width" FORCE)
    endif()
        
    string(REGEX MATCH "MEMORY_CLOCK_RATE=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MEMORY_CLOCK_RATE)
        set(NIHILUS_MEMORY_CLOCK_RATE "${CMAKE_MATCH_1}" CACHE STRING "GPU memory clock rate" FORCE)
    endif()
        
    string(REGEX MATCH "MAJOR_COMPUTE_CAPABILITY=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MAJOR_COMPUTE_CAPABILITY)
        set(NIHILUS_MAJOR_COMPUTE_CAPABILITY "${CMAKE_MATCH_1}" CACHE STRING "GPU major compute capability" FORCE)
    endif()
        
    string(REGEX MATCH "MINOR_COMPUTE_CAPABILITY=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MINOR_COMPUTE_CAPABILITY)
        set(NIHILUS_MINOR_COMPUTE_CAPABILITY "${CMAKE_MATCH_1}" CACHE STRING "GPU minor compute capability" FORCE)
    endif()
        
    string(REGEX MATCH "MAX_GRID_SIZE_X=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_MAX_GRID_SIZE_X)
        set(NIHILUS_MAX_GRID_SIZE_X "${CMAKE_MATCH_1}" CACHE STRING "GPU max grid size X" FORCE)
    endif()
        
    string(REGEX MATCH "GPU_ARCH_INDEX=([0-9]+)" _ "${GPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_GPU_ARCH_INDEX)
        set(NIHILUS_GPU_ARCH_INDEX "${CMAKE_MATCH_1}" CACHE STRING "GPU architecture index" FORCE)
    endif()
        
    if(NOT DEFINED NIHILUS_GPU_PROPERTIES_ERECTED)
        set(NIHILUS_GPU_PROPERTIES_ERECTED TRUE CACHE BOOL "GPU properties successfully detected" FORCE)
    endif()
    message(STATUS "GPU Properties detected successfully")
else()
    message(WARNING "GPU feature detector failed, using reasonable default values for unset properties")

    if(NOT DEFINED NIHILUS_SM_COUNT)
        set(NIHILUS_SM_COUNT "16" CACHE STRING "GPU SM count (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MAX_THREADS_PER_SM)
        set(NIHILUS_MAX_THREADS_PER_SM "1024" CACHE STRING "GPU max threads per SM (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MAX_THREADS_PER_BLOCK)
        set(NIHILUS_MAX_THREADS_PER_BLOCK "1024" CACHE STRING "GPU max threads per block (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_WARP_SIZE)
        set(NIHILUS_WARP_SIZE "32" CACHE STRING "GPU warp size (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_L2_CACHE_SIZE)
        set(NIHILUS_L2_CACHE_SIZE "2097152" CACHE STRING "GPU L2 cache size (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_SHARED_MEM_PER_BLOCK)
        set(NIHILUS_SHARED_MEM_PER_BLOCK "49152" CACHE STRING "GPU shared memory per block (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MEMORY_BUS_WIDTH)
        set(NIHILUS_MEMORY_BUS_WIDTH "256" CACHE STRING "GPU memory bus width (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MEMORY_CLOCK_RATE)
        set(NIHILUS_MEMORY_CLOCK_RATE "6000000" CACHE STRING "GPU memory clock rate (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MAJOR_COMPUTE_CAPABILITY)
        set(NIHILUS_MAJOR_COMPUTE_CAPABILITY "7" CACHE STRING "GPU major compute capability (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MINOR_COMPUTE_CAPABILITY)
        set(NIHILUS_MINOR_COMPUTE_CAPABILITY "0" CACHE STRING "GPU minor compute capability (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_MAX_GRID_SIZE_X)
        set(NIHILUS_MAX_GRID_SIZE_X "2147483647" CACHE STRING "GPU max grid size X (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_GPU_ARCH_INDEX)
        set(NIHILUS_GPU_ARCH_INDEX "0" CACHE STRING "GPU architecture index (fallback)" FORCE)
    endif()
endif()

if(NOT DEFINED NIHILUS_TOTAL_THREADS)
    math(EXPR NIHILUS_TOTAL_THREADS "${NIHILUS_SM_COUNT} * ${NIHILUS_MAX_THREADS_PER_SM}")
    set(NIHILUS_TOTAL_THREADS "${NIHILUS_TOTAL_THREADS}" CACHE STRING "GPU total concurrent threads" FORCE)
endif()

if(NOT DEFINED NIHILUS_OPTIMAL_BLOCK_SIZE)
    set(NIHILUS_OPTIMAL_BLOCK_SIZE "512" CACHE STRING "GPU optimal block size" FORCE)
endif()

if(NOT DEFINED NIHILUS_OPTIMAL_GRID_SIZE)
    math(EXPR NIHILUS_OPTIMAL_GRID_SIZE "${NIHILUS_TOTAL_THREADS} / ${NIHILUS_OPTIMAL_BLOCK_SIZE}")
    set(NIHILUS_OPTIMAL_GRID_SIZE "${NIHILUS_OPTIMAL_GRID_SIZE}" CACHE STRING "GPU optimal grid size" FORCE)
endif()

message(STATUS "GPU Configuration: ${NIHILUS_SM_COUNT} SMs, ${NIHILUS_TOTAL_THREADS} total threads, ${NIHILUS_OPTIMAL_GRID_SIZE} optimal grid size")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_gpu_properties.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cuda/nihilus_gpu_properties.hpp"
    @ONLY
)