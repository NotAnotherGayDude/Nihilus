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

if(UNIX OR APPLE)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCpuProperties.sh" "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Cpu-Properties -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\"${CMAKE_CXX_COMPILER}\" -DNIHILUS_DETECT_CPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Cpu-Properties --config=Release"
    )
    execute_process(
        COMMAND "chmod" "+x" "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCpuProperties.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT "${CHMOD_RESULT}" EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterCpuProperties.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCpuProperties.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Cpu-Properties/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCpuProperties.bat" 
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Cpu-Properties -DNIHILUS_DETECT_CPU_PROPERTIES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Cpu-Properties --config=Release"
    )
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCpuProperties.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Cpu-Properties/Release/feature_detector.exe")
endif()

if(NOT DEFINED NIHILUS_THREAD_COUNT OR 
   NOT DEFINED NIHILUS_CPU_L1_CACHE_SIZE OR 
   NOT DEFINED NIHILUS_INSTRUCTION_SET OR
   NOT NIHILUS_DETECT_CPU_PROPERTIES)

    execute_process(
        COMMAND "${FEATURE_TESTER_FILE}"
        RESULT_VARIABLE FEATURE_TESTER_EXIT_CODE
        OUTPUT_VARIABLE CPU_PROPERTIES_OUTPUT
        ERROR_VARIABLE FEATURE_TESTER_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

message(STATUS "CPU detector exit code: ${FEATURE_TESTER_EXIT_CODE}")
message(STATUS "CPU detector output: '${CPU_PROPERTIES_OUTPUT}'")
message(STATUS "CPU detector error: '${FEATURE_TESTER_ERROR}'")

if("${FEATURE_TESTER_EXIT_CODE}" EQUAL 0 AND "${CPU_PROPERTIES_OUTPUT}" MATCHES "CPU_SUCCESS=1")
    string(REGEX MATCH "THREAD_COUNT=([0-9]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_THREAD_COUNT)
        set(NIHILUS_THREAD_COUNT "${CMAKE_MATCH_1}" CACHE STRING "CPU thread count" FORCE)
    endif()
    
    string(REGEX MATCH "INSTRUCTION_SET=([0-9]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_INSTRUCTION_SET)
        set(NIHILUS_INSTRUCTION_SET "${CMAKE_MATCH_1}" CACHE STRING "CPU instruction set bitmask" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_AVX2=([0-1]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_HAS_AVX2)
        set(NIHILUS_HAS_AVX2 "${CMAKE_MATCH_1}" CACHE STRING "CPU has AVX2 support" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_AVX512=([0-1]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_HAS_AVX512)
        set(NIHILUS_HAS_AVX512 "${CMAKE_MATCH_1}" CACHE STRING "CPU has AVX512 support" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_NEON=([0-1]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_HAS_NEON)
        set(NIHILUS_HAS_NEON "${CMAKE_MATCH_1}" CACHE STRING "CPU has NEON support" FORCE)
    endif()
    
    string(REGEX MATCH "HAS_SVE2=([0-1]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_HAS_SVE2)
        set(NIHILUS_HAS_SVE2 "${CMAKE_MATCH_1}" CACHE STRING "CPU has SVE2 support" FORCE)
    endif()
    
    string(REGEX MATCH "L1_CACHE_SIZE=([0-9]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_CPU_L1_CACHE_SIZE)
        set(NIHILUS_CPU_L1_CACHE_SIZE "${CMAKE_MATCH_1}" CACHE STRING "CPU L1 cache size" FORCE)
    endif()
    
    string(REGEX MATCH "L2_CACHE_SIZE=([0-9]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_CPU_L2_CACHE_SIZE)
        set(NIHILUS_CPU_L2_CACHE_SIZE "${CMAKE_MATCH_1}" CACHE STRING "CPU L2 cache size" FORCE)
    endif()
    
    string(REGEX MATCH "L3_CACHE_SIZE=([0-9]+)" _ "${CPU_PROPERTIES_OUTPUT}")
    if(NOT DEFINED NIHILUS_CPU_L3_CACHE_SIZE)
        set(NIHILUS_CPU_L3_CACHE_SIZE "${CMAKE_MATCH_1}" CACHE STRING "CPU L3 cache size" FORCE)
    endif()
    
    if(NOT DEFINED NIHILUS_CPU_PROPERTIES_ERECTED)
        set(NIHILUS_CPU_PROPERTIES_ERECTED TRUE CACHE BOOL "CPU properties successfully detected" FORCE)
    endif()
    message(STATUS "CPU Properties detected successfully")
else()
    message(WARNING "CPU feature detector failed, using reasonable default values for unset properties")
    
    if(NOT DEFINED NIHILUS_THREAD_COUNT)
        set(NIHILUS_THREAD_COUNT "4" CACHE STRING "CPU thread count (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_INSTRUCTION_SET)
        set(NIHILUS_INSTRUCTION_SET "0" CACHE STRING "CPU instruction set bitmask (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_HAS_AVX2)
        set(NIHILUS_HAS_AVX2 "0" CACHE STRING "CPU has AVX2 support (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_HAS_AVX512)
        set(NIHILUS_HAS_AVX512 "0" CACHE STRING "CPU has AVX512 support (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_HAS_NEON)
        set(NIHILUS_HAS_NEON "0" CACHE STRING "CPU has NEON support (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_HAS_SVE2)
        set(NIHILUS_HAS_SVE2 "0" CACHE STRING "CPU has SVE2 support (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_CPU_L1_CACHE_SIZE)
        set(NIHILUS_CPU_L1_CACHE_SIZE "32768" CACHE STRING "CPU L1 cache size - 32KB (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_CPU_L2_CACHE_SIZE)
        set(NIHILUS_CPU_L2_CACHE_SIZE "262144" CACHE STRING "CPU L2 cache size - 256KB (fallback)" FORCE)
    endif()
    if(NOT DEFINED NIHILUS_CPU_L3_CACHE_SIZE)
        set(NIHILUS_CPU_L3_CACHE_SIZE "8388608" CACHE STRING "CPU L3 cache size - 8MB (fallback)" FORCE)
    endif()
endif()

if(NOT DEFINED NIHILUS_CPU_ARCH_INDEX)
    if("${NIHILUS_HAS_AVX512}")
        set(NIHILUS_CPU_ARCH_INDEX "2" CACHE STRING "CPU architecture index - AVX512" FORCE)
        set(NIHILUS_CPU_ALIGNMENT "64" CACHE STRING "CPU Alignment" FORCE)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX512,-mavx512f;-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c>>" CACHE STRING "SIMD flags" FORCE)
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=1;NIHILUS_AVX2=0;NIHILUS_NEON=0" CACHE STRING "SIMD definitions" FORCE)
        set(NIHILUS_INSTRUCTION_SET_NAME "AVX512" CACHE STRING "Instruction set name" FORCE)
    elseif("${NIHILUS_HAS_AVX2}")
        set(NIHILUS_CPU_ARCH_INDEX "1" CACHE STRING "CPU architecture index - AVX2" FORCE)
        set(NIHILUS_CPU_ALIGNMENT "32" CACHE STRING "CPU Alignment" FORCE)        
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mavx2;-mfma;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c>>" CACHE STRING "SIMD flags" FORCE)
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=0;NIHILUS_AVX2=1;NIHILUS_NEON=0" CACHE STRING "SIMD definitions" FORCE)
        set(NIHILUS_INSTRUCTION_SET_NAME "AVX2" CACHE STRING "Instruction set name" FORCE)
    elseif("${NIHILUS_HAS_SVE2}")
        set(NIHILUS_CPU_ARCH_INDEX "2" CACHE STRING "CPU architecture index - SVE2" FORCE)
        set(NIHILUS_CPU_ALIGNMENT "64" CACHE STRING "CPU Alignment" FORCE)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,,-march=armv8-a+sve;-msve-vector-bits=scalable;-march=armv8-a+sve+sve2>>" CACHE STRING "SIMD flags" FORCE)
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=1;NIHILUS_AVX512=0;NIHILUS_AVX2=0;NIHILUS_NEON=0" CACHE STRING "SIMD definitions" FORCE)
        set(NIHILUS_INSTRUCTION_SET_NAME "SVE2" CACHE STRING "Instruction set name" FORCE)
    elseif("${NIHILUS_HAS_NEON}")
        set(NIHILUS_CPU_ARCH_INDEX "1" CACHE STRING "CPU architecture index - NEON" FORCE)
        set(NIHILUS_CPU_ALIGNMENT "16" CACHE STRING "CPU Alignment" FORCE)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CUDA_COMPILER_ID:NVIDIA>,,$<IF:$<CXX_COMPILER_ID:MSVC>,,-march=armv8-a>>" CACHE STRING "SIMD flags" FORCE)
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=0;NIHILUS_AVX2=0;NIHILUS_NEON=1" CACHE STRING "SIMD definitions" FORCE)
        set(NIHILUS_INSTRUCTION_SET_NAME "NEON" CACHE STRING "Instruction set name" FORCE)
    else()
        set(NIHILUS_CPU_ARCH_INDEX "0" CACHE STRING "CPU architecture index - fallback" FORCE)
        set(NIHILUS_CPU_ALIGNMENT "8" CACHE STRING "CPU Alignment" FORCE)
        set(NIHILUS_SIMD_FLAGS "" CACHE STRING "SIMD flags" FORCE)
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=0;NIHILUS_AVX2=0;NIHILUS_NEON=0" CACHE STRING "SIMD definitions" FORCE)
        set(NIHILUS_INSTRUCTION_SET_NAME "NONE" CACHE STRING "Instruction set name" FORCE)
    endif()
endif()

message(STATUS "CPU Configuration: ${NIHILUS_THREAD_COUNT} threads, L1: ${NIHILUS_CPU_L1_CACHE_SIZE}B, arch index: ${NIHILUS_CPU_ARCH_INDEX}")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_cpu_properties.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cpu/nihilus_cpu_properties.hpp"
    @ONLY
)