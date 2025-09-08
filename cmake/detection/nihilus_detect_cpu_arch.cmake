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
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArch.sh" "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNIHILUS_DETECT_CPU_ARCH=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Arch --config=Release"
    )
    execute_process(
        COMMAND chmod +x "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArch.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterArch.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArch.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Arch/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArch.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch -DCMAKE_BUILD_TYPE=Release  -DNIHILUS_DETECT_CPU_ARCH=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Arch --config=Release"
    )
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterArch.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Arch/Release/feature_detector.exe")
endif()

if (NOT NIHILUS_CPU_ARCHITECTURE)

    execute_process(
        COMMAND "${FEATURE_TESTER_FILE}"
        RESULT_VARIABLE NIHILUS_CPU_ARCHITECTURE_NEW
    )

    if(NIHILUS_CPU_ARCHITECTURE_NEW EQUAL 0x8)
        set(NIHILUS_CPU_ARCHITECTURE 2)
        set(NIHILUS_CPU_ALIGNMENT 64)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CXX_COMPILER_ID:MSVC>,,-march=armv8-a+sve;-msve-vector-bits=scalable;-march=armv8-a+sve+sve2>")
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=1;NIHILUS_AVX512=0;NIHILUS_AVX2=0;NIHILUS_NEON=0")
        set(NIHILUS_INSTRUCTION_SET_NAME "SVE2")
    elseif(NIHILUS_CPU_ARCHITECTURE_NEW EQUAL 0x2)
        set(NIHILUS_CPU_ARCHITECTURE 2)
        set(NIHILUS_CPU_ALIGNMENT 64)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX512,-mavx512f;-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c>")
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=1;NIHILUS_AVX2=0;NIHILUS_NEON=0")
        set(NIHILUS_INSTRUCTION_SET_NAME "AVX512")
    elseif(NIHILUS_CPU_ARCHITECTURE_NEW EQUAL 0x4)
        set(NIHILUS_CPU_ALIGNMENT 16)
        set(NIHILUS_CPU_ARCHITECTURE 1)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CXX_COMPILER_ID:MSVC>,,-march=armv8-a>")
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=0;NIHILUS_AVX2=0;NIHILUS_NEON=1")
        set(NIHILUS_INSTRUCTION_SET_NAME "NEON")
    elseif(NIHILUS_CPU_ARCHITECTURE_NEW EQUAL 0x1)
        set(NIHILUS_CPU_ALIGNMENT 32)
        set(NIHILUS_CPU_ARCHITECTURE 1)
        set(NIHILUS_SIMD_FLAGS "$<IF:$<CXX_COMPILER_ID:MSVC>,/arch:AVX2,-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c>")
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=0;NIHILUS_AVX2=1;NIHILUS_NEON=0")
        set(NIHILUS_INSTRUCTION_SET_NAME "AVX2")
    else()
        set(NIHILUS_CPU_ARCHITECTURE 0)
        set(NIHILUS_CPU_ALIGNMENT 8)
        set(NIHILUS_SIMD_DEFINITIONS "NIHILUS_SVE2=0;NIHILUS_AVX512=0;NIHILUS_AVX2=1;NIHILUS_NEON=0")
        set(NIHILUS_INSTRUCTION_SET_NAME "NONE")
    endif()

    set(NIHILUS_SIMD_FLAGS "${NIHILUS_SIMD_FLAGS}" CACHE STRING "SIMD flags" FORCE)
    set(NIHILUS_SIMD_DEFINITIONS "${NIHILUS_SIMD_DEFINITIONS}" CACHE STRING "SIMD definitions" FORCE)
    set(NIHILUS_INSTRUCTION_SET_NAME "${NIHILUS_INSTRUCTION_SET_NAME}" CACHE STRING "Instruction set name" FORCE)
    set(NIHILUS_CPU_ARCHITECTURE "${NIHILUS_CPU_ARCHITECTURE}" CACHE STRING "CPU architecture" FORCE)
    set(NIHILUS_CPU_ARCHITECTURE_INCLUDE "${NIHILUS_CPU_ARCHITECTURE_INCLUDE}" CACHE STRING "CPU architecture include" FORCE)
    set(NIHILUS_CPU_ALIGNMENT "${NIHILUS_CPU_ALIGNMENT}" CACHE STRING "CPU Alignment" FORCE)

endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_cpu_arch.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cpu/nihilus_cpu_arch.hpp"
    @ONLY
)
