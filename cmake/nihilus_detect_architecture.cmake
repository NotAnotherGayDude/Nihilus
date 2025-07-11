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
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTesterArch.sh" "#!/bin/bash
\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNIHILUS_DETECT_ARCH=TRUE
\"${CMAKE_COMMAND}\" --build ./Build-Arch --config=Release")
    execute_process(
        COMMAND chmod +x "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTesterArch.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterArch.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTesterArch.sh"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Build-Arch/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTesterArch.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch -DCMAKE_BUILD_TYPE=Release  -DNIHILUS_DETECT_ARCH=TRUE
\"${CMAKE_COMMAND}\" --build ./Build-Arch --config=Release")
    execute_process(
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildFeatureTesterArch.bat"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Build-Arch/Release/feature_detector.exe")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(NIHILUS_AVX2_FLAGS "/arch:AVX2")
    set(NIHILUS_AVX512_FLAGS "/arch:AVX512")
    set(NIHILUS_NEON_FLAGS "")
    set(NIHILUS_SVE2_FLAGS "")
else()
    set(NIHILUS_AVX2_FLAGS "-mavx2;-mfma;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2")
    set(NIHILUS_AVX512_FLAGS "-mavx512f;-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2")
    set(NIHILUS_NEON_FLAGS "-mfpu=neon")
    set(NIHILUS_SVE2_FLAGS "-march=armv8-a+sve;-msve-vector-bits=scalable;-march=armv8-a+sve+sve2")
endif()

if (NOT NIHILUS_CPU_INSTRUCTIONS)

execute_process(
    COMMAND "${FEATURE_TESTER_FILE}"
    RESULT_VARIABLE NIHILUS_CPU_INSTRUCTIONS_NEW
)

set(SIMD_FLAG "")

math(EXPR NIHILUS_CPU_INSTRUCTIONS_NUMERIC "${NIHILUS_CPU_INSTRUCTIONS_NEW}")
math(EXPR NIHILUS_CPU_INSTRUCTIONS 0)
math(EXPR INSTRUCTION_PRESENT_SVE2 "(${NIHILUS_CPU_INSTRUCTIONS_NUMERIC} & 0x8)")
math(EXPR INSTRUCTION_PRESENT_AVX512 "(${NIHILUS_CPU_INSTRUCTIONS_NUMERIC} & 0x2)")
math(EXPR INSTRUCTION_PRESENT_NEON "(${NIHILUS_CPU_INSTRUCTIONS_NUMERIC} & 0x4)")
math(EXPR INSTRUCTION_PRESENT_AVX2 "(${NIHILUS_CPU_INSTRUCTIONS_NUMERIC} & 0x1)")

if(INSTRUCTION_PRESENT_SVE2)
    set(NIHILUS_CPU_INSTRUCTIONS 8)
    set(SIMD_FLAG "${NIHILUS_SVE2_FLAGS}")
elseif(INSTRUCTION_PRESENT_AVX512)
    set(NIHILUS_CPU_INSTRUCTIONS 2)
    set(SIMD_FLAG "${NIHILUS_AVX512_FLAGS}")
elseif(INSTRUCTION_PRESENT_NEON)
    set(NIHILUS_CPU_INSTRUCTIONS 4)
    set(SIMD_FLAG "${NIHILUS_NEON_FLAGS}")
elseif(INSTRUCTION_PRESENT_AVX2)
    set(NIHILUS_CPU_INSTRUCTIONS 1)
    set(SIMD_FLAG "${NIHILUS_AVX2_FLAGS}")
else()
    set(NIHILUS_CPU_INSTRUCTIONS 0)
endif()
endif()

if(NIHILUS_CPU_INSTRUCTIONS EQUAL 8)
    set(SIMD_FLAG "${NIHILUS_SVE2_FLAGS}")
    set(INSTRUCTION_SET_NAME "SVE2")
elseif(NIHILUS_CPU_INSTRUCTIONS EQUAL 2)
    set(SIMD_FLAG "${NIHILUS_AVX512_FLAGS}")
    set(INSTRUCTION_SET_NAME "AVX512")
elseif(NIHILUS_CPU_INSTRUCTIONS EQUAL 4)
    set(SIMD_FLAG "${NIHILUS_NEON_FLAGS}")
    set(INSTRUCTION_SET_NAME "NEON")
elseif(NIHILUS_CPU_INSTRUCTIONS EQUAL 1)
    set(SIMD_FLAG "${NIHILUS_AVX2_FLAGS}")
    set(INSTRUCTION_SET_NAME "AVX2")
else()
    set(NIHILUS_CPU_INSTRUCTIONS 0)
    set(INSTRUCTION_SET_NAME "NONE")
endif()

set(SIMD_FLAG "${SIMD_FLAG}" CACHE STRING "AVX flags" FORCE)
set(NIHILUS_CPU_INSTRUCTIONS "${NIHILUS_CPU_INSTRUCTIONS}" CACHE STRING "CPU Instruction Sets" FORCE)

file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/Include/nihilus/cpu/simd/nihilus_cpu_instructions.hpp" "/*
Copyright (c) 2025 RealTimeChris (Chris M.)

This file is part of software offered under a restricted-use license to a designated Licensee,
whose identity is confirmed in writing by the Author.

License Terms (Summary):
- Exclusive, non-transferable license for internal use only.
- Redistribution, sublicensing, or public disclosure is prohibited without written consent.
- Full ownership remains with the Author.
- License may terminate if unused for [X months], if materially breached, or by mutual agreement.
- No warranty is provided, express or implied.

Full license terms are provided in the LICENSE file distributed with this software.

Signed,
RealTimeChris (Chris M.)
2025
*/
#pragma once

#include <cstdint>

#undef NIHILUS_CPU_INSTRUCTIONS
#define NIHILUS_CPU_INSTRUCTIONS ${NIHILUS_CPU_INSTRUCTIONS}

#define NIHILUS_AVX2_BIT (1 << 0)
#define NIHILUS_AVX512_BIT (1 << 1)
#define NIHILUS_NEON_BIT (1 << 2)
#define NIHILUS_SVE2_BIT (1 << 3)

#if NIHILUS_CPU_INSTRUCTIONS & NIHILUS_AVX2_BIT
	#define NIHILUS_AVX2
static constexpr uint64_t cpu_arch_index{ 1 };
static constexpr uint64_t cpu_alignment{ 32 };
#elif NIHILUS_CPU_INSTRUCTIONS & NIHILUS_AVX512_BIT
	#define NIHILUS_AVX512
static constexpr uint64_t cpu_arch_index{ 2 };
static constexpr uint64_t cpu_alignment{ 64 };
#elif NIHILUS_CPU_INSTRUCTIONS & NIHILUS_NEON_BIT
	#define NIHILUS_NEON
static constexpr uint64_t cpu_arch_index{ 1 };
static constexpr uint64_t cpu_alignment{ 16 };
#elif NIHILUS_CPU_INSTRUCTIONS & NIHILUS_SVE2_BIT
	#define NIHILUS_SVE2
static constexpr uint64_t cpu_arch_index{ 2 };
static constexpr uint64_t cpu_alignment{ 64 };
#else
static constexpr uint64_t cpu_arch_index{ 0 };
static constexpr uint64_t cpu_alignment{ 8 };
#endif
")
