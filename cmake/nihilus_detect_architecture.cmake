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
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/BuildFeatureTesterArch.sh" "#!/bin/bash
\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNIHILUS_DETECT_ARCH=TRUE
\"${CMAKE_COMMAND}\" --build ./Build-Arch --config=Release")
    execute_process(
        COMMAND chmod +x "${CMAKE_SOURCE_DIR}/cmake/BuildFeatureTesterArch.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterArch.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/BuildFeatureTesterArch.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/Build-Arch/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/BuildFeatureTesterArch.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Arch -DCMAKE_BUILD_TYPE=Release  -DNIHILUS_DETECT_ARCH=TRUE
\"${CMAKE_COMMAND}\" --build ./Build-Arch --config=Release")
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/BuildFeatureTesterArch.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/Build-Arch/Release/feature_detector.exe")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(NIHILUS_AVX2_FLAGS "/arch:AVX2")
    set(NIHILUS_AVX512_FLAGS "/arch:AVX512")
    set(NIHILUS_NEON_FLAGS "")
    set(NIHILUS_SVE2_FLAGS "")
else()
    set(NIHILUS_AVX2_FLAGS "-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c")
    set(NIHILUS_AVX512_FLAGS "-mavx512f;-mfma;-mavx2;-mavx;-mlzcnt;-mpopcnt;-mbmi;-mbmi2;-msse4.2;-mf16c")
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
    set(NIHILUS_CPU_INSTRUCTIONS 4)
    set(NIHILUS_CPU_ALIGNMENT 64)
    set(SIMD_FLAG "${NIHILUS_SVE2_FLAGS}")
elseif(INSTRUCTION_PRESENT_AVX512)
    set(NIHILUS_CPU_INSTRUCTIONS 2)
    set(NIHILUS_CPU_ALIGNMENT 64)
    set(SIMD_FLAG "${NIHILUS_AVX512_FLAGS}")
elseif(INSTRUCTION_PRESENT_NEON)
    set(NIHILUS_CPU_ALIGNMENT 16)
    set(NIHILUS_CPU_INSTRUCTIONS 3)
    set(SIMD_FLAG "${NIHILUS_NEON_FLAGS}")
elseif(INSTRUCTION_PRESENT_AVX2)
    set(NIHILUS_CPU_ALIGNMENT 32)
    set(NIHILUS_CPU_INSTRUCTIONS 1)
    set(SIMD_FLAG "${NIHILUS_AVX2_FLAGS}")
else()
    set(NIHILUS_CPU_INSTRUCTIONS 0)
    set(NIHILUS_CPU_ALIGNMENT 8)
endif()
endif()

if(NIHILUS_CPU_INSTRUCTIONS EQUAL 4)
    set(SIMD_FLAG "${NIHILUS_SVE2_FLAGS}")
    set(INSTRUCTION_SET_NAME "SVE2")
elseif(NIHILUS_CPU_INSTRUCTIONS EQUAL 2)
    set(SIMD_FLAG "${NIHILUS_AVX512_FLAGS}")
    set(INSTRUCTION_SET_NAME "AVX512")
elseif(NIHILUS_CPU_INSTRUCTIONS EQUAL 3)
    set(SIMD_FLAG "${NIHILUS_NEON_FLAGS}")
    set(INSTRUCTION_SET_NAME "NEON")
elseif(NIHILUS_CPU_INSTRUCTIONS EQUAL 1)
    set(SIMD_FLAG "${NIHILUS_AVX2_FLAGS}")
    set(INSTRUCTION_SET_NAME "AVX2")
else()
    set(NIHILUS_CPU_INSTRUCTIONS 0)
    set(INSTRUCTION_SET_NAME "NONE")
endif()

set(SIMD_FLAG "${SIMD_FLAG}" CACHE STRING "SIMD flags" FORCE)
set(NIHILUS_CPU_INSTRUCTIONS "${NIHILUS_CPU_INSTRUCTIONS}" CACHE STRING "CPU Instruction Sets" FORCE)
set(NIHILUS_CPU_ALIGNMENT "${NIHILUS_CPU_ALIGNMENT}" CACHE STRING "CPU Alignment" FORCE)

file(WRITE "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cpu/simd/nihilus_cpu_instructions.hpp" "/*
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

#undef NIHILUS_CPU_INSTRUCTION_INDEX
#define NIHILUS_CPU_INSTRUCTION_INDEX ${NIHILUS_CPU_INSTRUCTIONS}

static constexpr uint64_t arch_alignments[]{ 8, 32, 64, 16, 64 };

static constexpr uint64_t arch_indices[]{ 0, 1, 2, 1, 2 };

#define NIHILUS_AVX2 NIHILUS_CPU_INSTRUCTION_INDEX & (1) && NIHILUS_ARCH_X64
#define NIHILUS_AVX512 NIHILUS_CPU_INSTRUCTION_INDEX & (2) && NIHILUS_ARCH_X64
#define NIHILUS_NEON NIHILUS_CPU_INSTRUCTION_INDEX & (3) && NIHILUS_ARCH_ARM64
#define NIHILUS_SVE2 NIHILUS_CPU_INSTRUCTION_INDEX & (4) && NIHILUS_ARCH_ARM64

namespace nihilus {

	struct cpu_arch_index_holder {
		static constexpr uint64_t cpu_arch_index{ arch_indices[NIHILUS_CPU_INSTRUCTION_INDEX] };
	};

	struct cpu_alignment_holder {
		static constexpr uint64_t cpu_alignment{ arch_alignments[NIHILUS_CPU_INSTRUCTION_INDEX] };
	};

}
")
