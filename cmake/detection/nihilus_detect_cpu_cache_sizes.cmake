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
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCacheSizes.sh" "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Cache-Sizes -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DNIHILUS_DETECT_CPU_CACHE_SIZES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Cache-Sizes --config=Release"
    )
    execute_process(
        COMMAND chmod +x "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCacheSizes.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTesterCacheSizes.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCacheSizes.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Cache-Sizes/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCacheSizes.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build-Cache-Sizes -DCMAKE_BUILD_TYPE=Release  -DNIHILUS_DETECT_CPU_CACHE_SIZES=TRUE\n"
        "\"${CMAKE_COMMAND}\" --build ./Build-Cache-Sizes --config=Release"
    )
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTesterCacheSizes.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build-Cache-Sizes/Release/feature_detector.exe")
endif()

if (NOT NIHILUS_CPU_CACHE_SIZE)

    execute_process(
        COMMAND "${FEATURE_TESTER_FILE}"
        RESULT_VARIABLE NIHILUS_CPU_CACHE_SIZE_NEW
    )

    if(NIHILUS_CPU_CACHE_SIZE_NEW GREATER 0)
        set(NIHILUS_CPU_CACHE_SIZE "${NIHILUS_CPU_CACHE_SIZE_NEW}" CACHE STRING "CPU L1 cache size " FORCE)
    else()
        message(WARNING "Feature detector failed, using default thread count of 1")
        set(NIHILUS_CPU_CACHE_SIZE "64" CACHE STRING "CPU L1 cache size (default fallback)" FORCE)
    endif()

endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_cpu_cache_sizes.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cpu/nihilus_cpu_cache_sizes.hpp"
    @ONLY
)
