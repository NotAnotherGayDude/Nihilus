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
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTester.sh" "#!/bin/bash\n"
        "\"${CMAKE_COMMAND}\" -S ./ -B ./Build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}\n"
        "\"${CMAKE_COMMAND}\" --build ./Build --config=Release"
    )
    execute_process(
        COMMAND chmod +x "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTester.sh"
        RESULT_VARIABLE CHMOD_RESULT
    )
    if(NOT ${CHMOD_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to set executable permissions for BuildFeatureTester.sh")
    endif()
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTester.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build/feature_detector")
elseif(WIN32)
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTester.bat" "\"${CMAKE_COMMAND}\" -S ./ -B ./Build -DCMAKE_BUILD_TYPE=Release\n"
        "\"${CMAKE_COMMAND}\" --build ./Build --config=Release"
    )
    execute_process(
        COMMAND "${CMAKE_SOURCE_DIR}/cmake/detection/BuildFeatureTester.bat"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/detection"
    )
    set(FEATURE_TESTER_FILE "${CMAKE_SOURCE_DIR}/cmake/detection/Build/Release/feature_detector.exe")
endif()

if (NOT NIHILUS_MAX_THREAD_COUNT)

    execute_process(
        COMMAND "${FEATURE_TESTER_FILE}"
        RESULT_VARIABLE NIHILUS_MAX_THREAD_COUNT_NEW
    )

    if(NIHILUS_MAX_THREAD_COUNT_NEW GREATER 0)
        set(NIHILUS_MAX_THREAD_COUNT "${NIHILUS_MAX_THREAD_COUNT_NEW}" CACHE STRING "CPU max thread count" FORCE)
    else()
        message(WARNING "Feature detector failed, using default thread count of 1")
        set(NIHILUS_MAX_THREAD_COUNT "1" CACHE STRING "CPU max thread count (default fallback)" FORCE)
    endif()

endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/detection/nihilus_max_thread_count.hpp.in"
    "${CMAKE_SOURCE_DIR}/include/nihilus-incl/cpu/nihilus_max_thread_count.hpp"
    @ONLY
)
