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

set(PROJECT_NAME "nihilus")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/../Install")

option(NIHILUS_DETECT_CPU_PROPERTIES "Override cpu-cache-size selection" OFF)

option(NNIHILUS_DETECT_GPU_PROPERTIES "Override gpu-cache-size selection" OFF)

option(NIHILUS_MODEL_SIZE_OVERRIDE "For manually setting which model size gets used in 'tests'" OFF)

option(NIHILUS_CUDA_ENABLED "Enable CUDA support" OFF)

if (NIHILUS_CUDA_ENABLED)
    project(
        "${PROJECT_NAME}"
        VERSION "1.0.0"
        DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
        LANGUAGES CUDA
    )
    set(NIHILUS_MAIN_FILE_EXTENSION ".cu")
else()
    project(
        "${PROJECT_NAME}"
        VERSION "1.0.0"
        DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
        LANGUAGES CXX
    )
    set(NIHILUS_MAIN_FILE_EXTENSION ".cpp")
endif()

include("cmake/detection/nihilus_detect_cpu_properties.cmake")

add_library("${PROJECT_NAME}" INTERFACE)
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

set_target_properties(
    "${PROJECT_NAME}" PROPERTIES
    OUTPUT_NAME "nihilus"
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    INTERPROCEDURAL_OPTIMIZATION TRUE
    POSITION_INDEPENDENT_CODE OFF
    $<$<CXX_COMPILER_ID:MSVC>:
        VS_GLOBAL_EnableCppCoreCheck true
        VS_GLOBAL_CodeAnalysisRuleSet AllRules.ruleset
        VS_GLOBAL_RunCodeAnalysis true
    >
)

include("${CMAKE_SOURCE_DIR}/cmake/flags_and_options.cmake")

if (NIHILUS_DEV)
    include(FetchContent)
    FetchContent_Declare(
        "Jsonifier"
        GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
        GIT_TAG        dev-new
    )

    FetchContent_MakeAvailable(Jsonifier)
endif()

if(NIHILUS_CUDA_ENABLED)
    add_subdirectory("cmake/cuda")
endif()

target_link_libraries(
    "${PROJECT_NAME}" INTERFACE
    "$<$<TARGET_EXISTS:nihilus::nihilus-cuda>:nihilus::nihilus-cuda>"
    "$<$<TARGET_EXISTS:Jsonifier::Jsonifier>:Jsonifier::Jsonifier>"
)

target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

target_compile_options(
    "${PROJECT_NAME}" INTERFACE
    #"${NIHILUS_COMPILE_OPTIONS}"
)

target_link_options(
    "${PROJECT_NAME}" INTERFACE
    #"${NIHILUS_LINK_OPTIONS}"
)

target_compile_definitions(
    "${PROJECT_NAME}" INTERFACE
    "${NIHILUS_COMPILE_DEFINITIONS}"
)

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/nihilus-incl/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CONFIG_FILE_NAME}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
    INSTALL_DESTINATION "share/nihilus"
    PATH_VARS EXPORTED_TARGETS_FILE_PATH
)

set(VERSION_FILE_NAME "${PROJECT_NAME}ConfigVersion.cmake")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_FILE_NAME}"
    "${CMAKE_CURRENT_BINARY_DIR}/${VERSION_FILE_NAME}"
    DESTINATION "share/nihilus"
)

install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.hpp"
)

if (NIHILUS_CUDA_ENABLED)
    set(NIHILUS_ALL_TARGETS "${PROJECT_NAME};${PROJECT_NAME}-cuda")
else()
    set(NIHILUS_ALL_TARGETS "${PROJECT_NAME}")
endif()

install(
    TARGETS ${NIHILUS_ALL_TARGETS}
    EXPORT "${EXPORTED_TARGETS_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)

install(
    EXPORT "${EXPORTED_TARGETS_NAME}"
    FILE "${EXPORTED_TARGETS_FILE_NAME}"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION "share/nihilus"
)

message(STATUS "Nihilus Configuration Summary:")
message(STATUS "==========================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Max Thread Count: ${NIHILUS_MAX_THREAD_COUNT}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CPU Variants: ${INSTRUCTION_SET_NAME}")

if (VS_LLAMA_CLEAN)
    add_subdirectory("./tests/vs-llama-clean")
elseif (VS_LLAMA_DEV)
    add_subdirectory("./tests/vs-llama-dev")
elseif (VS_LLAMA_BENCHMARK)
    add_subdirectory("./tests/vs-llama-benchmark")
elseif (NIHILUS_ONLY)
    add_subdirectory("./tests/nihilus-only")
elseif (VS_LLAMA_CUDA)
    add_subdirectory("./tests/vs-llama-cuda")
elseif (NIHILUS_ONLY_CUDA)
    add_subdirectory("./tests/nihilus-only-cuda")
endif()