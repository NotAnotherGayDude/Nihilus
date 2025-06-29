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
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(ENV{CMAKE_INSTALL_PREFIX} "${CMAKE_SOURCE_DIR}/../Install")

if (NIHILUS_VS_LLAMA_DEV)
    include(FetchContent)
    FetchContent_Declare(
        Jsonifier
        GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
        GIT_TAG        dev-new
    )

    FetchContent_MakeAvailable(Jsonifier)
endif()

project(
    "${PROJECT_NAME}"
    VERSION "1.0.0"
    DESCRIPTION "Nihilus: RealTime Tensor Math - You trained the model. Nihilus gives it a voice."
    LANGUAGES CXX
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(NIHILUS_ARCH_X64 TRUE)
    set(NIHILUS_BUILD_ALL_X64_VARIANTS TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(NIHILUS_ARCH_ARM64 TRUE)
    set(NIHILUS_BUILD_ALL_ARM_VARIANTS TRUE)
endif()

add_library("${PROJECT_NAME}" INTERFACE)
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

file(GLOB_RECURSE NIHILUS_HEADERS CONFIGURE_DEPENDS 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nihilus/*.hpp"
)

set_target_properties(
    "${PROJECT_NAME}" PROPERTIES
    OUTPUT_NAME "nihilus"
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    INTERFACE_COMPILE_FEATURES cxx_std_23
)

target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

target_link_libraries(
    "${PROJECT_NAME}" INTERFACE $<$<STREQUAL:$<UPPER_CASE:${NIHILUS_VS_LLAMA_DEV}>,TRUE>:Jsonifier::Jsonifier>
)

include("cmake/nihilus_detect_architecture.cmake")

set(NIHILUS_COMMON_COMPILE_FLAGS
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fno-asynchronous-unwind-tables>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fvisibility-inlines-hidden>>"
    "$<$<BOOL:${PGO_GENERATE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-generate>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fomit-frame-pointer>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fvisibility=hidden>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fno-trapping-math>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fno-signed-zeros>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-funroll-loops>>"
    "$<$<BOOL:${PGO_USE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-use>>"
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:${NIHILUS_UBSAN_ENABLED}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-ffast-math>>"
    "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>""$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wnull-dereference>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-g0>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-O3>>"
    "$<$<CONFIG:DEBUG>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-O0>>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprefetch-loop-arrays>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-falign-loops=32>"
    "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-ftree-vectorize>"
    #"$<$<CXX_COMPILER_ID:MSVC>:$<$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>:/fsanitize=address>>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wuninitialized>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wconversion>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wpedantic>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wshadow>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wextra>"
	"$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wall>"	
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/DNDEBUG>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/fp:fast>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/Ob2>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/GL>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/Gy>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/O2>>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wnull-dereference>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wuninitialized>"
    "$<$<CXX_COMPILER_ID:MSVC>:/permissive->"
	"$<$<CXX_COMPILER_ID:GNU>:-Wconversion>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Zc:lambda>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wpedantic>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wshadow>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wextra>"
	"$<$<CXX_COMPILER_ID:MSVC>:/Wall>"
	"$<$<CXX_COMPILER_ID:GNU>:-Wall>"
	"$<$<CXX_COMPILER_ID:MSVC>:/W4>"
    "${SIMD_FLAG}"
)

set(NIHILUS_COMMON_LINK_OPTIONS
    "$<$<BOOL:${PGO_GENERATE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-generate>>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-Wl,--discard-all>>"
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>,$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-Wl,--strip-all>>"
    "$<$<BOOL:${PGO_USE}>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-fprofile-use>>"
    "$<$<AND:$<STREQUAL:$<UPPER_CASE:${NIHILUS_UBSAN_ENABLED}>,TRUE>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-fsanitize=undefined>"
    "$<$<CONFIG:RELEASE>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>>:-s>>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>>:-fsanitize=address>"
    "$<$<CONFIG:RELEASE>:$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-flto=thin>>"
    "$<$<CONFIG:RELEASE>:$<$<STREQUAL:$<UPPER_CASE:$<CXX_COMPILER_ID>>,CLANG>:-Wl,-O3>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/DEBUG:NONE>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/OPT:ICF>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/OPT:REF>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:GNU>:-Wl,-O3>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:MSVC>:/LTCG>>"
    "$<$<CONFIG:RELEASE>:$<$<CXX_COMPILER_ID:GNU>:-flto>>"
    "$<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO>"
)

target_compile_options(
    "${PROJECT_NAME}" INTERFACE
    "${NIHILUS_COMMON_COMPILE_FLAGS}"
)

target_link_options(
    "${PROJECT_NAME}" INTERFACE
    "${NIHILUS_COMMON_LINK_OPTIONS}"
)

target_compile_definitions(
    "${PROJECT_NAME}" INTERFACE 
    "$<$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>:NIHILUS_ASAN_ENABLED>"
    "$<$<STREQUAL:${NIHILUS_DEV_ENABLED},TRUE>:NIHILUS_DEBUG>"
)

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/nihilus/${EXPORTED_TARGETS_FILE_NAME}")

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/${CONFIG_FILE_NAME}.in"
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

set(NIHILUS_ALL_TARGETS "${PROJECT_NAME}")

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

message(STATUS "")
message(STATUS "Nihilus Configuration Summary:")
message(STATUS "==========================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(NIHILUS_BUILD_ALL_X64_VARIANTS)
    message(STATUS "x64 Variants: ${INSTRUCTION_SET_NAME}")
endif()
if(NIHILUS_BUILD_ALL_ARM_VARIANTS)
    message(STATUS "ARM Variants: ${INSTRUCTION_SET_NAME}")
endif()

if (NIHILUS_VS_LLAMA)
    add_subdirectory("./tests/vs-llama-clean")
elseif (NIHILUS_VS_LLAMA_DEV)
    add_subdirectory("./tests/vs-llama-dev")
elseif (NIHILUS_VS_LLAMA_BENCHMARK)
    add_subdirectory("./tests/vs-llama-benchmark")
elseif (NIHILUS_NH_ONLY)
    add_subdirectory("./tests/nihilus-only")
endif()