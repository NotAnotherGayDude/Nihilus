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

option(NIHILUS_MAX_THREAD_COUNT "Override detected thread count" OFF)

option(NIHILUS_CPU_INSTRUCTIONS "Override cpu-arch selection" OFF)

option(NIHILUS_CPU_CACHE_SIZES "Override cpu-cache-sizes selection" OFF)

option(NIHILUS_GPU_INSTRUCTIONS "Override gpu-arch selection" OFF)

option(NIHILUS_MODEL_SIZE_OVERRIDE "For manually setting which model size gets used in 'tests'" OFF)

option(NIHILUS_CUDA "Enable CUDA support" OFF)

if (NIHILUS_CUDA)
    project(
        "${PROJECT_NAME}"
        VERSION "1.0.0"
        DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
        LANGUAGES CXX CUDA
    )
else()
    project(
        "${PROJECT_NAME}"
        VERSION "1.0.0"
        DESCRIPTION "Nihilus: You trained the model. Nihilus gives it a voice."
        LANGUAGES CXX
    )
endif()

set(NIHILUS_PLATFORM_WINDOWS "NIHILUS_PLATFORM_WINDOWS=1;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=0")
set(NIHILUS_PLATFORM_LINUX "NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=1")
set(NIHILUS_PLATFORM_MAC "NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=1;NIHILUS_PLATFORM_LINUX=0")
set(NIHILUS_PLATFORM_DEFS "$<IF:$<PLATFORM_ID:Windows>,${NIHILUS_PLATFORM_WINDOWS},$<IF:$<PLATFORM_ID:Linux>,${NIHILUS_PLATFORM_LINUX},${NIHILUS_PLATFORM_MAC}>>")
set(NIHILUS_COMPILER_CLANG_FLAGS "NIHILUS_COMPILER_CLANG=1;NIHILUS_COMPILER_MSVC=0;NIHILUS_COMPILER_GNUCXX=0")
set(NIHILUS_COMPILER_MSVC_FLAGS "NIHILUS_COMPILER_CLANG=0;NIHILUS_COMPILER_MSVC=1;NIHILUS_COMPILER_GNUCXX=0")
set(NIHILUS_COMPILER_GNUCXX_FLAGS "NIHILUS_COMPILER_CLANG=0;NIHILUS_COMPILER_MSVC=0;NIHILUS_COMPILER_GNUCXX=1")
set(NIHILUS_COMPILER_DEFS "$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,${NIHILUS_COMPILER_CLANG_FLAGS},$<IF:$<CXX_COMPILER_ID:MSVC>,${NIHILUS_COMPILER_MSVC_FLAGS},${NIHILUS_COMPILER_GNUCXX_FLAGS}>>")
set(NIHILUS_INLINE "$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],noinline>>")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(NIHILUS_ARCH_DEFS "NIHILUS_ARCH_X64=1;NIHILUS_ARCH_ARM64=0")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")  
    set(NIHILUS_ARCH_DEFS "NIHILUS_ARCH_ARM64=1;NIHILUS_ARCH_X64=0")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(NIHILUS_CLANG_COMPILE_OPTIONS
    "-O3;-ftrivial-auto-var-init=pattern;-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize;-Rpass-analysis=loop-vectorize;-flto=thin"
    "-funroll-loops;-fslp-vectorize;-fforce-enable-int128;-ftemplate-backtrace-limit=0;-fconstexpr-backtrace-limit=0;-fconstexpr-depth=2048;-ftemplate-depth=2048;-fconstexpr-steps=2000000"
    "-fno-stack-protector;-fomit-frame-pointer;-fno-plt;-fno-semantic-interposition;-fno-common;-fmerge-all-constants;-fno-ident;-fno-use-cxa-atexit;-fvisibility=hidden;-fvisibility-inlines-hidden"
    "-fdata-sections;-ffunction-sections;-Wall;-Wextra;-Wpedantic;-Weverything;-Wconversion;-Wsign-conversion;-Wfloat-conversion;-Wnull-dereference;-Wuninitialized"
    "-Wdouble-promotion;-Wformat=2;-Wformat-security;-Walloca;-Warray-bounds;-Wcast-align;-Wcast-qual;-Wctor-dtor-privacy;-Wdate-time;-Wextra-semi"
    "-Wimplicit-fallthrough;-Winvalid-pch;-Wmissing-declarations;-Wmissing-include-dirs;-Wnon-virtual-dtor;-Wold-style-cast;-Woverloaded-virtual;-Wredundant-decls;-Wsign-promo"
    "-Wswitch-default;-Wswitch-enum;-Wundef;-Wunused-macros;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;-Wno-exit-time-destructors"
)

if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    set(NIHILUS_CLANG_COMPILE_OPTIONS "${NIHILUS_CLANG_COMPILE_OPTIONS};-Wno-nan-infinity-disabled")
endif()

set(NIHILUS_GCC_COMPILE_OPTIONS
    "-O3;-Ofast;-flto=auto;-funroll-loops;-ftree-vectorize;-ftree-slp-vectorize;-fgcse-after-reload;-fipa-cp-clone;-floop-interchange;-floop-unroll-and-jam;-fpeel-loops"
    "-fpredictive-commoning;-ftrivial-auto-var-init=pattern;-freorder-blocks-and-partition;-fipa-ra;-fsplit-paths;-ftree-loop-distribute-patterns;-ftree-loop-distribution;-ftree-partial-pre;-funswitch-loops;-fversion-loops-for-strides"
    "-fgraphite-identity;-floop-nest-optimize;-ftemplate-backtrace-limit=0;-fconstexpr-loop-limit=1000000;-fconstexpr-ops-limit=2000000;-ftemplate-depth=2048;-fno-stack-protector;-fomit-frame-pointer"
    "-fno-plt;-fno-semantic-interposition;-fno-common;-fmerge-all-constants;-fdevirtualize-at-ltrans;-fipa-pta;-fno-ident;-fvisibility=hidden;-fvisibility-inlines-hidden;-fdata-sections;-ffunction-sections"
    "-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;-Wfloat-conversion;-Wnull-dereference;-Wuninitialized;-Wmaybe-uninitialized;-Wduplicated-cond;-Wduplicated-branches;-Wlogical-op"
    "-Wrestrict;-Wdouble-promotion;-Wformat=2;-Wformat-security;-Wformat-overflow=2;-Wformat-truncation=2;-Wstringop-overflow=4;-Walloc-zero;-Walloca;-Warray-bounds=2;-Wattribute-alias=2;-Wcast-align=strict"
    "-Wcast-qual;-Wcatch-value=3;-Wconditionally-supported;-Wctor-dtor-privacy;-Wdate-time;-Wextra-semi;-Wimplicit-fallthrough=5;-Winvalid-pch;-Wmissing-declarations;-Wmissing-include-dirs;-Wnoexcept;-Wnon-virtual-dtor"
    "-Wold-style-cast;-Woverloaded-virtual;-Wredundant-decls;-Wsign-promo;-Wstrict-null-sentinel;-Wstrict-overflow=5;-Wsuggest-attribute=cold;-Wsuggest-attribute=const;-Wsuggest-attribute=format;-Wsuggest-attribute=malloc"
    "-Wsuggest-attribute=noreturn;-Wsuggest-attribute=pure;-Wsuggest-final-methods;-Wsuggest-final-types;-Wsuggest-override;-Wswitch-default;-Wswitch-enum;-Wsync-nand;-Wtrampolines;-Wundef;-Wunused-const-variable=2"
    "-Wunused-macros;-Wuseless-cast;-Wzero-as-null-pointer-constant;-Wplacement-new=2;-Wvla;-Weffc++"
)

set(NIHILUS_MSVC_COMPILE_OPTIONS
    "/Ob3;/Ot;/Oy;/GT;/GL;/fp:precise;/Qpar;/Qvec-report:2;/constexpr:depth2048;/constexpr:backtrace0;/constexpr:steps2000000;/GS-;/Gy;/Gw;/Zc:inline;/Zc:throwingNew;/W4;/Wall;/permissive-;/Zc:__cplusplus;/wd4820;/wd4324"
    "/Zc:alignedNew;/Zc:auto;/Zc:forScope;/Zc:implicitNoexcept;/Zc:noexceptTypes;/Zc:referenceBinding;/Zc:rvalueCast;/Zc:sizedDealloc;/Zc:strictStrings;/Zc:ternary;/Zc:wchar_t"
)

set(NIHILUS_LINUX_LINK_OPTIONS "-Wl,--gc-sections;-Wl,--strip-debug;-Wl,--build-id=none")
set(NIHILUS_MACOS_LINK_OPTIONS "-Wl,-dead_strip")
set(NIHILUS_WINDOWS_LINK_OPTIONS "/LTCG;/DYNAMICBASE:NO;/OPT:REF;/OPT:ICF;/INCREMENTAL:NO;/MACHINE:X64;/DEBUG:NONE")

if(NIHILUS_CUDA)
    set(NIHILUS_CUDA_ENABLED_DEF "1")
else()
    set(NIHILUS_CUDA_ENABLED_DEF "0")
endif()

set(NIHILUS_COMMON_COMPILE_DEFINITIONS 
    "GSL_UNENFORCED_ON_CONTRACT_VIOLATION"
    "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF"
    "${NIHILUS_ARCH_DEFS}"
    "${NIHILUS_PLATFORM_DEFS}"
    "${NIHILUS_COMPILER_DEFS}"
    "NIHILUS_INLINE=${NIHILUS_INLINE}"
    "NIHILUS_CUDA_ENABLED=${NIHILUS_CUDA_ENABLED_DEF}"
    "$<$<NOT:$<STREQUAL:${NIHILUS_MODEL_SIZE_OVERRIDE},OFF>>:LLAMA_MODEL_SIZE=${NIHILUS_MODEL_SIZE_OVERRIDE}>"
    "$<$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>:NIHILUS_ASAN_ENABLED>"
    "$<$<STREQUAL:${NIHILUS_DEV},TRUE>:NIHILUS_DEV>"
    "$<$<STREQUAL:${NIHILUS_BENCHMARK},TRUE>:NIHILUS_BENCHMARK>"
)
set(NIHILUS_MSVC_COMPILE_DEFINITIONS "_SECURE_SCL=0;NOMINMAX;WIN32_LEAN_AND_MEAN")

if(NIHILUS_CUDA)
    add_subdirectory("cmake/cuda")
endif()

if (NIHILUS_DEV)
    include(FetchContent)
    FetchContent_Declare(
        "Jsonifier"
        GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
        GIT_TAG        dev-new
    )

    FetchContent_MakeAvailable(Jsonifier)
endif()

add_library("${PROJECT_NAME}" INTERFACE)
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

target_link_libraries("${PROJECT_NAME}" INTERFACE
    "$<$<TARGET_EXISTS:${PROJECT_NAME}-cuda>:${PROJECT_NAME}-cuda>"
    "$<$<TARGET_EXISTS:Jsonifier::Jsonifier>:Jsonifier::Jsonifier>"
)

file(GLOB_RECURSE NIHILUS_HEADERS CONFIGURE_DEPENDS 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*"
)

set_target_properties(
    "${PROJECT_NAME}" PROPERTIES
    OUTPUT_NAME "nihilus"
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    INTERFACE_COMPILE_FEATURES cxx_std_23
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

target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

include("cmake/detection/nihilus_detect_max_thread_count.cmake")
include("cmake/detection/nihilus_detect_cpu_architecture.cmake")
include("cmake/detection/nihilus_detect_cpu_cache_sizes.cmake")

target_compile_options("${PROJECT_NAME}" INTERFACE
    "${NIHILUS_SIMD_FLAGS}"
    "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:${NIHILUS_CLANG_COMPILE_OPTIONS}>"
    "$<$<CXX_COMPILER_ID:GNU>:${NIHILUS_GCC_COMPILE_OPTIONS}>"
    "$<$<CXX_COMPILER_ID:MSVC>:${NIHILUS_MSVC_COMPILE_OPTIONS}>"
)

target_link_options(${PROJECT_NAME} INTERFACE
    "$<$<PLATFORM_ID:Linux>:${NIHILUS_LINUX_LINK_OPTIONS}>"
    "$<$<PLATFORM_ID:Darwin>:${NIHILUS_MACOS_LINK_OPTIONS}>"
    "$<$<PLATFORM_ID:Windows>:${NIHILUS_WINDOWS_LINK_OPTIONS}>"
)

target_compile_definitions("${PROJECT_NAME}" INTERFACE
    "${NIHILUS_COMMON_COMPILE_DEFINITIONS}"
    "${NIHILUS_MSVC_COMPILE_DEFINITIONS}"
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

if (NIHILUS_CUDA)
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

if (NIHILUS_CLEAN)
    add_subdirectory("./tests/vs-llama-clean")
elseif (NIHILUS_DEV)
    add_subdirectory("./tests/vs-llama-dev")
elseif (NIHILUS_BENCHMARK)
    add_subdirectory("./tests/vs-llama-benchmark")
elseif (NIHILUS_NH_ONLY)
    add_subdirectory("./tests/nihilus-only")
elseif (NIHILUS_CUDA)
    add_subdirectory("./tests/nihilus-cuda")
endif()