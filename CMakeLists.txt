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
# RealTimeChris (Chris M.)
# 2025
# */

cmake_minimum_required(VERSION 3.28)

set(PROJECT_NAME "nihilus")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(ENV{CMAKE_INSTALL_PREFIX} "${CMAKE_SOURCE_DIR}/../Install")

if (NIHILUS_DEV)
    include(FetchContent)
    FetchContent_Declare(
        Jsonifier
        GIT_REPOSITORY https://github.com/realtimechris/jsonifier.git
        GIT_TAG        dev-new
    )

    FetchContent_MakeAvailable(Jsonifier)
endif()

option(NIHILUS_MAX_THREAD_COUNT "Override detected thread count" OFF)

option(NIHILUS_CPU_INSTRUCTIONS "Override cpu-arch selection" OFF)

option(NIHILUS_MODEL_SIZE_OVERRIDE "For manually setting which model size gets used in 'tests'" OFF)

project(
    "${PROJECT_NAME}"
    VERSION "1.0.0"
    DESCRIPTION "Nihilus: RealTime Tensor Math - You trained the model. Nihilus gives it a voice."
    LANGUAGES CXX
)

if (BUILDING_FOR_RUST)
    set(NIHILUS_PRIVATE_DEFINITION "PRIVATE")
    set(NIHILUS_PUBLIC_DEFINITION "PUBLIC")
    set(NIHILUS_STATIC_SOURCES "${CMAKE_SOURCE_DIR}/src/dllmain.cpp")
    set(NIHILUS_STATIC_DEFINITION "STATIC")
else()
    set(NIHILUS_PRIVATE_DEFINITION "INTERFACE")
    set(NIHILUS_PUBLIC_DEFINITION "INTERFACE")
    set(NIHILUS_STATIC_SOURCES "")
    set(NIHILUS_STATIC_DEFINITION "INTERFACE")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(NIHILUS_ARCH_X64 TRUE)
    set(NIHILUS_BUILD_ALL_X64_VARIANTS TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(NIHILUS_ARCH_ARM64 TRUE)
    set(NIHILUS_BUILD_ALL_ARM_VARIANTS TRUE)
endif()

add_library("${PROJECT_NAME}" "${NIHILUS_STATIC_DEFINITION}" "${NIHILUS_STATIC_SOURCES}")
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

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
    "${PROJECT_NAME}" "${NIHILUS_PUBLIC_DEFINITION}"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:include/>"
)

target_link_libraries(
    "${PROJECT_NAME}" "${NIHILUS_PRIVATE_DEFINITION}" $<$<STREQUAL:$<UPPER_CASE:${NIHILUS_DEV}>,TRUE>:Jsonifier::Jsonifier>
)

include("cmake/nihilus_detect_max_thread_count.cmake")
include("cmake/nihilus_detect_architecture.cmake")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_options(${PROJECT_NAME} "${NIHILUS_PRIVATE_DEFINITION}"
    ${AVX_FLAG}
    $<$<CXX_COMPILER_ID:Clang>:
        -O3
        -Ofast
        -ftrivial-auto-var-init=pattern
        -menable-unsafe-fp-math
        -Rpass=loop-vectorize
        -Rpass-missed=loop-vectorize
        -Rpass-analysis=loop-vectorize
        -flto=thin
        -ffast-math
        -funroll-loops
        -fslp-vectorize
        -fforce-enable-int128
        -ftemplate-backtrace-limit=0
        -fconstexpr-backtrace-limit=0
        -fconstexpr-depth=2048
        -ftemplate-depth=2048
        -fconstexpr-ops-limit=2000000
        -fno-stack-protector
        -fomit-frame-pointer
        -fno-plt
        -fno-semantic-interposition
        -fno-common
        -fmerge-all-constants        
        -fno-ident
        -fno-use-cxa-atexit
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fdata-sections
        -ffunction-sections
        -Wall
        -Wextra
        -Wpedantic
        -Weverything
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-padded
        -Wno-documentation
        -Wno-documentation-unknown-command
        -Wno-exit-time-destructors
        -Wno-global-constructors
        -Wno-weak-vtables
        -Wno-missing-prototypes
        -Wno-missing-variable-declarations
        -Wconversion
        -Wsign-conversion
        -Wfloat-conversion
        -Wnull-dereference
        -Wuninitialized
        -Wmaybe-uninitialized
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wrestrict
        -Wdouble-promotion
        -Wformat=2
        -Wformat-security
        -Wformat-overflow=2
        -Wformat-truncation=2
        -Wstringop-overflow=4
        -Walloc-zero
        -Walloca
        -Warray-bounds=2
        -Wattribute-alias=2
        -Wcast-align=strict
        -Wcast-qual
        -Wcatch-value=3
        -Wconditionally-supported
        -Wctor-dtor-privacy
        -Wdate-time
        -Wextra-semi
        -Wimplicit-fallthrough=5
        -Winvalid-pch
        -Wmissing-declarations
        -Wmissing-include-dirs
        -Wnoexcept
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wredundant-decls
        -Wsign-promo
        -Wstrict-null-sentinel
        -Wstrict-overflow=5
        -Wsuggest-attribute=cold
        -Wsuggest-attribute=const
        -Wsuggest-attribute=format
        -Wsuggest-attribute=malloc
        -Wsuggest-attribute=noreturn
        -Wsuggest-attribute=pure
        -Wsuggest-final-methods
        -Wsuggest-final-types
        -Wsuggest-override
        -Wswitch-default
        -Wswitch-enum
        -Wsync-nand
        -Wtrampolines
        -Wundef
        -Wunused-const-variable=2
        -Wunused-macros
        -Wuseless-cast
        -Wzero-as-null-pointer-constant
    >

    $<$<CXX_COMPILER_ID:GNU>:
        -O3
        -Ofast
        -march=native
        -mtune=native
        -flto=auto
        -ffast-math
        -funroll-loops
        -ftree-vectorize
        -ftree-slp-vectorize
        -fgcse-after-reload
        -fipa-cp-clone
        -floop-interchange
        -floop-unroll-and-jam
        -fpeel-loops
        -fpredictive-commoning
        -ftrivial-auto-var-init=pattern
        -freorder-blocks-and-partition
        -fipa-ra
        -fsplit-paths
        -ftree-loop-distribute-patterns
        -ftree-loop-distribution
        -ftree-partial-pre
        -funswitch-loops
        -fversion-loops-for-strides
        -fgraphite-identity
        -floop-nest-optimize        
        -ftemplate-backtrace-limit=0
        -fconstexpr-loop-limit=1000000
        -fconstexpr-ops-limit=2000000
        -ftemplate-depth=2048
        -fno-stack-protector
        -fomit-frame-pointer
        -fno-plt
        -fno-semantic-interposition
        -fno-common
        -fmerge-all-constants
        -fdevirtualize-at-ltrans
        -fipa-pta
        -fno-ident
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fdata-sections
        -ffunction-sections
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wfloat-conversion
        -Wnull-dereference
        -Wuninitialized
        -Wmaybe-uninitialized
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wrestrict
        -Wdouble-promotion
        -Wformat=2
        -Wformat-security
        -Wformat-overflow=2
        -Wformat-truncation=2
        -Wstringop-overflow=4
        -Walloc-zero
        -Walloca
        -Warray-bounds=2
        -Wattribute-alias=2
        -Wcast-align=strict
        -Wcast-qual
        -Wcatch-value=3
        -Wconditionally-supported
        -Wctor-dtor-privacy
        -Wdate-time
        -Wextra-semi
        -Wimplicit-fallthrough=5
        -Winvalid-pch
        -Wmissing-declarations
        -Wmissing-include-dirs
        -Wnoexcept
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Woverloaded-virtual
        -Wredundant-decls
        -Wsign-promo
        -Wstrict-null-sentinel
        -Wstrict-overflow=5
        -Wsuggest-attribute=cold
        -Wsuggest-attribute=const
        -Wsuggest-attribute=format
        -Wsuggest-attribute=malloc
        -Wsuggest-attribute=noreturn
        -Wsuggest-attribute=pure
        -Wsuggest-final-methods
        -Wsuggest-final-types
        -Wsuggest-override
        -Wswitch-default
        -Wswitch-enum
        -Wsync-nand
        -Wtrampolines
        -Wundef
        -Wunused-const-variable=2
        -Wunused-macros
        -Wuseless-cast
        -Wzero-as-null-pointer-constant
        -Wplacement-new=2
        -Wvla
        -Weffc++
    >

    $<$<CXX_COMPILER_ID:MSVC>:
        /Ob3
        /Ot
        /Oy
        /GT
        /GL
        /fp:fast
        /Qpar
        /Qvec-report:2
        /constexpr:depth2048
        /constexpr:backtrace0
        /constexpr:steps1000000
        /GS-
        /Gy
        /Gw
        /Zc:inline
        /Zc:throwingNew
        /W4
        /Wall
        /permissive-
        /Zc:__cplusplus
        /Zc:alignedNew
        /Zc:auto
        /Zc:forScope
        /Zc:implicitNoexcept
        /Zc:noexceptTypes
        /Zc:referenceBinding
        /Zc:rvalueCast
        /Zc:sizedDealloc
        /Zc:strictStrings
        /Zc:ternary
        /Zc:wchar_t
        /wd4820
    >
)

target_link_options(${PROJECT_NAME} "${NIHILUS_PRIVATE_DEFINITION}"
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
        $<$<CONFIG:Release>:-flto>
        -Wl,--gc-sections
        $<$<PLATFORM_ID:Linux>:
            -Wl,--strip-debug
            -Wl,--build-id=none
        >
    >

    $<$<CXX_COMPILER_ID:MSVC>:
        $<$<CONFIG:Release>:/LTCG>
        /DYNAMICBASE:NO
        /OPT:REF
        /OPT:ICF
        /INCREMENTAL:NO
        /MACHINE:X64
        $<$<CONFIG:Release>:/DEBUG:NONE>
    >
)

target_compile_definitions(${PROJECT_NAME} "${NIHILUS_PRIVATE_DEFINITION}"
    $<$<CONFIG:Release>:
        GSL_UNENFORCED_ON_CONTRACT_VIOLATION
        SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF
        $<$<CXX_COMPILER_ID:MSVC>:
            _SECURE_SCL=0
            _HAS_ITERATOR_DEBUGGING=0
            _ITERATOR_DEBUG_LEVEL=0
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        >
    >
    "$<$<NOT:$<STREQUAL:${NIHILUS_MODEL_SIZE_OVERRIDE},OFF>>:LLAMA_MODEL_SIZE=${NIHILUS_MODEL_SIZE_OVERRIDE}>"
    CONSTEXPR_EVERYTHING
    ENABLE_CONSTEXPR_MATH
)

target_compile_options(
    "${PROJECT_NAME}" "${NIHILUS_PRIVATE_DEFINITION}"
    "${NIHILUS_COMMON_COMPILE_FLAGS}"
)

target_link_options(
    "${PROJECT_NAME}" "${NIHILUS_PRIVATE_DEFINITION}"
    "${NIHILUS_COMMON_LINK_OPTIONS}"
)

target_compile_definitions(
    "${PROJECT_NAME}" "${NIHILUS_PRIVATE_DEFINITION}" 
    "$<$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>:NIHILUS_ASAN_ENABLED>"
    "$<$<STREQUAL:${NIHILUS_DEV},TRUE>:NIHILUS_DEV>"
    "$<$<STREQUAL:${NIHILUS_BENCHMARK},TRUE>:NIHILUS_BENCHMARK>"
)

set(CONFIG_FILE_NAME "${PROJECT_NAME}Config.cmake")
set(EXPORTED_TARGETS_NAME "${PROJECT_NAME}Targets")
set(EXPORTED_TARGETS_FILE_NAME "${EXPORTED_TARGETS_NAME}.cmake")
set(EXPORTED_TARGETS_FILE_PATH "share/nihilus-incl/${EXPORTED_TARGETS_FILE_NAME}")

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
message(STATUS "Max Thread Count: ${NIHILUS_MAX_THREAD_COUNT}")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
if(NIHILUS_BUILD_ALL_X64_VARIANTS)
    message(STATUS "x64 Variants: ${INSTRUCTION_SET_NAME}")
endif()
if(NIHILUS_BUILD_ALL_ARM_VARIANTS)
    message(STATUS "ARM Variants: ${INSTRUCTION_SET_NAME}")
endif()

if (NIHILUS_CLEAN)
    add_subdirectory("./tests/vs-llama-clean")
elseif (NIHILUS_DEV)
    add_subdirectory("./tests/vs-llama-dev")
elseif (NIHILUS_BENCHMARK)
    add_subdirectory("./tests/vs-llama-benchmark")
elseif (NIHILUS_NH_ONLY)
    add_subdirectory("./tests/nihilus-only")
endif()