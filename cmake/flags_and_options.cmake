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

set(NIHILUS_COMPILE_DEFINITIONS
    NIHILUS_COMPILER_CUDA=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,1,0>
    NIHILUS_ARCH_X64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>,1,0>
    NIHILUS_ARCH_ARM64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},ARM64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm64>>,1,0>
    NIHILUS_PLATFORM_WINDOWS=$<IF:$<PLATFORM_ID:Windows>,1,0>
    NIHILUS_PLATFORM_LINUX=$<IF:$<PLATFORM_ID:Linux>,1,0>
    NIHILUS_PLATFORM_MAC=$<IF:$<PLATFORM_ID:Darwin>,1,0>
    NIHILUS_COMPILER_CLANG=$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,1,0>
    NIHILUS_COMPILER_MSVC=$<IF:$<CXX_COMPILER_ID:MSVC>,1,0>
    NIHILUS_COMPILER_GNUCXX=$<IF:$<CXX_COMPILER_ID:GNU>,1,0>
    NIHILUS_DEV=$<IF:$<STREQUAL:${NIHILUS_DEV},TRUE>,1,0>
    NIHILUS_CUDA_TENSOR_CORES=$<IF:$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<VERSION_GREATER_EQUAL:${CMAKE_CUDA_COMPILER_VERSION},11.0>>,1,0>
    NIHILUS_CUDA_MAX_REGISTERS=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,128,0>
    "NIHILUS_HOST_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __host__ __device__,__noinline__ __host__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "NIHILUS_HOST=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __host__,__noinline__ __host__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "NIHILUS_DEVICE=$<IF:$<CUDA_COMPILER_ID:NVIDIA>,$<IF:$<CONFIG:Release>,__forceinline__ __device__,__noinline__ __device__>,$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>>"
    "NIHILUS_GLOBAL=__global__"
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;WIN32_LEAN_AND_MEAN>
    ${NIHILUS_SIMD_DEFINITIONS}
)

if(MSVC OR NVCC)
    string(REGEX REPLACE /Ob[0-2] "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REGEX REPLACE /O[0-2] "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
endif()

set(NIHILUS_CXX_COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:Clang>:
        -O3
        -funroll-loops
        -fvectorize
        -fslp-vectorize
        -finline-functions
        -fomit-frame-pointer
        -fmerge-all-constants
        -ffunction-sections
        -fdata-sections
        -falign-functions=32
        -fno-math-errno
        -ffp-contract=on
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fno-rtti
        -fno-asynchronous-unwind-tables
        -fno-unwind-tables
        -fno-stack-protector
        -fno-ident
        -pipe
        -fno-common
        -fwrapv
        -D_FORTIFY_SOURCE=0
        -Weverything
        -Wnon-virtual-dtor
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-unsafe-buffer-usage
        -Wno-padded
        -Wno-c++20-compat
        -Wno-exit-time-destructors
        -Werror
    >
    $<$<CXX_COMPILER_ID:AppleClang>:
        -O3
        -funroll-loops
        -fvectorize
        -fslp-vectorize
        -finline-functions
        -fomit-frame-pointer
        -fmerge-all-constants
        -ffunction-sections
        -fdata-sections
        -falign-functions=32
        -fno-math-errno
        -ffp-contract=on
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fno-rtti
        -fno-asynchronous-unwind-tables
        -fno-unwind-tables
        -fno-stack-protector
        -fno-ident
        -pipe
        -fno-common
        -fwrapv
        -D_FORTIFY_SOURCE=0
        -Weverything
        -Wnon-virtual-dtor
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-unsafe-buffer-usage
        -Wno-padded
        -Wno-c++20-compat
        -Wno-exit-time-destructors
        -Wno-poison-system-directories
        -Werror
    >
    $<$<CXX_COMPILER_ID:GNU>:
        -O3
        -funroll-loops
        -finline-functions
        -fomit-frame-pointer
        -fno-math-errno
        -falign-functions=32
        -falign-loops=32
        -fprefetch-loop-arrays
        -ftree-vectorize
        -fstrict-aliasing
        -ffunction-sections
        -fdata-sections
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fno-keep-inline-functions
        -fno-ident
        -fmerge-all-constants
        -fno-stack-protector
        -fno-rtti
        -fgcse-after-reload
        -ftree-loop-distribute-patterns
        -fpredictive-commoning
        -funswitch-loops
        -ftree-loop-vectorize
        -ftree-slp-vectorize
        -Wall
        -Wextra
        -Wpedantic
        -Wnon-virtual-dtor
        -Wlogical-op
        -Wduplicated-cond
        -Wduplicated-branches
        -Wnull-dereference
        -Wdouble-promotion
        -Werror
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /Ob3
        /Ot
        /Oy
        /GT
        /GL
        /fp:precise
        /Qpar
        /GS-
        /Gy
        /Gw
        /Zc:inline
        /Zc:throwingNew
        /W4
        /permissive-
        /Zc:__cplusplus
        /wd4820
        /wd4324
        /wd5002
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
        /WX
    >
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:
        -fno-plt
        -fno-semantic-interposition
    >
    ${NIHILUS_SIMD_FLAGS}
)

set(NIHILUS_CUDA_COMPILE_OPTIONS
    $<$<CUDA_COMPILER_ID:NVIDIA>:
        -O3
        --fmad=false
        --prec-div=true
        --prec-sqrt=true
        --restrict
        --extended-lambda
    >
    $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:
        -Xcompiler=/Ot,-Oy,-GT,-fp:precise,-Qpar,-GS-,-Gy,-Gw,-W4,-permissive-,-wd4820,-wd4324,-wd5002
    >
    $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:GNU>>:
        -Xcompiler=-O3,-funroll-loops,-finline-functions,-fomit-frame-pointer,-fno-math-errno
    >
    $<$<AND:$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:Clang>>:
        -Xcompiler=-O3,-funroll-loops,-finline-functions,-fomit-frame-pointer,-fno-math-errno
    >
)

set(NIHILUS_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:${NIHILUS_CXX_COMPILE_OPTIONS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${NIHILUS_CUDA_COMPILE_OPTIONS}>
)

set(NIHILUS_LINK_OPTIONS
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:
        -Wl,-dead_strip
        -Wl,-x
        -Wl,-S
    >
    $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<PLATFORM_ID:Darwin>>:
        -Wl,-dead_strip
        -Wl,-x
        -Wl,-S
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:
        -Wl,-dead_strip
        -Wl,-x
        -Wl,-S
    >
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:
        -Wl,--gc-sections
        -Wl,--strip-all
        -Wl,--build-id=none
        -Wl,--hash-style=gnu
        -Wl,-z,now
        -Wl,-z,relro
        -flto=thin
        -fwhole-program-vtables
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:
        -Wl,--gc-sections
        -Wl,--strip-all
        -Wl,--as-needed
        -Wl,-O3
    >
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:
        /DYNAMICBASE:NO
        /OPT:REF
        /OPT:ICF
        /INCREMENTAL:NO
        /MACHINE:X64
        /LTCG
    >
    $<$<CUDA_COMPILER_ID:NVIDIA>:
        -lcudart_static
        -lrt
        -ldl
        -lpthread
        --relocatable-device-code=false
    >
)