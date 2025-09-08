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

set(NIHILUS_PLATFORM_WINDOWS "NIHILUS_PLATFORM_WINDOWS=1;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=0")
set(NIHILUS_PLATFORM_LINUX "NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=1")
set(NIHILUS_PLATFORM_MAC "NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=1;NIHILUS_PLATFORM_LINUX=0")
set(NIHILUS_PLATFORM_DEFS "$<IF:$<PLATFORM_ID:Windows>,${NIHILUS_PLATFORM_WINDOWS},$<IF:$<PLATFORM_ID:Linux>,${NIHILUS_PLATFORM_LINUX},${NIHILUS_PLATFORM_MAC}>>")
set(NIHILUS_COMPILER_FLAGS_CLANG "NIHILUS_COMPILER_CLANG=1;NIHILUS_COMPILER_MSVC=0;NIHILUS_COMPILER_GNUCXX=0")
set(NIHILUS_COMPILER_FLAGS_MSVC "NIHILUS_COMPILER_CLANG=0;NIHILUS_COMPILER_MSVC=1;NIHILUS_COMPILER_GNUCXX=0")
set(NIHILUS_COMPILER_FLAGS_GCC "NIHILUS_COMPILER_CLANG=0;NIHILUS_COMPILER_MSVC=0;NIHILUS_COMPILER_GNUCXX=1")
set(NIHILUS_COMPILER_DEFS "$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,${NIHILUS_COMPILER_FLAGS_CLANG},$<IF:$<CXX_COMPILER_ID:MSVC>,${NIHILUS_COMPILER_FLAGS_MSVC},${NIHILUS_COMPILER_FLAGS_GCC}>>")
set(NIHILUS_INLINE "$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],noinline>>")

set(NIHILUS_CCOMPILE_DEFINITIONS
    "GSL_UNENFORCED_ON_CONTRACT_VIOLATION"
    "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF"
    "NIHILUS_ARCH_X64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>,1,0>"
    "NIHILUS_ARCH_ARM64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},ARM64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm64>>,1,0>"
    "${NIHILUS_PLATFORM_DEFS}"
    "${NIHILUS_COMPILER_DEFS}"
    "NIHILUS_INLINE=${NIHILUS_INLINE}"
    "NIHILUS_CUDA_ENABLED=$<IF:$<NOT:$<STREQUAL:${NIHILUS_CUDA},OFF>>,1,0>"
    "$<$<NOT:$<STREQUAL:${NIHILUS_MODEL_SIZE_OVERRIDE},OFF>>:LLAMA_MODEL_SIZE=${NIHILUS_MODEL_SIZE_OVERRIDE}>"
    "$<$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>:NIHILUS_ASAN_ENABLED>"
    "$<$<STREQUAL:${NIHILUS_DEV},TRUE>:NIHILUS_DEV>"
    "$<$<CXX_COMPILER_ID:MSVC>:_SECURE_SCL=0;NOMINMAX;WIN32_LEAN_AND_MEAN>"
    "${NIHILUS_SIMD_DEFINITIONS}"
)

set(NIHILUS_COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:Clang>:-O3;-flto=thin;-funroll-loops;-fvectorize;-fslp-vectorize;-finline-functions;-fomit-frame-pointer;-fmerge-all-constants;-ftemplate-depth=2048;-fconstexpr-depth=2048;-fconstexpr-steps=50000000;"
    "-ftemplate-backtrace-limit=0;-ffunction-sections;-fdata-sections;-falign-functions=32;-fno-math-errno;-fno-trapping-math;-ffp-contract=fast;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-rtti;-fno-asynchronous-unwind-tables;"
    "-fno-unwind-tables;-fno-stack-protector;-fno-ident;-pipe;-fno-common;-fwrapv;-D_FORTIFY_SOURCE=0;-Weverything;-Wnon-virtual-dtor;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;"
    "-Wno-exit-time-destructors>"
    
    "$<$<CXX_COMPILER_ID:GNU>:-O3;-flto;-funroll-loops;-finline-functions;-fomit-frame-pointer;-fno-math-errno;-ffinite-math-only;-fno-signed-zeros;-fno-trapping-math;-ftemplate-depth=2000;-fconstexpr-depth=2000;-fconstexpr-ops-limit=100000000;"
    "-fconstexpr-loop-limit=1000000;-falign-functions=32;-falign-loops=32;-fprefetch-loop-arrays;-ftree-vectorize;-fstrict-aliasing;-ffunction-sections;-fdata-sections;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-keep-inline-functions;"
    "-fno-ident;-fmerge-all-constants;-fno-stack-protector;-fno-rtti;-fgcse-after-reload;-ftree-loop-distribute-patterns;-fpredictive-commoning;-funswitch-loops;-ftree-loop-vectorize;-ftree-slp-vectorize;-Wall;-Wextra;-Wpedantic;-Wnon-virtual-dtor;"
    "-Wlogical-op;-Wduplicated-cond;-Wduplicated-branches;-Wnull-dereference;-Wdouble-promotion>"
    
    "$<$<CXX_COMPILER_ID:MSVC>:/Ob3;/Ot;/Oy;/GT;/GL;/fp:precise;/Qpar;/Qvec-report:2;/constexpr:depth2048;/constexpr:backtrace0;/constexpr:steps2000000;/GS-;/Gy;/Gw;/Zc:inline;/Zc:throwingNew;/W4;/permissive-;/Zc:__cplusplus;/wd4820;/wd4324;/Zc:alignedNew;"
    "/Zc:auto;/Zc:forScope;/Zc:implicitNoexcept;/Zc:noexceptTypes;/Zc:referenceBinding;/Zc:rvalueCast;/Zc:sizedDealloc;/Zc:strictStrings;/Zc:ternary;/Zc:wchar_t>"

    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-Wno-nan-infinity-disabled;-fno-plt;-fno-semantic-interposition>"
    "${NIHILUS_SIMD_FLAGS}"
)

set(NIHILUS_LINK_OPTIONS
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:-flto=thin;-Wl,-dead_strip;-Wl,-x;-Wl,-S>"    
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:-flto;-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-flto=thin;-Wl,--gc-sections;-Wl,--strip-all;-Wl,--build-id=none;-Wl,--hash-style=gnu;-Wl,-z,now;-Wl,-z,relro>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:-flto;-Wl,--gc-sections;-Wl,--strip-all;-Wl,--as-needed;-Wl,-O3>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:/LTCG;/DYNAMICBASE:NO;/OPT:REF;/OPT:ICF;/INCREMENTAL:NO;/MACHINE:X64>"
)