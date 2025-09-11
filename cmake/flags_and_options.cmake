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

set(NIHILUS_PLATFORM_DEFINITIONS
    "$<IF:$<PLATFORM_ID:Windows>,NIHILUS_PLATFORM_WINDOWS=1;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=0,>"
    "$<IF:$<PLATFORM_ID:Linux>,NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=0;NIHILUS_PLATFORM_LINUX=1,>"
    "$<IF:$<PLATFORM_ID:Darwin>,NIHILUS_PLATFORM_WINDOWS=0;NIHILUS_PLATFORM_MAC=1;NIHILUS_PLATFORM_LINUX=0,>"
)

set(NIHILUS_COMPILER_DEFINITIONS
    "$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,NIHILUS_COMPILER_CLANG=1;NIHILUS_COMPILER_MSVC=0;NIHILUS_COMPILER_GNUCXX=0,>"
    "$<IF:$<CXX_COMPILER_ID:MSVC>,NIHILUS_COMPILER_CLANG=0;NIHILUS_COMPILER_MSVC=1;NIHILUS_COMPILER_GNUCXX=0,>"
    "$<IF:$<CXX_COMPILER_ID:GNU>,NIHILUS_COMPILER_CLANG=0;NIHILUS_COMPILER_MSVC=0;NIHILUS_COMPILER_GNUCXX=1,>"
)

set(NIHILUS_INLINE_KEYWORD
    "$<IF:$<CONFIG:Release>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::forceinline]] inline,inline __attribute__((always_inline))>,$<IF:$<CXX_COMPILER_ID:MSVC>,[[msvc::noinline]],__attribute__((noinline))>>"
)

set(NIHILUS_COMMON_COMPILE_DEFINITIONS
    "NIHILUS_ARCH_X64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>,1,0>"
    "NIHILUS_ARCH_ARM64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},ARM64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm64>>,1,0>"
    "${NIHILUS_PLATFORM_DEFINITIONS}"
    "${NIHILUS_COMPILER_DEFINITIONS}"
    "NIHILUS_INLINE=${NIHILUS_INLINE_KEYWORD}"
    "$<$<STREQUAL:${NIHILUS_DEV},TRUE>:NIHILUS_DEV>"
    "$<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;WIN32_LEAN_AND_MEAN>"
    "${NIHILUS_SIMD_DEFINITIONS}"
)

set(NIHILUS_COMMON_COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:Clang>:-O3;-flto=thin;-funroll-loops;-fvectorize;-fslp-vectorize;-finline-functions;-fomit-frame-pointer;-fmerge-all-constants;-ftemplate-depth=2048;-fconstexpr-depth=2048;-fconstexpr-steps=50000000;-ftemplate-backtrace-limit=0;-ffunction-sections;-fdata-sections;-falign-functions=32;-fno-math-errno;-fno-trapping-math;-ffp-contract=fast;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-rtti;-fno-asynchronous-unwind-tables;-fno-unwind-tables;-fno-stack-protector;-fno-ident;-pipe;-fno-common;-fwrapv;-D_FORTIFY_SOURCE=0;-Weverything;-Wnon-virtual-dtor;-Wno-c++98-compat;-Wno-c++98-compat-pedantic;-Wno-unsafe-buffer-usage;-Wno-padded;-Wno-c++20-compat;-Wno-exit-time-destructors>"
    "$<$<CXX_COMPILER_ID:GNU>:-O3;-flto;-funroll-loops;-finline-functions;-fomit-frame-pointer;-fno-math-errno;-ffinite-math-only;-fno-signed-zeros;-fno-trapping-math;-ftemplate-depth=2000;-fconstexpr-depth=2000;-fconstexpr-ops-limit=100000000;-fconstexpr-loop-limit=1000000;-falign-functions=32;-falign-loops=32;-fprefetch-loop-arrays;-ftree-vectorize;-fstrict-aliasing;-ffunction-sections;-fdata-sections;-fvisibility=hidden;-fvisibility-inlines-hidden;-fno-keep-inline-functions;-fno-ident;-fmerge-all-constants;-fno-stack-protector;-fno-rtti;-fgcse-after-reload;-ftree-loop-distribute-patterns;-fpredictive-commoning;-funswitch-loops;-ftree-loop-vectorize;-ftree-slp-vectorize;-Wall;-Wextra;-Wpedantic;-Wnon-virtual-dtor;-Wlogical-op;-Wduplicated-cond;-Wduplicated-branches;-Wnull-dereference;-Wdouble-promotion>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Ob3;/Ot;/Oy;/GT;/GL;/fp:precise;/Qpar;/Qvec-report:2;/constexpr:depth2048;/constexpr:backtrace0;/constexpr:steps2000000;/GS-;/Gy;/Gw;/Zc:inline;/Zc:throwingNew;/W4;/permissive-;/Zc:__cplusplus;/wd4820;/wd4324;/Zc:alignedNew;/Zc:auto;/Zc:forScope;/Zc:implicitNoexcept;/Zc:noexceptTypes;/Zc:referenceBinding;/Zc:rvalueCast;/Zc:sizedDealloc;/Zc:strictStrings;/Zc:ternary;/Zc:wchar_t>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-Wno-nan-infinity-disabled;-fno-plt;-fno-semantic-interposition>"
    "${NIHILUS_SIMD_FLAGS}"
)

set(NIHILUS_COMMON_LINK_OPTIONS
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:-flto=thin;-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:-flto;-Wl,-dead_strip;-Wl,-x;-Wl,-S>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-flto=thin;-Wl,--gc-sections;-Wl,--strip-all;-Wl,--build-id=none;-Wl,--hash-style=gnu;-Wl,-z,now;-Wl,-z,relro>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:-flto;-Wl,--gc-sections;-Wl,--strip-all;-Wl,--as-needed;-Wl,-O3>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:/LTCG;/DYNAMICBASE:NO;/OPT:REF;/OPT:ICF;/INCREMENTAL:NO;/MACHINE:X64>"
)

set(NIHILUS_CUDA_INLINE_KEYWORD "$<IF:$<CONFIG:Release>,__forceinline__,__noinline__>")

set(NIHILUS_CUDA_COMPILE_DEFINITIONS
    "${NIHILUS_COMMON_COMPILE_DEFINITIONS}"
    "NIHILUS_CUDA_ENABLED=1"
    "NIHILUS_INLINE=${NIHILUS_CUDA_INLINE_KEYWORD}"
    "$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,NIHILUS_CUDA_HOST_CLANG=1;NIHILUS_CUDA_HOST_MSVC=0;NIHILUS_CUDA_HOST_GNUCXX=0,>"
    "$<IF:$<CXX_COMPILER_ID:MSVC>,NIHILUS_CUDA_HOST_CLANG=0;NIHILUS_CUDA_HOST_MSVC=1;NIHILUS_CUDA_HOST_GNUCXX=0,>"
    "$<IF:$<CXX_COMPILER_ID:GNU>,NIHILUS_CUDA_HOST_CLANG=0;NIHILUS_CUDA_HOST_MSVC=0;NIHILUS_CUDA_HOST_GNUCXX=1,>"
)

set(NIHILUS_CUDA_COMPILE_OPTIONS
    "$<$<CONFIG:Release>:--expt-relaxed-constexpr;--extended-lambda;--expt-extended-lambda;-O3;--use_fast_math;--restrict;--extra-device-vectorization;--ptxas-options=-O3;--ptxas-options=-v;--maxrregcount=255>"
    "$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Ob3;-Xcompiler=/Ot;-Xcompiler=/Oy;-Xcompiler=/GT;-Xcompiler=/GL;-Xcompiler=/fp:precise;-Xcompiler=/Qpar;-Xcompiler=/constexpr:depth2048;-Xcompiler=/constexpr:backtrace0;-Xcompiler=/constexpr:steps2000000;-Xcompiler=/GS-;-Xcompiler=/Gy;-Xcompiler=/Gw;-Xcompiler=/Zc:inline;-Xcompiler=/permissive->"
    "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Xcompiler=-O3;-Xcompiler=-flto=thin;-Xcompiler=-funroll-loops;-Xcompiler=-finline-functions;-Xcompiler=-fomit-frame-pointer;-Xcompiler=-fmerge-all-constants;-Xcompiler=-ftemplate-depth=2048;-Xcompiler=-fconstexpr-depth=2048;-Xcompiler=-fconstexpr-steps=50000000;-Xcompiler=-ffunction-sections;-Xcompiler=-fdata-sections;-Xcompiler=-falign-functions=32;-Xcompiler=-fno-math-errno;-Xcompiler=-fno-trapping-math;-Xcompiler=-ffp-contract=fast;-Xcompiler=-fvisibility=hidden;-Xcompiler=-fvisibility-inlines-hidden;-Xcompiler=-fno-rtti;-Xcompiler=-fno-stack-protector>"
    "$<$<CXX_COMPILER_ID:GNU>:-Xcompiler=-O3;-Xcompiler=-flto;-Xcompiler=-funroll-loops;-Xcompiler=-finline-functions;-Xcompiler=-fomit-frame-pointer;-Xcompiler=-fno-math-errno;-Xcompiler=-ffinite-math-only;-Xcompiler=-fno-signed-zeros;-Xcompiler=-fno-trapping-math;-Xcompiler=-ftemplate-depth=2000;-Xcompiler=-fconstexpr-depth=2000;-Xcompiler=-fconstexpr-ops-limit=100000000;-Xcompiler=-fconstexpr-loop-limit=1000000;-Xcompiler=-falign-functions=32;-Xcompiler=-falign-loops=32;-Xcompiler=-fprefetch-loop-arrays;-Xcompiler=-ftree-vectorize;-Xcompiler=-fstrict-aliasing;-Xcompiler=-ffunction-sections;-Xcompiler=-fdata-sections;-Xcompiler=-fvisibility=hidden;-Xcompiler=-fvisibility-inlines-hidden;-Xcompiler=-fno-keep-inline-functions;-Xcompiler=-fmerge-all-constants;-Xcompiler=-fno-stack-protector;-Xcompiler=-fno-rtti;-Xcompiler=-fgcse-after-reload;-Xcompiler=-ftree-loop-distribute-patterns;-Xcompiler=-fpredictive-commoning;-Xcompiler=-funswitch-loops;-Xcompiler=-ftree-loop-vectorize;-Xcompiler=-ftree-slp-vectorize>"
)

set(NIHILUS_CUDA_LINK_OPTIONS
    "$<$<CONFIG:Release>:-lcudart;-lcuda;-lcublas;-lcurand;-lcufft>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:-Xlinker=-flto=thin;-Xlinker=-dead_strip>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:-Xlinker=-flto;-Xlinker=-dead_strip>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-Xlinker=-flto=thin;-Xlinker=--gc-sections;-Xlinker=--strip-all>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:-Xlinker=-flto;-Xlinker=--gc-sections;-Xlinker=--strip-all;-Xlinker=--as-needed>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:-Xlinker=/LTCG;-Xlinker=/OPT:REF;-Xlinker=/OPT:ICF;-Xlinker=/INCREMENTAL:NO>"
)