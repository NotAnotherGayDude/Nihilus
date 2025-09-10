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
set(NIHILUS_CUDA_COMPILER_FLAGS_CLANG "NIHILUS_CUDA_HOST_CLANG=1;NIHILUS_CUDA_HOST_MSVC=0;NIHILUS_CUDA_HOST_GNUCXX=0")
set(NIHILUS_CUDA_COMPILER_FLAGS_MSVC "NIHILUS_CUDA_HOST_CLANG=0;NIHILUS_CUDA_HOST_MSVC=1;NIHILUS_CUDA_HOST_GNUCXX=0")
set(NIHILUS_CUDA_COMPILER_FLAGS_GCC "NIHILUS_CUDA_HOST_CLANG=0;NIHILUS_CUDA_HOST_MSVC=0;NIHILUS_CUDA_HOST_GNUCXX=1")
set(NIHILUS_CUDA_COMPILER_DEFS "$<IF:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,${NIHILUS_CUDA_COMPILER_FLAGS_CLANG},$<IF:$<CXX_COMPILER_ID:MSVC>,${NIHILUS_CUDA_COMPILER_FLAGS_MSVC},${NIHILUS_CUDA_COMPILER_FLAGS_GCC}>>")
set(NIHILUS_CUDA_INLINE "$<IF:$<CONFIG:Release>,__forceinline__,__noinline__>")

set(NIHILUS_CUDA_COMPILE_DEFINITIONS
    "GSL_UNENFORCED_ON_CONTRACT_VIOLATION"
    "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF"
    "NIHILUS_ARCH_X64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},AMD64>>,1,0>"
    "NIHILUS_ARCH_ARM64=$<IF:$<OR:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},ARM64>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm64>>,1,0>"
    "${NIHILUS_PLATFORM_DEFS}"
    "${NIHILUS_CUDA_COMPILER_DEFS}"
    "NIHILUS_INLINE=${NIHILUS_CUDA_INLINE}"
    "NIHILUS_CUDA_ENABLED=1"
    "$<$<NOT:$<STREQUAL:${NIHILUS_MODEL_SIZE_OVERRIDE},OFF>>:LLAMA_MODEL_SIZE=${NIHILUS_MODEL_SIZE_OVERRIDE}>"
    "$<$<STREQUAL:${NIHILUS_ASAN_ENABLED},TRUE>:NIHILUS_ASAN_ENABLED>"
    "$<$<STREQUAL:${NIHILUS_DEV},TRUE>:NIHILUS_DEV>"
    "${NIHILUS_SIMD_DEFINITIONS}"
)

set(NIHILUS_CUDA_LINK_OPTIONS
    "$<$<CONFIG:Release>:-lcudart;-lcuda;-lcublas;-lcurand;-lcufft>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Darwin>>:-Xlinker=-flto=thin;-Xlinker=-dead_strip>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Darwin>>:-Xlinker=-flto;-Xlinker=-dead_strip>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>:-Xlinker=-flto=thin;-Xlinker=--gc-sections;-Xlinker=--strip-all>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<PLATFORM_ID:Linux>>:-Xlinker=-flto;-Xlinker=--gc-sections;-Xlinker=--strip-all;-Xlinker=--as-needed>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<PLATFORM_ID:Windows>>:-Xlinker=/LTCG;-Xlinker=/OPT:REF;-Xlinker=/OPT:ICF;-Xlinker=/INCREMENTAL:NO>"
)

set(NIHILUS_CUDA_COMPILE_OPTIONS
    "$<$<CONFIG:Release>:--expt-relaxed-constexpr;--extended-lambda;--expt-extended-lambda;-O3;--use_fast_math;--restrict;--extra-device-vectorization;--ptxas-options=-O3;--ptxas-options=-v;--maxrregcount=255>"

    "$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Ob3;-Xcompiler=/Ot;-Xcompiler=/Oy;-Xcompiler=/GT;-Xcompiler=/GL;"
    "-Xcompiler=/fp:precise;-Xcompiler=/Qpar;-Xcompiler=/constexpr:depth2048;"
    "-Xcompiler=/constexpr:backtrace0;-Xcompiler=/constexpr:steps2000000;-Xcompiler=/GS-;"
    "-Xcompiler=/Gy;-Xcompiler=/Gw;-Xcompiler=/Zc:inline;-Xcompiler=/permissive->"

    "$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Xcompiler=-O3;-Xcompiler=-flto=thin;-Xcompiler=-funroll-loops;"
    "-Xcompiler=-finline-functions;-Xcompiler=-fomit-frame-pointer;"
    "-Xcompiler=-fmerge-all-constants;-Xcompiler=-ftemplate-depth=2048;"
    "-Xcompiler=-fconstexpr-depth=2048;-Xcompiler=-fconstexpr-steps=50000000;"
    "-Xcompiler=-ffunction-sections;-Xcompiler=-fdata-sections;"
    "-Xcompiler=-falign-functions=32;-Xcompiler=-fno-math-errno;"
    "-Xcompiler=-fno-trapping-math;-Xcompiler=-ffp-contract=fast;"
    "-Xcompiler=-fvisibility=hidden;-Xcompiler=-fvisibility-inlines-hidden;"
    "-Xcompiler=-fno-rtti;-Xcompiler=-fno-stack-protector>"

    "$<$<CXX_COMPILER_ID:GNU>:-Xcompiler=-O3;-Xcompiler=-flto;-Xcompiler=-funroll-loops;"
    "-Xcompiler=-finline-functions;-Xcompiler=-fomit-frame-pointer;"
    "-Xcompiler=-fno-math-errno;-Xcompiler=-ffinite-math-only;"
    "-Xcompiler=-fno-signed-zeros;-Xcompiler=-fno-trapping-math;"
    "-Xcompiler=-ftemplate-depth=2000;-Xcompiler=-fconstexpr-depth=2000;"
    "-Xcompiler=-fconstexpr-ops-limit=100000000;-Xcompiler=-fconstexpr-loop-limit=1000000;"
    "-Xcompiler=-falign-functions=32;-Xcompiler=-falign-loops=32;"
    "-Xcompiler=-fprefetch-loop-arrays;-Xcompiler=-ftree-vectorize;"
    "-Xcompiler=-fstrict-aliasing;-Xcompiler=-ffunction-sections;"
    "-Xcompiler=-fdata-sections;-Xcompiler=-fvisibility=hidden;"
    "-Xcompiler=-fvisibility-inlines-hidden;-Xcompiler=-fno-keep-inline-functions;"
    "-Xcompiler=-fmerge-all-constants;-Xcompiler=-fno-stack-protector;"
    "-Xcompiler=-fno-rtti;-Xcompiler=-fgcse-after-reload;-Xcompiler=-ftree-loop-distribute-patterns;"
    "-Xcompiler=-fpredictive-commoning;-Xcompiler=-funswitch-loops;"
    "-Xcompiler=-ftree-loop-vectorize;-Xcompiler=-ftree-slp-vectorize>"
)