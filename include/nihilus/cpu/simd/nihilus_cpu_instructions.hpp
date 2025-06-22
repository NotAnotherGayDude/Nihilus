/*
Copyright (c) 2025 RealTimeChris (Chris M.)

This file is part of software offered under a restricted-use license to a designated Licensee,
whose identity is confirmed in writing by the Author.

License Terms (Summary):
- Exclusive, non-transferable license for internal use only.
- Redistribution, sublicensing, or public disclosure is prohibited without written consent.
- Full ownership remains with the Author.
- License may terminate if unused for [X months], if materially breached, or by mutual agreement.
- No warranty is provided, express or implied.

Full license terms are provided in the LICENSE file distributed with this software.

Signed,
RealTimeChris (Chris M.)
2025
*/
#pragma once

#undef NIHILUS_CPU_INSTRUCTIONS
#define NIHILUS_CPU_INSTRUCTIONS 1

#define NIHILUS_AVX2_BIT (1 << 0)
#define NIHILUS_AVX512_BIT (1 << 1)
#define NIHILUS_NEON_BIT (1 << 2)
#define NIHILUS_SVE2_BIT (1 << 3)

#if NIHILUS_CPU_INSTRUCTIONS & NIHILUS_AVX2_BIT
	#define NIHILUS_AVX2
static constexpr uint64_t cpu_arch_index{ 1 };
static constexpr uint64_t cpu_alignment{ 32 };
#elif NIHILUS_CPU_INSTRUCTIONS & NIHILUS_AVX512_BIT
	#define NIHILUS_AVX512
static constexpr uint64_t cpu_arch_index{ 2 };
static constexpr uint64_t cpu_alignment{ 64 };
#elif NIHILUS_CPU_INSTRUCTIONS & NIHILUS_NEON_BIT
	#define NIHILUS_NEON
static constexpr uint64_t cpu_arch_index{ 1 };
static constexpr uint64_t cpu_alignment{ 16 };
#elif NIHILUS_CPU_INSTRUCTIONS & NIHILUS_SVE2_BIT
	#define NIHILUS_SVE2
static constexpr uint64_t cpu_arch_index{ 2 };
static constexpr uint64_t cpu_alignment{ 64 };
#else
static constexpr uint64_t cpu_arch_index{ 0 };
static constexpr uint64_t cpu_alignment{ 8 };
#endif
