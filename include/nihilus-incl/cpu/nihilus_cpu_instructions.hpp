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

#include <nihilus-incl/common/config.hpp>

#define NIHILUS_AVX2_BIT 1
#define NIHILUS_AVX512_BIT 2
#define NIHILUS_NEON_BIT 3
#define NIHILUS_SVE2_BIT 4

#undef NIHILUS_CPU_INSTRUCTION_INDEX
#define NIHILUS_CPU_INSTRUCTION_INDEX 1

#ifndef NIHILUS_CHECK_CPU_INSTRUCTIONS
	#define NIHILUS_CHECK_CPU_INSTRUCTIONS(x, y) (x == y)
#endif

static constexpr uint64_t arch_indices[]{ 0, 1, 2, 1, 2 };

static constexpr uint64_t cpu_alignments[]{ 16, 32, 64, 16, 64 };

#define NIHILUS_AVX2 (NIHILUS_CHECK_CPU_INSTRUCTIONS(NIHILUS_CPU_INSTRUCTION_INDEX, NIHILUS_AVX2_BIT) && NIHILUS_ARCH_X64)
#define NIHILUS_AVX512 (NIHILUS_CHECK_CPU_INSTRUCTIONS(NIHILUS_CPU_INSTRUCTION_INDEX, NIHILUS_AVX512_BIT) && NIHILUS_ARCH_X64)
#define NIHILUS_NEON (NIHILUS_CHECK_CPU_INSTRUCTIONS(NIHILUS_CPU_INSTRUCTION_INDEX, NIHILUS_NEON_BIT) && NIHILUS_ARCH_ARM64)
#define NIHILUS_SVE2 (NIHILUS_CHECK_CPU_INSTRUCTIONS(NIHILUS_CPU_INSTRUCTION_INDEX, NIHILUS_SVE2_BIT) && NIHILUS_ARCH_ARM64)

namespace nihilus {

	struct cpu_arch_index_holder {
		static constexpr static_aligned_const cpu_arch_index_raw{ arch_indices[NIHILUS_CPU_INSTRUCTION_INDEX] };
		static constexpr const uint64_t& cpu_arch_index{ *cpu_arch_index_raw };
	};

	struct cpu_alignment_holder {
		static constexpr static_aligned_const cpu_alignment_raw{ cpu_alignments[NIHILUS_CPU_INSTRUCTION_INDEX] };
		static constexpr const uint64_t& cpu_alignment{ *cpu_alignment_raw };
	};
}
