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

#include <cstdint>

#undef NIHILUS_CPU_INSTRUCTION_INDEX
#define NIHILUS_CPU_INSTRUCTION_INDEX 1

static constexpr uint64_t arch_alignments[]{ 8, 32, 64, 16, 64 };

static constexpr uint64_t arch_indices[]{ 0, 1, 2, 1, 2 };

#define NIHILUS_AVX2 NIHILUS_CPU_INSTRUCTION_INDEX & (1) && NIHILUS_ARCH_X64
#define NIHILUS_AVX512 NIHILUS_CPU_INSTRUCTION_INDEX & (2) && NIHILUS_ARCH_X64
#define NIHILUS_NEON NIHILUS_CPU_INSTRUCTION_INDEX & (3) && NIHILUS_ARCH_ARM64
#define NIHILUS_SVE2 NIHILUS_CPU_INSTRUCTION_INDEX & (4) && NIHILUS_ARCH_ARM64

namespace nihilus {

	struct cpu_arch_index_holder {
		static constexpr uint64_t cpu_arch_index{ arch_indices[NIHILUS_CPU_INSTRUCTION_INDEX] };
	};

	struct cpu_alignment_holder {
		static constexpr uint64_t cpu_alignment{ arch_alignments[NIHILUS_CPU_INSTRUCTION_INDEX] };
	};

}
