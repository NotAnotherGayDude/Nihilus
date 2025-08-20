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

#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/core_traits.hpp>
#include <nihilus-incl/cpu/cpu_arch.hpp>

namespace nihilus {

	template<model_config config, processing_phases processing_phase, device_types dev_type, typename core_type> struct kernel_dispatcher;

	template<model_config config, processing_phases processing_phase, device_types dev_type, typename core_type> struct kernel_dispatcher {
		NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			kernel_dispatcher_impl<cpu_arch_index_holder::cpu_arch_index, core_type::core_type, processing_phase>::impl(params, thread_index, thread_count, current_block);
		}

		NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
			kernel_dispatcher_impl<cpu_arch_index_holder::cpu_arch_index, core_type::core_type, processing_phase>::impl(params, thread_index, thread_count);
		}
	};

}
