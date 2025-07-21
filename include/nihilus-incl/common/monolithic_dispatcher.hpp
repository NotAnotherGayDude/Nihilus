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

	template<model_config config, device_types dev_type, typename core_type> struct kernel_dispatcher;

	template<model_config config, device_types dev_type, single_input_types core_type> struct kernel_dispatcher<config, dev_type, core_type> {
		NIHILUS_INLINE static void impl(core_type& params, uint64_t thread_index, uint64_t thread_count, uint64_t runtime_dim) {
			kernel_dispatcher_impl<config.cpu_arch_index, core_type::kernel_type, typename core_type::transform_type, core_type, typename core_type::output_type,
				typename core_type::input_01_type::output_type>::impl(thread_index, thread_count, params, get_adjacent_value<config, core_type::type, 0>::impl(params));
		}
	};

	template<model_config config, device_types dev_type, double_input_types core_type> struct kernel_dispatcher<config, dev_type, core_type> {
		NIHILUS_INLINE static void impl(core_type& params, uint64_t thread_index, uint64_t thread_count, uint64_t runtime_dim) {
			kernel_dispatcher_impl<config.cpu_arch_index, core_type::kernel_type, typename core_type::transform_type, core_type, typename core_type::output_type,
				typename core_type::input_01_type::output_type, typename core_type::input_02_type::output_type>::impl(thread_index, thread_count, params,
				get_adjacent_value<config, core_type::type, 0>::impl(params), get_adjacent_value<config, core_type::type, 1>::impl(params));
		}
	};

	template<model_config config, device_types dev_type, triple_input_types core_type> struct kernel_dispatcher<config, dev_type, core_type> {
		NIHILUS_INLINE static void impl(core_type& params, uint64_t thread_index, uint64_t thread_count, uint64_t runtime_dim) {
			kernel_dispatcher_impl<config.cpu_arch_index, core_type::kernel_type, typename core_type::transform_type, core_type, typename core_type::output_type,
				typename core_type::input_01_type::output_type, typename core_type::input_02_type::output_type, typename core_type::input_03_type::output_type>::impl(thread_index,
				thread_count, params, get_adjacent_value<config, core_type::type, 0>::impl(params), get_adjacent_value<config, core_type::type, 1>::impl(params),
				get_adjacent_value<config, core_type::type, 2>::impl(params));
		}
	};

}