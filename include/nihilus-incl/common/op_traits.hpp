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

#include <nihilus-incl/common/dim_traits.hpp>
#include <nihilus-incl/common/kernel_traits.hpp>

namespace nihilus {

	enum class processing_phases {
		prompt_eval_time,
		eval_time,
	};
	template<typename config_type, typename core_traits_type, device_types devive_type, uint64_t, core_types core_type, processing_phases processing_phase>
	struct kernel_dispatcher_impl;
	template<integral_or_enum_types auto index, typename derived_type_new> struct core_elem_base {
		using derived_type = derived_type_new;
		mutable uint64_t total_required_bytes_rt{};
		using enum_type = decltype(index);

		NIHILUS_HOST constexpr decltype(auto) operator[](tag<index>) & noexcept {
			return *static_cast<derived_type*>(this);
		}
	};template<typename... value_types> struct get_first_type;

	template<typename value_type, typename... value_types> struct get_first_type<value_type, value_types...> {
		using type = value_type;
	};

	template<typename... value_types> using get_first_type_t = get_first_type<value_types...>::type;

	template<device_types device_type> constexpr allocation_strategy_types allocation_strategy_type{ [] {
		if constexpr (device_type == device_types::gpu) {
			return allocation_strategy_types::alloc;
		} else {
			return allocation_strategy_types::mmap;
		}
	}() };
}
