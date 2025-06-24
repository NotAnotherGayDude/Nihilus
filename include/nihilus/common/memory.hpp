/*
Copyright (c) 2025 RealTimeChris (Chris model_traits_type.)

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
RealTimeChris (Chris model_traits_type.)
2025
*/

#pragma once

#include <nihilus/common/kernel_traits.hpp>
#include <nihilus/common/kernel_type_profile_traits.hpp>
#include <nihilus/common/model_traits.hpp>
#include <nihilus/common/core_traits.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/array.hpp>
#include <latch>

namespace nihilus {
	
	template<typename base_type, auto type> struct memory_allocator : public base_type {
		NIHILUS_FORCE_INLINE memory_allocator() noexcept								   = default;
		NIHILUS_FORCE_INLINE memory_allocator& operator=(const memory_allocator&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_allocator(const memory_allocator&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE memory_allocator& operator=(memory_allocator&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE memory_allocator(memory_allocator&&) noexcept				   = delete;
		using model_traits_type															   = base_type::model_traits_type;
		NIHILUS_FORCE_INLINE constexpr static void impl(uint64_t& total_required_bytes) {
			if constexpr (base_type::alc_type == nihilus::alloc_type::per_block_alloc) {
				total_required_bytes += nihilus::round_up_to_multiple<cpu_alignment>(base_type::total_required_bytes) * model_traits_type::block_count;
			} else if constexpr (base_type::alc_type == nihilus::alloc_type::single_alloc) {
				total_required_bytes += nihilus::round_up_to_multiple<cpu_alignment>(base_type::total_required_bytes);
			}
		}
	};

}
