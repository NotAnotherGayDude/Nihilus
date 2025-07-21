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

#include <nihilus-incl/common/behavioral_axes.hpp>
#include <latch>

namespace nihilus {

	template<model_config config, typename... bases> struct core_bases : public bases... {
		NIHILUS_INLINE core_bases()				 = default;
		core_bases& operator=(core_bases&&)		 = delete;
		core_bases(core_bases&&)				 = delete;
		core_bases& operator=(const core_bases&) = delete;
		core_bases(const core_bases&)			 = delete;
		template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) const {
			(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
		}

		template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) {
			(impl_internal_filtered<mixin_type, bases>(args...), ...);
		}

	  protected:
		template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered(arg_types&&... args) const {
			( void )(args, ...);
			if constexpr (mixin_type<config, base_type>::filter()) {
				if constexpr (has_return_type<mixin_type<config, base_type>>) {
					while (!mixin_type<config, base_type>::impl(*static_cast<const base_type*>(this), detail::forward<arg_types>(args)...)) {
					}
				} else {
					mixin_type<config, base_type>::impl(*static_cast<const base_type*>(this), detail::forward<arg_types>(args)...);
				}
			}
		}

	  protected:
		template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered(arg_types&&... args) {
			( void )(args, ...);
			if constexpr (mixin_type<config, base_type>::filter()) {
				if constexpr (has_return_type<mixin_type<config, base_type>>) {
					while (!mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), detail::forward<arg_types>(args)...)) {
					}
				} else {
					mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), detail::forward<arg_types>(args)...);
				}
			}
		}
	};

	template<model_config config, typename index_sequence> struct get_core_bases;

	template<model_config config, size_t... index> struct get_core_bases<config, std::index_sequence<index...>> {
		using type = core_bases<config, core_traits<config, static_cast<typename model_traits_type<config>::op_type_type>(index)>...>;
	};

	template<model_config config> using get_core_bases_t = typename get_core_bases<config, std::make_index_sequence<static_cast<uint64_t>(op_types::count)>>::type;

	template<model_config config_new> static constexpr get_core_bases_t<config_new> core_bases_val{};

	template<uint64_t max_depth> NIHILUS_INLINE constexpr uint64_t calculate_peak_concurrent_memory(const array<depth_and_bytes, op_types::count>& depths) {
		array<depth_and_bytes, op_types::count> depths_new{ depths };
		array<uint64_t, max_depth> depth_byte_counts{};

		for (uint64_t x = 0; x < max_depth; ++x) {
			for (uint64_t y = 0; y < depths.size(); ++y) {
				depth_byte_counts[x] += depths_new[y].required_bytes;
			}
		}

		uint64_t max_size{};
		for (uint64_t x = 0; x < depth_byte_counts.size(); ++x) {
			if (max_size < depth_byte_counts[x]) {
				max_size = depth_byte_counts[x];
			}
		}
		return max_size;
	}

	constexpr std::size_t alignment = cpu_alignment;
	constexpr std::size_t align_up(std::size_t x) {
		return (x + (alignment - 1)) & ~(alignment - 1);
	}

	constexpr bool overlaps(int64_t a_start, int64_t a_end, int64_t b_start, int64_t b_end) {
		return !(a_end < b_start || b_end < a_start);
	}

	constexpr memory_plan compute_offsets(array<depth_and_bytes, op_types::count> tensors, std::size_t max_size_guess ) {
		memory_plan result{ max_size_guess };
		std::size_t alloc_count	   = 0;
		std::size_t current_offset = 0;

		for (uint64_t x = 0; x < tensors.size(); ++x) {
			result.offsets[x].offset = current_offset;
			current_offset		  += tensors[x].required_bytes;
		}

		result.memory_total = current_offset;
		return result;
	}

	template<model_config config_new> struct core_bases_traits_type {
		using model_traits_type = model_traits_type<config_new>;
		static constexpr uint64_t max_depth{ []() {
			uint64_t return_value{};
			core_bases_val<config_new>.template impl<max_depth_calculator>(return_value);
			return return_value;
		}() };

		static constexpr uint64_t allocation_count{ []() {
			uint64_t return_value{};
			core_bases_val<config_new>.template impl<memory_planner>(return_value);
			return return_value;
		}() };
		static constexpr auto depths{ []() {
			array<depth_and_bytes, op_types::count> return_value{};
			depth_and_bytes fill_value{};
			fill_value.last_used_depth	= max_depth;
			fill_value.first_used_depth = -1;
			return_value.fill(fill_value);
			std::sort(return_value.begin(), return_value.end(), std::less<depth_and_bytes>{});
			core_bases_val<config_new>.template impl<memory_planner_depths>(return_value);
			return return_value;
		}() };
		static constexpr uint64_t max_size{ []() {
			return calculate_peak_concurrent_memory<max_depth>(depths);
		}() };
		static constexpr auto memory_plan_val{ []() {
			return compute_offsets(depths, max_size);
		}() };
		static constexpr array<uint64_t, max_depth + 1> ops_per_depth{ []() {
			array<uint64_t, max_depth + 1> return_value{};
			core_bases_val<config_new>.template impl<ops_per_depth_calculator>(return_value);
			return return_value;
		}() };
	};
}