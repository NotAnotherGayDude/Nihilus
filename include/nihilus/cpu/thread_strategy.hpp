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
#include <nihilus/common/common.hpp>
#include <nihilus/common/array.hpp>
#include <nihilus/common/tuple.hpp>
#include <latch>

namespace nihilus {
	/*
	template<typename op_type_type> struct core_traits_dynamic {
		op_type_type type{};
		uint64_t depth{};
	};

	template<nihilus::model_config config_new, typename op_type_type, nihilus::thread_strategy_type thread_strategy_type> struct count_active_ops;

	template<nihilus::model_config config_new, typename op_type_type, nihilus::thread_strategy_type thread_strategy_type>
		requires(thread_strategy_type == thread_strategy_type::per_block || thread_strategy_type == thread_strategy_type::global_output||
			thread_strategy_type == thread_strategy_type::global_input)
	struct count_active_ops<config_new, op_type_type, thread_strategy_type> {
		template<op_type_type current_index = static_cast<op_type_type>(0)> constexpr void impl(uint64_t current_max_depth = 0) {
			if constexpr (static_cast<uint64_t>(current_index) < static_cast<uint64_t>(op_type_type::count)) {
				constexpr uint64_t current_index_new = static_cast<uint64_t>(current_index);
				current_max_depth += (nihilus::core_traits<config_new, current_index>::krn_type != nihilus::kernel_type::none &&
										 nihilus::core_traits<config_new, current_index>::thread_strategy_type == thread_strategy_type &&
										 nihilus::core_traits<config_new, current_index>::depth != 0)
					? 1
					: 0;
				return count_active_ops<config_new, op_type_type, thread_strategy_type, static_cast<op_type_type>(current_index_new + 1)>(current_max_depth);
			} else {
				return current_max_depth;
			}
		}
	};

	template<nihilus::model_config config_new, nihilus::thread_strategy_type thread_strategy_type, typename op_type_type, uint64_t size,
		op_type_type current_index = static_cast<op_type_type>(0)>
	constexpr auto generate_core_traits_dynamic_array(nihilus::array<core_traits_dynamic<op_type_type>, size> values, uint64_t current_index_newer = 0) {
		constexpr uint64_t current_index_new = static_cast<uint64_t>(current_index);
		if constexpr (current_index_new < static_cast<uint64_t>(op_type_type::count)) {
			if constexpr (nihilus::core_traits<config_new, current_index>::krn_type != nihilus::kernel_type::none &&
				nihilus::core_traits<config_new, current_index>::thread_strategy_type == thread_strategy_type && nihilus::core_traits<config_new, current_index>::depth != 0) {
				core_traits_dynamic<op_type_type> return_values{};
				return_values.depth			= nihilus::core_traits<config_new, current_index>::depth;
				return_values.type			= nihilus::core_traits<config_new, current_index>::type;
				values[current_index_newer] = return_values;
				++current_index_newer;
			}
			return generate_core_traits_dynamic_array<config_new, thread_strategy_type, op_type_type, size, static_cast<op_type_type>(current_index_new + 1)>(values,
				current_index_newer);
		} else {
			return values;
		}
	}

	template<model_config config_new, thread_strategy_type thread_strategy_type, typename op_type_type> constexpr auto generate_core_traits_dynamic_array() {
		constexpr uint64_t active_op_count{ count_active_ops<config_new, op_type_type, thread_strategy_type>() };
		array<core_traits_dynamic<op_type_type>, active_op_count> return_values{};
		return generate_core_traits_dynamic_array<config_new, thread_strategy_type, op_type_type>(return_values);
	}

	template<typename op_type_type, uint64_t depth_count, uint64_t array_size>
	constexpr auto count_ops_per_depth(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		if constexpr (array_size == 0) {
			return 0;
		}
		array<uint64_t, depth_count> return_values{};

		for (uint64_t i = 0; i < array_size; ++i) {
			++return_values[traits_array[i].depth];
		}
		return return_values;
	}

	template<typename op_type_type, uint64_t array_size> constexpr uint64_t count_unique_depths(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		if constexpr (array_size == 0) {
			return 0;
		}
		uint64_t max_depth{};

		for (uint64_t i = 0; i < array_size; ++i) {
			max_depth = traits_array[i].depth > max_depth ? traits_array[i].depth : max_depth;
		}
		return max_depth + 1;
	}

	template<typename op_type_type, uint64_t depth_count, uint64_t array_size>
	constexpr uint64_t count_max_ops_per_depth(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		if constexpr (array_size == 0) {
			return 0;
		}
		array<uint64_t, depth_count> array{};
		for (uint64_t i = 0; i < array_size; ++i) {
			++array[traits_array[i].depth - 1];
		}
		uint64_t max_depth_count{};

		for (uint64_t i = 0; i < depth_count; ++i) {
			max_depth_count = array[i] > max_depth_count ? array[i] : max_depth_count;
		}
		return max_depth_count;
	}

	template<model_config config_new, typename op_type_type, uint64_t max_ops_per_depth, uint64_t depth_count, uint64_t array_size>
	constexpr auto construct_thread_strategy(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		array<array<op_type_type, max_ops_per_depth>, depth_count> result{};
		array<uint64_t, depth_count> counters{};

		for (uint64_t d = 0; d < depth_count; ++d) {
			counters[d] = 0;
			for (uint64_t op = 0; op < max_ops_per_depth; ++op) {
				result[d][op] = op_type_type::count;
			}
		}

		for (uint64_t i = 0; i < traits_array.size(); ++i) {
			uint64_t depth_index = traits_array[i].depth;

			if (depth_index < depth_count && counters[depth_index] < max_ops_per_depth) {
				result[depth_index][counters[depth_index]] = traits_array[i].type;
				++counters[depth_index];
			}
		}

		return result;
	}

	template<model_config config_new, typename op_type_type, thread_strategy_type thread_strategy_type> struct thread_strategy_pre;

	template<model_config config_new, typename op_type_type> struct thread_strategy_pre<config_new, op_type_type, thread_strategy_type::global_input> {
		static constexpr auto dynamic_core_traits_array{ generate_core_traits_dynamic_array<config_new, thread_strategy_type::global_input, op_type_type>() };
		static constexpr uint64_t unique_depth_count{ count_unique_depths<op_type_type>(dynamic_core_traits_array) };
		static constexpr auto max_ops_per_depth{ count_max_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto actual_ops_per_depth{ count_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto final_ops{ construct_thread_strategy<config_new, op_type_type, max_ops_per_depth, unique_depth_count>(dynamic_core_traits_array) };
	};

	template<model_config config_new, typename op_type_type> struct thread_strategy_pre<config_new, op_type_type, thread_strategy_type::per_block> {
		static constexpr auto dynamic_core_traits_array{ generate_core_traits_dynamic_array<config_new, thread_strategy_type::per_block, op_type_type>() };
		static constexpr uint64_t unique_depth_count{ count_unique_depths<op_type_type>(dynamic_core_traits_array) };
		static constexpr auto max_ops_per_depth{ count_max_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto actual_ops_per_depth{ count_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto final_ops{ construct_thread_strategy<config_new, op_type_type, max_ops_per_depth, unique_depth_count>(dynamic_core_traits_array) };
	};

	template<model_config config_new, typename op_type_type> struct thread_strategy_pre<config_new, op_type_type, thread_strategy_type::global_output> {
		static constexpr auto dynamic_core_traits_array{ generate_core_traits_dynamic_array<config_new, thread_strategy_type::global_output, op_type_type>() };
		static constexpr uint64_t unique_depth_count{ count_unique_depths<op_type_type>(dynamic_core_traits_array) };
		static constexpr auto actual_ops_per_depth{ count_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto max_ops_per_depth{ count_max_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto final_ops{ construct_thread_strategy<config_new, op_type_type, max_ops_per_depth, unique_depth_count>(dynamic_core_traits_array) };
	};

	template<typename model_type, uint64_t current_depth, typename... bases> struct thread_core_bases : public bases... {
		NIHILUS_FORCE_INLINE thread_core_bases() noexcept							= default;
		NIHILUS_FORCE_INLINE thread_core_bases& operator=(thread_core_bases&&)		= delete;
		NIHILUS_FORCE_INLINE thread_core_bases(thread_core_bases&&)					= delete;
		NIHILUS_FORCE_INLINE thread_core_bases& operator=(const thread_core_bases&) = delete;
		NIHILUS_FORCE_INLINE thread_core_bases(const thread_core_bases&)			= delete;
		template<template<typename, auto> typename mixin_type, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE constexpr void impl_internal(arg_types&&... args) {
			return static_cast<mixin_type<op_entity_type, op_entity_type::type>*>(static_cast<op_entity_type*>(this))->impl(std::forward<arg_types>(args)...);
		}

		template<template<typename, auto> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE constexpr void impl(arg_types&&... args) {
			(impl_internal<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<typename, auto> typename mixin_type, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE static constexpr void impl_internal_static(arg_types&&... args) {
			return mixin_type<op_entity_type, op_entity_type::type>::impl(std::forward<arg_types>(args)...);
		}

		template<template<typename, auto> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE static constexpr void impl_static(arg_types&&... args) {
			(impl_internal_static<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}
	};

	template<typename model_type, typename... bases> struct thread_depth_bases : public bases... {
		NIHILUS_FORCE_INLINE thread_depth_bases() noexcept							  = default;
		NIHILUS_FORCE_INLINE thread_depth_bases& operator=(thread_depth_bases&&)	  = delete;
		NIHILUS_FORCE_INLINE thread_depth_bases(thread_depth_bases&&)				  = delete;
		NIHILUS_FORCE_INLINE thread_depth_bases& operator=(const thread_depth_bases&) = delete;
		NIHILUS_FORCE_INLINE thread_depth_bases(const thread_depth_bases&)			  = delete;
		template<template<typename, auto> typename mixin_type, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE constexpr void impl_internal(arg_types&&... args) {
			return static_cast<op_entity_type*>(this)->template impl<mixin_type>(std::forward<arg_types>(args)...);
		}

		template<template<typename, auto> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE constexpr void impl(arg_types&&... args) {
			(impl_internal<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}
	};

	template<nihilus::model_config config, typename model_type, typename thread_strategy_type, uint64_t current_depth, typename index_sequence> struct get_depth_level_core_bases;

	template<nihilus::model_config config, typename model_type, typename thread_strategy_type, uint64_t current_depth, uint64_t... index>
	struct get_depth_level_core_bases<config, model_type, thread_strategy_type, current_depth, std::index_sequence<index...>> {
		using type = thread_core_bases<model_type, current_depth, nihilus::core_traits<config, thread_strategy_type::final_ops[current_depth][index]>...>;
	};

	template<nihilus::model_config config, typename model_type, typename thread_strategy_type, uint64_t current_depth> using get_depth_level_core_bases_t =
		typename get_depth_level_core_bases<config, model_type, thread_strategy_type, current_depth,
			std::make_index_sequence<thread_strategy_type::actual_ops_per_depth[current_depth]>>::type;

	template<nihilus::model_config config, typename model_type, typename thread_strategy_type, typename index_sequence> struct get_thread_strategy_core_bases;

	template<nihilus::model_config config, typename model_type, typename thread_strategy_type, uint64_t... index>
	struct get_thread_strategy_core_bases<config, model_type, thread_strategy_type, std::index_sequence<index...>> {
		using type = thread_depth_bases<model_type, get_depth_level_core_bases_t<config, model_type, thread_strategy_type, index>...>;
	};

	template<nihilus::model_config config, typename model_type, typename op_type_type, nihilus::thread_strategy_type thread_strategy_type>
	using get_thread_strategy_core_bases_t =
		typename get_thread_strategy_core_bases<config, model_type, op_type_type, std::make_index_sequence<thread_strategy_type::unique_depth_count>>::type;

	template<nihilus::model_config config, typename model_type> struct thread_strategy
		: public nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_input>,
		  public nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::per_block>,
		  public nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_output>,
		  public get_thread_strategy_core_bases_t<config, model_type,
			  nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::none>,
			  nihilus::thread_strategy_type::none>,
		  public get_thread_strategy_core_bases_t<config, model_type,
			  nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_input>,
			  nihilus::thread_strategy_type::global_input>,
		  public get_thread_strategy_core_bases_t<config, model_type,
			  nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::per_block>,
			  nihilus::thread_strategy_type::per_block>,
		  public get_thread_strategy_core_bases_t<config, model_type,
			  nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_output>,
			  nihilus::thread_strategy_type::global_output> {
		using global_input_type	 = nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_input>;
		using per_block_type	 = nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::per_block>;
		using global_output_type = nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_output>;

		template<template<typename, auto> typename mixin_type, nihilus::thread_strategy_type thread_strategy_type, typename... arg_types>
		NIHILUS_FORCE_INLINE constexpr void impl(uint64_t thread_index, uint64_t thread_count) {
			if constexpr (thread_strategy_type == nihilus::thread_strategy_type::global_input) {
				get_thread_strategy_core_bases_t<config, model_type,
					nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_input>,
					nihilus::thread_strategy_type::global_input>::template impl<mixin_type>(thread_index, thread_count);
			} else if constexpr (thread_strategy_type == nihilus::thread_strategy_type::per_block) {
				for (uint64_t x = 0; x < model_type::model_traits_type::block_count; ++x) {
					get_thread_strategy_core_bases_t<config, model_type,
						nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::per_block>,
						nihilus::thread_strategy_type::per_block>::template impl<mixin_type>(thread_index, thread_count);
				}
			} else if constexpr (thread_strategy_type == nihilus::thread_strategy_type::global_output) {
				get_thread_strategy_core_bases_t<config, model_type,
					nihilus::thread_strategy_pre<config, typename nihilus::model_traits_type<config>::op_type_type, nihilus::thread_strategy_type::global_output>,
					nihilus::thread_strategy_type::global_output>::template impl<mixin_type>(thread_index, thread_count);
			}
		}
	};*/
}
