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

	template<typename base_type> struct memory_planner_construction : public base_type {
		NIHILUS_FORCE_INLINE memory_planner_construction() noexcept												 = default;
		NIHILUS_FORCE_INLINE memory_planner_construction& operator=(const memory_planner_construction&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_planner_construction(const memory_planner_construction&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE memory_planner_construction& operator=(memory_planner_construction&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE memory_planner_construction(memory_planner_construction&&) noexcept				 = delete;
		using op_type_type																						 = typename base_type::model_traits_type::op_type_type;
		using model_traits_type = typename base_type::model_traits_type;
		template<typename core_traits_type> static constexpr uint64_t get_multiplier() {
			if constexpr (core_traits_type::alc_type == alloc_type::per_block_alloc) {
				return model_traits_type::block_count;
			} else if constexpr (core_traits_type::alc_type == alloc_type::single_alloc) {
				return 1;
			} else {
				return 0;
			}
		}

		NIHILUS_FORCE_INLINE constexpr static void impl(uint64_t& current_size) {
			using core_traits_type = base_type;
			using output_type	   = core_traits_type::output_type;
			if constexpr (core_traits_type::alc_type == alloc_type::per_block_alloc) {
				++current_size;
			} else if constexpr (core_traits_type::alc_type == alloc_type::single_alloc) {
				++current_size;
			} else {
				return;
			}
		}

		template<uint64_t size> NIHILUS_FORCE_INLINE constexpr static void impl(array<op_type_type, size>& value, uint64_t& current_index) {
			if constexpr (base_type::alc_type == alloc_type::per_block_alloc) {
				value[current_index] = base_type::type;
				++current_index;
			}
		}
	};	

	template<typename base_type> struct memory_planner : public base_type {
		NIHILUS_FORCE_INLINE memory_planner() noexcept								 = default;
		NIHILUS_FORCE_INLINE memory_planner& operator=(const memory_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_planner(const memory_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE memory_planner& operator=(memory_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE memory_planner(memory_planner&&) noexcept				 = delete;
		using output_type															 = base_type::output_type;
		template<typename memory_buffer_type> NIHILUS_FORCE_INLINE void impl(memory_buffer_type& memory_buffer) {
			if constexpr (base_type::total_required_bytes > 0) {
				output_type* ptr = static_cast<output_type*>(memory_buffer.claim_memory(base_type::total_required_bytes));
				if constexpr (array_type<decltype(this->data)>) {
					for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
						this->data[x] = ptr;
					}
				} else {
					this->data = ptr;
				}
			}
		}
	};

	template<model_config config, typename derived_type> struct memory_strategy {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		using base_type			= derived_type;
		using op_type_type		= model_traits_type::op_type_type;

		static constexpr uint64_t memory_count{ [] {
			uint64_t return_value{};
			get_constexpr_core_bases_t<config>::template impl<memory_planner_construction>(return_value);
			return return_value;
		}() };

		static constexpr auto active_types{ [] {
			uint64_t current_index{};
			array<op_type_type, memory_count> return_value{};
			get_constexpr_core_bases_t<config>::template impl<memory_planner_construction>(return_value, current_index);
			return return_value;
		}() };

		template<template<typename> typename thread_function, typename... arg_types> NIHILUS_FORCE_INLINE void impl(arg_types&&... args) {}
	};

}
