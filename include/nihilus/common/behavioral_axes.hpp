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

#include <nihilus/common/monolithic_dispatcher.hpp>
#include <nihilus/common/behavioral_axes.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/tuple.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	using namespace std::chrono_literals;
	inline std::atomic_uint64_t depths_global{};
	inline std::atomic_uint64_t depths{};
	inline std::atomic_uint64_t count{};

	NIHILUS_FORCE_INLINE void spinlock_nanoseconds(uint64_t nanoseconds) {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		auto start = std::chrono::high_resolution_clock::now();
		auto end   = std::chrono::high_resolution_clock::now();
		do {
			end = std::chrono::high_resolution_clock::now();
		} while ((end - start).count() < nanoseconds);
#else
		// Linux/Unix implementation
		auto start	= std::chrono::high_resolution_clock::now();
		auto target = start + std::chrono::nanoseconds(nanoseconds);
		do {
		} while (std::chrono::high_resolution_clock::now() < target);
#endif
	}

	int64_t current_iteration{};

	template<nihilus::model_config config, typename base_type_new> struct execution_planner {
		NIHILUS_FORCE_INLINE execution_planner() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_planner(const execution_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_planner& operator=(execution_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_planner(execution_planner&&) noexcept				 = delete;
		using output_type																	 = base_type_new::output_type;
		using base_type																		 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return nihilus::blocking<base_type_new>;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type_new& core, uint64_t thread_count) {
			for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
				core.sync_flag_start[x].init(thread_count);
				core.sync_flag_end[x].init(thread_count);
				core.remaining_thread_count.store(thread_count, std::memory_order_release);
			}
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct weight_mapper {
		NIHILUS_FORCE_INLINE weight_mapper() noexcept								 = default;
		NIHILUS_FORCE_INLINE weight_mapper& operator=(const weight_mapper&) noexcept = delete;
		NIHILUS_FORCE_INLINE weight_mapper(const weight_mapper&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE weight_mapper& operator=(weight_mapper&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE weight_mapper(weight_mapper&&) noexcept				 = delete;
		using base_type																 = base_type_new;
		using model_traits_type														 = nihilus::model_traits_type<base_type::config>;
		using op_type_type															 = typename model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return static_cast<uint64_t>(core_traits_type::type) <= 12;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type_new& core, nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_types::count>& data) {
			if constexpr (nihilus::array_type<decltype(core.data)>) {
				for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&core.data[x]);
				}
			} else {
				for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&core.data);
				}
			}
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct memory_mapper {
		NIHILUS_FORCE_INLINE memory_mapper() noexcept								 = default;
		NIHILUS_FORCE_INLINE memory_mapper& operator=(const memory_mapper&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_mapper(const memory_mapper&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE memory_mapper& operator=(memory_mapper&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE memory_mapper(memory_mapper&&) noexcept				 = delete;
		using base_type																 = base_type_new;
		using model_traits_type														 = nihilus::model_traits_type<base_type::config>;
		using op_type_type															 = typename model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return base_type::total_required_bytes > 0;
		}
		template<typename memory_buffer_type> NIHILUS_FORCE_INLINE static void impl(base_type_new& core, memory_buffer_type& memory_buffer) {
			auto* new_ptr	  = memory_buffer.claim_memory(base_type::total_required_bytes);
			using output_type = typename base_type::output_type;
			if constexpr (nihilus::array_type<decltype(core.data)>) {
				for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
					core.data[x] = static_cast<output_type*>(new_ptr);
				}
			} else {
				core.data = static_cast<output_type*>(new_ptr);
			}
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct dim_updater {
		NIHILUS_FORCE_INLINE dim_updater() noexcept								 = default;
		NIHILUS_FORCE_INLINE dim_updater& operator=(const dim_updater&) noexcept = delete;
		NIHILUS_FORCE_INLINE dim_updater(const dim_updater&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE dim_updater& operator=(dim_updater&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE dim_updater(dim_updater&&) noexcept				 = delete;
		using base_type															 = base_type_new;
		using model_traits_type													 = nihilus::model_traits_type<base_type::config>;
		using op_type_type														 = typename model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return base_type::runtime_dims;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type_new& core, uint64_t new_dims) {
			core.get_mutable_dim() = new_dims;
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct memory_calculator {
		NIHILUS_FORCE_INLINE memory_calculator() noexcept									 = default;
		NIHILUS_FORCE_INLINE memory_calculator& operator=(const memory_calculator&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_calculator(const memory_calculator&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE memory_calculator& operator=(memory_calculator&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE memory_calculator(memory_calculator&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		using model_traits_type																 = nihilus::model_traits_type<base_type::config>;
		using op_type_type																	 = typename model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::total_required_bytes > 0;
		}
		NIHILUS_FORCE_INLINE static constexpr void impl(uint64_t& total_required_bytes) {
			total_required_bytes += base_type::total_required_bytes;
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct global_input_thread_function {
		NIHILUS_FORCE_INLINE global_input_thread_function() noexcept											   = default;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(const global_input_thread_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(global_input_thread_function&&) noexcept				   = delete;
		using base_type																							   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_input && core_traits_type::krn_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
		}
	};

	template<nihilus::model_config config, nihilus::blocking base_type_new> struct global_input_thread_function<config, base_type_new> {
		NIHILUS_FORCE_INLINE global_input_thread_function() noexcept											   = default;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(const global_input_thread_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(global_input_thread_function&&) noexcept				   = delete;
		using base_type																							   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_input && core_traits_type::krn_type != nihilus::kernel_types::none;
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct per_block_thread_function {
		NIHILUS_FORCE_INLINE per_block_thread_function() noexcept											 = default;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::per_block && core_traits_type::krn_type != nihilus::kernel_types::transpose &&
				core_traits_type::krn_type != nihilus::kernel_types::view && core_traits_type::krn_type != nihilus::kernel_types::permute &&
				core_traits_type::krn_type != nihilus::kernel_types::reshape && core_traits_type::krn_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count, uint64_t current_block) {
			nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
		}
	};

	template<nihilus::model_config config, nihilus::blocking base_type_new> struct per_block_thread_function<config, base_type_new> {
		NIHILUS_FORCE_INLINE per_block_thread_function() noexcept											 = default;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::per_block && core_traits_type::krn_type != nihilus::kernel_types::transpose &&
				core_traits_type::krn_type != nihilus::kernel_types::view && core_traits_type::krn_type != nihilus::kernel_types::permute &&
				core_traits_type::krn_type != nihilus::kernel_types::reshape && core_traits_type::krn_type != nihilus::kernel_types::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count, uint64_t current_block) {
			core.sync_flag_start[current_block].arrive_and_wait(thread_index);
			nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
			core.sync_flag_end[current_block].arrive_and_wait(thread_index);
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct global_output_thread_function {
		NIHILUS_FORCE_INLINE global_output_thread_function() noexcept												 = default;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				 = delete;
		using base_type																								 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_output && core_traits_type::krn_type != nihilus::kernel_types::transpose &&
				core_traits_type::krn_type != nihilus::kernel_types::view && core_traits_type::krn_type != nihilus::kernel_types::permute &&
				core_traits_type::krn_type != nihilus::kernel_types::reshape && core_traits_type::krn_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
		}
	};

	template<nihilus::model_config config, nihilus::blocking base_type_new> struct global_output_thread_function<config, base_type_new> {
		NIHILUS_FORCE_INLINE global_output_thread_function() noexcept												 = default;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				 = delete;
		using base_type																								 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_output && core_traits_type::krn_type != nihilus::kernel_types::transpose &&
				core_traits_type::krn_type != nihilus::kernel_types::view && core_traits_type::krn_type != nihilus::kernel_types::permute &&
				core_traits_type::krn_type != nihilus::kernel_types::reshape && core_traits_type::krn_type != nihilus::kernel_types::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			core.sync_flag_start[0].arrive_and_wait(thread_index);
			nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
			core.sync_flag_end[0].arrive_and_wait(thread_index);
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct main_thread_per_block_function {
		NIHILUS_FORCE_INLINE main_thread_per_block_function() noexcept												   = default;
		NIHILUS_FORCE_INLINE main_thread_per_block_function& operator=(const main_thread_per_block_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE main_thread_per_block_function(const main_thread_per_block_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE main_thread_per_block_function& operator=(main_thread_per_block_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE main_thread_per_block_function(main_thread_per_block_function&&) noexcept				   = delete;
		using base_type																								   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return nihilus::blocking<base_type_new> && core_traits_type::layer_type == nihilus::thread_strategy_type::per_block &&
				core_traits_type::layer_type == nihilus::thread_strategy_type::per_block && core_traits_type::krn_type != nihilus::kernel_types::reshape &&
				core_traits_type::krn_type != nihilus::kernel_types::transpose && core_traits_type::krn_type != nihilus::kernel_types::view &&
				core_traits_type::krn_type != nihilus::kernel_types::permute && core_traits_type::krn_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t current_block) {
			core.sync_flag_start[current_block].main_wait();
			core.sync_flag_end[current_block].main_wait();
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct main_thread_global_output_function {
		NIHILUS_FORCE_INLINE main_thread_global_output_function() noexcept													   = default;
		NIHILUS_FORCE_INLINE main_thread_global_output_function& operator=(const main_thread_global_output_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE main_thread_global_output_function(const main_thread_global_output_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE main_thread_global_output_function& operator=(main_thread_global_output_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE main_thread_global_output_function(main_thread_global_output_function&&) noexcept				   = delete;
		using base_type																										   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return nihilus::blocking<base_type_new> && core_traits_type::layer_type == nihilus::thread_strategy_type::global_output &&
				core_traits_type::krn_type != nihilus::kernel_types::transpose && core_traits_type::krn_type != nihilus::kernel_types::view &&
				core_traits_type::krn_type != nihilus::kernel_types::permute && core_traits_type::krn_type != nihilus::kernel_types::reshape &&
				core_traits_type::krn_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core) {
			core.sync_flag_start[0].main_wait();
			core.sync_flag_end[0].main_wait();
		}
	};

#if defined(NIHILUS_DEBUG)

	template<nihilus::model_config config, typename base_type> struct tensor_debugger_impl {
		NIHILUS_FORCE_INLINE tensor_debugger_impl() noexcept									   = default;
		NIHILUS_FORCE_INLINE tensor_debugger_impl& operator=(const tensor_debugger_impl&) noexcept = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl(const tensor_debugger_impl&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl& operator=(tensor_debugger_impl&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl(tensor_debugger_impl&&) noexcept				   = delete;
		using output_type																		   = base_type::output_type;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return true;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core) {
			if constexpr (nihilus::array_type<decltype(core.data)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					nihilus::tensor_debugger::compare_tensor_data(core, x, current_iteration);
				}
			} else {
				nihilus::tensor_debugger::compare_tensor_data(core, 0, current_iteration);
			}
		}
	};
#endif

}