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
#if !defined(NIHILUS_DEV)
	#if defined(NIHILUS_PLATFORM_WINDOWS)
		auto start = std::chrono::high_resolution_clock::now();
		auto end   = std::chrono::high_resolution_clock::now();
		do {
			end = std::chrono::high_resolution_clock::now();
		} while ((end - start).count() < static_cast<int64_t>(nanoseconds));
	#else
		// Linux/Unix implementation
		auto start	= std::chrono::high_resolution_clock::now();
		auto target = start + std::chrono::nanoseconds(nanoseconds);
		do {
		} while (std::chrono::high_resolution_clock::now() < target);
	#endif
#endif
	}

	template<model_config config_new> struct core_bases_traits;

	int64_t current_iteration{};

	template<nihilus::model_config config, typename base_type_new> struct execution_planner {
		NIHILUS_FORCE_INLINE execution_planner() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_planner(const execution_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_planner& operator=(execution_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_planner(execution_planner&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return active_op_type<base_type>;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			if constexpr (array_type<decltype(parse_core.run_checkers)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					parse_core.run_checkers[x].init(detail::max(1ull, thread_count / core_bases_traits<config>::ops_per_depth[base_type::depth]));
				}
			} else {
				parse_core.run_checkers.init(detail::max(1ull, thread_count / core_bases_traits<config>::ops_per_depth[base_type::depth]));
			}
		}
	};

	template<nihilus::model_config config, blocking base_type_new> struct execution_planner<config, base_type_new> {
		NIHILUS_FORCE_INLINE execution_planner() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_planner(const execution_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_planner& operator=(execution_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_planner(execution_planner&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return active_op_type<base_type>;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			if constexpr (array_type<decltype(parse_core.run_checkers)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					parse_core.run_checkers[x].init(thread_count);
				}
			} else {
				parse_core.run_checkers.init(thread_count);
			}
			if constexpr (array_type<decltype(parse_core.sync_flag_start)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					parse_core.sync_flag_start[x].init(thread_count);
					parse_core.sync_flag_end[x].init(thread_count);
				}
			} else {
				parse_core.sync_flag_start.init(thread_count);
				parse_core.sync_flag_end.init(thread_count);
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
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return static_cast<uint64_t>(base_type::type) <= 12;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_types::count>& data) {
			if constexpr (nihilus::array_type<decltype(parse_core.data)>) {
				for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&parse_core.data[x]);
				}
			} else {
				for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&parse_core.data);
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
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return base_type::total_required_bytes > 0;
		}
		template<typename memory_buffer_type> NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, memory_buffer_type& memory_buffer) {
			auto* new_ptr	  = memory_buffer.claim_memory(base_type::total_required_bytes);
			using output_type = typename base_type::output_type;
			if constexpr (nihilus::array_type<decltype(parse_core.data)>) {
				for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
					parse_core.data[x] = static_cast<output_type*>(new_ptr);
				}
			} else {
				parse_core.data = static_cast<output_type*>(new_ptr);
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
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			bool return_value{};
			for (uint64_t x = 0; x < 4; ++x) {
				return_value |= base_type::runtime_dims[x];
			}
			return return_value;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t new_dims) {
			parse_core.get_mutable_dim() = new_dims;
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct max_depth_calculator {
		NIHILUS_FORCE_INLINE max_depth_calculator() noexcept									   = default;
		NIHILUS_FORCE_INLINE max_depth_calculator& operator=(const max_depth_calculator&) noexcept = delete;
		NIHILUS_FORCE_INLINE max_depth_calculator(const max_depth_calculator&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE max_depth_calculator& operator=(max_depth_calculator&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE max_depth_calculator(max_depth_calculator&&) noexcept				   = delete;
		using base_type																			   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return true;
		}
		NIHILUS_FORCE_INLINE static constexpr void impl(const base_type& parse_core, uint64_t& current_max_depth) {
			current_max_depth = base_type::depth > current_max_depth ? base_type::depth : current_max_depth;
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct ops_per_depth_calculator {
		NIHILUS_FORCE_INLINE ops_per_depth_calculator() noexcept										   = default;
		NIHILUS_FORCE_INLINE ops_per_depth_calculator& operator=(const ops_per_depth_calculator&) noexcept = delete;
		NIHILUS_FORCE_INLINE ops_per_depth_calculator(const ops_per_depth_calculator&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE ops_per_depth_calculator& operator=(ops_per_depth_calculator&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE ops_per_depth_calculator(ops_per_depth_calculator&&) noexcept				   = delete;
		using base_type																					   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return true;
		}
		template<uint64_t max_depth> NIHILUS_FORCE_INLINE static constexpr void impl(const base_type& parse_core, array<uint64_t, max_depth>& ops_per_depth) {
			++ops_per_depth[base_type::depth];
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct memory_calculator {
		NIHILUS_FORCE_INLINE constexpr memory_calculator() noexcept							 = default;
		NIHILUS_FORCE_INLINE memory_calculator& operator=(const memory_calculator&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_calculator(const memory_calculator&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE memory_calculator& operator=(memory_calculator&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE memory_calculator(memory_calculator&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return base_type::total_required_bytes > 0;
		}
		NIHILUS_FORCE_INLINE static constexpr void impl(const base_type& parse_core, uint64_t& total_required_bytes) {
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
			return base_type::layer_type == nihilus::thread_strategy_type::global_input && base_type::kernel_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core) {
			int64_t thread_count{ parse_core.run_checkers.thread_count };
			if (int64_t thread_index = parse_core.run_checkers.do_we_run(); thread_index < thread_count) {
				nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				nihilus::spinlock_nanoseconds(spinlock_time);
				count.fetch_add(1, std::memory_order_release);
			}
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
			return base_type::layer_type == nihilus::thread_strategy_type::per_block && base_type::kernel_type != nihilus::kernel_types::transpose &&
				base_type::kernel_type != nihilus::kernel_types::view && base_type::kernel_type != nihilus::kernel_types::permute &&
				base_type::kernel_type != nihilus::kernel_types::reshape && base_type::kernel_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t current_block) {
			int64_t thread_count{ parse_core.run_checkers[current_block].thread_count };
			if (int64_t thread_index = parse_core.run_checkers[current_block].do_we_run(); thread_index < thread_count) {
				nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				nihilus::spinlock_nanoseconds(spinlock_time);
				count.fetch_add(1, std::memory_order_release);
			}
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
			return base_type::layer_type == nihilus::thread_strategy_type::per_block && base_type::kernel_type != nihilus::kernel_types::transpose &&
				base_type::kernel_type != nihilus::kernel_types::view && base_type::kernel_type != nihilus::kernel_types::permute &&
				base_type::kernel_type != nihilus::kernel_types::reshape && base_type::kernel_type != nihilus::kernel_types::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t current_block) {
			int64_t thread_count{ parse_core.run_checkers[current_block].thread_count };
			if (int64_t thread_index = parse_core.run_checkers[current_block].do_we_run(); thread_index < thread_count) {
				parse_core.sync_flag_start[current_block].arrive_and_wait();
				nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				//nihilus::spinlock_nanoseconds(spinlock_time);
				parse_core.sync_flag_end[current_block].arrive_and_wait();
				//count.fetch_add(1, std::memory_order_release);
			}
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
			return base_type::layer_type == nihilus::thread_strategy_type::global_output && base_type::kernel_type != nihilus::kernel_types::transpose &&
				base_type::kernel_type != nihilus::kernel_types::view && base_type::kernel_type != nihilus::kernel_types::permute &&
				base_type::kernel_type != nihilus::kernel_types::reshape && base_type::kernel_type != nihilus::kernel_types::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core) {
			int64_t thread_count{ parse_core.run_checkers.thread_count };
			if (int64_t thread_index = parse_core.run_checkers.do_we_run(); thread_index < thread_count) {
				nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				//nihilus::spinlock_nanoseconds(spinlock_time);
				//count.fetch_add(1, std::memory_order_release);
			}
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
			return base_type::layer_type == nihilus::thread_strategy_type::global_output && base_type::kernel_type != nihilus::kernel_types::transpose &&
				base_type::kernel_type != nihilus::kernel_types::view && base_type::kernel_type != nihilus::kernel_types::permute &&
				base_type::kernel_type != nihilus::kernel_types::reshape && base_type::kernel_type != nihilus::kernel_types::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core) {
			int64_t thread_count{ parse_core.run_checkers.thread_count };
			if (int64_t thread_index = parse_core.run_checkers.do_we_run(); thread_index < thread_count) {
				parse_core.sync_flag_start.arrive_and_wait();
				nihilus::kernel_dispatcher<config, nihilus::device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				//				nihilus::spinlock_nanoseconds(spinlock_time);
				parse_core.sync_flag_end.arrive_and_wait();
				//count.fetch_add(1, std::memory_order_release);
			}
		}
	};

#if defined(NIHILUS_DEV)

	template<nihilus::model_config config, typename base_type_new> struct tensor_debugger_impl {
		NIHILUS_FORCE_INLINE tensor_debugger_impl() noexcept									   = default;
		NIHILUS_FORCE_INLINE tensor_debugger_impl& operator=(const tensor_debugger_impl&) noexcept = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl(const tensor_debugger_impl&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl& operator=(tensor_debugger_impl&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl(tensor_debugger_impl&&) noexcept				   = delete;
		using base_type																			   = base_type_new;
		using output_type																		   = base_type::output_type;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return true;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core) {
			if constexpr (nihilus::array_type<decltype(parse_core.data)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					nihilus::tensor_debugger::compare_tensor_data(parse_core, x, current_iteration);
				}
			} else {
				nihilus::tensor_debugger::compare_tensor_data(parse_core, 0, current_iteration);
			}
		}
	};


	template<nihilus::model_config config, typename base_type_new> struct execution_checker {
		NIHILUS_FORCE_INLINE execution_checker() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_checker& operator=(const execution_checker&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_checker(const execution_checker&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_checker& operator=(execution_checker&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_checker(execution_checker&&) noexcept				 = delete;
		using output_type																	 = base_type_new::output_type;
		using base_type																		 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return active_op_type<base_type>;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			if constexpr (array_type<decltype(parse_core.run_checkers)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					if (parse_core.run_checkers[x].flag.load() < parse_core.run_checkers[x].thread_count) {
						std::cout << "Failed to finish op of type: " << ( int32_t )base_type::type << ", FOR BLOCK: " << x << std::endl;
						std::cout << "Actuated Count: " << parse_core.run_checkers[x].flag.load() << ", Expected Count: " << parse_core.run_checkers[x].thread_count << std::endl;
					}
				}
			} else {
				if (parse_core.run_checkers.flag.load() < parse_core.run_checkers.thread_count) {
					std::cout << "Failed to finish op of type: " << ( int32_t )base_type::type << std::endl;
					std::cout << "Actuated Count: " << parse_core.run_checkers.flag.load() << ", Expected Count: " << parse_core.run_checkers.thread_count << std::endl;
				}
			}
		}
	};

	template<nihilus::model_config config, blocking base_type_new> struct execution_checker<config, base_type_new> {
		NIHILUS_FORCE_INLINE execution_checker() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_checker& operator=(const execution_checker&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_checker(const execution_checker&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_checker& operator=(execution_checker&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_checker(execution_checker&&) noexcept				 = delete;
		using output_type																	 = base_type_new::output_type;
		using base_type																		 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			return active_op_type<base_type>;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			if constexpr (array_type<decltype(parse_core.sync_flag_start)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					if (parse_core.run_checkers[x].flag.load() < parse_core.run_checkers[x].thread_count) {
						std::cout << "Failed to finish op of type: " << ( int32_t )base_type::type << ", FOR BLOCK: " << x << std::endl;
						std::cout << "Actuated Count: " << parse_core.run_checkers[x].flag.load() << ", Expected Count: " << parse_core.run_checkers[x].thread_count << std::endl;
					}
				}
			} else {
				if (parse_core.run_checkers.flag.load() < parse_core.run_checkers.thread_count) {
					std::cout << "Failed to finish op of type: " << ( int32_t )base_type::type << std::endl;
					std::cout << "Actuated Count: " << parse_core.run_checkers.flag.load() << ", Expected Count: " << parse_core.run_checkers.thread_count << std::endl;
				}
			}
		}
	};

#endif

}