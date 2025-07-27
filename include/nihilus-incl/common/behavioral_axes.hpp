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

#include <nihilus-incl/common/monolithic_dispatcher.hpp>
#include <nihilus-incl/common/memory_buffer.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	using namespace std::chrono_literals;

	NIHILUS_INLINE void spinlock_nanoseconds(uint64_t nanoseconds) {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		auto start = std::chrono::high_resolution_clock::now();
		auto end   = std::chrono::high_resolution_clock::now();
		do {
			end = std::chrono::high_resolution_clock::now();
		} while ((end - start).count() < static_cast<int64_t>(nanoseconds));
#else
		auto start	= std::chrono::high_resolution_clock::now();
		auto target = start + std::chrono::nanoseconds(nanoseconds);
		do {
		} while (std::chrono::high_resolution_clock::now() < target);
#endif
	}

	template<model_config config_new> struct core_bases_traits_type;

	template<model_config config, typename base_type> NIHILUS_INLINE static constexpr uint64_t get_dims(uint64_t, uint64_t sequence_length) {
		return sequence_length;
	}

	template<model_config config, typename base_type>
		requires(final_output_type<base_type>)
	NIHILUS_INLINE static constexpr uint64_t get_dims(uint64_t block_idx, uint64_t sequence_length) {
		uint64_t mult = runtime_dims_multipliers<base_type, config>[block_idx];
		return mult * sequence_length + (1 - mult);
	}

	template<kernel_types kernel_type, int64_t ops_per_depth> NIHILUS_INLINE constexpr int64_t get_thread_count(uint64_t base_count) {
		if constexpr (kernel_type == kernel_types::sub || kernel_type == kernel_types::reshape || kernel_type == kernel_types::permute || kernel_type == kernel_types::transpose ||
			kernel_type == kernel_types::view || kernel_type == kernel_types::cont || kernel_type == kernel_types::copy || kernel_type == kernel_types::none) {
			return 1;
		} else if constexpr (kernel_type == kernel_types::add || kernel_type == kernel_types::mul || kernel_type == kernel_types::mul_mat ||
			kernel_type == kernel_types::rms_norm || kernel_type == kernel_types::rms_norm_mul || kernel_type == kernel_types::add_rms_norm_mul ||
			kernel_type == kernel_types::rope || kernel_type == kernel_types::silu) {
			return static_cast<int64_t>(base_count);
		} else if constexpr (kernel_type == kernel_types::get_rows) {
			return 1;
		} else if constexpr (kernel_type == kernel_types::softmax) {
			return static_cast<int64_t>(base_count);
		} else {
			return static_cast<int64_t>(base_count);
		}
	}

	template<model_config config, typename base_type_new> struct execution_planner {
		NIHILUS_INLINE execution_planner() noexcept									   = default;
		NIHILUS_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_INLINE execution_planner(const execution_planner&) noexcept			   = delete;
		NIHILUS_INLINE execution_planner& operator=(execution_planner&&) noexcept	   = delete;
		NIHILUS_INLINE execution_planner(execution_planner&&) noexcept				   = delete;
		using base_type																   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return active_op_types<base_type> || active_input_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			//std::cout << "CURRENT THREAD COUNT(REAL): " << thread_count << std::endl;
			if constexpr (active_op_types<base_type>) {
				if constexpr (array_types<decltype(parse_core.run_checkers)>) {
					for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
						parse_core.run_checkers[x].init(get_thread_count<base_type::kernel_type, core_bases_traits_type<config>::ops_per_depth[base_type::depth]>(thread_count));
					}
				} else {
					parse_core.run_checkers.init(get_thread_count<base_type::kernel_type, core_bases_traits_type<config>::ops_per_depth[base_type::depth]>(thread_count));
				}
			}
		}
	};

	template<model_config config, blocking_types base_type_new> struct execution_planner<config, base_type_new> {
		NIHILUS_INLINE execution_planner() noexcept									   = default;
		NIHILUS_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_INLINE execution_planner(const execution_planner&) noexcept			   = delete;
		NIHILUS_INLINE execution_planner& operator=(execution_planner&&) noexcept	   = delete;
		NIHILUS_INLINE execution_planner(execution_planner&&) noexcept				   = delete;
		using base_type																   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return blocking_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			//std::cout << "CURRENT THREAD COUNT: " << thread_count << std::endl;
			if constexpr (array_types<decltype(parse_core.sync_flag_start)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					parse_core.sync_flag_start[x].init(static_cast<int64_t>(thread_count));
					parse_core.sync_flag_end[x].init(static_cast<int64_t>(thread_count));
				}
			} else {
				parse_core.sync_flag_start.init(static_cast<int64_t>(thread_count));
				parse_core.sync_flag_end.init(static_cast<int64_t>(thread_count));
			}
		}
	};

	template<model_config config, typename base_type_new> struct weight_mapper {
		NIHILUS_INLINE weight_mapper() noexcept								   = default;
		NIHILUS_INLINE weight_mapper& operator=(const weight_mapper&) noexcept = delete;
		NIHILUS_INLINE weight_mapper(const weight_mapper&) noexcept			   = delete;
		NIHILUS_INLINE weight_mapper& operator=(weight_mapper&&) noexcept	   = delete;
		NIHILUS_INLINE weight_mapper(weight_mapper&&) noexcept				   = delete;
		using base_type														   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return static_cast<uint64_t>(base_type::type) <= 12;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, array<array<void*, model_traits_type<config>::block_count>, op_types::count>& data) {
			if constexpr (array_types<decltype(parse_core.data)>) {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&parse_core.data[x]);
				}
			} else {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&parse_core.data);
				}
			}
		}
	};

	struct depth_and_bytes {
		int64_t first_used_depth{ -1 };
		int64_t last_used_depth{};
		int64_t required_bytes{};
		op_types type{};
		bool active{};
		NIHILUS_INLINE constexpr bool operator<(const depth_and_bytes& other) const {
			return first_used_depth < other.first_used_depth;
		}
	};

	template<model_config config, typename base_type_new> struct memory_planner_depths {
		NIHILUS_INLINE memory_planner_depths() noexcept										   = default;
		NIHILUS_INLINE memory_planner_depths& operator=(const memory_planner_depths&) noexcept = delete;
		NIHILUS_INLINE memory_planner_depths(const memory_planner_depths&) noexcept			   = delete;
		NIHILUS_INLINE memory_planner_depths& operator=(memory_planner_depths&&) noexcept	   = delete;
		NIHILUS_INLINE memory_planner_depths(memory_planner_depths&&) noexcept				   = delete;
		using base_type																		   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::allocation_strategy_type != allocation_strategy_types::mmap && base_type::allocation_strategy_type != allocation_strategy_types::none;
		}

		NIHILUS_INLINE static constexpr void impl(const base_type&, array<depth_and_bytes, op_types::count>& depths_new) {
			if (depths_new[base_type::type].first_used_depth == -1) {
				depths_new[base_type::type].last_used_depth	 = base_type::depth;
				depths_new[base_type::type].required_bytes	 = base_type::total_required_bytes;
				depths_new[base_type::type].first_used_depth = base_type::depth;
				depths_new[base_type::type].type			 = base_type::type;
			}
			if constexpr (single_input_types<base_type>) {
				depths_new[base_type::input_01_type::type].last_used_depth = base_type::depth;
			} else if constexpr (double_input_types<base_type>) {
				depths_new[base_type::input_01_type::type].last_used_depth = base_type::depth;
				depths_new[base_type::input_02_type::type].last_used_depth = base_type::depth;
			} else if constexpr (triple_input_types<base_type>) {
				depths_new[base_type::input_01_type::type].last_used_depth = base_type::depth;
				depths_new[base_type::input_02_type::type].last_used_depth = base_type::depth;
				depths_new[base_type::input_03_type::type].last_used_depth = base_type::depth;
			}
		}
	};

	struct offset_and_size {
		int64_t offset{};
		int64_t size{};
		int64_t last_used_depth{};
		op_types type{};
	};

	struct memory_plan {
		NIHILUS_INLINE constexpr memory_plan(uint64_t size_new) : memory_total{ size_new } {
		}
		array<offset_and_size, op_types::count> offsets{};
		uint64_t memory_total{};
	};

	template<model_config config, typename model_type> struct thread_pool;

	template<model_config config, typename base_type_new> struct memory_mapper {
		NIHILUS_INLINE memory_mapper() noexcept								   = default;
		NIHILUS_INLINE memory_mapper& operator=(const memory_mapper&) noexcept = delete;
		NIHILUS_INLINE memory_mapper(const memory_mapper&) noexcept			   = delete;
		NIHILUS_INLINE memory_mapper& operator=(memory_mapper&&) noexcept	   = delete;
		NIHILUS_INLINE memory_mapper(memory_mapper&&) noexcept				   = delete;
		using base_type														   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::allocation_strategy_type != allocation_strategy_types::mmap;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, const memory_plan& plan, memory_buffer<config>& memory_buffer) {
			using output_type = typename base_type::output_type;
			if constexpr (base_type::type == op_types::l_out) {
				parse_core.data = static_cast<core_traits<config, op_types::inp_embd>*>(static_cast<thread_pool<config, model<config>>*>(&parse_core))->data;
			} else {
				if constexpr (remapped_op_types<base_type>) {
					using input_01_type	  = typename base_type::input_01_type;
					using other_data_type = decltype(input_01_type::data);
					if constexpr (array_types<decltype(parse_core.data)> && array_types<other_data_type>) {
						for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
							parse_core.data[x] = static_cast<input_01_type*>(static_cast<thread_pool<config, model<config>>*>(&parse_core))->data[x];
						}
					} else if constexpr (array_types<decltype(parse_core.data)> && !array_types<other_data_type>) {
						for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
							parse_core.data[x] = static_cast<input_01_type*>(static_cast<thread_pool<config, model<config>>*>(&parse_core))->data;
						}
					} else if constexpr (!array_types<decltype(parse_core.data)> && array_types<other_data_type>) {
						parse_core.data = static_cast<input_01_type*>(static_cast<thread_pool<config, model<config>>*>(&parse_core))->data[0];
					} else if constexpr (!array_types<decltype(parse_core.data)> && !array_types<other_data_type>) {
						parse_core.data = static_cast<input_01_type*>(static_cast<thread_pool<config, model<config>>*>(&parse_core))->data;
					} else {
						std::cout << "Sorry, but failed to map op of type: " << base_type::type << std::endl;
					}
				} else {
					auto* new_ptr = memory_buffer.claim_memory(static_cast<uint64_t>(plan.offsets[base_type::type].offset));
					if constexpr (array_types<decltype(parse_core.data)>) {
						for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
							parse_core.data[x] = static_cast<output_type*>(new_ptr);
						}
					} else {
						parse_core.data = static_cast<output_type*>(new_ptr);
					}
				}
			}
		}
	};

	template<model_config config, typename base_type_new> struct max_depth_calculator {
		NIHILUS_INLINE max_depth_calculator() noexcept										 = default;
		NIHILUS_INLINE max_depth_calculator& operator=(const max_depth_calculator&) noexcept = delete;
		NIHILUS_INLINE max_depth_calculator(const max_depth_calculator&) noexcept			 = delete;
		NIHILUS_INLINE max_depth_calculator& operator=(max_depth_calculator&&) noexcept		 = delete;
		NIHILUS_INLINE max_depth_calculator(max_depth_calculator&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return true;
		}
		NIHILUS_INLINE static constexpr void impl(const base_type&, uint64_t& current_max_depth) {
			current_max_depth = base_type::depth > current_max_depth ? base_type::depth : current_max_depth;
		}
	};

	template<model_config config, typename base_type_new> struct ops_per_depth_calculator {
		NIHILUS_INLINE ops_per_depth_calculator() noexcept											 = default;
		NIHILUS_INLINE ops_per_depth_calculator& operator=(const ops_per_depth_calculator&) noexcept = delete;
		NIHILUS_INLINE ops_per_depth_calculator(const ops_per_depth_calculator&) noexcept			 = delete;
		NIHILUS_INLINE ops_per_depth_calculator& operator=(ops_per_depth_calculator&&) noexcept		 = delete;
		NIHILUS_INLINE ops_per_depth_calculator(ops_per_depth_calculator&&) noexcept				 = delete;
		using base_type																				 = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return true;
		}
		template<uint64_t max_depth> NIHILUS_INLINE static constexpr void impl(const base_type&, array<uint64_t, max_depth>& ops_per_depth) {
			++ops_per_depth[base_type::depth];
		}
	};

	template<model_config config, typename base_type_new> struct dim_updater {
		NIHILUS_INLINE dim_updater() noexcept							   = default;
		NIHILUS_INLINE dim_updater& operator=(const dim_updater&) noexcept = delete;
		NIHILUS_INLINE dim_updater(const dim_updater&) noexcept			   = delete;
		NIHILUS_INLINE dim_updater& operator=(dim_updater&&) noexcept	   = delete;
		NIHILUS_INLINE dim_updater(dim_updater&&) noexcept				   = delete;
		using base_type													   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return active_op_types<base_type> || active_input_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t runtime_dim) {
			if constexpr (base_type::runtime_dims) {
				parse_core.get_mutable_dim() = get_dims<config, base_type>(0, runtime_dim) * base_type::runtime_dim_multiplier;
			}
		}
	};

	template<model_config config, typename base_type_new> struct run_checker_resetter {
		NIHILUS_INLINE run_checker_resetter() noexcept										 = default;
		NIHILUS_INLINE run_checker_resetter& operator=(const run_checker_resetter&) noexcept = delete;
		NIHILUS_INLINE run_checker_resetter(const run_checker_resetter&) noexcept			 = delete;
		NIHILUS_INLINE run_checker_resetter& operator=(run_checker_resetter&&) noexcept		 = delete;
		NIHILUS_INLINE run_checker_resetter(run_checker_resetter&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return active_op_types<base_type> || active_input_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core) {
			if constexpr (active_op_types<base_type>) {
				parse_core.run_checkers.reset();
			}
		}
	};

	template<model_config config, typename base_type_new> struct global_input_thread_function {
		NIHILUS_INLINE global_input_thread_function() noexcept												 = default;
		NIHILUS_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_input_thread_function(const global_input_thread_function&) noexcept			 = delete;
		NIHILUS_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept		 = delete;
		NIHILUS_INLINE global_input_thread_function(global_input_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		using return_type																					 = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::global_input && base_type::kernel_type != kernel_types::none;
		}
		NIHILUS_INLINE static bool impl(base_type& parse_core) {
			int64_t thread_count{ parse_core.run_checkers.thread_count };
			if (int64_t thread_index = parse_core.run_checkers.do_we_run(); thread_index < thread_count) {
				kernel_dispatcher<config, device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				if constexpr (config.benchmark) {
					//spinlock_nanoseconds(spinlock_time);
				}
				return false;
			} else {
				return true;
			}
		}
	};

	template<model_config config, typename base_type_new> struct per_block_thread_function {
		NIHILUS_INLINE per_block_thread_function() noexcept											   = default;
		NIHILUS_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				   = delete;
		using base_type																				   = base_type_new;
		using return_type																			   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::per_block && base_type::kernel_type != kernel_types::transpose &&
				base_type::kernel_type != kernel_types::view && base_type::kernel_type != kernel_types::permute && base_type::kernel_type != kernel_types::reshape &&
				base_type::kernel_type != kernel_types::none;
		}
		NIHILUS_INLINE static bool impl(base_type& parse_core, uint64_t current_block) {
			int64_t thread_count{ parse_core.run_checkers[current_block].thread_count };
			if (int64_t thread_index = parse_core.run_checkers[current_block].do_we_run(); thread_index < thread_count) {
				kernel_dispatcher<config, device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				if constexpr (config.benchmark) {
					//spinlock_nanoseconds(spinlock_time);
				}
				return false;
			} else {
				return true;
			}
		}
	};

	template<model_config config, blocking_types base_type_new> struct per_block_thread_function<config, base_type_new> {
		NIHILUS_INLINE per_block_thread_function() noexcept											   = default;
		NIHILUS_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				   = delete;
		using base_type																				   = base_type_new;
		using return_type																			   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::per_block && base_type::kernel_type != kernel_types::transpose &&
				base_type::kernel_type != kernel_types::view && base_type::kernel_type != kernel_types::permute && base_type::kernel_type != kernel_types::reshape &&
				base_type::kernel_type != kernel_types::none;
		}

		NIHILUS_INLINE static bool impl(base_type& parse_core, uint64_t) {
			int64_t thread_index{};
			int64_t thread_count = parse_core.sync_flag_start.arrive_and_wait_get_thread(thread_index);
			kernel_dispatcher<config, device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
			if constexpr (config.benchmark) {
				//spinlock_nanoseconds(spinlock_time);
			}
			parse_core.sync_flag_end.arrive_and_wait();
			return true;
		}
	};

	template<model_config config, typename base_type_new> struct global_output_thread_function {
		NIHILUS_INLINE global_output_thread_function() noexcept												   = default;
		NIHILUS_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				   = delete;
		using base_type																						   = base_type_new;
		using return_type																					   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::global_output && base_type::kernel_type != kernel_types::transpose &&
				base_type::kernel_type != kernel_types::view && base_type::kernel_type != kernel_types::permute && base_type::kernel_type != kernel_types::reshape &&
				base_type::kernel_type != kernel_types::none;
		}
		NIHILUS_INLINE static bool impl(base_type& parse_core) {
			int64_t thread_count{ parse_core.run_checkers.thread_count };
			if (int64_t thread_index = parse_core.run_checkers.do_we_run(); thread_index < thread_count) {
				kernel_dispatcher<config, device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
				if constexpr (config.benchmark) {
					//spinlock_nanoseconds(spinlock_time);
				}
				return false;
			} else {
				return true;
			}
		}
	};

	template<model_config config, blocking_types base_type_new> struct global_output_thread_function<config, base_type_new> {
		NIHILUS_INLINE global_output_thread_function() noexcept												   = default;
		NIHILUS_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				   = delete;
		using base_type																						   = base_type_new;
		using return_type																					   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::global_output && base_type::kernel_type != kernel_types::transpose &&
				base_type::kernel_type != kernel_types::view && base_type::kernel_type != kernel_types::permute && base_type::kernel_type != kernel_types::reshape &&
				base_type::kernel_type != kernel_types::none;
		}

		NIHILUS_INLINE static bool impl(base_type& parse_core) {
			int64_t thread_index{};
			int64_t thread_count = parse_core.sync_flag_start.arrive_and_wait_get_thread(thread_index);
			kernel_dispatcher<config, device_types::cpu, base_type>::impl(parse_core, thread_index, thread_count);
			if constexpr (config.benchmark) {
				//spinlock_nanoseconds(spinlock_time);
			}
			parse_core.sync_flag_end.arrive_and_wait();
			return true;
		}
	};

	template<model_config config, typename base_type_new> struct tensor_debugger_impl {
		NIHILUS_INLINE tensor_debugger_impl() noexcept										 = default;
		NIHILUS_INLINE tensor_debugger_impl& operator=(const tensor_debugger_impl&) noexcept = delete;
		NIHILUS_INLINE tensor_debugger_impl(const tensor_debugger_impl&) noexcept			 = delete;
		NIHILUS_INLINE tensor_debugger_impl& operator=(tensor_debugger_impl&&) noexcept		 = delete;
		NIHILUS_INLINE tensor_debugger_impl(tensor_debugger_impl&&) noexcept				 = delete;
		using base_type																		 = base_type_new;
		using output_type																	 = base_type::output_type;
		NIHILUS_INLINE static constexpr bool filter() {
			return config.dev;
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t current_block, uint64_t current_iteration_new, uint64_t runtime_dim) {
			runtime_dim = get_dims<config, base_type>(current_block, runtime_dim) * base_type::runtime_dim_multiplier;
			tensor_debugger::compare_tensor_data(parse_core, current_block, current_iteration_new, runtime_dim);
		}
	};

	template<model_config config, typename base_type_new> struct execution_checker {
		NIHILUS_INLINE execution_checker() noexcept									   = default;
		NIHILUS_INLINE execution_checker& operator=(const execution_checker&) noexcept = delete;
		NIHILUS_INLINE execution_checker(const execution_checker&) noexcept			   = delete;
		NIHILUS_INLINE execution_checker& operator=(execution_checker&&) noexcept	   = delete;
		NIHILUS_INLINE execution_checker(execution_checker&&) noexcept				   = delete;
		using output_type															   = base_type_new::output_type;
		using base_type																   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return active_op_types<base_type> && config.dev;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core) {
			if constexpr (array_types<decltype(parse_core.run_checkers)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					if (parse_core.run_checkers[x].flag.load() < parse_core.run_checkers[x].thread_count) {
						std::stringstream stream{};
						stream << "Failed to finish op of type: " << base_type::type << ", FOR BLOCK: " << x << std::endl;
						stream << "Actuated Count: " << parse_core.run_checkers[x].flag.load() << ", Expected Count: " << parse_core.run_checkers[x].thread_count << std::endl;
						log<log_levels::status>(stream.str());
					}
				}
			} else {
				if (parse_core.run_checkers.flag.load() < parse_core.run_checkers.thread_count) {
					std::stringstream stream{};
					stream << "Failed to finish op of type: " << base_type::type << std::endl;
					stream << "Actuated Count: " << parse_core.run_checkers.flag.load() << ", Expected Count: " << parse_core.run_checkers.thread_count << std::endl;
					log<log_levels::status>(stream.str());
				}
			}
		}
	};

	template<model_config config, blocking_types base_type_new> struct execution_checker<config, base_type_new> {
		NIHILUS_INLINE execution_checker() noexcept									   = default;
		NIHILUS_INLINE execution_checker& operator=(const execution_checker&) noexcept = delete;
		NIHILUS_INLINE execution_checker(const execution_checker&) noexcept			   = delete;
		NIHILUS_INLINE execution_checker& operator=(execution_checker&&) noexcept	   = delete;
		NIHILUS_INLINE execution_checker(execution_checker&&) noexcept				   = delete;
		using output_type															   = base_type_new::output_type;
		using base_type																   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return active_op_types<base_type> && config.dev;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core) {
			if constexpr (array_types<decltype(parse_core.run_checkers)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					if (parse_core.run_checkers[x].flag.load() < parse_core.run_checkers[x].thread_count) {
						std::cout << "Failed to finish op of type: " << base_type::type << ", FOR BLOCK: " << x << std::endl;
						std::cout << "Actuated Count: " << parse_core.run_checkers[x].flag.load() << ", Expected Count: " << parse_core.run_checkers[x].thread_count << std::endl;
					}
				}
			} else {
				if (parse_core.run_checkers.flag.load() < parse_core.run_checkers.thread_count) {
					std::cout << "Failed to finish op of type: " << base_type::type << std::endl;
					std::cout << "Actuated Count: " << parse_core.run_checkers.flag.load() << ", Expected Count: " << parse_core.run_checkers.thread_count << std::endl;
				}
			}
		}
	};

}
