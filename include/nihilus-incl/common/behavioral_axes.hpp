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

	enum class thread_strategy_types : uint8_t {
		none,
		global_input,
		global_output,
		per_block,
	};

	enum class allocation_strategy_types : uint8_t {
		none,
		mmap,
		remap,
		alloc,
	};

	enum class data_strategy_types : uint8_t {
		none,
		global,
		per_block,
	};

	using namespace std::chrono_literals;

	NIHILUS_INLINE void spinlock_nanoseconds(uint64_t nanoseconds) {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		auto start = clock_type::now();
		auto end   = clock_type::now();
		do {
			end = clock_type::now();
		} while ((end - start).count() < static_cast<int64_t>(nanoseconds));
#else
		auto start	= clock_type::now();
		auto target = start + std::chrono::nanoseconds(nanoseconds);
		do {
		} while (clock_type::now() < target);
#endif
	}

	/*

	template<model_config config, typename base_type_new> struct current_chunk_resetter {
		NIHILUS_INLINE current_chunk_resetter() noexcept								 = default;
		NIHILUS_INLINE current_chunk_resetter& operator=(const current_chunk_resetter&) noexcept = delete;
		NIHILUS_INLINE current_chunk_resetter(const current_chunk_resetter&) noexcept			 = delete;
		NIHILUS_INLINE current_chunk_resetter& operator=(current_chunk_resetter&&) noexcept		 = delete;
		NIHILUS_INLINE current_chunk_resetter(current_chunk_resetter&&) noexcept				 = delete;
		using base_type															 = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_chunk_types<base_type> || has_latch_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core) {
			for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
				if constexpr (base_type::kernel_type == kernel_types::mul_mat) {
					parse_core.current_chunk[x].store(0);
				}
				if constexpr (base_type::global_input_count > 0) {
					if (x == 0) {
						parse_core.latch[x].reset(0);
					} else {
						parse_core.latch[x].reset(base_type::global_input_count);
					}
				} else {
					parse_core.latch[x].reset(base_type::global_input_count);
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
			return base_type::data_strategy_type != data_strategy_types::global && base_type::data_strategy_type != data_strategy_types::none &&
				base_type::data_strategy_type != data_strategy_types::global;
		}

		NIHILUS_INLINE static constexpr void impl(const base_type&, array<depth_and_bytes, op_types::count>& depths_new) {
			if (depths_new[base_type::core_type].first_used_depth == -1) {
				depths_new[base_type::core_type].last_used_depth	= base_type::depth;
				depths_new[base_type::core_type].required_bytes	= base_type::total_required_bytes;
				depths_new[base_type::core_type].first_used_depth = base_type::depth;
				depths_new[base_type::core_type].type				= base_type::core_type;
			}
			if constexpr (single_input_types<base_type>) {
				depths_new[base_type::input_01_type::core_type].last_used_depth = base_type::depth;
			} else if constexpr (double_input_types<base_type>) {
				depths_new[base_type::input_01_type::core_type].last_used_depth = base_type::depth;
				depths_new[base_type::input_02_type::core_type].last_used_depth = base_type::depth;
			} else if constexpr (triple_input_types<base_type>) {
				depths_new[base_type::input_01_type::core_type].last_used_depth = base_type::depth;
				depths_new[base_type::input_02_type::core_type].last_used_depth = base_type::depth;
				depths_new[base_type::input_03_type::core_type].last_used_depth = base_type::depth;
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

	template<model_config config, typename base_type_new> struct memory_mapper {
		NIHILUS_INLINE memory_mapper() noexcept								   = default;
		NIHILUS_INLINE memory_mapper& operator=(const memory_mapper&) noexcept = delete;
		NIHILUS_INLINE memory_mapper(const memory_mapper&) noexcept			   = delete;
		NIHILUS_INLINE memory_mapper& operator=(memory_mapper&&) noexcept	   = delete;
		NIHILUS_INLINE memory_mapper(memory_mapper&&) noexcept				   = delete;
		using base_type														   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::data_strategy_type != data_strategy_types::global;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, const memory_plan& plan, memory_buffer<config>& memory_buffer) {
			using output_type = typename base_type::output_type;
			if constexpr (base_type::data_strategy_type == data_strategy_types::global) {
				using input_01_type	  = typename base_type::input_01_type;
				using other_data_type = decltype(input_01_type::data);
				if constexpr (array_types<decltype(parse_core.data)> && array_types<other_data_type>) {
					for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
						parse_core.data[x] = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data[x];
					}
				} else if constexpr (array_types<decltype(parse_core.data)> && !array_types<other_data_type>) {
					for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
						parse_core.data[x] = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data;
					}
				} else if constexpr (!array_types<decltype(parse_core.data)> && array_types<other_data_type>) {
					parse_core.data = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data[0];
				} else if constexpr (!array_types<decltype(parse_core.data)> && !array_types<other_data_type>) {
					parse_core.data = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data;
				} else {
					std::cout << "Sorry, but failed to map op of type: " << base_type::core_type << ", " << std::endl;
				}
			} else {
				auto* new_ptr = memory_buffer.claim_memory(static_cast<uint64_t>(plan.offsets[base_type::core_type].offset));
				if constexpr (array_types<decltype(parse_core.data)>) {
					for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
						parse_core.data[x] = static_cast<output_type*>(new_ptr);
					}
				} else {
					parse_core.data = static_cast<output_type*>(new_ptr);
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
			//current_max_depth = base_type::depth > current_max_depth ? base_type::depth : current_max_depth;
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
			//++ops_per_depth[base_type::depth];
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
			return has_latch_types<base_type> || active_input_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t runtime_dim) {
			//if constexpr (base_type::runtime_dims != 5) {
			//				parse_core.get_mutable_dim() = get_dims<config, base_type>(0, runtime_dim) * base_type::runtime_dim_multiplier;
			//}
		}
	};

	template<model_config config, typename base_type_new, processing_phase phase> struct global_input_thread_function {
		NIHILUS_INLINE global_input_thread_function() noexcept												 = default;
		NIHILUS_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_input_thread_function(const global_input_thread_function&) noexcept			 = delete;
		NIHILUS_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept		 = delete;
		NIHILUS_INLINE global_input_thread_function(global_input_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		using return_type																					 = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_latch_types<base_type> && base_type::thread_strategy_type == thread_strategy_types::global_input && base_type::kernel_type != kernel_types::none;
		}
		NIHILUS_INLINE static bool impl(base_type& parse_core, int64_t thread_index_new, int64_t thread_count) {
			thread_count = parse_core.latch[0].thread_count;
			if constexpr (config.dev) {
				static_cast<thread_pool<config>*>(&parse_core)->perf_stats.collector[thread_index_new][base_type::core_type].start();
			}
			if (parse_core.latch[0].is_ready(thread_index_new)) {
				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[GLOBAL_INPUT] Thread: " << std::to_string(thread_index_new) << "/" << thread_count << " (ID: " << std::this_thread::get_id()
						   << "), for Op: " << base_type::core_type << " STARTED" << std::endl;
					log<log_levels::status>(stream.str());
				}

				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[GLOBAL_INPUT] Thread: " << std::to_string(thread_index_new) << "/" << thread_count << " (ID: " << std::this_thread::get_id()
						   << "), for Op: " << base_type::core_type << " EXECUTING Kernel" << std::endl;
					log<log_levels::status>(stream.str());
				}

				//kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count, 0);
				parse_core.latch[0].complete_work();
				if constexpr (config.dev) {
					const auto byte_count = type_traits<typename base_type::output_type>::total_byte_size(parse_core);
					std::stringstream stream{};
					stream << "[GLOBAL_INPUT] Thread: " << std::to_string(thread_index_new) << "/" << thread_count << " (ID: " << std::this_thread::get_id()
						   << "), for Op: " << base_type::core_type << " EXECUTING Kernel" << std::endl;
					log<log_levels::status>(stream.str());
					static_cast<thread_pool<config>*>(&parse_core)->perf_stats.collector[thread_index_new][base_type::core_type].end(byte_count);
				}


				if constexpr (config.dev) {
					std::stringstream stream{};
					//if (thread_index_new == 0) {
					//tensor_debugger<config>::compare_tensor_data(parse_core, 0, static_cast<thread_pool<config>*>(&parse_core)->perf_stats.current_iteration,
					//parse_core.get_mutable_dim());
					//}
					stream << "[GLOBAL_INPUT] Thread: " << std::to_string(thread_index_new) << "/" << thread_count << " (ID: " << std::this_thread::get_id()
						   << "), for Op: " << base_type::core_type << " COMPLETED succesfully" << std::endl;
					log<log_levels::status>(stream.str());
				}

				return true;
			} else {
				if constexpr (config.dev) {
					//std::stringstream stream{};
					//stream << "[GLOBAL_INPUT] Thread assignment FAILED - no work available" << std::endl;
					//log<log_levels::status>(stream.str());
				}
				return true;
			}
		}
	};

	template<model_config config, typename base_type_new, processing_phase phase> struct per_block_thread_function {
		NIHILUS_INLINE per_block_thread_function() noexcept											   = default;
		NIHILUS_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				   = delete;
		using base_type																				   = base_type_new;
		using return_type																			   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_latch_types<base_type> && base_type::thread_strategy_type == thread_strategy_types::per_block && base_type::kernel_type != kernel_types::none;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t current_block, int64_t thread_index_new, int64_t thread_count) {
			if constexpr (config.dev) {
				static_cast<thread_pool<config>*>(&parse_core)->perf_stats.collector[thread_index_new][base_type::core_type].start();
			}
			thread_count = parse_core.latch[0].thread_count;
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[PER_BLOCK] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << "/" << thread_count << " (ID: " << std::this_thread::get_id()
					   << "), for Op: " << base_type::core_type << " STARTED" << std::endl;
				log<log_levels::status>(stream.str());
			}
			if (parse_core.latch[current_block].is_ready(thread_index_new)) {
				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[PER_BLOCK] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << "/" << thread_count
						   << " (ID: " << std::this_thread::get_id() << "), for Op: " << base_type::core_type << " EXECUTING kernel" << std::endl;
					log<log_levels::status>(stream.str());
				}

				//kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count, current_block);

				parse_core.latch[current_block].complete_work();

				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[PER_BLOCK] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << "/" << thread_count
						   << " (ID: " << std::this_thread::get_id() << "), for Op: " << base_type::core_type << " COMPLETED succesfully" << std::endl;
					const auto byte_count = type_traits<typename base_type::output_type>::total_byte_size(parse_core);
					static_cast<thread_pool<config>*>(&parse_core)->perf_stats.collector[thread_index_new][base_type::core_type].end(byte_count);
				}
			}
		}
	};

	template<model_config config, has_chunk_types base_type_new, processing_phase phase> struct per_block_thread_function<config, base_type_new, phase> {
		NIHILUS_INLINE per_block_thread_function() noexcept											   = default;
		NIHILUS_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				   = delete;
		using base_type																				   = base_type_new;
		using return_type																			   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::per_block && base_type::kernel_type != kernel_types::none;
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t current_block, int64_t thread_index_new, int64_t thread_count) {
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[PER_BLOCK_BLOCKING] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << " (ID: " << std::this_thread::get_id()
					   << "), for Op: " << base_type::core_type << " ARRIVING at start barrier" << std::endl;
				log<log_levels::status>(stream.str());
				static_cast<thread_pool<config>*>(&parse_core)->perf_stats.collector[thread_index_new][base_type::core_type].start();
			}
			if constexpr (base_type::kernel_type == kernel_types::mul_mat) {
				if (thread_index_new == 0) {
					//parse_core.current_chunk[current_block].store(0);
				}
			}
			parse_core.latch[current_block].is_ready(thread_index_new);

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[PER_BLOCK_BLOCKING] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << "/" << thread_count
					   << " (ID: " << std::this_thread::get_id() << "), for Op: " << base_type::core_type << " EXECUTING kernel" << std::endl;
				log<log_levels::status>(stream.str());
			}

			//kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count, current_block);

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[PER_BLOCK_BLOCKING] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << "/" << thread_count
					   << " (ID: " << std::this_thread::get_id() << "), for Op: " << base_type::core_type << " ARRIVING at end barrier" << std::endl;
				log<log_levels::status>(stream.str());
			}

			parse_core.latch[current_block].complete_work();

			//parse_core.sync_flag_end.arrive_and_wait_get_thread();
			if constexpr (config.dev) {
				std::stringstream stream{};
				//if (thread_index_new == 0) {
				//tensor_debugger<config>::compare_tensor_data(parse_core, 0, static_cast<thread_pool<config>*>(&parse_core)->perf_stats.current_iteration,
				//parse_core.get_mutable_dim());
				//}
				stream << "[PER_BLOCK_BLOCKING] Block " << current_block << ", Thread: " << std::to_string(thread_index_new) << "/" << thread_count
					   << " (ID: " << std::this_thread::get_id() << "), for Op: " << base_type::core_type << " COMPLETED succesfully" << std::endl;
				log<log_levels::status>(stream.str());
				const auto byte_count = type_traits<typename base_type::output_type>::total_byte_size(parse_core);
				static_cast<thread_pool<config>*>(&parse_core)->perf_stats.collector[thread_index_new][base_type::core_type].end(byte_count);
			}
		}
	};

	template<model_config config, typename base_type_new, processing_phase phase> struct global_output_thread_function {
		NIHILUS_INLINE global_output_thread_function() noexcept												   = default;
		NIHILUS_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				   = delete;
		using base_type																						   = base_type_new;
		using return_type																					   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_latch_types<base_type> && base_type::thread_strategy_type == thread_strategy_types::global_output && base_type::kernel_type != kernel_types::none;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, int64_t thread_index_new, int64_t thread_count) {
			thread_count = parse_core.latch[0].thread_count;
			if (parse_core.latch[0].is_ready(thread_index_new)) {
				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[GLOBAL_OUTPUT] Thread for op: " << base_type::core_type << ", " << std::to_string(thread_index_new) << "/" << thread_count
						   << " (ID: " << std::this_thread::get_id() << ") STARTED" << std::endl;
					log<log_levels::status>(stream.str());
				}

				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[GLOBAL_OUTPUT] Thread for op: " << base_type::core_type << ", " << std::to_string(thread_index_new) << " EXECUTING kernel: " << base_type::core_type
						   << ", " << std::endl;
					log<log_levels::status>(stream.str());
				}

				//kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count, 0);

				parse_core.latch[0].complete_work();

				if constexpr (config.dev) {
					std::stringstream stream{};
					stream << "[GLOBAL_OUTPUT] Thread for op: " << base_type::core_type << ", " << std::to_string(thread_index_new) << " FINISHED kernel execution" << std::endl;
					log<log_levels::status>(stream.str());
				}

				if constexpr (config.dev) {
					//spinlock_nanoseconds(spinlock_time);
				}
			}
		}
	};

	template<model_config config, has_chunk_types base_type_new, processing_phase phase> struct global_output_thread_function<config, base_type_new, phase> {
		NIHILUS_INLINE global_output_thread_function() noexcept												   = default;
		NIHILUS_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				   = delete;
		using base_type																						   = base_type_new;
		using return_type																					   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::thread_strategy_type == thread_strategy_types::global_output && base_type::kernel_type != kernel_types::none;
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, int64_t thread_index_new, int64_t thread_count) {
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[GLOBAL_OUTPUT_BLOCKING] Thread (ID: " << std::this_thread::get_id() << ") ARRIVING at start barrier..." << std::endl;
				log<log_levels::status>(stream.str());
			}
			if constexpr (base_type::kernel_type == kernel_types::mul_mat) {
				if (thread_index_new == 0) {
					//parse_core.current_chunk[0].store(0);
				}
			}

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[GLOBAL_OUTPUT_BLOCKING] Thread for op: " << base_type::core_type << ", " << std::to_string(thread_index_new) << "/" << thread_count
					   << " (ID: " << std::this_thread::get_id() << ") PASSED start barrier, beginning execution..." << std::endl;
				log<log_levels::status>(stream.str());
			}

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[GLOBAL_OUTPUT_BLOCKING] Thread for op: " << base_type::core_type << ", " << std::to_string(thread_index_new)
					   << " EXECUTING kernel: " << base_type::core_type << ", " << std::endl;
				log<log_levels::status>(stream.str());
			}

			//kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count, 0);

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[GLOBAL_OUTPUT_BLOCKING] Thread for op: " << base_type::core_type << ", " << std::to_string(thread_index_new)
					   << " FINISHED kernel, arriving at end barrier..." << std::endl;
				log<log_levels::status>(stream.str());
			}

			if constexpr (config.dev) {
				//spinlock_nanoseconds(spinlock_time);
			}

			//parse_core.sync_flag_end.arrive_and_wait_get_thread();

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[GLOBAL_OUTPUT_BLOCKING] Thread " << std::to_string(thread_index_new) << " PASSED end barrier, operation complete!" << std::endl;
				log<log_levels::status>(stream.str());
			}
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
			return (config.dev && static_cast<uint64_t>(base_type::core_type) > 12);
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t current_block, uint64_t current_iteration_new) {
			uint64_t runtime_dim{};
			if constexpr (base_type::runtime_dims != 5) {
				runtime_dim = parse_core.get_mutable_dim();
			}
			tensor_debugger<config>::compare_tensor_data(parse_core, current_block, current_iteration_new, runtime_dim);
		}
	};*/

	template<typename value_type>
	concept final_output_type = std::remove_cvref_t<value_type>::type == op_types::ffn_inp_norm_out_ffn_norm || std::remove_cvref_t<value_type>::type == op_types::ffn_silu ||
		std::remove_cvref_t<value_type>::type == op_types::ffn_gate || std::remove_cvref_t<value_type>::type == op_types::ffn_up ||
		std::remove_cvref_t<value_type>::type == op_types::ffn_gate_par || std::remove_cvref_t<value_type>::type == op_types::ffn_out ||
		std::remove_cvref_t<value_type>::type == op_types::l_out_final_norm;

	template<typename core_traits_type, model_config config> static constexpr auto runtime_dims_multipliers{ []() {
		array<uint64_t, model_traits_type<config>::block_count> multipliers{};
		if constexpr (final_output_type<core_traits_type>) {
			for (uint64_t i = 0; i < model_traits_type<config>::block_count; ++i) {
				multipliers[i] = 1;
			}
			multipliers[model_traits_type<config>::block_count - 1] = 0;
		} else {
			multipliers.fill(1);
		}
		return multipliers;
	}() };

	template<model_config config, uint64_t current_index = 0> consteval array<uint64_t, core_types::count> get_depths_map(array<uint64_t, core_types::count> values = {}) {
		constexpr uint64_t max_index{ static_cast<uint64_t>(core_types::count) };
		if constexpr (current_index < max_index) {
			values[current_index] = core_traits<config, static_cast<core_types>(current_index)>::depth;
			return get_depths_map<config, current_index + 1>(values);
		} else {
			return values;
		}
	}

	template<model_config config> struct thread_pool;

	template<model_config config> struct memory_plan {
		uint64_t currently_allocated_bytes = 0;
		uint64_t peak_allocated_bytes	   = 0;
		array<uint64_t, core_types::count> allocations{};
		array<uint64_t, core_types::count> offsets{};
		array<bool, core_types::count> is_active{};
		static constexpr auto depths_map{ get_depths_map<config>() };
	};

	template<model_config config, typename base_type_new> struct total_bytes_collector {
		NIHILUS_INLINE total_bytes_collector() noexcept										   = default;
		NIHILUS_INLINE total_bytes_collector& operator=(const total_bytes_collector&) noexcept = delete;
		NIHILUS_INLINE total_bytes_collector(const total_bytes_collector&) noexcept			   = delete;
		NIHILUS_INLINE total_bytes_collector& operator=(total_bytes_collector&&) noexcept	   = delete;
		NIHILUS_INLINE total_bytes_collector(total_bytes_collector&&) noexcept				   = delete;

		using base_type = base_type_new;

		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::core_type != core_types::weights;
		}

		NIHILUS_INLINE static constexpr void impl(memory_plan<config>& mem_plan) {
			uint64_t current_depth = (mem_plan.depths_map[base_type::core_type] == std::numeric_limits<uint64_t>::max()) ? 0 : mem_plan.depths_map[base_type::core_type];
			if (current_depth == 0) {
				mem_plan.currently_allocated_bytes = 0;
				mem_plan.peak_allocated_bytes	   = 0;
			}

			if constexpr (base_type::depth != std::numeric_limits<uint64_t>::max() && base_type::depth >= 2) {
				uint64_t free_depth = mem_plan.depths_map[base_type::core_type] - 2;
				if (mem_plan.is_active[free_depth]) {
					mem_plan.currently_allocated_bytes -= mem_plan.allocations[free_depth];
					mem_plan.is_active[free_depth] = false;
				}
			}
			mem_plan.offsets[current_depth]			   = mem_plan.currently_allocated_bytes;
			mem_plan.allocations[current_depth]		 = base_type::total_required_bytes;
			mem_plan.is_active[current_depth]		   = true;
			mem_plan.currently_allocated_bytes += base_type::total_required_bytes;

			if (mem_plan.currently_allocated_bytes > mem_plan.peak_allocated_bytes) {
				mem_plan.peak_allocated_bytes = mem_plan.currently_allocated_bytes;
			}
		}
	};

	template<model_config config> struct core_bases_traits_type;

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
		if constexpr (kernel_type == kernel_types::get_rows || kernel_type == kernel_types::sub || kernel_type == kernel_types::view || kernel_type == kernel_types::none) {
			return 1;
		} else if constexpr (kernel_type == kernel_types::add || kernel_type == kernel_types::mul || kernel_type == kernel_types::mul_mat || kernel_type == kernel_types::none ||
			kernel_type == kernel_types::none || kernel_type == kernel_types::none || kernel_type == kernel_types::rope || kernel_type == kernel_types::silu ||
			kernel_type == kernel_types::softmax) {
			return static_cast<int64_t>(base_count);
		} else {
			return static_cast<int64_t>(base_count);
		}
	}

	template<model_config config> struct thread_pool;

	template<model_config config, typename base_type_new> struct execution_planner {
		NIHILUS_INLINE execution_planner() noexcept									   = default;
		NIHILUS_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_INLINE execution_planner(const execution_planner&) noexcept			   = delete;
		NIHILUS_INLINE execution_planner& operator=(execution_planner&&) noexcept	   = delete;
		NIHILUS_INLINE execution_planner(execution_planner&&) noexcept				   = delete;
		using base_type																   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_latch_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t thread_count) {
			if constexpr (array_types<decltype(parse_core.latch)>) {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					parse_core.latch[x].init(thread_count);
				}
			} else {
				parse_core.latch.init(thread_count);
			}
		}
	};

	template<model_config config, typename base_type> struct memory_mapper_impl {
		static constexpr uint64_t max_index{ static_cast<uint64_t>(base_type::enum_type::count) };
		template<uint64_t current_index = 0>
		NIHILUS_INLINE static void impl(base_type& value, const memory_plan<config>& plan, memory_buffer<config>& memory_buffer, uint64_t internal_offset = 0) {
			if constexpr (current_index < max_index) {
				auto& ref		= get<current_index>(value.values);
				using core_type = std::remove_cvref_t<decltype(ref)>;
				if constexpr (array_types<decltype(ref.data)>) {
					using data_type = std::remove_cvref_t<decltype(ref.data[0])>;
					data_type ptr	= static_cast<data_type>(memory_buffer.claim_memory(plan.offsets[base_type::core_type]));
					for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
						ref.data[x] = ptr;
					}
				} else {
					using data_type = std::remove_cvref_t<decltype(ref.data)>;
					data_type ptr	= static_cast<data_type>(memory_buffer.claim_memory(plan.offsets[base_type::core_type]));
					ref.data = ptr;
				}
				impl<current_index + 1>(value, plan, memory_buffer, internal_offset);
			}
		};
	};

	template<model_config config, typename base_type_new> struct memory_mapper {
		NIHILUS_INLINE memory_mapper() noexcept								   = default;
		NIHILUS_INLINE memory_mapper& operator=(const memory_mapper&) noexcept = delete;
		NIHILUS_INLINE memory_mapper(const memory_mapper&) noexcept			   = delete;
		NIHILUS_INLINE memory_mapper& operator=(memory_mapper&&) noexcept	   = delete;
		NIHILUS_INLINE memory_mapper(memory_mapper&&) noexcept				   = delete;
		using base_type														   = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_total_required_bytes_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core, const memory_plan<config>& plan, memory_buffer<config>& memory_buffer) {
			memory_mapper_impl<config, base_type>::impl(parse_core, plan, memory_buffer);
		}
	};

	template<model_config config, typename base_type_new> struct current_chunk_resetter {
		NIHILUS_INLINE current_chunk_resetter() noexcept										 = default;
		NIHILUS_INLINE current_chunk_resetter& operator=(const current_chunk_resetter&) noexcept = delete;
		NIHILUS_INLINE current_chunk_resetter(const current_chunk_resetter&) noexcept			 = delete;
		NIHILUS_INLINE current_chunk_resetter& operator=(current_chunk_resetter&&) noexcept		 = delete;
		NIHILUS_INLINE current_chunk_resetter(current_chunk_resetter&&) noexcept				 = delete;
		using base_type																			 = base_type_new;
		NIHILUS_INLINE static constexpr bool filter() {
			return has_chunk_types<base_type>;
		}
		NIHILUS_INLINE static void impl(base_type& parse_core) {
			for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
				if constexpr (array_types<decltype(parse_core.current_chunk)>) {
					parse_core.current_chunk[x].store(0);
				} else {
					parse_core.current_chunk.store(0);
				}
			}
		}
	};

	template<model_config config, typename core_traits_type> struct weight_mapper {
		template<weight_types weight_type, typename op_traits_type>
		NIHILUS_INLINE static void pack_weight_pointers_impl(op_traits_type& op, array<array<void*, model_traits_type<config>::block_count>, weight_types::count>& data) {
			if constexpr (array_types<decltype(op.data)>) {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					data[weight_type][x] = static_cast<void*>(&op.data[x]);
				}
			} else {
				data[weight_type][0] = static_cast<void*>(&op.data);
			}
		}

		NIHILUS_INLINE static void impl(core_traits_type& core_traits, array<array<void*, model_traits_type<config>::block_count>, weight_types::count>& data) {
			pack_weight_pointers_impl<weight_types::attn_q>(get<weight_types::attn_q>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::attn_k>(get<weight_types::attn_k>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::attn_v>(get<weight_types::attn_v>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::attn_output>(get<weight_types::attn_output>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::attn_norm>(get<weight_types::attn_norm>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::ffn_gate>(get<weight_types::ffn_gate>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::ffn_up>(get<weight_types::ffn_up>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::ffn_down>(get<weight_types::ffn_down>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::ffn_norm>(get<weight_types::ffn_norm>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::token_embd>(get<weight_types::token_embd>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::rope_freqs>(get<weight_types::rope_freqs>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::output_norm>(get<weight_types::output_norm>(core_traits.values), data);
			pack_weight_pointers_impl<weight_types::output>(get<weight_types::output>(core_traits.values), data);
		};
	};

	template<model_config config, typename base_type_new, processing_phase phase> struct global_input_thread_function {
		NIHILUS_INLINE global_input_thread_function() noexcept										   = default;
		NIHILUS_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_input_thread_function(const global_input_thread_function&) noexcept		  = delete;
		NIHILUS_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept	  = delete;
		NIHILUS_INLINE global_input_thread_function(global_input_thread_function&&) noexcept			  = delete;
		using base_type																				   = base_type_new;
		using return_type																			   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::core_type == core_types::token_embeddings;
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, int64_t thread_index_new, int64_t thread_count) {
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << thread_index_new << " [STARTING] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
			parse_core.latch.arrive_and_wait();
			kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count);

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << thread_index_new << " [FINISHED] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
		}
	};

	template<model_config config, typename base_type_new, processing_phase phase> struct per_block_thread_function {
		NIHILUS_INLINE per_block_thread_function() noexcept											   = default;
		NIHILUS_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				   = delete;
		using base_type																				   = base_type_new;
		using return_type																			   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::core_type != core_types::weights && base_type::core_type != core_types::global_inputs && base_type::core_type != core_types::final_norm_and_sampling &&
				base_type::core_type != core_types::token_embeddings;
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, uint64_t current_block, int64_t thread_index_new, int64_t thread_count) {
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << thread_index_new << " [STARTING] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << ", for [BLOCK]: " << current_block << std::endl;
				log<log_levels::status>(stream.str());
			}
			parse_core.latch[current_block].arrive_and_wait();
			kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count, current_block);

			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << thread_index_new << " [FINISHED] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << ", for [BLOCK]: " << current_block << std::endl;
				log<log_levels::status>(stream.str());
			}
		}
	};

	template<model_config config, typename base_type_new, processing_phase phase> struct global_output_thread_function {
		NIHILUS_INLINE global_output_thread_function() noexcept												   = default;
		NIHILUS_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			   = delete;
		NIHILUS_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept	   = delete;
		NIHILUS_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				   = delete;
		using base_type																						   = base_type_new;
		using return_type																					   = bool;
		NIHILUS_INLINE static constexpr bool filter() {
			return base_type::core_type == core_types::final_norm_and_sampling;
		}

		NIHILUS_INLINE static void impl(base_type& parse_core, int64_t thread_index_new, int64_t thread_count) {
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << thread_index_new << " [STARTING] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
			parse_core.latch.arrive_and_wait();

			kernel_dispatcher<config, phase, device_types::cpu, base_type>::impl(parse_core, thread_index_new, thread_count);
			if constexpr (config.dev) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << thread_index_new << " [FINISHED] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
		}
	};

}