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
#include <nihilus-incl/common/core_bases.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	NIHILUS_INLINE bool pin_thread_to_core(int32_t core_id) {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		DWORD_PTR mask	 = 1ULL << core_id;
		HANDLE thread	 = GetCurrentThread();
		DWORD_PTR result = SetThreadAffinityMask(thread, mask);
		if (result == 0) {
			std::cerr << "Failed to set thread affinity on Windows. Error: " << GetLastError() << std::endl;
			return false;
		}
		return true;

#elif defined(NIHILUS_PLATFORM_LINUX)
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(core_id, &cpuset);

		pthread_t current_thread = pthread_self();
		int32_t result			 = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
		if (result != 0) {
			std::cerr << "Failed to set thread affinity on Linux. Error: " << result << std::endl;
			return false;
		}
		return true;

#elif defined(NIHILUS_PLATFORM_MAC)
		thread_port_t thread				 = mach_thread_self();
		thread_affinity_policy_data_t policy = { core_id };
		kern_return_t result				 = thread_policy_set(thread, THREAD_AFFINITY_POLICY, ( thread_policy_t )&policy, 1);
		mach_port_deallocate(mach_task_self(), thread);
		if (result != KERN_SUCCESS) {
			std::cerr << "Failed to set thread affinity on macOS. Error: " << result << std::endl;
			return false;
		}
		return true;

#else
		std::cerr << "Thread pinning is not supported on this platform." << std::endl;
		return false;
#endif
	}

	NIHILUS_INLINE void raise_current_thread_priority() {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST)) {
			std::cerr << "Failed to set thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif defined(NIHILUS_PLATFORM_LINUX) || defined(NIHILUS_PLATFORM_MAC)
		pthread_t this_thread = pthread_self();

		sched_param sch_params;
		sch_params.sched_priority = 0;

		int32_t policy;
		if (pthread_getschedparam(this_thread, &policy, &sch_params) != 0) {
			std::cerr << "Failed to get thread sched param: " << strerror(errno) << std::endl;
			return;
		}

		int32_t max_priority = sched_get_priority_max(policy);
		if (max_priority == -1) {
			std::cerr << "Failed to get detail::max thread priority: " << strerror(errno) << std::endl;
			return;
		}

		sch_params.sched_priority = max_priority;

		if (pthread_setschedparam(this_thread, policy, &sch_params) != 0) {
			std::cerr << "Failed to set thread priority: " << strerror(errno) << std::endl;
		}
#else
	#warning "Thread priority adjustment not supported on this platform."
#endif
	}

	struct benchmark_stats {
		std::chrono::high_resolution_clock::time_point sampling_start = {};
		std::chrono::high_resolution_clock::time_point prompt_start	  = {};
		std::chrono::high_resolution_clock::time_point token_start	  = {};
		std::chrono::high_resolution_clock::time_point eval_start	  = {};
		std::chrono::high_resolution_clock::time_point load_start	  = {};
		double total_load_time_ns									  = 0;
		double total_prompt_eval_time_ns							  = 0;
		double total_eval_time_ns									  = 0;
		double total_sampling_time_ns								  = 0;
		int32_t prompt_token_count									  = 0;
		int32_t generated_token_count								  = 0;
		int32_t total_sampling_runs									  = 0;
		uint64_t current_iteration									  = 0;
		op_latch debug_counter{};
	};

	template<model_config config> struct perf_base {};

	template<model_config config>
		requires(config.dev || config.benchmark)
	struct perf_base<config> {
		benchmark_stats perf_stats{};
	};

	template<model_config config, typename model_type> struct thread_pool : public get_core_bases_t<config>, public perf_base<config> {
		using core_base_type											   = get_core_bases_t<config>;
		NIHILUS_INLINE thread_pool() noexcept							   = default;
		NIHILUS_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
		NIHILUS_INLINE thread_pool(const thread_pool&) noexcept			   = delete;

		static constexpr array<uint64_t, config.max_thread_count> p_core_ids = [] {
			array<uint64_t, config.max_thread_count> return_value{};
			constexpr uint64_t max_p_cores = 8;
			for (size_t x = 0; x < config.max_thread_count; ++x) {
				return_value[x] = x / 2;
			}

			return return_value;
		}();

		using ThreadFunctionPtr = void (thread_pool::*)(uint64_t);

		template<size_t index = 0>
		static constexpr array<ThreadFunctionPtr, config.max_thread_count> generate_function_ptrs(array<ThreadFunctionPtr, config.max_thread_count> values = {}) {
			if constexpr (index < config.max_thread_count) {
				if (index % 2 == 0) {
					if (index % 3 == 0) {
						values[index] = &thread_pool::thread_function<true, true>;
					} else {
						values[index] = &thread_pool::thread_function<true, false>;
					}
				} else {
					if (index % 3 == 0) {
						values[index] = &thread_pool::thread_function<false, true>;
					} else {
						values[index] = &thread_pool::thread_function<false, false>;
					}
				}
				return generate_function_ptrs<index + 1>(values);
			}
			return values;
		}

		NIHILUS_INLINE thread_pool(int64_t thread_count_new) {
			thread_count = static_cast<uint64_t>(thread_count_new);
			runtime_dims.resize(static_cast<uint64_t>(thread_count));
			threads.resize(static_cast<uint64_t>(thread_count));
			thread_latch.init(thread_count_new);
			if constexpr (config.dev) {
				perf_base<config>::perf_stats.debug_counter.init(thread_count);
			}
			static constexpr auto function_ptrs = generate_function_ptrs();

			for (uint64_t x = 0; x < static_cast<uint64_t>(thread_count); ++x) {
				threads[x] = std::thread{ function_ptrs[x], this, x };
			}
		}

		template<bool pin_to_core = false, bool raise_priority = false> NIHILUS_INLINE void thread_function(uint64_t thread_index) {
			if constexpr (pin_to_core) {
				//int32_t core_id = p_core_ids[thread_index % p_core_ids.size()];
				//pin_thread_to_core(core_id);
			}
			if constexpr (raise_priority) {
				//raise_current_thread_priority();
			}
			while (!stop.load()) {
				thread_latch.worker_wait(thread_index);
				if (!stop.load()) {
					core_base_type::template impl<global_input_thread_function>(runtime_dims[thread_index]);
					for (uint64_t x = 0; x < model_type::model_traits_type::block_count; ++x) {
						core_base_type::template impl<per_block_thread_function>(x, runtime_dims[thread_index]);
						if constexpr (config.dev) {
							perf_base<config>::perf_stats.debug_counter.arrive_and_wait();
							if (thread_index == 0) {
								core_base_type::template impl<tensor_debugger_impl>(x, perf_base<config>::perf_stats.current_iteration, runtime_dims[thread_index]);
							}
						}
					}
					core_base_type::template impl<global_output_thread_function>(runtime_dims[thread_index]);
					thread_latch.arrive_and_wait(thread_index);
				}
			}
		}

		NIHILUS_INLINE void execute_tasks(uint64_t runtime_dims_new) {
			core_base_type::template impl<execution_planner>(thread_count);
			depths.store(0, std::memory_order_release);
			for (uint64_t x = 0; x < threads.size(); ++x) {
				runtime_dims[x] = runtime_dims_new;
			}
			thread_latch.count_down();
			thread_latch.main_wait();
		}

		NIHILUS_INLINE ~thread_pool() {
			stop.store(true);
			thread_latch.count_down();
			for (auto& value: threads) {
				if (value.joinable()) {
					value.join();
				}
			}
		};

	  protected:
		vector<uint64_t> runtime_dims{};
		main_gate_latch thread_latch{};
		vector<std::thread> threads{};
		atomic_flag_wrapper stop{};
		uint64_t thread_count{};
	};

}