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

#include <nihilus-incl/infra/monolithic_dispatcher.hpp>
#include <nihilus-incl/cpu/nihilus_cpu_properties.hpp>
#include <nihilus-incl/infra/core_bases.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	NIHILUS_HOST bool pin_thread_to_core(uint64_t core_id) {
#if NIHILUS_PLATFORM_WINDOWS
		DWORD_PTR mask	 = 1ULL << core_id;
		HANDLE thread	 = GetCurrentThread();
		DWORD_PTR result = SetThreadAffinityMask(thread, mask);
		if (result == 0) {
			std::cerr << "Failed to set thread affinity on Windows. Error: " << GetLastError() << std::endl;
			return false;
		}
		return true;

#elif NIHILUS_PLATFORM_LINUX
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

#elif NIHILUS_PLATFORM_MAC
		thread_port_t thread				 = mach_thread_self();
		thread_affinity_policy_data_t policy = { static_cast<int32_t>(core_id) };

		kern_return_t result = thread_policy_set(thread, THREAD_AFFINITY_POLICY, std::bit_cast<thread_policy_t>(&policy), THREAD_AFFINITY_POLICY_COUNT);

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

	NIHILUS_HOST void raise_current_thread_priority() {
#if NIHILUS_PLATFORM_WINDOWS
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST)) {
			std::cerr << "Failed to set thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif NIHILUS_PLATFORM_LINUX || NIHILUS_PLATFORM_MAC
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
		array<array<benchmarking::event_collector, core_types::count>, cpu_properties::thread_count> collector{};
		clock_type::time_point sampling_start		= {};
		clock_type::time_point prompt_start			= {};
		clock_type::time_point token_start			= {};
		clock_type::time_point eval_start			= {};
		clock_type::time_point load_start			= {};
		double total_prompt_eval_time_ns			= {};
		double total_sampling_time_ns				= {};
		double total_eval_time_ns					= {};
		double total_load_time_ns					= {};
		uint64_t generated_token_count				= {};
		uint64_t prompt_token_count					= {};
		uint64_t total_sampling_runs				= {};
		uint64_t current_iteration					= {};
		aligned_vector<uint64_t> runtime_dimensions = {};
	};

	template<const model_config& config> struct perf_base {};

	template<const model_config& config>
		requires(config.benchmark || config.dev)
	struct perf_base<config> {
		benchmark_stats perf_stats{};
	};

	template<const model_config& config> struct thread_pool : public get_core_bases_t<config, core_types>, public perf_base<config> {
		using core_bases_type											   = get_core_bases_t<config, core_types>;
		NIHILUS_HOST thread_pool() noexcept							   = default;
		NIHILUS_HOST thread_pool& operator=(const thread_pool&) noexcept = delete;
		NIHILUS_HOST thread_pool(const thread_pool&) noexcept			   = delete;

		NIHILUS_HOST thread_pool(int64_t thread_count_new) {
			thread_count = thread_count_new;
			threads.resize(static_cast<uint64_t>(thread_count));
			thread_latch.init(static_cast<typename main_gate_latch::value_type>(thread_count_new));
			if constexpr (config.benchmark ) {
				perf_base<config>::perf_stats.runtime_dimensions.resize(static_cast<uint64_t>(thread_count));
			}

			for (int64_t x = 0; x < thread_count; ++x) {
				threads[x] = std::thread{ &thread_pool::thread_function, this, x };
			}
		}

		template<processing_phases processing_phase, size_t... indices> NIHILUS_HOST void execute_blocks(std::index_sequence<indices...>) {
			(core_bases_type::template impl_thread<per_block_thread_function, processing_phase>(static_cast<int64_t>(indices), thread_count), ...);
		}

		NIHILUS_HOST void thread_function(int64_t thread_index) {
			if (thread_index % 4 == 0 && (thread_index < static_cast<int64_t>(cpu_properties::thread_count) / 3)) {
				//raise_current_thread_priority();
			}
			while (!stop.load()) {
				thread_latch.worker_wait(static_cast<typename main_gate_latch::value_type>(thread_index));
				if (!stop.load()) {
					if (processing_phase.load() == processing_phases::prompt_eval_time) {
						core_bases_type::template impl_thread<global_input_thread_function, processing_phases::prompt_eval_time>(thread_count);
						execute_blocks<processing_phases::prompt_eval_time>(std::make_index_sequence<static_cast<size_t>(model_traits_type<config>::block_count)>{});
						core_bases_type::template impl_thread<global_output_thread_function, processing_phases::prompt_eval_time>(thread_count);
					} else {
						core_bases_type::template impl_thread<global_input_thread_function, processing_phases::eval_time>(thread_count);
						execute_blocks<processing_phases::eval_time>(std::make_index_sequence<static_cast<size_t>(model_traits_type<config>::block_count)>{});
						core_bases_type::template impl_thread<global_output_thread_function, processing_phases::eval_time>(thread_count);
					}
					thread_latch.arrive();
				}
			}
		}

		template<processing_phases phase_new> NIHILUS_HOST void execute_tasks(uint64_t runtime_dimensions_new) {
			processing_phase.store(phase_new);
			core_bases_type::template impl<sync_resetter>(thread_count);
			core_bases_type::template impl<dim_updater>(runtime_dimensions_new);
			if constexpr (config.benchmark ) {
				for (uint64_t x = 0; x < threads.size(); ++x) {
					perf_base<config>::perf_stats.runtime_dimensions[x] = runtime_dimensions_new;
				}
			}
			thread_latch.count_down();
			thread_latch.main_wait();
		}

	  protected:
		atomic_flag_wrapper<processing_phases> processing_phase{};
		aligned_vector<std::thread> threads{};
		atomic_flag_wrapper<bool> stop{};
		main_gate_latch thread_latch{};
		int64_t thread_count{};

		NIHILUS_HOST ~thread_pool() {
			stop.store(true);
			thread_latch.count_down();
			for (auto& value: threads) {
				if (value.joinable()) {
					value.join();
				}
			}
		}
	};


}
