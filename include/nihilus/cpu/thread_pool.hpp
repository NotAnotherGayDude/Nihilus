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
#include <nihilus/common/core_bases.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/tuple.hpp>
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

	NIHILUS_INLINE void reset_current_thread_priority() {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_NORMAL)) {
			std::cerr << "Failed to reset thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif defined(NIHILUS_PLATFORM_LINUX) || defined(NIHILUS_PLATFORM_MAC)
		pthread_t this_thread = pthread_self();

		sched_param sch_params;
		int32_t policy;
		if (pthread_getschedparam(this_thread, &policy, &sch_params) != 0) {
			std::cerr << "Failed to get thread sched param: " << strerror(errno) << std::endl;
			return;
		}

		int32_t min_priority = sched_get_priority_min(policy);
		int32_t max_priority = sched_get_priority_max(policy);
		if (min_priority == -1 || max_priority == -1) {
			std::cerr << "Failed to get detail::min/max priority: " << strerror(errno) << std::endl;
			return;
		}

		sch_params.sched_priority = (min_priority + max_priority) / 2;

		if (pthread_setschedparam(this_thread, policy, &sch_params) != 0) {
			std::cerr << "Failed to reset thread priority: " << strerror(errno) << std::endl;
		}
#else
	#warning "Thread priority adjustment not supported on this platform."
#endif
	};

	template<model_config config, typename model_type> struct thread_pool : public get_core_bases_t<config> {
		using core_base_type											   = get_core_bases_t<config>;
		NIHILUS_INLINE thread_pool() noexcept							   = default;
		NIHILUS_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
		NIHILUS_INLINE thread_pool(const thread_pool&) noexcept			   = delete;

		NIHILUS_INLINE thread_pool(uint64_t thread_count_new) {
			threads.resize(thread_count_new);
			thread_count = thread_count_new;
			thread_latch.init(thread_count_new);
			for (uint64_t x = 0; x < thread_count_new; ++x) {
				threads[x] = std::thread{ [&, x] {
					thread_function<false>(x);
				} };
			}
		}

		template<bool raise_priority> NIHILUS_INLINE void thread_function(uint64_t thread_index) {
			while (!stop.load(std::memory_order_acquire)) {
				thread_latch.worker_wait(thread_index);
				if (!stop.load(std::memory_order_acquire)) {
					core_base_type::template impl<global_input_thread_function>();
					for (uint64_t x = 0; x < model_type::model_traits_type::block_count; ++x) {
						core_base_type::template impl<per_block_thread_function>(x);
					}
					core_base_type::template impl<global_output_thread_function>();
					thread_latch.arrive_and_wait(thread_index);
				}
			}
		}

		NIHILUS_INLINE void execute_tasks() {
			core_base_type::template impl<execution_planner>(thread_count);
			depths.store(0, std::memory_order_release);
			thread_latch.count_down();
			thread_latch.main_wait();
		}

		NIHILUS_INLINE ~thread_pool() {
			stop.store(true, std::memory_order_release);
			thread_latch.count_down();
			for (auto& value: threads) {
				if (value.joinable()) {
					value.join();
				}
			}
		};

	  protected:
		std::vector<std::thread> threads{};
		char padding[32]{};
		alignas(64) std::atomic_bool stop{};
		char padding02[63]{};
		alignas(64) uint64_t thread_count{};
		main_gate_latch thread_latch;
	};

}