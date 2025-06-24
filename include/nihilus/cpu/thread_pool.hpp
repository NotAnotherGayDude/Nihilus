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
#include <nihilus/cpu/thread_strategy.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	NIHILUS_FORCE_INLINE bool pin_thread_to_core(int core_id) {
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
		int result				 = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
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

	NIHILUS_FORCE_INLINE void raise_current_thread_priority() {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST)) {
			std::cerr << "Failed to set thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif defined(NIHILUS_PLATFORM_LINUX) || defined(NIHILUS_PLATFORM_MAC)
		pthread_t this_thread = pthread_self();

		sched_param sch_params;
		sch_params.sched_priority = 0;

		int policy;
		if (pthread_getschedparam(this_thread, &policy, &sch_params) != 0) {
			std::cerr << "Failed to get thread sched param: " << strerror(errno) << std::endl;
			return;
		}

		int max_priority = sched_get_priority_max(policy);
		if (max_priority == -1) {
			std::cerr << "Failed to get max thread priority: " << strerror(errno) << std::endl;
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

	NIHILUS_FORCE_INLINE void reset_current_thread_priority() {
#if defined(NIHILUS_PLATFORM_WINDOWS)
		HANDLE thread = GetCurrentThread();
		if (!SetThreadPriority(thread, THREAD_PRIORITY_NORMAL)) {
			std::cerr << "Failed to reset thread priority on Windows. Error: " << GetLastError() << std::endl;
		}
#elif defined(NIHILUS_PLATFORM_LINUX) || defined(NIHILUS_PLATFORM_MAC)
		pthread_t this_thread = pthread_self();

		sched_param sch_params;
		int policy;
		if (pthread_getschedparam(this_thread, &policy, &sch_params) != 0) {
			std::cerr << "Failed to get thread sched param: " << strerror(errno) << std::endl;
			return;
		}

		int min_priority = sched_get_priority_min(policy);
		int max_priority = sched_get_priority_max(policy);
		if (min_priority == -1 || max_priority == -1) {
			std::cerr << "Failed to get min/max priority: " << strerror(errno) << std::endl;
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

#if defined(NIHILUS_PLATFORM_WINDOWS)

	#include <windows.h>
#endif

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
		NIHILUS_FORCE_INLINE static void impl(base_type_new& core, nihilus::array<nihilus::array<void*, model_traits_type::block_count>, op_type_type::count>& data) {
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

	template<nihilus::model_config config, typename base_type_new> struct global_input_thread_function : public base_type_new {
		NIHILUS_FORCE_INLINE global_input_thread_function() noexcept											   = default;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(const global_input_thread_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(global_input_thread_function&&) noexcept				   = delete;
		using base_type																							   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_input && core_traits_type::krn_type != nihilus::kernel_type::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			nihilus::kernel_dispatcher<config, nihilus::device_type::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
		}
	};

	template<nihilus::model_config config, nihilus::blocking base_type_new> struct global_input_thread_function<config, base_type_new> : public base_type_new {
		NIHILUS_FORCE_INLINE global_input_thread_function() noexcept											   = default;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(const global_input_thread_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function& operator=(global_input_thread_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE global_input_thread_function(global_input_thread_function&&) noexcept				   = delete;
		using base_type																							   = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_input && core_traits_type::krn_type != nihilus::kernel_type::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			core.sync_flag_start[0].arrive_and_wait(thread_index);
			nihilus::kernel_dispatcher<config, nihilus::device_type::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
			core.sync_flag_end[0].arrive_and_wait(thread_index);
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct per_block_thread_function : public base_type_new {
		NIHILUS_FORCE_INLINE per_block_thread_function() noexcept											 = default;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::per_block && core_traits_type::krn_type != nihilus::kernel_type::transpose &&
				core_traits_type::krn_type != nihilus::kernel_type::view && core_traits_type::krn_type != nihilus::kernel_type::permute &&
				core_traits_type::krn_type != nihilus::kernel_type::reshape && core_traits_type::krn_type != nihilus::kernel_type::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count, uint64_t current_block) {
			nihilus::kernel_dispatcher<config, nihilus::device_type::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
		}
	};

	template<nihilus::model_config config, nihilus::blocking base_type_new> struct per_block_thread_function<config, base_type_new> : public base_type_new {
		NIHILUS_FORCE_INLINE per_block_thread_function() noexcept											 = default;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(const per_block_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function& operator=(per_block_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE per_block_thread_function(per_block_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::per_block && core_traits_type::krn_type != nihilus::kernel_type::transpose &&
				core_traits_type::krn_type != nihilus::kernel_type::view && core_traits_type::krn_type != nihilus::kernel_type::permute &&
				core_traits_type::krn_type != nihilus::kernel_type::reshape && core_traits_type::krn_type != nihilus::kernel_type::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count, uint64_t current_block) {
			core.sync_flag_start[current_block].arrive_and_wait(thread_index);
			nihilus::kernel_dispatcher<config, nihilus::device_type::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
			core.sync_flag_end[current_block].arrive_and_wait(thread_index);
		}
	};

	template<nihilus::model_config config, typename base_type_new> struct global_output_thread_function : public base_type_new {
		NIHILUS_FORCE_INLINE global_output_thread_function() noexcept												 = default;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				 = delete;
		using base_type																								 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_output && core_traits_type::krn_type != nihilus::kernel_type::transpose &&
				core_traits_type::krn_type != nihilus::kernel_type::view && core_traits_type::krn_type != nihilus::kernel_type::permute &&
				core_traits_type::krn_type != nihilus::kernel_type::reshape && core_traits_type::krn_type != nihilus::kernel_type::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			nihilus::kernel_dispatcher<config, nihilus::device_type::cpu, base_type>::impl(core, thread_index, thread_count);
			nihilus::spinlock_nanoseconds(spinlock_time);
		}
	};

	template<nihilus::model_config config, nihilus::blocking base_type_new> struct global_output_thread_function<config, base_type_new> : public base_type_new {
		NIHILUS_FORCE_INLINE global_output_thread_function() noexcept												 = default;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(const global_output_thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function& operator=(global_output_thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE global_output_thread_function(global_output_thread_function&&) noexcept				 = delete;
		using base_type																								 = base_type_new;
		NIHILUS_FORCE_INLINE static constexpr bool filter() {
			using core_traits_type = base_type;
			return core_traits_type::layer_type == nihilus::thread_strategy_type::global_output && core_traits_type::krn_type != nihilus::kernel_type::transpose &&
				core_traits_type::krn_type != nihilus::kernel_type::view && core_traits_type::krn_type != nihilus::kernel_type::permute &&
				core_traits_type::krn_type != nihilus::kernel_type::reshape && core_traits_type::krn_type != nihilus::kernel_type::none;
		}

		NIHILUS_FORCE_INLINE static void impl(base_type& core, uint64_t thread_index, uint64_t thread_count) {
			core.sync_flag_start[0].arrive_and_wait(thread_index);
			nihilus::kernel_dispatcher<config, nihilus::device_type::cpu, base_type>::impl(core, thread_index, thread_count);
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
				core_traits_type::layer_type == nihilus::thread_strategy_type::per_block && core_traits_type::krn_type != nihilus::kernel_type::reshape &&
				core_traits_type::krn_type != nihilus::kernel_type::transpose && core_traits_type::krn_type != nihilus::kernel_type::view &&
				core_traits_type::krn_type != nihilus::kernel_type::permute && core_traits_type::krn_type != nihilus::kernel_type::none;
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
				core_traits_type::krn_type != nihilus::kernel_type::transpose && core_traits_type::krn_type != nihilus::kernel_type::view &&
				core_traits_type::krn_type != nihilus::kernel_type::permute && core_traits_type::krn_type != nihilus::kernel_type::reshape &&
				core_traits_type::krn_type != nihilus::kernel_type::none;
		}
		NIHILUS_FORCE_INLINE static void impl(base_type& core) {
			core.sync_flag_start[0].main_wait();
			core.sync_flag_end[0].main_wait();
		}
	};

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
					nihilus::tensor_debugger::compare_tensor_data(core, x);
				}
			} else {
				nihilus::tensor_debugger::compare_tensor_data(core, 0);
			}
		}
	};

	template<nihilus::model_config config, typename model_type> struct thread_pool : public get_core_bases_t<config> {
		using core_base_type													 = get_core_bases_t<config>;
		NIHILUS_FORCE_INLINE thread_pool() noexcept								 = default;
		NIHILUS_FORCE_INLINE thread_pool& operator=(const thread_pool&) noexcept = delete;
		NIHILUS_FORCE_INLINE thread_pool(const thread_pool&) noexcept			 = delete;

		NIHILUS_FORCE_INLINE thread_pool(uint64_t thread_count_new) {
			threads.resize(thread_count_new);
			thread_count = thread_count_new;
			thread_latch.init(thread_count_new);
			for (uint64_t x = 0; x < thread_count_new; ++x) {
				threads[x] = std::thread{ [&, x] {
					thread_function_impl<false>(x);
				} };
			}
			core_base_type::template impl<execution_planner>(thread_count);
		}

		template<bool raise_priority> NIHILUS_FORCE_INLINE void thread_function_impl(uint64_t thread_index) {
			while (!stop.load(std::memory_order_acquire)) {
				thread_latch.worker_wait(thread_index);
				if (!stop.load(std::memory_order_acquire)) {
					core_base_type::template impl<global_input_thread_function>(thread_index, thread_count);
					for (uint64_t x = 0; x < model_type::model_traits_type::block_count; ++x) {
						core_base_type::template impl<per_block_thread_function>(thread_index, thread_count, x);
					}
					core_base_type::template impl<global_output_thread_function>(thread_index, thread_count);
					thread_latch.arrive_and_wait(thread_index);
				}
			}
		}

		NIHILUS_FORCE_INLINE void execute_tasks() {
			nihilus::depths.store(0, std::memory_order_release);
			thread_latch.count_down();
			for (uint64_t x = 0; x < model_type::model_traits_type::block_count; ++x) {
				core_base_type::template impl<main_thread_per_block_function>(x);
			}
			core_base_type::template impl<main_thread_global_output_function>();
			thread_latch.main_wait();
		}

		NIHILUS_FORCE_INLINE ~thread_pool() {
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
		nihilus::op_latch thread_latch;
	};

}