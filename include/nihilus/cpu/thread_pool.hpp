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

	template<typename base_type_new> struct execution_planner : public base_type_new {
		NIHILUS_FORCE_INLINE execution_planner() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_planner(const execution_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_planner& operator=(execution_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_planner(execution_planner&&) noexcept				 = delete;
		using output_type																	 = base_type_new::output_type;
		using base_type																		 = base_type_new;
		using model_traits_type																 = typename base_type::model_traits_type;
		using op_type_type																	 = typename model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE void impl(uint64_t, array<array<void*, model_traits_type::block_count>, op_type_type::count>& data) {
			if constexpr (array_type<decltype(this->data)>) {
				for (size_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&this->data[x]);
				}
			} else {
				for (size_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&this->data);
				}
			}
		}
	};

	template<blocking base_type_new> struct execution_planner<base_type_new> : public base_type_new {
		NIHILUS_FORCE_INLINE execution_planner() noexcept									 = default;
		NIHILUS_FORCE_INLINE execution_planner& operator=(const execution_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE execution_planner(const execution_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE execution_planner& operator=(execution_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE execution_planner(execution_planner&&) noexcept				 = delete;
		using output_type																	 = base_type_new::output_type;
		using base_type																		 = base_type_new;
		using model_traits_type																 = typename base_type::model_traits_type;
		using op_type_type																	 = typename model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE void impl(uint64_t thread_count, array<array<void*, model_traits_type::block_count>, op_type_type::count>& data) {
			if constexpr (array_type<decltype(this->data)>) {
				for (size_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&this->data[x]);
				}
			} else {
				for (size_t x = 0; x < model_traits_type::block_count; ++x) {
					data[base_type::type][x] = reinterpret_cast<void*>(&this->data);
				}
			}
			for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
				this->sync_flag_start[x].init(thread_count);
				this->sync_flag_end[x].init(thread_count);
			}
		}
	};

	inline std::atomic_size_t depths_global{};
	inline std::atomic_size_t depths{};
	inline std::atomic_size_t count{};

	template<typename base_type_new> struct constexpr_execution_planner {
		NIHILUS_FORCE_INLINE constexpr_execution_planner() noexcept												 = default;
		NIHILUS_FORCE_INLINE constexpr_execution_planner& operator=(const constexpr_execution_planner&) noexcept = delete;
		NIHILUS_FORCE_INLINE constexpr_execution_planner(const constexpr_execution_planner&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE constexpr_execution_planner& operator=(constexpr_execution_planner&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE constexpr_execution_planner(constexpr_execution_planner&&) noexcept				 = delete;
		using output_type																						 = base_type_new::output_type;
		using base_type																							 = base_type_new;
		using op_type_type																						 = base_type_new::model_traits_type::op_type_type;
		NIHILUS_FORCE_INLINE constexpr static void impl(uint64_t& count_new, layer_op_type layer_type) {
			if ((base_type::layer_type == layer_type && base_type::krn_type != kernel_type::permute && base_type::krn_type != kernel_type::reshape &&
					base_type::krn_type != kernel_type::transpose && base_type::krn_type != kernel_type::view) &&
				base_type::krn_type != kernel_type::none) {
				count_new += base_type::layer_type == layer_type;
			}
		}
		template<uint64_t size> NIHILUS_FORCE_INLINE constexpr static void impl(array<op_type_type, size>& value, uint64_t& current_index, layer_op_type layer_type) {
			if ((base_type::layer_type == layer_type && base_type::krn_type != kernel_type::permute && base_type::krn_type != kernel_type::reshape &&
					base_type::krn_type != kernel_type::transpose && base_type::krn_type != kernel_type::view) &&
				base_type::krn_type != kernel_type::none) {
				value[current_index] = base_type::type;
				++current_index;
			}
		}
	};

	template<typename base_type> struct tensor_debugger_impl {
		NIHILUS_FORCE_INLINE tensor_debugger_impl() noexcept									   = default;
		NIHILUS_FORCE_INLINE tensor_debugger_impl& operator=(const tensor_debugger_impl&) noexcept = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl(const tensor_debugger_impl&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl& operator=(tensor_debugger_impl&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE tensor_debugger_impl(tensor_debugger_impl&&) noexcept				   = delete;
		using output_type																		   = base_type::output_type;
		NIHILUS_FORCE_INLINE static void impl(base_type& core) {
			if constexpr (array_type<decltype(core.data)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					tensor_debugger::compare_tensor_data(core, x);
				}
			} else {
				tensor_debugger::compare_tensor_data(core, 0);
			}
		}
	};

	template<typename base_type> struct update_runtime_dims {
		NIHILUS_FORCE_INLINE update_runtime_dims() noexcept										 = default;
		NIHILUS_FORCE_INLINE update_runtime_dims& operator=(const update_runtime_dims&) noexcept = delete;
		NIHILUS_FORCE_INLINE update_runtime_dims(const update_runtime_dims&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE update_runtime_dims& operator=(update_runtime_dims&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE update_runtime_dims(update_runtime_dims&&) noexcept				 = delete;
		using output_type																		 = base_type::output_type;
		NIHILUS_FORCE_INLINE static void impl(base_type& core, size_t new_seq_length) {
			if constexpr (array_type<decltype(core.data)>) {
				for (uint64_t x = 0; x < base_type::model_traits_type::block_count; ++x) {
					tensor_debugger::compare_tensor_data(core, x);
				}
			} else {
				tensor_debugger::compare_tensor_data(core, 0);
			}
		}
	};

	template<model_config config, typename base_type_new> struct thread_function : public base_type_new {
		NIHILUS_FORCE_INLINE thread_function() noexcept									 = default;
		NIHILUS_FORCE_INLINE thread_function& operator=(const thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE thread_function(const thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE thread_function& operator=(thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE thread_function(thread_function&&) noexcept				 = delete;
		using output_type																 = base_type_new::output_type;
		using base_type																	 = base_type_new;
		NIHILUS_FORCE_INLINE void impl(uint64_t thread_index, uint64_t thread_count) {
			kernel_dispatcher<config, device_type::cpu, base_type>::impl(*this, thread_index, thread_count);
			spinlock_nanoseconds(spinlock_time / thread_count);
		}
		NIHILUS_FORCE_INLINE void impl() {};
	};

	template<model_config config, blocking base_type_new> struct thread_function<config, base_type_new> : public base_type_new {
		NIHILUS_FORCE_INLINE thread_function() noexcept									 = default;
		NIHILUS_FORCE_INLINE thread_function& operator=(const thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE thread_function(const thread_function&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE thread_function& operator=(thread_function&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE thread_function(thread_function&&) noexcept				 = delete;
		using output_type																 = base_type_new::output_type;
		using base_type																	 = base_type_new;
		NIHILUS_FORCE_INLINE void impl(uint64_t thread_index, uint64_t thread_count, uint64_t current_index = 0) {
			this->sync_flag_start[current_index].arrive_and_wait(thread_index);
			kernel_dispatcher<config, device_type::cpu, base_type>::impl(*this, thread_index, thread_count);
			spinlock_nanoseconds(spinlock_time / thread_count);
			this->sync_flag_end[current_index].arrive_and_wait(thread_index);
		}

		NIHILUS_FORCE_INLINE void impl(uint64_t current_index = 0) {
			this->sync_flag_start[current_index].main_wait();
			this->sync_flag_end[current_index].main_wait();
		}
	};

	template<model_config config, typename base_type_new> struct main_thread_function : public base_type_new {
		NIHILUS_FORCE_INLINE main_thread_function() noexcept									   = default;
		NIHILUS_FORCE_INLINE main_thread_function& operator=(const main_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE main_thread_function(const main_thread_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE main_thread_function& operator=(main_thread_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE main_thread_function(main_thread_function&&) noexcept				   = delete;
		using output_type																		   = base_type_new::output_type;
		using base_type																			   = base_type_new;

		NIHILUS_FORCE_INLINE void impl(uint64_t current_index = 0) {
			this->sync_flag_start[current_index].main_wait();
			this->sync_flag_end[current_index].main_wait();
		}
	};

	template<model_config config, blocking base_type_new> struct main_thread_function<config, base_type_new> : public base_type_new {
		NIHILUS_FORCE_INLINE main_thread_function() noexcept									   = default;
		NIHILUS_FORCE_INLINE main_thread_function& operator=(const main_thread_function&) noexcept = delete;
		NIHILUS_FORCE_INLINE main_thread_function(const main_thread_function&) noexcept			   = delete;
		NIHILUS_FORCE_INLINE main_thread_function& operator=(main_thread_function&&) noexcept	   = delete;
		NIHILUS_FORCE_INLINE main_thread_function(main_thread_function&&) noexcept				   = delete;
		using output_type																		   = base_type_new::output_type;
		using base_type																			   = base_type_new;

		NIHILUS_FORCE_INLINE void impl(uint64_t current_index = 0) {
			this->sync_flag_start[current_index].main_wait();
			this->sync_flag_end[current_index].main_wait();
		}
	};

	template<model_config config, typename base_type_new> struct threading_strategy {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		using base_type			= base_type_new;
		using op_type_type		= model_traits_type::op_type_type;

		static constexpr uint64_t global_input_count{ [] {
			uint64_t return_value{};
			get_constexpr_core_bases_t<config>::template impl<constexpr_execution_planner>(return_value, layer_op_type::global_input);
			return return_value;
		}() };

		static constexpr uint64_t per_block_count{ [] {
			uint64_t return_value{};
			get_constexpr_core_bases_t<config>::template impl<constexpr_execution_planner>(return_value, layer_op_type::per_block);
			return return_value;
		}() };

		static constexpr uint64_t global_output_count{ [] {
			uint64_t return_value{};
			get_constexpr_core_bases_t<config>::template impl<constexpr_execution_planner>(return_value, layer_op_type::global_output);
			return return_value;
		}() };

		static constexpr auto global_input{ [] {
			uint64_t current_index{};
			array<op_type_type, global_input_count> return_value{};
			get_constexpr_core_bases_t<config>::template impl<constexpr_execution_planner>(return_value, current_index, layer_op_type::global_input);
			return return_value;
		}() };

		static constexpr auto per_block{ [] {
			uint64_t current_index{};
			array<op_type_type, per_block_count> return_value{};
			get_constexpr_core_bases_t<config>::template impl<constexpr_execution_planner>(return_value, current_index, layer_op_type::per_block);
			return return_value;
		}() };

		static constexpr auto global_output{ [] {
			uint64_t current_index{};
			array<op_type_type, global_output_count> return_value{};
			get_constexpr_core_bases_t<config>::template impl<constexpr_execution_planner>(return_value, current_index, layer_op_type::global_output);
			return return_value;
		}() };

		template<template<model_config, typename> typename thread_function, uint64_t current_index = 0>
		NIHILUS_FORCE_INLINE void impl_global_input(uint64_t thread_index, uint64_t thread_count) {
			if constexpr (current_index < global_input_count) {
				static constexpr op_type_type op_type = global_input[current_index];
				using core_traits_type				  = core_traits<config, op_type>;
				static_cast<thread_function<config, core_traits_type>*>(static_cast<core_traits_type*>(static_cast<base_type_new*>(this)))->impl(thread_index, thread_count);
				impl_global_input<thread_function, current_index + 1>(thread_index, thread_count);
			}
		}

		template<template<model_config, typename> typename thread_function, uint64_t current_index = 0>
		NIHILUS_FORCE_INLINE void impl_per_block(uint64_t thread_index, uint64_t thread_count, uint64_t current_index_new) {
			if constexpr (current_index < per_block_count) {
				static constexpr op_type_type op_type = per_block[current_index];
				using core_traits_type				  = core_traits<config, op_type>;
				if constexpr (blocking<core_traits_type>) {
					static_cast<thread_function<config, core_traits_type>*>(static_cast<core_traits_type*>(static_cast<base_type_new*>(this)))
						->impl(thread_index, thread_count, current_index_new);
				} else {
					static_cast<thread_function<config, core_traits_type>*>(static_cast<core_traits_type*>(static_cast<base_type_new*>(this)))->impl(thread_index, thread_count);
				}
				impl_per_block<thread_function, current_index + 1>(thread_index, thread_count, current_index_new);
			}
		}

		template<template<model_config, typename> typename thread_function, uint64_t current_index = 0>
		NIHILUS_FORCE_INLINE void impl_global_output(uint64_t thread_index, uint64_t thread_count) {
			if constexpr (current_index < global_output_count) {
				static constexpr op_type_type op_type = global_output[current_index];
				using core_traits_type				  = core_traits<config, op_type>;
				static_cast<thread_function<config, core_traits_type>*>(static_cast<core_traits_type*>(static_cast<base_type_new*>(this)))->impl(thread_index, thread_count);
				impl_global_output<thread_function, current_index + 1>(thread_index, thread_count);
			}
		};

		template<template<model_config, typename> typename thread_function> NIHILUS_FORCE_INLINE void impl(uint64_t thread_index, uint64_t thread_count) {
			impl_global_input<thread_function>(thread_index, thread_count);
			for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
				impl_per_block<thread_function>(thread_index, thread_count, x);
			}
			impl_global_output<thread_function>(thread_index, thread_count);
		}

		template<template<model_config, typename> typename thread_function, uint64_t current_index = 0> NIHILUS_FORCE_INLINE void impl_global_output_main() {
			if constexpr (current_index < global_output_count) {
				static constexpr op_type_type op_type = global_output[current_index];
				using core_traits_type				  = core_traits<config, op_type>;
				static_cast<thread_function<config, core_traits_type>*>(static_cast<core_traits_type*>(static_cast<base_type_new*>(this)))->impl();
				impl_global_output_main<thread_function, current_index + 1>();
			}
		};

		template<template<model_config, typename> typename thread_function, uint64_t current_index = 0> NIHILUS_FORCE_INLINE void impl_per_block_main(uint64_t current_index_new) {
			if constexpr (current_index < per_block_count) {
				static constexpr op_type_type op_type = per_block[current_index];
				using core_traits_type				  = core_traits<config, op_type>;
				if constexpr (blocking<core_traits_type>) {
					static_cast<thread_function<config, core_traits_type>*>(static_cast<core_traits_type*>(static_cast<base_type_new*>(this)))->impl(current_index_new);
				}

				impl_per_block_main<thread_function, current_index + 1>(current_index_new);
			}
		}

		template<template<model_config, typename> typename thread_function> NIHILUS_FORCE_INLINE void impl_main() {
			for (uint64_t x = 0; x < model_traits_type::block_count; ++x) {
				impl_per_block_main<thread_function>(x);
			}
			impl_global_output_main<thread_function>();
		};
	};

	template<model_config config, typename base_type_new> struct thread_pool : public threading_strategy<config, base_type_new> {
		using base_type															 = base_type_new;
		NIHILUS_FORCE_INLINE thread_pool() noexcept								 = delete;
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
		}

		template<bool raise_priority> NIHILUS_FORCE_INLINE void thread_function_impl(uint64_t thread_index) {
			while (!stop.load(std::memory_order_acquire)) {
				thread_latch.worker_wait(thread_index);
				if (!stop.load(std::memory_order_acquire)) {
					threading_strategy<config, base_type>::template impl<thread_function>(thread_index, thread_count);
					thread_latch.arrive_and_wait(thread_index);
				}
			}
		}

		NIHILUS_FORCE_INLINE void execute_tasks() {
			depths.store(0, std::memory_order_release);
			thread_latch.count_down();
			threading_strategy<config, base_type>::template impl_main<main_thread_function>();
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
		op_latch thread_latch;
	};

}