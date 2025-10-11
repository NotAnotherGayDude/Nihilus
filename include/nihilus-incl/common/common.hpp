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

#include <nihilus-incl/common/allocator.hpp>
#include <nihilus-incl/common/config.hpp>
#include <nihilus-incl/common/data_types.hpp>
#include <nihilus-incl/common/concepts.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/common/aligned_vector.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <csignal>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	NIHILUS_HOST static void nihilus_pause() noexcept {
#if NIHILUS_ARCH_X64
		_mm_pause();
#elif NIHILUS_ARCH_ARM64
		__asm__ __volatile__("yield" ::: "memory");
#else
		__asm__ __volatile__("" ::: "memory");
#endif
	}

	NIHILUS_HOST static uint64_t get_time_based_seed() noexcept {
		if constexpr (std::is_same_v<std::chrono::duration<uint64_t, std::nano>, clock_type::duration>) {
			return static_cast<uint64_t>(clock_type::now().time_since_epoch().count());
		} else {
			return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::duration<uint64_t, std::nano>>(clock_type::now().time_since_epoch()).count());
		}
	}

	enum class sort_methods {
		less_than,
		greater_than,
	};

	template<sort_methods sort_method, typename value_type> struct sort;

	template<typename value_type> struct sort<sort_methods::greater_than, value_type> {
		NIHILUS_HOST static void impl(value_type* values, uint64_t count) {
			for (uint64_t i = 0; i < count - 1; ++i) {
				uint64_t min_idx = i;
				for (uint64_t j = i + 1; j < count; ++j) {
					if (values[j] > values[min_idx]) {
						min_idx = j;
					}
				}
				if (min_idx != i) {
					std::swap(values[i], values[min_idx]);
				}
			}
		}
	};

	template<typename value_type> struct sort<sort_methods::less_than, value_type> {
		NIHILUS_HOST static void impl(value_type* values, uint64_t count) {
			for (uint64_t i = 0; i < count - 1; ++i) {
				uint64_t min_idx = i;
				for (uint64_t j = i + 1; j < count; ++j) {
					if (values[j] < values[min_idx]) {
						min_idx = j;
					}
				}
				if (min_idx != i) {
					std::swap(values[i], values[min_idx]);
				}
			}
		}
	};

	template<typename value_type>
	concept time_t = is_specialization_v<value_type, std::chrono::duration>;

	template<time_t value_type = std::chrono::nanoseconds> class stop_watch {
	  public:
		using hr_clock = std::conditional_t<clock_type::is_steady, clock_type, std::chrono::steady_clock>;
		static constexpr static_aligned_const lock_free{ std::atomic<value_type>::is_always_lock_free };
		using time_type = std::conditional_t<lock_free, value_type, uint64_t>;

		NIHILUS_HOST constexpr stop_watch() noexcept {
		}

		NIHILUS_HOST stop_watch(uint64_t newTime) noexcept {
			total_time_units.store(newTime);
		}

		NIHILUS_HOST constexpr stop_watch& operator=(stop_watch&& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_HOST constexpr stop_watch(stop_watch&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_HOST constexpr stop_watch& operator=(const stop_watch& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_HOST constexpr stop_watch(const stop_watch& other) noexcept {
			*this = other;
		}

		NIHILUS_HOST bool has_time_elapsed() noexcept {
			return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >= total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_HOST void reset(time_type newTimeValue = time_type{}) noexcept {
			if NIHILUS_LIKELY (newTimeValue != time_type{}) {
				total_time_units.store(newTimeValue, std::memory_order_release);
			}
			start_time_units.store(get_current_time(), std::memory_order_release);
		}

		NIHILUS_HOST uint64_t get_total_wait_time() const noexcept {
			return get_value_as_uint(total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_HOST time_type total_time_elapsed() noexcept {
			return get_current_time() - start_time_units.load(std::memory_order_acquire);
		}

		NIHILUS_HOST uint64_t total_time_elapsed_uint64() noexcept {
			return get_value_as_uint(get_current_time()) - get_value_as_uint(start_time_units.load(std::memory_order_acquire));
		}

	  protected:
		std::atomic<time_type> total_time_units{};
		std::atomic<time_type> start_time_units{};

		NIHILUS_HOST time_type get_current_time() {
			if constexpr (lock_free) {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch());
			} else {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch()).count();
			}
		}

		NIHILUS_HOST uint64_t get_value_as_uint(time_type time) {
			if constexpr (lock_free) {
				return time.count();
			} else {
				return time;
			}
		}
	};

	struct NIHILUS_ALIGN(1024 * 8)
		jiff_the_jiping_jippalon_the_grand_jiper_of_the_jappaloneans_the_grand_chopper_of_jammals_onions_the_grand_jammer_of_jammals_vaccuum_the_grand_erector_of_jammals_pillar {};

	static constexpr jiff_the_jiping_jippalon_the_grand_jiper_of_the_jappaloneans_the_grand_chopper_of_jammals_onions_the_grand_jammer_of_jammals_vaccuum_the_grand_erector_of_jammals_pillar
		jiff{};

	template<auto value_new> struct make_static {
		static constexpr auto value{ value_new };
	};

	template<int8_types value_type> NIHILUS_HOST static bool is_alpha(value_type c) noexcept {
		NIHILUS_ALIGN(64)
		static constexpr const static_aligned_const<bool>* __restrict alpha_table{ [] {
			NIHILUS_ALIGN(64)
			constexpr array<static_aligned_const<bool>, 256> return_values{ [] {
				array<static_aligned_const<bool>, 256> return_values_new{};
				for (int32_t i = 'A'; i <= 'Z'; ++i) {
					return_values_new[static_cast<uint64_t>(i)] = static_aligned_const<bool>{ true };
				}

				for (int32_t i = 'a'; i <= 'z'; ++i) {
					return_values_new[static_cast<uint64_t>(i)] = static_aligned_const<bool>{ true };
				}
				return return_values_new;
			}() };
			return make_static<return_values>::value.data();
		}() };
		return alpha_table[static_cast<uint8_t>(c)];
	}

	template<int8_types value_type> NIHILUS_HOST static bool is_space(value_type c) noexcept {
		NIHILUS_ALIGN(64)
		static constexpr const static_aligned_const<bool>* __restrict space_table{ [] {
			NIHILUS_ALIGN(64)
			constexpr array<static_aligned_const<bool>, 256> return_values{ [] {
				array<static_aligned_const<bool>, 256> return_values_new{};
				return_values_new[static_cast<uint64_t>('\r')] = static_aligned_const<bool>{ true };
				return_values_new[static_cast<uint64_t>('\n')] = static_aligned_const<bool>{ true };
				return_values_new[static_cast<uint64_t>(' ')]  = static_aligned_const<bool>{ true };
				return_values_new[static_cast<uint64_t>('\t')] = static_aligned_const<bool>{ true };
				return_values_new[static_cast<uint64_t>('\v')] = static_aligned_const<bool>{ true };
				return_values_new[static_cast<uint64_t>('\f')] = static_aligned_const<bool>{ true };
				return return_values_new;
			}() };
			return make_static<return_values>::value.data();
		}() };
		return space_table[static_cast<uint8_t>(c)];
	}

	template<int8_types value_type> NIHILUS_HOST static constexpr bool is_digit(value_type c) noexcept {
		return static_cast<uint8_t>(c - '0') < 10;
	}

	template<integral_or_enum_types value_type_new> struct NIHILUS_ALIGN(64) atomic_flag_wrapper {
		static constexpr static_aligned_const spin_cycles{ 500000ull };
		using value_type = typename std::atomic_signed_lock_free::value_type;
		NIHILUS_HOST constexpr atomic_flag_wrapper() noexcept {
		}
		NIHILUS_HOST constexpr atomic_flag_wrapper& operator=(const atomic_flag_wrapper&) noexcept {
			return *this;
		}

		NIHILUS_HOST constexpr atomic_flag_wrapper(const atomic_flag_wrapper&) noexcept {
		}

		NIHILUS_HOST void store(value_type_new value_new) {
			flag.store(static_cast<value_type>(value_new), std::memory_order_release);
		}

		NIHILUS_HOST value_type_new load() {
			return static_cast<value_type_new>(flag.load(std::memory_order_acquire));
		}

		NIHILUS_HOST void clear() {
			flag.store(0, std::memory_order_release);
		}

		NIHILUS_HOST void test_and_set() {
			flag.store(1, std::memory_order_release);
		}

		NIHILUS_HOST void notify_one() {
			flag.notify_one();
		}

		NIHILUS_HOST void notify_all() {
			flag.notify_all();
		}

		NIHILUS_HOST value_type_new fetch_add(value_type value) {
			return static_cast<value_type_new>(flag.fetch_add(value, std::memory_order_seq_cst));
		}

		NIHILUS_HOST value_type_new fetch_sub(value_type value) {
			return static_cast<value_type_new>(flag.fetch_sub(value, std::memory_order_seq_cst));
		}

		NIHILUS_HOST bool test() {
			return flag.load(std::memory_order_acquire) == 1;
		}

		NIHILUS_HOST void hybrid_wait(value_type expected) {
			for (uint32_t i = 0; i < spin_cycles; ++i) {
				if (flag.load(std::memory_order_acquire) == expected) {
					return;
				}
				nihilus_pause();
			}
			for (value_type v = flag.load(std::memory_order_acquire); v != expected; v = flag.load(std::memory_order_acquire)) {
				flag.wait(v, std::memory_order_acquire);
			}
		}

		NIHILUS_HOST void wait() {
			value_type current_value_01{ flag.load(std::memory_order_acquire) };
			while (current_value_01 != 0) {
				current_value_01 = flag.load(std::memory_order_acquire);
				nihilus_pause();
			}
		}

	  protected:
		NIHILUS_ALIGN(64) std::atomic_signed_lock_free flag {};
	};

	struct NIHILUS_ALIGN(64) main_gate_latch {
		NIHILUS_HOST main_gate_latch() {
		}
		NIHILUS_HOST main_gate_latch& operator=(const main_gate_latch&) = delete;
		NIHILUS_HOST main_gate_latch(const main_gate_latch&)			= delete;
		using value_type												= std::atomic_signed_lock_free::value_type;
		aligned_vector<atomic_flag_wrapper<value_type>> flags{};
		atomic_flag_wrapper<value_type> global_counter{};
		NIHILUS_ALIGN(64) value_type thread_count {};

		NIHILUS_HOST void init(value_type thread_count_new) {
			thread_count = thread_count_new;
			flags.resize(static_cast<typename aligned_vector<atomic_flag_wrapper<int64_t>>::size_type>(thread_count));
			global_counter.store(thread_count);
		}

		NIHILUS_HOST void worker_wait(value_type thread_index) {
			flags[thread_index].hybrid_wait(1);
			flags[thread_index].clear();
		}

		NIHILUS_HOST void arrive() {
			global_counter.fetch_sub(1);
		}

		NIHILUS_HOST void count_down() {
			for (value_type x = 0; x < thread_count; ++x) {
				flags[x].test_and_set();
				flags[x].notify_one();
			}
		}

		NIHILUS_HOST void main_wait() {
			global_counter.wait();
			global_counter.store(thread_count);
		}
	};

	template<printable_enum_types auto current_index> consteval std::string_view get_enum_name() {
#if NIHILUS_COMPILER_MSVC || (defined(NIHILUS_COMPILER_CUDA) && NIHILUS_COMPILER_CUDA)
		NIHILUS_ALIGN(64) constexpr static_aligned_const<const char*> pretty_function_tail[]{ { ">(void)" } };
#else
		NIHILUS_ALIGN(64) constexpr static_aligned_const<const char*> pretty_function_tail[]{ { "]" } };
#endif
		std::string_view str = std::source_location::current().function_name();
#if NIHILUS_COMPILER_GNUCXX
		str			   = str.substr(str.find("=") + 2);
		uint64_t end   = str.find(';');
		str			   = str.substr(0, end);
		uint64_t start = str.find_last_of(':') + 1;
		return str.substr(start);
#else
		str			   = str.substr(str.find("=") + 2);
		uint64_t start = str.find_last_of(':') + 1;
		uint64_t end   = str.find(pretty_function_tail->value);
		return str.substr(start, end - start);
#endif
	}

	template<printable_enum_types current_type, size_t... I> consteval auto get_enum_names_impl(std::index_sequence<I...>) {
		return array<std::string_view, current_type::count>{ get_enum_name<static_cast<current_type>(I)>()... };
	}

	template<printable_enum_types current_type> consteval auto get_enum_names() {
		return get_enum_names_impl<current_type>(std::make_index_sequence<static_cast<uint64_t>(current_type::count)>{});
	}

	template<printable_enum_types enum_type> struct names {
		static constexpr auto data{ get_enum_names<enum_type>() };
	};

	template<printable_enum_types enum_type> NIHILUS_HOST std::string_view get_name(enum_type type) {
		if (static_cast<uint64_t>(type) < names<enum_type>::data.size()) {
			return names<enum_type>::data[type];
		} else {
			return "Unknown Type.";
		}
	}

	template<printable_enum_types enum_type> NIHILUS_HOST static std::ostream& operator<<(std::ostream& os, enum_type type) {
		os << get_name(type);
		return os;
	}

	struct cli_params {
		uint64_t thread_count{ std::thread::hardware_concurrency() };
		bool no_conversation{ false };
		uint64_t batch_size{ 512 };
		uint64_t n_predict{ 128 };
		std::string model_file{};
		uint64_t n_tokens{ 0 };
		std::string prompt{};
		uint64_t seed{ 0 };
	};

	struct execution_parameters {
		const int32_t* input_tokens{};
		uint64_t kv_cache_seq_len{};
		uint64_t position_offset{};
		uint64_t sequence_length{};
		std::atomic<float> seed{};
		uint64_t max_new_tokens{};
		std::string_view prompt{};
		uint64_t thread_count{};
		uint64_t token_count{};
		uint64_t random_seed{};
		int32_t eos_token_id{};
		uint64_t sequence_id{};
		bool clear_kv_cache{};
		uint64_t batch_size{};
		float temperature{};
		bool is_prefill{};
		bool use_cache{};
		int32_t top_k{};
		float top_p{};
	};
}
