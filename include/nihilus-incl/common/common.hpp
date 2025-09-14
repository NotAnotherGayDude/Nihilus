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

#include <nihilus-incl/benchmarking/event_counter.hpp>
#include <nihilus-incl/common/allocator.hpp>
#include <nihilus-incl/common/config.hpp>
#include <nihilus-incl/common/exception.hpp>
#include <nihilus-incl/common/data_types.hpp>
#include <nihilus-incl/common/concepts.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/common/vector.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <fstream>
#include <csignal>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	NIHILUS_INLINE static void nihilus_pause() noexcept {
#if NIHILUS_ARCH_X64
		_mm_pause();
#elif NIHILUS_ARCH_ARM64
		__asm__ __volatile__("yield" ::: "memory");
#else
		__asm__ __volatile__("" ::: "memory");
#endif
	}

	enum class sort_methods {
		less_than,
		greater_than,
	};

	template<sort_methods sort_method, typename value_type, uint64_t size_hint = 0> struct sort;

	template<typename value_type>
	concept token_prob_types = requires() {
		detail::remove_cvref_t<value_type>::token_id;
		detail::remove_cvref_t<value_type>::probability;
	};

	template<typename value_type>
	concept core_base_creation_data_types = requires() {
		detail::remove_cvref_t<value_type>::dimensions;
		detail::remove_cvref_t<value_type>::n_dimensions;
		detail::remove_cvref_t<value_type>::layer_number;
		detail::remove_cvref_t<value_type>::op_type;
		detail::remove_cvref_t<value_type>::offset;
		detail::remove_cvref_t<value_type>::type;
	};

	template<typename T> struct greater_comparator {
		NIHILUS_INLINE static bool impl(const T& a, const T& b) {
			return a > b;
		}
	};

	template<typename T> struct less_comparator {
		NIHILUS_INLINE static bool impl(const T& a, const T& b) {
			return a < b;
		}
	};

	template<token_prob_types T> struct greater_comparator<T> {
		NIHILUS_INLINE static bool impl(const T& a, const T& b) {
			return a.probability > b.probability;
		}
	};

	template<token_prob_types T> struct less_comparator<T> {
		NIHILUS_INLINE static bool impl(const T& a, const T& b) {
			return a.probability < b.probability;
		}
	};

	template<core_base_creation_data_types T> struct greater_comparator<T> {
		NIHILUS_INLINE static bool impl(const T& a, const T& b) {
			uint64_t a_dims = a.total_dims();
			uint64_t b_dims = b.total_dims();
			return (a_dims != b_dims) ? (a_dims > b_dims) : (a.layer_number > b.layer_number);
		}
	};

	template<core_base_creation_data_types T> struct less_comparator<T> {
		NIHILUS_INLINE static bool impl(const T& a, const T& b) {
			uint64_t a_dims = a.total_dims();
			uint64_t b_dims = b.total_dims();
			return (a_dims != b_dims) ? (a_dims < b_dims) : (a.layer_number < b.layer_number);
		}
	};

	template<typename value_type> struct insertion_sort {
		template<typename Comparator> NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			for (uint64_t i = 1; i < count; ++i) {
				value_type key = std::move(values[i]);
				uint64_t j	   = i;

				while (j > 0 && Comparator::impl(key, values[j - 1])) {
					values[j] = std::move(values[j - 1]);
					--j;
				}
				values[j] = std::move(key);
			}
		}
	};

	template<typename value_type> struct introsort {
		template<typename Comparator> NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			if (count <= 1)
				return;

			uint64_t max_depth = 2 * log2_floor(count);
			introsort_impl<Comparator>(values, 0, count - 1, max_depth);
		}

	  private:
		NIHILUS_INLINE static uint64_t log2_floor(uint64_t n) {
			uint64_t result = 0;
			while (n >>= 1)
				++result;
			return result;
		}

		template<typename Comparator> NIHILUS_INLINE static void introsort_impl(value_type* values, uint64_t low, uint64_t high, uint64_t max_depth) {
			while (high - low > 16) {
				if (max_depth == 0) {
					heapsort_range<Comparator>(values, low, high + 1);
					return;
				}
				--max_depth;

				uint64_t pivot = partition<Comparator>(values, low, high);

				if (pivot - low < high - pivot) {
					introsort_impl<Comparator>(values, low, pivot, max_depth);
					low = pivot + 1;
				} else {
					introsort_impl<Comparator>(values, pivot + 1, high, max_depth);
					high = pivot;
				}
			}
			insertion_sort<value_type>::template impl<Comparator>(values + low, high - low + 1);
		}

		template<typename Comparator> NIHILUS_INLINE static uint64_t partition(value_type* values, uint64_t low, uint64_t high) {
			uint64_t mid = low + (high - low) / 2;
			if (Comparator::impl(values[mid], values[low]))
				std::swap(values[low], values[mid]);
			if (Comparator::impl(values[high], values[low]))
				std::swap(values[low], values[high]);
			if (Comparator::impl(values[high], values[mid]))
				std::swap(values[mid], values[high]);
			std::swap(values[mid], values[high]);

			value_type& pivot = values[high];
			uint64_t i		  = low;

			for (uint64_t j = low; j < high; ++j) {
				if (Comparator::impl(values[j], pivot)) {
					if (i != j)
						std::swap(values[i], values[j]);
					++i;
				}
			}
			std::swap(values[i], values[high]);
			return i;
		}

		template<typename Comparator> NIHILUS_INLINE static void heapsort_range(value_type* values, uint64_t start, uint64_t count) {

			for (int64_t i = (count / 2) - 1; i >= 0; --i) {
				heapify<Comparator>(values + start, count, i);
			}

			for (uint64_t i = count - 1; i > 0; --i) {
				std::swap(values[start], values[start + i]);
				heapify<Comparator>(values + start, i, 0);
			}
		}

		template<typename Comparator> NIHILUS_INLINE static void heapify(value_type* values, uint64_t count, uint64_t root) {
			uint64_t largest = root;
			uint64_t left	 = 2 * root + 1;
			uint64_t right	 = 2 * root + 2;

			if (left < count && Comparator::impl(values[largest], values[left]))
				largest = left;
			if (right < count && Comparator::impl(values[largest], values[right]))
				largest = right;

			if (largest != root) {
				std::swap(values[root], values[largest]);
				heapify<Comparator>(values, count, largest);
			}
		}
	};

	template<typename value_type, uint64_t size_hint> struct sort<sort_methods::greater_than, value_type, size_hint> {
		NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			using Comparator = greater_comparator<value_type>;

			if constexpr (size_hint > 0 && size_hint <= 32) {
				insertion_sort<value_type>::template impl<Comparator>(values, count);
			} else {
				if (count <= 32) {
					insertion_sort<value_type>::template impl<Comparator>(values, count);
				} else {
					introsort<value_type>::template impl<Comparator>(values, count);
				}
			}
		}
	};

	template<typename value_type, uint64_t size_hint> struct sort<sort_methods::less_than, value_type, size_hint> {
		NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			using Comparator = less_comparator<value_type>;

			if constexpr (size_hint > 0 && size_hint <= 32) {
				insertion_sort<value_type>::template impl<Comparator>(values, count);
			} else {
				if (count <= 32) {
					insertion_sort<value_type>::template impl<Comparator>(values, count);
				} else {
					introsort<value_type>::template impl<Comparator>(values, count);
				}
			}
		}
	};

	template<token_prob_types value_type, uint64_t size_hint> struct sort<sort_methods::greater_than, value_type, size_hint> {
		NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			if (count <= 16) {
				for (uint64_t i = 1; i < count; ++i) {
					value_type key = values[i];
					uint64_t j	   = i;

					while (j > 0 && values[j - 1].probability < key.probability) {
						values[j] = values[j - 1];
						--j;
					}
					values[j] = key;
				}
			} else {
				using Comparator = greater_comparator<value_type >;
				introsort<value_type>::template impl<Comparator>(values, count);
			}
		}
	};

	template<token_prob_types value_type, uint64_t size_hint> struct sort<sort_methods::less_than, value_type, size_hint> {
		NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			if (count <= 16) {
				for (uint64_t i = 1; i < count; ++i) {
					value_type key = values[i];
					uint64_t j	   = i;

					while (j > 0 && values[j - 1].probability > key.probability) {
						values[j] = values[j - 1];
						--j;
					}
					values[j] = key;
				}
			} else {
				using Comparator = less_comparator<value_type >;
				introsort<value_type>::template impl<Comparator>(values, count);
			}
		}
	};

	template<typename value_type> struct index_sorter {
		template<typename Comparator> NIHILUS_INLINE static void sort_by_indices(value_type* values, uint64_t count) {
			std::vector<uint64_t> indices(count);
			std::iota(indices.begin(), indices.end(), 0);

			struct index_comparator {
				const value_type* values_ptr;

				NIHILUS_INLINE index_comparator(const value_type* ptr) : values_ptr(ptr) {
				}

				NIHILUS_INLINE static bool impl(uint64_t a, uint64_t b, const value_type* values) {
					return Comparator::impl(values[a], values[b]);
				}
			};

			sort_indices_impl<Comparator>(indices.data(), count, values);

			rearrange_by_indices(values, indices.data(), count);
		}

	  private:
		template<typename Comparator> NIHILUS_INLINE static void sort_indices_impl(uint64_t* indices, uint64_t count, const value_type* values) {
			if (count <= 1)
				return;

			uint64_t max_depth = 2 * log2_floor(count);
			sort_indices_recursive<Comparator>(indices, 0, count - 1, max_depth, values);
		}

		template<typename Comparator>
		NIHILUS_INLINE static void sort_indices_recursive(uint64_t* indices, uint64_t low, uint64_t high, uint64_t max_depth, const value_type* values) {
			while (high - low > 16) {
				if (max_depth == 0) {
					heap_sort_indices<Comparator>(indices, low, high + 1, values);
					return;
				}
				--max_depth;

				uint64_t pivot = partition_indices<Comparator>(indices, low, high, values);

				if (pivot - low < high - pivot) {
					sort_indices_recursive<Comparator>(indices, low, pivot, max_depth, values);
					low = pivot + 1;
				} else {
					sort_indices_recursive<Comparator>(indices, pivot + 1, high, max_depth, values);
					high = pivot;
				}
			}

			insertion_sort_indices<Comparator>(indices + low, high - low + 1, values);
		}

		template<typename Comparator> NIHILUS_INLINE static uint64_t partition_indices(uint64_t* indices, uint64_t low, uint64_t high, const value_type* values) {
			uint64_t mid = low + (high - low) / 2;
			if (Comparator::impl(values[indices[mid]], values[indices[low]]))
				std::swap(indices[low], indices[mid]);
			if (Comparator::impl(values[indices[high]], values[indices[low]]))
				std::swap(indices[low], indices[high]);
			if (Comparator::impl(values[indices[high]], values[indices[mid]]))
				std::swap(indices[mid], indices[high]);
			std::swap(indices[mid], indices[high]);

			uint64_t pivot_idx = indices[high];
			uint64_t i		   = low;

			for (uint64_t j = low; j < high; ++j) {
				if (Comparator::impl(values[indices[j]], values[pivot_idx])) {
					if (i != j)
						std::swap(indices[i], indices[j]);
					++i;
				}
			}
			std::swap(indices[i], indices[high]);
			return i;
		}

		template<typename Comparator> NIHILUS_INLINE static void insertion_sort_indices(uint64_t* indices, uint64_t count, const value_type* values) {
			for (uint64_t i = 1; i < count; ++i) {
				uint64_t key = indices[i];
				uint64_t j	 = i;

				while (j > 0 && Comparator::impl(values[key], values[indices[j - 1]])) {
					indices[j] = indices[j - 1];
					--j;
				}
				indices[j] = key;
			}
		}

		template<typename Comparator> NIHILUS_INLINE static void heap_sort_indices(uint64_t* indices, uint64_t start, uint64_t count, const value_type* values) {
			for (int64_t i = (count / 2) - 1; i >= 0; --i) {
				heapify_indices<Comparator>(indices + start, count, i, values);
			}

			for (uint64_t i = count - 1; i > 0; --i) {
				std::swap(indices[start], indices[start + i]);
				heapify_indices<Comparator>(indices + start, i, 0, values);
			}
		}

		template<typename Comparator> NIHILUS_INLINE static void heapify_indices(uint64_t* indices, uint64_t count, uint64_t root, const value_type* values) {
			uint64_t largest = root;
			uint64_t left	 = 2 * root + 1;
			uint64_t right	 = 2 * root + 2;

			if (left < count && Comparator::impl(values[indices[largest]], values[indices[left]]))
				largest = left;
			if (right < count && Comparator::impl(values[indices[largest]], values[indices[right]]))
				largest = right;

			if (largest != root) {
				std::swap(indices[root], indices[largest]);
				heapify_indices<Comparator>(indices, count, largest, values);
			}
		}

		NIHILUS_INLINE static uint64_t log2_floor(uint64_t n) {
			uint64_t result = 0;
			while (n >>= 1)
				++result;
			return result;
		}

		NIHILUS_INLINE static void rearrange_by_indices(value_type* values, uint64_t* indices, uint64_t count) {
			for (uint64_t i = 0; i < count; ++i) {
				if (indices[i] != i) {
					value_type temp = std::move(values[i]);
					uint64_t j		= i;

					while (indices[j] != i) {
						uint64_t next = indices[j];
						values[j]	  = std::move(values[next]);
						indices[j]	  = j;
						j			  = next;
					}

					values[j]  = std::move(temp);
					indices[j] = j;
				}
			}
		}
	};

	template<core_base_creation_data_types value_type, sort_methods sort_method, uint64_t size_hint> struct sort<sort_method, value_type, size_hint> {
		NIHILUS_INLINE static void impl(value_type* values, uint64_t count) {
			if constexpr (sort_method == sort_methods::greater_than) {
				using Comparator = greater_comparator<value_type>;

				if (count > 64) {
					index_sorter<value_type>::template sort_by_indices<Comparator>(values, count);
				} else {
					introsort<value_type>::template impl<Comparator>(values, count);
				}
			} else {
				using Comparator = less_comparator<value_type>;

				if (count > 64) {
					index_sorter<value_type>::template sort_by_indices<Comparator>(values, count);
				} else {
					introsort<value_type>::template impl<Comparator>(values, count);
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

		NIHILUS_INLINE constexpr stop_watch() noexcept {
		}

		NIHILUS_INLINE stop_watch(uint64_t newTime) noexcept {
			total_time_units.store(newTime);
		}

		NIHILUS_INLINE constexpr stop_watch& operator=(stop_watch&& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_INLINE constexpr stop_watch(stop_watch&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE constexpr stop_watch& operator=(const stop_watch& other) noexcept {
			if NIHILUS_LIKELY (this != &other) {
				total_time_units.store(other.total_time_units.load(std::memory_order_acquire), std::memory_order_release);
				start_time_units.store(other.start_time_units.load(std::memory_order_acquire), std::memory_order_release);
			}
			return *this;
		}

		NIHILUS_INLINE constexpr stop_watch(const stop_watch& other) noexcept {
			*this = other;
		}

		NIHILUS_INLINE bool has_time_elapsed() noexcept {
			return ((get_current_time() - start_time_units.load(std::memory_order_acquire)) >= total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE void reset(time_type newTimeValue = time_type{}) noexcept {
			if NIHILUS_LIKELY (newTimeValue != time_type{}) {
				total_time_units.store(newTimeValue, std::memory_order_release);
			}
			start_time_units.store(get_current_time(), std::memory_order_release);
		}

		NIHILUS_INLINE uint64_t get_total_wait_time() const noexcept {
			return get_value_as_uint(total_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE time_type total_time_elapsed() noexcept {
			return get_current_time() - start_time_units.load(std::memory_order_acquire);
		}

		NIHILUS_INLINE uint64_t total_time_elapsed_uint64() noexcept {
			return get_value_as_uint(get_current_time()) - get_value_as_uint(start_time_units.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE constexpr ~stop_watch() {
		}

	  protected:
		std::atomic<time_type> total_time_units{};
		std::atomic<time_type> start_time_units{};

		NIHILUS_INLINE time_type get_current_time() {
			if constexpr (lock_free) {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch());
			} else {
				return std::chrono::duration_cast<value_type>(hr_clock::now().time_since_epoch()).count();
			}
		}

		NIHILUS_INLINE uint64_t get_value_as_uint(time_type time) {
			if constexpr (lock_free) {
				return time.count();
			} else {
				return time;
			}
		}
	};

	struct alignas(1024 * 8)
		jiff_the_jiping_jippalon_the_grand_jiper_of_the_jappaloneans_the_grand_chopper_of_jammals_onions_the_grand_jammer_of_jammals_vaccuum_the_grand_erector_of_jammals_pillar {};

	static constexpr jiff_the_jiping_jippalon_the_grand_jiper_of_the_jappaloneans_the_grand_chopper_of_jammals_onions_the_grand_jammer_of_jammals_vaccuum_the_grand_erector_of_jammals_pillar
		jiff{};

	template<auto value_new> struct make_static {
		static constexpr auto value{ value_new };
	};

	template<int8_types value_type> NIHILUS_INLINE static bool is_alpha(value_type c) noexcept {
		alignas(64) static constexpr const static_aligned_const<bool>* __restrict alpha_table{ [] {
			alignas(64) constexpr array<static_aligned_const<bool>, 256> return_values{ [] {
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

	template<int8_types value_type> NIHILUS_INLINE static bool is_space(value_type c) noexcept {
		alignas(64) static constexpr const static_aligned_const<bool>* __restrict space_table{ [] {
			alignas(64) constexpr array<static_aligned_const<bool>, 256> return_values{ [] {
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

	template<int8_types value_type> NIHILUS_INLINE static constexpr bool is_digit(value_type c) noexcept {
		return static_cast<uint8_t>(c - '0') < 10;
	}

	template<integral_or_enum_types value_type_new> struct alignas(64) atomic_flag_wrapper {
		static constexpr static_aligned_const spin_cycles{ 500000ull };
		using value_type										= typename std::atomic_signed_lock_free::value_type;
		NIHILUS_INLINE constexpr atomic_flag_wrapper() noexcept = default;
		NIHILUS_INLINE constexpr atomic_flag_wrapper& operator=(const atomic_flag_wrapper&) noexcept {
			return *this;
		}

		NIHILUS_INLINE constexpr atomic_flag_wrapper(const atomic_flag_wrapper&) noexcept {
		}

		NIHILUS_INLINE void store(value_type_new value_new) {
			flag.store(static_cast<value_type>(value_new), std::memory_order_release);
		}

		NIHILUS_INLINE value_type_new load() {
			return static_cast<value_type_new>(flag.load(std::memory_order_acquire));
		}

		NIHILUS_INLINE void clear() {
			flag.store(0, std::memory_order_release);
		}

		NIHILUS_INLINE void test_and_set() {
			flag.store(1, std::memory_order_release);
		}

		NIHILUS_INLINE void notify_one() {
			flag.notify_one();
		}

		NIHILUS_INLINE void notify_all() {
			flag.notify_all();
		}

		NIHILUS_INLINE value_type_new fetch_add(value_type value) {
			return static_cast<value_type_new>(flag.fetch_add(value, std::memory_order_seq_cst));
		}

		NIHILUS_INLINE value_type_new fetch_sub(value_type value) {
			return static_cast<value_type_new>(flag.fetch_sub(value, std::memory_order_seq_cst));
		}

		NIHILUS_INLINE bool test() {
			return flag.load(std::memory_order_acquire) == 1;
		}

		NIHILUS_INLINE void hybrid_wait(value_type expected) {
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

		NIHILUS_INLINE void wait() {
			value_type current_value_01{ flag.load(std::memory_order_acquire) };
			while (current_value_01 != 0) {
				current_value_01 = flag.load(std::memory_order_acquire);
				nihilus_pause();
			}
		}

	  protected:
		alignas(64) std::atomic_signed_lock_free flag{};
	};

	struct alignas(64) main_gate_latch {
		NIHILUS_INLINE main_gate_latch()								  = default;
		NIHILUS_INLINE main_gate_latch& operator=(const main_gate_latch&) = delete;
		NIHILUS_INLINE main_gate_latch(const main_gate_latch&)			  = delete;
		using value_type												  = std::atomic_signed_lock_free::value_type;
		aligned_vector<atomic_flag_wrapper<value_type>> flags{};
		atomic_flag_wrapper<value_type> global_counter{};
		alignas(64) value_type thread_count{};

		NIHILUS_INLINE void init(value_type thread_count_new) {
			thread_count = thread_count_new;
			flags.resize(static_cast<typename aligned_vector<atomic_flag_wrapper<int64_t>>::size_type>(thread_count));
			global_counter.store(thread_count);
		}

		NIHILUS_INLINE void worker_wait(value_type thread_index) {
			flags[thread_index].hybrid_wait(1);
			flags[thread_index].clear();
		}

		NIHILUS_INLINE void arrive() {
			global_counter.fetch_sub(1);
		}

		NIHILUS_INLINE void count_down() {
			for (value_type x = 0; x < thread_count; ++x) {
				flags[x].test_and_set();
				flags[x].notify_one();
			}
		}

		NIHILUS_INLINE void main_wait() {
			global_counter.wait();
			global_counter.store(thread_count);
		}
	};

	template<printable_enum_types auto current_index> consteval std::string_view get_enum_name() {
#if NIHILUS_COMPILER_MSVC || NIHILUS_COMPILER_CUDA
		alignas(64) constexpr static_aligned_const<const char*> pretty_function_tail[]{ { ">(void)" } };
#else
		alignas(64) constexpr static_aligned_const<const char*> pretty_function_tail[]{ { "]" } };
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
		return get_enum_names_impl<current_type>(std::make_index_sequence<static_cast<size_t>(current_type::count)>{});
	}

	template<printable_enum_types enum_type> struct names {
		static constexpr auto data{ get_enum_names<enum_type>() };
	};

	template<printable_enum_types enum_type> NIHILUS_INLINE std::string_view get_name(enum_type type) {
		if (static_cast<uint64_t>(type) < names<enum_type>::data.size()) {
			return names<enum_type>::data[type];
		} else {
			return "Unknown Type.";
		}
	}

	template<printable_enum_types enum_type> NIHILUS_INLINE static std::ostream& operator<<(std::ostream& os, enum_type type) {
		os << get_name(type);
		return os;
	}

	enum class data_types : uint64_t {
		f32	  = 0,
		f16	  = 1,
		q8_0  = 8,
		i8	  = 24,
		i16	  = 25,
		i32	  = 26,
		i64	  = 27,
		f64	  = 28,
		bf16  = 30,
		count = 39,
	};

	enum class core_types : uint8_t {
		weights,
		global_inputs,
		token_embeddings,
		mega_qkv_prep_and_cache_publish,
		mega_attention_apply,
		mega_ffn,
		final_norm_and_sampling,
		count,
	};

	enum class kernel_types : uint8_t {
		none,
		get_rows,
		rms_norm,
		mul,
		mul_mat,
		reshape,
		transpose,
		permute,
		view,
		rope,
		softmax,
		silu,
		copy,
		cont,
		add,
		sub,
		top_k_filter,
		top_p_filter,
		repetition_penalty,
		presence_penalty,
		temperature_scale,
		frequency_penalty,
		vocab_mask,
		sample_logits,
		count,
	};

	enum class composite_kernel_types : uint8_t {
		none,
		view,
		get_rows,
		mega_qkv_prep_and_cache,
		mega_attention_apply,
		mega_ffn,
		final_norm_and_sampling,
		count,
	};

	enum class weight_types : uint8_t {
		attn_q,
		attn_k,
		attn_v,
		attn_output,
		attn_norm,
		ffn_gate,
		ffn_up,
		ffn_down,
		ffn_norm,
		token_embd,
		rope_freqs,
		output_norm,
		output,
		count,
	};

	enum class global_input_types : uint8_t {
		inp_tokens,
		inp_pos,
		cache_k,
		cache_v,
		kq_mask,
		inp_out_ids,
		temperature,
		top_k,
		top_p,
		repetition_penalty,
		presence_penalty,
		frequency_penalty,
		rep_window,
		token_history,
		rng_state,
		logits_bias,
		allowed_vocab_mask,
		count,
	};

	enum class token_embedding_types : uint8_t {
		get_rows,
		count,
	};

	enum class mega_qkv_prep_and_cache_publish_types : uint8_t {
		q_out,
		count,
	};

	enum class mega_attention_apply_types {
		ffn_inp,
		count,
	};

	enum class mega_ffn_types {
		l_out,
		count,
	};

	enum class final_norm_and_sampling_types {
		result_token_id,
		count,
	};

	enum class global_output_types : uint8_t {
		result_output_composite,
		count,
	};

	enum class rope_and_cache_types : uint8_t {
		rope_q_permute_type,
		rope_k_copy_type,
		k_rope_view_type,
		v_rope_view_type,
		count,
	};

	enum class attention_scores_types : uint8_t {
		kq_scores_type,
		count,
	};

	enum class attention_weighted_values_types : uint8_t {
		attention_output_type,
		count,
	};

	enum class attention_output_projection_types : uint8_t {
		attn_output_type,
		count,
	};

	enum class ffn_parallel_projection_types : uint8_t {
		ffn_gate_type,
		ffn_up_type,
		count,
	};

	enum class ffn_down_projection_types : uint8_t {
		ffn_down_type,
		count,
	};

	enum class device_types : uint8_t {
		cpu,
		gpu,
		numa,
	};

	enum class model_arches : uint8_t {
		llama,
		deci,
		falcon,
		baichuan,
		grok,
		gpt2,
		gptj,
		gptneox,
		mpt,
		starcoder,
		refact,
		bert,
		nomic_bert,
		jina_bert_v2,
		bloom,
		stablelm,
		qwen,
		qwen2,
		qwen2moe,
		qwen2vl,
		phi2,
		phi3,
		phimoe,
		plamo,
		codeshell,
		orion,
		internlm2,
		minicpm,
		minicpm3,
		gemma,
		gemma2,
		starcoder2,
		mamba,
		xverse,
		command_r,
		cohere2,
		dbrx,
		olmo,
		olmo2,
		olmoe,
		openelm,
		arctic,
		deepseek,
		deepseek2,
		chatglm,
		bitnet,
		t5,
		t5encoder,
		jais,
		nemotron,
		exaone,
		rwkv6,
		rwkv6qwen2,
		granite,
		granite_moe,
		chameleon,
		wavtokenizer_dec,
		unknown,
		count,
	};

	enum class kernel_type_profiles : uint8_t {
		fp16_mha,
		fp16_moe,
		bf16_mha,
		bf16_gqa,
		q4_mha,
		q4_gqa,
		q4_moe,
		q8_mha,
		q8_gqa,
		q8_moe,
		mixed_fp16_fp32,
		mixed_bf16_fp32,
		count,
	};

	enum class norm_types : uint8_t {
		rms_standard,
		rms_parallel,
		rms_grouped,
		layer_norm_standard,
		layer_norm_no_bias,
		rms_norm_welford,
		adaptive_norm,
		count,
	};

	enum class kv_cache_strategies : uint8_t {
		contiguous,
		paged,
		compressed,
		streaming,
		hierarchical,
		count,
	};

	enum class rope_scaling_types : uint8_t {
		none,
		linear,
		dynamic,
		yarn,
		longrope,
		count,
	};

	enum class model_generations : uint8_t {
		v1_v2,
		v3,
		v3_1,
		v3_2,
		count,
	};

	enum class model_sizes : uint8_t {
		llm_unknown,
		llm_14M,
		llm_17M,
		llm_22M,
		llm_33M,
		llm_60M,
		llm_70M,
		llm_80M,
		llm_109M,
		llm_137M,
		llm_160M,
		llm_220M,
		llm_250M,
		llm_270M,
		llm_335M,
		llm_410M,
		llm_450M,
		llm_770M,
		llm_780M,
		llm_0_5B,
		llm_1B,
		llm_1_3B,
		llm_1_4B,
		llm_1_5B,
		llm_1_6B,
		llm_2B,
		llm_2_8B,
		llm_3B,
		llm_4B,
		llm_6B,
		llm_6_9B,
		llm_7B,
		llm_8B,
		llm_9B,
		llm_11B,
		llm_12B,
		llm_13B,
		llm_14B,
		llm_15B,
		llm_16B,
		llm_20B,
		llm_30B,
		llm_32B,
		llm_34B,
		llm_35B,
		llm_40B,
		llm_65B,
		llm_70B,
		llm_405B,
		llm_SMALL,
		llm_MEDIUM,
		llm_LARGE,
		llm_XL,
		llm_A1_7B,
		llm_A2_7B,
		llm_8x7B,
		llm_8x22B,
		llm_16x12B,
		llm_16x3_8B,
		llm_10B_128x3_66B,
		llm_57B_A14B,
		llm_27B,
		count,
	};

	enum class tokenizer_types : uint8_t {
		none,
		spm,
		bpe,
		wpm,
		ugm,
		rwkv,
		count,
	};

	enum class tokenizer_pre_types : uint8_t {
		default_pre,
		llama3,
		deepseek_llm,
		deepseek_coder,
		falcon,
		mpt,
		starcoder,
		gpt2,
		refact,
		command_r,
		stablelm2,
		qwen2,
		olmo,
		dbrx,
		smaug,
		poro,
		chatglm3,
		chatglm4,
		viking,
		jais,
		tekken,
		smollm,
		codeshell,
		bloom,
		gpt3_finnish,
		exaone,
		chameleon,
		minerva,
		deepseek3_llm,
		count,
	};

	enum class rope_types : int8_t {
		none_rope = -1,
		norm,
		neox,
		mrope,
		vision,
		count,
	};

	enum class token_types : uint8_t {
		undefined_token,
		normal,
		unknown,
		control,
		user_defined,
		unused,
		byte,
		count,
	};

	enum class tokens : uint16_t {
		undefined	 = 0,
		unknown		 = 1 << 0,
		unused		 = 1 << 1,
		normal		 = 1 << 2,
		control		 = 1 << 3,
		user_defined = 1 << 4,
		byte		 = 1 << 5,
		normalized	 = 1 << 6,
		lstrip		 = 1 << 7,
		rstrip		 = 1 << 8,
		single_word	 = 1 << 9,
		count,
	};

	enum class model_format {
		nh_void,
		gguf,
		count,
	};

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
