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

#include <nihilus-incl/common/optional.hpp>
#include <nihilus-incl/common/common.hpp>
#include <span>

namespace nihilus {

	struct xoshiro_256 {
		array<uint64_t, 4ull> state{};

		NIHILUS_HOST xoshiro_256() noexcept
			: state{ [&] {
				  array<uint64_t, 4ull> return_values{};
				  auto x	 = get_time_based_seed() >> 12ull;
				  auto x01	 = x ^ x << 25ull;
				  auto x02	 = x01 ^ x01 >> 27ull;
				  uint64_t s = x02 * 0x2545F4914F6CDD1Dull;
				  for (uint64_t y = 0; y < 4; ++y) {
					  return_values[y] = splitmix64(s);
				  }
				  return return_values;
			  }() } {
		}

		NIHILUS_HOST uint64_t operator()() noexcept {
			const uint64_t result = rotl(state[1ull] * 5ull, 7ull) * 9ull;
			const uint64_t t	  = state[1ull] << 17ull;

			state[2ull] ^= state[0ull];
			state[3ull] ^= state[1ull];
			state[1ull] ^= state[2ull];
			state[0ull] ^= state[3ull];

			state[2ull] ^= t;

			state[3ull] = rotl(state[3ull], 45ull);

			return result;
		}

	  protected:
		NIHILUS_HOST uint64_t rotl(const uint64_t x, const uint64_t k) const noexcept {
			return (x << k) | (x >> (64ull - k));
		}

		NIHILUS_HOST uint64_t splitmix64(uint64_t& seed64) const noexcept {
			uint64_t result = seed64 += 0x9E3779B97F4A7C15ull;
			result			= (result ^ (result >> 30ull)) * 0xBF58476D1CE4E5B9ull;
			result			= (result ^ (result >> 27ull)) * 0x94D049BB133111EBull;
			return result ^ (result >> 31ull);
		}
	};

	template<typename value_type_new, uint64_t size_new> struct ring_buffer_interface {
		using value_type = value_type_new;
		using pointer	 = value_type*;
		using size_type	 = uint64_t;
		friend class iterator;

		static_assert(std::has_single_bit(size_new));
		NIHILUS_HOST ring_buffer_interface() noexcept {
		}

		NIHILUS_HOST size_type get_used_space() noexcept {
			if (full_val) {
				return size_new;
			}
			return (head >= tail) ? (head - tail) : (size_new + head - tail);
		}

		NIHILUS_HOST value_type operator*() noexcept {
			return evicted_value;
		}

		template<typename value_type_newer> NIHILUS_HOST void write_data_impl(value_type_newer&& data) noexcept {
			values[head] = detail::forward<value_type_newer>(data);
			head		 = (head + 1) & (size_new - 1);
			full_val	 = head == tail;
		}

		template<typename value_type_newer> NIHILUS_HOST bool write_data(value_type_newer&& data) noexcept {
			if (full_val) {
				evicted_value = values[tail];
				tail		  = (tail + 1) & (size_new - 1);
				write_data_impl(data);
				return true;
			} else {
				write_data_impl(data);
				return false;
			}
		}

		array<std::remove_cvref_t<value_type>, size_new> values{};
		value_type evicted_value{};
		size_type tail{};
		size_type head{};
		bool full_val{};
	};

	struct moving_averager : ring_buffer_interface<double, 64> {
		double running_sum{};

		NIHILUS_HOST moving_averager() noexcept {
		}

		NIHILUS_HOST moving_averager& operator+=(double value) noexcept {
			if (write_data(value)) {
				running_sum -= **this;
			}
			running_sum += value;
			return *this;
		}

		NIHILUS_HOST operator double() noexcept {
			if (auto used_space = get_used_space(); used_space != 0) {
				return running_sum / static_cast<double>(used_space);
			}
			return running_sum;
		}
	};

	struct jitter_generator : moving_averager, xoshiro_256 {
		NIHILUS_HOST jitter_generator& operator+=(double value) noexcept {
			moving_averager::operator+=(value);
			return *this;
		}

		NIHILUS_HOST double get_next_jitter_amount() noexcept {
			return (static_cast<double>(xoshiro_256::operator()()) / static_cast<double>(std::numeric_limits<uint64_t>::max()) * 0.05) * moving_averager::operator double();
		}
	};

}
