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

#include <nihilus-incl/benchmarking/metrics.hpp>

#if NIHILUS_PLATFORM_WINDOWS

	#include <intrin.h>
	#include <vector>

namespace nihilus::benchmarking {

	template<typename event_count> struct event_collector_type : std::vector<event_count> {
		using duration_type = decltype(clock_type::now());
		duration_type clock_start{};
		int64_t current_index{};
		uint64_t cycle_start{};
		bool started{};

		NIHILUS_INLINE event_collector_type() : std::vector<event_count>{} {};

		NIHILUS_INLINE void reset() {
			started		  = false;
			current_index = 0;
		}

		NIHILUS_INLINE operator bool() {
			return started;
		}

		NIHILUS_INLINE void start() {
			clock_start = clock_type::now();
			cycle_start = __rdtsc();
			return;
		}

		NIHILUS_INLINE performance_metrics operator*() {
			return collect_metrics(std::span<event_count>{ std::vector<event_count>::data(), std::vector<event_count>::size() }, current_index);
		}

		NIHILUS_INLINE void end(uint64_t bytes_processed) {
			if (!started) {
				started = true;
			}
			volatile uint64_t cycleEnd = __rdtsc();
			const auto endClock		   = clock_type::now();
			if (std::vector<event_count>::size() < current_index + 1) {
				std::vector<event_count>::emplace_back();
			}
			std::vector<event_count>::operator[](current_index).cycles_val.emplace(cycleEnd - cycle_start);
			std::vector<event_count>::operator[](current_index).elapsed = endClock - clock_start;
			std::vector<event_count>::operator[](current_index).bytes_processed_val.emplace(bytes_processed);
			++current_index;
			return;
		}
	};

}
#endif
