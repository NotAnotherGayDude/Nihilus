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
// Sampled mostly from https://github.com/fastfloat/fast_float
#pragma once

#include <nihilus-incl/benchmarking/metrics.hpp>

#if defined(NIHILUS_PLATFORM_LINUX)

	#include <linux/perf_event.h>
	#include <asm/unistd.h>
	#include <sys/ioctl.h>
	#include <unistd.h>
	#include <cstring>
	#include <vector>

namespace nihilus::benchmarking::internal {

	NIHILUS_INLINE size_t rdtsc() {
	#if defined(__x86_64__)
		uint32_t a, d;
		__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
		return static_cast<unsigned long>(a) | (static_cast<unsigned long>(d) << 32);
	#elif defined(__i386__)
		size_t x;
		__asm__ volatile("rdtsc" : "=A"(x));
		return x;
	#else
		return 0;
	#endif
	}

	class linux_events {
	  protected:
		std::vector<uint64_t> temp_result_vec{};
		std::vector<uint64_t> ids{};
		perf_event_attr attribs{};
		size_t num_events{};
		bool working{};
		int32_t fd{};

	  public:
		NIHILUS_INLINE explicit linux_events(std::vector<int32_t> config_vec) : working(true) {
			memset(&attribs, 0, sizeof(attribs));
			attribs.type		   = PERF_TYPE_HARDWARE;
			attribs.size		   = sizeof(attribs);
			attribs.disabled	   = 1;
			attribs.exclude_kernel = 1;
			attribs.exclude_hv	   = 1;

			attribs.sample_period	  = 0;
			attribs.read_format		  = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
			const int32_t pid		  = 0;
			const int32_t cpu		  = -1;
			const unsigned long flags = 0;

			int32_t group = -1;
			num_events	  = config_vec.size();
			ids.resize(config_vec.size());
			uint32_t i = 0;
			for (auto config: config_vec) {
				attribs.config = static_cast<long long unsigned int>(config);
				int32_t _fd	   = static_cast<int32_t>(syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags));
				if (_fd == -1) {
					reportError("perf_event_open");
				}
				ioctl(_fd, PERF_EVENT_IOC_ID, &ids[i++]);
				if (group == -1) {
					group = _fd;
					fd	  = _fd;
				}
			}

			temp_result_vec.resize(num_events * 2 + 1);
		}

		NIHILUS_INLINE ~linux_events() {
			if (fd != -1) {
				close(fd);
			}
		}

		NIHILUS_INLINE void run() {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
					reportError("ioctl(PERF_EVENT_IOC_RESET)");
				}

				if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
					reportError("ioctl(PERF_EVENT_IOC_ENABLE)");
				}
			}
		}

		NIHILUS_INLINE void end(std::vector<uint64_t>& results) {
			if (fd != -1) {
				if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
					reportError("ioctl(PERF_EVENT_IOC_DISABLE)");
				}

				if (read(fd, temp_result_vec.data(), temp_result_vec.size() * 8) == -1) {
					reportError("read");
				}
			}

			for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
				results[i / 2] = temp_result_vec[i];
			}
			for (uint32_t i = 2; i < temp_result_vec.size(); i += 2) {
				if (ids[i / 2 - 1] != temp_result_vec[i]) {
					reportError("event mismatch");
				}
			}
		}

		bool isWorking() {
			return working;
		}

	  protected:
		NIHILUS_INLINE void reportError(const std::string&) {
			working = false;
		}
	};

	template<typename event_count> struct event_collector_type : public linux_events, public std::vector<event_count> {
		using duration_type = decltype(clock_type::now());
		std::vector<uint64_t> results{};
		volatile uint64_t cycle_start{};
		duration_type clock_start{};
		int64_t current_index{};
		bool started{};

		NIHILUS_INLINE event_collector_type()
			: linux_events{ std::vector<int32_t>{ PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES,
				  PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES } },
			  std::vector<event_count>{} {
		}

		NIHILUS_INLINE void reset() {
			started		  = false;
			current_index = 0;
			std::vector<event_count>::clear();
		}

		NIHILUS_INLINE operator bool() {
			return started;
		}

		NIHILUS_INLINE bool hasEvents() {
			return linux_events::isWorking();
		}

		NIHILUS_INLINE void start() {
			if (hasEvents()) {
				linux_events::run();
			}
			clock_start = clock_type::now();
			cycle_start = rdtsc();
		}

		NIHILUS_INLINE performance_metrics operator*() {
			return collect_metrics(std::span<event_count>{ std::vector<event_count>::data(), std::vector<event_count>::size() }, std::vector<event_count>::size());
		}

		NIHILUS_INLINE void end(uint64_t bytes_processed) {
			if (!started) {
				started = true;
			}
			volatile uint64_t cycleEnd = rdtsc();
			const auto endClock		   = clock_type::now();

			std::vector<event_count>::emplace_back();
			std::vector<event_count>::operator[](current_index).cyclesVal.emplace(cycleEnd - cycle_start);
			std::vector<event_count>::operator[](current_index).elapsed = endClock - clock_start;
			std::vector<event_count>::operator[](current_index).bytesProcessedVal.emplace(bytes_processed);

			if (hasEvents()) {
				if (results.size() != linux_events::temp_result_vec.size()) {
					results.resize(linux_events::temp_result_vec.size());
				}
				linux_events::end(results);
				std::vector<event_count>::operator[](current_index).instructionsVal.emplace(results[1]);
				std::vector<event_count>::operator[](current_index).branchesVal.emplace(results[2]);
				std::vector<event_count>::operator[](current_index).branch_missesVal.emplace(results[3]);
				std::vector<event_count>::operator[](current_index).cache_referencesVal.emplace(results[4]);
				std::vector<event_count>::operator[](current_index).cache_missesVal.emplace(results[5]);
			}
			++current_index;
		}
	};
}

#endif
