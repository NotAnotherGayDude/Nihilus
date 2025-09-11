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

#include <nihilus-incl/common/config.hpp>

namespace nihilus {

	struct gpu_properties {
	  protected:
		static constexpr static_aligned_const sm_count_raw{ 70ull };
		static constexpr static_aligned_const max_threads_per_sm_raw{ 1536ull };
		static constexpr static_aligned_const max_threads_per_block_raw{ 1024ull };
		static constexpr static_aligned_const warp_size_raw{ 32ull };
		static constexpr static_aligned_const l2_cache_size_raw{ 2097152ull };
		static constexpr static_aligned_const shared_mem_per_block_raw{ 49152ull };
		static constexpr static_aligned_const memory_bus_width_raw{ 256ull };
		static constexpr static_aligned_const memory_clock_rate_raw{ 14001000ull };
		static constexpr static_aligned_const major_compute_capability_raw{ 12ull };
		static constexpr static_aligned_const minor_compute_capability_raw{ 0ull };
		static constexpr static_aligned_const max_grid_size_x_raw{ 2147483647ull };
		static constexpr static_aligned_const gpu_arch_index_raw{ 4ull };
		static constexpr static_aligned_const total_threads_raw{ 107520ull };
		static constexpr static_aligned_const optimal_block_size_raw{ 512ull };
		static constexpr static_aligned_const optimal_grid_size_raw{ 210ull };
		
	  public:
		static constexpr const uint64_t& sm_count{ *sm_count_raw };
		static constexpr const uint64_t& max_threads_per_sm{ *max_threads_per_sm_raw };
		static constexpr const uint64_t& max_threads_per_block{ *max_threads_per_block_raw };
		static constexpr const uint64_t& warp_size{ *warp_size_raw };
		static constexpr const uint64_t& l2_cache_size{ *l2_cache_size_raw };
		static constexpr const uint64_t& shared_mem_per_block{ *shared_mem_per_block_raw };
		static constexpr const uint64_t& memory_bus_width{ *memory_bus_width_raw };
		static constexpr const uint64_t& memory_clock_rate{ *memory_clock_rate_raw };
		static constexpr const uint64_t& major_compute_capability{ *major_compute_capability_raw };
		static constexpr const uint64_t& minor_compute_capability{ *minor_compute_capability_raw };
		static constexpr const uint64_t& max_grid_size_x{ *max_grid_size_x_raw };
		static constexpr const uint64_t& total_threads{ *total_threads_raw };
		static constexpr const uint64_t& optimal_block_size{ *optimal_block_size_raw };
		static constexpr const uint64_t& optimal_grid_size{ *optimal_grid_size_raw };
		static constexpr const uint64_t& gpu_arch_index{ *gpu_arch_index_raw };
	};
}
