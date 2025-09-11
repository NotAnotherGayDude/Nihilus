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
// cuda_12.cuh

#if NIHILUS_CUDA_ENABLED

	#include <nihilus-incl/infra/core_bases.hpp>
	#include <nihilus-incl/common/common.hpp>
	#include <nihilus-incl/common/type_traits.hpp>
	#include <nihilus-incl/infra/core_traits.hpp>
	#include <cuda_runtime.h>

namespace nihilus {

	template<core_types kernel_type> struct kernel_scaling_factors {
		static constexpr float memory_bound_factor	= 1.0f;
		static constexpr float compute_bound_factor = 1.0f;
	};

	template<typename output_type, core_types kernel_type>
	NIHILUS_INLINE static constexpr int64_t calculate_gpu_launch_params(output_type& output, int64_t& work_per_thread, int64_t& grid_size, int64_t& block_size) {
		const auto dims				= output.get_array_rt();
		const uint64_t total_elems	= dims[0] * dims[1] * dims[2] * dims[3];
		const uint64_t element_size = sizeof(typename output_type::output_type);
		const uint64_t total_bytes	= total_elems * element_size;

		block_size = gpu_properties::optimal_block_size;
		block_size = (block_size / gpu_properties::warp_size) * gpu_properties::warp_size;

		const uint64_t l2_working_set  = static_cast<uint64_t>(gpu_properties::l2_cache_size * 0.75f);
		const uint64_t bytes_per_block = (total_bytes / gpu_properties::optimal_grid_size) + 1;

		float scaling_factor;
		if (bytes_per_block > l2_working_set) {
			scaling_factor = kernel_scaling_factors<kernel_type>::memory_bound_factor;
			grid_size	   = static_cast<int64_t>(total_bytes / l2_working_set) + 1;
			grid_size	   = static_cast<int64_t>(static_cast<float>(grid_size) * scaling_factor);
		} else {
			scaling_factor = kernel_scaling_factors<kernel_type>::compute_bound_factor;
			grid_size	   = static_cast<int64_t>(static_cast<float>(gpu_properties::optimal_grid_size) * scaling_factor);
		}

		const int64_t min_grid_size = gpu_properties::sm_count;
		const int64_t max_grid_size = detail::min(static_cast<int64_t>(gpu_properties::max_grid_size_x), static_cast<int64_t>(total_elems / gpu_properties::warp_size) + 1);
		grid_size					= detail::max(min_grid_size, detail::min(grid_size, max_grid_size));

		const int64_t total_threads = grid_size * block_size;
		work_per_thread				= (total_elems + total_threads - 1) / total_threads;

		const int64_t coalescing_factor = gpu_properties::warp_size / element_size;
		if (coalescing_factor > 1) {
			work_per_thread = ((work_per_thread + coalescing_factor - 1) / coalescing_factor) * coalescing_factor;
		}

		return total_threads;
	}

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk, int64_t chunk_size) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count) {
			int64_t chunk_size{};
			const int64_t chunk_count = calculate_chunk_count<typename core_traits_type::token_embeddings_type>(params.values, chunk_size, thread_count);
			int64_t current_chunk	  = params.current_chunk_prompt_eval.fetch_add(1);
			for (; current_chunk < chunk_count; current_chunk = params.current_chunk_prompt_eval.fetch_add(1)) {
				process_chunk(params, current_chunk, chunk_size);
			}

			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk, int64_t chunk_size) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk, int64_t current_block) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk, int64_t current_block) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

}
#endif