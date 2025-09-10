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

#if NIHILUS_CUDA_ENABLED

	#include <nihilus-incl/infra/core_bases.hpp>
	#include <nihilus-incl/common/common.hpp>
	#include <nihilus-incl/common/type_traits.hpp>
	#include <nihilus-incl/infra/core_traits.hpp>
	#include <cuda_runtime.h>

namespace nihilus {

	template<typename output_type> NIHILUS_INLINE static constexpr int64_t calculate_chunk_count_gpu(output_type& output, int64_t& chunk_size, int64_t thread_count) {
		const auto dims			   = output.get_array_rt();
		const uint64_t total_bytes = type_traits<typename output_type::output_type>::total_byte_size(dims);
		uint64_t chunk_count	   = detail::max(1, total_bytes / static_cast<uint64_t>(static_cast<float>(gpu_cache_size_holder::l2_data_cache_size) * 0.5f));
		chunk_count				   = (chunk_count == 1) ? thread_count : chunk_count;
		const uint64_t total_elems = dims[0] * dims[1] * dims[2] * dims[3];
		chunk_size				   = total_elems / chunk_count;
		return chunk_count;
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
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk, int64_t current_block) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk, int64_t current_block) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count) {
			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		}
	};

	template<model_config config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_index, int64_t thread_count) {
			//int64_t chunk_count{ /* GET CHUNK COUNT */ };
			//int64_t current_chunk{ params.current_chunk_prompt_eval.fetch_add(1) };
			//for (; current_chunk < chunk_count; current_chunk = params.current_chunk_prompt_eval.fetch_add(1)) {
			//process_chunk(params, thread_index, thread_count, current_chunk);
			//}
			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		}
	};

}
#endif