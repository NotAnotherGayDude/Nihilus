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

	#include <nihilus-incl/common/kernel_traits.hpp>
	#include <nihilus-incl/infra/core_traits.hpp>
	#include <cuda_runtime.h>

namespace nihilus {

	template<typename core_type>
	NIHILUS_INLINE __global__ static void token_embeddings_process_chunk_prompt_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
	}

	// token_embeddings, eval_time
	template<typename core_type>
	NIHILUS_INLINE __global__ static void token_embeddings_process_chunk_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
	}

	template<typename core_type> NIHILUS_INLINE __global__ static void token_embeddings_impl_eval_time(core_type& params, int64_t thread_index, int64_t thread_count) {
	}

	// mega_qkv_prep_and_cache_publish, eval_time
	template<typename core_type> NIHILUS_INLINE __global__ static void mega_qkv_prep_and_cache_publish_process_chunk_eval_time(core_type& params, int64_t thread_index,
		int64_t thread_count, int64_t current_chunk, int64_t current_block) {
	}

	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_qkv_prep_and_cache_publish_impl_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
	}

	// mega_qkv_prep_and_cache_publish, prompt_eval_time
	template<typename core_type> NIHILUS_INLINE __global__ static void mega_qkv_prep_and_cache_publish_process_chunk_prompt_eval_time(core_type& params, int64_t thread_index,
		int64_t thread_count, int64_t current_chunk, int64_t current_block) {
	}

	template<typename core_type> NIHILUS_INLINE __global__ static void mega_qkv_prep_and_cache_publish_impl_prompt_eval_time(core_type& params, int64_t thread_index,
		int64_t thread_count, int64_t current_block) {
	}

	// mega_attention_apply, eval_time
	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_attention_apply_process_chunk_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
		// PROCESS DATA.
	}

	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_attention_apply_impl_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
	}

	// mega_attention_apply, prompt_eval_time
	template<typename core_type> NIHILUS_INLINE __global__ static void mega_attention_apply_process_chunk_prompt_eval_time(core_type& params, int64_t thread_index,
		int64_t thread_count, int64_t current_chunk) {
		// PROCESS DATA.
	}

	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_attention_apply_impl_prompt_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
	}

	// mega_ffn, eval_time
	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_ffn_process_chunk_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
		// PROCESS DATA.
	}

	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_ffn_impl_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
	}

	// mega_ffn, prompt_eval_time
	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_ffn_process_chunk_prompt_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
		// PROCESS DATA.
	}

	template<typename core_type>
	NIHILUS_INLINE __global__ static void mega_ffn_impl_prompt_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
	}

	// final_norm_and_sampling, eval_time
	template<typename core_type>
	NIHILUS_INLINE __global__ static void final_norm_and_sampling_process_chunk_eval_time(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
		// PROCESS DATA.
	}

	template<typename core_type> NIHILUS_INLINE __global__ static void final_norm_and_sampling_impl_eval_time(core_type& params, int64_t thread_index, int64_t thread_count) {
	}

	// final_norm_and_sampling, prompt_eval_time
	template<typename core_type> NIHILUS_INLINE __global__ static void final_norm_and_sampling_process_chunk_prompt_eval_time(core_type& params, int64_t thread_index,
		int64_t thread_count, int64_t current_chunk) {
		// PROCESS DATA.
	}

	template<typename core_type>
	NIHILUS_INLINE __global__ static void final_norm_and_sampling_impl_prompt_eval_time(core_type& params, int64_t thread_index, int64_t thread_count) {
	}

}
#endif