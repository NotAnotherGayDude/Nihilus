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

	template<> struct kernel_dispatcher_impl_dev<4, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
		}

		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::token_embeddings, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
		}

		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		template<typename core_type>
		NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk, int64_t current_block) {
		}

		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		template<typename core_type>
		NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk, int64_t current_block) {
		}

		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::mega_attention_apply, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::mega_ffn, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {}
	};

	template<> struct kernel_dispatcher_impl_dev<4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE __global__ static void process_chunk(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_chunk) {
			// PROCESS DATA.
		}
		template<typename core_type> NIHILUS_INLINE __global__ static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::token_embeddings, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::mega_ffn, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

	template<> struct kernel_dispatcher_impl<device_types::gpu, 4, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE  static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
		}
	};

}
#endif