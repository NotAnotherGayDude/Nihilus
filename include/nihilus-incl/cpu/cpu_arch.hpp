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

#include <nihilus-incl/cpu/simd/avx_2.hpp>
#include <nihilus-incl/cpu/simd/avx_512.hpp>
#include <nihilus-incl/cpu/simd/arm_neon.hpp>
#include <nihilus-incl/cpu/simd/arm_sve2.hpp>

namespace nihilus {

#if !NIHILUS_AVX2 && !NIHILUS_AVX512 && !NIHILUS_NEON && !NIHILUS_SVE2

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::token_embeddings, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
			using token_embeddings_type	  = typename core_type::token_embeddings_type;
			static constexpr auto dim0101 = token_embeddings_type::get_array()[0];
			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::mega_attention_apply, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::mega_ffn, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		};
	};

	template<> struct kernel_dispatcher_impl<device_types::cpu, 0, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		template<typename core_type> NIHILUS_INLINE static void impl(core_type& params, int64_t thread_index, int64_t thread_count) {
			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		};
	};

#endif

}