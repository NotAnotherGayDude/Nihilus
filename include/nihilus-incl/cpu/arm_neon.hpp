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

#include <nihilus-incl/common/sub_kernel_traits.hpp>
#include <nihilus-incl/cpu/common.hpp>

#if NIHILUS_NEON

namespace nihilus {

	template<typename output_type> NIHILUS_HOST static constexpr int64_t calculate_chunk_count(output_type& output, uint64_t& chunk_size, int64_t thread_count) {
		const auto dims			   = output.get_array_rt();
		const uint64_t total_bytes = type_traits<typename output_type::output_type>::total_byte_size(dims);
		uint64_t chunk_count	   = detail::max(static_cast<uint64_t>(1), total_bytes / static_cast<uint64_t>(static_cast<float>(cpu_properties::l1_cache_size) * 0.5f));
		chunk_count				   = (chunk_count == 1) ? static_cast<uint64_t>(thread_count) : chunk_count;
		const uint64_t total_elems = static_cast<uint64_t>(dims[0]) * static_cast<uint64_t>(dims[1]) * static_cast<uint64_t>(dims[2]) * static_cast<uint64_t>(dims[3]);
		chunk_size				   = total_elems / chunk_count;
		return static_cast<int64_t>(chunk_count);
	}

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t) {
			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t) {
			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
		}

		NIHILUS_HOST static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
		}

		NIHILUS_HOST static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t) {
			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 1, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type& params, int64_t) {
			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		}
	};

};

#endif
