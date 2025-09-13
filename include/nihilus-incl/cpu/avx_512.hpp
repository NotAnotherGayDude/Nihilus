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

#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/cpu/common.hpp>

#if NIHILUS_AVX512

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

	NIHILUS_HOST void dequantize_q8_0_to_f32(const block_q8_0<half>* __restrict src, float* __restrict dst, uint64_t count) {
		constexpr uint64_t block_size = 32;

		const uint64_t full_blocks = count / block_size;
		const uint64_t remainder   = count % block_size;

		for (uint64_t block_idx = 0; block_idx < full_blocks; ++block_idx) {
			const block_q8_0<half>& block	   = src[block_idx];
			const float scale				   = fp16_to_fp32(block.d);
			const int8_t* __restrict quantized = block.qs;
			const uint64_t base_offset		   = block_idx * block_size;

			for (uint64_t j = 0; j < block_size; ++j) {
				dst[base_offset + j] = scale * static_cast<float>(quantized[j]);
			}
		}
		if (remainder > 0) {
			const block_q8_0<half>& final_block = src[full_blocks];
			const float scale					= fp16_to_fp32(final_block.d);
			const int8_t* __restrict quantized	= final_block.qs;
			const uint64_t base_offset			= full_blocks * block_size;

			for (uint64_t j = 0; j < remainder; ++j) {
				dst[base_offset + j] = scale * static_cast<float>(quantized[j]);
			}
		}
	}

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type& params, int64_t current_chunk, uint64_t chunk_size) {
			auto& get_rows_op						= params.values.template get_core<token_embeddings_types, token_embeddings_types::get_rows>();
			auto& weights_core						= get_adjacent_value<typename core_traits_type::config_type, core_types::weights>::impl(params);
			auto& inputs_core						= get_adjacent_value<typename core_traits_type::config_type, core_types::global_inputs>::impl(params);
			auto& token_embd_op						= weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op						= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();
			const auto* __restrict weight_data		= token_embd_op.get_data();
			const auto* __restrict token_ids		= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>().get_data();
			constexpr uint64_t embedding_length		= model_traits_type<typename core_traits_type::config_type>::embedding_length;
			constexpr uint64_t blocks_per_embedding = embedding_length / 32;
			const uint64_t sequence_length			= inp_tokens_op.get_seq_length_dim();
			const uint64_t start_token				= static_cast<uint64_t>(current_chunk) * chunk_size;
			const uint64_t end_token				= detail::min(start_token + chunk_size, sequence_length);
			auto* __restrict output_data			= get_rows_op.get_data();
			for (uint64_t token_idx = start_token; token_idx < end_token; ++token_idx) {
				const int32_t token_id		   = token_ids[token_idx];
				const auto* __restrict src_row = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
				auto* __restrict dst_row	   = output_data + (token_idx * embedding_length);
				dequantize_q8_0_to_f32(src_row, dst_row, embedding_length);
			}
		}

		NIHILUS_HOST static void impl(core_traits_type& params, int64_t thread_count) {
			uint64_t chunk_size{};
			const int64_t chunk_count = calculate_chunk_count<typename core_traits_type::token_embeddings_type>(params.values, chunk_size, thread_count);
			int64_t current_chunk	  = params.current_chunk_prompt_eval.fetch_add(1);
			for (; current_chunk < chunk_count; current_chunk = params.current_chunk_prompt_eval.fetch_add(1)) {
				process_chunk(params, current_chunk, chunk_size);
			}

			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type& params, int64_t current_chunk, uint64_t chunk_size) {
			auto& get_rows_op						= params.values.template get_core<token_embeddings_types, token_embeddings_types::get_rows>();
			auto& weights_core						= get_adjacent_value<typename core_traits_type::config_type, core_types::weights>::impl(params);
			auto& inputs_core						= get_adjacent_value<typename core_traits_type::config_type, core_types::global_inputs>::impl(params);
			auto& token_embd_op						= weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op						= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();
			const auto* __restrict weight_data		= token_embd_op.get_data();
			const auto* __restrict token_ids		= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>().get_data();
			constexpr uint64_t embedding_length		= model_traits_type<typename core_traits_type::config_type>::embedding_length;
			constexpr uint64_t blocks_per_embedding = embedding_length / 32;
			const uint64_t sequence_length			= inp_tokens_op.get_seq_length_dim();
			const uint64_t start_token				= static_cast<uint64_t>(current_chunk) * chunk_size;
			const uint64_t end_token				= detail::min(start_token + chunk_size, sequence_length);
			auto* __restrict output_data			= get_rows_op.get_data();
			for (uint64_t token_idx = start_token; token_idx < end_token; ++token_idx) {
				const int32_t token_id		   = token_ids[token_idx];
				const auto* __restrict src_row = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
				auto* __restrict dst_row	   = output_data + (token_idx * embedding_length);
				dequantize_q8_0_to_f32(src_row, dst_row, embedding_length);
			}
		}

		NIHILUS_HOST static void impl(core_traits_type& params, int64_t thread_count) {
			uint64_t chunk_size{};
			const int64_t chunk_count = calculate_chunk_count<typename core_traits_type::token_embeddings_type>(params.values, chunk_size, thread_count);
			int64_t current_chunk	  = params.current_chunk_prompt_eval.fetch_add(1);
			for (; current_chunk < chunk_count; current_chunk = params.current_chunk_prompt_eval.fetch_add(1)) {
				process_chunk(params, current_chunk, chunk_size);
			}

			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
		}

		NIHILUS_HOST static void impl(core_traits_type&, int64_t, int64_t) {
			//params.latch_eval[current_block].fetch_sub(1);
			//params.latch_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
		}

		NIHILUS_HOST static void impl(core_traits_type&, int64_t, int64_t) {
			//params.latch_prompt_eval[current_block].fetch_sub(1);
			//params.latch_prompt_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type&, int64_t, int64_t) {
			//params.latch_eval[current_block].fetch_sub(1);
			//params.latch_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type&, int64_t, int64_t) {
			//params.latch_prompt_eval[current_block].fetch_sub(1);
			//params.latch_prompt_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type&, int64_t, int64_t) {
			//params.latch_eval[current_block].fetch_sub(1);
			//params.latch_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type&, int64_t, int64_t) {
			//params.latch_prompt_eval[current_block].fetch_sub(1);
			//params.latch_prompt_eval[current_block].wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type&, int64_t) {
			//params.latch_eval.fetch_sub(1);
			//params.latch_eval.wait();
		}
	};

	template<typename config_type, typename core_traits_type>
	struct kernel_dispatcher_impl<config_type, core_traits_type, device_types::cpu, 2, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_HOST static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_HOST static void impl(core_traits_type&, int64_t) {
			//params.latch_prompt_eval.fetch_sub(1);
			//params.latch_prompt_eval.wait();
		}
	};

};

#endif
