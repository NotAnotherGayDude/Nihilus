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
#include <nihilus-incl/infra/core_traits.hpp>
#include <nihilus-incl/cpu/common.hpp>

#if NIHILUS_AVX2

namespace nihilus {

	template<typename output_type> NIHILUS_INLINE static constexpr int64_t calculate_chunk_count(output_type& output, uint64_t& chunk_size, int64_t thread_count) {
		const auto dims			   = output.get_array_rt();
		const uint64_t total_bytes = type_traits<typename output_type::output_type>::total_byte_size(dims);
		uint64_t chunk_count	   = detail::max(static_cast<uint64_t>(1), total_bytes / static_cast<uint64_t>(static_cast<float>(cpu_properties::l1_cache_size) * 0.5f));
		chunk_count				   = (chunk_count == 1) ? static_cast<uint64_t>(thread_count) : chunk_count;
		const uint64_t total_elems = static_cast<uint64_t>(dims[0]) * static_cast<uint64_t>(dims[1]) * static_cast<uint64_t>(dims[2]) * static_cast<uint64_t>(dims[3]);
		chunk_size				   = total_elems / chunk_count;
		return static_cast<int64_t>(chunk_count);
	}

	NIHILUS_INLINE void dequantize_q8_0_to_f32(const block_q8_0<half>* __restrict src, float* __restrict dst, uint64_t count) {
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

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::token_embeddings, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk, uint64_t chunk_size) {
			auto& get_rows_op						= params.values.template get_core<token_embedding_types, token_embedding_types::get_rows>();
			auto& weights_core						= get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core						= get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_op						= weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op						= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();
			const auto* __restrict weight_data		= token_embd_op.data;
			const auto* __restrict token_ids		= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>().data;
			constexpr uint64_t embedding_length		= model_traits_type<core_traits_type::config>::embedding_length;
			constexpr uint64_t blocks_per_embedding = embedding_length / 32;
			const uint64_t sequence_length			= inp_tokens_op.get_mutable_dim();
			const uint64_t start_token				= static_cast<uint64_t>(current_chunk) * chunk_size;
			const uint64_t end_token				= detail::min(start_token + chunk_size, sequence_length);
			auto* __restrict output_data			= get_rows_op.data;
			for (uint64_t token_idx = start_token; token_idx < end_token; ++token_idx) {
				const int32_t token_id		   = token_ids[token_idx];
				const auto* __restrict src_row = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
				auto* __restrict dst_row	   = output_data + (token_idx * embedding_length);
				dequantize_q8_0_to_f32(src_row, dst_row, embedding_length);
			}
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_count) {
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

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::token_embeddings, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type& params, int64_t current_chunk, uint64_t chunk_size) {
			auto& get_rows_op						= params.values.template get_core<token_embedding_types, token_embedding_types::get_rows>();
			auto& weights_core						= get_adjacent_value<core_traits_type::config, core_types::weights>::impl(params);
			auto& inputs_core						= get_adjacent_value<core_traits_type::config, core_types::global_inputs>::impl(params);
			auto& token_embd_op						= weights_core.values.template get_core<weight_types, weight_types::token_embd>();
			auto& inp_tokens_op						= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>();
			const auto* __restrict weight_data		= token_embd_op.data;
			const auto* __restrict token_ids		= inputs_core.values.template get_core<global_input_types, global_input_types::inp_tokens>().data;
			constexpr uint64_t embedding_length		= model_traits_type<core_traits_type::config>::embedding_length;
			constexpr uint64_t blocks_per_embedding = embedding_length / 32;
			const uint64_t sequence_length			= inp_tokens_op.get_mutable_dim();
			const uint64_t start_token				= static_cast<uint64_t>(current_chunk) * chunk_size;
			const uint64_t end_token				= detail::min(start_token + chunk_size, sequence_length);
			auto* __restrict output_data			= get_rows_op.data;
			for (uint64_t token_idx = start_token; token_idx < end_token; ++token_idx) {
				const int32_t token_id		   = token_ids[token_idx];
				const auto* __restrict src_row = weight_data + (static_cast<uint64_t>(token_id) * blocks_per_embedding);
				auto* __restrict dst_row	   = output_data + (token_idx * embedding_length);
				dequantize_q8_0_to_f32(src_row, dst_row, embedding_length);
			}
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t thread_count) {
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

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::mega_qkv_prep_and_cache_publish, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t, int64_t) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::mega_qkv_prep_and_cache_publish, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t, int64_t) {
		}

		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::mega_attention_apply, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::mega_attention_apply, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::mega_ffn, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_eval[current_block].fetch_sub(1);
			params.latch_eval[current_block].wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::mega_ffn, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t, int64_t current_block) {
			params.latch_prompt_eval[current_block].fetch_sub(1);
			params.latch_prompt_eval[current_block].wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::final_norm_and_sampling, processing_phases::eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t) {
			params.latch_eval.fetch_sub(1);
			params.latch_eval.wait();
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, core_types::final_norm_and_sampling, processing_phases::prompt_eval_time> {
		NIHILUS_INLINE static void process_chunk(core_traits_type&, int64_t) {
			// PROCESS DATA.
		}
		NIHILUS_INLINE static void impl(core_traits_type& params, int64_t) {
			params.latch_prompt_eval.fetch_sub(1);
			params.latch_prompt_eval.wait();
		}
	};
	/*
	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, block_q8_0<half>> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, block_q8_0<half>> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, block_q8_0<half>, float, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, block_q8_0<half>, float, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, block_q8_0<half>, float, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, block_q8_0<half>, float, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	NIHILUS_INLINE static void quantize_row_q8_0_avx2(const float* __restrict src, block_q8_0<half>* __restrict dst, uint64_t n) {
		static constexpr uint64_t QK = Q_SIZE;
		const uint64_t nb			 = n / QK;
		const __m256 signBit		 = _mm256_set1_ps(-0.0f);

		for (uint64_t i = 0; i < nb; i++) {
			__m256 v0 = _mm256_load_ps(src + i * QK + 0);
			__m256 v1 = _mm256_load_ps(src + i * QK + 8);
			__m256 v2 = _mm256_load_ps(src + i * QK + 16);
			__m256 v3 = _mm256_load_ps(src + i * QK + 24);

			__m256 maxAbs = _mm256_andnot_ps(signBit, v0);
			maxAbs		  = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
			maxAbs		  = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
			maxAbs		  = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

			__m128 max4			  = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
			max4				  = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
			max4				  = _mm_max_ss(max4, _mm_movehdup_ps(max4));
			const float maxScalar = _mm_cvtss_f32(max4);

			const float d	 = maxScalar / 127.f;
			dst[i].d		 = fp32_to_fp16(d);
			const float id	 = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
			const __m256 mul = _mm256_set1_ps(id);

			v0 = _mm256_mul_ps(v0, mul);
			v1 = _mm256_mul_ps(v1, mul);
			v2 = _mm256_mul_ps(v2, mul);
			v3 = _mm256_mul_ps(v3, mul);

			v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
			v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
			v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
			v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

			__m256i i0 = _mm256_cvtps_epi32(v0);
			__m256i i1 = _mm256_cvtps_epi32(v1);
			__m256i i2 = _mm256_cvtps_epi32(v2);
			__m256i i3 = _mm256_cvtps_epi32(v3);

			i0 = _mm256_packs_epi32(i0, i1);
			i2 = _mm256_packs_epi32(i2, i3);
			i0 = _mm256_packs_epi16(i0, i2);

			const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
			i0				   = _mm256_permutevar8x32_epi32(i0, perm);

			_mm256_store_si256(( __m256i* )dst[i].qs, i0);
		}
	}

	template<uint64_t size_new> struct vec_scale_mul_f32_to_q8_0;

	template<uint64_t size_new>
		requires(size_new > 0 && size_new <= 32)
	struct vec_scale_mul_f32_to_q8_0<size_new> {
		NIHILUS_INLINE static void impl(block_q8_0<half>* __restrict y_q8, const float* __restrict x, const float scale, const float* __restrict z, uint64_t block_idx,
			uint64_t element_offset) {
			alignas(64) float temp_results[32] = { 0 };

			if constexpr (size_new == 8) {
				const __m256 scale_vec = _mm256_set1_ps(scale);
				__m256 ax0			   = _mm256_load_ps(x);
				__m256 az0			   = _mm256_load_ps(z);
				__m256 temp			   = _mm256_mul_ps(ax0, az0);
				__m256 result		   = _mm256_mul_ps(temp, scale_vec);
				_mm256_store_ps(&temp_results[element_offset], result);
			} else if constexpr (size_new == 4) {
				const __m128 scale_vec = _mm_set1_ps(scale);
				__m128 ax0			   = _mm_load_ps(x);
				__m128 az0			   = _mm_load_ps(z);
				__m128 temp			   = _mm_mul_ps(ax0, az0);
				__m128 result		   = _mm_mul_ps(temp, scale_vec);
				_mm_store_ps(&temp_results[element_offset], result);
			} else {
				for (uint64_t i = 0; i < size_new; ++i) {
					temp_results[element_offset + i] = x[i] * scale * z[i];
				}
			}

			if (element_offset + size_new >= 32) {
				quantize_row_q8_0_avx2(temp_results, &y_q8[block_idx], 1);
			}
		}
	};

	NIHILUS_INLINE float simd_sum_squares(const float* __restrict data, uint64_t size) {
		__m256 sum_vec			= _mm256_setzero_ps();
		const uint64_t simd_end = size & ~7ULL;

		for (uint64_t i = 0; i < simd_end; i += 8) {
			__m256 x_vec = _mm256_load_ps(&data[i]);
			sum_vec		 = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
		}

		__m128 sum_high	  = _mm256_extractf128_ps(sum_vec, 1);
		__m128 sum_low	  = _mm256_castps256_ps128(sum_vec);
		__m128 sum_quad	  = _mm_add_ps(sum_low, sum_high);
		__m128 sum_dual	  = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
		__m128 sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 1));
		float sum		  = _mm_cvtss_f32(sum_single);

		for (uint64_t i = simd_end; i < size; ++i) {
			sum += data[i] * data[i];
		}

		return sum;
	}

	template<uint64_t size_new> struct vec_scale_mul_f32_contiguous;

	template<uint64_t size_new> struct vec_scale_mul_f32_contiguous {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			if constexpr (size_new == 4) {
				const __m128 scale_vec = _mm_set1_ps(scale);
				__m128 ax0			   = _mm_load_ps(x);
				__m128 az0			   = _mm_load_ps(z);
				__m128 temp			   = _mm_mul_ps(ax0, az0);
				__m128 result		   = _mm_mul_ps(temp, scale_vec);
				_mm_store_ps(y, result);
			} else if constexpr (size_new == 8) {
				const __m256 scale_vec = _mm256_set1_ps(scale);
				__m256 ax0			   = _mm256_load_ps(x);
				__m256 az0			   = _mm256_load_ps(z);
				__m256 temp			   = _mm256_mul_ps(ax0, az0);
				__m256 result		   = _mm256_mul_ps(temp, scale_vec);
				_mm256_store_ps(y, result);
			} else {
				for (uint64_t i = 0; i < size_new; ++i) {
					y[i] = x[i] * scale * z[i];
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, block_q8_0<half>, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, block_q8_0<half>, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, block_q8_0<half>, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, block_q8_0<half>, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul, processing_phases::prompt_eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_traits_type, float, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul, processing_phases::eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_traits_type, float, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul, processing_phases::prompt_eval_time, core_traits_type, block_q8_0<half>, float, float>
		: public kernel_base<kernel_types::mul, core_traits_type, block_q8_0<half>, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul, processing_phases::eval_time, core_traits_type, block_q8_0<half>, float, float>
		: public kernel_base<kernel_types::mul, core_traits_type, block_q8_0<half>, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::prompt_eval_time, core_traits_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, block_q8_0<half>, int32_t> {
		using input_type01			   = typename core_traits_type::input_01_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t blocks_per_row = ne00 / Q_SIZE;
			const uint64_t ne1						 = output[1];
			const auto* src_base					 = input01.data;
			float* dst_base							 = output.data;
			const int32_t* token_ids				 = input02.data;

			for (uint64_t row_idx = 0; row_idx < ne1; ++row_idx) {
				const int32_t token_id = static_cast<uint32_t>(token_ids[row_idx]);
				const auto* src_blocks	= src_base + token_id * blocks_per_row;
				float* dst_row			= dst_base + row_idx * ne0;

				for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
					const auto& src_block = src_blocks[block_idx];
					float* dst_block	  = dst_row + block_idx * Q_SIZE;
					const float d		  = fp16_to_fp32(src_block.d);

					for (uint64_t j = 0; j < Q_SIZE; ++j) {
						dst_block[j] = src_block.qs[j] * d;
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::eval_time, core_traits_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, block_q8_0<half>, int32_t> {
		using input_type01			   = typename core_traits_type::input_01_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t blocks_per_row = ne00 / Q_SIZE;
			const int32_t token_id					 = static_cast<uint32_t>(input02.data[0]);
			const auto* src_blocks					 = input01.data + token_id * blocks_per_row;
			float* dst_row							 = output.data;

			for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
				const auto& src_block = src_blocks[block_idx];
				float* dst_block	  = dst_row + block_idx * Q_SIZE;
				const float d		  = fp16_to_fp32(src_block.d);

				for (uint64_t j = 0; j < Q_SIZE; ++j) {
					dst_block[j] = src_block.qs[j] * d;
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::prompt_eval_time, core_traits_type, float, block_q8_0<half>, block_q8_0<half>>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, float, block_q8_0<half>, block_q8_0<half>> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::eval_time, core_traits_type, float, block_q8_0<half>, block_q8_0<half>>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, float, block_q8_0<half>, block_q8_0<half>> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::prompt_eval_time, core_traits_type, block_q8_0<half>, half, float>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, block_q8_0<half>, half, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::eval_time, core_traits_type, block_q8_0<half>, half, float>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, block_q8_0<half>, half, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::softmax, processing_phases::prompt_eval_time, core_traits_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::softmax, core_traits_type, float, block_q8_0<half>, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::softmax, processing_phases::eval_time, core_traits_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::softmax, core_traits_type, float, block_q8_0<half>, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rope, processing_phases::prompt_eval_time, core_traits_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::rope, core_traits_type, float, float, int32_t, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rope, processing_phases::eval_time, core_traits_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::rope, core_traits_type, float, float, int32_t, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::prompt_eval_time, core_traits_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne1		 = output[1];
			const uint64_t ne0		 = output[0];
			const float* src_base	 = input01.data;
			float* dst_base			 = output.data;
			const int32_t* token_ids = input02.data;

			for (uint64_t row_idx = 0; row_idx < ne1; ++row_idx) {
				const int32_t token_id = static_cast<uint32_t>(token_ids[row_idx]);
				const float* src_row	= src_base + token_id * ne0;
				float* dst_row			= dst_base + row_idx * ne0;

				uint64_t i = 0;
				for (; i + 8 <= ne0; i += 8) {
					__m256 src_vec = _mm256_loadu_ps(src_row + i);
					_mm256_storeu_ps(dst_row + i, src_vec);
				}
				for (; i < ne0; ++i) {
					dst_row[i] = src_row[i];
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::eval_time, core_traits_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne0		= output[0];
			const int32_t token_id = static_cast<uint32_t>(input02.data[0]);
			const float* src_row	= input01.data + token_id * ne0;
			float* dst_row			= output.data;

			uint64_t i = 0;
			for (; i + 8 <= ne0; i += 8) {
				__m256 src_vec = _mm256_loadu_ps(src_row + i);
				_mm256_storeu_ps(dst_row + i, src_vec);
			}
			for (; i < ne0; ++i) {
				dst_row[i] = src_row[i];
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::silu, processing_phases::prompt_eval_time, core_traits_type, float, float>
		: public kernel_base<kernel_types::silu, core_traits_type, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::silu, processing_phases::eval_time, core_traits_type, float, float>
		: public kernel_base<kernel_types::silu, core_traits_type, float, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne1	= output[1];

			const uint64_t chunk_count = count_elements(output) / thread_count;
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01);
				}
			} else {
				for (uint64_t index = thread_index; index < chunk_count; ++index) {
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, float, block_q8_0<half>, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_traits_type, float, block_q8_0<half>, block_q8_0<half>> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, float, block_q8_0<half>, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_traits_type, float, block_q8_0<half>, block_q8_0<half>> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, int32_t, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, int32_t, float> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rope_copy, processing_phases::prompt_eval_time, core_traits_type, float, float, int32_t, float, int16_t>
		: public kernel_base<kernel_types::rope_copy, core_traits_type, float, float, int32_t, float, int16_t> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		using input_type04			   = typename core_traits_type::input_04_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne30 = input_type04::get_array()[0];
		static constexpr uint64_t ne32 = input_type04::get_array()[2];
		static constexpr uint64_t ne33 = input_type04::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03,
			const typename core_traits_type::input_04_type& input04) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03,
			const typename core_traits_type::input_04_type& input04) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03, const typename core_traits_type::input_04_type& input04) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne31 = input04[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03, input04);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03, input04);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rope_copy, processing_phases::eval_time, core_traits_type, float, float, int32_t, float, int16_t>
		: public kernel_base<kernel_types::rope_copy, core_traits_type, float, float, int32_t, float, int16_t> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		using input_type04			   = typename core_traits_type::input_04_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne30 = input_type04::get_array()[0];
		static constexpr uint64_t ne32 = input_type04::get_array()[2];
		static constexpr uint64_t ne33 = input_type04::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03,
			const typename core_traits_type::input_04_type& input04) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03,
			const typename core_traits_type::input_04_type& input04) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03, const typename core_traits_type::input_04_type& input04) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne31 = input04[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03, input04);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03, input04);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, nihilus::kernel_types::none, nihilus::processing_phases::prompt_eval_time, core_traits_type, float,
		nihilus::block_q8_0<nihilus::half>, nihilus::block_q8_0<nihilus::half>, short> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type> struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, nihilus::kernel_types::none, nihilus::processing_phases::eval_time, core_traits_type, float,
		nihilus::block_q8_0<nihilus::half>, nihilus::block_q8_0<nihilus::half>, short> {
		using input_type01			   = typename core_traits_type::input_01_type;
		using input_type02			   = typename core_traits_type::input_02_type;
		using input_type03			   = typename core_traits_type::input_03_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne20 = input_type03::get_array()[0];
		static constexpr uint64_t ne22 = input_type03::get_array()[2];
		static constexpr uint64_t ne23 = input_type03::get_array()[3];
		static constexpr uint64_t ne0  = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2  = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3  = core_traits_type::get_array()[3];

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_block(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}
			}
		};

		template<bool blocks_per_element> NIHILUS_INLINE static void produce_single_element(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output,
			const typename core_traits_type::input_01_type& input01, const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if constexpr (blocks_per_element) {
				const uint64_t blocks_per_element_count{};

				for (uint64_t x = 0; x < blocks_per_element_count; ++x) {
					// process with actual working function.
				}

			} else {
				const uint64_t elements_per_block_count{};

				for (uint64_t x = 0; x < elements_per_block_count; ++x) {
					// process with actual working function.
				}
			}
		};

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01 = input01[1];
			const uint64_t ne11 = input02[1];
			const uint64_t ne21 = input03[1];
			const uint64_t ne1	= output[1];

			uint64_t sync_ith = output.current_chunk_prompt_eval[current_block].fetch_sub(1);

			const uint64_t chunk_count = detail::max((type_traits<typename core_traits_type::output_type>::total_byte_size(output) / (cpu_properties::cpu_properties) * 4) / 3, 1);
			const uint64_t block_byte_size{};
			const uint64_t element_byte_size{};
			if (block_byte_size > element_byte_size) {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_block<false>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			} else {
				for (uint64_t index = sync_ith; index < chunk_count; index = output.current_chunk_prompt_eval[current_block].fetch_sub(1)) {
					//std::cout<< "CURRENT INDEX: " << index << "CHUNK COUNT: " << chunk_count << std::endl;
					produce_single_element<true>(thread_index, thread_count, current_block, output, input01, input02, input03);
				}
			}
		}
	};

	template<uint64_t size_new> struct vec_add_rms_norm_f32 {};

	template<uint64_t size_new>
		requires(size_new > 0 && size_new < 4)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			for (uint64_t i = 0; i < size_new; ++i) {
				const float added = x[i] + z[i];
				y[i]			  = added * scale;
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 4)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			const __m128 scale_vec = _mm_set1_ps(scale);
			__m128 ax0			   = _mm_load_ps(x);
			__m128 az0			   = _mm_load_ps(z);
			__m128 added		   = _mm_add_ps(ax0, az0);
			__m128 ay0			   = _mm_mul_ps(added, scale_vec);
			_mm_store_ps(y, ay0);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 4 && size_new < 8)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			vec_add_rms_norm_f32<4>::impl(y, x, z, scale);

			constexpr uint64_t remainder = size_new - 8ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_f32<remainder>::impl(y + 4, x + 4, z + 4, scale);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 8)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			const __m256 scale_vec = _mm256_set1_ps(scale);
			__m256 ax0			   = _mm256_load_ps(x);
			__m256 az0			   = _mm256_load_ps(z);
			__m256 added		   = _mm256_add_ps(ax0, az0);
			__m256 ay0			   = _mm256_mul_ps(added, scale_vec);
			_mm256_store_ps(y, ay0);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 8 && size_new < 16)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			vec_add_rms_norm_f32<8>::impl(y, x, z, scale);

			constexpr uint64_t remainder = size_new - 8ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_f32<remainder>::impl(y + 8, x + 8, z + 8, scale);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 16)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);

			__m256 added0 = _mm256_add_ps(ax0, az0);
			__m256 added1 = _mm256_add_ps(ax1, az1);

			__m256 ay0 = _mm256_mul_ps(added0, scale_vec);
			__m256 ay1 = _mm256_mul_ps(added1, scale_vec);

			_mm256_store_ps(y, ay0);
			_mm256_store_ps(y + 8, ay1);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 16 && size_new < Q_SIZE)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			vec_add_rms_norm_f32<16>::impl(y, x, z, scale);

			constexpr uint64_t remainder = size_new - 16ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_f32<remainder>::impl(y + 16, x + 16, z + 16, scale);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == Q_SIZE)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 ax2 = _mm256_load_ps(x + 16);
			__m256 ax3 = _mm256_load_ps(x + 24);

			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);
			__m256 az2 = _mm256_load_ps(z + 16);
			__m256 az3 = _mm256_load_ps(z + 24);

			__m256 added0 = _mm256_add_ps(ax0, az0);
			__m256 added1 = _mm256_add_ps(ax1, az1);
			__m256 added2 = _mm256_add_ps(ax2, az2);
			__m256 added3 = _mm256_add_ps(ax3, az3);

			__m256 ay0 = _mm256_mul_ps(added0, scale_vec);
			__m256 ay1 = _mm256_mul_ps(added1, scale_vec);
			__m256 ay2 = _mm256_mul_ps(added2, scale_vec);
			__m256 ay3 = _mm256_mul_ps(added3, scale_vec);

			_mm256_store_ps(y, ay0);
			_mm256_store_ps(y + 8, ay1);
			_mm256_store_ps(y + 16, ay2);
			_mm256_store_ps(y + 24, ay3);
		}
	};

	template<uint64_t size_new>
		requires(size_new > Q_SIZE && size_new < 64)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			vec_add_rms_norm_f32<32>::impl(y, x, z, scale);

			constexpr uint64_t remainder = size_new - 32ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_f32<remainder>::impl(y + Q_SIZE, x + Q_SIZE, z + Q_SIZE, scale);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 64)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			for (uint64_t i = 0; i < 64ULL; i += 32ULL) {
				_mm_prefetch(static_cast<const char*>(x + i + 64ULL), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 64ULL), _MM_HINT_T0);

				__m256 ax0 = _mm256_load_ps(x + i);
				__m256 ax1 = _mm256_load_ps(x + i + 8ULL);
				__m256 ax2 = _mm256_load_ps(x + i + 16ULL);
				__m256 ax3 = _mm256_load_ps(x + i + 24ULL);

				__m256 az0 = _mm256_load_ps(z + i);
				__m256 az1 = _mm256_load_ps(z + i + 8ULL);
				__m256 az2 = _mm256_load_ps(z + i + 16ULL);
				__m256 az3 = _mm256_load_ps(z + i + 24ULL);

				__m256 added0 = _mm256_add_ps(ax0, az0);
				__m256 added1 = _mm256_add_ps(ax1, az1);
				__m256 added2 = _mm256_add_ps(ax2, az2);
				__m256 added3 = _mm256_add_ps(ax3, az3);

				__m256 ay0 = _mm256_mul_ps(added0, scale_vec);
				__m256 ay1 = _mm256_mul_ps(added1, scale_vec);
				__m256 ay2 = _mm256_mul_ps(added2, scale_vec);
				__m256 ay3 = _mm256_mul_ps(added3, scale_vec);

				_mm256_store_ps(y + i, ay0);
				_mm256_store_ps(y + i + 8ULL, ay1);
				_mm256_store_ps(y + i + 16ULL, ay2);
				_mm256_store_ps(y + i + 24ULL, ay3);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new > 64)
	struct vec_add_rms_norm_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale) {
			const __m256 scale_vec		 = _mm256_set1_ps(scale);
			static constexpr uint64_t np = size_new & ~63ULL;
			uint64_t i					 = 0;

			for (; i < np; i += 64ULL) {
				_mm_prefetch(static_cast<const char*>(x + i + 128ULL), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 128ULL), _MM_HINT_T0);

				for (uint64_t j = 0; j < 64ULL; j += 32ULL) {
					__m256 ax0 = _mm256_load_ps(x + i + j);
					__m256 ax1 = _mm256_load_ps(x + i + j + 8ULL);
					__m256 ax2 = _mm256_load_ps(x + i + j + 16ULL);
					__m256 ax3 = _mm256_load_ps(x + i + j + 24ULL);

					__m256 az0 = _mm256_load_ps(z + i + j);
					__m256 az1 = _mm256_load_ps(z + i + j + 8ULL);
					__m256 az2 = _mm256_load_ps(z + i + j + 16ULL);
					__m256 az3 = _mm256_load_ps(z + i + j + 24ULL);

					__m256 added0 = _mm256_add_ps(ax0, az0);
					__m256 added1 = _mm256_add_ps(ax1, az1);
					__m256 added2 = _mm256_add_ps(ax2, az2);
					__m256 added3 = _mm256_add_ps(ax3, az3);

					__m256 ay0 = _mm256_mul_ps(added0, scale_vec);
					__m256 ay1 = _mm256_mul_ps(added1, scale_vec);
					__m256 ay2 = _mm256_mul_ps(added2, scale_vec);
					__m256 ay3 = _mm256_mul_ps(added3, scale_vec);

					_mm256_stream_ps(y + i + j, ay0);
					_mm256_stream_ps(y + i + j + 8ULL, ay1);
					_mm256_stream_ps(y + i + j + 16ULL, ay2);
					_mm256_stream_ps(y + i + j + 24ULL, ay3);
				}
			}

			if (i < size_new) {
				constexpr uint64_t remainder = size_new % 64ULL;
				if constexpr (remainder > 0) {
					vec_add_rms_norm_f32<remainder>::impl(y + i, x + i, z + i, scale);
				}
			}
		}
	};

	NIHILUS_INLINE float simd_sum_squares_add(const float* __restrict x_data, const float* __restrict z_data, uint64_t size) {
		__m256 sum_vec			= _mm256_setzero_ps();
		const uint64_t simd_end = size & ~7ULL;

		for (uint64_t i = 0; i < simd_end; i += 8) {
			__m256 x_vec = _mm256_load_ps(&x_data[i]);
			__m256 z_vec = _mm256_load_ps(&z_data[i]);
			__m256 added = _mm256_add_ps(x_vec, z_vec);
			sum_vec		 = _mm256_fmadd_ps(added, added, sum_vec);
		}

		__m128 sum_high	  = _mm256_extractf128_ps(sum_vec, 1);
		__m128 sum_low	  = _mm256_castps256_ps128(sum_vec);
		__m128 sum_quad	  = _mm_add_ps(sum_low, sum_high);
		__m128 sum_dual	  = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
		__m128 sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 1));
		float sum		  = _mm_cvtss_f32(sum_single);

		for (uint64_t i = simd_end; i < size; ++i) {
			const float added = x_data[i] + z_data[i];
			sum += added * added;
		}

		return sum;
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t thread_index, uint64_t thread_count, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t src0_stride_03			 = ne02 * ne01 * ne00;
			const uint64_t src0_stride_02			 = ne01 * ne00;
			static constexpr uint64_t src0_stride_01 = ne00;

			const uint64_t src1_stride_13			 = ne12 * ne11 * ne10;
			const uint64_t src1_stride_12			 = ne11 * ne10;
			static constexpr uint64_t src1_stride_11 = ne10;

			const uint64_t dst_stride_03			= ne02 * ne01 * ne00;
			const uint64_t dst_stride_02			= ne01 * ne00;
			static constexpr uint64_t dst_stride_01 = ne00;

			static constexpr float eps = core_traits_type::model_traits_type::layer_norm_rms_epsilon;
			const uint64_t total_rows  = ne01 * ne02 * ne03;

			if constexpr (is_power_of_2able) {
				const uint64_t log2_ne01			= tzcnt(ne01);
				static constexpr uint64_t log2_ne02 = tzcnt_constexpr(ne02);
				const uint64_t log2_ne11			= tzcnt(ne11);
				static constexpr uint64_t log2_ne12 = tzcnt_constexpr(ne12);
				static constexpr uint64_t log2_ne13 = tzcnt_constexpr(ne13);
				static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
				const uint64_t log2_ne02_ne01		= log2_ne02 + log2_ne01;

				for (uint64_t row_idx = thread_index; row_idx < total_rows; row_idx += thread_count) {
					const uint64_t i03 = row_idx >> log2_ne02_ne01;
					const uint64_t i02 = (row_idx - (i03 << log2_ne02_ne01)) >> log2_ne01;
					const uint64_t i01 = row_idx - (i03 << log2_ne02_ne01) - (i02 << log2_ne01);
					const uint64_t i13 = i03 & ((1ULL << log2_ne13) - 1ULL);
					const uint64_t i12 = i02 & ((1ULL << log2_ne12) - 1ULL);
					const uint64_t i11 = i01 & ((1ULL << log2_ne11) - 1ULL);

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr = &src0_data[src0_offset];
					const float* __restrict src1_ptr = &src1_data[src1_offset];
					float* __restrict dst_ptr		 = &dst_data[dst_offset];

					const float sum	  = simd_sum_squares_add(src0_ptr, src1_ptr, ne00);
					const float mean  = sum / static_cast<float>(ne00);
					const float scale = 1.0f / sqrtf_fast(mean + eps);

					if constexpr (ne10 == ne00) {
						const uint64_t nr0 = ne00 >> log2_ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							vec_add_rms_norm_f32<ne10>::impl(dst_ptr + (r << log2_ne10), src0_ptr + (r << log2_ne10), src1_ptr + (r << log2_ne10), scale);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10 = i0 & ((1ULL << log2_ne10) - 1ULL);
							const float added  = src0_ptr[i0] + src1_ptr[i10];
							dst_ptr[i0]		   = added * scale;
						}
					}
				}
			} else {
				for (uint64_t row_idx = thread_index; row_idx < total_rows; row_idx += thread_count) {
					const uint64_t i03 = row_idx / (ne02 * ne01);
					const uint64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
					const uint64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;
					const uint64_t i13 = i03 % ne13;
					const uint64_t i12 = i02 % ne12;
					const uint64_t i11 = i01 % ne11;

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr = &src0_data[src0_offset];
					const float* __restrict src1_ptr = &src1_data[src1_offset];
					float* __restrict dst_ptr		 = &dst_data[dst_offset];

					const float sum	  = simd_sum_squares_add(src0_ptr, src1_ptr, ne00);
					const float mean  = sum / static_cast<float>(ne00);
					const float scale = 1.0f / sqrtf_fast(mean + eps);

					if constexpr (ne10 == ne00) {
						const uint64_t nr0 = ne00 / ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							vec_add_rms_norm_f32<ne10>::impl(dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr + r * ne10, scale);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10 = i0 % ne10;
							const float added  = src0_ptr[i0] + src1_ptr[i10];
							dst_ptr[i0]		   = added * scale;
						}
					}
				}
			}
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t thread_index			  = static_cast<uint64_t>(thread_index);
			const uint64_t thread_count			  = static_cast<uint64_t>(thread_count);
			const float* __restrict src01 = input02.data;

			const bool is_power_of_2 = ((ne01 & (ne01 - 1ULL)) == 0) && ((ne02 & (ne02 - 1ULL)) == 0) && ((ne10 & (ne10 - 1ULL)) == 0) && ((ne11 & (ne11 - 1ULL)) == 0) &&
				((ne12 & (ne12 - 1ULL)) == 0) && ((ne13 & (ne13 - 1ULL)) == 0);

			if (is_power_of_2) {
				process_tensor_elements<true>(thread_index, thread_count, ne01, ne11, input01.data, src01, output.data);
			} else {
				process_tensor_elements<false>(thread_index, thread_count, ne01, ne11, input01.data, src01, output.data);
			}
		}
	};

	template<uint64_t size_new> struct vec_add_rms_norm_mul_q8_f32 {};

	template<uint64_t size_new>
		requires(size_new > 0 && size_new < 4)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			for (uint64_t i = 0; i < size_new; ++i) {
				const float added			   = x[i] + z[i];
				const float normalized		   = added * scale;
				const float dequantized_weight = static_cast<float>(w_quants[w_offset + i]) * w_scales[w_offset / Q_SIZE];
				y[i]						   = normalized * dequantized_weight;
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 4)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			const __m128 scale_vec = _mm_set1_ps(scale);

			__m128 ax0	 = _mm_load_ps(x);
			__m128 az0	 = _mm_load_ps(z);
			__m128 added = _mm_add_ps(ax0, az0);

			__m128 normalized = _mm_mul_ps(added, scale_vec);

			const float block_scale	 = w_scales[w_offset / Q_SIZE];
			const __m128 w_scale_vec = _mm_set1_ps(block_scale);

			__m128i q8_weights		   = _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset]));
			__m128i q8_weights_128	   = _mm_cvtepi8_epi32(q8_weights);
			__m128 dequantized_weights = _mm_cvtepi32_ps(q8_weights_128);
			dequantized_weights		   = _mm_mul_ps(dequantized_weights, w_scale_vec);

			__m128 result = _mm_mul_ps(normalized, dequantized_weights);
			_mm_store_ps(y, result);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 4 && size_new < 8)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			vec_add_rms_norm_mul_q8_f32<4>::impl(y, x, z, scale, w_scales, w_quants, w_offset);

			constexpr uint64_t remainder = size_new - 4ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_mul_q8_f32<remainder>::impl(y + 4, x + 4, z + 4, scale, w_scales, w_quants, w_offset + 4);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 8)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			__m256 ax0	 = _mm256_load_ps(x);
			__m256 az0	 = _mm256_load_ps(z);
			__m256 added = _mm256_add_ps(ax0, az0);

			__m256 normalized = _mm256_mul_ps(added, scale_vec);

			const float block_scale	 = w_scales[w_offset / Q_SIZE];
			const __m256 w_scale_vec = _mm256_set1_ps(block_scale);

			__m128i q8_weights		   = _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset]));
			__m256i q8_weights_256	   = _mm256_cvtepi8_epi32(q8_weights);
			__m256 dequantized_weights = _mm256_cvtepi32_ps(q8_weights_256);
			dequantized_weights		   = _mm256_mul_ps(dequantized_weights, w_scale_vec);

			__m256 result = _mm256_mul_ps(normalized, dequantized_weights);
			_mm256_store_ps(y, result);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 8 && size_new < 16)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			vec_add_rms_norm_mul_q8_f32<8>::impl(y, x, z, scale, w_scales, w_quants, w_offset);

			constexpr uint64_t remainder = size_new - 8ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_mul_q8_f32<remainder>::impl(y + 8, x + 8, z + 8, scale, w_scales, w_quants, w_offset + 8);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 16)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);

			__m256 added0 = _mm256_add_ps(ax0, az0);
			__m256 added1 = _mm256_add_ps(ax1, az1);

			__m256 normalized0 = _mm256_mul_ps(added0, scale_vec);
			__m256 normalized1 = _mm256_mul_ps(added1, scale_vec);

			const float block_scale	 = w_scales[w_offset / Q_SIZE];
			const __m256 w_scale_vec = _mm256_set1_ps(block_scale);

			__m128i q8_weights0			= _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset]));
			__m256i q8_weights0_256		= _mm256_cvtepi8_epi32(q8_weights0);
			__m256 dequantized_weights0 = _mm256_cvtepi32_ps(q8_weights0_256);
			dequantized_weights0		= _mm256_mul_ps(dequantized_weights0, w_scale_vec);

			__m128i q8_weights1			= _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset + 8]));
			__m256i q8_weights1_256		= _mm256_cvtepi8_epi32(q8_weights1);
			__m256 dequantized_weights1 = _mm256_cvtepi32_ps(q8_weights1_256);
			dequantized_weights1		= _mm256_mul_ps(dequantized_weights1, w_scale_vec);

			__m256 result0 = _mm256_mul_ps(normalized0, dequantized_weights0);
			__m256 result1 = _mm256_mul_ps(normalized1, dequantized_weights1);

			_mm256_store_ps(y, result0);
			_mm256_store_ps(y + 8, result1);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 16 && size_new < Q_SIZE)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			vec_add_rms_norm_mul_q8_f32<16>::impl(y, x, z, scale, w_scales, w_quants, w_offset);

			constexpr uint64_t remainder = size_new - 16ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_mul_q8_f32<remainder>::impl(y + 16, x + 16, z + 16, scale, w_scales, w_quants, w_offset + 16);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == Q_SIZE)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			const __m256 scale_vec	 = _mm256_set1_ps(scale);
			const float block_scale	 = w_scales[w_offset / Q_SIZE];
			const __m256 w_scale_vec = _mm256_set1_ps(block_scale);

			for (uint64_t i = 0; i < Q_SIZE; i += 8) {
				__m256 ax	 = _mm256_load_ps(x + i);
				__m256 az	 = _mm256_load_ps(z + i);
				__m256 added = _mm256_add_ps(ax, az);

				__m256 normalized = _mm256_mul_ps(added, scale_vec);

				__m128i q8_weights		   = _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset + i]));
				__m256i q8_weights_256	   = _mm256_cvtepi8_epi32(q8_weights);
				__m256 dequantized_weights = _mm256_cvtepi32_ps(q8_weights_256);
				dequantized_weights		   = _mm256_mul_ps(dequantized_weights, w_scale_vec);

				__m256 result = _mm256_mul_ps(normalized, dequantized_weights);
				_mm256_store_ps(y + i, result);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new > Q_SIZE && size_new < 64)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			vec_add_rms_norm_mul_q8_f32<Q_SIZE>::impl(y, x, z, scale, w_scales, w_quants, w_offset);

			constexpr uint64_t remainder = size_new - 32ULL;
			if constexpr (remainder > 0) {
				vec_add_rms_norm_mul_q8_f32<remainder>::impl(y + Q_SIZE, x + Q_SIZE, z + Q_SIZE, scale, w_scales, w_quants, w_offset + Q_SIZE);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 64)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			for (uint64_t i = 0; i < 64; i += Q_SIZE) {
				_mm_prefetch(static_cast<const char*>(x + i + 64), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 64), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(&w_quants[w_offset + i + 64]), _MM_HINT_T0);

				const float block_scale	 = w_scales[(w_offset + i) / Q_SIZE];
				const __m256 w_scale_vec = _mm256_set1_ps(block_scale);

				for (uint64_t j = 0; j < Q_SIZE; j += 8) {
					const uint64_t idx = i + j;

					__m256 ax	 = _mm256_load_ps(x + idx);
					__m256 az	 = _mm256_load_ps(z + idx);
					__m256 added = _mm256_add_ps(ax, az);

					__m256 normalized = _mm256_mul_ps(added, scale_vec);

					__m128i q8_weights		   = _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset + idx]));
					__m256i q8_weights_256	   = _mm256_cvtepi8_epi32(q8_weights);
					__m256 dequantized_weights = _mm256_cvtepi32_ps(q8_weights_256);
					dequantized_weights		   = _mm256_mul_ps(dequantized_weights, w_scale_vec);

					__m256 result = _mm256_mul_ps(normalized, dequantized_weights);
					_mm256_store_ps(y + idx, result);
				}
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new > 64)
	struct vec_add_rms_norm_mul_q8_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z, const float scale, const float* __restrict w_scales,
			const int8_t* __restrict w_quants, uint64_t w_offset) {
			const __m256 scale_vec		 = _mm256_set1_ps(scale);
			static constexpr uint64_t np = size_new & ~63ULL;
			uint64_t i					 = 0;

			for (; i < np; i += 64) {
				_mm_prefetch(static_cast<const char*>(x + i + 128), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 128), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(&w_quants[w_offset + i + 128]), _MM_HINT_T0);

				for (uint64_t j = 0; j < 64; j += Q_SIZE) {
					const uint64_t block_idx = (w_offset + i + j) / Q_SIZE;
					const float block_scale	 = w_scales[block_idx];
					const __m256 w_scale_vec = _mm256_set1_ps(block_scale);

					for (uint64_t k = 0; k < Q_SIZE; k += 8) {
						const uint64_t idx = i + j + k;

						__m256 ax	 = _mm256_load_ps(x + idx);
						__m256 az	 = _mm256_load_ps(z + idx);
						__m256 added = _mm256_add_ps(ax, az);

						__m256 normalized = _mm256_mul_ps(added, scale_vec);

						__m128i q8_weights		   = _mm_loadl_epi64(static_cast<const __m128i*>(&w_quants[w_offset + idx]));
						__m256i q8_weights_256	   = _mm256_cvtepi8_epi32(q8_weights);
						__m256 dequantized_weights = _mm256_cvtepi32_ps(q8_weights_256);
						dequantized_weights		   = _mm256_mul_ps(dequantized_weights, w_scale_vec);

						__m256 result = _mm256_mul_ps(normalized, dequantized_weights);
						_mm256_stream_ps(y + idx, result);
					}
				}
			}

			if (i < size_new) {
				constexpr uint64_t remainder = size_new % 64ULL;
				if constexpr (remainder > 0) {
					vec_add_rms_norm_mul_q8_f32<remainder>::impl(y + i, x + i, z + i, scale, w_scales, w_quants, w_offset + i);
				}
			}
		}
	};

	NIHILUS_INLINE float simd_sum_squares_add_for_q8(const float* __restrict x_data, const float* __restrict z_data, uint64_t size) {
		__m256 sum_vec			= _mm256_setzero_ps();
		const uint64_t simd_end = size & ~7ULL;

		for (uint64_t i = 0; i < simd_end; i += 8) {
			__m256 x_vec = _mm256_load_ps(&x_data[i]);
			__m256 z_vec = _mm256_load_ps(&z_data[i]);
			__m256 added = _mm256_add_ps(x_vec, z_vec);
			sum_vec		 = _mm256_fmadd_ps(added, added, sum_vec);
		}

		__m128 sum_high	  = _mm256_extractf128_ps(sum_vec, 1);
		__m128 sum_low	  = _mm256_castps256_ps128(sum_vec);
		__m128 sum_quad	  = _mm_add_ps(sum_low, sum_high);
		__m128 sum_dual	  = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
		__m128 sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 1));
		float sum		  = _mm_cvtss_f32(sum_single);

		for (uint64_t i = simd_end; i < size; ++i) {
			const float added = x_data[i] + z_data[i];
			sum += added * added;
		}

		return sum;
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::prompt_eval_time, core_traits_type, float, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, float, block_q8_0<half>> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;
		using input_type03 = typename core_traits_type::input_03_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t thread_index, uint64_t thread_count, uint64_t ne01, uint64_t ne11, uint64_t ne21,
			const float* __restrict src0_data, const float* __restrict src1_data, const block_q8_0<half>* __restrict src2_data, float* __restrict dst_data) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];
			static constexpr uint64_t ne20 = input_type03::get_array()[0];
			static constexpr uint64_t ne22 = input_type03::get_array()[2];
			static constexpr uint64_t ne23 = input_type03::get_array()[3];

			const uint64_t src0_stride_03			 = ne02 * ne01 * ne00;
			const uint64_t src0_stride_02			 = ne01 * ne00;
			static constexpr uint64_t src0_stride_01 = ne00;

			const uint64_t src1_stride_13			 = ne12 * ne11 * ne10;
			const uint64_t src1_stride_12			 = ne11 * ne10;
			static constexpr uint64_t src1_stride_11 = ne10;

			const uint64_t src2_stride_23			 = ne22 * ne21 * (ne20 / Q_SIZE);
			const uint64_t src2_stride_22			 = ne21 * (ne20 / Q_SIZE);
			static constexpr uint64_t src2_stride_21 = ne20 / Q_SIZE;

			const uint64_t dst_stride_03			= ne02 * ne01 * ne00;
			const uint64_t dst_stride_02			= ne01 * ne00;
			static constexpr uint64_t dst_stride_01 = ne00;

			static constexpr float eps = core_traits_type::model_traits_type::layer_norm_rms_epsilon;
			const uint64_t total_rows  = ne01 * ne02 * ne03;

			if constexpr (is_power_of_2able) {
				const uint64_t log2_ne01			= tzcnt(ne01);
				static constexpr uint64_t log2_ne02 = tzcnt_constexpr(ne02);
				const uint64_t log2_ne11			= tzcnt(ne11);
				static constexpr uint64_t log2_ne12 = tzcnt_constexpr(ne12);
				static constexpr uint64_t log2_ne13 = tzcnt_constexpr(ne13);
				static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
				const uint64_t log2_ne21			= tzcnt(ne21);
				static constexpr uint64_t log2_ne22 = tzcnt_constexpr(ne22);
				static constexpr uint64_t log2_ne23 = tzcnt_constexpr(ne23);
				static constexpr uint64_t log2_ne20 = tzcnt_constexpr(ne20);
				const uint64_t log2_ne02_ne01		= log2_ne02 + log2_ne01;

				for (uint64_t row_idx = thread_index; row_idx < total_rows; row_idx += thread_count) {
					const uint64_t i03 = row_idx >> log2_ne02_ne01;
					const uint64_t i02 = (row_idx - (i03 << log2_ne02_ne01)) >> log2_ne01;
					const uint64_t i01 = row_idx - (i03 << log2_ne02_ne01) - (i02 << log2_ne01);
					const uint64_t i13 = i03 & ((1ULL << log2_ne13) - 1ULL);
					const uint64_t i12 = i02 & ((1ULL << log2_ne12) - 1ULL);
					const uint64_t i11 = i01 & ((1ULL << log2_ne11) - 1ULL);
					const uint64_t i23 = i03 & ((1ULL << log2_ne23) - 1ULL);
					const uint64_t i22 = i02 & ((1ULL << log2_ne22) - 1ULL);
					const uint64_t i21 = i01 & ((1ULL << log2_ne21) - 1ULL);

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t src2_offset = i23 * src2_stride_23 + i22 * src2_stride_22 + i21 * src2_stride_21;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr			= &src0_data[src0_offset];
					const float* __restrict src1_ptr			= &src1_data[src1_offset];
					const block_q8_0<half>* __restrict src2_ptr = &src2_data[src2_offset];
					float* __restrict dst_ptr					= &dst_data[dst_offset];

					const float sum	  = simd_sum_squares_add_for_q8(src0_ptr, src1_ptr, ne00);
					const float mean  = sum / static_cast<float>(ne00);
					const float scale = 1.0f / sqrtf_fast(mean + eps);

					float w_scales[ne00 / Q_SIZE];
					for (uint64_t block_idx = 0; block_idx < ne00 / Q_SIZE; ++block_idx) {
						w_scales[block_idx] = static_cast<float>(src2_ptr[block_idx].d);
					}
					const int8_t* w_quants = static_cast<const int8_t*>(&src2_ptr[0].qs[0]);

					if constexpr (ne10 == ne00 && ne20 == ne00) {
						const uint64_t nr0 = ne00 >> log2_ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							const uint64_t offset = r << log2_ne10;
							vec_add_rms_norm_mul_q8_f32<ne10>::impl(dst_ptr + offset, src0_ptr + offset, src1_ptr + offset, scale, w_scales, w_quants, offset);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10			   = i0 & ((1ULL << log2_ne10) - 1ULL);
							const uint64_t i20			   = i0 & ((1ULL << log2_ne20) - 1ULL);
							const float added			   = src0_ptr[i0] + src1_ptr[i10];
							const float normalized		   = added * scale;
							const float block_scale		   = static_cast<float>(src2_ptr[i20 / Q_SIZE].d);
							const float dequantized_weight = static_cast<float>(src2_ptr[i20 / Q_SIZE].qs[i20 % Q_SIZE]) * block_scale;
							dst_ptr[i0]					   = normalized * dequantized_weight;
						}
					}
				}
			} else {
				for (uint64_t row_idx = thread_index; row_idx < total_rows; row_idx += thread_count) {
					const uint64_t i03 = row_idx / (ne02 * ne01);
					const uint64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
					const uint64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;
					const uint64_t i13 = i03 % ne13;
					const uint64_t i12 = i02 % ne12;
					const uint64_t i11 = i01 % ne11;
					const uint64_t i23 = i03 % ne23;
					const uint64_t i22 = i02 % ne22;
					const uint64_t i21 = i01 % ne21;

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t src2_offset = i23 * src2_stride_23 + i22 * src2_stride_22 + i21 * src2_stride_21;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr			= &src0_data[src0_offset];
					const float* __restrict src1_ptr			= &src1_data[src1_offset];
					const block_q8_0<half>* __restrict src2_ptr = &src2_data[src2_offset];
					float* __restrict dst_ptr					= &dst_data[dst_offset];

					const float sum	  = simd_sum_squares_add_for_q8(src0_ptr, src1_ptr, ne00);
					const float mean  = sum / static_cast<float>(ne00);
					const float scale = 1.0f / sqrtf_fast(mean + eps);

					float w_scales[ne00 / Q_SIZE];
					for (uint64_t block_idx = 0; block_idx < ne00 / Q_SIZE; ++block_idx) {
						w_scales[block_idx] = static_cast<float>(src2_ptr[block_idx].d);
					}
					const int8_t* w_quants = static_cast<const int8_t*>(&src2_ptr[0].qs[0]);

					if constexpr (ne10 == ne00 && ne20 == ne00) {
						const uint64_t nr0 = ne00 / ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							const uint64_t offset = r * ne10;
							vec_add_rms_norm_mul_q8_f32<ne10>::impl(dst_ptr + offset, src0_ptr + offset, src1_ptr + offset, scale, w_scales, w_quants, offset);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10			   = i0 % ne10;
							const uint64_t i20			   = i0 % ne20;
							const float added			   = src0_ptr[i0] + src1_ptr[i10];
							const float normalized		   = added * scale;
							const float block_scale		   = static_cast<float>(src2_ptr[i20 / Q_SIZE].d);
							const float dequantized_weight = static_cast<float>(src2_ptr[i20 / Q_SIZE].qs[i20 % Q_SIZE]) * block_scale;
							dst_ptr[i0]					   = normalized * dequantized_weight;
						}
					}
				}
			}
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];
			static constexpr uint64_t ne20 = input_type03::get_array()[0];
			const uint64_t ne21			   = input03[1];
			static constexpr uint64_t ne22 = input_type03::get_array()[2];
			static constexpr uint64_t ne23 = input_type03::get_array()[3];

			const uint64_t thread_index						 = static_cast<uint64_t>(thread_index);
			const uint64_t thread_count						 = static_cast<uint64_t>(thread_count);
			const float* __restrict src01			 = input02.data;
			const block_q8_0<half>* __restrict src02 = input03.data[current_block];

			const bool is_power_of_2 = ((ne01 & (ne01 - 1ULL)) == 0) && ((ne02 & (ne02 - 1ULL)) == 0) && ((ne10 & (ne10 - 1ULL)) == 0) && ((ne11 & (ne11 - 1ULL)) == 0) &&
				((ne12 & (ne12 - 1ULL)) == 0) && ((ne13 & (ne13 - 1ULL)) == 0) && ((ne20 & (ne20 - 1ULL)) == 0) && ((ne21 & (ne21 - 1ULL)) == 0) && ((ne22 & (ne22 - 1ULL)) == 0) &&
				((ne23 & (ne23 - 1ULL)) == 0);

			if (is_power_of_2) {
				process_tensor_elements<true>(thread_index, thread_count, ne01, ne11, ne21, input01.data, src01, src02, output.data);
			} else {
				process_tensor_elements<false>(thread_index, thread_count, ne01, ne11, ne21, input01.data, src01, src02, output.data);
			}
		}
	};

	NIHILUS_INLINE float simd_sum_squares(const float* __restrict data, uint64_t size) {
		__m256 sum_vec			= _mm256_setzero_ps();
		const uint64_t simd_end = size & ~7ULL;

		for (uint64_t i = 0; i < simd_end; i += 8) {
			__m256 x_vec = _mm256_load_ps(&data[i]);
			sum_vec		 = _mm256_fmadd_ps(x_vec, x_vec, sum_vec);
		}

		__m128 sum_high	  = _mm256_extractf128_ps(sum_vec, 1);
		__m128 sum_low	  = _mm256_castps256_ps128(sum_vec);
		__m128 sum_quad	  = _mm_add_ps(sum_low, sum_high);
		__m128 sum_dual	  = _mm_add_ps(sum_quad, _mm_movehl_ps(sum_quad, sum_quad));
		__m128 sum_single = _mm_add_ss(sum_dual, _mm_shuffle_ps(sum_dual, sum_dual, 1));
		float sum		  = _mm_cvtss_f32(sum_single);

		for (uint64_t i = simd_end; i < size; ++i) {
			sum += data[i] * data[i];
		}

		return sum;
	}

	template<uint64_t size_new> struct vec_scale_mul_f32 {};

	template<uint64_t size_new>
		requires(size_new > 0 && size_new < 4)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			for (uint64_t i = 0; i < size_new; ++i) {
				y[i] = x[i] * scale * z[i];
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 4)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			const __m128 scale_vec = _mm_set1_ps(scale);
			__m128 ax0			   = _mm_load_ps(x);
			__m128 az0			   = _mm_load_ps(z);
			__m128 temp			   = _mm_mul_ps(ax0, az0);
			__m128 ay0			   = _mm_mul_ps(temp, scale_vec);
			_mm_store_ps(y, ay0);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 4 && size_new < 8)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			vec_scale_mul_f32<4>::impl(y, x, scale, z);

			constexpr uint64_t remainder = size_new - 8ULL;
			if constexpr (remainder > 0) {
				vec_scale_mul_f32<remainder>::impl(y + 4, x + 4, scale, z + 4);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 8)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			const __m256 scale_vec = _mm256_set1_ps(scale);
			__m256 ax0			   = _mm256_load_ps(x);
			__m256 az0			   = _mm256_load_ps(z);
			__m256 temp			   = _mm256_mul_ps(ax0, az0);
			__m256 ay0			   = _mm256_mul_ps(temp, scale_vec);
			_mm256_store_ps(y, ay0);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 8 && size_new < 16)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			vec_scale_mul_f32<8>::impl(y, x, scale, z);

			constexpr uint64_t remainder = size_new - 8ULL;
			if constexpr (remainder > 0) {
				vec_scale_mul_f32<remainder>::impl(y + 8, x + 8, scale, z + 8);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 16)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);

			__m256 temp0 = _mm256_mul_ps(ax0, az0);
			__m256 temp1 = _mm256_mul_ps(ax1, az1);
			__m256 ay0	 = _mm256_mul_ps(temp0, scale_vec);
			__m256 ay1	 = _mm256_mul_ps(temp1, scale_vec);

			_mm256_store_ps(y, ay0);
			_mm256_store_ps(y + 8, ay1);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 16 && size_new < Q_SIZE)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			vec_scale_mul_f32<16>::impl(y, x, scale, z);

			constexpr uint64_t remainder = size_new - 16ULL;
			if constexpr (remainder > 0) {
				vec_scale_mul_f32<remainder>::impl(y + 16, x + 16, scale, z + 16);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == Q_SIZE)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 ax2 = _mm256_load_ps(x + 16);
			__m256 ax3 = _mm256_load_ps(x + 24);

			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);
			__m256 az2 = _mm256_load_ps(z + 16);
			__m256 az3 = _mm256_load_ps(z + 24);

			__m256 temp0 = _mm256_mul_ps(ax0, az0);
			__m256 temp1 = _mm256_mul_ps(ax1, az1);
			__m256 temp2 = _mm256_mul_ps(ax2, az2);
			__m256 temp3 = _mm256_mul_ps(ax3, az3);

			__m256 ay0 = _mm256_mul_ps(temp0, scale_vec);
			__m256 ay1 = _mm256_mul_ps(temp1, scale_vec);
			__m256 ay2 = _mm256_mul_ps(temp2, scale_vec);
			__m256 ay3 = _mm256_mul_ps(temp3, scale_vec);

			_mm256_store_ps(y, ay0);
			_mm256_store_ps(y + 8, ay1);
			_mm256_store_ps(y + 16, ay2);
			_mm256_store_ps(y + 24, ay3);
		}
	};

	template<uint64_t size_new>
		requires(size_new > Q_SIZE && size_new < 64)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			vec_scale_mul_f32<32>::impl(y, x, scale, z);

			constexpr uint64_t remainder = size_new - 32ULL;
			if constexpr (remainder > 0) {
				vec_scale_mul_f32<remainder>::impl(y + Q_SIZE, x + Q_SIZE, scale, z + Q_SIZE);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 64)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			const __m256 scale_vec = _mm256_set1_ps(scale);

			for (uint64_t i = 0; i < 64ULL; i += 32ULL) {
				_mm_prefetch(static_cast<const char*>(x + i + 64ULL), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 64ULL), _MM_HINT_T0);

				__m256 ax0 = _mm256_load_ps(x + i);
				__m256 ax1 = _mm256_load_ps(x + i + 8ULL);
				__m256 ax2 = _mm256_load_ps(x + i + 16ULL);
				__m256 ax3 = _mm256_load_ps(x + i + 24ULL);

				__m256 az0 = _mm256_load_ps(z + i);
				__m256 az1 = _mm256_load_ps(z + i + 8ULL);
				__m256 az2 = _mm256_load_ps(z + i + 16ULL);
				__m256 az3 = _mm256_load_ps(z + i + 24ULL);

				__m256 temp0 = _mm256_mul_ps(ax0, az0);
				__m256 temp1 = _mm256_mul_ps(ax1, az1);
				__m256 temp2 = _mm256_mul_ps(ax2, az2);
				__m256 temp3 = _mm256_mul_ps(ax3, az3);

				__m256 ay0 = _mm256_mul_ps(temp0, scale_vec);
				__m256 ay1 = _mm256_mul_ps(temp1, scale_vec);
				__m256 ay2 = _mm256_mul_ps(temp2, scale_vec);
				__m256 ay3 = _mm256_mul_ps(temp3, scale_vec);

				_mm256_store_ps(y + i, ay0);
				_mm256_store_ps(y + i + 8ULL, ay1);
				_mm256_store_ps(y + i + 16ULL, ay2);
				_mm256_store_ps(y + i + 24ULL, ay3);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new > 64)
	struct vec_scale_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float scale, const float* __restrict z) {
			const __m256 scale_vec		 = _mm256_set1_ps(scale);
			static constexpr uint64_t np = size_new & ~63ULL;
			uint64_t i					 = 0;

			for (; i < np; i += 64ULL) {
				_mm_prefetch(static_cast<const char*>(x + i + 128ULL), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 128ULL), _MM_HINT_T0);

				for (uint64_t j = 0; j < 64ULL; j += 32ULL) {
					__m256 ax0 = _mm256_load_ps(x + i + j);
					__m256 ax1 = _mm256_load_ps(x + i + j + 8ULL);
					__m256 ax2 = _mm256_load_ps(x + i + j + 16ULL);
					__m256 ax3 = _mm256_load_ps(x + i + j + 24ULL);

					__m256 az0 = _mm256_load_ps(z + i + j);
					__m256 az1 = _mm256_load_ps(z + i + j + 8ULL);
					__m256 az2 = _mm256_load_ps(z + i + j + 16ULL);
					__m256 az3 = _mm256_load_ps(z + i + j + 24ULL);

					__m256 temp0 = _mm256_mul_ps(ax0, az0);
					__m256 temp1 = _mm256_mul_ps(ax1, az1);
					__m256 temp2 = _mm256_mul_ps(ax2, az2);
					__m256 temp3 = _mm256_mul_ps(ax3, az3);

					__m256 ay0 = _mm256_mul_ps(temp0, scale_vec);
					__m256 ay1 = _mm256_mul_ps(temp1, scale_vec);
					__m256 ay2 = _mm256_mul_ps(temp2, scale_vec);
					__m256 ay3 = _mm256_mul_ps(temp3, scale_vec);

					_mm256_stream_ps(y + i + j, ay0);
					_mm256_stream_ps(y + i + j + 8ULL, ay1);
					_mm256_stream_ps(y + i + j + 16ULL, ay2);
					_mm256_stream_ps(y + i + j + 24ULL, ay3);
				}
			}

			if (i < size_new) {
				constexpr uint64_t remainder = size_new % 64ULL;
				if constexpr (remainder > 0) {
					vec_scale_mul_f32<remainder>::impl(y + i, x + i, scale, z + i);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rms_norm_mul_transpose, processing_phases::prompt_eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::rms_norm_mul_transpose, core_traits_type, float, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t thread_index, uint64_t thread_count, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t src0_stride_03			 = ne02 * ne01 * ne00;
			const uint64_t src0_stride_02			 = ne01 * ne00;
			static constexpr uint64_t src0_stride_01 = ne00;

			const uint64_t src1_stride_13			 = ne12 * ne11 * ne10;
			const uint64_t src1_stride_12			 = ne11 * ne10;
			static constexpr uint64_t src1_stride_11 = ne10;

			const uint64_t dst_stride_03			= ne02 * ne01 * ne00;
			const uint64_t dst_stride_02			= ne01 * ne00;
			static constexpr uint64_t dst_stride_01 = ne00;

			static constexpr float eps = core_traits_type::model_traits_type::layer_norm_rms_epsilon;
			const uint64_t total_rows  = ne01 * ne02 * ne03;

			if constexpr (is_power_of_2able) {
				const uint64_t log2_ne01			= tzcnt(ne01);
				static constexpr uint64_t log2_ne02 = tzcnt_constexpr(ne02);
				const uint64_t log2_ne11			= tzcnt(ne11);
				static constexpr uint64_t log2_ne12 = tzcnt_constexpr(ne12);
				static constexpr uint64_t log2_ne13 = tzcnt_constexpr(ne13);
				static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
				const uint64_t log2_ne02_ne01		= log2_ne02 + log2_ne01;

				for (uint64_t row_idx = thread_index; row_idx < total_rows; row_idx += thread_count) {
					const uint64_t i03 = row_idx >> log2_ne02_ne01;
					const uint64_t i02 = (row_idx - (i03 << log2_ne02_ne01)) >> log2_ne01;
					const uint64_t i01 = row_idx - (i03 << log2_ne02_ne01) - (i02 << log2_ne01);
					const uint64_t i13 = i03 & ((1ULL << log2_ne13) - 1ULL);
					const uint64_t i12 = i02 & ((1ULL << log2_ne12) - 1ULL);
					const uint64_t i11 = i01 & ((1ULL << log2_ne11) - 1ULL);

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr = &src0_data[src0_offset];
					const float* __restrict src1_ptr = &src1_data[src1_offset];
					float* __restrict dst_ptr		 = &dst_data[dst_offset];

					const float sum	  = simd_sum_squares(src0_ptr, ne00);
					const float mean  = sum / static_cast<float>(ne00);
					const float scale = 1.0f / sqrtf_fast(mean + eps);

					if constexpr (ne10 == ne00) {
						const uint64_t nr0 = ne00 >> log2_ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							vec_scale_mul_f32<ne10>::impl(dst_ptr + (r << log2_ne10), src0_ptr + (r << log2_ne10), scale, src1_ptr);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10 = i0 & ((1ULL << log2_ne10) - 1ULL);
							dst_ptr[i0]		   = src0_ptr[i0] * scale * src1_ptr[i10];
						}
					}
				}
			} else {
				for (uint64_t row_idx = thread_index; row_idx < total_rows; row_idx += thread_count) {
					const uint64_t i03 = row_idx / (ne02 * ne01);
					const uint64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
					const uint64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;
					const uint64_t i13 = i03 % ne13;
					const uint64_t i12 = i02 % ne12;
					const uint64_t i11 = i01 % ne11;

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr = &src0_data[src0_offset];
					const float* __restrict src1_ptr = &src1_data[src1_offset];
					float* __restrict dst_ptr		 = &dst_data[dst_offset];

					const float sum	  = simd_sum_squares(src0_ptr, ne00);
					const float mean  = sum / static_cast<float>(ne00);
					const float scale = 1.0f / sqrtf_fast(mean + eps);

					if constexpr (ne10 == ne00) {
						const uint64_t nr0 = ne00 / ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							vec_scale_mul_f32<ne10>::impl(dst_ptr + r * ne10, src0_ptr + r * ne10, scale, src1_ptr);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10 = i0 % ne10;
							dst_ptr[i0]		   = src0_ptr[i0] * scale * src1_ptr[i10];
						}
					}
				}
			}
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t thread_index			  = static_cast<uint64_t>(thread_index);
			const uint64_t thread_count			  = static_cast<uint64_t>(thread_count);
			const float* __restrict src01 = input02.data[current_block];

			const bool is_power_of_2 = ((ne01 & (ne01 - 1ULL)) == 0) && ((ne02 & (ne02 - 1ULL)) == 0) && ((ne10 & (ne10 - 1ULL)) == 0) && ((ne11 & (ne11 - 1ULL)) == 0) &&
				((ne12 & (ne12 - 1ULL)) == 0) && ((ne13 & (ne13 - 1ULL)) == 0);

			if (is_power_of_2) {
				process_tensor_elements<true>(thread_index, thread_count, ne01, ne11, input01.data, src01, output.data);
			} else {
				process_tensor_elements<false>(thread_index, thread_count, ne01, ne11, input01.data, src01, output.data);
			}
		}
	};

	template<uint64_t size_new> struct vec_mul_f32 {};

	template<uint64_t size_new>
		requires(size_new > 0 && size_new < 8)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			for (uint64_t i = 0; i < size_new; ++i) {
				y[i] = x[i] * z[i];
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 8)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			__m256 ax0 = _mm256_load_ps(x);
			__m256 az0 = _mm256_load_ps(z);
			__m256 ay0 = _mm256_mul_ps(ax0, az0);
			_mm256_store_ps(y, ay0);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 8 && size_new < 16)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			vec_mul_f32<8>::impl(y, x, z);

			constexpr uint64_t remainder = size_new - 8ULL;
			if constexpr (remainder > 0) {
				vec_mul_f32<remainder>::impl(y + 8, x + 8, z + 8);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 16)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);

			__m256 ay0 = _mm256_mul_ps(ax0, az0);
			__m256 ay1 = _mm256_mul_ps(ax1, az1);

			_mm256_store_ps(y, ay0);
			_mm256_store_ps(y + 8, ay1);
		}
	};

	template<uint64_t size_new>
		requires(size_new > 16 && size_new < Q_SIZE)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			vec_mul_f32<16>::impl(y, x, z);

			constexpr uint64_t remainder = size_new - 16ULL;
			if constexpr (remainder > 0) {
				vec_mul_f32<remainder>::impl(y + 16, x + 16, z + 16);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == Q_SIZE)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			__m256 ax0 = _mm256_load_ps(x);
			__m256 ax1 = _mm256_load_ps(x + 8);
			__m256 ax2 = _mm256_load_ps(x + 16);
			__m256 ax3 = _mm256_load_ps(x + 24);

			__m256 az0 = _mm256_load_ps(z);
			__m256 az1 = _mm256_load_ps(z + 8);
			__m256 az2 = _mm256_load_ps(z + 16);
			__m256 az3 = _mm256_load_ps(z + 24);

			__m256 ay0 = _mm256_mul_ps(ax0, az0);
			__m256 ay1 = _mm256_mul_ps(ax1, az1);
			__m256 ay2 = _mm256_mul_ps(ax2, az2);
			__m256 ay3 = _mm256_mul_ps(ax3, az3);

			_mm256_store_ps(y, ay0);
			_mm256_store_ps(y + 8, ay1);
			_mm256_store_ps(y + 16, ay2);
			_mm256_store_ps(y + 24, ay3);
		}
	};

	template<uint64_t size_new>
		requires(size_new > Q_SIZE && size_new < 64)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			vec_mul_f32<32>::impl(y, x, z);

			constexpr uint64_t remainder = size_new - 32ULL;
			if constexpr (remainder > 0) {
				vec_mul_f32<remainder>::impl(y + Q_SIZE, x + Q_SIZE, z + Q_SIZE);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new == 64)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			for (uint64_t i = 0; i < 64ULL; i += 32ULL) {
				_mm_prefetch(static_cast<const char*>(x + i + 64ULL), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 64ULL), _MM_HINT_T0);

				__m256 ax0 = _mm256_load_ps(x + i);
				__m256 ax1 = _mm256_load_ps(x + i + 8ULL);
				__m256 ax2 = _mm256_load_ps(x + i + 16ULL);
				__m256 ax3 = _mm256_load_ps(x + i + 24ULL);

				__m256 az0 = _mm256_load_ps(z + i);
				__m256 az1 = _mm256_load_ps(z + i + 8ULL);
				__m256 az2 = _mm256_load_ps(z + i + 16ULL);
				__m256 az3 = _mm256_load_ps(z + i + 24ULL);

				__m256 ay0 = _mm256_mul_ps(ax0, az0);
				__m256 ay1 = _mm256_mul_ps(ax1, az1);
				__m256 ay2 = _mm256_mul_ps(ax2, az2);
				__m256 ay3 = _mm256_mul_ps(ax3, az3);

				_mm256_store_ps(y + i, ay0);
				_mm256_store_ps(y + i + 8ULL, ay1);
				_mm256_store_ps(y + i + 16ULL, ay2);
				_mm256_store_ps(y + i + 24ULL, ay3);
			}
		}
	};

	template<uint64_t size_new>
		requires(size_new > 64)
	struct vec_mul_f32<size_new> {
		NIHILUS_INLINE static void impl(float* __restrict y, const float* __restrict x, const float* __restrict z) {
			const uint64_t np = size_new & ~63ULL;
			uint64_t i		  = 0;

			for (; i < np; i += 64ULL) {
				_mm_prefetch(static_cast<const char*>(x + i + 128ULL), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(z + i + 128ULL), _MM_HINT_T0);

				for (uint64_t j = 0; j < 64ULL; j += 32ULL) {
					__m256 ax0 = _mm256_load_ps(x + i + j);
					__m256 ax1 = _mm256_load_ps(x + i + j + 8ULL);
					__m256 ax2 = _mm256_load_ps(x + i + j + 16ULL);
					__m256 ax3 = _mm256_load_ps(x + i + j + 24ULL);

					__m256 az0 = _mm256_load_ps(z + i + j);
					__m256 az1 = _mm256_load_ps(z + i + j + 8ULL);
					__m256 az2 = _mm256_load_ps(z + i + j + 16ULL);
					__m256 az3 = _mm256_load_ps(z + i + j + 24ULL);

					__m256 ay0 = _mm256_mul_ps(ax0, az0);
					__m256 ay1 = _mm256_mul_ps(ax1, az1);
					__m256 ay2 = _mm256_mul_ps(ax2, az2);
					__m256 ay3 = _mm256_mul_ps(ax3, az3);

					_mm256_stream_ps(y + i + j, ay0);
					_mm256_stream_ps(y + i + j + 8ULL, ay1);
					_mm256_stream_ps(y + i + j + 16ULL, ay2);
					_mm256_stream_ps(y + i + j + 24ULL, ay3);
				}
			}

			if (i < size_new) {
				constexpr uint64_t remainder = size_new % 64ULL;
				if constexpr (remainder > 0) {
					vec_mul_f32<remainder>::impl(y + i, x + i, z + i);
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul, processing_phases::prompt_eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_traits_type, float, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t thread_index, uint64_t thread_count, uint64_t nr, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t src0_stride_03			 = ne02 * ne01 * ne00;
			const uint64_t src0_stride_02			 = ne01 * ne00;
			static constexpr uint64_t src0_stride_01 = ne00;

			const uint64_t src1_stride_13			 = ne12 * ne11 * ne10;
			const uint64_t src1_stride_12			 = ne11 * ne10;
			static constexpr uint64_t src1_stride_11 = ne10;

			const uint64_t dst_stride_03			= ne02 * ne01 * ne00;
			const uint64_t dst_stride_02			= ne01 * ne00;
			static constexpr uint64_t dst_stride_01 = ne00;

			if constexpr (is_power_of_2able) {
				const uint64_t log2_ne01			= tzcnt(ne01);
				static constexpr uint64_t log2_ne02 = tzcnt_constexpr(ne02);
				const uint64_t log2_ne11			= tzcnt(ne11);
				static constexpr uint64_t log2_ne12 = tzcnt_constexpr(ne12);
				static constexpr uint64_t log2_ne13 = tzcnt_constexpr(ne13);
				static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);

				const uint64_t log2_ne02_ne01 = log2_ne02 + log2_ne01;

				for (uint64_t ir = thread_index; ir < nr; ir += thread_count) {
					const uint64_t i03 = ir >> log2_ne02_ne01;
					const uint64_t i02 = (ir - (i03 << log2_ne02_ne01)) >> log2_ne01;
					const uint64_t i01 = ir - (i03 << log2_ne02_ne01) - (i02 << log2_ne01);

					const uint64_t i13 = i03 & ((1ULL << log2_ne13) - 1ULL);
					const uint64_t i12 = i02 & ((1ULL << log2_ne12) - 1ULL);
					const uint64_t i11 = i01 & ((1ULL << log2_ne11) - 1ULL);

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr = &src0_data[src0_offset];
					const float* __restrict src1_ptr = &src1_data[src1_offset];
					float* __restrict dst_ptr		 = &dst_data[dst_offset];

					if constexpr (ne10 == ne00) {
						const uint64_t nr0 = ne00 >> log2_ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							vec_mul_f32<ne10>::impl(dst_ptr + (r << log2_ne10), src0_ptr + (r << log2_ne10), src1_ptr);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10 = i0 & ((1ULL << log2_ne10) - 1ULL);
							dst_ptr[i0]		   = src0_ptr[i0] * src1_ptr[i10];
						}
					}
				}
			} else {
				for (uint64_t ir = thread_index; ir < nr; ir += thread_count) {
					const uint64_t i03 = ir / (ne02 * ne01);
					const uint64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
					const uint64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

					const uint64_t i13 = i03 % ne13;
					const uint64_t i12 = i02 % ne12;
					const uint64_t i11 = i01 % ne11;

					const uint64_t src0_offset = i03 * src0_stride_03 + i02 * src0_stride_02 + i01 * src0_stride_01;
					const uint64_t src1_offset = i13 * src1_stride_13 + i12 * src1_stride_12 + i11 * src1_stride_11;
					const uint64_t dst_offset  = i03 * dst_stride_03 + i02 * dst_stride_02 + i01 * dst_stride_01;

					const float* __restrict src0_ptr = &src0_data[src0_offset];
					const float* __restrict src1_ptr = &src1_data[src1_offset];
					float* __restrict dst_ptr		 = &dst_data[dst_offset];

					if constexpr (ne10 == ne00) {
						const uint64_t nr0 = ne00 / ne10;
						for (uint64_t r = 0; r < nr0; ++r) {
							vec_mul_f32<ne10>::impl(dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
						}
					} else {
						for (uint64_t i0 = 0; i0 < ne00; ++i0) {
							const uint64_t i10 = i0 % ne10;
							dst_ptr[i0]		   = src0_ptr[i0] * src1_ptr[i10];
						}
					}
				}
			}
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t thread_index = static_cast<uint64_t>(thread_index);
			const uint64_t thread_count = static_cast<uint64_t>(thread_count);
			const uint64_t nr  = ne01 * ne02 * ne03;

			const bool is_power_of_2 = ((ne01 & (ne01 - 1ULL)) == 0) && ((ne02 & (ne02 - 1ULL)) == 0) && ((ne10 & (ne10 - 1ULL)) == 0) && ((ne11 & (ne11 - 1ULL)) == 0) &&
				((ne12 & (ne12 - 1ULL)) == 0) && ((ne13 & (ne13 - 1ULL)) == 0);

			if (is_power_of_2) {
				process_tensor_elements<true>(thread_index, thread_count, nr, ne01, ne11, input01.data, input02.data, output.data);
			} else {
				process_tensor_elements<false>(thread_index, thread_count, nr, ne01, ne11, input01.data, input02.data, output.data);
			}
		}
	};

	NIHILUS_INLINE static void dequantize_row_q8_0(const block_q8_0<half>* __restrict x, float* __restrict y, uint64_t k) {
		static constexpr int64_t qk = Q_SIZE;

		const uint64_t nb = k & ~(Q_SIZE - 1);

		for (uint64_t i = 0; i < nb; i++) {
			const float d = fp16_to_fp32(x[i].d);

			for (uint64_t j = 0; j < qk; ++j) {
				y[i * qk + j] = x[i].qs[j] * d;
			}
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::prompt_eval_time, core_traits_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_traits_type::input_01_type;
		using input_type02 = core_traits_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];

			const uint64_t nr  = count_elements(input02);
			const uint64_t thread_index = static_cast<uint64_t>(thread_index);
			const uint64_t thread_count = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + thread_count - 1ull) / thread_count;
			const uint64_t ir0 = dr * thread_index;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const block_q8_0<half>* __restrict input01_data = input01.data;
			const int32_t* __restrict input02_data			= input02.data;
			float* __restrict output_data					= output.data;

			static constexpr uint64_t blocks_per_row	  = ne00 / Q_SIZE;
			static constexpr uint64_t input01_stride_dim1 = blocks_per_row;
			const uint64_t input01_stride_dim2			  = ne01 * blocks_per_row;
			const uint64_t input01_stride_dim3			  = ne01 * ne02 * blocks_per_row;

			static constexpr uint64_t input02_stride_dim1 = 1;
			static constexpr uint64_t input02_stride_dim2 = ne10;
			const uint64_t input02_stride_dim3			  = ne10 * ne11;

			static constexpr uint64_t output_stride_dim1 = ne00;
			const uint64_t output_stride_dim2			 = output[1] * ne00;
			const uint64_t output_stride_dim3			 = output[1] * output[2] * ne00;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12 = i / (ne11 * ne10);
				const uint64_t i11 = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10 = i - i12 * ne11 * ne10 - i11 * ne10;

				const uint64_t input02_idx = i10 * input02_stride_dim1 + i11 * input02_stride_dim2 + i12 * input02_stride_dim3;
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * input01_stride_dim1 + i11 * input01_stride_dim2 + i12 * input01_stride_dim3;

				const uint64_t output_idx = i10 * output_stride_dim1 + i11 * output_stride_dim2 + i12 * output_stride_dim3;

				dequantize_row_q8_0(&input01_data[input01_block_idx], &output_data[output_idx], ne00);
			}
		}
	};

	NIHILUS_INLINE static void vec_cpy_f32(const uint64_t n, float* __restrict y, const float* __restrict x) {
		const uint64_t np = n & ~(Q_SIZE - 1);
		for (uint64_t i = 0; i < np; i += Q_SIZE) {
			__m256 ax0 = _mm256_load_ps(x + i);
			__m256 ax1 = _mm256_load_ps(x + i + 8);
			__m256 ax2 = _mm256_load_ps(x + i + 16);
			__m256 ax3 = _mm256_load_ps(x + i + 24);
			_mm256_store_ps(y + i, ax0);
			_mm256_store_ps(y + i + 8, ax1);
			_mm256_store_ps(y + i + 16, ax2);
			_mm256_store_ps(y + i + 24, ax3);
		}
		for (uint64_t i = np; i < n; ++i) {
			y[i] = x[i];
		}
	}

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::prompt_eval_time, core_traits_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			const uint64_t ne00 = input01[0];
			const uint64_t ne01 = input01[1];
			const uint64_t ne02 = input01[2];
			const uint64_t ne10 = input02[0];
			const uint64_t ne11 = input02[1];

			const uint64_t nr  = count_elements(input02);
			const uint64_t thread_index = static_cast<uint64_t>(thread_index);
			const uint64_t thread_count = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + thread_count - 1ull) / thread_count;
			const uint64_t ir0 = dr * thread_index;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const float* __restrict input01_data   = input01.data;
			const int32_t* __restrict input02_data = input02.data;
			float* __restrict output_data		   = output.data;

			const uint64_t blocks_per_row		   = ne00 / Q_SIZE;
			const uint64_t output_elements_per_row = ne00;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12		   = i / (ne11 * ne10);
				const uint64_t i11		   = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10		   = (i - i12 * ne11 * ne10 - i11 * ne10);
				const uint64_t input02_idx = i10 + i11 * ne10 + i12 * (ne10 * ne11);
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * blocks_per_row + i11 * (ne01 * blocks_per_row) + i12 * (ne01 * ne02 * blocks_per_row);
				const float* __restrict src_ptr	 = &input01_data[input01_block_idx];

				const uint64_t output_idx = i10 * output_elements_per_row + i11 * (output[1] * output_elements_per_row) + i12 * (output[1] * output[2] * output_elements_per_row);
				float* __restrict dst_ptr = &output_data[output_idx];

				vec_cpy_f32(ne00, dst_ptr, src_ptr);
			}
		}
	};

	NIHILUS_INLINE static half fp32_to_fp16_f16c(float f) {
		static constexpr float scale_to_inf	 = fp32_from_bits(0x77800000);
		static constexpr float scale_to_zero = fp32_from_bits(0x08800000);
		float base							 = (fabsf(f) * scale_to_inf) * scale_to_zero;

		const uint32_t w	  = fp32_to_bits(f);
		const uint32_t shl1_w = w + w;
		const uint32_t sign	  = w & UINT32_C(0x80000000);
		uint32_t bias		  = shl1_w & UINT32_C(0xFF000000);
		if (bias < UINT32_C(0x71000000)) {
			bias = UINT32_C(0x71000000);
		}

		base						 = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
		const uint32_t bits			 = fp32_to_bits(base);
		const uint32_t exp_bits		 = (bits >> 13) & UINT32_C(0x00007C00);
		const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
		const uint32_t nonsign		 = exp_bits + mantissa_bits;
		return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
	}

	NIHILUS_INLINE void quantize_row_q8_0(const float* __restrict x, block_q8_0<half>* __restrict vy, int64_t k) {
		const int64_t nb = k / Q_SIZE;

		block_q8_0<half>* __restrict y = vy;
		for (int64_t i = 0; i < nb; i++) {
			__m256 v0 = _mm256_load_ps(x);
			__m256 v1 = _mm256_load_ps(x + 8);
			__m256 v2 = _mm256_load_ps(x + 16);
			__m256 v3 = _mm256_load_ps(x + 24);
			x += 32;
			const __m256 signBit = _mm256_set1_ps(-0.0f);
			__m256 maxAbs		 = _mm256_andnot_ps(signBit, v0);
			maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
			maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
			maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

			__m128 max4			  = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
			max4				  = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
			max4				  = _mm_max_ss(max4, _mm_movehdup_ps(max4));
			const float maxScalar = _mm_cvtss_f32(max4);

			const float d	 = maxScalar / 127.f;
			y[i].d			 = fp32_to_fp16_f16c(d);
			const float id	 = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
			const __m256 mul = _mm256_set1_ps(id);

			v0 = _mm256_mul_ps(v0, mul);
			v1 = _mm256_mul_ps(v1, mul);
			v2 = _mm256_mul_ps(v2, mul);
			v3 = _mm256_mul_ps(v3, mul);

			v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
			v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
			v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
			v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

			__m256i i0 = _mm256_cvtps_epi32(v0);
			__m256i i1 = _mm256_cvtps_epi32(v1);
			__m256i i2 = _mm256_cvtps_epi32(v2);
			__m256i i3 = _mm256_cvtps_epi32(v3);

			i0 = _mm256_packs_epi32(i0, i1);
			i2 = _mm256_packs_epi32(i2, i3);
			i0 = _mm256_packs_epi16(i0, i2);

			const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
			i0				   = _mm256_permutevar8x32_epi32(i0, perm);

			_mm256_store_si256(( __m256i* )y[i].qs, i0);
		}
	}

	NIHILUS_INLINE float unhalf(half d) {
		return fp16_to_fp32(d);
	}

	NIHILUS_INLINE __m256 madd(__m256 a, __m256 b, __m256 c) {
		return _mm256_fmadd_ps(a, b, c);
	}

	NIHILUS_INLINE float hsum(__m128 x) {
		x = _mm_add_ps(x, _mm_movehl_ps(x, x));
		x = _mm_add_ss(x, _mm_movehdup_ps(x));
		return _mm_cvtss_f32(x);
	}

	NIHILUS_INLINE float hsum(__m256 x) {
		return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
	}

	NIHILUS_INLINE static __m256 sum_i16_pairs_float(const __m256i x) {
		const __m256i ones		   = _mm256_set1_epi16(1);
		const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
		return _mm256_cvtepi32_ps(summed_pairs);
	}

	NIHILUS_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
		const __m256i dot = _mm256_maddubs_epi16(ax, sy);
		return sum_i16_pairs_float(dot);
	}

	NIHILUS_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
		const __m256i ax = _mm256_sign_epi8(x, x);
		const __m256i sy = _mm256_sign_epi8(y, x);
		return mul_sum_us8_pairs_float(ax, sy);
	}

	NIHILUS_INLINE float hsum_float_8(const __m256 x) {
		__m128 res = _mm256_extractf128_ps(x, 1);
		res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
		res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
		res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
		return _mm_cvtss_f32(res);
	}

	template<uint64_t n_blocks> struct vec_dot_q8_0_q8_0 {};

	template<uint64_t n_blocks>
		requires(n_blocks > 0 && n_blocks < 4)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			const block_q8_0<half>* __restrict x = vx;
			const block_q8_0<half>* __restrict y = vy;
			float sumf							 = 0.0f;

			for (uint64_t ib = 0; ib < n_blocks; ++ib) {
				int64_t sumi = 0;
				for (int64_t j = 0; j < Q_SIZE; j++) {
					sumi += x[ib].qs[j] * y[ib].qs[j];
				}
				sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
			}
			*s = sumf;
		}
	};

	template<uint64_t n_blocks>
		requires(n_blocks == 4)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			const block_q8_0<half>* __restrict x = vx;
			const block_q8_0<half>* __restrict y = vy;
			__m256 acc							 = _mm256_setzero_ps();

			for (uint64_t ib = 0; ib < 4; ++ib) {
				const __m256 d = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
				__m256i qx	   = _mm256_load_si256(( const __m256i* )x[ib].qs);
				__m256i qy	   = _mm256_load_si256(( const __m256i* )y[ib].qs);
				const __m256 q = mul_sum_i8_pairs_float(qx, qy);
				acc			   = _mm256_fmadd_ps(d, q, acc);
			}
			*s = hsum_float_8(acc);
		}
	};

	template<uint64_t n_blocks>
		requires(n_blocks > 4 && n_blocks < 8)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			vec_dot_q8_0_q8_0<4>::impl(s, vx, vy);

			static constexpr uint64_t remainder = n_blocks - 4;
			if constexpr (remainder > 0) {
				float remainder_sum;
				vec_dot_q8_0_q8_0<remainder>::impl(&remainder_sum, vx + 4, vy + 4);
				*s += remainder_sum;
			}
		}
	};

	template<uint64_t n_blocks>
		requires(n_blocks == 8)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			const block_q8_0<half>* __restrict x = vx;
			const block_q8_0<half>* __restrict y = vy;
			__m256 acc0							 = _mm256_setzero_ps();
			__m256 acc1							 = _mm256_setzero_ps();

			for (uint64_t ib = 0; ib < 8; ib += 2) {
				const __m256 d0 = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
				const __m256 d1 = _mm256_set1_ps(fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d));

				__m256i qx0 = _mm256_load_si256(( const __m256i* )x[ib].qs);
				__m256i qy0 = _mm256_load_si256(( const __m256i* )y[ib].qs);
				__m256i qx1 = _mm256_load_si256(( const __m256i* )x[ib + 1].qs);
				__m256i qy1 = _mm256_load_si256(( const __m256i* )y[ib + 1].qs);

				const __m256 q0 = mul_sum_i8_pairs_float(qx0, qy0);
				const __m256 q1 = mul_sum_i8_pairs_float(qx1, qy1);

				acc0 = _mm256_fmadd_ps(d0, q0, acc0);
				acc1 = _mm256_fmadd_ps(d1, q1, acc1);
			}

			__m256 final_acc = _mm256_add_ps(acc0, acc1);
			*s				 = hsum_float_8(final_acc);
		}
	};

	template<uint64_t n_blocks>
		requires(n_blocks > 8 && n_blocks < 16)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			vec_dot_q8_0_q8_0<8>::impl(s, vx, vy);

			static constexpr uint64_t remainder = n_blocks - 8;
			if constexpr (remainder > 0) {
				float remainder_sum;
				vec_dot_q8_0_q8_0<remainder>::impl(&remainder_sum, vx + 8, vy + 8);
				*s += remainder_sum;
			}
		}
	};

	template<uint64_t n_blocks>
		requires(n_blocks == 16)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			const block_q8_0<half>* __restrict x = vx;
			const block_q8_0<half>* __restrict y = vy;
			__m256 acc0							 = _mm256_setzero_ps();
			__m256 acc1							 = _mm256_setzero_ps();
			__m256 acc2							 = _mm256_setzero_ps();
			__m256 acc3							 = _mm256_setzero_ps();

			for (uint64_t ib = 0; ib < 16; ib += 4) {
				_mm_prefetch(static_cast<const char*>(&x[ib + 8]), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(&y[ib + 8]), _MM_HINT_T0);

				const __m256 d0 = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
				const __m256 d1 = _mm256_set1_ps(fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d));
				const __m256 d2 = _mm256_set1_ps(fp16_to_fp32(x[ib + 2].d) * fp16_to_fp32(y[ib + 2].d));
				const __m256 d3 = _mm256_set1_ps(fp16_to_fp32(x[ib + 3].d) * fp16_to_fp32(y[ib + 3].d));

				__m256i qx0 = _mm256_load_si256(( const __m256i* )x[ib].qs);
				__m256i qy0 = _mm256_load_si256(( const __m256i* )y[ib].qs);
				__m256i qx1 = _mm256_load_si256(( const __m256i* )x[ib + 1].qs);
				__m256i qy1 = _mm256_load_si256(( const __m256i* )y[ib + 1].qs);
				__m256i qx2 = _mm256_load_si256(( const __m256i* )x[ib + 2].qs);
				__m256i qy2 = _mm256_load_si256(( const __m256i* )y[ib + 2].qs);
				__m256i qx3 = _mm256_load_si256(( const __m256i* )x[ib + 3].qs);
				__m256i qy3 = _mm256_load_si256(( const __m256i* )y[ib + 3].qs);

				const __m256 q0 = mul_sum_i8_pairs_float(qx0, qy0);
				const __m256 q1 = mul_sum_i8_pairs_float(qx1, qy1);
				const __m256 q2 = mul_sum_i8_pairs_float(qx2, qy2);
				const __m256 q3 = mul_sum_i8_pairs_float(qx3, qy3);

				acc0 = _mm256_fmadd_ps(d0, q0, acc0);
				acc1 = _mm256_fmadd_ps(d1, q1, acc1);
				acc2 = _mm256_fmadd_ps(d2, q2, acc2);
				acc3 = _mm256_fmadd_ps(d3, q3, acc3);
			}

			__m256 final_acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
			*s				 = hsum_float_8(final_acc);
		}
	};

	template<uint64_t remainder_blocks> struct process_remainder {
		NIHILUS_INLINE static void impl(float* sum, const block_q8_0<half>* vx, const block_q8_0<half>* vy) {
			if constexpr (remainder_blocks == 1) {
				vec_dot_q8_0_q8_0<device_types::cpu, 1>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 2) {
				vec_dot_q8_0_q8_0<2>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 3) {
				vec_dot_q8_0_q8_0<3>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 4) {
				vec_dot_q8_0_q8_0<4>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 5) {
				vec_dot_q8_0_q8_0<5>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 6) {
				vec_dot_q8_0_q8_0<6>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 7) {
				vec_dot_q8_0_q8_0<7>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 8) {
				vec_dot_q8_0_q8_0<8>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 9) {
				vec_dot_q8_0_q8_0<9>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 10) {
				vec_dot_q8_0_q8_0<10>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 11) {
				vec_dot_q8_0_q8_0<11>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 12) {
				vec_dot_q8_0_q8_0<12>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 13) {
				vec_dot_q8_0_q8_0<13>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 14) {
				vec_dot_q8_0_q8_0<14>::impl(sum, vx, vy);
			} else if constexpr (remainder_blocks == 15) {
				vec_dot_q8_0_q8_0<15>::impl(sum, vx, vy);
			}
		}
	};

	template<uint64_t n_blocks>
		requires(n_blocks > 16)
	struct vec_dot_q8_0_q8_0<n_blocks> {
		NIHILUS_INLINE static void impl(float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
			static constexpr uint64_t chunk_size	   = 16;
			static constexpr uint64_t total_chunks	   = (n_blocks + chunk_size - 1) / chunk_size;
			static constexpr uint64_t remainder_blocks = n_blocks % chunk_size;

			float total_sum = 0.0f;

			for (uint64_t chunk_id = 0; chunk_id < total_chunks - 1; ++chunk_id) {
				float chunk_sum;
				vec_dot_q8_0_q8_0<16>::impl(&chunk_sum, vx + chunk_id * 16, vy + chunk_id * 16);
				total_sum += chunk_sum;
			}

			if (total_chunks > 0) {
				uint64_t final_chunk_id = total_chunks - 1;
				float final_chunk_sum;

				if (remainder_blocks == 0) {
					vec_dot_q8_0_q8_0<16>::impl(&final_chunk_sum, vx + final_chunk_id * 16, vy + final_chunk_id * 16);
				} else {
					if constexpr (remainder_blocks > 0) {
						process_remainder<remainder_blocks>::impl(&final_chunk_sum, vx + final_chunk_id * 16, vy + final_chunk_id * 16);
					}
				}
				total_sum += final_chunk_sum;
			}

			*s = total_sum;
		}
	};

	NIHILUS_INLINE void vec_dot_q8_0_q8_0_impl(int64_t n, float* __restrict s, const block_q8_0<half>* __restrict vx, const block_q8_0<half>* __restrict vy) {
		static constexpr int64_t qk = Q_SIZE;
		const int64_t nb			= n / qk;

		const block_q8_0<half>* __restrict x = vx;
		const block_q8_0<half>* __restrict y = vy;

		int64_t ib	   = 0;
		float sumf = 0;
		__m256 acc = _mm256_setzero_ps();
		for (; ib < nb; ++ib) {
			const __m256 d = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
			__m256i qx	   = _mm256_load_si256(( const __m256i* )x[ib].qs);
			__m256i qy	   = _mm256_load_si256(( const __m256i* )y[ib].qs);

			const __m256 q = mul_sum_i8_pairs_float(qx, qy);
			acc = _mm256_fmadd_ps(d, q, acc);
		}

		sumf = hsum_float_8(acc);
		for (; ib < nb; ++ib) {
			int64_t sumi = 0;

			for (int64_t j = 0; j < qk; j++) {
				sumi += x[ib].qs[j] * y[ib].qs[j];
			}

			sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
		}

		*s = sumf;
	}
 
	template<uint64_t size_new> struct vec_dot_q8_f32 {};

	template<uint64_t size_new>
		requires(size_new > 0 && size_new < Q_SIZE)
	struct vec_dot_q8_f32<size_new> {
		NIHILUS_INLINE static float impl(const float* __restrict x, const block_q8_0<half>* __restrict y_blocks, uint64_t block_offset) {
			float sum = 0.0f;
			for (uint64_t i = 0; i < size_new; ++i) {
				const uint64_t block_idx = (block_offset + i) / Q_SIZE;
				const uint64_t elem_idx	 = (block_offset + i) % Q_SIZE;
				const float scale		 = static_cast<float>(y_blocks[block_idx].d);
				const int8_t quant		 = y_blocks[block_idx].qs[elem_idx];
				sum += x[i] * static_cast<float>(quant) * scale;
			}
			return sum;
		}
	};

	template<uint64_t size_new>
		requires(size_new == Q_SIZE)
	struct vec_dot_q8_f32<size_new> {
		NIHILUS_INLINE static float impl(const float* __restrict x, const block_q8_0<half>* __restrict y_blocks, uint64_t block_offset) {
			const uint64_t block_idx = block_offset / Q_SIZE;
			const float scale		 = fp16_to_fp32(y_blocks[block_idx].d);

			float sum = 0.0f;
			for (uint64_t i = 0; i < Q_SIZE; ++i) {
				const int8_t quant = y_blocks[block_idx].qs[i];
				sum += x[i] * static_cast<float>(quant);
			}

			return sum * scale;
		}
	};

	template<uint64_t size_new>
		requires(size_new > Q_SIZE && size_new < 64)
	struct vec_dot_q8_f32<size_new> {
		NIHILUS_INLINE static float impl(const float* __restrict x, const block_q8_0<half>* __restrict y_blocks, uint64_t block_offset) {
			float sum					 = vec_dot_q8_f32<Q_SIZE>::impl(x, y_blocks, block_offset);
			constexpr uint64_t remainder = size_new - Q_SIZE;
			if constexpr (remainder > 0) {
				sum += vec_dot_q8_f32<remainder>::impl(x + Q_SIZE, y_blocks, block_offset + Q_SIZE);
			}
			return sum;
		}
	};

	template<uint64_t size_new>
		requires(size_new >= 64)
	struct vec_dot_q8_f32<size_new> {
		NIHILUS_INLINE static float impl(const float* __restrict x, const block_q8_0<half>* __restrict y_blocks, uint64_t block_offset) {
			float sum					 = 0.0f;
			static constexpr uint64_t np = size_new & ~(Q_SIZE - 1);
			uint64_t i					 = 0;

			for (; i < np; i += Q_SIZE) {
				_mm_prefetch(static_cast<const char*>(x + i + Q_SIZE), _MM_HINT_T0);
				_mm_prefetch(static_cast<const char*>(&y_blocks[(block_offset + i + Q_SIZE) / Q_SIZE]), _MM_HINT_T0);
				sum += vec_dot_q8_f32<Q_SIZE>::impl(x + i, y_blocks, block_offset + i);
			}

			if (i < size_new) {
				constexpr uint64_t remainder = size_new % Q_SIZE;
				if constexpr (remainder > 0) {
					sum += vec_dot_q8_f32<remainder>::impl(x + i, y_blocks, block_offset + i);
				}
			}
			return sum;
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::prompt_eval_time, core_traits_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, float, block_q8_0<half>, float> {
		using input_type01						 = typename core_traits_type::input_01_type;
		using input_type02						 = typename core_traits_type::input_02_type;
		static constexpr uint64_t ne00			 = input_type01::get_array()[0];
		static constexpr uint64_t ne02			 = input_type01::get_array()[2];
		static constexpr uint64_t ne03			 = input_type01::get_array()[3];
		static constexpr uint64_t ne10			 = input_type02::get_array()[0];
		static constexpr uint64_t ne12			 = input_type02::get_array()[2];
		static constexpr uint64_t ne13			 = input_type02::get_array()[3];
		static constexpr uint64_t ne0			 = core_traits_type::get_array()[0];
		static constexpr uint64_t ne2			 = core_traits_type::get_array()[2];
		static constexpr uint64_t ne3			 = core_traits_type::get_array()[3];
		static constexpr int64_t r2				 = ne12 / ne02;
		static constexpr int64_t r3				 = ne13 / ne03;
		static constexpr uint64_t blocks_per_row = (ne10 + Q_SIZE - 1) / Q_SIZE;
		static constexpr int64_t nr0			 = ne0;
		static constexpr int64_t nr1			 = 1;

		static constexpr uint64_t chunk_size	 = 8;
		static constexpr int64_t nchunk0	 = (nr0 + chunk_size - 1) / chunk_size;
		static constexpr int64_t nchunk1	 = (nr1 + chunk_size - 1) / chunk_size;
		static constexpr int64_t chunk_count = nchunk0 * nchunk1;
		static constexpr int64_t dr0		 = (nr0 + nchunk0 - 1) / nchunk0;
		static constexpr int64_t dr1		 = (nr1 + nchunk1 - 1) / nchunk1;

		NIHILUS_INLINE static float scalar_dot_product_q8_f32(const float* __restrict input_vector,
			const block_q8_0<half>* __restrict weight_row,
			int64_t vector_size
		) {
			const int64_t num_blocks = (vector_size + Q_SIZE - 1) / Q_SIZE;
			float sum				 = 0.0f;

			for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
				const block_q8_0<half>& weight_block = weight_row[block_idx];
				const float weight_scale			 = fp16_to_fp32(weight_block.d);

				const int64_t elements_in_block = detail::min(Q_SIZE, vector_size - block_idx * Q_SIZE);

				for (int64_t elem_idx = 0; elem_idx < elements_in_block; ++elem_idx) {
					const int64_t global_idx = block_idx * Q_SIZE + elem_idx;

					if (global_idx < vector_size) {
						const float input_val	= input_vector[global_idx];
						const int8_t weight_val = weight_block.qs[elem_idx];

						sum += input_val * (static_cast<float>(weight_val) * weight_scale);
					}
				}
			}

			return sum;
		}

		template<bool is_power_of_2able> NIHILUS_INLINE static int64_t process_tensor_elements(int64_t chunk_id, const float* __restrict src0_data,
			const block_q8_0<half>* __restrict src1_data, float* __restrict dst_data, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, core_traits_type& output) {
			const uint64_t ne01		  = input01[1];
			const uint64_t ne11		  = input02[1];
			const uint64_t output_ne1 = output[1];
			const int64_t actual_nr1  = output_ne1 * ne2 * ne3;

			const int64_t ir0_start = dr0 * chunk_id;
			const int64_t ir0_end			= detail::min(ir0_start + dr0, nr0);
			const auto elenent_count01		= count_elements(input01);
			const auto elenent_count02		= count_elements(input02);
			const auto elenent_count_output = count_elements(output);
			if (ir0_start >= ir0_end || ir0_start < 0 || ir0_end > nr0) {
				return 0;
			}

			for (int64_t ir1 = 0; ir1 < actual_nr1; ++ir1) {
				const int64_t i13 = ir1 / (ne12 * output_ne1);
				const int64_t i12 = (ir1 - i13 * ne12 * output_ne1) / output_ne1;
				const int64_t i11 = ir1 - i13 * ne12 * output_ne1 - i12 * output_ne1;
				const int64_t i03 = i13 / r3;
				const int64_t i02 = i12 / r2;

				const uint64_t src0_offset = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i11 * ne00;
				const uint64_t dst_offset  = ir1 * nr0;

				const int64_t max_input_elements = ne10;
				if (src0_offset + max_input_elements > elenent_count02) {
					continue;
				}

				const float* __restrict src0_ptr = &src0_data[src0_offset];
				float* __restrict dst_ptr		 = &dst_data[dst_offset];

				for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
					if (ir0 >= ne01) {
						dst_ptr[ir0] = 0.0f;
						continue;
					}

					if (dst_offset + ir0 >= elenent_count_output) {
						continue;
					}

					const int64_t weight_row_offset = ir0 * blocks_per_row;
					if (weight_row_offset + blocks_per_row > elenent_count01) {
						dst_ptr[ir0] = 0.0f;
						continue;
					}
					dst_ptr[ir0] = scalar_dot_product_q8_f32(src0_ptr,
						&src1_data[weight_row_offset],
						ne10
					);
				}
			}
			return 1;
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, typename core_traits_type::input_01_type& input01,
			typename core_traits_type::input_02_type& input02) {
			int64_t current_chunk = thread_index;

			while (current_chunk < chunk_count ) {
				current_chunk = output.current_chunk_prompt_eval.fetch_sub(chunks_completed);
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::eval_time, core_traits_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, float, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			if (thread_index != 0)
				return;

			static constexpr uint64_t ne00 = core_traits_type::input_01_type::get_array()[0];
			static constexpr uint64_t ne10 = core_traits_type::input_02_type::get_array()[0];
			static constexpr uint64_t ne0  = core_traits_type::get_array()[0];

			const float* __restrict src0_data = input02.data;
			const block_q8_0<half>* __restrict src1_data;
			if constexpr (array_types<decltype(input01.data)>) {
				src1_data = input01.data[current_block];
			} else {
				src1_data = input01.data;
			}

			float* __restrict dst_data = output.data;

			for (uint64_t i0 = 0; i0 < ne0; ++i0) {
				dst_data[i0] = vec_dot_q8_f32<ne10>::impl(src0_data, src1_data, i0 * (ne10 / Q_SIZE) * Q_SIZE);
			}
		}
	};	

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::prompt_eval_time, core_traits_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, float, half, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t ne00	  = input_type01::get_array()[0];
			static constexpr uint64_t ne01	  = input_type01::get_array()[1];
			static constexpr uint64_t ne02	  = input_type01::get_array()[2];
			static constexpr uint64_t ne03	  = input_type01::get_array()[3];
			static constexpr uint64_t ne12	  = input_type02::get_array()[2];
			static constexpr uint64_t ne13	  = input_type02::get_array()[3];
			const uint64_t ne10				  = input02[0];
			const uint64_t ne11				  = input02[1];
			const half* src0_data			  = input01.data;
			const float* __restrict src1_data = input02.data;
			float* __restrict dst_data		  = output.data;

			static constexpr uint64_t r2 = ne12 / ne02;
			static constexpr uint64_t r3 = ne13 / ne03;

			for (uint64_t i13 = 0; i13 < ne13; ++i13) {
				for (uint64_t i12 = 0; i12 < ne12; ++i12) {
					for (uint64_t i11 = 0; i11 < ne11; ++i11) {
						const uint64_t i03 = i13 / r3;
						const uint64_t i02 = i12 / r2;

						for (uint64_t i01 = 0; i01 < ne01; ++i01) {
							float sum = 0.0f;

							for (uint64_t i00 = 0; i00 < ne00; ++i00) {
								const uint64_t src0_idx = i00 + i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;
								const uint64_t src1_idx = i00 + i11 * ne10 + i12 * ne10 * ne11 + i13 * ne10 * ne11 * ne12;

								sum += fp16_to_fp32(src0_data[src0_idx]) * src1_data[src1_idx];
							}

							const uint64_t dst_idx = i01 + i11 * ne01 + i12 * ne01 * ne11 + i13 * ne01 * ne11 * ne12;
							dst_data[dst_idx]	   = sum;
						}
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::softmax, processing_phases::prompt_eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_traits_type, float, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data  = input01.data;
			const float* __restrict mask_data = input02.data;
			float* __restrict dst_data		  = output.data;

			for (uint64_t i03 = 0; i03 < ne03; ++i03) {
				for (uint64_t i02 = 0; i02 < ne02; ++i02) {
					for (uint64_t i01 = 0; i01 < ne01; ++i01) {
						const uint64_t row_offset = i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;

						float max_val = -INFINITY;
						for (uint64_t i00 = 0; i00 < ne00; ++i00) {
							const uint64_t src_idx	= row_offset + i00;
							const uint64_t mask_idx = i00 + i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;

							float val = src_data[src_idx];
							if (mask_data != nullptr) {
								val += mask_data[mask_idx];
							}

							if (val > max_val) {
								max_val = val;
							}
						}

						float sum = 0.0f;
						for (uint64_t i00 = 0; i00 < ne00; ++i00) {
							const uint64_t src_idx	= row_offset + i00;
							const uint64_t mask_idx = i00 + i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;

							float val = src_data[src_idx];
							if (mask_data != nullptr) {
								val += mask_data[mask_idx];
							}

							float exp_val	  = expf(val - max_val);
							dst_data[src_idx] = exp_val;
							sum += exp_val;
						}

						const float inv_sum = 1.0f / sum;
						for (uint64_t i00 = 0; i00 < ne00; ++i00) {
							const uint64_t dst_idx = row_offset + i00;
							dst_data[dst_idx] *= inv_sum;
						}
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::copy, processing_phases::prompt_eval_time, core_traits_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_traits_type, half, half, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const half* src_data = input01.data;
			half* dst_data		 = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = ne00 * ne01 * ne02_runtime * ne03;

			for (uint64_t i = 0; i < total_elements; ++i) {
				dst_data[i] = src_data[i];
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rope, processing_phases::prompt_eval_time, core_traits_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::rope, core_traits_type, float, float, int32_t, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;
		using input_type03 = typename core_traits_type::input_03_type;

		static constexpr float constexpr_pow(float base, float exp) {
			if (exp == 0.0f)
				return 1.0f;
			if (exp == 1.0f)
				return base;

			float result		= 1.0f;
			int64_t int_exp		= static_cast<int64_t>(exp);
			float current_power = base;

			while (int_exp > 0) {
				if (int_exp & 1)
					result *= current_power;
				current_power *= current_power;
				int_exp >>= 1;
			}

			float frac = exp - static_cast<int64_t>(exp);
			if (frac > 0.0f) {
				result *= (1.0f + frac * (base - 1.0f) / base);
			}

			return result;
		}

		template<uint64_t N> static constexpr auto make_freq_table() {
			array<float, N> freqs{};
			constexpr float rope_freq_base = core_traits_type::model_traits_type::rope_freq_base;
			constexpr uint32_t rope_dim	   = core_traits_type::model_traits_type::rope_dimension_count;

			for (uint64_t i = 0; i < N; ++i) {
				const float freq_exponent = (2.0f * static_cast<float>(i)) / static_cast<float>(rope_dim);
				const float theta_power	  = constexpr_pow(rope_freq_base, freq_exponent);
				freqs[i]				  = 1.0f / theta_power;
			}
			return freqs;
		}

		static constexpr float rope_freq_base		   = core_traits_type::model_traits_type::rope_freq_base;
		static constexpr uint32_t rope_dimension_count = core_traits_type::model_traits_type::rope_dimension_count;
		static constexpr uint64_t rope_dimension_count			   = core_traits_type::model_traits_type::rope_dimension_count;
		static constexpr uint32_t attention_head_count = core_traits_type::model_traits_type::attention_head_count;

		static constexpr uint64_t batch_size	  = input_type01::get_array()[0];
		static constexpr uint64_t num_heads		  = input_type01::get_array()[2];
		static constexpr uint64_t tensor_rope_dimension_count = input_type01::get_array()[3];

		static constexpr uint64_t rope_dim		= rope_dimension_count;
		static constexpr uint64_t half_rope_dim = rope_dim / 2;

		static constexpr auto freq_table = make_freq_table<half_rope_dim>();

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const float* __restrict src_data		  = input01.data;
			const int32_t* pos_data					  = input02.data;
			const float* __restrict freq_scaling_data = input03.data;
			float* __restrict dst_data				  = output.data;

			const uint64_t seq_len = input01[1];

			const uint64_t total_work_items = batch_size * seq_len * num_heads;
			const uint64_t total_elements	= batch_size * seq_len * num_heads * rope_dimension_count;

			const uint64_t work_per_thread = (total_work_items + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = (work_start + work_per_thread < total_work_items) ? work_start + work_per_thread : total_work_items;

			const int64_t nr = total_work_items;
			int64_t ir		 = 0;

			for (int64_t i3 = 0; i3 < batch_size; i3++) {
				for (int64_t i2 = 0; i2 < seq_len; i2++) {
					const int64_t position = pos_data[i2];

					for (int64_t i1 = 0; i1 < num_heads; i1++) {
						if (ir++ < work_start)
							continue;
						if (ir > work_end)
							break;

						const uint64_t src_offset = i3 * seq_len * num_heads * rope_dimension_count + i2 * num_heads * rope_dimension_count + i1 * rope_dimension_count;
						const uint64_t dst_offset = i3 * seq_len * num_heads * rope_dimension_count + i2 * num_heads * rope_dimension_count + i1 * rope_dimension_count;

						for (int64_t i0 = 0; i0 < rope_dim; i0 += 2) {
							const uint64_t dim_pair = i0 / 2;
							float freq				= freq_table[dim_pair];

							if (freq_scaling_data != nullptr) {
								const uint64_t scaling_idx = (dim_pair < input_type02::get_array()[0]) ? dim_pair : 0;
								freq *= freq_scaling_data[scaling_idx];
							}

							const float angle	  = static_cast<float>(position) * freq;
							const float cos_theta = cosf(angle);
							const float sin_theta = sinf(angle);

							const float x0 = src_data[src_offset + i0];
							const float x1 = src_data[src_offset + i0 + 1];

							dst_data[dst_offset + i0]	  = x0 * cos_theta - x1 * sin_theta;
							dst_data[dst_offset + i0 + 1] = x0 * sin_theta + x1 * cos_theta;
						}

						for (int64_t i0 = rope_dim; i0 < rope_dimension_count; i0++) {
							dst_data[dst_offset + i0] = src_data[src_offset + i0];
						}
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::copy, processing_phases::prompt_eval_time, core_traits_type, float, float>
		: public kernel_base<kernel_types::copy, core_traits_type, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			float* __restrict dst_data		 = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = count_elements(output);

			for (uint64_t i = 0; i < total_elements; ++i) {
				dst_data[i] = src_data[i];
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::cont, processing_phases::prompt_eval_time, core_traits_type, float, float>
		: public kernel_base<kernel_types::cont, core_traits_type, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			float* __restrict dst_data		 = output.data;

			const uint64_t ne02			  = input01[2];
			const uint64_t total_elements = ne00 * ne01 * ne02 * ne03;

			const uint64_t work_per_thread = (total_elements + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = detail::min(work_start + work_per_thread, total_elements);

			const uint64_t src_stride0 = input01.strides[0];
			const uint64_t src_stride1 = input01.strides[1];
			const uint64_t src_stride2 = input01.strides[2];
			const uint64_t src_stride3 = input01.strides[3];

			for (uint64_t linear_idx = work_start; linear_idx < work_end; ++linear_idx) {
				const uint64_t i3		  = linear_idx / (ne00 * ne01 * ne02);
				const uint64_t remaining3 = linear_idx % (ne00 * ne01 * ne02);
				const uint64_t i2		  = remaining3 / (ne00 * ne01);
				const uint64_t remaining2 = remaining3 % (ne00 * ne01);
				const uint64_t i1		  = remaining2 / ne00;
				const uint64_t i0		  = remaining2 % ne00;

				const uint64_t src_idx = i3 * src_stride3 + i2 * src_stride2 + i1 * src_stride1 + i0 * src_stride0;
				dst_data[linear_idx]   = src_data[src_idx];
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::silu, processing_phases::prompt_eval_time, core_traits_type, float, float>
		: public kernel_base<kernel_types::silu, core_traits_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, int64_t, core_traits_type&, const typename core_traits_type::input_01_type&) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			if (thread_index != 0)
				return;

			static constexpr uint64_t ne00 = core_traits_type::input_01_type::get_array()[0];
			static constexpr uint64_t ne10 = core_traits_type::input_02_type::get_array()[0];
			static constexpr float eps	   = core_traits_type::model_traits_type::layer_norm_rms_epsilon;

			const float sum	  = simd_sum_squares_add(input01.data, input02.data, ne00);
			const float mean  = sum / static_cast<float>(ne00);
			const float scale = 1.0f / sqrtf_fast(mean + eps);

			if constexpr (ne10 == ne00) {
				vec_add_rms_norm_f32<ne00>::impl(output.data, input01.data, input02.data, scale);
			} else {
				static constexpr bool is_power_of_2 = (ne10 & (ne10 - 1)) == 0;
				if constexpr (is_power_of_2) {
					static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10 = i0 & ((1ULL << log2_ne10) - 1ULL);
						const float added  = input01.data[i0] + input02.data[i10];
						output.data[i0]	   = added * scale;
					}
				} else {
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10 = i0 % ne10;
						const float added  = input01.data[i0] + input02.data[i10];
						output.data[i0]	   = added * scale;
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::none, processing_phases::eval_time, core_traits_type, float, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_traits_type, float, float, float, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			if (thread_index != 0)
				return;

			static constexpr uint64_t ne00 = core_traits_type::input_01_type::get_array()[0];
			static constexpr uint64_t ne10 = core_traits_type::input_02_type::get_array()[0];
			static constexpr uint64_t ne20 = core_traits_type::input_03_type::get_array()[0];
			static constexpr float eps	   = core_traits_type::model_traits_type::layer_norm_rms_epsilon;

			const float sum	  = simd_sum_squares_add_for_q8(input01.data, input02.data, ne00);
			const float mean  = sum / static_cast<float>(ne00);
			const float scale = 1.0f / sqrtf_fast(mean + eps);

			const block_q8_0<half>* src2_data	 = input03.data[current_block];
			static constexpr uint64_t num_blocks = ne00 / Q_SIZE;
			float w_scales[num_blocks];
			for (uint64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
				w_scales[block_idx] = static_cast<float>(src2_data[block_idx].d);
			}
			const int8_t* w_quants = static_cast<const int8_t*>(&src2_data[0].qs[0]);

			if constexpr (ne10 == ne00 && ne20 == ne00) {
				vec_add_rms_norm_mul_q8_f32<ne00>::impl(output.data, input01.data, input02.data, scale, w_scales, w_quants, 0);
			} else {
				static constexpr bool is_power_of_2 = ((ne10 & (ne10 - 1)) == 0) && ((ne20 & (ne20 - 1)) == 0);
				if constexpr (is_power_of_2) {
					static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
					static constexpr uint64_t log2_ne20 = tzcnt_constexpr(ne20);
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10			   = i0 & ((1ULL << log2_ne10) - 1ULL);
						const uint64_t i20			   = i0 & ((1ULL << log2_ne20) - 1ULL);
						const float added			   = input01.data[i0] + input02.data[i10];
						const float normalized		   = added * scale;
						const float block_scale		   = w_scales[i20 / Q_SIZE];
						const float dequantized_weight = static_cast<float>(w_quants[i20]) * block_scale;
						output.data[i0]				   = normalized * dequantized_weight;
					}
				} else {
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10			   = i0 % ne10;
						const uint64_t i20			   = i0 % ne20;
						const float added			   = input01.data[i0] + input02.data[i10];
						const float normalized		   = added * scale;
						const float block_scale		   = w_scales[i20 / Q_SIZE];
						const float dequantized_weight = static_cast<float>(w_quants[i20]) * block_scale;
						output.data[i0]				   = normalized * dequantized_weight;
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rms_norm_mul_transpose, processing_phases::eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::rms_norm_mul_transpose, core_traits_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			if (thread_index != 0)
				return;
			static constexpr uint64_t ne00 = core_traits_type::input_01_type::get_array()[0];
			static constexpr uint64_t ne10 = core_traits_type::input_02_type::get_array()[0];
			static constexpr float eps	   = core_traits_type::model_traits_type::layer_norm_rms_epsilon;
			const float sum				   = simd_sum_squares(input01.data, ne00);
			const float mean			   = sum / static_cast<float>(ne00);
			const float scale			   = 1.0f / sqrtf_fast(mean + eps);
			const float* src1_data		   = input02.data[current_block];
			if constexpr (ne10 == ne00) {
				vec_scale_mul_f32<ne00>::impl(output.data, input01.data, scale, src1_data);
			} else {
				static constexpr bool is_power_of_2 = (ne10 & (ne10 - 1)) == 0;
				if constexpr (is_power_of_2) {
					static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10 = i0 & ((1ULL << log2_ne10) - 1ULL);
						output.data[i0]	   = input01.data[i0] * scale * src1_data[i10];
					}
				} else {
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10 = i0 % ne10;
						output.data[i0]	   = input01.data[i0] * scale * src1_data[i10];
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul, processing_phases::eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_traits_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			if (thread_index != 0)
				return;
			static constexpr uint64_t ne00 = core_traits_type::input_01_type::get_array()[0];
			static constexpr uint64_t ne10 = core_traits_type::input_02_type::get_array()[0];
			if constexpr (ne10 == ne00) {
				vec_mul_f32<ne00>::impl(output.data, input01.data, input02.data);
			} else {
				static constexpr bool is_power_of_2 = (ne10 & (ne10 - 1)) == 0;
				if constexpr (is_power_of_2) {
					static constexpr uint64_t log2_ne10 = tzcnt_constexpr(ne10);
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10 = i0 & ((1ULL << log2_ne10) - 1ULL);
						output.data[i0]	   = input01.data[i0] * input02.data[i10];
					}
				} else {
					for (uint64_t i0 = 0; i0 < ne00; ++i0) {
						const uint64_t i10 = i0 % ne10;
						output.data[i0]	   = input01.data[i0] * input02.data[i10];
					}
				}
			}
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::eval_time, core_traits_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			if (thread_index != 0)
				return;
			static constexpr uint64_t ne00			 = core_traits_type::input_01_type::get_array()[0];
			static constexpr uint64_t blocks_per_row = ne00 / Q_SIZE;
			const uint64_t token_id					 = static_cast<uint64_t>(input02.data[0]);
			dequantize_row_q8_0(&input01.data[current_block], output.data, ne00);
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::get_rows, processing_phases::eval_time, core_traits_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_traits_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
			if (thread_index != 0)
				return;
			const uint64_t ne00		= input01[0];
			const uint64_t token_id = static_cast<uint64_t>(input02.data[0]);
			vec_cpy_f32(ne00, output.data, &input01.data[token_id * ne00]);
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::mul_mat, processing_phases::eval_time, core_traits_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_traits_type, float, half, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::softmax, processing_phases::eval_time, core_traits_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_traits_type, float, float, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::copy, processing_phases::eval_time, core_traits_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_traits_type, half, half, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02) {
		}
	};

	template<const model_config& config, typename core_traits_type>
	struct kernel_dispatcher_impl<config, core_traits_type, device_types::cpu, 1, kernel_types::rope, processing_phases::eval_time, core_traits_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::rope, core_traits_type, float, float, int32_t, float> {
		using input_type01 = typename core_traits_type::input_01_type;
		using input_type02 = typename core_traits_type::input_02_type;
		using input_type03 = typename core_traits_type::input_03_type;

		static constexpr float constexpr_pow(float base, float exp) {
			if (exp == 0.0f)
				return 1.0f;
			if (exp == 1.0f)
				return base;

			float result		= 1.0f;
			int64_t int_exp		= static_cast<int64_t>(exp);
			float current_power = base;

			while (int_exp > 0) {
				if (int_exp & 1)
					result *= current_power;
				current_power *= current_power;
				int_exp >>= 1;
			}

			float frac = exp - static_cast<int64_t>(exp);
			if (frac > 0.0f) {
				result *= (1.0f + frac * (base - 1.0f) / base);
			}

			return result;
		}

		template<uint64_t N> static constexpr auto make_freq_table() {
			array<float, N> freqs{};
			constexpr float rope_freq_base = core_traits_type::model_traits_type::rope_freq_base;
			constexpr uint32_t rope_dim	   = core_traits_type::model_traits_type::rope_dimension_count;

			for (uint64_t i = 0; i < N; ++i) {
				const float freq_exponent = (2.0f * static_cast<float>(i)) / static_cast<float>(rope_dim);
				const float theta_power	  = constexpr_pow(rope_freq_base, freq_exponent);
				freqs[i]				  = 1.0f / theta_power;
			}
			return freqs;
		}

		static constexpr float rope_freq_base		   = core_traits_type::model_traits_type::rope_freq_base;
		static constexpr uint32_t rope_dimension_count = core_traits_type::model_traits_type::rope_dimension_count;
		static constexpr uint64_t rope_dimension_count			   = core_traits_type::model_traits_type::rope_dimension_count;
		static constexpr uint32_t attention_head_count = core_traits_type::model_traits_type::attention_head_count;

		static constexpr uint64_t batch_size	  = input_type01::get_array()[0];
		static constexpr uint64_t num_heads		  = input_type01::get_array()[2];
		static constexpr uint64_t tensor_rope_dimension_count = input_type01::get_array()[3];

		static constexpr uint64_t rope_dim		= rope_dimension_count;
		static constexpr uint64_t half_rope_dim = rope_dim / 2;

		static constexpr auto freq_table = make_freq_table<half_rope_dim>();

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_traits_type& output, const typename core_traits_type::input_01_type& input01,
			const typename core_traits_type::input_02_type& input02, const typename core_traits_type::input_03_type& input03) {
			const float* __restrict src_data		  = input01.data;
			const int32_t* pos_data					  = input02.data;
			const float* __restrict freq_scaling_data = input03.data;
			float* __restrict dst_data				  = output.data;

			const uint64_t seq_len = input01[1];

			const uint64_t total_work_items = batch_size * seq_len * num_heads;
			const uint64_t total_elements	= batch_size * seq_len * num_heads * rope_dimension_count;

			const uint64_t work_per_thread = (total_work_items + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = (work_start + work_per_thread < total_work_items) ? work_start + work_per_thread : total_work_items;

			const int64_t nr = total_work_items;
			int64_t ir		 = 0;

			for (int64_t i3 = 0; i3 < batch_size; i3++) {
				for (int64_t i2 = 0; i2 < seq_len; i2++) {
					const int64_t position = pos_data[i2];

					for (int64_t i1 = 0; i1 < num_heads; i1++) {
						if (ir++ < work_start)
							continue;
							break;

						const uint64_t src_offset = i3 * seq_len * num_heads * rope_dimension_count + i2 * num_heads * rope_dimension_count + i1 * rope_dimension_count;
						const uint64_t dst_offset = i3 * seq_len * num_heads * rope_dimension_count + i2 * num_heads * rope_dimension_count + i1 * rope_dimension_count;

						for (int64_t i0 = 0; i0 < rope_dim; i0 += 2) {
							const uint64_t dim_pair = i0 / 2;
							float freq				= freq_table[dim_pair];

							if (freq_scaling_data != nullptr) {
								const uint64_t scaling_idx = (dim_pair < input_type02::get_array()[0]) ? dim_pair : 0;
								freq *= freq_scaling_data[scaling_idx];
							}

							const float angle	  = static_cast<float>(position) * freq;
							const float cos_theta = cosf(angle);
							const float sin_theta = sinf(angle);

							const float x0 = src_data[src_offset + i0];
							const float x1 = src_data[src_offset + i0 + 1];

							dst_data[dst_offset + i0]	  = x0 * cos_theta - x1 * sin_theta;
							dst_data[dst_offset + i0 + 1] = x0 * sin_theta + x1 * cos_theta;
						}

						for (int64_t i0 = rope_dim; i0 < rope_dimension_count; i0++) {
							dst_data[dst_offset + i0] = src_data[src_offset + i0];
						}
					}
				}
			}
		}
	};
	*/

};

#endif
