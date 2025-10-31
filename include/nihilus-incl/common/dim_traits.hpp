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

#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/db/file_io.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/infra/model_traits.hpp>
#include <nihilus-incl/common/type_traits.hpp>
#include <nihilus-incl/common/dimensions.hpp>

namespace nihilus {

	template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

	template<llama_arch_config_types config_type_new, enum_types auto enum_value_new, typename enum_type = decltype(enum_value_new)> struct dim_traits;

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::attn_q>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::embedding_length, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::attn_q };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::attn_k>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::n_embd_kv_gqa, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::attn_k };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::attn_v>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::n_embd_kv_gqa, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::attn_v };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::attn_output>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::embedding_length, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::attn_output };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::attn_norm>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::attn_norm };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::ffn_gate>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::feed_forward_length, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::ffn_gate };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::ffn_up>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::feed_forward_length, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::ffn_up };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::ffn_down>
		: public dimensions<model_dimensions<config_type_new>::feed_forward_length, model_dimensions<config_type_new>::embedding_length, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::ffn_down };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::ffn_norm>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::ffn_norm };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::token_embd>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::vocab_size, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::token_embd };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::rope_freqs>
		: public dimensions<model_dimensions<config_type_new>::rope_dimension_count / 2, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::rope_freqs };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::output_norm>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::output_norm };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::output>
		: public dimensions<model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::vocab_size, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::output };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::inp_tokens>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  config_type_new::batched_processing ? config_type_new::batch_size : config_type_new::max_sequence_length,
			  config_type_new::batched_processing ? config_type_new::max_sequence_length : 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::inp_tokens };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::inp_pos>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  config_type_new::batched_processing ? config_type_new::batch_size : config_type_new::max_sequence_length,
			  config_type_new::batched_processing ? config_type_new::max_sequence_length : 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::inp_pos };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::cache_k>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  config_type_new::batched_processing ? config_type_new::batch_size : config_type_new ::max_sequence_length,
			  config_type_new::batched_processing ? config_type_new::max_sequence_length : model_dimensions<config_type_new>::n_embd_kv_gqa,
			  config_type_new::batched_processing ? model_dimensions<config_type_new>::n_embd_kv_gqa : 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::cache_k };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::cache_v>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  config_type_new::batched_processing ? config_type_new::batch_size : config_type_new ::max_sequence_length,
			  config_type_new::batched_processing ? config_type_new::max_sequence_length : model_dimensions<config_type_new>::n_embd_kv_gqa,
			  config_type_new::batched_processing ? model_dimensions<config_type_new>::n_embd_kv_gqa : 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::cache_v };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::kq_mask> : public dimensions<32ull, 32ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::kq_mask };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::inp_out_ids>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::inp_out_ids };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::temperature>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::temperature };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::top_k>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::top_k };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::top_p>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::top_p };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::repetition_penalty>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::repetition_penalty };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::presence_penalty>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::presence_penalty };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::frequency_penalty>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::frequency_penalty };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::rep_window>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing>(), config_type_new::batched_processing ? config_type_new::batch_size : 1ull, 1ull, 1ull,
			  1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::rep_window };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::token_history>
		: public rt_dimensions<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  config_type_new::batched_processing ? config_type_new::batch_size : config_type_new::max_sequence_length,
			  config_type_new::batched_processing ? config_type_new::max_sequence_length : 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::token_history };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::rng_state> : public dimensions<256ull, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::rng_state };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::logits_bias>
		: public dimensions<model_dimensions<config_type_new>::vocab_size, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::logits_bias };
	};

	template<llama_arch_config_types config_type_new> struct dim_traits<config_type_new, tensor_types::allowed_vocab_mask>
		: public dimensions<model_dimensions<config_type_new>::vocab_size, 1ull, 1ull, 1ull> {
		static constexpr auto sub_kernel_type{ tensor_types::allowed_vocab_mask };
	};

}
