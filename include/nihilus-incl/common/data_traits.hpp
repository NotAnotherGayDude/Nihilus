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

#include <nihilus-incl/common/dim_traits.hpp>
#include <nihilus-incl/common/type_traits.hpp>
#include <nihilus-incl/common/data.hpp>

namespace nihilus {

	template<tensor_types tensor_type, typename config_type_new> struct data_traits;

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::attn_q, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::attn_q>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::attn_q };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::attn_k, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::attn_k>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::attn_k };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::attn_v, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::attn_v>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::attn_v };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::attn_output, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::attn_output>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::attn_output };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::attn_norm, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::attn_norm>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::attn_norm };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::ffn_gate, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::ffn_gate>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::ffn_gate };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::ffn_up, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::ffn_up>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::ffn_up };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::ffn_down, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::ffn_down>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::ffn_down };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::ffn_norm, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::ffn_norm>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{
			config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count
		};
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::ffn_norm };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::token_embd, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::token_embd>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::token_embd };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::rope_freqs, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::rope_freqs>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::rope_freqs };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::output_norm, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::output_norm>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::output_norm };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::output, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::output>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ config_type_new::device_type == device_types::cpu ? 0 : type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::output };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::inp_tokens, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::inp_tokens>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::inp_tokens };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::inp_pos, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::inp_pos>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::inp_pos };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::cache_k, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::cache_k>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::cache_k };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::cache_v, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::cache_v>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) * model_dimensions<config_type_new>::block_count };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
		static constexpr tensor_types tensor_type{ tensor_types::cache_v };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::kq_mask, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::kq_mask>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::kq_mask };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::inp_out_ids, config_type_new> {
		using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
		using dims_type	  = dim_traits<config_type_new, tensor_types::inp_out_ids>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::inp_out_ids };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::temperature, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::temperature>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::temperature };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::top_k, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::top_k>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::top_k };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::top_p, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::top_p>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::top_p };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::repetition_penalty, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::repetition_penalty>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::repetition_penalty };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::presence_penalty, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::presence_penalty>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::presence_penalty };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::frequency_penalty, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::frequency_penalty>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::frequency_penalty };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::rep_window, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::rep_window>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::rep_window };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::token_history, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::token_history>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::token_history };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::rng_state, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::rng_state>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::rng_state };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::logits_bias, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::logits_bias>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::logits_bias };
	};

	template<llama_arch_config_types config_type_new> struct data_traits<tensor_types::allowed_vocab_mask, config_type_new> {
		using output_type		   = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
		using dims_type			   = dim_traits<config_type_new, tensor_types::allowed_vocab_mask>;
		static constexpr auto dims = dims_type::dims;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
		static constexpr tensor_types tensor_type{ tensor_types::allowed_vocab_mask };
	};

	template<integral_or_enum_types auto enum_value_new, typename config_type_new> struct indexed_data_holder;

	template<tensor_types enum_value_new, typename config_type_new> struct indexed_data_holder<enum_value_new, config_type_new>
		: public data_holder<typename data_traits<enum_value_new, config_type_new>::output_type, data_traits<enum_value_new, config_type_new>::data_strategy_type,
			  data_traits<enum_value_new, config_type_new>::total_required_bytes, model_dimensions<config_type_new>::block_count>,
		  public data_traits<enum_value_new, config_type_new>,
		  public core_elem_base<enum_value_new, struct indexed_data_holder<enum_value_new, config_type_new>> {
		static constexpr auto enum_value{ enum_value_new };
	};

	template<typename config_type_new, typename enum_type> struct data_holder_aggregator;

	template<llama_arch_config_types config_type_new> struct data_holder_aggregator<config_type_new, tensor_types> {
		static constexpr array values{ tensor_types::attn_q, tensor_types::attn_k, tensor_types::attn_v, tensor_types::attn_output, tensor_types::attn_norm, tensor_types::ffn_gate,
			tensor_types::ffn_up, tensor_types::ffn_down, tensor_types::ffn_norm, tensor_types::token_embd, tensor_types::rope_freqs, tensor_types::output_norm,
			tensor_types::output, tensor_types::inp_tokens, tensor_types::inp_pos, tensor_types::cache_k, tensor_types::cache_v, tensor_types::kq_mask, tensor_types::inp_out_ids,
			tensor_types::temperature, tensor_types::top_k, tensor_types::top_p, tensor_types::repetition_penalty, tensor_types::presence_penalty, tensor_types::frequency_penalty,
			tensor_types::rep_window, tensor_types::token_history, tensor_types::rng_state, tensor_types::logits_bias, tensor_types::allowed_vocab_mask };
	};

}
