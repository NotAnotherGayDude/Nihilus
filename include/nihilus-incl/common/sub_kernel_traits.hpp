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

#include <nihilus-incl/common/op_traits.hpp>

namespace nihilus {

	template<typename config_type_new, enum_types auto enum_value_new, typename enum_type = decltype(enum_value_new)> struct sub_kernel_traits_new;
	/*

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::attn_q>
		: public core_elem_base<weight_types::attn_q, sub_kernel_traits_new<config_type_new, weight_types::attn_q>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_q>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_q>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::attn_q };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::attn_k>
		: public core_elem_base<weight_types::attn_k, sub_kernel_traits_new<config_type_new, weight_types::attn_k>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_k>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_k>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::attn_k };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::attn_v>
		: public core_elem_base<weight_types::attn_v, sub_kernel_traits_new<config_type_new, weight_types::attn_v>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_v>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_v>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::attn_v };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::attn_output>
		: public core_elem_base<weight_types::attn_output, sub_kernel_traits_new<config_type_new, weight_types::attn_output>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_output>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_output>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::attn_output };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::attn_norm>
		: public core_elem_base<weight_types::attn_norm, sub_kernel_traits_new<config_type_new, weight_types::attn_norm>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_norm>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::norm_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::norm_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::attn_norm>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::attn_norm };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::ffn_gate>
		: public core_elem_base<weight_types::ffn_gate, sub_kernel_traits_new<config_type_new, weight_types::ffn_gate>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_gate>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_gate>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::ffn_gate };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::ffn_up>
		: public core_elem_base<weight_types::ffn_up, sub_kernel_traits_new<config_type_new, weight_types::ffn_up>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_up>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_up>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::ffn_up };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::ffn_down>
		: public core_elem_base<weight_types::ffn_down, sub_kernel_traits_new<config_type_new, weight_types::ffn_down>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_down>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_down>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::ffn_down };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::ffn_norm>
		: public core_elem_base<weight_types::ffn_norm, sub_kernel_traits_new<config_type_new, weight_types::ffn_norm>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_norm>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::per_block, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::norm_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::norm_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::ffn_norm>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ weight_types::ffn_norm };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::token_embd>
		: public core_elem_base<weight_types::token_embd, sub_kernel_traits_new<config_type_new, weight_types::token_embd>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::token_embd>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::token_embd>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ weight_types::token_embd };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::rope_freqs>
		: public core_elem_base<weight_types::rope_freqs, sub_kernel_traits_new<config_type_new, weight_types::rope_freqs>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::rope_freqs>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::rope_freqs>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ weight_types::rope_freqs };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::output_norm>
		: public core_elem_base<weight_types::output_norm, sub_kernel_traits_new<config_type_new, weight_types::output_norm>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::output_norm>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::norm_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::norm_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::output_norm>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ weight_types::output_norm };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, weight_types::output>
		: public core_elem_base<weight_types::output, sub_kernel_traits_new<config_type_new, weight_types::output>>,
		  public kernel_dims_new<0, dim_traits<config_type_new, weight_types::output>::dims>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
		using dims_type	  = kernel_dims_new<0, dim_traits<config_type_new, weight_types::output>::dims>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ weight_types::output };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::inp_tokens>
		: public core_elem_base<global_input_types::inp_tokens, sub_kernel_traits_new<config_type_new, global_input_types::inp_tokens>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::inp_tokens>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::inp_tokens>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::inp_tokens };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::inp_pos>
		: public core_elem_base<global_input_types::inp_pos, sub_kernel_traits_new<config_type_new, global_input_types::inp_pos>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::inp_pos>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::inp_pos>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::inp_pos };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::cache_k>
		: public core_elem_base<global_input_types::cache_k, sub_kernel_traits_new<config_type_new, global_input_types::cache_k>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::cache_k>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::cache_k>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ global_input_types::cache_k };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::cache_v>
		: public core_elem_base<global_input_types::cache_v, sub_kernel_traits_new<config_type_new, global_input_types::cache_v>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::cache_v>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::cache_v>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::per_block> };
		static constexpr auto sub_kernel_type{ global_input_types::cache_v };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::kq_mask>
		: public core_elem_base<global_input_types::kq_mask, sub_kernel_traits_new<config_type_new, global_input_types::kq_mask>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::kq_mask>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::mask_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::mask_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::kq_mask>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::kq_mask };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::inp_out_ids>
		: public core_elem_base<global_input_types::inp_out_ids, sub_kernel_traits_new<config_type_new, global_input_types::inp_out_ids>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::inp_out_ids>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::inp_out_ids>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::inp_out_ids };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::temperature>
		: public core_elem_base<global_input_types::temperature, sub_kernel_traits_new<config_type_new, global_input_types::temperature>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::temperature>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::temperature>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::temperature };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::top_k>
		: public core_elem_base<global_input_types::top_k, sub_kernel_traits_new<config_type_new, global_input_types::top_k>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::top_k>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::top_k>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::top_k };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::top_p>
		: public core_elem_base<global_input_types::top_p, sub_kernel_traits_new<config_type_new, global_input_types::top_p>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::top_p>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::top_p>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::top_p };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::repetition_penalty>
		: public core_elem_base<global_input_types::repetition_penalty, sub_kernel_traits_new<config_type_new, global_input_types::repetition_penalty>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::repetition_penalty>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::repetition_penalty>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::repetition_penalty };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::presence_penalty>
		: public core_elem_base<global_input_types::presence_penalty, sub_kernel_traits_new<config_type_new, global_input_types::presence_penalty>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::presence_penalty>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::presence_penalty>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::presence_penalty };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::frequency_penalty>
		: public core_elem_base<global_input_types::frequency_penalty, sub_kernel_traits_new<config_type_new, global_input_types::frequency_penalty>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::frequency_penalty>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::frequency_penalty>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::frequency_penalty };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::rep_window>
		: public core_elem_base<global_input_types::rep_window, sub_kernel_traits_new<config_type_new, global_input_types::rep_window>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::rep_window>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::rep_window>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::rep_window };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::token_history>
		: public core_elem_base<global_input_types::token_history, sub_kernel_traits_new<config_type_new, global_input_types::token_history>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::token_history>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::token_history>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::token_history };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::rng_state>
		: public core_elem_base<global_input_types::rng_state, sub_kernel_traits_new<config_type_new, global_input_types::rng_state>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::rng_state>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::rng_state>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::rng_state };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::logits_bias>
		: public core_elem_base<global_input_types::logits_bias, sub_kernel_traits_new<config_type_new, global_input_types::logits_bias>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::logits_bias>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size, dim_traits<config_type_new, global_input_types::logits_bias>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::logits_bias };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_input_types::allowed_vocab_mask>
		: public core_elem_base<global_input_types::allowed_vocab_mask, sub_kernel_traits_new<config_type_new, global_input_types::allowed_vocab_mask>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::allowed_vocab_mask>>()>,
		  public data_mixin<config_type_new, data_strategy_types::global, typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::mask_type> {
		using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::mask_type;
		using dims_type	  = kernel_dims_new<get_runtime_mask<config_type_new::batched_processing>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_input_types::allowed_vocab_mask>>()>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::dims)),
			model_traits_type<config_type_new>::block_count, data_strategy_types::global> };
		static constexpr auto sub_kernel_type{ global_input_types::allowed_vocab_mask };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>
		: public core_elem_base<token_embeddings_sub_kernel_types::token_embeddings_get_rows,
			  sub_kernel_traits_new<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::token_embd>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, global_input_types::inp_tokens>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>>()>;
		static constexpr auto sub_kernel_type{ token_embeddings_sub_kernel_types::token_embeddings_get_rows };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::norm_rms_norm, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::norm_rms_norm };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::attn_norm_mul, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, weight_types::attn_norm>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::attn_norm_mul };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::q_mul_mat, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::attn_q>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::q_reshape, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_reshape };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::q_rope, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::inp_pos>;
		using input_type_03 = sub_kernel_traits_new<config_type_new, weight_types::rope_freqs>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_rope };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::k_mul_mat, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::attn_k>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::k_reshape, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_reshape };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::k_rope, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::inp_pos>;
		using input_type_03 = sub_kernel_traits_new<config_type_new, weight_types::rope_freqs>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_rope };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::v_mul_mat, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::attn_v>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::k_cache_view, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_input_types::cache_k>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_cache_view };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_cpy>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::k_cache_cpy, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_cpy>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_cpy>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_cpy>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_cache_cpy };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::v_transpose, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_transpose };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::v_cache_view, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_input_types::cache_v>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_cache_view };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_cpy>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::v_cache_cpy, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_cpy>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_cpy>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_cpy>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_cache_cpy };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_view>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::v_view, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_view>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_view>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_input_types::cache_v>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 0>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::v_view>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_view };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_view>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::k_view, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_view>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_view>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_input_types::cache_k>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::k_view>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_view };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::q_permute, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_permute };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat>
		: public core_elem_base<attn_prep_and_score_sub_kernel_types::kq_mul_mat, sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::attention_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::k_view>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::kq_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::kq_soft_max, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::attention_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::kq_mask>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kq_soft_max };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::kqv_mul_mat, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, attn_prep_and_score_sub_kernel_types::v_view>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 2>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kqv_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::kqv_permute, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kqv_permute };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::kqv_cont, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kqv_cont };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat,
			  sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::attn_output>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_inp_add, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_inp_add };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_norm_mul, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, weight_types::ffn_norm>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_norm_mul };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::ffn_gate>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_gate_silu, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_gate_silu };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::ffn_up>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::ffn_down>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::layer_out_add, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::layer_out_add };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::l_out_get_rows, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::inp_out_ids>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::l_out_get_rows };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add>
		: public core_elem_base<attn_out_and_ffn_sub_kernel_types::extracted_add, sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add>>()>;
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::extracted_add };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::output_norm_rms_norm,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::output_norm_rms_norm };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::output_norm_mul,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, weight_types::output_norm>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::output_norm_mul };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::output_projection_mul_mat,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat>>()> {
		using output_type				= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01				= sub_kernel_traits_new<config_type_new, weight_types::output>;
		using input_type_02				= sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul>;
		static constexpr bool quantized = !detail::is_same_v<output_type, typename input_type_01::output_type>;
		using dims_type					= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
							get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
								dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::output_projection_mul_mat };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_repetition_penalty,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::token_history>;
		using input_type_03 = sub_kernel_traits_new<config_type_new, global_input_types::repetition_penalty>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_repetition_penalty };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_presence_penalty,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::token_history>;
		using input_type_03 = sub_kernel_traits_new<config_type_new, global_input_types::presence_penalty>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_presence_penalty };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_frequency_penalty,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::token_history>;
		using input_type_03 = sub_kernel_traits_new<config_type_new, global_input_types::frequency_penalty>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_frequency_penalty };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_logits_bias,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::logits_bias>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_logits_bias };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_vocab_mask,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::allowed_vocab_mask>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_vocab_mask };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_temperature,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::temperature>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_temperature };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::compute_softmax,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::compute_softmax };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_top_k_filter,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::top_k>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_top_k_filter };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::apply_top_p_filter,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::logit_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::top_p>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_top_p_filter };
	};

	template<typename config_type_new> struct sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::sample_token>
		: public core_elem_base<global_output_and_sampling_sub_kernel_types::sample_token,
			  sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::sample_token>>,
		  public kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
			  get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
				  dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::sample_token>>()> {
		using output_type	= typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
		using input_type_01 = sub_kernel_traits_new<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter>;
		using input_type_02 = sub_kernel_traits_new<config_type_new, global_input_types::rng_state>;
		using dims_type		= kernel_dims_new<get_runtime_mask<config_type_new::batched_processing, 1>(),
				get_shifted_dimensions_new<config_type_new::batched_processing, config_type_new::batch_size,
					dim_traits<config_type_new, global_output_and_sampling_sub_kernel_types::sample_token>>()>;
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::sample_token };
	};

	template<typename config_type_new, typename enum_type> struct sub_kernel_new_aggregator;

	template<llama_arch_config_types config_type_new> struct sub_kernel_new_aggregator<config_type_new, weight_types> {
		static constexpr array values{ weight_types::attn_q, weight_types::attn_k, weight_types::attn_v, weight_types::attn_output, weight_types::attn_norm, weight_types::ffn_gate,
			weight_types::ffn_up, weight_types::ffn_down, weight_types::ffn_norm, weight_types::token_embd, weight_types::rope_freqs, weight_types::output_norm,
			weight_types::output };
	};

	template<llama_arch_config_types config_type_new> struct sub_kernel_new_aggregator<config_type_new, global_input_types> {
		static constexpr array values{ global_input_types::inp_tokens, global_input_types::inp_pos, global_input_types::cache_k, global_input_types::cache_v,
			global_input_types::kq_mask, global_input_types::inp_out_ids, global_input_types::temperature, global_input_types::top_k, global_input_types::top_p,
			global_input_types::repetition_penalty, global_input_types::presence_penalty, global_input_types::frequency_penalty, global_input_types::rep_window,
			global_input_types::token_history, global_input_types::rng_state, global_input_types::logits_bias, global_input_types::allowed_vocab_mask };
	};

	template<llama_arch_config_types config_type_new> struct sub_kernel_new_aggregator<config_type_new, token_embeddings_sub_kernel_types> {
		static constexpr array values{ token_embeddings_sub_kernel_types::token_embeddings_get_rows };
	};

	template<llama_arch_config_types config_type_new> struct sub_kernel_new_aggregator<config_type_new, attn_prep_and_score_sub_kernel_types> {
		static constexpr array values{ attn_prep_and_score_sub_kernel_types::norm_rms_norm, attn_prep_and_score_sub_kernel_types::attn_norm_mul,
			attn_prep_and_score_sub_kernel_types::q_mul_mat, attn_prep_and_score_sub_kernel_types::q_reshape, attn_prep_and_score_sub_kernel_types::q_rope,
			attn_prep_and_score_sub_kernel_types::k_mul_mat, attn_prep_and_score_sub_kernel_types::k_reshape, attn_prep_and_score_sub_kernel_types::k_rope,
			attn_prep_and_score_sub_kernel_types::v_mul_mat, attn_prep_and_score_sub_kernel_types::k_cache_view, attn_prep_and_score_sub_kernel_types::k_cache_cpy,
			attn_prep_and_score_sub_kernel_types::v_transpose, attn_prep_and_score_sub_kernel_types::v_cache_view, attn_prep_and_score_sub_kernel_types::v_cache_cpy,
			attn_prep_and_score_sub_kernel_types::v_view, attn_prep_and_score_sub_kernel_types::k_view, attn_prep_and_score_sub_kernel_types::q_permute,
			attn_prep_and_score_sub_kernel_types::kq_mul_mat };
	};

	template<llama_arch_config_types config_type_new> struct sub_kernel_new_aggregator<config_type_new, attn_out_and_ffn_sub_kernel_types> {
		static constexpr array values{ attn_out_and_ffn_sub_kernel_types::kq_soft_max, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat,
			attn_out_and_ffn_sub_kernel_types::kqv_permute, attn_out_and_ffn_sub_kernel_types::kqv_cont, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat,
			attn_out_and_ffn_sub_kernel_types::ffn_inp_add, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul,
			attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat,
			attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat, attn_out_and_ffn_sub_kernel_types::layer_out_add,
			attn_out_and_ffn_sub_kernel_types::l_out_get_rows, attn_out_and_ffn_sub_kernel_types::extracted_add };
	};

	template<llama_arch_config_types config_type_new> struct sub_kernel_new_aggregator<config_type_new, global_output_and_sampling_sub_kernel_types> {
		static constexpr array values{ global_output_and_sampling_sub_kernel_types::output_norm_rms_norm, global_output_and_sampling_sub_kernel_types::output_norm_mul,
			global_output_and_sampling_sub_kernel_types::output_projection_mul_mat, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty,
			global_output_and_sampling_sub_kernel_types::apply_presence_penalty, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty,
			global_output_and_sampling_sub_kernel_types::apply_logits_bias, global_output_and_sampling_sub_kernel_types::apply_vocab_mask,
			global_output_and_sampling_sub_kernel_types::apply_temperature, global_output_and_sampling_sub_kernel_types::compute_softmax,
			global_output_and_sampling_sub_kernel_types::apply_top_k_filter, global_output_and_sampling_sub_kernel_types::apply_top_p_filter,
			global_output_and_sampling_sub_kernel_types::sample_token };
	};

	template<typename value_type>
	concept has_total_required_bytes = requires() { detail::remove_cvref_t<value_type>::total_required_bytes; };

	template<typename config_type_new, typename value_type, uint64_t current_index, uint64_t max_index> constexpr uint64_t get_total_required_bytes_impl() {
		if constexpr (has_total_required_bytes<value_type>) {
			return value_type::total_required_bytes;
		} else if constexpr (current_index == max_index - 1) {
			return get_total_required_bytes<round_up_to_multiple<64>(type_traits<typename value_type::output_type>::total_byte_size(value_type::dims)),
				model_traits_type<config_type_new>::block_count, data_strategy_types::global>;
		} else {
			return 0;
		}
	}

	template<typename config_type_new, typename enum_type, size_t... indices> constexpr uint64_t get_total_required_bytes_new(std::index_sequence<indices...>) {
		uint64_t return_value{ (
			get_total_required_bytes_impl<config_type_new, sub_kernel_traits_new<config_type_new, static_cast<enum_type>(indices)>, indices, sizeof...(indices)>() + ...) };
		return return_value;
	}

	template<typename config_type_new, typename enum_type> constexpr uint64_t get_total_required_bytes_new() {
		return get_total_required_bytes_new<config_type_new, enum_type>(std::make_index_sequence<sub_kernel_new_aggregator<config_type_new, enum_type>::values.size()>{});
	}*/

}
