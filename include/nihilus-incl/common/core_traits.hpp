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

#include <nihilus-incl/common/core_bases.hpp>
#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/common/kernel_type_profile_traits.hpp>
#include <nihilus-incl/common/model_traits.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <latch>

namespace nihilus {

	template<typename type01, typename type02> struct requires_dequant_or_quant {
		static constexpr bool required{ !std::is_same_v<type01, type02> };
	};

	template<model_config config, data_strategy_types data_strategy_type, typename output_type> struct data_mixin;

	template<model_config config, typename output_type> struct data_mixin<config, data_strategy_types::global, output_type> {
		output_type* data{};
	};

	template<model_config config, typename output_type> struct data_mixin<config, data_strategy_types::per_block, output_type> {
		array<output_type*, model_traits_type<config>::block_count> data{};
	};

	template<model_config config, enum_types auto enum_value_new, composite_kernel_types kernel_type, data_strategy_types data_strategy_type,
		allocation_strategy_types allocation_strategy_type, typename composite_kernel_type_new, typename... input_composite_kernel_types_new>
	struct op_traits : public data_mixin<config, data_strategy_type, typename composite_kernel_type_new::output_type>, public composite_kernel_type_new {
		using output_type = composite_kernel_type_new::output_type;
		using dims_type	  = composite_kernel_type_new::dims_type;
		static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::get_array()) };
		static constexpr auto enum_value{ enum_value_new };
	};

	template<model_config config, core_types core_type, typename derived_type> struct core_traits_base {
		static decltype(auto) decl_elem(tag<core_type>);
		constexpr decltype(auto) operator[](tag<core_type>) & {
			return *static_cast<derived_type*>(this);
		}
		constexpr decltype(auto) operator[](tag<core_type>) const& {
			return *static_cast<const derived_type*>(this);
		}
	};

	template<model_config config> struct core_traits<config, core_types::weights> : public core_traits_base<config, core_types::weights, core_traits<config, core_types::weights>> {
		static constexpr core_types core_type{ core_types::weights };
		static constexpr uint64_t depth{ 0 };
		using attn_q_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::embedding_length, 1, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using attn_k_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using attn_v_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using attn_output_weight_kernel_traits =
			kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::embedding_length, 1, 1>, kernel_types::none,
				typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using attn_norm_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

		using ffn_gate_weight_kernel_traits =
			kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::feed_forward_length, 1, 1>, kernel_types::none,
				typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using ffn_up_weight_kernel_traits =
			kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::feed_forward_length, 1, 1>, kernel_types::none,
				typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using ffn_down_weight_kernel_traits =
			kernel_traits<config, core_trait_dims<model_traits_type<config>::feed_forward_length, model_traits_type<config>::embedding_length, 1, 1>, kernel_types::none,
				typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using ffn_norm_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

		using token_embd_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::vocab_size, 1, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using rope_freqs_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::rope_dimension_count / 2, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

		using output_norm_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

		using output_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::vocab_size, 1, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

		using attn_q_weight_type =
			op_traits<config, weight_types::attn_q, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_q_weight_kernel_traits>;
		using attn_k_weight_type =
			op_traits<config, weight_types::attn_k, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_k_weight_kernel_traits>;
		using attn_v_weight_type =
			op_traits<config, weight_types::attn_v, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_v_weight_kernel_traits>;
		using attn_output_weight_type = op_traits<config, weight_types::attn_output, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap,
			attn_output_weight_kernel_traits>;
		using attn_norm_weight_type	  = op_traits<config, weight_types::attn_norm, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap,
			  attn_norm_weight_kernel_traits>;
		using ffn_gate_weight_type =
			op_traits<config, weight_types::ffn_gate, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_gate_weight_kernel_traits>;
		using ffn_up_weight_type =
			op_traits<config, weight_types::ffn_up, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_up_weight_kernel_traits>;
		using ffn_down_weight_type =
			op_traits<config, weight_types::ffn_down, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_down_weight_kernel_traits>;
		using ffn_norm_weight_type =
			op_traits<config, weight_types::ffn_norm, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_norm_weight_kernel_traits>;
		using token_embd_weight_type  = op_traits<config, weight_types::token_embd, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap,
			 token_embd_weight_kernel_traits>;
		using rope_freqs_weight_type  = op_traits<config, weight_types::rope_freqs, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap,
			 rope_freqs_weight_kernel_traits>;
		using output_norm_weight_type = op_traits<config, weight_types::output_norm, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap,
			output_norm_weight_kernel_traits>;
		using output_weight_type =
			op_traits<config, weight_types::output, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap, output_weight_kernel_traits>;

		using list_of_traits = tuple<attn_q_weight_type, attn_k_weight_type, attn_v_weight_type, attn_output_weight_type, attn_norm_weight_type, ffn_gate_weight_type,
			ffn_up_weight_type, ffn_down_weight_type, ffn_norm_weight_type, token_embd_weight_type, rope_freqs_weight_type, output_norm_weight_type, output_weight_type>;
		list_of_traits values{};
	};


	template<model_config config> struct core_traits<config, core_types::global_inputs>
		: public core_traits_base<config, core_types::global_inputs, core_traits<config, core_types::global_inputs>> {
		static constexpr core_types core_type{ core_types::global_inputs };
		static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
		using enum_type = global_input_types;

		using inp_tokens_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::input_token_type>;

		using inp_pos_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::position_type>;

		using cache_k_kernel_traits = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::block_count, model_traits_type<config>::attention_head_count_kv,
				config.default_max_sequence_length, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type>;

		using cache_v_kernel_traits = kernel_traits<config,
			core_trait_dims<config.default_max_sequence_length, model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count_kv,
				model_traits_type<config>::block_count, 0>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type>;

		using kq_mask_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::block_count, model_traits_type<config>::block_count, 1, 1>,
			kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::mask_type>;

		using inp_out_ids_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::none,
			typename kernel_type_profile_traits<config.kernel_profile>::output_token_type>;

		using cache_k_type =
			op_traits<config, global_input_types::cache_k, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::alloc, cache_k_kernel_traits>;
		using cache_v_type =
			op_traits<config, global_input_types::cache_v, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::alloc, cache_v_kernel_traits>;
		using inp_tokens_type = op_traits<config, global_input_types::inp_tokens, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc,
			inp_tokens_kernel_traits>;
		using inp_pos_type =
			op_traits<config, global_input_types::inp_pos, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc, inp_pos_kernel_traits>;
		using kq_mask_type =
			op_traits<config, global_input_types::kq_mask, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc, kq_mask_kernel_traits>;
		using inp_out_ids_type = op_traits<config, global_input_types::inp_out_ids, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc,
			inp_out_ids_kernel_traits>;

		using list_of_traits = tuple<inp_tokens_type, inp_pos_type, cache_k_type, cache_v_type, kq_mask_type, inp_out_ids_type>;
		list_of_traits values{};

		static constexpr uint64_t total_required_bytes{ inp_tokens_type::total_required_bytes + inp_pos_type::total_required_bytes +
			(model_traits_type<config>::block_count * cache_k_type::total_required_bytes) + (model_traits_type<config>::block_count * cache_v_type::total_required_bytes) +
			kq_mask_type::total_required_bytes + inp_out_ids_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::global_outputs>
		: public core_traits_base<config, core_types::global_outputs, core_traits<config, core_types::global_outputs>> {
		static constexpr core_types core_type{ core_types::global_outputs };
		static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
		using enum_type = global_output_types;

		using rms_norm_kernel_traits = kernel_traits<config,
			get_new_dims_new_1_t<kernel_types::rms_norm, typename core_traits<config, core_types::token_embeddings>::input_embedding_kernel_traits>, kernel_types::rms_norm,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, typename core_traits<config, core_types::token_embeddings>::input_embedding_kernel_traits>;

		using mul_kernel_traits = kernel_traits<config,
			get_new_dims_new_2_t<kernel_types::mul, rms_norm_kernel_traits, typename core_traits<config, core_types::weights>::output_norm_weight_kernel_traits>, kernel_types::mul,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_traits,
			typename core_traits<config, core_types::weights>::output_norm_weight_kernel_traits>;

		using mul_mat_kernel_traits =
			kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, typename core_traits<config, core_types::weights>::output_weight_kernel_traits, mul_kernel_traits>,
				kernel_types::mul_mat, typename kernel_type_profile_traits<config.kernel_profile>::compute_type,
				typename core_traits<config, core_types::weights>::output_weight_kernel_traits, mul_kernel_traits>;

		using logit_sample_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::logit_sample,
			typename kernel_type_profile_traits<config.kernel_profile>::output_token_type, mul_mat_kernel_traits>;

		using result_output_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_logit_sample,
			typename kernel_type_profile_traits<config.kernel_profile>::output_token_type, rms_norm_kernel_traits, mul_kernel_traits, mul_mat_kernel_traits,
			logit_sample_kernel_traits>;

		using result_output_type = op_traits<config, global_output_types::result_output_composite, composite_kernel_types::rms_norm_mul_mul_mat_logit_sample,
			data_strategy_types::global, allocation_strategy_types::alloc, result_output_composite_kernel_traits>;

		using list_of_traits = tuple<result_output_type>;

		list_of_traits values{};
		op_latch latch{};

		static constexpr uint64_t total_required_bytes{ result_output_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::token_embeddings>
		: public core_traits_base<config, core_types::token_embeddings, core_traits<config, core_types::token_embeddings>> {
		static constexpr core_types core_type{ core_types::token_embeddings };
		static constexpr uint64_t depth{ 0 };
		using enum_type = token_embedding_types;

		using input_01_type = typename core_traits<config, core_types::weights>::token_embd_weight_type;
		using input_02_type = typename core_traits<config, core_types::global_inputs>::inp_tokens_type;

		using input_embedding_kernel_traits = kernel_traits<config, get_new_dims_new_2_t<kernel_types::get_rows, input_01_type, input_02_type>, kernel_types::get_rows,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

		using token_embeddings_type = op_traits<config, token_embedding_types::get_rows, composite_kernel_types::get_rows, data_strategy_types::global,
			allocation_strategy_types::alloc, input_embedding_kernel_traits>;

		using list_of_traits = tuple<token_embeddings_type>;

		list_of_traits values{};

		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		op_latch latch{};

		static constexpr uint64_t total_required_bytes{ token_embeddings_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::qkv_projection_layer>
		: public core_traits_base<config, core_types::qkv_projection_layer, core_traits<config, core_types::qkv_projection_layer>> {
		static constexpr core_types core_type{ core_types::qkv_projection_layer };
		static constexpr uint64_t depth{ 1 };
		using enum_type = qkv_projection_types;

		using input_01_type = typename core_traits<config, core_types::token_embeddings>::token_embeddings_type;
		using input_02_type = typename core_traits<config, core_types::weights>::attn_norm_weight_type;
		using input_03_type = typename core_traits<config, core_types::weights>::attn_q_weight_type;
		using input_04_type = typename core_traits<config, core_types::weights>::attn_k_weight_type;
		using input_05_type = typename core_traits<config, core_types::weights>::attn_v_weight_type;
		using input_06_type = typename core_traits<config, core_types::global_inputs>::cache_k_type;
		using input_07_type = typename core_traits<config, core_types::global_inputs>::cache_v_type;

		using rms_norm_kernel_trait = kernel_traits<config, get_new_dims_new_1_t<kernel_types::rms_norm, input_01_type>, kernel_types::rms_norm,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type>;

		using mul_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, rms_norm_kernel_trait, input_02_type>, kernel_types::mul,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, input_02_type>;

		using q_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_03_type, mul_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_03_type, mul_kernel_trait>;

		using q_reshape_kernel_trait = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count, config.default_max_sequence_length, 1, 2>,
			kernel_types::reshape, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, q_mul_mat_kernel_trait>;

		using k_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_04_type, mul_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_04_type, mul_kernel_trait>;

		using k_reshape_kernel_trait = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count_kv, config.default_max_sequence_length, 1, 2>,
			kernel_types::reshape, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, k_mul_mat_kernel_trait>;

		using v_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_05_type, mul_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_05_type, mul_kernel_trait>;

		using v_transpose_kernel_trait = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1, 0>,
			kernel_types::transpose, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, v_mul_mat_kernel_trait>;

		using v_cache_view_kernel_trait = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1, 0>,
			kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_07_type>;

		using v_cache_copy_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::copy, v_transpose_kernel_trait, v_cache_view_kernel_trait>, kernel_types::copy,
			typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, v_transpose_kernel_trait, v_cache_view_kernel_trait>;

		using q_cur_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_reshape,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, mul_kernel_trait, q_mul_mat_kernel_trait, q_reshape_kernel_trait>;

		using k_cur_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_reshape,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, mul_kernel_trait, k_mul_mat_kernel_trait, k_reshape_kernel_trait>;

		using v_cur_composite_kernel_traits =
			composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_transpose_copy, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type,
				rms_norm_kernel_trait, mul_kernel_trait, v_mul_mat_kernel_trait, v_transpose_kernel_trait, v_cache_copy_kernel_trait>;

		using q_cur_type = op_traits<config, qkv_projection_types::q_cur, composite_kernel_types::rms_norm_mul_mul_mat_reshape, data_strategy_types::per_block,
			allocation_strategy_types::alloc, q_cur_composite_kernel_traits>;

		using k_cur_type = op_traits<config, qkv_projection_types::k_cur, composite_kernel_types::rms_norm_mul_mul_mat_reshape, data_strategy_types::per_block,
			allocation_strategy_types::alloc, k_cur_composite_kernel_traits>;

		using v_cur_type = op_traits<config, qkv_projection_types::v_cur, composite_kernel_types::rms_norm_mul_mul_mat_transpose_copy, data_strategy_types::per_block,
			allocation_strategy_types::alloc, v_cur_composite_kernel_traits>;

		using list_of_traits = tuple<q_cur_type, k_cur_type, v_cur_type>;

		list_of_traits values{};
		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ q_cur_type::total_required_bytes + k_cur_type::total_required_bytes + v_cur_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::rope_and_cache_operations>
		: public core_traits_base<config, core_types::rope_and_cache_operations, core_traits<config, core_types::rope_and_cache_operations>> {
		static constexpr core_types core_type{ core_types::rope_and_cache_operations };
		static constexpr uint64_t depth{ 2 };
		using enum_type = rope_and_cache_types;

		using input_01_type = typename core_traits<config, core_types::qkv_projection_layer>::q_cur_type;
		using input_02_type = typename core_traits<config, core_types::qkv_projection_layer>::k_cur_type;
		using input_03_type = typename core_traits<config, core_types::qkv_projection_layer>::v_cur_type;
		using input_04_type = typename core_traits<config, core_types::global_inputs>::inp_pos_type;
		using input_05_type = typename core_traits<config, core_types::weights>::rope_freqs_weight_type;
		using input_06_type = typename core_traits<config, core_types::global_inputs>::cache_k_type;
		using input_07_type = typename core_traits<config, core_types::global_inputs>::cache_v_type;

		using rope_q_kernel_trait = kernel_traits<config, get_new_dims_new_3_t<kernel_types::rope, input_01_type, input_04_type, input_05_type>, kernel_types::rope,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_04_type, input_05_type>;

		using rope_k_kernel_trait = kernel_traits<config, get_new_dims_new_3_t<kernel_types::rope, input_02_type, input_04_type, input_05_type>, kernel_types::rope,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_02_type, input_04_type, input_05_type>;

		using k_cache_view_kernel_trait = kernel_traits<config, core_trait_dims<model_traits_type<config>::n_embd_kv_gqa, config.default_max_sequence_length, 1, 1, 1>,
			kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_06_type>;

		using k_cache_copy_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::copy, rope_k_kernel_trait, k_cache_view_kernel_trait>, kernel_types::copy,
			typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, rope_k_kernel_trait, k_cache_view_kernel_trait>;

		using k_rope_view_kernel_trait = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count, model_traits_type<config>::attention_head_count_kv,
				1>,
			kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_06_type>;

		using v_rope_view_kernel_trait = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::attention_head_count, model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count_kv,
				1>,
			kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_07_type>;

		using rope_q_permute_kernel_trait = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::rope_dimension_count, config.default_max_sequence_length, model_traits_type<config>::attention_head_count, 1>,
			kernel_types::permute, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rope_q_kernel_trait>;

		using rope_q_permute_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rope_permute,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rope_q_kernel_trait, rope_q_permute_kernel_trait>;

		using rope_k_copy_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rope_copy,
			typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, rope_k_kernel_trait, k_cache_view_kernel_trait, k_cache_copy_kernel_trait>;

		using k_rope_view_composite_trait_type =
			composite_kernel_traits<config, composite_kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, k_rope_view_kernel_trait>;

		using v_rope_view_composite_trait_type =
			composite_kernel_traits<config, composite_kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, v_rope_view_kernel_trait>;

		using rope_q_permute_type = op_traits<config, rope_and_cache_types::rope_q_permute_type, composite_kernel_types::rope_permute, data_strategy_types::per_block,
			allocation_strategy_types::alloc, rope_q_permute_composite_kernel_traits>;

		using rope_k_copy_type = op_traits<config, rope_and_cache_types::rope_k_copy_type, composite_kernel_types::rope_copy, data_strategy_types::per_block,
			allocation_strategy_types::alloc, rope_k_copy_composite_kernel_traits>;

		using k_rope_view_type = op_traits<config, rope_and_cache_types::k_rope_view_type, composite_kernel_types::view, data_strategy_types::per_block,
			allocation_strategy_types::alloc, k_rope_view_composite_trait_type>;

		using v_rope_view_type = op_traits<config, rope_and_cache_types::v_rope_view_type, composite_kernel_types::view, data_strategy_types::per_block,
			allocation_strategy_types::alloc, v_rope_view_composite_trait_type>;

		using list_of_traits = tuple<rope_q_permute_type, rope_k_copy_type, k_rope_view_type, v_rope_view_type>;
		list_of_traits values{};

		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ rope_q_permute_type::total_required_bytes + rope_k_copy_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::attention_scores_computation>
		: public core_traits_base<config, core_types::attention_scores_computation, core_traits<config, core_types::attention_scores_computation>> {
		static constexpr core_types core_type{ core_types::attention_scores_computation };
		static constexpr uint64_t depth{ 3 };
		using enum_type = attention_scores_types;

		using input_01_type = typename core_traits<config, core_types::rope_and_cache_operations>::rope_q_permute_type;
		using input_02_type = typename core_traits<config, core_types::rope_and_cache_operations>::k_rope_view_type;

		using kq_scores_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_02_type, input_01_type>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_02_type, input_01_type>;

		using kq_scores_composite_kernel_traits =
			composite_kernel_traits<config, composite_kernel_types::mul_mat, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, kq_scores_kernel_trait>;

		using kq_scores_type = op_traits<config, attention_scores_types::kq_scores_type, composite_kernel_types::mul_mat, data_strategy_types::per_block,
			allocation_strategy_types::alloc, kq_scores_composite_kernel_traits>;

		using list_of_traits = tuple<kq_scores_type>;

		list_of_traits values{};

		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ kq_scores_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::attention_weighted_values>
		: public core_traits_base<config, core_types::attention_weighted_values, core_traits<config, core_types::attention_weighted_values>> {
		static constexpr core_types core_type{ core_types::attention_weighted_values };
		static constexpr uint64_t depth{ 4 };
		using enum_type = attention_weighted_values_types;

		using input_01_type = typename core_traits<config, core_types::attention_scores_computation>::kq_scores_type;
		using input_02_type = typename core_traits<config, core_types::global_inputs>::kq_mask_type;
		using input_03_type = typename core_traits<config, core_types::rope_and_cache_operations>::v_rope_view_type;

		using softmax_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::softmax, input_01_type, input_02_type>, kernel_types::softmax,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

		using attention_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_03_type, softmax_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_03_type, softmax_kernel_trait>;

		using attention_permute_kernel_trait = kernel_traits<config,
			core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count, config.default_max_sequence_length, 1, 2>,
			kernel_types::permute, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, attention_mul_mat_kernel_trait>;

		using attention_cont_kernel_trait = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, config.default_max_sequence_length, 1, 1, 1>,
			kernel_types::cont, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, attention_permute_kernel_trait>;

		using attention_output_composite_kernel_traits =
			composite_kernel_traits<config, composite_kernel_types::softmax_mul_mat_permute_cont, typename kernel_type_profile_traits<config.kernel_profile>::compute_type,
				softmax_kernel_trait, attention_mul_mat_kernel_trait, attention_permute_kernel_trait, attention_cont_kernel_trait>;

		using attention_output_type = op_traits<config, attention_weighted_values_types::attention_output_type, composite_kernel_types::softmax_mul_mat_permute_cont,
			data_strategy_types::per_block, allocation_strategy_types::alloc, attention_output_composite_kernel_traits>;

		using list_of_traits = tuple<attention_output_type>;

		list_of_traits values{};
		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ attention_output_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::attention_output_projection>
		: public core_traits_base<config, core_types::attention_output_projection, core_traits<config, core_types::attention_output_projection>> {
		static constexpr core_types core_type{ core_types::attention_output_projection };
		static constexpr uint64_t depth{ 5 };
		using enum_type = attention_output_projection_types;

		using input_01_type = typename core_traits<config, core_types::attention_weighted_values>::attention_output_type;
		using input_02_type = typename core_traits<config, core_types::weights>::attn_output_weight_type;

		using attn_output_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_02_type, input_01_type>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_02_type, input_01_type>;

		using attn_output_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, attn_output_mul_mat_kernel_trait>;

		using attn_output_type = op_traits<config, attention_output_projection_types::attn_output_type, composite_kernel_types::mul_mat, data_strategy_types::per_block,
			allocation_strategy_types::alloc, attn_output_composite_kernel_traits>;

		using list_of_traits = tuple<attn_output_type>;

		list_of_traits values{};
		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ attn_output_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::ffn_parallel_projections>
		: public core_traits_base<config, core_types::ffn_parallel_projections, core_traits<config, core_types::ffn_parallel_projections>> {
		static constexpr core_types core_type{ core_types::ffn_parallel_projections };
		static constexpr uint64_t depth{ 6 };
		using enum_type = ffn_parallel_projection_types;

		using input_01_type = typename core_traits<config, core_types::attention_output_projection>::attn_output_type;
		using input_02_type = typename core_traits<config, core_types::token_embeddings>::token_embeddings_type;
		using input_03_type = typename core_traits<config, core_types::weights>::ffn_norm_weight_type;
		using input_04_type = typename core_traits<config, core_types::weights>::ffn_gate_weight_type;
		using input_05_type = typename core_traits<config, core_types::weights>::ffn_up_weight_type;

		using add_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::add, input_01_type, input_02_type>, kernel_types::add,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

		using rms_norm_kernel_trait = kernel_traits<config, get_new_dims_new_1_t<kernel_types::rms_norm, add_kernel_trait>, kernel_types::rms_norm,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, add_kernel_trait>;

		using mul_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, rms_norm_kernel_trait, input_03_type>, kernel_types::mul,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, input_03_type>;

		using gate_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_04_type, mul_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_04_type, mul_kernel_trait>;

		using gate_silu_kernel_trait = kernel_traits<config, get_new_dims_new_1_t<kernel_types::silu, gate_mul_mat_kernel_trait>, kernel_types::silu,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, gate_mul_mat_kernel_trait>;

		using up_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_05_type, mul_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_05_type, mul_kernel_trait>;

		using ffn_gate_composite_kernel_traits =
			composite_kernel_traits<config, composite_kernel_types::add_rms_norm_mul_mat_silu, typename kernel_type_profile_traits<config.kernel_profile>::compute_type,
				add_kernel_trait, rms_norm_kernel_trait, mul_kernel_trait, gate_mul_mat_kernel_trait, gate_silu_kernel_trait>;

		using ffn_up_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::add_rms_norm_mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, add_kernel_trait, rms_norm_kernel_trait, mul_kernel_trait, up_mul_mat_kernel_trait>;

		using ffn_gate_type = op_traits<config, ffn_parallel_projection_types::ffn_gate_type, composite_kernel_types::add_rms_norm_mul_mat_silu, data_strategy_types::per_block,
			allocation_strategy_types::alloc, ffn_gate_composite_kernel_traits>;

		using ffn_up_type = op_traits<config, ffn_parallel_projection_types::ffn_up_type, composite_kernel_types::add_rms_norm_mul_mat, data_strategy_types::per_block,
			allocation_strategy_types::alloc, ffn_up_composite_kernel_traits>;

		using list_of_traits = tuple<ffn_gate_type, ffn_up_type>;

		list_of_traits values{};

		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ ffn_gate_type::total_required_bytes + ffn_up_type::total_required_bytes };
	};

	template<model_config config> struct core_traits<config, core_types::ffn_down_projection>
		: public core_traits_base<config, core_types::ffn_down_projection, core_traits<config, core_types::ffn_down_projection>> {
		static constexpr core_types core_type{ core_types::ffn_down_projection };
		static constexpr uint64_t depth{ 7 };
		using enum_type = ffn_down_projection_types;

		using input_01_type = typename core_traits<config, core_types::ffn_parallel_projections>::ffn_gate_type;
		using input_02_type = typename core_traits<config, core_types::ffn_parallel_projections>::ffn_up_type;
		using input_03_type = typename core_traits<config, core_types::weights>::ffn_down_weight_type;
		using input_04_type = typename core_traits<config, core_types::attention_output_projection>::attn_output_type;

		using mul_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, input_01_type, input_02_type>, kernel_types::mul,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

		using mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_03_type, mul_kernel_trait>, kernel_types::mul_mat,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_03_type, mul_kernel_trait>;

		using add_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::add, mul_mat_kernel_trait, input_04_type>, kernel_types::add,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, mul_mat_kernel_trait, input_04_type>;

		using ffn_down_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::mul_mul_mat_add,
			typename kernel_type_profile_traits<config.kernel_profile>::compute_type, mul_kernel_trait, mul_mat_kernel_trait, add_kernel_trait>;

		using ffn_down_type = op_traits<config, ffn_down_projection_types::ffn_down_type, composite_kernel_types::mul_mul_mat_add, data_strategy_types::per_block,
			allocation_strategy_types::alloc, ffn_down_composite_kernel_traits>;

		using list_of_traits = tuple<ffn_down_type>;

		list_of_traits values{};
		array<atomic_flag_wrapper, model_traits_type<config>::block_count> current_chunk{};
		array<op_latch, model_traits_type<config>::block_count> latch{};

		static constexpr uint64_t total_required_bytes{ ffn_down_type::total_required_bytes };
	};

	template<model_config config> struct model;

	template<model_config config_new, auto kernel_type> struct get_adjacent_value;

	template<model_config config_new, auto kernel_type> struct get_adjacent_value {
		using model_traits_type	   = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using derived_derived_type = model<config_new>;
		using thread_pool_type	   = typename derived_derived_type::thread_pool_type;
		template<typename derived_type> NIHILUS_INLINE static auto& impl(derived_type& parse_core) {
			return *static_cast<typename derived_derived_type::thread_pool_type*>(&parse_core);
		}
	};
}
