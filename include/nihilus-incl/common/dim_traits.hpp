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

namespace nihilus {

	template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

	template<typename config_type_new, enum_types auto enum_value_new, typename enum_type = decltype(enum_value_new)> struct dim_traits_newer;

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::attn_q> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::embedding_length, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::attn_q };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::attn_k> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::n_embd_kv_gqa, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::attn_k };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::attn_v> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::n_embd_kv_gqa, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::attn_v };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::attn_output> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::embedding_length, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::attn_output };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::attn_norm> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::attn_norm };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::ffn_gate> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::feed_forward_length, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::ffn_gate };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::ffn_up> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::feed_forward_length, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::ffn_up };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::ffn_down> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::feed_forward_length, model_traits_type<config_type_new>::embedding_length, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::ffn_down };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::ffn_norm> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::ffn_norm };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::token_embd> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::vocab_size, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::token_embd };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::rope_freqs> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::rope_dimension_count / 2, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::rope_freqs };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::output_norm> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::output_norm };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, weight_types::output> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, model_traits_type<config_type_new>::vocab_size, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ weight_types::output };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::inp_tokens> {
		static constexpr array<uint64_t, 4> dims{ config_type_new::max_sequence_length, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::inp_tokens };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::inp_pos> {
		static constexpr array<uint64_t, 4> dims{ config_type_new::max_sequence_length, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::inp_pos };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::cache_k> {
		static constexpr array<uint64_t, 4> dims{ config_type_new::max_sequence_length, model_traits_type<config_type_new>::n_embd_kv_gqa, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::cache_k };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::cache_v> {
		static constexpr array<uint64_t, 4> dims{ config_type_new::max_sequence_length, model_traits_type<config_type_new>::n_embd_kv_gqa, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::cache_v };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::kq_mask> {
		static constexpr array<uint64_t, 4> dims{ 32ull, 32ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::kq_mask };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::inp_out_ids> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::inp_out_ids };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::temperature> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::temperature };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::top_k> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::top_k };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::top_p> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::top_p };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::repetition_penalty> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::repetition_penalty };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::presence_penalty> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::presence_penalty };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::frequency_penalty> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::frequency_penalty };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::rep_window> {
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::rep_window };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::token_history> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::context_length, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::token_history };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::rng_state> {
		static constexpr array<uint64_t, 4> dims{ 256ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::rng_state };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::logits_bias> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::vocab_size, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::logits_bias };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_input_types::allowed_vocab_mask> {
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::vocab_size, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_input_types::allowed_vocab_mask };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::token_embd>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::inp_tokens>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_02[0], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ token_embeddings_sub_kernel_types::token_embeddings_get_rows };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::norm_rms_norm };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::norm_rms_norm>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, weight_types::attn_norm>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::attn_norm_mul };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::attn_q>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::rope_dimension_count, model_traits_type<config_type_new>::attention_head_count,
			input_dims_01[1], 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_reshape };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_reshape>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::inp_pos>::dims;
		static constexpr array<uint64_t, 4> input_dims_03 = dim_traits_newer<config_type_new, weight_types::rope_freqs>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], input_dims_01[2], 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_rope };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::attn_k>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::rope_dimension_count, model_traits_type<config_type_new>::attention_head_count_kv,
			input_dims_01[1], 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_reshape };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_reshape>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::inp_pos>::dims;
		static constexpr array<uint64_t, 4> input_dims_03 = dim_traits_newer<config_type_new, weight_types::rope_freqs>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], input_dims_01[2], 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_rope };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::attn_v>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::attn_norm_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_input_types::cache_k>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0] * input_dims_01[1], 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_cache_view };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_cpy> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_rope>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_cache_view>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_02[0], 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_cache_cpy };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_01[0], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_transpose };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_input_types::cache_v>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_cache_view };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_cpy> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_transpose>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_cache_view>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_cache_cpy };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_view> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_input_types::cache_v>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::kq_mask>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_02[0], model_traits_type<config_type_new>::rope_dimension_count,
			model_traits_type<config_type_new>::attention_head_count_kv, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::v_view };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_view> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_input_types::cache_k>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::kq_mask>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::rope_dimension_count, input_dims_02[0],
			model_traits_type<config_type_new>::attention_head_count_kv, 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::k_view };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_rope>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[2], input_dims_01[1], 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::q_permute };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::k_view>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::q_permute>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], input_dims_02[2], 1ull };
		static constexpr auto sub_kernel_type{ attn_prep_and_score_sub_kernel_types::kq_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::kq_mul_mat>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::kq_mask>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], input_dims_01[2], 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kq_soft_max };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_prep_and_score_sub_kernel_types::v_view>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kq_soft_max>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], input_dims_02[2], 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kqv_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[2], input_dims_01[1], 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kqv_permute };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_permute>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ model_traits_type<config_type_new>::embedding_length, input_dims_01[2], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::kqv_cont };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::attn_output>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::kqv_cont>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::attn_output_mul_mat>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, token_embeddings_sub_kernel_types::token_embeddings_get_rows>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_inp_add };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_rms_norm>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, weight_types::ffn_norm>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_norm_mul };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::ffn_gate>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_gate_silu };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::ffn_up>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_norm_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_silu>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_up_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::ffn_down>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_gate_par_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_down_mul_mat>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::ffn_inp_add>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::layer_out_add };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::layer_out_add>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_input_types::inp_out_ids>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_02[0], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::l_out_get_rows };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::l_out_get_rows>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ attn_out_and_ffn_sub_kernel_types::extracted_add };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, attn_out_and_ffn_sub_kernel_types::extracted_add>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::output_norm_rms_norm };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_rms_norm>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, weight_types::output_norm>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::output_norm_mul };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, weight_types::output>::dims;
		static constexpr array<uint64_t, 4> input_dims_02 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::output_norm_mul>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[1], input_dims_02[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::output_projection_mul_mat };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::output_projection_mul_mat>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_repetition_penalty };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_repetition_penalty>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_presence_penalty };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_presence_penalty>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_frequency_penalty };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_frequency_penalty>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_logits_bias };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_logits_bias>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_vocab_mask };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_vocab_mask>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_temperature };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_temperature>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::compute_softmax };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::compute_softmax>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_top_k_filter };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_k_filter>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ input_dims_01[0], input_dims_01[1], 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::apply_top_p_filter };
	};

	template<typename config_type_new> struct dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::sample_token> {
	  protected:
		static constexpr array<uint64_t, 4> input_dims_01 = dim_traits_newer<config_type_new, global_output_and_sampling_sub_kernel_types::apply_top_p_filter>::dims;

	  public:
		static constexpr array<uint64_t, 4> dims{ 1ull, 1ull, 1ull, 1ull };
		static constexpr auto sub_kernel_type{ global_output_and_sampling_sub_kernel_types::sample_token };
	};
}
