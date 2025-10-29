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

#include <nihilus-incl/infra/model.hpp>
#include <nihilus-incl/infra/model_parser.hpp>
#include <nihilus-incl/common/common.hpp>
#include <cstdint>

namespace nihilus {

	enum class tensor_types {
		// Weights.
		attn_q,
		attn_k,
		attn_v,
		attn_output,
		attn_norm,
		ffn_gate,
		ffn_up,
		ffn_down,
		ffn_norm,
		token_embd,
		rope_freqs,
		output_norm,
		output,
		// Global Inputs.
		inp_tokens,
		inp_pos,
		cache_k,
		cache_v,
		kq_mask,
		inp_out_ids,
		temperature,
		top_k,
		top_p,
		repetition_penalty,
		presence_penalty,
		frequency_penalty,
		rep_window,
		token_history,
		rng_state,
		logits_bias,
		allowed_vocab_mask,
		// Token-Embeddings Mega-Kernel.
		inp_embd_get_rows,
		// Attn Mega-Kernel.
		norm_pre_attn_rms_norm,
		attn_norm_mul,
		qcur_mul_mat,
		qcur_reshape,
		qcur_rope,
		kcur_mul_mat,
		kcur_reshape,
		kcur_rope,
		vcur_mul_mat,
		k_cache_view,
		k_cache_cpy,
		vcur_transpose,
		v_cache_view,
		v_cache_cpy,
		v_from_cache_view,
		k_from_cache_view,
		q_permute,
		kq_mul_mat,
		// Ffn Mega-Kernel.
		kq_soft_max,
		kqv_mul_mat,
		kqv_merged_permute,
		kqv_merged_cont,
		kqv_out_mul_mat,
		ffn_inp_add,
		norm_pre_ffn_rms_norm,
		ffn_norm_mul,
		ffn_gate_mul_mat,
		ffn_silu_unary,
		ffn_up_mul_mat,
		ffn_gate_par_mul,
		ffn_out_mul_mat,
		layer_out_add,
		// Global-Output Mega-Kernel.
		node_1016_get_rows,
		node_1017_get_rows,
		final_ffn_inp_add,
		final_norm_pre_rms_norm,
		final_norm_mul,
		result_norm_mul,
		result_output_mul_mat,
		apply_repetition_penalty,
		apply_presence_penalty,
		apply_frequency_penalty,
		apply_logits_bias,
		apply_vocab_mask,
		apply_temperature,
		compute_softmax,
		apply_top_k_filter,
		apply_top_p_filter,
		sample_token,
		count
	};

}
