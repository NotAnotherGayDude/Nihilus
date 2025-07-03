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

#include <nihilus/common/tuple.hpp>

namespace nihilus {

	template<typename weight_type_new, typename activation_type_new, typename compute_type_new, typename scale_type_new, typename index_type_new, typename output_type_new>
	struct kernel_type_profile_traits_impl {
		using weight_type			  = weight_type_new;
		using activation_type		  = activation_type_new;
		using compute_type			  = compute_type_new;
		using scale_type			  = scale_type_new;
		using index_type			  = index_type_new;
		using output_type			  = output_type_new;
		using cache_type			  = scale_type_new;
		using token_embd_weight_type  = weight_type;
		using attn_q_weight_type	  = weight_type;
		using attn_k_weight_type	  = weight_type;
		using attn_v_weight_type	  = weight_type;
		using attn_output_weight_type = weight_type;
		using attn_norm_weight_type	  = compute_type;
		using ffn_gate_weight_type	  = weight_type;
		using ffn_up_weight_type	  = weight_type;
		using ffn_down_weight_type	  = weight_type;
		using ffn_norm_weight_type	  = compute_type;
		using output_weight_type	  = weight_type;
		using output_norm_weight_type = output_type_new;
		using rope_freqs_weight_type  = compute_type;
		using embedding_type		  = compute_type;
		using query_type			  = compute_type;
		using key_type				  = compute_type;
		using value_type			  = compute_type;
		using attention_score_type	  = compute_type;
		using softmax_type			  = compute_type;
		using hidden_type			  = compute_type;
		using residual_type			  = compute_type;
		using norm_output_type		  = compute_type;
		using ffn_intermediate_type	  = compute_type;
		using logit_type			  = compute_type;
		using input_token_type		  = index_type;
		using output_token_type		  = index_type;
		using position_type			  = index_type;
		using kv_cache_type			  = cache_type;
		using inp_embd_type			  = compute_type;
		using inp_tokens_type		  = index_type;
		using inp_pos_type			  = index_type;
		using inp_out_ids_type		  = index_type;
		using cache_k_type			  = cache_type;
		using cache_v_type			  = cache_type;
		using kq_mask_type			  = compute_type;
		using norm_types			  = compute_type;
		using attn_norm_type		  = compute_type;
		using qcur_type				  = compute_type;
		using qcur_reshaped_type	  = compute_type;
		using qcur_rope_type		  = compute_type;
		using kcur_type				  = compute_type;
		using kcur_reshaped_type	  = compute_type;
		using kcur_rope_type		  = compute_type;
		using vcur_type				  = compute_type;
		using k_cache_view_type		  = cache_type;
		using k_cache_view_copy_type  = cache_type;
		using vcur_transposed_type	  = compute_type;
		using v_cache_view_type		  = cache_type;
		using v_cache_view_copy_type  = cache_type;
		using v_type				  = compute_type;
		using k_type				  = compute_type;
		using q_type				  = compute_type;
		using kq_type				  = compute_type;
		using kq_soft_max_type		  = compute_type;
		using kqv_type				  = compute_type;
		using kqv_merged_type		  = compute_type;
		using kqv_merged_cont_type	  = compute_type;
		using kqv_out_type			  = compute_type;
		using ffn_inp_type			  = compute_type;
		using norm_out_type			  = compute_type;
		using ffn_norm_type			  = compute_type;
		using ffn_gate_type			  = compute_type;
		using ffn_silu_type			  = compute_type;
		using ffn_up_type			  = compute_type;
		using ffn_gate_par_type		  = compute_type;
		using ffn_out_type			  = compute_type;
		using l_out_type			  = compute_type;
		using attn_residual_type	  = compute_type;
		using prev_residual_type	  = compute_type;
		using final_norm_type		  = compute_type;
		using result_norm_type		  = compute_type;
		using result_output_type	  = compute_type;
	};

	template<kernel_type_profiles kernel_profile> struct kernel_type_profile_traits;

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q8_gqa> : public kernel_type_profile_traits_impl<block_q8_0<half>, float, float, int16_t, int32_t, float> {};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_mha> : public kernel_type_profile_traits_impl<block_q8_0<half>, float, float, int16_t, int32_t, float> {
	};

}
