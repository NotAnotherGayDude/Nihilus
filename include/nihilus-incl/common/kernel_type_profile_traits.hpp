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

#include <nihilus-incl/common/concepts.hpp>
#include <nihilus-incl/common/tuple.hpp>

namespace nihilus {

	template<typename weight_type_new, typename activation_type_new, typename compute_type_new, typename embedding_type_new, typename logit_type_new, typename token_type_new,
		typename attention_type_new, typename norm_type_new, typename scale_type_new, typename zero_point_type_new, typename kv_cache_type_new, typename mask_type_new,
		typename index_type_new>
	struct kernel_type_profile_traits_impl {
		using weight_type	  = weight_type_new;
		using activation_type = activation_type_new;
		using compute_type	  = compute_type_new;
		using embedding_type  = embedding_type_new;
		using logit_type	  = logit_type_new;
		using token_type	  = token_type_new;
		using attention_type  = attention_type_new;
		using norm_type		  = norm_type_new;
		using scale_type	  = scale_type_new;
		using zero_point_type = zero_point_type_new;
		using kv_cache_type	  = kv_cache_type_new;
		using mask_type		  = mask_type_new;
		using index_type	  = index_type_new;
	};

	template<kernel_type_profiles kernel_type_profile> struct kernel_type_profile_traits;

	template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_mha> : public kernel_type_profile_traits_impl<half,// weight_type
																					   half,// activation_type
																					   half,// compute_type
																					   half,// embedding_type
																					   half,// logit_type
																					   int32_t,// token_token_type
																					   half,// attention_type
																					   half,// norm_type
																					   half,// scale_type
																					   int8_t,// zero_point_type
																					   half,// kv_cache_type
																					   half,// mask_type
																					   int32_t// index_type
																					   > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::fp16_mha };
		static constexpr const char name[]{ "FP16-MHA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::fp16_moe> : public kernel_type_profile_traits_impl<half,// weight_type
																					   half,// activation_type
																					   half,// compute_type (Assuming hardware support)
																					   half,// embedding_type
																					   half,// logit_type
																					   int32_t,// token_token_type
																					   half,// attention_type
																					   half,// norm_type
																					   half,// scale_type
																					   int8_t,// zero_point_type
																					   half,// kv_cache_type
																					   half,// mask_type
																					   int32_t// index_type
																					   > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::fp16_moe };
		static constexpr const char name[]{ "FP16-MoE" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::bf16_mha> : public kernel_type_profile_traits_impl<bf16_t,// weight_type
																					   bf16_t,// activation_type
																					   bf16_t,// compute_type (Best for high dynamic range)
																					   bf16_t,// embedding_type
																					   bf16_t,// logit_type
																					   int32_t,// token_type
																					   bf16_t,// attention_type
																					   bf16_t,// norm_type
																					   bf16_t,// scale_type
																					   int8_t,// zero_point_type
																					   bf16_t,// kv_cache_type
																					   bf16_t,// mask_type
																					   int32_t// index_type
																					   > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::bf16_mha };
		static constexpr const char name[]{ "BF16-MHA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::bf16_gqa> : public kernel_type_profile_traits_impl<bf16_t,// weight_type
																					   bf16_t,// activation_type
																					   bf16_t,// compute_type
																					   bf16_t,// embedding_type
																					   bf16_t,// logit_type
																					   int32_t,// token_type
																					   bf16_t,// attention_type
																					   bf16_t,// norm_type
																					   bf16_t,// scale_type
																					   int8_t,// zero_point_type
																					   bf16_t,// kv_cache_type
																					   bf16_t,// mask_type
																					   int32_t// index_type
																					   > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::bf16_gqa };
		static constexpr const char name[]{ "BF16-GQA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q4_mha> : public kernel_type_profile_traits_impl<block_q4_k<half>,// weight_type (using common k-quant block)
																					 half,// activation_type
																					 float,// compute_type (Promote to FP32)
																					 half,// embedding_type
																					 float,// logit_type
																					 int32_t,// token_type
																					 float,// attention_type
																					 float,// norm_type
																					 half,// scale_type
																					 int8_t,// zero_point_type
																					 half,// kv_cache_type
																					 float,// mask_type
																					 int32_t// index_type
																					 > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::q4_mha };
		static constexpr const char name[]{ "Q4-MHA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q4_gqa> : public kernel_type_profile_traits_impl<block_q4_k<half>,// weight_type
																					 half,// activation_type
																					 float,// compute_type
																					 half,// embedding_type
																					 float,// logit_type
																					 int32_t,// token_type
																					 float,// attention_type
																					 float,// norm_type
																					 half,// scale_type
																					 int8_t,// zero_point_type
																					 half,// kv_cache_type
																					 float,// mask_type
																					 int32_t// index_type
																					 > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::q4_gqa };
		static constexpr const char name[]{ "Q4-GQA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q4_moe> : public kernel_type_profile_traits_impl<block_q4_k<half>,// weight_type
																					 half,// activation_type
																					 float,// compute_type
																					 half,// embedding_type
																					 float,// logit_type
																					 int32_t,// token_type
																					 float,// attention_type
																					 float,// norm_type
																					 half,// scale_type
																					 int8_t,// zero_point_type
																					 half,// kv_cache_type
																					 float,// mask_type
																					 int32_t// index_type
																					 > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::q4_moe };
		static constexpr const char name[]{ "Q4-MoE" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q8_mha> : public kernel_type_profile_traits_impl<block_q8_0<half>,// weight_type
																					 half,// activation_type
																					 float,// compute_type
																					 half,// embedding_type
																					 float,// logit_type
																					 int32_t,// token_type
																					 float,// attention_type
																					 float,// norm_type
																					 half,// scale_type
																					 int8_t,// zero_point_type
																					 half,// kv_cache_type
																					 float,// mask_type
																					 int32_t// index_type
																					 > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::q8_mha };
		static constexpr const char name[]{ "Q8-MHA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q8_gqa> : public kernel_type_profile_traits_impl<block_q8_0<half>,// weight_type
																					 half,// activation_type
																					 float,// compute_type
																					 half,// embedding_type
																					 float,// logit_type
																					 int32_t,// token_type
																					 float,// attention_type
																					 float,// norm_type
																					 half,// scale_type
																					 int8_t,// zero_point_type
																					 half,// kv_cache_type
																					 float,// mask_type
																					 int32_t// index_type
																					 > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::q8_gqa };
		static constexpr const char name[]{ "Q8-GQA" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q8_moe> : public kernel_type_profile_traits_impl<block_q8_0<half>,// weight_type
																					 half,// activation_type
																					 float,// compute_type
																					 half,// embedding_type
																					 float,// logit_type
																					 int32_t,// token_type
																					 float,// attention_type
																					 float,// norm_type
																					 half,// scale_type
																					 int8_t,// zero_point_type
																					 half,// kv_cache_type
																					 float,// mask_type
																					 int32_t// index_type
																					 > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::q8_moe };
		static constexpr const char name[]{ "Q8-MoE" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::mixed_fp16_fp32> : public kernel_type_profile_traits_impl<half,// weight_type (Storage is FP16)
																							  half,// activation_type
																							  float,// compute_type (Compute is FP32)
																							  half,// embedding_type
																							  float,// logit_type
																							  int32_t,// token_type
																							  float,// attention_type
																							  float,// norm_type
																							  half,// scale_type
																							  int8_t,// zero_point_type
																							  half,// kv_cache_type
																							  float,// mask_type
																							  int32_t// index_type
																							  > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::mixed_fp16_fp32 };
		static constexpr const char name[]{ "Mixed-FP16/FP32" };
	};

	template<> struct kernel_type_profile_traits<kernel_type_profiles::mixed_bf16_fp32> : public kernel_type_profile_traits_impl<bf16_t,// weight_type (Storage is BF16)
																							  bf16_t,// activation_type
																							  float,// compute_type (Compute is FP32)
																							  bf16_t,// embedding_type
																							  float,// logit_type
																							  int32_t,// token_type
																							  float,// attention_type
																							  float,// norm_type
																							  bf16_t,// scale_type
																							  int8_t,// zero_point_type
																							  bf16_t,// kv_cache_type
																							  float,// mask_type
																							  int32_t// index_type
																							  > {
		static constexpr kernel_type_profiles type{ kernel_type_profiles::mixed_bf16_fp32 };
		static constexpr const char name[]{ "Mixed-BF16/FP32" };
	};
}
