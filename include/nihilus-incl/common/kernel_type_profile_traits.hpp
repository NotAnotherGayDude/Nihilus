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

#include <nihilus-incl/common/tuple.hpp>

namespace nihilus {

	template<typename weight_type_new, typename activation_type_new, typename compute_type_new, typename embedding_type_new, typename logit_type_new, typename input_token_type_new,
		typename output_token_type_new, typename position_type_new, typename attention_type_new, typename norm_type_new, typename scale_type_new, typename zero_point_type_new,
		typename kv_cache_type_new, typename mask_type_new, typename index_type_new, typename size_type_new>
	struct kernel_type_profile_traits_impl {
		using weight_type		= weight_type_new;
		using activation_type	= activation_type_new;
		using compute_type		= compute_type_new;
		using embedding_type	= embedding_type_new;
		using logit_type		= logit_type_new;
		using attention_type	= attention_type_new;
		using norm_type			= norm_type_new;
		using kv_cache_type		= kv_cache_type_new;
		using input_token_type	= input_token_type_new;
		using output_token_type = output_token_type_new;
		using position_type		= position_type_new;
		using scale_type		= scale_type_new;
		using zero_point_type	= zero_point_type_new;
		using mask_type			= mask_type_new;
		using index_type		= index_type_new;
		using size_type			= size_type_new;
	};

	template<kernel_type_profiles kernel_profile> struct kernel_type_profile_traits;

	template<> struct kernel_type_profile_traits<kernel_type_profiles::q8_gqa> : public kernel_type_profile_traits_impl<block_q8_0<half>, half, float, half, float, int32_t,
																					 int32_t, int32_t, float, float, half, int8_t, half, float, int32_t, uint64_t> {};

}
