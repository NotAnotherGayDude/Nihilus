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

#include <nihilus-incl/common/kernel_type_profile_traits.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/tuple.hpp>

namespace nihilus {

	template<model_arches arch, model_sizes model_size, model_generations model_generation> struct model_traits;

	template<> struct model_traits<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1> {
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3_1 };
		static constexpr auto model_size{ model_sizes::llm_8B };
		static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
		static constexpr float rope_freq_base			  = 500000.0f;
		static constexpr uint32_t vocab_size			  = 128256;
		static constexpr uint32_t embedding_length		  = 4096;
		static constexpr uint32_t block_count			  = 32;
		static constexpr uint32_t feed_forward_length	  = 14336;
		static constexpr uint32_t attention_head_count	  = 32;
		static constexpr uint32_t attention_head_count_kv = 8;
		static constexpr uint32_t rope_dimension_count	  = 128;
		static constexpr uint32_t context_length		  = 131072;
		static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llm_3B, model_generations::v3_2> {
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3_2 };
		static constexpr auto model_size{ model_sizes::llm_3B };
		static constexpr float layer_norm_rms_epsilon	  = 1e-5f;
		static constexpr float rope_freq_base			  = 500000.0f;
		static constexpr uint32_t vocab_size			  = 128256;
		static constexpr uint32_t embedding_length		  = 3072;
		static constexpr uint32_t block_count			  = 28;
		static constexpr uint32_t feed_forward_length	  = 8192;
		static constexpr uint32_t attention_head_count	  = 24;
		static constexpr uint32_t attention_head_count_kv = 8;
		static constexpr uint32_t rope_dimension_count	  = 128;
		static constexpr uint32_t context_length		  = 131072;
		static constexpr uint64_t n_embd_kv_gqa			  = rope_dimension_count * attention_head_count_kv;
	};

}
