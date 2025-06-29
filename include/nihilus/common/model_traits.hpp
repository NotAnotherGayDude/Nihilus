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

#include <nihilus/common/kernel_type_profile_traits.hpp>
#include <nihilus/common/tuple.hpp>

namespace nihilus {

	template<model_arches arch, auto model_size, auto model_generation> struct model_traits;

	template<> struct model_traits<model_arches::llama, model_sizes::llama_1B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_1B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 2048;
		static constexpr uint64_t block_count		   = 16;
		static constexpr uint64_t feed_forward_length  = 8192;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 64;
		static constexpr uint64_t rope_dimension_count = 64;
		static constexpr uint64_t total_parameters	   = 1000000000;
		static constexpr uint64_t kv_cache_layers	   = 16;
		static constexpr uint64_t intermediate_size	   = 8192;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_3B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_3B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 3072;
		static constexpr uint64_t block_count		   = 28;
		static constexpr uint64_t feed_forward_length  = 8192;
		static constexpr uint64_t head_count		   = 24;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 3000000000;
		static constexpr uint64_t kv_cache_layers	   = 28;
		static constexpr uint64_t intermediate_size	   = 8192;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_7B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_7B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 4096;
		static constexpr uint64_t block_count		   = 32;
		static constexpr uint64_t feed_forward_length  = 11008;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 32;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 7000000000;
		static constexpr uint64_t kv_cache_layers	   = 32;
		static constexpr uint64_t intermediate_size	   = 11008;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_8B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_8B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 4096;
		static constexpr uint64_t block_count		   = 32;
		static constexpr uint64_t feed_forward_length  = 11008;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 32;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 8000000000;
		static constexpr uint64_t kv_cache_layers	   = 32;
		static constexpr uint64_t intermediate_size	   = 11008;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_11B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_11B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 4096;
		static constexpr uint64_t block_count		   = 32;
		static constexpr uint64_t feed_forward_length  = 11008;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 32;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 11000000000;
		static constexpr uint64_t kv_cache_layers	   = 32;
		static constexpr uint64_t intermediate_size	   = 11008;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_13B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_13B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 5120;
		static constexpr uint64_t block_count		   = 40;
		static constexpr uint64_t feed_forward_length  = 13824;
		static constexpr uint64_t head_count		   = 40;
		static constexpr uint64_t head_count_kv		   = 40;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 13000000000;
		static constexpr uint64_t kv_cache_layers	   = 40;
		static constexpr uint64_t intermediate_size	   = 13824;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_70B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_70B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 8192;
		static constexpr uint64_t block_count		   = 80;
		static constexpr uint64_t feed_forward_length  = 28672;
		static constexpr uint64_t head_count		   = 64;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 70000000000;
		static constexpr uint64_t kv_cache_layers	   = 80;
		static constexpr uint64_t intermediate_size	   = 28672;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_90B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_90B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 8192;
		static constexpr uint64_t block_count		   = 80;
		static constexpr uint64_t feed_forward_length  = 28672;
		static constexpr uint64_t head_count		   = 64;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 90000000000;
		static constexpr uint64_t kv_cache_layers	   = 80;
		static constexpr uint64_t intermediate_size	   = 28672;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_405B, model_generations::v1_v2> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v1_v2 };
		static constexpr auto model_size{ model_sizes::llama_405B };
		static constexpr uint64_t vocab_size		   = 32000;
		static constexpr uint64_t embedding_dim		   = 16384;
		static constexpr uint64_t block_count		   = 126;
		static constexpr uint64_t feed_forward_length  = 53248;
		static constexpr uint64_t head_count		   = 128;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 405000000000;
		static constexpr uint64_t kv_cache_layers	   = 126;
		static constexpr uint64_t intermediate_size	   = 53248;
		static constexpr uint64_t max_sequence_length  = 2048;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_1B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_1B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 2048;
		static constexpr uint64_t block_count		   = 16;
		static constexpr uint64_t feed_forward_length  = 8192;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 64;
		static constexpr uint64_t rope_dimension_count = 64;
		static constexpr uint64_t total_parameters	   = 1000000000;
		static constexpr uint64_t kv_cache_layers	   = 16;
		static constexpr uint64_t intermediate_size	   = 8192;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_3B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_3B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 3072;
		static constexpr uint64_t block_count		   = 28;
		static constexpr uint64_t feed_forward_length  = 8192;
		static constexpr uint64_t head_count		   = 24;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 3000000000;
		static constexpr uint64_t kv_cache_layers	   = 28;
		static constexpr uint64_t intermediate_size	   = 8192;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_7B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_7B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 4096;
		static constexpr uint64_t block_count		   = 32;
		static constexpr uint64_t feed_forward_length  = 11008;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 32;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 7000000000;
		static constexpr uint64_t kv_cache_layers	   = 32;
		static constexpr uint64_t intermediate_size	   = 11008;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_8B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_8B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 4096;
		static constexpr uint64_t block_count		   = 32;
		static constexpr uint64_t feed_forward_length  = 14336;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 8000000000;
		static constexpr uint64_t kv_cache_layers	   = 32;
		static constexpr uint64_t intermediate_size	   = 14336;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_11B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_11B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 4096;
		static constexpr uint64_t block_count		   = 32;
		static constexpr uint64_t feed_forward_length  = 14336;
		static constexpr uint64_t head_count		   = 32;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 11000000000;
		static constexpr uint64_t kv_cache_layers	   = 32;
		static constexpr uint64_t intermediate_size	   = 14336;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_13B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_13B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 5120;
		static constexpr uint64_t block_count		   = 40;
		static constexpr uint64_t feed_forward_length  = 13824;
		static constexpr uint64_t head_count		   = 40;
		static constexpr uint64_t head_count_kv		   = 40;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 13000000000;
		static constexpr uint64_t kv_cache_layers	   = 40;
		static constexpr uint64_t intermediate_size	   = 13824;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_70B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_70B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 8192;
		static constexpr uint64_t block_count		   = 80;
		static constexpr uint64_t feed_forward_length  = 28672;
		static constexpr uint64_t head_count		   = 64;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 70000000000;
		static constexpr uint64_t kv_cache_layers	   = 80;
		static constexpr uint64_t intermediate_size	   = 28672;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_90B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_90B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 8192;
		static constexpr uint64_t block_count		   = 80;
		static constexpr uint64_t feed_forward_length  = 28672;
		static constexpr uint64_t head_count		   = 64;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 90000000000;
		static constexpr uint64_t kv_cache_layers	   = 80;
		static constexpr uint64_t intermediate_size	   = 28672;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

	template<> struct model_traits<model_arches::llama, model_sizes::llama_405B, model_generations::v3> {
		using op_type_type = op_types;
		static constexpr auto arch{ model_arches::llama };
		static constexpr auto model_generation{ model_generations::v3 };
		static constexpr auto model_size{ model_sizes::llama_405B };
		static constexpr uint64_t vocab_size		   = 128256;
		static constexpr uint64_t embedding_dim		   = 16384;
		static constexpr uint64_t block_count		   = 126;
		static constexpr uint64_t feed_forward_length  = 53248;
		static constexpr uint64_t head_count		   = 128;
		static constexpr uint64_t head_count_kv		   = 8;
		static constexpr uint64_t head_dim			   = 128;
		static constexpr uint64_t rope_dimension_count = 128;
		static constexpr uint64_t total_parameters	   = 405000000000;
		static constexpr uint64_t kv_cache_layers	   = 126;
		static constexpr uint64_t intermediate_size	   = 53248;
		static constexpr uint64_t max_sequence_length  = 8192;
		static constexpr uint64_t n_embd_head_kv	   = embedding_dim / head_count;
		static constexpr uint64_t n_embd_kv_gqa		   = n_embd_head_kv * head_count_kv;
	};

}
