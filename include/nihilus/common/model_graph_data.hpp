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

#include <nihilus/common/memory_buffer.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/tokenizer.hpp>

namespace nihilus {

	struct core_base_creation_data {
		array<uint64_t, 4> dimensions{ { 1, 1, 1, 1 } };
		uint32_t n_dimensions{};
		mutable void* data{};
		std::string name{};
		uint64_t offset{};
		data_types type{};

		NIHILUS_FORCE_INLINE uint64_t core_total_dims() const {
			return dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3];
		}

		NIHILUS_FORCE_INLINE uint64_t core_total_byte_size() const {
			uint64_t total_elements = core_total_dims();
			uint64_t block_size		= core_block_size();
			uint64_t type_size		= core_type_size();
			uint64_t num_blocks		= (total_elements + block_size - 1) / block_size;
			return num_blocks * type_size;
		}

		NIHILUS_INLINE uint64_t core_block_size() const {
			return get_type_traits(type).block_size;
		}

		NIHILUS_INLINE uint64_t core_type_size() const {
			return get_type_traits(type).type_size;
		}

		NIHILUS_INLINE uint64_t core_row_size(int64_t dims_new) const {
			return core_type_size() * dims_new / core_block_size();
		}
	};

	template<model_arches arch> struct construction_parameters;

	template<> struct construction_parameters<model_arches::llama> {
		uint64_t rope_dimension_count{};
		uint64_t feed_forward_length{};
		uint64_t embedding_length{};
		double f_attention_scale{};
		double rms_norm_epsilon{};
		double rope_attn_factor{};
		uint64_t context_length{};
		uint64_t n_expert_used{};
		uint64_t head_count_kv{};
		double rope_freq_scale{};
		double rope_ext_factor{};
		double rope_freq_base{};
		double rope_beta_fast{};
		double rope_beta_slow{};
		uint64_t block_count{};
		uint64_t head_count{};
		uint64_t vocab_size{};
		uint64_t rope_type{};
		uint64_t n_expert{};
	};

	template<model_config config> struct model_graph_data {
		NIHILUS_INLINE model_graph_data()									= default;
		NIHILUS_INLINE model_graph_data& operator=(model_graph_data&&)		= default;
		NIHILUS_INLINE model_graph_data(model_graph_data&&)					= default;
		NIHILUS_INLINE model_graph_data& operator=(const model_graph_data&) = delete;
		NIHILUS_INLINE model_graph_data(const model_graph_data&)			= delete;
		std::unordered_map<op_types, core_base_creation_data> cores{};
		construction_parameters<config.arch> cparams{};
	};

}