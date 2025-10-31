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

#include <nihilus-incl/common/config.hpp>

namespace nihilus {

	template<uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> struct dimensions {
		static constexpr array<uint64_t, 4> dims{ dim_00, dim_01, dim_02, dim_03 };
	};

	enum class incorrect_runtime_dims {
		incorrect_runtime_dim,
	};

	template<typename value_type> constexpr uint64_t popcount(value_type value) noexcept {
		uint64_t count = 0;
		value &= 0xF;
		while (value != 0) {
			value &= (value - 1);
			count++;
		}
		return count;
	}

	constexpr bool has_at_most_two_bits_set(uint64_t value) noexcept {
		return popcount(value & 0xF) <= 2;
	}

	template<uint64_t... runtime_mask> consteval uint64_t get_runtime_mask_impl() {
		constexpr uint64_t result = ((1ULL << runtime_mask) | ...);
		static_assert(((runtime_mask < 4) && ...), "Sorry, but you can only define a dimension within the first 4 dimensions as runtime mutable!");
		static_assert(has_at_most_two_bits_set(result), "Sorry, but you can only define one or two of the first 4 dimensions as runtime mutable!");
		return result;
	}

	template<bool batched, uint64_t... runtime_mask>
		requires(sizeof...(runtime_mask) != 0)
	consteval uint64_t get_runtime_mask() {
		return ((get_runtime_mask_impl<runtime_mask...>() << static_cast<int32_t>(batched)) | static_cast<uint64_t>(batched)) & 0xF;
	}

	template<bool batched, uint64_t... runtime_mask>
		requires(sizeof...(runtime_mask) == 0)
	consteval uint64_t get_runtime_mask() {
		return ((get_runtime_mask_impl<0>() << static_cast<int32_t>(batched)) | static_cast<uint64_t>(batched)) & 0xF;
	}

	template<uint64_t runtime_mask_new, uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> struct rt_dimensions : public dimensions<dim_00, dim_01, dim_02, dim_03> {
		using base_type = dimensions<dim_00, dim_01, dim_02, dim_03>;
		static_assert(has_at_most_two_bits_set(runtime_mask_new), "Sorry, but you can only define one or two of the first 4 dimensions as runtime mutable!");
		static constexpr uint64_t runtime_mask{ runtime_mask_new & 0xF };

		mutable array<uint64_t, 4> rt_dims{ base_type::dims[0], base_type::dims[1], base_type::dims[2], base_type::dims[3] };

		static constexpr array<uint64_t, 4> get_array() {
			return array<uint64_t, 4>{ base_type::dims[0], base_type::dims[1], base_type::dims[2], base_type::dims[3] };
		}

		NIHILUS_HOST const array<uint64_t, 4>& get_array_rt() const {
			return rt_dims;
		}

		template<typename index_tag> NIHILUS_HOST uint64_t& get_dims(index_tag index) {
			static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
			static_assert(static_assert_printer_val<((runtime_mask & (1ULL << index)) != 0), incorrect_runtime_dims::incorrect_runtime_dim, index>::impl);
			return rt_dims[index];
		}

		template<typename index_tag> NIHILUS_HOST uint64_t get_dims(index_tag index) const {
			static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
			static_assert(static_assert_printer_val<((runtime_mask & (1ULL << index)) != 0), incorrect_runtime_dims::incorrect_runtime_dim, index>::impl);
			return rt_dims[index];
		}
	};

	template<typename value_type>
	concept dim_00_runtime_mutable = (value_type::runtime_mask & 0b0001) != 0;

	template<typename value_type>
	concept dim_01_runtime_mutable = (value_type::runtime_mask & 0b0010) != 0;

	template<typename value_type>
	concept dim_02_runtime_mutable = (value_type::runtime_mask & 0b0100) != 0;

	template<typename value_type>
	concept dim_03_runtime_mutable = (value_type::runtime_mask & 0b1000) != 0;

	template<typename value_type>
	concept runtime_dims_types =
		dim_00_runtime_mutable<value_type> || dim_01_runtime_mutable<value_type> || dim_02_runtime_mutable<value_type> || dim_03_runtime_mutable<value_type>;

	template<typename config_type> struct model_dimensions {
		enum {
			vocab_size				= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::vocab_size,
			block_count				= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::block_count,
			feed_forward_length		= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::feed_forward_length,
			embedding_length		= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::embedding_length,
			attention_head_count	= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::attention_head_count,
			attention_head_count_kv = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::attention_head_count_kv,
			rope_dimension_count	= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::rope_dimension_count,
			context_length			= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::context_length,
			n_embd_kv_gqa			= model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>::n_embd_kv_gqa,
			sequence_length			= config_type::max_sequence_length,
			batch_size				= config_type::batch_size
		};
	};

}
