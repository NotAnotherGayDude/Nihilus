/*
Copyright (c) 2025 RealTimeChris (Chris model_traits_type.)

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
RealTimeChris (Chris model_traits_type.)
2025
*/

#pragma once

#include <nihilus/common/kernel_traits.hpp>
#include <nihilus/common/kernel_type_profile_traits.hpp>
#include <nihilus/common/model_traits.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/array.hpp>
#include <nihilus/common/tuple.hpp>
#include <latch>

namespace nihilus {

	enum class alloc_type : uint8_t {
		no_alloc,
		single_alloc,
		per_block_alloc,
	};

	enum class layer_op_type : uint8_t {
		none,
		global_input,
		global_output,
		per_block,
	};

	template<typename type01, typename type02> struct requires_dequant_or_quant {
		static constexpr bool required{ !std::is_same_v<type01, type02> };
	};

	template<model_config config, auto op_type> struct core_traits;

	template<typename op_type_type> struct core_traits_dynamic {
		op_type_type type{};
		uint64_t depth{};
	};

	template<model_config config, typename op_type_type, op_type_type current_index = static_cast<op_type_type>(0)>
	constexpr uint64_t count_active_ops(uint64_t current_max_depth = 0) {
		if constexpr (static_cast<uint64_t>(current_index) < static_cast<uint64_t>(op_type_type::count)) {
			constexpr uint64_t current_index_new = static_cast<uint64_t>(current_index);
			current_max_depth += core_traits<config, current_index>::krn_type != kernel_type::none ? 1 : 0;
			return count_active_ops<config, op_type_type, static_cast<op_type_type>(current_index_new + 1)>(current_max_depth);
		} else {
			return current_max_depth;
		}
	}

	template<model_config config, typename op_type_type, uint64_t size, op_type_type current_index = static_cast<op_type_type>(0)>
	constexpr auto generate_core_traits_dynamic_array(array<core_traits_dynamic<op_type_type>, size> values, uint64_t current_index_newer = 0) {
		constexpr uint64_t current_index_new = static_cast<uint64_t>(current_index);
		if constexpr (current_index_new < static_cast<uint64_t>(op_type_type::count)) {
			if constexpr (core_traits<config, current_index>::krn_type != kernel_type::none) {
				core_traits_dynamic<op_type_type> return_values{};
				return_values.depth			= core_traits<config, current_index>::depth;
				return_values.type			= core_traits<config, current_index>::type;
				values[current_index_newer] = return_values;
				++current_index_newer;
			}
			return generate_core_traits_dynamic_array<config, op_type_type, size, static_cast<op_type_type>(current_index_new + 1)>(values, current_index_newer);
		} else {
			return values;
		}
	}

	template<model_config config, typename op_type_type> constexpr auto generate_core_traits_dynamic_array() {
		constexpr uint64_t active_op_count{ count_active_ops<config, op_type_type>() };
		array<core_traits_dynamic<op_type_type>, active_op_count> return_values{};
		return generate_core_traits_dynamic_array<config, op_type_type>(return_values);
	}

	template<typename op_type_type, uint64_t depth_count, uint64_t array_size>
	constexpr auto count_ops_per_depth(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		if constexpr (array_size == 0) {
			return 0;
		}
		array<uint64_t, depth_count> return_values{};

		for (uint64_t i = 0; i < array_size; ++i) {
			++return_values[traits_array[i].depth - 1];
		}
		return return_values;
	}

	template<typename op_type_type, uint64_t array_size> constexpr uint64_t count_unique_depths(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		if constexpr (array_size == 0) {
			return 0;
		}
		uint64_t max_depth{};

		for (uint64_t i = 0; i < array_size; ++i) {
			max_depth = traits_array[i].depth > max_depth ? traits_array[i].depth : max_depth;
		}
		return max_depth;
	}

	template<typename op_type_type, size_t depth_count, uint64_t array_size>
	constexpr uint64_t count_max_ops_per_depth(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		if constexpr (array_size == 0) {
			return 0;
		}
		array<uint64_t, depth_count> array{};
		for (uint64_t i = 0; i < array_size; ++i) {
			++array[traits_array[i].depth - 1];
		}
		uint64_t max_depth_count{};

		for (uint64_t i = 0; i < depth_count; ++i) {
			max_depth_count = array[i] > max_depth_count ? array[i] : max_depth_count;
		}
		return max_depth_count;
	}

	template<model_config config, typename op_type_type, uint64_t max_ops_per_depth, uint64_t depth_count, size_t array_size>
	constexpr auto construct_thread_strategy(const nihilus::array<core_traits_dynamic<op_type_type>, array_size>& traits_array) {
		array<array<op_type_type, max_ops_per_depth>, depth_count> result{};
		array<uint64_t, depth_count> counters{};

		for (uint64_t d = 0; d < depth_count; ++d) {
			counters[d] = 0;
			for (uint64_t op = 0; op < max_ops_per_depth; ++op) {
				result[d][op] = op_type_type::count;
			}
		}

		for (uint64_t i = 0; i < traits_array.size(); ++i) {
			uint64_t depth_index = traits_array[i].depth - 1;

			if (depth_index < depth_count && counters[depth_index] < max_ops_per_depth) {
				result[depth_index][counters[depth_index]] = traits_array[i].type;
				++counters[depth_index];
			}
		}

		return result;
	}

	template<model_config config, typename op_type_type> struct thread_strategy {
		static constexpr auto dynamic_core_traits_array{ generate_core_traits_dynamic_array<config, op_type_type>() };
		static constexpr uint64_t active_op_count{ count_active_ops<config, op_type_type>() };
		static constexpr uint64_t unique_depth_count{ count_unique_depths<op_type_type>(dynamic_core_traits_array) };
		static constexpr auto actual_ops_per_depth{ count_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto max_ops_per_depth{ count_max_ops_per_depth<op_type_type, unique_depth_count>(dynamic_core_traits_array) };
		static constexpr auto final_ops{ construct_thread_strategy<config, op_type_type, max_ops_per_depth, unique_depth_count>(dynamic_core_traits_array) };
	};

	template<kernel_type kernel_type01, kernel_type kernel_type02> struct output_transform {};

	template<> struct output_transform<kernel_type::silu, kernel_type::mul_mat> {
		template<typename value_type> NIHILUS_FORCE_INLINE static void impl(value_type*, uint64_t) {};
	};

	template<model_config config, auto op_type> struct depth_tracker {};

	template<typename derived_type, uint64_t... indices> struct core_trait_dims;

	template<typename derived_type, uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new>
	struct core_trait_dims<derived_type, dim00_new, dim01_new, dim02_new, dim03_new> {
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00, dim01, dim02, dim03 } };
		}

		template<size_t index> NIHILUS_FORCE_INLINE uint64_t& operator[](tag<index>) const {
			if constexpr (index == 0) {
				return dim00;
			} else if constexpr (index == 1) {
				return dim01;
			} else if constexpr (index == 2) {
				return dim02;
			} else {
				return dim03;
			}
		}
	};

	template<typename derived_type, uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new>
	struct core_trait_dims<derived_type, dim00_new, dim01_new, dim02_new, dim03_new, 0> {
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_FORCE_INLINE uint64_t& operator[](uint64_t index) {
			return static_cast<derived_type*>(this)->dim00;
		}
	};

	template<typename derived_type, uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new>
	struct core_trait_dims<derived_type, dim00_new, dim01_new, dim02_new, dim03_new, 1> {
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_FORCE_INLINE uint64_t& operator[](uint64_t index) {
			return static_cast<derived_type*>(this)->dim01;
		}
	};

	template<model_config config> using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

	template<model_config config> struct model;

	template<model_config config> struct core_traits<config, llama_op_types::token_embd_weight>
		: core_trait_dims<core_traits<config, llama_op_types::token_embd_weight>, model_traits_type<config>::embedding_dim, model_traits_type<config>::vocab_size, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::token_embd_weight>, model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::token_embd_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::inp_tokens>
		: core_trait_dims<core_traits<config, llama_op_types::inp_tokens>, model_traits_type<config>::max_sequence_length, 1, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::input_token_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::inp_tokens>, model_traits_type::max_sequence_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_tokens };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::inp_pos>
		: core_trait_dims<core_traits<config, llama_op_types::inp_pos>, model_traits_type<config>::max_sequence_length, 1, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::position_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::inp_pos>, model_traits_type::max_sequence_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_pos };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::inp_out_ids>
		: core_trait_dims<core_traits<config, llama_op_types::inp_out_ids>, model_traits_type<config>::max_sequence_length, 1, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::output_token_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::inp_out_ids>, model_traits_type::max_sequence_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_out_ids };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::rope_freqs_weight>
		: core_trait_dims<core_traits<config, llama_op_types::rope_freqs_weight>, model_traits_type<config>::rope_dimension_count / 2, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::rope_freqs_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::rope_freqs_weight>, model_traits_type::rope_dimension_count / 2, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::rope_freqs_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::output_weight>
		: core_trait_dims<core_traits<config, llama_op_types::output_weight>, model_traits_type<config>::embedding_dim, model_traits_type<config>::vocab_size, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::output_weight>, model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::output_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::output_norm_weight>
		: core_trait_dims<core_traits<config, llama_op_types::output_norm_weight>, model_traits_type<config>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::output_norm_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::output_norm_weight>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::output_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_q_weight>
		: core_trait_dims<core_traits<config, llama_op_types::attn_q_weight>, model_traits_type<config>::embedding_dim, model_traits_type<config>::embedding_dim, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::attn_q_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::attn_q_weight>, model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_q_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_k_weight>
		: core_trait_dims<core_traits<config, llama_op_types::attn_k_weight>, model_traits_type<config>::embedding_dim,
			  (model_traits_type<config>::head_dim * model_traits_type<config>::head_count_kv), 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::attn_k_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::attn_k_weight>, model_traits_type::embedding_dim,
			(model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_k_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_v_weight>
		: core_trait_dims<core_traits<config, llama_op_types::attn_v_weight>, model_traits_type<config>::embedding_dim,
			  (model_traits_type<config>::head_dim * model_traits_type<config>::head_count_kv), 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::attn_v_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::attn_v_weight>, model_traits_type::embedding_dim,
			(model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_v_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_output_weight>
		: core_trait_dims<core_traits<config, llama_op_types::attn_output_weight>, model_traits_type<config>::embedding_dim, model_traits_type<config>::embedding_dim, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::attn_output_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::attn_output_weight>, model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_output_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_norm_weight>
		: core_trait_dims<core_traits<config, llama_op_types::attn_norm_weight>, model_traits_type<config>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::attn_norm_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::attn_norm_weight>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_gate_weight>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_gate_weight>, model_traits_type<config>::embedding_dim, model_traits_type<config>::feed_forward_length, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_gate_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_gate_weight>, model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_up_weight>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_up_weight>, model_traits_type<config>::embedding_dim, model_traits_type<config>::feed_forward_length, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_up_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_up_weight>, model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_up_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_down_weight>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_down_weight>, model_traits_type<config>::feed_forward_length, model_traits_type<config>::embedding_dim, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_down_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_down_weight>, model_traits_type::feed_forward_length, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_down_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_norm_weight>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_norm_weight>, model_traits_type<config>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_norm_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::ffn_norm_weight>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::cache_k>
		: core_trait_dims<core_traits<config, llama_op_types::cache_k>, model_traits_type<config>::n_embd_kv_gqa, model_traits_type<config>::embedding_dim, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::cache_k>, model_traits_type::n_embd_kv_gqa, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::cache_k };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::cache_v>
		: core_trait_dims<core_traits<config, llama_op_types::cache_v>, model_traits_type<config>::n_embd_kv_gqa, model_traits_type<config>::embedding_dim, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::cache_v>, model_traits_type::n_embd_kv_gqa, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::cache_v };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kq_mask>
		: core_trait_dims<core_traits<config, llama_op_types::kq_mask>, model_traits_type<config>::max_sequence_length, model_traits_type<config>::max_sequence_length, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kq_mask_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::kq_mask>, model_traits_type::max_sequence_length, model_traits_type::max_sequence_length, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::kq_mask };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::inp_embd>
		: core_trait_dims<core_traits<config, llama_op_types::inp_embd>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::token_embd_weight>;
		using input_type02														 = core_traits<config, llama_op_types::inp_tokens>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::embedding_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::inp_embd>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_input };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::inp_embd };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::norm>
		: core_trait_dims<core_traits<config, llama_op_types::norm>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::inp_embd>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::norm_output_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_norm>
		: core_trait_dims<core_traits<config, llama_op_types::attn_norm>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::norm>;
		using input_type02														 = core_traits<config, llama_op_types::attn_norm_weight>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::norm_output_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::attn_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::attn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::qcur>
		: core_trait_dims<core_traits<config, llama_op_types::qcur>, model_traits_type<config>::head_count * model_traits_type<config>::head_dim,
			  model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::attn_q_weight>;
		using input_type02														 = core_traits<config, llama_op_types::attn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::qcur>, model_traits_type::head_count * model_traits_type::head_dim,
			model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::qcur };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::qcur_reshaped>
		: core_trait_dims<core_traits<config, llama_op_types::qcur_reshaped>, model_traits_type<config>::head_dim, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::qcur>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::qcur_reshaped>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::reshape };
		static constexpr llama_op_types type{ llama_op_types::qcur_reshaped };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::qcur_rope>
		: core_trait_dims<core_traits<config, llama_op_types::qcur_rope>, model_traits_type<config>::head_dim, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::qcur_reshaped>;
		using input_type02														 = core_traits<config, llama_op_types::inp_pos>;
		using input_type03														 = core_traits<config, llama_op_types::rope_freqs_weight>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::qcur_rope>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rope };
		static constexpr llama_op_types type{ llama_op_types::qcur_rope };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kcur>
		: core_trait_dims<core_traits<config, llama_op_types::kcur>, model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim,
			  model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::attn_k_weight>;
		using input_type02														 = core_traits<config, llama_op_types::attn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::key_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kcur>, model_traits_type::n_embd_kv_gqa, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kcur };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kcur_reshaped>
		: core_trait_dims<core_traits<config, llama_op_types::kcur_reshaped>, model_traits_type<config>::head_dim, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kcur>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::key_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kcur_reshaped>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::reshape };
		static constexpr llama_op_types type{ llama_op_types::kcur_reshaped };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kcur_rope>
		: core_trait_dims<core_traits<config, llama_op_types::kcur_rope>, model_traits_type<config>::head_dim, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kcur_reshaped>;
		using input_type02														 = core_traits<config, llama_op_types::inp_pos>;
		using input_type03														 = core_traits<config, llama_op_types::rope_freqs_weight>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::key_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kcur_rope>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ std::max(std::max(input_type01::depth, input_type02::depth), input_type03::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rope };
		static constexpr llama_op_types type{ llama_op_types::kcur_rope };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::vcur>
		: core_trait_dims<core_traits<config, llama_op_types::vcur>, model_traits_type<config>::head_dim * model_traits_type<config>::head_count_kv,
			  model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::attn_v_weight>;
		using input_type02														 = core_traits<config, llama_op_types::attn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::vcur>, model_traits_type::head_dim * model_traits_type::head_count_kv,
			model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::vcur };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::k_cache_view>
		: core_trait_dims<core_traits<config, llama_op_types::k_cache_view>, 2, model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::cache_k>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::k_cache_view>, 1, model_traits_type::head_count_kv * model_traits_type::head_dim,
			model_traits_type::max_sequence_length, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::k_cache_view };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::k_cache_view_copy>
		: core_trait_dims<core_traits<config, llama_op_types::k_cache_view_copy>, model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim,
			  model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kcur_rope>;
		using output_type														 = typename core_traits<config, llama_op_types::k_cache_view>::output_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::k_cache_view_copy>, model_traits_type::n_embd_kv_gqa, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::copy };
		static constexpr llama_op_types type{ llama_op_types::k_cache_view_copy };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::vcur_transposed>
		: core_trait_dims<core_traits<config, llama_op_types::vcur_transposed>, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_dim * model_traits_type<config>::head_count_kv, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::vcur>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::vcur_transposed>, model_traits_type::max_sequence_length,
			model_traits_type::head_dim * model_traits_type::head_count_kv, 1, 1, 0>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::transpose };
		static constexpr llama_op_types type{ llama_op_types::vcur_transposed };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::v_cache_view>
		: core_trait_dims<core_traits<config, llama_op_types::v_cache_view>, 2, model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::cache_v>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::v_cache_view>, 1, model_traits_type::head_count_kv * model_traits_type::head_dim,
			model_traits_type::max_sequence_length, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::v_cache_view };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::v_cache_view_copy>
		: core_trait_dims<core_traits<config, llama_op_types::v_cache_view_copy>, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::vcur_transposed>;
		using output_type														 = typename core_traits<config, llama_op_types::v_cache_view>::output_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::v_cache_view_copy>, model_traits_type::max_sequence_length, model_traits_type::n_embd_kv_gqa, 1, 1, 0>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::copy };
		static constexpr llama_op_types type{ llama_op_types::v_cache_view_copy };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::v> : core_trait_dims<core_traits<config, llama_op_types::v>, model_traits_type<config>::head_dim,
																					  model_traits_type<config>::block_count, model_traits_type<config>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::cache_v>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::scale_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::v>, model_traits_type::head_dim, model_traits_type::block_count, model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::v };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::k> : core_trait_dims<core_traits<config, llama_op_types::k>, model_traits_type<config>::head_dim,
																					  model_traits_type<config>::block_count, model_traits_type<config>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::cache_k>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::k>, model_traits_type::head_dim, model_traits_type::block_count, model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::k };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::q> : core_trait_dims<core_traits<config, llama_op_types::q>, model_traits_type<config>::head_dim,
																					  model_traits_type<config>::max_sequence_length, model_traits_type<config>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::qcur_rope>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::query_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::q>, model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::permute };
		static constexpr llama_op_types type{ llama_op_types::q };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kq>
		: core_trait_dims<core_traits<config, llama_op_types::kq>, model_traits_type<config>::max_sequence_length, 1, model_traits_type<config>::head_count, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::k>;
		using input_type02														 = core_traits<config, llama_op_types::q>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::attention_score_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kq>, model_traits_type::max_sequence_length, 1, model_traits_type::head_count, 1, 0>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kq };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kq_soft_max>
		: core_trait_dims<core_traits<config, llama_op_types::kq_soft_max>, model_traits_type<config>::max_sequence_length, 1, model_traits_type<config>::head_count, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kq>;
		using input_type02														 = core_traits<config, llama_op_types::kq_mask>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::softmax_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::kq_soft_max>, model_traits_type::max_sequence_length, 1, model_traits_type::head_count, 1, 0>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::softmax };
		static constexpr llama_op_types type{ llama_op_types::kq_soft_max };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kqv>
		: core_trait_dims<core_traits<config, llama_op_types::kqv>, model_traits_type<config>::head_dim, 1, model_traits_type<config>::head_count, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::v>;
		using input_type02														 = core_traits<config, llama_op_types::kq_soft_max>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kqv>, model_traits_type::head_dim, 1, model_traits_type::head_count, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kqv };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kqv_merged>
		: core_trait_dims<core_traits<config, llama_op_types::kqv_merged>, model_traits_type<config>::head_dim, model_traits_type<config>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kqv>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kqv_merged>, model_traits_type::head_dim, model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::permute };
		static constexpr llama_op_types type{ llama_op_types::kqv_merged };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kqv_merged_cont>
		: core_trait_dims<core_traits<config, llama_op_types::kqv_merged_cont>, model_traits_type<config>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kqv_merged>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::kqv_merged_cont>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::cont };
		static constexpr llama_op_types type{ llama_op_types::kqv_merged_cont };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::kqv_out>
		: core_trait_dims<core_traits<config, llama_op_types::kqv_out>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::attn_output_weight>;
		using input_type02														 = core_traits<config, llama_op_types::kqv_merged_cont>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::hidden_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::kqv_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kqv_out };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_inp>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_inp>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kqv_out>;
		using input_type02														 = core_traits<config, llama_op_types::l_out>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_inp>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add };
		static constexpr llama_op_types type{ llama_op_types::ffn_inp };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::norm_out>
		: core_trait_dims<core_traits<config, llama_op_types::norm_out>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_inp>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::norm_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::norm_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_norm>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_norm>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::norm_out>;
		using input_type02														 = core_traits<config, llama_op_types::ffn_norm_weight>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::residual_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_gate>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_gate>, model_traits_type<config>::feed_forward_length, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_gate_weight>;
		using input_type02														 = core_traits<config, llama_op_types::ffn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_gate>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_silu>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_silu>, model_traits_type<config>::feed_forward_length, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_gate>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_silu>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::silu };
		static constexpr llama_op_types type{ llama_op_types::ffn_silu };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_up>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_up>, model_traits_type<config>::feed_forward_length, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_up_weight>;
		using input_type02														 = core_traits<config, llama_op_types::ffn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_up>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_up };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_gate_par>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_gate_par>, model_traits_type<config>::feed_forward_length, model_traits_type<config>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_silu>;
		using input_type02														 = core_traits<config, llama_op_types::ffn_up>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::ffn_intermediate_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_gate_par>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate_par };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::ffn_out>
		: core_trait_dims<core_traits<config, llama_op_types::ffn_out>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits(uint64_t thread_count) noexcept : sync_flag_start{ thread_count }, sync_flag_end{ thread_count } {};
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_down_weight>;
		using input_type02														 = core_traits<config, llama_op_types::ffn_gate_par>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::hidden_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::ffn_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_out };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::l_out>
		: core_trait_dims<core_traits<config, llama_op_types::l_out>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::ffn_out>;
		using input_type02														 = core_traits<config, llama_op_types::ffn_inp>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::l_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add };
		static constexpr llama_op_types type{ llama_op_types::l_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::attn_residual>
		: core_trait_dims<core_traits<config, llama_op_types::attn_residual>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::kqv_out>;
		using input_type02														 = core_traits<config, llama_op_types::inp_out_ids>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::attn_residual>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::attn_residual };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::prev_residual>
		: core_trait_dims<core_traits<config, llama_op_types::prev_residual>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::l_out>;
		using input_type02														 = core_traits<config, llama_op_types::inp_out_ids>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::prev_residual>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::prev_residual };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::final_norm>
		: core_trait_dims<core_traits<config, llama_op_types::final_norm>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::l_out>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::norm_output_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::final_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::final_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::result_norm>
		: core_trait_dims<core_traits<config, llama_op_types::result_norm>, model_traits_type<config>::embedding_dim, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::final_norm>;
		using input_type02														 = core_traits<config, llama_op_types::output_norm_weight>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::norm_output_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::result_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::result_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config> struct core_traits<config, llama_op_types::result_output>
		: core_trait_dims<core_traits<config, llama_op_types::result_output>, model_traits_type<config>::vocab_size, model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using input_type01														 = core_traits<config, llama_op_types::output_weight>;
		using input_type02														 = core_traits<config, llama_op_types::result_norm>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::logit_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::result_output>, model_traits_type::vocab_size, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr layer_op_type layer_type{ layer_op_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::result_output };
		array<op_latch, model_traits_type::block_count> sync_flag_start;
		array<op_latch, model_traits_type::block_count> sync_flag_end;
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	/*

	template<typename op_type_type, uint64_t depth_count, uint64_t maxIndex, uint64_t currentIndex = 0>
	constexpr auto collectTupleRefsImpl(array<array<tuple_reference<op_type_type>, maxIndex>, depth_count>& tupleRefs) {
		if constexpr (currentIndex < maxIndex) {
			tupleRefs[currentIndex].oldIndex = currentIndex;
			return collectTupleRefsImpl<maxIndex, currentIndex + 1>(tuple, tupleRefs);
		}
		return tupleRefs;
	}

	template<model_config config, typename op_type_type> constexpr auto collectTupleRefs() {
		constexpr auto depth_count = count_depths<config, op_type_type>;
		array<tuple_reference, depth_count> tupleRefs{};
		return collectTupleRefsImpl<tupleSize>(tuple, tupleRefs);
	}

	template<typename value_type> inline constexpr auto tupleRefs{ collectTupleRefs(jsonifier::concepts::coreV<value_type>) };
	template<typename value_type> inline constexpr auto sortedTupleReferencesByLength{ sortTupleRefsByLength(tupleRefs<value_type>) };
	template<typename value_type> inline constexpr auto tupleReferencesByLength{ consolidateTupleRefs(sortedTupleReferencesByLength<value_type>) };

	template<typename value_type, uint64_t... indices> constexpr auto createNewTupleImpl(std::index_sequence<indices...>) noexcept {
		return makeTuple(get<sortedTupleReferencesByLength<value_type>[indices].oldIndex>(jsonifier::concepts::coreV<value_type>)...);
	}

	template<typename value_type> constexpr auto createNewTuple() noexcept {
		constexpr auto& tupleRefs = sortedTupleReferencesByLength<value_type>;
		return createNewTupleImpl<value_type>(std::make_index_sequence<tupleRefs.size()>{});
	}
	*/
	template<model_config config, auto krn_type, uint64_t index> struct get_adjacent_value;

	template<model_config config, auto krn_type, uint64_t index> struct get_adjacent_value {
		using derived_type		   = core_traits<config, krn_type>;
		using model_traits_type	   = model_traits<config.arch, config.model_size, config.model_generation>;
		using derived_derived_type = model<config>;
		NIHILUS_FORCE_INLINE static auto& impl(derived_type& core) {
			if constexpr (index == 0) {
				using input_type01 = typename derived_type::input_type01;
				return *static_cast<input_type01*>(static_cast<derived_type*>(&core));
			} else if constexpr (index == 1) {
				using input_type02 = typename derived_type::input_type02;
				return *static_cast<input_type02*>(static_cast<derived_type*>(&core));
			} else if constexpr (index == 2) {
				using input_type03 = typename derived_type::input_type03;
				return *static_cast<input_type03*>(static_cast<derived_type*>(&core));
			}
		}
	};

	template<typename... bases> struct core_bases : bases... {
		NIHILUS_FORCE_INLINE core_bases() noexcept					  = default;
		NIHILUS_FORCE_INLINE core_bases& operator=(core_bases&&)	  = delete;
		NIHILUS_FORCE_INLINE core_bases(core_bases&&)				  = delete;
		NIHILUS_FORCE_INLINE core_bases& operator=(const core_bases&) = delete;
		NIHILUS_FORCE_INLINE core_bases(const core_bases&)			  = delete;

		template<template<typename> typename mixin_type, typename op_entity_type, typename... arg_types> NIHILUS_FORCE_INLINE void impl_internal(arg_types&&... args) {
			return mixin_type<op_entity_type>::impl(*static_cast<op_entity_type*>(this), std::forward<arg_types>(args)...);
		}

		template<template<typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE void impl(arg_types&&... args) {
			(impl_internal<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<model_config, typename> typename mixin_type, model_config config, typename op_entity_type>
		NIHILUS_FORCE_INLINE void impl_thread_internal(size_t thread_index, size_t thread_count) {
			return static_cast<op_entity_type*>(static_cast<mixin_type<config, op_entity_type>*>(this))->impl_thread(thread_index, thread_count);
		}

		template<template<model_config, typename> typename mixin_type, model_config config> NIHILUS_FORCE_INLINE void impl_thread(size_t thread_index, size_t thread_count) {
			(impl_thread_internal<mixin_type, config, bases>(thread_index, thread_count), ...);
		}

		template<template<typename> typename mixin_type, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE static constexpr void impl_internal_constexpr(arg_types&&... args) {
			return mixin_type<op_entity_type>::impl_constexpr(std::forward<arg_types>(args)...);
		}

		template<template<typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE static constexpr void impl_constexpr(arg_types&&... args) {
			(impl_internal_constexpr<mixin_type, bases>(args...), ...);
		}
	};

	template<model_config config, typename index_sequence> struct get_core_bases_base;

	template<model_config config> using op_type_type_t = typename model_traits<config.arch, config.model_size, config.model_generation>::op_type_type;

	template<model_config config, uint64_t... index> struct get_core_bases_base<config, std::index_sequence<index...>> {
		using type = core_bases<core_traits<config, static_cast<op_type_type_t<config>>(index)>...>;
	};

	template<model_config config> using get_core_bases_config_base_t =
		typename get_core_bases_base<config, std::make_index_sequence<static_cast<uint64_t>(op_type_type_t<config>::count)>>::type;

	template<typename derived_type, typename... bases> struct thread_core_bases : bases... {
		NIHILUS_FORCE_INLINE thread_core_bases() noexcept							= default;
		NIHILUS_FORCE_INLINE thread_core_bases& operator=(thread_core_bases&&)		= delete;
		NIHILUS_FORCE_INLINE thread_core_bases(thread_core_bases&&)					= delete;
		NIHILUS_FORCE_INLINE thread_core_bases& operator=(const thread_core_bases&) = delete;
		NIHILUS_FORCE_INLINE thread_core_bases(const thread_core_bases&)			= delete;

		template<template<model_config, typename> typename mixin_type, model_config config, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE void impl_one_internal(arg_types&&... args) {
			return static_cast<mixin_type<config, op_entity_type>*>(static_cast<op_entity_type*>(this))->impl(std::forward<arg_types>(args)...);
		}

		template<template<model_config, typename> typename mixin_type, model_config config, typename... arg_types> NIHILUS_FORCE_INLINE void impl_one(arg_types&&... args) {
			(impl_one_internal<mixin_type, config, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<model_config, typename> typename mixin_type, model_config config, typename op_entity_type>
		NIHILUS_FORCE_INLINE void impl_thread_internal(size_t thread_index, size_t thread_count) {
			static_cast<mixin_type<config, op_entity_type>*>(static_cast<op_entity_type*>(this))->impl_thread(thread_index, thread_count);
			//static_cast<op_entity_type*>(static_cast<model_type*>(static_cast<derived_type*>(this)))->impl_thread(thread_index, thread_count);
			//return static_cast<op_entity_type*>(static_cast<mixin_type<config, op_entity_type>*>(this))->impl_thread(thread_index, thread_count);
		}

		template<template<model_config, typename> typename mixin_type, model_config config> NIHILUS_FORCE_INLINE void impl_thread(size_t thread_index, size_t thread_count) {
			(impl_thread_internal<mixin_type, config, bases>(thread_index, thread_count), ...);
		}
	};

	template<typename derived_type, typename... bases> struct thread_depth_bases : bases... {
		NIHILUS_FORCE_INLINE thread_depth_bases() noexcept							  = default;
		NIHILUS_FORCE_INLINE thread_depth_bases& operator=(thread_depth_bases&&)	  = delete;
		NIHILUS_FORCE_INLINE thread_depth_bases(thread_depth_bases&&)				  = delete;
		NIHILUS_FORCE_INLINE thread_depth_bases& operator=(const thread_depth_bases&) = delete;
		NIHILUS_FORCE_INLINE thread_depth_bases(const thread_depth_bases&)			  = delete;

		template<template<model_config, typename> typename mixin_type, model_config config, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE void impl_internal(arg_types&&... args) {
			static_cast<derived_type*>(this)->template impl_one<mixin_type, config>(args...);
			//return mixin_type<config, op_entity_type>::impl(std::forward<arg_types>(args)...);
		}

		template<template<model_config, typename> typename mixin_type, model_config config, typename... arg_types> NIHILUS_FORCE_INLINE void impl(arg_types&&... args) {
			(impl_internal<mixin_type, config, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<model_config, typename> typename mixin_type, model_config config, typename op_entity_type>
		NIHILUS_FORCE_INLINE void impl_thread_internal(size_t thread_index, size_t thread_count) {
			using model_type = typename derived_type::derived_type;
			static_cast<op_entity_type*>(static_cast<model_type*>(static_cast<derived_type*>(this)))->template impl_thread<mixin_type, config>(thread_index, thread_count);
		}

		template<template<model_config, typename> typename mixin_type, model_config config> NIHILUS_FORCE_INLINE void impl_thread(size_t thread_index, size_t thread_count) {
			(impl_thread_internal<mixin_type, config, bases>(thread_index, thread_count), ...);
		}
	};

	template<model_config config, typename derived_type, typename thread_strategy_type, size_t current_depth, typename index_sequence>
	struct get_depth_level_thread_core_bases_base;

	template<model_config config, typename derived_type, typename thread_strategy_type, size_t current_depth, uint64_t... index>
	struct get_depth_level_thread_core_bases_base<config, derived_type, thread_strategy_type, current_depth, std::index_sequence<index...>> {
		using type = thread_core_bases<derived_type, core_traits<config, thread_strategy_type::final_ops[current_depth][index]>...>;
	};

	template<model_config config, typename derived_type, typename thread_strategy_type, size_t current_depth> using get_depth_level_thread_core_bases_config_base_t =
		typename get_depth_level_thread_core_bases_base<config, derived_type, thread_strategy_type, current_depth,
			std::make_index_sequence<thread_strategy_type::actual_ops_per_depth[current_depth]>>::type;

	template<model_config config, typename derived_type, typename thread_strategy_type, typename index_sequence> struct get_depth_level_thread_strategy_thread_core_bases_base;

	template<model_config config, typename derived_type, typename thread_strategy_type, uint64_t... index>
	struct get_depth_level_thread_strategy_thread_core_bases_base<config, derived_type, thread_strategy_type, std::index_sequence<index...>> {
		using type = thread_depth_bases<derived_type, get_depth_level_thread_core_bases_config_base_t<config, derived_type, thread_strategy_type, index>...>;
	};

	template<model_config config, typename derived_type, typename thread_strategy_type> using get_depth_level_thread_strategy_thread_core_bases_config_base_t =
		typename get_depth_level_thread_strategy_thread_core_bases_base<config, derived_type, thread_strategy_type,
			std::make_index_sequence<thread_strategy_type::unique_depth_count>>::type;
}
