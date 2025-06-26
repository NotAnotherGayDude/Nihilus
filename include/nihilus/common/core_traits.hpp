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

	enum class thread_strategy_type : uint8_t {
		none,
		weight_mapping,
		thread_prep,
		global_input,
		global_output,
		per_block,
	};

	template<typename type01, typename type02> struct requires_dequant_or_quant {
		static constexpr bool required{ !std::is_same_v<type01, type02> };
	};

	template<model_config config_new, auto op_type> struct core_traits;

	template<kernel_type kernel_type01, kernel_type kernel_type02> struct output_transform {};

	template<> struct output_transform<kernel_type::silu, kernel_type::mul_mat> {
		template<typename value_type> NIHILUS_FORCE_INLINE static void impl(value_type*, uint64_t) {};
	};

	template<model_config config_new, auto op_type> struct depth_tracker {};

	//template<kernel_type type, typename input_type01, typename input_type02, typename input_type03> struct dimension_transformer<type, input_type01, input_type02, input_type03> {};

	template<model_config config_new> using model_traits_type = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;

	template<model_config config_new> struct model;

	template<model_config config_new> struct core_traits<config_new, llama_op_types::token_embd_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::token_embd_weight>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::vocab_size, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::token_embd_weight>, model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::token_embd_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::inp_tokens>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::inp_tokens>, model_traits_type<config_new>::max_sequence_length, 1, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::input_token_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::inp_tokens>, model_traits_type::max_sequence_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_tokens };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::inp_pos>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::inp_pos>, model_traits_type<config_new>::max_sequence_length, 1, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::position_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::inp_pos>, model_traits_type::max_sequence_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_pos };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::inp_out_ids>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::inp_out_ids>, model_traits_type<config_new>::max_sequence_length, 1, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::output_token_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::inp_out_ids>, model_traits_type::max_sequence_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::inp_out_ids };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::rope_freqs_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::rope_freqs_weight>, model_traits_type<config_new>::rope_dimension_count / 2, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::rope_freqs_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::rope_freqs_weight>, model_traits_type::rope_dimension_count / 2, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::rope_freqs_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::output_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::output_weight>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::vocab_size, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::output_weight>, model_traits_type::embedding_dim, model_traits_type::vocab_size, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::output_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::output_norm_weight>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::output_norm_weight>, model_traits_type<config_new>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::output_norm_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::output_norm_weight>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::output_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::attn_q_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::attn_q_weight>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::embedding_dim, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_q_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::attn_q_weight>, model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_q_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::attn_k_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::attn_k_weight>, model_traits_type<config_new>::embedding_dim,
			  (model_traits_type<config_new>::head_dim * model_traits_type<config_new>::head_count_kv), 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_k_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::attn_k_weight>, model_traits_type::embedding_dim,
			(model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_k_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::attn_v_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::attn_v_weight>, model_traits_type<config_new>::embedding_dim,
			  (model_traits_type<config_new>::head_dim * model_traits_type<config_new>::head_count_kv), 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_v_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::attn_v_weight>, model_traits_type::embedding_dim,
			(model_traits_type::head_dim * model_traits_type::head_count_kv), 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_v_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::attn_output_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::attn_output_weight>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::embedding_dim,
			  1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_output_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::attn_output_weight>, model_traits_type::embedding_dim, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_output_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::attn_norm_weight>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::attn_norm_weight>, model_traits_type<config_new>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_norm_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::attn_norm_weight>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::attn_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_gate_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_gate_weight>, model_traits_type<config_new>::embedding_dim,
			  model_traits_type<config_new>::feed_forward_length, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_gate_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_gate_weight>, model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_up_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_up_weight>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::feed_forward_length,
			  1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_up_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_up_weight>, model_traits_type::embedding_dim, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_up_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_down_weight>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_down_weight>, model_traits_type<config_new>::feed_forward_length,
			  model_traits_type<config_new>::embedding_dim, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_down_weight_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_down_weight>, model_traits_type::feed_forward_length, model_traits_type::embedding_dim, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_down_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_norm_weight>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::ffn_norm_weight>, model_traits_type<config_new>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_norm_weight_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::ffn_norm_weight>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::ffn_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::cache_k>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::cache_k>, model_traits_type<config_new>::n_embd_kv_gqa * model_traits_type<config_new>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::cache_k>, model_traits_type::n_embd_kv_gqa * model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::cache_k };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::cache_v>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::cache_v>, model_traits_type<config_new>::n_embd_kv_gqa * model_traits_type<config_new>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::cache_v>, model_traits_type::n_embd_kv_gqa * model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::per_block_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::cache_v };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kq_mask>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kq_mask>, model_traits_type<config_new>::max_sequence_length, model_traits_type<config_new>::max_sequence_length,
			  1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::kq_mask_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::kq_mask>, model_traits_type::max_sequence_length, model_traits_type::max_sequence_length, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_type krn_type{ kernel_type::none };
		static constexpr llama_op_types type{ llama_op_types::kq_mask };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::inp_embd>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::inp_embd>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::token_embd_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::inp_tokens>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::embedding_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::inp_embd>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_input };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::inp_embd };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::norm_attn_norm>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::norm_attn_norm>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1,
			  1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::inp_embd>;
		using input_type02														 = core_traits<config_new, llama_op_types::attn_norm_weight>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::norm_output_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::norm_attn_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm_mul };
		static constexpr llama_op_types type{ llama_op_types::norm_attn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::qcur>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::qcur>, model_traits_type<config_new>::head_count * model_traits_type<config_new>::head_dim,
			  model_traits_type<config_new>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::attn_q_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::norm_attn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::qcur>, model_traits_type::head_count * model_traits_type::head_dim,
			model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::qcur };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::qcur_reshaped>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::qcur_reshaped>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::qcur>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::qcur_reshaped>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::reshape };
		static constexpr llama_op_types type{ llama_op_types::qcur_reshaped };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::qcur_rope>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::qcur_rope>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::qcur_reshaped>;
		using input_type02														 = core_traits<config_new, llama_op_types::inp_pos>;
		using input_type03														 = core_traits<config_new, llama_op_types::rope_freqs_weight>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::qcur_rope>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rope };
		static constexpr llama_op_types type{ llama_op_types::qcur_rope };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kcur>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kcur>, model_traits_type<config_new>::head_count_kv * model_traits_type<config_new>::head_dim,
			  model_traits_type<config_new>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::attn_k_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::norm_attn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::key_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::kcur>, model_traits_type::n_embd_kv_gqa, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kcur };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kcur_reshaped>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kcur_reshaped>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kcur>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::key_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::kcur_reshaped>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::reshape };
		static constexpr llama_op_types type{ llama_op_types::kcur_reshaped };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kcur_rope>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kcur_rope>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kcur_reshaped>;
		using input_type02														 = core_traits<config_new, llama_op_types::inp_pos>;
		using input_type03														 = core_traits<config_new, llama_op_types::rope_freqs_weight>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::key_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::kcur_rope>, model_traits_type::head_dim, model_traits_type::max_sequence_length,
			model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ std::max(std::max(input_type01::depth, input_type02::depth), input_type03::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rope };
		static constexpr llama_op_types type{ llama_op_types::kcur_rope };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::vcur>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::vcur>, model_traits_type<config_new>::head_dim * model_traits_type<config_new>::head_count_kv,
			  model_traits_type<config_new>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::attn_v_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::norm_attn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::vcur>, model_traits_type::head_dim * model_traits_type::head_count_kv,
			model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::vcur };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::k_cache_view>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::k_cache_view>, 2, model_traits_type<config_new>::head_count_kv * model_traits_type<config_new>::head_dim, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::cache_k>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::k_cache_view>, 1, model_traits_type::head_count_kv * model_traits_type::head_dim,
			model_traits_type::max_sequence_length, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::k_cache_view };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::k_cache_view_copy>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::k_cache_view_copy>, model_traits_type<config_new>::head_count_kv * model_traits_type<config_new>::head_dim,
			  model_traits_type<config_new>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kcur_rope>;
		using output_type														 = typename core_traits<config_new, llama_op_types::k_cache_view>::output_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::k_cache_view_copy>, model_traits_type::n_embd_kv_gqa, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::copy };
		static constexpr llama_op_types type{ llama_op_types::k_cache_view_copy };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::vcur_transposed>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::vcur_transposed>, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_dim * model_traits_type<config_new>::head_count_kv, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::vcur>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::vcur_transposed>, model_traits_type::max_sequence_length,
			model_traits_type::head_dim * model_traits_type::head_count_kv, 1, 1, 0>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::transpose };
		static constexpr llama_op_types type{ llama_op_types::vcur_transposed };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::v_cache_view>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::v_cache_view>, 2, model_traits_type<config_new>::head_count_kv * model_traits_type<config_new>::head_dim, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::cache_v>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::v_cache_view>, 1, model_traits_type::head_count_kv * model_traits_type::head_dim,
			model_traits_type::max_sequence_length, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::v_cache_view };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::v_cache_view_copy>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::v_cache_view_copy>, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_count_kv * model_traits_type<config_new>::head_dim, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::vcur_transposed>;
		using output_type														 = typename core_traits<config_new, llama_op_types::v_cache_view>::output_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::v_cache_view_copy>, model_traits_type::max_sequence_length, model_traits_type::n_embd_kv_gqa, 1, 1, 0>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::copy };
		static constexpr llama_op_types type{ llama_op_types::v_cache_view_copy };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::v>
		: core_trait_dims<core_traits<config_new, llama_op_types::v>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::block_count,
			  model_traits_type<config_new>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::cache_v>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::scale_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::v>, model_traits_type::max_sequence_length, model_traits_type::head_dim,
			model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::v };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::k>
		: core_trait_dims<core_traits<config_new, llama_op_types::k>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::block_count,
			  model_traits_type<config_new>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::cache_k>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::k>, model_traits_type::head_dim, model_traits_type::block_count, model_traits_type::head_count_kv, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::view };
		static constexpr llama_op_types type{ llama_op_types::k };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::q>
		: core_trait_dims<core_traits<config_new, llama_op_types::q>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::max_sequence_length,
			  model_traits_type<config_new>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::qcur_rope>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::q>, model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::permute };
		static constexpr llama_op_types type{ llama_op_types::q };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kq>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kq>, model_traits_type<config_new>::max_sequence_length, 1, model_traits_type<config_new>::head_count, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::k>;
		using input_type02														 = core_traits<config_new, llama_op_types::q>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::attention_score_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::kq>, model_traits_type::max_sequence_length, 1, model_traits_type::head_count, 1, 0>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kq };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kq_soft_max>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kq_soft_max>, model_traits_type<config_new>::head_count, 1, model_traits_type<config_new>::head_count, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept {};
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kq>;
		using input_type02														 = core_traits<config_new, llama_op_types::kq_mask>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::softmax_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::kq_soft_max>, model_traits_type::head_count, model_traits_type::head_dim, model_traits_type::head_count, 1, 0>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::softmax };
		static constexpr llama_op_types type{ llama_op_types::kq_soft_max };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kqv>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kqv>, model_traits_type<config_new>::head_dim, 1, model_traits_type<config_new>::head_count, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::v>;
		using input_type02														 = core_traits<config_new, llama_op_types::kq_soft_max>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::kqv>, model_traits_type::head_dim, 1, model_traits_type::head_count, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kqv };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kqv_merged>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kqv_merged>, model_traits_type<config_new>::head_dim, model_traits_type<config_new>::head_count, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kqv>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::kqv_merged>, model_traits_type::head_dim, model_traits_type::head_count, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::permute };
		static constexpr llama_op_types type{ llama_op_types::kqv_merged };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kqv_merged_cont>
		: public config_holder<config_new>, core_trait_dims<core_traits<config_new, llama_op_types::kqv_merged_cont>, model_traits_type<config_new>::embedding_dim, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kqv_merged>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type = core_trait_dims<core_traits<config_new, llama_op_types::kqv_merged_cont>, model_traits_type::embedding_dim, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::cont };
		static constexpr llama_op_types type{ llama_op_types::kqv_merged_cont };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::kqv_out>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::kqv_out>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::attn_output_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::kqv_merged_cont>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::hidden_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::kqv_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::kqv_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_inp_norm_out_ffn_norm>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_inp_norm_out_ffn_norm>, model_traits_type<config_new>::embedding_dim,
			  model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kqv_out>;
		using input_type02														 = core_traits<config_new, llama_op_types::l_out>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_inp_norm_out_ffn_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add_rms_norm_mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_inp_norm_out_ffn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_inp>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_inp>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kqv_out>;
		using input_type02														 = core_traits<config_new, llama_op_types::l_out>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_inp>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add };
		static constexpr llama_op_types type{ llama_op_types::ffn_inp };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::norm_out>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::norm_out>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_inp>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::norm_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::norm_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_norm>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_norm>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::norm_out>;
		using input_type02														 = core_traits<config_new, llama_op_types::ffn_norm_weight>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_gate>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_gate>, model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::max_sequence_length,
			  1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_gate_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::ffn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_gate>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_silu>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_silu>, model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::max_sequence_length,
			  1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_gate>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_silu>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::silu };
		static constexpr llama_op_types type{ llama_op_types::ffn_silu };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_up>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_up>, model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::max_sequence_length,
			  1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_up_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::ffn_norm>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_up>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_up };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_gate_par>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_gate_par>, model_traits_type<config_new>::feed_forward_length,
			  model_traits_type<config_new>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_silu>;
		using input_type02														 = core_traits<config_new, llama_op_types::ffn_up>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_gate_par>, model_traits_type::feed_forward_length, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::ffn_gate_par };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::ffn_out>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::ffn_out>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_down_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::ffn_gate_par>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::hidden_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::ffn_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::ffn_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::l_out>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::l_out>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1, 1,
			  1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::ffn_out>;
		using input_type02														 = core_traits<config_new, llama_op_types::ffn_inp>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::l_out>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_type krn_type{ kernel_type::add };
		static constexpr llama_op_types type{ llama_op_types::l_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::attn_residual>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::attn_residual>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length,
			  1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::kqv_out>;
		using input_type02														 = core_traits<config_new, llama_op_types::inp_out_ids>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::attn_residual>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::attn_residual };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::prev_residual>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::prev_residual>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length,
			  1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::l_out>;
		using input_type02														 = core_traits<config_new, llama_op_types::inp_out_ids>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::prev_residual>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::get_rows };
		static constexpr llama_op_types type{ llama_op_types::prev_residual };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::final_norm>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::final_norm>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1,
			  1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::l_out>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::norm_output_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::final_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_type01::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::rms_norm };
		static constexpr llama_op_types type{ llama_op_types::final_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::result_norm>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::result_norm>, model_traits_type<config_new>::embedding_dim, model_traits_type<config_new>::max_sequence_length, 1,
			  1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::final_norm>;
		using input_type02														 = core_traits<config_new, llama_op_types::output_norm_weight>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::norm_output_type;
		using transform_type													 = output_transform<input_type01::krn_type, input_type02::krn_type>;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::result_norm>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::mul };
		static constexpr llama_op_types type{ llama_op_types::result_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new> struct core_traits<config_new, llama_op_types::result_output>
		: public config_holder<config_new>,
		  core_trait_dims<core_traits<config_new, llama_op_types::result_output>, model_traits_type<config_new>::vocab_size, model_traits_type<config_new>::max_sequence_length, 1,
			  1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_type01														 = core_traits<config_new, llama_op_types::output_weight>;
		using input_type02														 = core_traits<config_new, llama_op_types::result_norm>;
		using output_type														 = typename kernel_type_profile_traits<config_new.kernel_profile>::logit_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config_new, llama_op_types::result_output>, model_traits_type::vocab_size, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr auto config{ config_holder<config_new>::config };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_type krn_type{ kernel_type::mul_mat };
		static constexpr llama_op_types type{ llama_op_types::result_output };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		uint64_t dim01{ model_traits_type::max_sequence_length };
		output_type* data{};
		int32_t value{};
	};

	template<model_config config_new, auto krn_type, uint64_t index> struct get_adjacent_value;

	template<model_config config_new, auto krn_type, uint64_t index> struct get_adjacent_value {
		using derived_type		   = core_traits<config_new, krn_type>;
		using model_traits_type	   = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using derived_derived_type = model<config_new>;
		using thread_pool_type	   = typename derived_derived_type::thread_pool_type;
		NIHILUS_FORCE_INLINE static auto& impl(derived_type& core) {
			if constexpr (index == 0) {
				return *static_cast<typename derived_type::input_type01*>(static_cast<typename derived_derived_type::thread_pool_type::core_base_type*>(&core));
			} else if constexpr (index == 1) {
				return *static_cast<typename derived_type::input_type02*>(static_cast<typename derived_derived_type::thread_pool_type::core_base_type*>(&core));
			} else if constexpr (index == 2) {
				return *static_cast<typename derived_type::input_type03*>(static_cast<typename derived_derived_type::thread_pool_type::core_base_type*>(&core));
			}
		}
	};

	template<nihilus::model_config config, typename... bases> struct core_bases : public bases... {
		template<template<nihilus::model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE constexpr void impl(arg_types&&... args) {
			(impl_internal_filtered<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<nihilus::model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE static constexpr void impl_static(arg_types&&... args) {
			(impl_internal_filtered_static<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}

	  protected:
		template<template<nihilus::model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_FORCE_INLINE constexpr void impl_internal_filtered(arg_types&&... args) {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), std::forward<arg_types>(args)...);
			}
		}

		template<template<nihilus::model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_FORCE_INLINE static constexpr void impl_internal_filtered_static(arg_types&&... args) {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(std::forward<arg_types>(args)...);
			}
		}
	};

	template<nihilus::model_config config, typename index_sequence> struct get_core_bases;

	template<nihilus::model_config config, uint64_t... index> struct get_core_bases<config, std::index_sequence<index...>> {
		using type = core_bases<config, nihilus::core_traits<config, static_cast<typename nihilus::model_traits_type<config>::op_type_type>(index)>...>;
	};

	template<nihilus::model_config config> using get_core_bases_t =
		typename get_core_bases<config, std::make_index_sequence<static_cast<uint64_t>(nihilus::model_traits_type<config>::op_type_type::count)>>::type;
}
