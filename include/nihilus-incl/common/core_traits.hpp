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

#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/common/kernel_type_profile_traits.hpp>
#include <nihilus-incl/common/model_traits.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <latch>

namespace nihilus {

	enum class thread_strategy_types : uint8_t {
		none,
		global_input,
		global_output,
		per_block,
	};

	enum class allocation_strategy_types : uint8_t {
		none,
		per_block,
		remap,
		mmap,
		global,
	};

	template<typename type01, typename type02> struct requires_dequant_or_quant {
		static constexpr bool required{ !std::is_same_v<type01, type02> };
	};

	template<model_config config_new, op_types op_type> struct core_traits_new;

	template<model_config config_new, op_types op_type> struct core_traits;

	template<kernel_types kernel_type01, kernel_types kernel_type02> struct output_transform {};

	template<model_config config_new> using model_traits_type = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;

	template<model_config config_new> struct model;

	template<model_config config_new> struct core_traits<config_new, op_types::token_embd_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::vocab_size, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, model_traits_type::vocab_size, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::token_embd_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::inp_tokens> : public config_holder<config_new>,
																							 public core_trait_dims<config_new.default_max_context_length, 1, 1, 1, 0> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::input_token_type;
		using core_traits_dims_type													 = core_trait_dims<config_new.default_max_context_length, 1, 1, 1, 0>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_input };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::inp_tokens };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::inp_pos> : public config_holder<config_new>,
																						  public core_trait_dims<config_new.default_max_context_length, 1, 1, 1, 0> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::position_type;
		using core_traits_dims_type													 = core_trait_dims<config_new.default_max_context_length, 1, 1, 1, 0>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_input };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::inp_pos };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::inp_out_ids> : public config_holder<config_new>, public core_trait_dims<1, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::output_token_type;
		using core_traits_dims_type													 = core_trait_dims<1, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_input };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::inp_out_ids };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::rope_freqs_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::rope_dimension_count / 2, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::rope_dimension_count / 2, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::rope_freqs_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::output_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::vocab_size, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, model_traits_type::vocab_size, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::output_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::output_norm_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::output_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_q_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::embedding_length, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::embedding_length, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::attn_q_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_k_weight>
		: public config_holder<config_new>,
		  public core_trait_dims<model_traits_type<config_new>::embedding_length,
			  (model_traits_type<config_new>::rope_dimension_count * model_traits_type<config_new>::attention_head_count_kv), 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type =
			core_trait_dims<model_traits_type::embedding_length, (model_traits_type::rope_dimension_count * model_traits_type::attention_head_count_kv), 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::attn_k_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_v_weight>
		: public config_holder<config_new>,
		  public core_trait_dims<model_traits_type<config_new>::embedding_length,
			  (model_traits_type<config_new>::rope_dimension_count * model_traits_type<config_new>::attention_head_count_kv), 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type =
			core_trait_dims<model_traits_type::embedding_length, (model_traits_type::rope_dimension_count * model_traits_type::attention_head_count_kv), 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::attn_v_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_output_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::embedding_length, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::embedding_length, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::attn_output_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_norm_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::attn_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_gate_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::ffn_gate_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_up_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::none };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::ffn_up_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_down_weight>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::embedding_length, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::feed_forward_length, model_traits_type::embedding_length, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::ffn_down_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_norm_weight> : public config_holder<config_new>,
																								  public core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::weight_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::mmap };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::ffn_norm_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::cache_k>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::n_embd_kv_gqa * model_traits_type<config_new>::embedding_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::n_embd_kv_gqa * model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::cache_k };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::cache_v>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::n_embd_kv_gqa * model_traits_type<config_new>::embedding_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::n_embd_kv_gqa * model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::cache_v };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kq_mask>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::block_count, model_traits_type<config_new>::block_count, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::mask_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::block_count, model_traits_type::block_count, 1, 1>;
		static constexpr input_types input_type{ input_types::none };
		static constexpr uint64_t depth{ 0 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_input };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types op_type{ op_types::kq_mask };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::embedding_lookup>
		: public config_holder<config_new>, public get_new_dims_2_t<config_new, kernel_types::embedding_lookup, op_types::token_embd_weight, op_types::inp_tokens, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::token_embd_weight>;
		using input_02_type															 = core_traits<config_new, op_types::inp_tokens>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type = get_new_dims_2_t<config_new, kernel_types::embedding_lookup, op_types::token_embd_weight, op_types::inp_tokens, runtime_dims<1>>;
		static constexpr input_types input_type{ input_types::two };
		static constexpr uint64_t global_input_count{ 0 };
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_input };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::embedding_lookup };
		static constexpr op_types op_type{ op_types::embedding_lookup };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<linked_latch<false>, model_traits_type::block_count> latch{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attention_preprocessing>
		: public config_holder<config_new>,
		  public get_new_dims_6_t<config_new, kernel_types::fused_qkv_rope, op_types::embedding_lookup, op_types::attn_norm_weight, op_types::rope_freqs_weight,
			  op_types::attn_q_weight, op_types::attn_k_weight, op_types::attn_v_weight, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type	= int32_t;
		using model_traits_type = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type = core_traits<config_new, op_types::embedding_lookup>;
		using input_02_type = core_traits<config_new, op_types::attn_norm_weight>;
		using input_03_type = core_traits<config_new, op_types::rope_freqs_weight>;
		using input_04_type = core_traits<config_new, op_types::attn_q_weight>;
		using input_05_type = core_traits<config_new, op_types::attn_k_weight>;
		using input_06_type = core_traits<config_new, op_types::attn_v_weight>;
		using output_type			= typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type = get_new_dims_6_t<config_new, kernel_types::fused_qkv_rope, op_types::embedding_lookup, op_types::attn_norm_weight,
			op_types::rope_freqs_weight, op_types::attn_q_weight, op_types::attn_k_weight, op_types::attn_v_weight, runtime_dims<1>>;
		static constexpr input_types input_type{ input_types::six };
		static constexpr uint64_t depth{
			detail::max(input_01_type::depth,
				detail::max(input_02_type::depth, detail::max(input_03_type::depth, detail::max(input_04_type::depth, detail::max(input_05_type::depth, input_06_type::depth))))) +
			1
		};
		static constexpr uint64_t global_input_count{ 1 };
		static constexpr uint64_t runtime_dim_multiplier{ 3 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides) * 3) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::fused_qkv_rope };
		static constexpr op_types op_type{ op_types::attention_preprocessing };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<linked_latch<true>, model_traits_type::block_count> latch{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attention_computation>
		: public config_holder<config_new>,
		  public get_new_dims_2_t<config_new, kernel_types::fused_attention, op_types::attention_preprocessing, op_types::kq_mask, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attention_preprocessing>;
		using input_02_type															 = core_traits<config_new, op_types::kq_mask>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type = get_new_dims_2_t<config_new, kernel_types::fused_attention, op_types::attention_preprocessing, op_types::kq_mask, runtime_dims<1>>;
		static constexpr input_types input_type{ input_types::two };
		static constexpr uint64_t global_input_count{ 0 };
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr bool dequantization{ false };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::fused_attention };
		static constexpr op_types op_type{ op_types::attention_computation };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<linked_latch<true>, model_traits_type::block_count> latch{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attention_output>
		: public config_holder<config_new>,
		  public get_new_dims_3_t<config_new, kernel_types::fused_attn_out, op_types::attention_computation, op_types::attn_output_weight, op_types::embedding_lookup,
			  runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attention_computation>;
		using input_02_type															 = core_traits<config_new, op_types::attn_output_weight>;
		using input_03_type															 = core_traits<config_new, op_types::embedding_lookup>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type =
			get_new_dims_3_t<config_new, kernel_types::fused_attn_out, op_types::attention_computation, op_types::attn_output_weight, op_types::embedding_lookup, runtime_dims<1>>;
		static constexpr input_types input_type{ input_types::three };
		static constexpr uint64_t global_input_count{ 1 };
		static constexpr uint64_t depth{ detail::max(detail::max(input_01_type::depth, input_02_type::depth), input_03_type::depth) + 1 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::fused_attn_out };
		static constexpr op_types op_type{ op_types::attention_output };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<linked_latch<true>, model_traits_type::block_count> latch{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::swiglu_transformation>
		: public config_holder<config_new>,
		  public get_new_dims_5_t<config_new, kernel_types::fused_swiglu, op_types::attention_output, op_types::ffn_norm_weight, op_types::ffn_gate_weight, op_types::ffn_up_weight,
			  op_types::ffn_down_weight, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attention_output>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_norm_weight>;
		using input_03_type															 = core_traits<config_new, op_types::ffn_gate_weight>;
		using input_04_type															 = core_traits<config_new, op_types::ffn_up_weight>;
		using input_05_type															 = core_traits<config_new, op_types::ffn_down_weight>;
		using output_type			= typename kernel_type_profile_traits<config_new.kernel_profile>::compute_type;
		using core_traits_dims_type = get_new_dims_5_t<config_new, kernel_types::fused_swiglu, op_types::attention_output, op_types::ffn_norm_weight, op_types::ffn_gate_weight,
			op_types::ffn_up_weight, op_types::ffn_down_weight, runtime_dims<1>>;
		static constexpr input_types input_type{ input_types::five };
		static constexpr uint64_t global_input_count{ 0 };
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::fused_swiglu };
		static constexpr op_types op_type{ op_types::swiglu_transformation };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<linked_latch<true>, model_traits_type::block_count> latch{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::final_projection>
		: public config_holder<config_new>,
		  public get_new_dims_3_t<config_new, kernel_types::fused_final_proj, op_types::swiglu_transformation, op_types::output_norm_weight, op_types::output_weight,
			  runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::swiglu_transformation>;
		using input_02_type															 = core_traits<config_new, op_types::output_norm_weight>;
		using input_03_type															 = core_traits<config_new, op_types::output_weight>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::logit_type;
		using core_traits_dims_type =
			get_new_dims_3_t<config_new, kernel_types::fused_final_proj, op_types::swiglu_transformation, op_types::output_norm_weight, op_types::output_weight, runtime_dims<1>>;
		static constexpr input_types input_type{ input_types::three };
		static constexpr uint64_t global_input_count{ 0 };
		static constexpr uint64_t depth{ detail::max(detail::max(input_01_type::depth, input_02_type::depth), input_03_type::depth) + 1 };
		static constexpr uint64_t runtime_dim_multiplier{ 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(
			type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array(), strides)) };
		static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_output };
		static constexpr allocation_strategy_types allocation_strategy_type{ allocation_strategy_types::global };
		static constexpr kernel_types kernel_type{ kernel_types::fused_final_proj };
		static constexpr op_types op_type{ op_types::final_projection };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<linked_latch<true>, model_traits_type::block_count> latch{};
		output_type* data{};
	};

	template<model_config config_new, auto kernel_type> struct get_adjacent_value;

	template<model_config config_new, auto kernel_type> struct get_adjacent_value {
		using derived_type		   = core_traits<config_new, kernel_type>;
		using model_traits_type	   = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using derived_derived_type = model<config_new>;
		using thread_pool_type	   = typename derived_derived_type::thread_pool_type;
		NIHILUS_INLINE static auto& impl(derived_type& parse_core) {
			return *static_cast<typename derived_derived_type::thread_pool_type*>(&parse_core);
		}
	};

	template<typename value_type>
	concept final_output_type = std::remove_cvref_t<value_type>::type == op_types::final_projection;

	template<typename core_traits_type, model_config config> static constexpr auto runtime_dims_multipliers{ []() {
		array<uint64_t, model_traits_type<config>::block_count> multipliers{};
		if constexpr (final_output_type<core_traits_type>) {
			for (uint64_t i = 0; i < model_traits_type<config>::block_count; ++i) {
				multipliers[i] = 1;
			}
			multipliers[model_traits_type<config>::block_count - 1] = 0;
		} else {
			multipliers.fill(1);
		}
		return multipliers;
	}() };
}
