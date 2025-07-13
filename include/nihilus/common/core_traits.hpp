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

	enum class thread_strategy_type : uint8_t {
		none,
		global_input,
		global_output,
		per_block,
	};

	template<typename type01, typename type02> struct requires_dequant_or_quant {
		static constexpr bool required{ !std::is_same_v<type01, type02> };
	};

	template<model_config config_new, op_types op_type> struct core_traits;

	template<kernel_types kernel_type01, kernel_types kernel_type02> struct output_transform {};

	template<> struct output_transform<kernel_types::silu, kernel_types::mul_mat> {
		template<typename value_type> NIHILUS_INLINE static void impl(value_type*, uint64_t) {};
	};

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
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::token_embd_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::inp_tokens>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::feed_forward_length, 1, 1, 1, 0> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::input_token_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::feed_forward_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::inp_tokens };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::feed_forward_length };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::inp_pos> : public config_holder<config_new>,
																						  public core_trait_dims<model_traits_type<config_new>::feed_forward_length, 1, 1, 1, 0> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::position_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::feed_forward_length, 1, 1, 1, 0>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::inp_pos };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		uint64_t dim00{ model_traits_type::feed_forward_length };
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
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::inp_out_ids };
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
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::rope_freqs_weight };
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
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::output_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::output_norm_weight_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::output_norm_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_q_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::embedding_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::attn_q_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_k_weight>
		: public config_holder<config_new>,
		  public core_trait_dims<model_traits_type<config_new>::embedding_length,
			  (model_traits_type<config_new>::head_dim * model_traits_type<config_new>::attention_head_count_kv), 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_k_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, (model_traits_type::head_dim * model_traits_type::attention_head_count_kv), 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::attn_k_weight };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<output_type*, model_traits_type::block_count> data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_v_weight>
		: public config_holder<config_new>,
		  public core_trait_dims<model_traits_type<config_new>::embedding_length,
			  (model_traits_type<config_new>::head_dim * model_traits_type<config_new>::attention_head_count_kv), 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_v_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, (model_traits_type::head_dim * model_traits_type::attention_head_count_kv), 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::attn_v_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_output_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::embedding_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::attn_output_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::attn_norm_weight_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::attn_norm_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_gate_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::ffn_gate_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_up_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::ffn_up_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_down_weight_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::feed_forward_length, model_traits_type::embedding_length, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::ffn_down_weight };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_norm_weight_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::ffn_norm_weight };
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
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::cache_k };
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
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::cache_v };
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
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::kq_mask_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::block_count, model_traits_type::block_count, 1, 1>;
		static constexpr uint64_t depth{ 0 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::none };
		static constexpr kernel_types kernel_type{ kernel_types::none };
		static constexpr op_types type{ op_types::kq_mask };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::inp_embd>
		: public config_holder<config_new>, public get_new_dims_2_t<config_new, kernel_types::get_rows, op_types::token_embd_weight, op_types::inp_tokens, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::token_embd_weight>;
		using input_02_type															 = core_traits<config_new, op_types::inp_tokens>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::embedding_type;
		using core_traits_dims_type = get_new_dims_2_t<config_new, kernel_types::get_rows, op_types::token_embd_weight, op_types::inp_tokens, runtime_dims<1>>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_input };
		static constexpr kernel_types kernel_type{ kernel_types::get_rows };
		static constexpr op_types type{ op_types::inp_embd };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		core_latch run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::norm_attn_norm>
		: public config_holder<config_new>, public get_new_dims_2_t<config_new, kernel_types::rms_norm_mul, op_types::inp_embd, op_types::attn_norm_weight, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::inp_embd>;
		using input_02_type															 = core_traits<config_new, op_types::attn_norm_weight>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::norm_output_type;
		using transform_type														 = output_transform<input_01_type::kernel_type, input_02_type::kernel_type>;
		using core_traits_dims_type = get_new_dims_2_t<config_new, kernel_types::rms_norm_mul, op_types::inp_embd, op_types::attn_norm_weight, runtime_dims<1>>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::rms_norm_mul };
		static constexpr op_types type{ op_types::norm_attn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::qcur_mul_mat>
		: public config_holder<config_new>, public get_new_dims_2_t<config_new, kernel_types::mul_mat, op_types::attn_q_weight, op_types::norm_attn_norm, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attn_q_weight>;
		using input_02_type															 = core_traits<config_new, op_types::norm_attn_norm>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = get_new_dims_2_t<config_new, kernel_types::mul_mat, op_types::attn_q_weight, op_types::norm_attn_norm, runtime_dims<1>>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::qcur_mul_mat };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::qcur_reshaped>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::rope_dimension_count, model_traits_type<config_new>::block_count, 1, 1, 2> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::qcur_mul_mat>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::rope_dimension_count, model_traits_type::block_count, 1, 1, 2>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::reshape };
		static constexpr op_types type{ op_types::qcur_reshaped };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::qcur_rope>
		: public config_holder<config_new>,
		  public get_new_dims_3_t<config_new, kernel_types::rope, op_types::qcur_reshaped, op_types::inp_pos, op_types::rope_freqs_weight, runtime_dims<2>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::qcur_reshaped>;
		using input_02_type															 = core_traits<config_new, op_types::inp_pos>;
		using input_03_type															 = core_traits<config_new, op_types::rope_freqs_weight>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = get_new_dims_3_t<config_new, kernel_types::rope, op_types::qcur_reshaped, op_types::inp_pos, op_types::rope_freqs_weight, runtime_dims<2>>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::rope };
		static constexpr op_types type{ op_types::qcur_rope };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kcur_mul_mat>
		: public config_holder<config_new>, public get_new_dims_2_t<config_new, kernel_types::mul_mat, op_types::attn_q_weight, op_types::norm_attn_norm, runtime_dims<1>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attn_q_weight>;
		using input_02_type															 = core_traits<config_new, op_types::norm_attn_norm>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = get_new_dims_2_t<config_new, kernel_types::mul_mat, op_types::attn_q_weight, op_types::norm_attn_norm, runtime_dims<1>>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::kcur_mul_mat };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kcur_reshaped>
		: public config_holder<config_new>, public core_trait_dims<model_traits_type<config_new>::rope_dimension_count, model_traits_type<config_new>::block_count, 1, 1, 2> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kcur_mul_mat>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::rope_dimension_count, model_traits_type::block_count, 1, 1, 2>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::reshape };
		static constexpr op_types type{ op_types::kcur_reshaped };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kcur_rope>
		: public config_holder<config_new>,
		  public get_new_dims_3_t<config_new, kernel_types::rope, op_types::kcur_reshaped, op_types::inp_pos, op_types::rope_freqs_weight, runtime_dims<2>> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kcur_reshaped>;
		using input_02_type															 = core_traits<config_new, op_types::inp_pos>;
		using input_03_type															 = core_traits<config_new, op_types::rope_freqs_weight>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type = get_new_dims_3_t<config_new, kernel_types::rope, op_types::kcur_reshaped, op_types::inp_pos, op_types::rope_freqs_weight, runtime_dims<2>>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::rope };
		static constexpr op_types type{ op_types::kcur_rope };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::vcur_mul_mat>
		: public config_holder<config_new>, public get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::attn_v_weight, op_types::norm_attn_norm> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attn_v_weight>;
		using input_02_type															 = core_traits<config_new, op_types::norm_attn_norm>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type = get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::attn_v_weight, op_types::norm_attn_norm>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };


		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::vcur_mul_mat };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::k_cache_view> : public config_holder<config_new>,
																							   public get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_k> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::cache_k>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_k>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };


		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::view };
		static constexpr op_types type{ op_types::k_cache_view };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::k_cache_view_copy> : public config_holder<config_new>,
																									public get_new_dims_1_2_t<config_new, kernel_types::copy, op_types::kcur_rope> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kcur_rope>;
		using output_type															 = typename core_traits<config_new, op_types::k_cache_view>::output_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::copy, op_types::kcur_rope>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };


		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::copy };
		static constexpr op_types type{ op_types::k_cache_view_copy };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::vcur_transposed>
		: public config_holder<config_new>, public get_new_dims_1_2_t<config_new, kernel_types::transpose, op_types::vcur_mul_mat> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::vcur_mul_mat>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::transpose, op_types::vcur_mul_mat>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::transpose };
		static constexpr op_types type{ op_types::vcur_transposed };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

		template<model_config config_new> struct core_traits<config_new, op_types::v_cache_view> : public config_holder<config_new>,
																							   public get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_v> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::cache_v>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_v>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::view };
		static constexpr op_types type{ op_types::v_cache_view };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::v_cache_view_copy>
		: public config_holder<config_new>, public get_new_dims_1_2_t<config_new, kernel_types::copy, op_types::vcur_transposed> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::vcur_transposed>;
		using output_type															 = typename core_traits<config_new, op_types::v_cache_view>::output_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::copy, op_types::vcur_transposed>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::copy };
		static constexpr op_types type{ op_types::v_cache_view_copy };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::v> : public config_holder<config_new>,
																					public get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_v> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::cache_v>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::scale_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_v>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::view };
		static constexpr op_types type{ op_types::v };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::k> : public config_holder<config_new>,
																					public get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_k> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::cache_k>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::kv_cache_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::view, op_types::cache_k>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::view };
		static constexpr op_types type{ op_types::k };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::q> : public config_holder<config_new>,
																					public get_new_dims_1_2_t<config_new, kernel_types::permute, op_types::qcur_rope> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::qcur_rope>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::query_type;
		using core_traits_dims_type													 = get_new_dims_1_2_t<config_new, kernel_types::permute, op_types::qcur_rope>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::permute };
		static constexpr op_types type{ op_types::q };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kq> : public config_holder<config_new>,
																					 public get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::k, op_types::q> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::k>;
		using input_02_type															 = core_traits<config_new, op_types::q>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::attention_score_type;
		using core_traits_dims_type													 = get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::k, op_types::q>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::kq };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kq_soft_max>
		: public config_holder<config_new>, public get_new_dims_2_2_t<config_new, kernel_types::softmax, op_types::kq, op_types::kq_mask> {
		NIHILUS_INLINE constexpr core_traits() noexcept {};
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kq>;
		using input_02_type															 = core_traits<config_new, op_types::kq_mask>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::softmax_type;
		using core_traits_dims_type													 = get_new_dims_2_2_t<config_new, kernel_types::softmax, op_types::kq, op_types::kq_mask>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::softmax };
		static constexpr op_types type{ op_types::kq_soft_max };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kqv>
		: public config_holder<config_new>, public get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::v, op_types::kq_soft_max> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::v>;
		using input_02_type															 = core_traits<config_new, op_types::kq_soft_max>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type													 = get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::v, op_types::kq_soft_max>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::kqv };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kqv_merged>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::head_dim, model_traits_type<config_new>::attention_head_count, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kqv>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::head_dim, model_traits_type::attention_head_count, 1, 1>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::permute };
		static constexpr op_types type{ op_types::kqv_merged };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kqv_merged_cont> : public config_holder<config_new>,
																								  core_trait_dims<model_traits_type<config_new>::embedding_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kqv_merged>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::value_type;
		using core_traits_dims_type													 = core_trait_dims<model_traits_type::embedding_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ 0 };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::cont };
		static constexpr op_types type{ op_types::kqv_merged_cont };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::kqv_out>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::attn_output_weight>;
		using input_02_type															 = core_traits<config_new, op_types::kqv_merged_cont>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::hidden_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::kqv_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_inp_norm_out_ffn_norm>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kqv_out>;
		using input_02_type															 = core_traits<config_new, op_types::l_out>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using transform_type														 = output_transform<input_01_type::kernel_type, input_02_type::kernel_type>;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::add_rms_norm_mul };
		static constexpr op_types type{ op_types::ffn_inp_norm_out_ffn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_inp>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kqv_out>;
		using input_02_type															 = core_traits<config_new, op_types::l_out>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::add };
		static constexpr op_types type{ op_types::ffn_inp };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::norm_out>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_inp>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
		static constexpr op_types type{ op_types::norm_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_norm>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::norm_out>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_norm_weight>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using transform_type														 = output_transform<input_01_type::kernel_type, input_02_type::kernel_type>;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul };
		static constexpr op_types type{ op_types::ffn_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_gate>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_gate_weight>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_norm>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::feed_forward_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::ffn_gate };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_silu>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_gate>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::feed_forward_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::silu };
		static constexpr op_types type{ op_types::ffn_silu };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_up>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_up_weight>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_norm>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::feed_forward_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::ffn_up };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_gate_par>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::feed_forward_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_silu>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_up>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::ffn_intermediate_type;
		using transform_type														 = output_transform<input_01_type::kernel_type, input_02_type::kernel_type>;
		using core_traits_dims_type = core_trait_dims<model_traits_type::feed_forward_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul };
		static constexpr op_types type{ op_types::ffn_gate_par };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::ffn_out>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_down_weight>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_gate_par>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::hidden_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::ffn_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<op_latch, model_traits_type::block_count> sync_flag_start{};
		array<op_latch, model_traits_type::block_count> sync_flag_end{};
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::l_out>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::ffn_out>;
		using input_02_type															 = core_traits<config_new, op_types::ffn_inp>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::per_block };
		static constexpr kernel_types kernel_type{ kernel_types::add };
		static constexpr op_types type{ op_types::l_out };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		array<core_latch, model_traits_type::block_count> run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::attn_residual>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::kqv_out>;
		using input_02_type															 = core_traits<config_new, op_types::inp_out_ids>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_types kernel_type{ kernel_types::get_rows };
		static constexpr op_types type{ op_types::attn_residual };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		core_latch run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::prev_residual>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::l_out>;
		using input_02_type															 = core_traits<config_new, op_types::inp_out_ids>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::residual_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_types kernel_type{ kernel_types::get_rows };
		static constexpr op_types type{ op_types::prev_residual };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		core_latch run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::final_norm>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::l_out>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::norm_output_type;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ input_01_type::depth + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_types kernel_type{ kernel_types::rms_norm };
		static constexpr op_types type{ op_types::final_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		core_latch run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::result_norm>
		: public config_holder<config_new>, core_trait_dims<model_traits_type<config_new>::embedding_length, model_traits_type<config_new>::feed_forward_length, 1, 1, 1> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::final_norm>;
		using input_02_type															 = core_traits<config_new, op_types::output_norm_weight>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::norm_output_type;
		using transform_type														 = output_transform<input_01_type::kernel_type, input_02_type::kernel_type>;
		using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1, 1>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_types kernel_type{ kernel_types::mul };
		static constexpr op_types type{ op_types::result_norm };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		core_latch run_checkers{};
		output_type* data{};
	};

	template<model_config config_new> struct core_traits<config_new, op_types::result_output>
		: public config_holder<config_new>, public get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::output_weight, op_types::result_norm> {
		NIHILUS_INLINE constexpr core_traits() noexcept								 = default;
		NIHILUS_INLINE constexpr core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_INLINE constexpr core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_INLINE constexpr core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_INLINE constexpr core_traits(core_traits&&) noexcept				 = delete;
		using transform_type														 = int32_t;
		using model_traits_type														 = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using input_01_type															 = core_traits<config_new, op_types::output_weight>;
		using input_02_type															 = core_traits<config_new, op_types::result_norm>;
		using output_type															 = typename kernel_type_profile_traits<config_new.kernel_profile>::logit_type;
		using core_traits_dims_type = get_new_dims_2_2_t<config_new, kernel_types::mul_mat, op_types::output_weight, op_types::result_norm>;
		static constexpr uint64_t depth{ detail::max(input_01_type::depth, input_02_type::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_01_type::output_type, typename input_02_type::output_type>::required };
		static constexpr array<uint64_t, 4> strides{ type_traits<output_type>::impl(core_traits_dims_type::get_array()) };
		static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment>(type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) +
			(dequantization ? type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array()) : 0)) };
		static constexpr thread_strategy_type layer_type{ thread_strategy_type::global_output };
		static constexpr kernel_types kernel_type{ kernel_types::mul_mat };
		static constexpr op_types type{ op_types::result_output };
		static constexpr uint64_t count{ total_required_bytes / sizeof(output_type) };
		op_latch sync_flag_start{};
		op_latch sync_flag_end{};
		core_latch run_checkers{};
		output_type* data{};
	};

	template<model_config config_new, auto kernel_type, uint64_t index> struct get_adjacent_value;

	template<model_config config_new, auto kernel_type, uint64_t index> struct get_adjacent_value {
		using derived_type		   = core_traits<config_new, kernel_type>;
		using model_traits_type	   = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;
		using derived_derived_type = model<config_new>;
		using thread_pool_type	   = typename derived_derived_type::thread_pool_type;
		NIHILUS_INLINE static auto& impl(derived_type& parse_core) {
			if constexpr (index == 0) {
				return *static_cast<typename derived_type::input_01_type*>(static_cast<typename derived_derived_type::thread_pool_type::core_base_type*>(&parse_core));
			} else if constexpr (index == 1) {
				return *static_cast<typename derived_type::input_02_type*>(static_cast<typename derived_derived_type::thread_pool_type::core_base_type*>(&parse_core));
			} else if constexpr (index == 2) {
				return *static_cast<typename derived_type::input_03_type*>(static_cast<typename derived_derived_type::thread_pool_type::core_base_type*>(&parse_core));
			}
		}
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02, core_traits_types core_traits03, runtime_dims_t... dimension_type>
	struct get_new_dims<kernel_type, core_traits01, core_traits02, core_traits03, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		using dim_traits03			 = typename core_traits03::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();
		static constexpr auto dims03 = dim_traits03::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::rope) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02, core_traits03, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02>
	struct get_new_dims<kernel_type, core_traits01, core_traits02> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::get_rows) {
				return core_trait_dims<dims01[0], dims02[0], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::rms_norm_mul) {
				return core_trait_dims<dims01[0], dims01[0], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02, runtime_dims_t... dimension_type>
	struct get_new_dims<kernel_type, core_traits01, core_traits02, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::get_rows) {
				return core_trait_dims<dims01[0], dims02[0], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::rms_norm_mul) {
				return core_trait_dims<dims01[0], dims01[0], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, runtime_dims_t... dimension_type> struct get_new_dims<kernel_type, core_traits01, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();

		static constexpr auto get_new_dims_fn() {
			return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01> struct get_new_dims<kernel_type, core_traits01> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();

		static constexpr auto get_new_dims_fn() {
			return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
		}

		using type = decltype(get_new_dims_fn());
	};
}