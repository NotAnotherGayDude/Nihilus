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

	template<model_config config, llama_op_types op_type> struct core_traits;

	template<kernel_type kernel_type01, kernel_type kernel_type02> struct output_transform {};

	template<> struct output_transform<kernel_type::silu, kernel_type::mul_mat> {
		template<typename value_type> NIHILUS_FORCE_INLINE static void impl(value_type*, uint64_t) {};
	};

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

		template<uint64_t index> NIHILUS_FORCE_INLINE uint64_t operator[](tag<index>) const {
			if constexpr (index == 0) {
				return static_cast<const derived_type*>(this)->dim00;
			} else if constexpr (index == 1) {
				return static_cast<const derived_type*>(this)->dim01;
			} else if constexpr (index == 2) {
				return static_cast<const derived_type*>(this)->dim02;
			} else {
				return static_cast<const derived_type*>(this)->dim03;
			}
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			if (index == 0) {
				return static_cast<const derived_type*>(this)->dim00;
			} else if (index == 1) {
				return static_cast<const derived_type*>(this)->dim01;
			} else if (index == 2) {
				return static_cast<const derived_type*>(this)->dim02;
			} else {
				return static_cast<const derived_type*>(this)->dim03;
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

		template<uint64_t index> NIHILUS_FORCE_INLINE uint64_t operator[](tag<index>) const {
			if constexpr (index == 0) {
				return static_cast<const derived_type*>(this)->dim00;
			} else if constexpr (index == 1) {
				return static_cast<const derived_type*>(this)->dim01;
			} else if constexpr (index == 2) {
				return static_cast<const derived_type*>(this)->dim02;
			} else {
				return static_cast<const derived_type*>(this)->dim03;
			}
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			if (index == 0) {
				return static_cast<const derived_type*>(this)->dim00;
			} else if (index == 1) {
				return static_cast<const derived_type*>(this)->dim01;
			} else if (index == 2) {
				return static_cast<const derived_type*>(this)->dim02;
			} else {
				return static_cast<const derived_type*>(this)->dim03;
			}
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

		template<uint64_t index> NIHILUS_FORCE_INLINE uint64_t operator[](tag<index>) const {
			if constexpr (index == 0) {
				return static_cast<const derived_type*>(this)->dim00;
			} else if constexpr (index == 1) {
				return static_cast<const derived_type*>(this)->dim01;
			} else if constexpr (index == 2) {
				return static_cast<const derived_type*>(this)->dim02;
			} else {
				return static_cast<const derived_type*>(this)->dim03;
			}
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			if (index == 0) {
				return static_cast<const derived_type*>(this)->dim00;
			} else if (index == 1) {
				return static_cast<const derived_type*>(this)->dim01;
			} else if (index == 2) {
				return static_cast<const derived_type*>(this)->dim02;
			} else {
				return static_cast<const derived_type*>(this)->dim03;
			}
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
		using core_traits_dims_type = core_trait_dims<core_traits<config, llama_op_types::cache_k>, model_traits_type::n_embd_kv_gqa, model_traits_type::embedding_dim, 1, 1>;
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
		using this_type															 = core_traits<config, llama_op_types::inp_embd>;
		using input_type01														 = core_traits<config, llama_op_types::token_embd_weight>;
		using input_type02														 = core_traits<config, llama_op_types::inp_tokens>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::embedding_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::inp_embd>, model_traits_type::embedding_dim, model_traits_type::max_sequence_length, 1, 1, 1>;
		static constexpr uint64_t depth{ std::max(input_type01::depth, input_type02::depth) + 1 };
		static constexpr bool dequantization{ requires_dequant_or_quant<typename input_type01::output_type, typename input_type02::output_type>::required };
		static constexpr alloc_type alc_type{ alloc_type::single_alloc };
		static constexpr array<uint64_t, 4> strides{ { type_traits<output_type>::impl(core_traits_dims_type::get_array()) } };
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
		using this_type															 = core_traits<config, llama_op_types::norm>;
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
		using this_type															 = core_traits<config, llama_op_types::attn_norm>;
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
		using this_type															 = core_traits<config, llama_op_types::qcur>;
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
		using this_type															 = core_traits<config, llama_op_types::qcur_reshaped>;
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
		using this_type															 = core_traits<config, llama_op_types::qcur_rope>;
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
		using this_type															 = core_traits<config, llama_op_types::kcur>;
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
		using this_type															 = core_traits<config, llama_op_types::kcur_reshaped>;
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
		using this_type															 = core_traits<config, llama_op_types::kcur_rope>;
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
		using this_type															 = core_traits<config, llama_op_types::vcur>;
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
		: core_trait_dims<core_traits<config, llama_op_types::k_cache_view>, model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim,
			  model_traits_type<config>::max_sequence_length, 1, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using this_type															 = core_traits<config, llama_op_types::k_cache_view>;
		using input_type01														 = core_traits<config, llama_op_types::cache_k>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::k_cache_view>, model_traits_type::n_embd_kv_gqa, model_traits_type::max_sequence_length, 1, 1, 1>;
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
		using this_type															 = core_traits<config, llama_op_types::k_cache_view_copy>;
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
		using this_type															 = core_traits<config, llama_op_types::vcur_transposed>;
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
		: core_trait_dims<core_traits<config, llama_op_types::v_cache_view>, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count_kv * model_traits_type<config>::head_dim, 1, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using this_type															 = core_traits<config, llama_op_types::v_cache_view>;
		using input_type01														 = core_traits<config, llama_op_types::cache_v>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::v_cache_view>, model_traits_type::max_sequence_length, model_traits_type::n_embd_kv_gqa, 1, 1, 0>;
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
		using this_type															 = core_traits<config, llama_op_types::v_cache_view_copy>;
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

	template<model_config config> struct core_traits<config, llama_op_types::v>
		: core_trait_dims<core_traits<config, llama_op_types::v>, model_traits_type<config>::max_sequence_length, model_traits_type<config>::head_dim,
			  model_traits_type<config>::head_count_kv, 1, 0> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using this_type															 = core_traits<config, llama_op_types::v>;
		using input_type01														 = core_traits<config, llama_op_types::cache_v>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::scale_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::v>, model_traits_type::max_sequence_length, model_traits_type::head_dim, model_traits_type::head_count_kv, 1, 0>;
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

	template<model_config config> struct core_traits<config, llama_op_types::k>
		: core_trait_dims<core_traits<config, llama_op_types::k>, model_traits_type<config>::head_dim, model_traits_type<config>::max_sequence_length,
			  model_traits_type<config>::head_count_kv, 1, 1> {
		NIHILUS_FORCE_INLINE core_traits() noexcept								 = default;
		NIHILUS_FORCE_INLINE core_traits& operator=(const core_traits&) noexcept = delete;
		NIHILUS_FORCE_INLINE core_traits(const core_traits&) noexcept			 = delete;
		NIHILUS_FORCE_INLINE core_traits& operator=(core_traits&&) noexcept		 = delete;
		NIHILUS_FORCE_INLINE core_traits(core_traits&&) noexcept				 = delete;
		using transform_type													 = int32_t;
		using model_traits_type													 = model_traits<config.arch, config.model_size, config.model_generation>;
		using this_type															 = core_traits<config, llama_op_types::k>;
		using input_type01														 = core_traits<config, llama_op_types::cache_k>;
		using output_type														 = typename kernel_type_profile_traits<config.kernel_profile>::scale_type;
		using core_traits_dims_type =
			core_trait_dims<core_traits<config, llama_op_types::k>, model_traits_type::head_dim, model_traits_type::max_sequence_length, model_traits_type::head_count_kv, 1, 1>;
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
		using this_type															 = core_traits<config, llama_op_types::q>;
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
		using this_type															 = core_traits<config, llama_op_types::kq>;
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
		using this_type															 = core_traits<config, llama_op_types::kq_soft_max>;
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
		using this_type															 = core_traits<config, llama_op_types::kqv>;
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
		using this_type															 = core_traits<config, llama_op_types::kqv_merged>;
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
		using this_type															 = core_traits<config, llama_op_types::kqv_merged_cont>;
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
		using this_type															 = core_traits<config, llama_op_types::kqv_out>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_inp>;
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
		using this_type															 = core_traits<config, llama_op_types::norm_out>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_norm>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_gate>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_silu>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_up>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_gate_par>;
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
		using this_type															 = core_traits<config, llama_op_types::ffn_out>;
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
		using this_type															 = core_traits<config, llama_op_types::l_out>;
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
		using this_type															 = core_traits<config, llama_op_types::attn_residual>;
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
		using this_type															 = core_traits<config, llama_op_types::prev_residual>;
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
		using this_type															 = core_traits<config, llama_op_types::final_norm>;
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
		using this_type															 = core_traits<config, llama_op_types::result_norm>;
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
		using this_type															 = core_traits<config, llama_op_types::result_output>;
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

	template<model_config config, auto krn_type, uint64_t index> struct get_adjacent_value;

	template<model_config config, auto krn_type, uint64_t index> struct get_adjacent_value {
		using derived_type		   = core_traits<config, krn_type>;
		using model_traits_type	   = model_traits<config.arch, config.model_size, config.model_generation>;
		using derived_derived_type = model<config>;
		NIHILUS_FORCE_INLINE static auto& impl(derived_type& core) {
			if constexpr (index == 0) {
				using input_type01 = typename derived_type::input_type01;
				return *static_cast<input_type01*>(static_cast<derived_derived_type*>(&core));
			} else if constexpr (index == 1) {
				using input_type02 = typename derived_type::input_type02;
				return *static_cast<input_type02*>(static_cast<derived_derived_type*>(&core));
			} else if constexpr (index == 2) {
				using input_type03 = typename derived_type::input_type03;
				return *static_cast<input_type03*>(static_cast<derived_derived_type*>(&core));
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

		template<template<typename> typename mixin_type, typename op_entity_type, typename... arg_types>
		NIHILUS_FORCE_INLINE static constexpr void impl_internal_constexpr(arg_types&&... args) {
			return mixin_type<op_entity_type>::impl(std::forward<arg_types>(args)...);
		}

		template<template<typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE static constexpr void impl_constexpr(arg_types&&... args) {
			(impl_internal_constexpr<mixin_type, bases>(args...), ...);
		}
	};

	template<model_config config, typename index_sequence> struct get_constexpr_core_bases_base;

	template<model_config config> using op_type_type_t = typename model_traits<config.arch, config.model_size, config.model_generation>::op_type_type;

	template<model_config config, uint64_t... index> struct get_constexpr_core_bases_base<config, std::index_sequence<index...>> {
		using type = core_bases<core_traits<config, static_cast<op_type_type_t<config>>(index)>...>;
	};

	template<model_config config> using get_constexpr_core_bases_config_base_t =
		typename get_constexpr_core_bases_base<config, std::make_index_sequence<static_cast<uint64_t>(op_type_type_t<config>::count)>>::type;

	template<model_config config, typename enum_holder_type, typename index_sequence> struct get_runtime_core_bases_base;

	template<model_config config> using op_type_type_t = typename model_traits<config.arch, config.model_size, config.model_generation>::op_type_type;

	template<model_config config, typename enum_holder_type, uint64_t... index> struct get_runtime_core_bases_base<config, enum_holder_type, std::index_sequence<index...>> {
		using type = core_bases<core_traits<config, enum_holder_type::active_traits[index]>...>;
	};

	template<model_config config, typename enum_holder_type> using get_runtime_core_bases_config_base_t =
		typename get_runtime_core_bases_base<config, enum_holder_type, std::make_index_sequence<static_cast<uint64_t>(op_type_type_t<config>::count)>>::type;

}
