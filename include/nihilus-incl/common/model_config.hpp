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

#include <nihilus-incl/common/utility.hpp>

namespace nihilus {

	template<typename value_type, typename derived_type> struct config_type_base_class {
	  protected:
		template<typename, typename> friend struct is_same_enum_type;
		enum class enum_type : value_type {
			disabled = std::numeric_limits<value_type>::min(),
			enabled	 = std::numeric_limits<value_type>::max(),
		} value{};

		NIHILUS_HOST constexpr config_type_base_class() noexcept {
		}

		NIHILUS_HOST constexpr config_type_base_class(enum_type value_new) noexcept : value{ value_new } {
		}

		NIHILUS_HOST constexpr config_type_base_class(value_type value_new) noexcept : value{ static_cast<enum_type>(value_new) } {
		}

	  public:
		static constexpr enum_type disabled{ enum_type::disabled };
		static constexpr enum_type enabled{ enum_type::enabled };

		NIHILUS_HOST constexpr operator value_type() const {
			return static_cast<value_type>(value);
		}
	};

	struct exception_type : config_type_base_class<bool, exception_type> {
	  protected:
		using base_type = config_type_base_class<bool, exception_type>;

	  public:
		NIHILUS_HOST constexpr exception_type() noexcept {
		}
		NIHILUS_HOST constexpr exception_type(enum_type value_new) noexcept : base_type{ value_new } {
		}
	};

	struct benchmark_type : config_type_base_class<bool, benchmark_type> {
	  protected:
		using base_type = config_type_base_class<bool, benchmark_type>;

	  public:
		NIHILUS_HOST constexpr benchmark_type() noexcept {
		}
		NIHILUS_HOST constexpr benchmark_type(enum_type value_new) noexcept : base_type{ value_new } {
		}
	};

	struct dev_type : config_type_base_class<bool, dev_type> {
	  protected:
		using base_type = config_type_base_class<bool, dev_type>;

	  public:
		NIHILUS_HOST constexpr dev_type() noexcept {
		}
		NIHILUS_HOST constexpr dev_type(enum_type value_new) noexcept : base_type{ value_new } {
		}
	};

	struct use_rotary_embeddings_type : config_type_base_class<bool, use_rotary_embeddings_type> {
	  protected:
		using base_type = config_type_base_class<bool, use_rotary_embeddings_type>;

	  public:
		NIHILUS_HOST constexpr use_rotary_embeddings_type() noexcept {
		}
		NIHILUS_HOST constexpr use_rotary_embeddings_type(enum_type value_new) noexcept : base_type{ value_new } {
		}
	};

	struct default_max_sequence_length_type : config_type_base_class<uint64_t, default_max_sequence_length_type> {
	  protected:
		using base_type = config_type_base_class<uint64_t, default_max_sequence_length_type>;

	  public:
		NIHILUS_HOST constexpr default_max_sequence_length_type() noexcept {
		}
		NIHILUS_HOST constexpr explicit default_max_sequence_length_type(uint64_t value_new) noexcept : base_type{ value_new } {
		}
	};

	struct batch_size_type : config_type_base_class<uint64_t, batch_size_type> {
	  protected:
		using base_type = config_type_base_class<uint64_t, batch_size_type>;

	  public:
		NIHILUS_HOST constexpr batch_size_type() noexcept {
		}
		NIHILUS_HOST constexpr explicit batch_size_type(uint64_t value_new) noexcept : base_type{ value_new } {
		}
	};

	struct kv_cache_block_size_type : config_type_base_class<uint64_t, kv_cache_block_size_type> {
	  protected:
		using base_type = config_type_base_class<uint64_t, kv_cache_block_size_type>;

	  public:
		NIHILUS_HOST constexpr kv_cache_block_size_type() noexcept {
		}
		NIHILUS_HOST constexpr explicit kv_cache_block_size_type(uint64_t value_new) noexcept : base_type{ value_new } {
		}
	};

	template<typename value_type, typename enum_type> struct is_same_enum_type {
		static constexpr bool value{ detail::is_same_v<typename detail::remove_cvref_t<value_type>::enum_type, enum_type> };
	};

	template<typename value_type, typename enum_type> static constexpr bool is_same_enum_type_v = is_same_enum_type<value_type, enum_type>::value;

	template<typename value_type>
	concept exception_types = detail::is_same_v<exception_type, detail::remove_cvref_t<value_type>> || is_same_enum_type_v<exception_type, value_type>;

	template<typename value_type>
	concept default_max_sequence_length_types = detail::is_same_v<default_max_sequence_length_type, detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept batch_size_types = detail::is_same_v<batch_size_type, detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept kv_cache_block_size_types = detail::is_same_v<kv_cache_block_size_type, detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept use_rotary_embeddings_types =
		detail::is_same_v<use_rotary_embeddings_type, detail::remove_cvref_t<value_type>> || is_same_enum_type_v<use_rotary_embeddings_type, value_type>;

	template<typename value_type>
	concept benchmark_types = detail::is_same_v<benchmark_type, detail::remove_cvref_t<value_type>> || is_same_enum_type_v<benchmark_type, value_type>;

	template<typename value_type>
	concept dev_types = detail::is_same_v<dev_type, detail::remove_cvref_t<value_type>> || is_same_enum_type_v<dev_type, value_type>;

	struct model_config {
		model_generations model_generation{ model_generations::v3_1 };
		model_sizes model_size{ model_sizes::llm_8B };
		kernel_type_profiles kernel_type_profile{};
		model_arches model_arch{};
		exception_type exceptions{};
		default_max_sequence_length_type max_sequence_length{ 1024 };
		batch_size_type batch_size{ 1 };
		kv_cache_strategies kv_cache_strategy{};
		user_input_types user_input_type{};
		rope_scaling_types rope_scaling_type{};
		tokenizer_pre_types tokenizer_pre_type{ tokenizer_pre_types::llama3 };
		kv_cache_block_size_type kv_cache_block_size{};
		use_rotary_embeddings_type use_rotary_embeddings{};
		rms_norm_types rms_norm_type{};
		tokenizer_types tokenizer_type{ tokenizer_types::bpe };
		device_types device_type{};
		model_formats model_format{ model_formats::gguf };
		float norm_epsilon{};
		benchmark_type benchmark{};
		dev_type dev{};

		template<detail::same_as<model_generations> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.model_generation = value;
			return return_value;
		}

		template<detail::same_as<model_sizes> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.model_size = value;
			return return_value;
		}

		template<detail::same_as<kernel_type_profiles> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.kernel_type_profile = value;
			return return_value;
		}

		template<detail::same_as<model_arches> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.model_arch = value;
			return return_value;
		}

		template<exception_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.exceptions = value;
			return return_value;
		}

		template<default_max_sequence_length_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.max_sequence_length = value;
			return return_value;
		}

		template<batch_size_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.batch_size = value;
			return return_value;
		}

		template<detail::same_as<kv_cache_strategies> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.kv_cache_strategy = value;
			return return_value;
		}

		template<detail::same_as<user_input_types> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.user_input_type = value;
			return return_value;
		}

		template<detail::same_as<rope_scaling_types> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.rope_scaling_type = value;
			return return_value;
		}

		template<detail::same_as<tokenizer_pre_types> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.tokenizer_pre_type = value;
			return return_value;
		}

		template<kv_cache_block_size_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.kv_cache_block_size = value;
			return return_value;
		}

		template<use_rotary_embeddings_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.use_rotary_embeddings = value;
			return return_value;
		}

		template<detail::same_as<rms_norm_types> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.rms_norm_type = value;
			return return_value;
		}

		template<detail::same_as<tokenizer_types> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.tokenizer_type = value;
			return return_value;
		}

		template<detail::same_as<device_types> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.device_type = value;
			return return_value;
		}

		template<detail::same_as<model_formats> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.model_format = value;
			return return_value;
		}

		template<detail::same_as<float> value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.norm_epsilon = value;
			return return_value;
		}

		template<benchmark_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.benchmark = value;
			return return_value;
		}

		template<dev_types value_type> consteval auto update(const value_type value) const {
			model_config return_value{ *this };
			return_value.dev = value;
			return return_value;
		}
	};

	template<const model_config& config> struct model_config_type {
		static constexpr model_generations model_generation		  = config.model_generation;
		static constexpr model_sizes model_size					  = config.model_size;
		static constexpr kernel_type_profiles kernel_type_profile = config.kernel_type_profile;
		static constexpr model_arches model_arch				  = config.model_arch;
		static constexpr bool exceptions						  = config.exceptions;
		static constexpr uint64_t max_sequence_length			  = config.max_sequence_length;
		static constexpr uint64_t batch_size					  = config.batch_size;
		static constexpr kv_cache_strategies kv_cache_strategy	  = config.kv_cache_strategy;
		static constexpr user_input_types user_input_type		  = config.user_input_type;
		static constexpr rope_scaling_types rope_scaling_type	  = config.rope_scaling_type;
		static constexpr tokenizer_pre_types tokenizer_pre_type	  = config.tokenizer_pre_type;
		static constexpr uint64_t kv_cache_block_size			  = config.kv_cache_block_size;
		static constexpr bool use_rotary_embeddings				  = config.use_rotary_embeddings;
		static constexpr rms_norm_types rms_norm_type			  = config.rms_norm_type;
		static constexpr tokenizer_types tokenizer_type			  = config.tokenizer_type;
		static constexpr device_types device_type				  = config.device_type;
		static constexpr model_formats model_format				  = config.model_format;
		static constexpr float norm_epsilon						  = config.norm_epsilon;
		static constexpr bool benchmark							  = config.benchmark;
		static constexpr bool dev								  = config.dev;

		NIHILUS_HOST static constexpr const model_config& get_config() {
			return config;
		}
	};

	template<typename... arg_types> NIHILUS_HOST static consteval auto generate_model_config(arg_types... args) {
		model_config config_new{};
		((config_new = config_new.update(args)), ...);
		return config_new;
	};

}
