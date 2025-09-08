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

#include <nihilus-incl/cpu/memory_mapped_file.hpp>
#include <nihilus-incl/common/parse_entity.hpp>
#include <nihilus-incl/infra/tokenizer.hpp>
#include <nihilus-incl/infra/core_traits.hpp>

namespace nihilus {

	enum class gguf_metadata_value_type : uint32_t {
		uint8	= 0,
		int8	= 1,
		uint16	= 2,
		int16	= 3,
		uint32	= 4,
		int32	= 5,
		float32 = 6,
		boolean = 7,
		string	= 8,
		array	= 9,
		uint64	= 10,
		int64	= 11,
		float64 = 12,
		unset	= 13,
	};

	enum class void_metadata_value_type : uint32_t {
		uint8	= 0,
		int8	= 1,
		uint16	= 2,
		int16	= 3,
		uint32	= 4,
		int32	= 5,
		float32 = 6,
		boolean = 7,
		string	= 8,
		array	= 9,
		uint64	= 10,
		int64	= 11,
		float64 = 12,
		unset	= 13,
	};

	struct void_metadata {
		uint32_t magic{};
		uint64_t version{};
		uint64_t thread_count{};
		uint64_t alignment{};
		uint64_t scalar_metadata_offset{};
		uint64_t scalar_metadata_length{};
		uint64_t scalar_offset{};
		uint64_t scalar_length{};
		uint64_t tensor_metadata_offset{};
		uint64_t tensor_metadata_length{};
		uint64_t tensor_offset{};
		uint64_t tensor_length{};
	};

	template<typename value_type> struct void_scalar_metadata {
		std::string_view name{};
		uint64_t offset_into_metadata_data{};
		void_metadata_value_type type{};
		value_type value{};
		uint64_t length{};
	};

	enum class void_device_types { cpu = 0, gpu = 1 };

	struct void_tensor_metadata {
		std::string_view name{};
		uint64_t offset_into_tensor_data{};
		void_device_types device_type{};
		uint64_t offset_per_thread{};
		array<uint64_t, 4> dims{};
		int64_t layer_number{};// -1 for "Per-Model" tensors.
		data_types type{};
	};

	struct void_model_file {
		void_metadata metadata{};
		aligned_vector<void_scalar_metadata<void*>> scalar_metadata{};
		aligned_vector<void_tensor_metadata> tensor_metadata{};
	};

	template<model_config config> struct stream_iterator {
		memory_mapped_file<config>* file{};
		uint64_t current_index = 0;
		uint64_t length		   = 0;
		bool valid			   = true;

		NIHILUS_INLINE stream_iterator(memory_mapped_file<config>* s) : file(s), length{ file->size() } {
		}

		template<typename value_type> NIHILUS_INLINE value_type read() {
			char values[sizeof(value_type)];
			std::copy_n(static_cast<char*>(file->data()) + current_index, sizeof(value_type), values);
			value_type dst = std::bit_cast<value_type>(values);
			current_index += sizeof(value_type);
			return dst;
		}

		NIHILUS_INLINE bool map_pointer(void* dst, const uint64_t offset, uint64_t count = 0) {
			*static_cast<void**>(dst) = static_cast<uint8_t*>(file->data()) + offset;
			return true;
		}

		template<typename value_type = uint8_t> NIHILUS_INLINE bool has_bytes(uint64_t size = sizeof(value_type)) const {
			return (current_index + size <= length);
		}
	};

#if NIHILUS_CUDA_ENABLED
	template<model_config config>
		requires(config.device_type == device_types::gpu)
	struct stream_iterator<config> {
		memory_mapped_file<config>* file{};
		uint64_t current_index = 0;
		uint64_t length		   = 0;
		bool valid			   = true;

		NIHILUS_INLINE stream_iterator(memory_mapped_file<config>* s) : file(s), length{ file->size() } {
		}

		template<typename value_type> NIHILUS_INLINE value_type read() {
			char values[sizeof(value_type)];
			std::copy_n(static_cast<char*>(file->data()) + current_index, sizeof(value_type), values);
			value_type dst = std::bit_cast<value_type>(values);
			current_index += sizeof(value_type);
			return dst;
		}

		NIHILUS_INLINE bool map_pointer(void* dst, const uint64_t offset, uint64_t count = 0) {
			memory_transfer<config>::host_to_device(static_cast<uint8_t*>(file->data()) + offset, static_cast<uint8_t*>(*static_cast<void**>(dst)), count);
			return true;
		}

		template<typename value_type = uint8_t> NIHILUS_INLINE bool has_bytes(uint64_t size = sizeof(value_type)) const {
			return (current_index + size <= length);
		}
	};
#endif

	template<model_config config, typename value_type, auto...> struct value_reader;

	template<model_config config, typename value_type>
		requires(( std::is_standard_layout_v<value_type> && std::is_trivial_v<value_type> && !std::is_enum_v<value_type> ) || std::is_same_v<gguf_metadata_value_type, value_type>)
	struct value_reader<config, value_type> {
		NIHILUS_INLINE static value_type gather_value(stream_iterator<config>& input) {
			if (input.template has_bytes<value_type>()) {
				return input.template read<value_type>();
			} else {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Sorry, but that index is out of range!", location>::impl();
				return {};
			}
		}
	};

	template<typename enum_type> struct string_to_enum;

	template<> struct string_to_enum<tokenizer_types> {
		NIHILUS_INLINE static tokenizer_types impl(std::string_view value) {
			if (string_literal_comparitor<"gpt2">::impl(value.data())) {
				return tokenizer_types::bpe;
			}

			return tokenizer_types::none;
		}
	};

	template<> struct string_to_enum<tokenizer_pre_types> {
		NIHILUS_INLINE static tokenizer_pre_types impl(std::string_view value) {
			if (string_literal_comparitor<"default">::impl(value.data())) {
				return tokenizer_pre_types::default_pre;
			}
			if (string_literal_comparitor<"llama3">::impl(value.data()) || string_literal_comparitor<"llama-v3">::impl(value.data()) ||
				string_literal_comparitor<"llama-bpe">::impl(value.data()) || string_literal_comparitor<"falcon3">::impl(value.data())) {
				return tokenizer_pre_types::llama3;
			}
			if (string_literal_comparitor<"deepseek-llm">::impl(value.data())) {
				return tokenizer_pre_types::deepseek_llm;
			}
			if (string_literal_comparitor<"deepseek-coder">::impl(value.data())) {
				return tokenizer_pre_types::deepseek_coder;
			}
			if (string_literal_comparitor<"deepseek-v3">::impl(value.data())) {
				return tokenizer_pre_types::deepseek3_llm;
			}
			if (string_literal_comparitor<"falcon">::impl(value.data())) {
				return tokenizer_pre_types::falcon;
			}
			if (string_literal_comparitor<"mpt">::impl(value.data())) {
				return tokenizer_pre_types::mpt;
			}
			if (string_literal_comparitor<"starcoder">::impl(value.data())) {
				return tokenizer_pre_types::starcoder;
			}
			if (string_literal_comparitor<"gpt-2">::impl(value.data()) || string_literal_comparitor<"phi-2">::impl(value.data()) ||
				string_literal_comparitor<"jina-es">::impl(value.data()) || string_literal_comparitor<"jina-de">::impl(value.data()) ||
				string_literal_comparitor<"gigachat">::impl(value.data()) || string_literal_comparitor<"jina-v1-en">::impl(value.data()) ||
				string_literal_comparitor<"jina-v2-es">::impl(value.data()) || string_literal_comparitor<"jina-v2-de">::impl(value.data()) ||
				string_literal_comparitor<"jina-v2-code">::impl(value.data()) || string_literal_comparitor<"roberta-bpe">::impl(value.data())) {
				return tokenizer_pre_types::gpt2;
			}
			if (string_literal_comparitor<"refact">::impl(value.data())) {
				return tokenizer_pre_types::refact;
			}
			if (string_literal_comparitor<"command-r">::impl(value.data())) {
				return tokenizer_pre_types::command_r;
			}
			if (string_literal_comparitor<"stablelm2">::impl(value.data())) {
				return tokenizer_pre_types::stablelm2;
			}
			if (string_literal_comparitor<"qwen2">::impl(value.data()) || string_literal_comparitor<"deepseek-r1-qwen">::impl(value.data()) ||
				string_literal_comparitor<"megrez">::impl(value.data())) {
				return tokenizer_pre_types::qwen2;
			}
			if (string_literal_comparitor<"olmo">::impl(value.data())) {
				return tokenizer_pre_types::olmo;
			}
			if (string_literal_comparitor<"dbrx">::impl(value.data())) {
				return tokenizer_pre_types::dbrx;
			}
			if (string_literal_comparitor<"smaug-bpe">::impl(value.data())) {
				return tokenizer_pre_types::smaug;
			}
			if (string_literal_comparitor<"poro-chat">::impl(value.data())) {
				return tokenizer_pre_types::poro;
			}
			if (string_literal_comparitor<"chatglm-bpe">::impl(value.data())) {
				return tokenizer_pre_types::chatglm4;
			}
			if (string_literal_comparitor<"viking">::impl(value.data())) {
				return tokenizer_pre_types::viking;
			}
			if (string_literal_comparitor<"jais">::impl(value.data())) {
				return tokenizer_pre_types::jais;
			}
			if (string_literal_comparitor<"tekken">::impl(value.data())) {
				return tokenizer_pre_types::tekken;
			}
			if (string_literal_comparitor<"smollm">::impl(value.data())) {
				return tokenizer_pre_types::smollm;
			}
			if (string_literal_comparitor<"codeshell">::impl(value.data())) {
				return tokenizer_pre_types::codeshell;
			}
			if (string_literal_comparitor<"bloom">::impl(value.data())) {
				return tokenizer_pre_types::bloom;
			}
			if (string_literal_comparitor<"gpt3-finnish">::impl(value.data())) {
				return tokenizer_pre_types::gpt3_finnish;
			}
			if (string_literal_comparitor<"exaone">::impl(value.data())) {
				return tokenizer_pre_types::exaone;
			}
			if (string_literal_comparitor<"chameleon">::impl(value.data())) {
				return tokenizer_pre_types::chameleon;
			}
			if (string_literal_comparitor<"minerva-7b">::impl(value.data())) {
				return tokenizer_pre_types::minerva;
			}

			return tokenizer_pre_types::default_pre;
		}
	};

	template<model_config config, typename value_type>
		requires(std::is_enum_v<value_type> && !std::is_same_v<gguf_metadata_value_type, value_type>)
	struct value_reader<config, value_type> {
		NIHILUS_INLINE static value_type gather_value(stream_iterator<config>& input) {
			uint64_t length = value_reader<config, uint64_t>::gather_value(input);
			if (!input.template has_bytes<uint8_t>(length)) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Sorry, but that index is out of range!", location>::impl();
			}
			const char* string_ptr{ static_cast<const char*>(input.file->data()) + input.current_index };
			input.current_index += length;
			std::string_view result(string_ptr, length);
			return string_to_enum<value_type>::impl(result);
		}
	};

	template<model_config config, typename value_type>
		requires(is_specialization_v<value_type, std::unordered_map>)
	struct value_reader<config, value_type> {
		NIHILUS_INLINE static value_type gather_value(stream_iterator<config>& input) {
			value_reader<config, gguf_metadata_value_type>::gather_value(input);
			uint64_t length{ value_reader<config, uint64_t>::gather_value(input) };
			constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
			if (length > MAX_ARRAY_LENGTH) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Sorry, but that index is out of range!", location>::impl();
			}
			value_type value{};
			value.reserve(length);
			for (uint64_t x = 0; x < length; ++x) {
				value.emplace(value_reader<config, typename value_type::key_type>::gather_value(input), static_cast<typename value_type::mapped_type>(x));
			}
			return value;
		}
	};

	template<model_config config, typename value_type>
		requires(is_specialization_v<value_type, aligned_vector>)
	struct value_reader<config, value_type> {
		NIHILUS_INLINE static value_type gather_value(stream_iterator<config>& input) {
			value_reader<config, gguf_metadata_value_type>::gather_value(input);
			uint64_t length{ value_reader<config, uint64_t>::gather_value(input) };
			constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
			if (length > MAX_ARRAY_LENGTH) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Array length exceeds maximum allowed size!", location>::impl();
			}
			value_type value{};
			value.reserve(length);
			for (uint64_t x = 0; x < length; ++x) {
				value.emplace_back(value_reader<config, typename value_type::value_type>::gather_value(input));
			}
			return value;
		}
	};

	using gguf_string_t = std::string_view;

	template<model_config config> struct value_reader<config, gguf_string_t> {
		NIHILUS_INLINE static std::string_view gather_value(stream_iterator<config>& input) {
			uint64_t length = value_reader<config, uint64_t>::gather_value(input);
			if (!input.template has_bytes<uint8_t>(length)) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Sorry, but that index is out of range!", location>::impl();
			}
			const char* string_ptr{ static_cast<const char*>(input.file->data()) + input.current_index };
			input.current_index += length;
			std::string_view result(string_ptr, length);
			return result;
		}
	};

	struct metadata_base {
		aligned_vector<std::string_view> languages;
		std::string_view quantize_imatrix_dataset;
		int32_t quantize_imatrix_entries_count;
		std::string_view quantize_imatrix_file;
		aligned_vector<std::string_view> tags;
		int32_t quantize_imatrix_chunks_count;
		uint32_t quantization_version;
		std::string_view architecture;
		std::string_view size_label;
		uint64_t metadata_kv_count;
		std::string_view finetune;
		std::string_view license;
		uint64_t tensor_count;
		uint32_t file_type;
		uint32_t alignment;
	};

	template<tokenizer_types type> struct tokenizer_base;

	template<> struct tokenizer_base<tokenizer_types::bpe> {
		std::unordered_map<std::string_view, int32_t> ggml_tokens;
		aligned_vector<std::string_view> ggml_merges;
		aligned_vector<int32_t> ggml_token_type;
		std::string_view chat_template;
		std::string_view ggml_model;
	};

	template<model_config config> struct gguf_metadata : public metadata_base,
														 public tokenizer_base<config.tokenizer_type>,
														 public model_traits<config.arch, config.model_size, config.model_generation>,
														 public tokenizer_traits<config.arch, config.tokenizer_type, config.tokenizer_pre_type> {};

	template<model_config config> struct parse_core<gguf_metadata<config>> {
		using value_type				  = gguf_metadata<config>;
		static constexpr auto parse_value = create_value<make_parse_entity<&value_type::architecture, "general.architecture">(),
			make_parse_entity<&value_type::finetune, "general.finetune">(), make_parse_entity<&value_type::size_label, "general.size_label">(),
			make_parse_entity<&value_type::license, "general.license">(), make_parse_entity<&value_type::tags, "general.tags">(),
			make_parse_entity<&value_type::languages, "general.languages">(), make_parse_entity<&value_type::block_count, "llama.block_count">(),
			make_parse_entity<&value_type::context_length, "llama.context_length">(), make_parse_entity<&value_type::embedding_length, "llama.embedding_length">(),
			make_parse_entity<&value_type::feed_forward_length, "llama.feed_forward_length">(),
			make_parse_entity<&value_type::attention_head_count, "llama.attention.head_count">(),
			make_parse_entity<&value_type::attention_head_count_kv, "llama.attention.head_count_kv">(), make_parse_entity<&value_type::rope_freq_base, "llama.rope.freq_base">(),
			make_parse_entity<&value_type::layer_norm_rms_epsilon, "llama.attention.layer_norm_rms_epsilon">(), make_parse_entity<&value_type::file_type, "general.file_type">(),
			make_parse_entity<&value_type::vocab_size, "llama.vocab_size">(), make_parse_entity<&value_type::rope_dimension_count, "llama.rope.dimension_count">(),
			make_parse_entity<&value_type::type, "tokenizer.ggml.model">(), make_parse_entity<&value_type::pre_type, "tokenizer.ggml.pre">(),
			make_parse_entity<&value_type::ggml_tokens, "tokenizer.ggml.tokens">(), make_parse_entity<&value_type::ggml_token_type, "tokenizer.ggml.token_type">(),
			make_parse_entity<&value_type::ggml_merges, "tokenizer.ggml.merges">(), make_parse_entity<&value_type::ggml_model, "tokenizer.ggml.model">(),
			make_parse_entity<&value_type::special_bos_id, "tokenizer.ggml.bos_token_id">(), make_parse_entity<&value_type::special_eos_id, "tokenizer.ggml.eos_token_id">(),
			make_parse_entity<&value_type::chat_template, "tokenizer.chat_template">(), make_parse_entity<&value_type::quantization_version, "general.quantization_version">(),
			make_parse_entity<&value_type::quantize_imatrix_file, "quantize.imatrix.file">(),
			make_parse_entity<&value_type::quantize_imatrix_dataset, "quantize.imatrix.dataset">(),
			make_parse_entity<&value_type::quantize_imatrix_entries_count, "quantize.imatrix.entries_count">(),
			make_parse_entity<&value_type::quantize_imatrix_chunks_count, "quantize.imatrix.chunks_count">()>();
	};

	template<typename value_type01, typename value_type02>
	concept is_comparable = requires() { value_type01{} != value_type02{}; };

	template<typename value_type01, is_comparable<value_type01> value_type02> NIHILUS_INLINE bool compare_equal(const value_type01& value01, const value_type02& value02) noexcept {
		if constexpr (std::is_floating_point_v<value_type01>) {
			constexpr value_type01 epsilon = std::numeric_limits<value_type01>::epsilon() * 10;
			return std::abs(value01 - value02) <= epsilon;
		} else {
			return value01 == value02;
		}
	}

	template<model_config config, typename value_type> struct parse_types_impl {
		inline static constexpr auto memberCount = core_tuple_size<value_type>;

		template<uint64_t index> using member_type_t =
			std::remove_reference_t<decltype(get_member<get<index>(parse_core<value_type>::parse_value).member_ptr>(std::declval<value_type&>()))>;

		template<uint64_t index> NIHILUS_INLINE static void process_index(value_type& value, std::string_view string, stream_iterator<config>& stream) {
			static constexpr auto tupleElem	 = get<index>(parse_core<value_type>::parse_value);
			static constexpr auto string_lit = tupleElem.name;
			static constexpr auto ptrNew	 = tupleElem.member_ptr;
			static constexpr auto keySize	 = string_lit.size();
			if NIHILUS_LIKELY ((string.size() <= keySize) && string_literal_comparitor<string_lit>::impl(string.data())) {
				auto& ref = get_member<ptrNew>(value);
				if constexpr (!std::is_const_v<std::remove_reference_t<decltype(ref)>>) {
					ref = value_reader<config, member_type_t<index>>::gather_value(stream);
				} else {
					member_type_t<index> value_new{ value_reader<config, detail::remove_const_t<member_type_t<index>>>::gather_value(stream) };
					if (!compare_equal(value_new, ref)) {
						static constexpr string_literal sl_new{ "Sorry, but member of name: " + string_lit + " was not equal!" };
						static constexpr auto location = std::source_location::current();
						nihilus_exception<config, sl_new, location>::impl();
						return;
					}
				}
				( void )ref;
			}
		}
	};

	template<template<model_config, typename> typename parsing_type, model_config config, typename value_type, size_t... indices>
	inline static constexpr auto generate_function_ptrs(std::index_sequence<indices...>) noexcept {
		using function_type = decltype(&parse_types_impl<config, value_type>::template process_index<0>);
		return array<function_type, sizeof...(indices)>{ { &parsing_type<config, value_type>::template process_index<indices>... } };
	}

	template<template<model_config, typename> typename parsing_type, model_config config, typename value_type>
	static constexpr auto function_ptrs{ generate_function_ptrs<parsing_type, config, value_type>(std::make_index_sequence<core_tuple_size<value_type>>{}) };

	template<model_config config, uint64_t current_index = 0, uint64_t max_index = 10, typename enum_type>
		requires(std::is_same_v<gguf_metadata_value_type, enum_type>)
	NIHILUS_INLINE void calculate_and_skip_unknown_value(stream_iterator<config>& input, enum_type type) {
		switch (static_cast<uint64_t>(type)) {
			case static_cast<uint64_t>(enum_type::uint8):
			case static_cast<uint64_t>(enum_type::int8):
			case static_cast<uint64_t>(enum_type::boolean): {
				input.current_index += 1;
				break;
			}
			case static_cast<uint64_t>(enum_type::uint16):
			case static_cast<uint64_t>(enum_type::int16): {
				input.current_index += 2;
				break;
			}
			case static_cast<uint64_t>(enum_type::uint32):
			case static_cast<uint64_t>(enum_type::int32):
			case static_cast<uint64_t>(enum_type::float32): {
				input.current_index += 4;
				break;
			}
			case static_cast<uint64_t>(enum_type::uint64):
			case static_cast<uint64_t>(enum_type::int64):
			case static_cast<uint64_t>(enum_type::float64): {
				input.current_index += 8;
				break;
			}
			case static_cast<uint64_t>(enum_type::string): {
				if (!input.template has_bytes<uint64_t>()) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Insufficient bytes for string length!", location>::impl();
					return;
				}
				uint64_t string_length = input.template read<uint64_t>();

				if (!input.template has_bytes<uint8_t>(string_length)) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Insufficient bytes for string content!", location>::impl();
					return;
				}
				input.current_index += string_length;
				break;
			}
			case static_cast<uint64_t>(enum_type::array): {
				if (!input.template has_bytes<gguf_metadata_value_type>()) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Insufficient bytes for array type!", location>::impl();
					return;
				}
				gguf_metadata_value_type array_types = input.template read<gguf_metadata_value_type>();

				if (!input.template has_bytes<uint64_t>()) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Insufficient bytes for array length!", location>::impl();
					return;
				}
				uint64_t array_length = input.template read<uint64_t>();

				constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
				if (array_length > MAX_ARRAY_LENGTH) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Array length exceeds maximum allowed size during skip!", location>::impl();
					return;
				}

				for (uint64_t i = 0; i < array_length; ++i) {
					if constexpr (current_index < max_index) {
						calculate_and_skip_unknown_value<config, current_index + 1>(input, array_types);
					}
				}
				break;
			}
			default: {
				return;
			}
		}
	}

	template<model_config config> struct value_reader<config, gguf_metadata<config>> {
		NIHILUS_INLINE static gguf_metadata<config> gather_value(stream_iterator<config>& input) {
			gguf_metadata<config> value{};
			uint32_t magic = value_reader<config, uint32_t>::gather_value(input);
			if (magic != 0x46554747) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Sorry, but that magic value was incorrect!", location>::impl();
			}
			value_reader<config, uint32_t>::gather_value(input);
			value.tensor_count		= value_reader<config, uint64_t>::gather_value(input);
			value.metadata_kv_count = value_reader<config, uint64_t>::gather_value(input);

			static constexpr uint64_t MAX_TENSOR_COUNT	 = 100000;
			static constexpr uint64_t MAX_METADATA_COUNT = 10000;

			if (value.tensor_count > MAX_TENSOR_COUNT) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Tensor count exceeds reasonable maximum!", location>::impl();
			}
			if (value.metadata_kv_count > MAX_METADATA_COUNT) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Metadata count exceeds reasonable maximum!", location>::impl();
			}

			for (uint64_t x = 0; x < value.metadata_kv_count; ++x) {
				std::string_view new_string			= value_reader<config, gguf_string_t>::gather_value(input);
				gguf_metadata_value_type value_type = value_reader<config, gguf_metadata_value_type>::gather_value(input);
				auto index							= hash_map<gguf_metadata<config>, const char*>::find_index(new_string.data(), new_string.data() + new_string.size());
				if (index < function_ptrs<parse_types_impl, config, gguf_metadata<config>>.size()) {
					function_ptrs<parse_types_impl, config, gguf_metadata<config>>[index](value, new_string, input);
				} else {
					calculate_and_skip_unknown_value<config>(input, value_type);
				}
			}
			return value;
		}
	};

	template<model_arches arch> struct string_to_op_type;

	template<> struct string_to_op_type<model_arches::llama> {
		NIHILUS_INLINE static weight_types impl(std::string_view input) noexcept {
			if (string_literal_comparitor<"token_embd.weight">::impl(input.data())) {
				return weight_types::token_embd;
			}
			if (string_literal_comparitor<"rope_freqs.weight">::impl(input.data())) {
				return weight_types::rope_freqs;
			}
			if (string_literal_comparitor<"output_norm.weight">::impl(input.data())) {
				return weight_types::output_norm;
			}
			if (string_literal_comparitor<"output.weight">::impl(input.data())) {
				return weight_types::output;
			}

			if (string_literal_comparitor<".attn_q.weight">::impl(input.data() + input.find(".attn_q.weight"))) {
				return weight_types::attn_q;
			}
			if (string_literal_comparitor<".attn_norm.weight">::impl(input.data() + input.find(".attn_norm.weight"))) {
				return weight_types::attn_norm;
			}

			if (string_literal_comparitor<"blk.">::impl(input.data()) && string_literal_comparitor<".weight">::impl(input.data() + input.size() - 7)) {
				auto second_dot = input.find('.', 4);
				if (second_dot != std::string_view::npos) {
					auto suffix = input.substr(second_dot + 1);

					if (string_literal_comparitor<"attn_q.weight">::impl(suffix.data())) {
						return weight_types::attn_q;
					}
					if (string_literal_comparitor<"attn_norm.weight">::impl(suffix.data())) {
						return weight_types::attn_norm;
					}
					if (string_literal_comparitor<"attn_k.weight">::impl(suffix.data())) {
						return weight_types::attn_k;
					}
					if (string_literal_comparitor<"attn_v.weight">::impl(suffix.data())) {
						return weight_types::attn_v;
					}
					if (string_literal_comparitor<"attn_output.weight">::impl(suffix.data())) {
						return weight_types::attn_output;
					}
					if (string_literal_comparitor<"ffn_down.weight">::impl(suffix.data())) {
						return weight_types::ffn_down;
					}
					if (string_literal_comparitor<"ffn_gate.weight">::impl(suffix.data())) {
						return weight_types::ffn_gate;
					}
					if (string_literal_comparitor<"ffn_up.weight">::impl(suffix.data())) {
						return weight_types::ffn_up;
					}
					if (string_literal_comparitor<"ffn_norm.weight">::impl(suffix.data())) {
						return weight_types::ffn_norm;
					}
				}
			}

			return weight_types::count;
		}
	};

	struct core_base_creation_data {
		array<uint64_t, 4> dimensions{ { 1, 1, 1, 1 } };
		uint32_t n_dimensions{};
		uint64_t layer_number{};
		weight_types op_type{};
		uint64_t offset{};
		data_types type{};

		NIHILUS_INLINE uint64_t total_dims() const {
			return dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3];
		}

		NIHILUS_INLINE uint64_t total_byte_size() const {
			uint64_t total_elements = total_dims();
			uint64_t block_size_val = block_size();
			uint64_t type_size_val	= type_size();
			uint64_t num_blocks		= (total_elements + block_size_val - 1) / block_size_val;
			return num_blocks * type_size_val;
		}

		template<typename op_traits> NIHILUS_INLINE bool operator==(const op_traits&) const {
			static constexpr auto other_dims = op_traits::get_array();
			return dimensions[0] == other_dims[0] && dimensions[1] == other_dims[1] && dimensions[2] == other_dims[2] && dimensions[3] == other_dims[3];
		}

		NIHILUS_INLINE uint64_t block_size() const {
			return get_type_traits(type).block_size;
		}

		NIHILUS_INLINE uint64_t type_size() const {
			return get_type_traits(type).type_size;
		}
	};

	template<model_config, model_arches> struct core_traits_comparitor;

	template<model_config config> struct core_traits_comparitor<config, model_arches::llama> {
		NIHILUS_INLINE static bool impl(const core_base_creation_data& parse_core) noexcept {
			switch (static_cast<uint64_t>(parse_core.op_type)) {
				case static_cast<uint64_t>(weight_types::token_embd): {
					return parse_core == typename core_traits<config, core_types::weights>::token_embd_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::rope_freqs): {
					return parse_core == typename core_traits<config, core_types::weights>::rope_freqs_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::output_norm): {
					return parse_core == typename core_traits<config, core_types::weights>::output_norm_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::output): {
					return parse_core == typename core_traits<config, core_types::weights>::output_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::attn_q): {
					return parse_core == typename core_traits<config, core_types::weights>::attn_q_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::attn_norm): {
					return parse_core == typename core_traits<config, core_types::weights>::attn_norm_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::attn_k): {
					return parse_core == typename core_traits<config, core_types::weights>::attn_k_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::attn_v): {
					return parse_core == typename core_traits<config, core_types::weights>::attn_v_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::attn_output): {
					return parse_core == typename core_traits<config, core_types::weights>::attn_output_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::ffn_down): {
					return parse_core == typename core_traits<config, core_types::weights>::ffn_down_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::ffn_gate): {
					return parse_core == typename core_traits<config, core_types::weights>::ffn_gate_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::ffn_up): {
					return parse_core == typename core_traits<config, core_types::weights>::ffn_up_weight_type{};
				}
				case static_cast<uint64_t>(weight_types::ffn_norm): {
					return parse_core == typename core_traits<config, core_types::weights>::ffn_norm_weight_type{};
				}
				default: {
					return false;
				}
			}
		}
	};

	NIHILUS_INLINE constexpr uint64_t parse_number(std::string_view str) noexcept {
		int64_t result = 0;
		for (char c: str) {
			if (c >= '0' && c <= '9') {
				result = result * 10 + (c - '0');
			} else {
				break;
			}
		}
		return static_cast<uint64_t>(result);
	}

	NIHILUS_INLINE uint64_t extract_layer_number(std::string_view name) noexcept {
		if NIHILUS_LIKELY (name[0] == 'b' && name.starts_with("blk.")) {
			uint64_t start = 4;
			uint64_t end   = name.find('.', start);
			if (end != std::string_view::npos) {
				return parse_number(name.substr(start, end - start));
			}
		}
		if NIHILUS_LIKELY (name[0] == 'c' && name.starts_with("cache_")) {
			for (uint64_t i = 7; i < name.size(); ++i) {
				if (name[i] == 'l' && i + 1 < name.size()) {
					return parse_number(name.substr(i + 1));
				}
			}
		}
		return 0;
	}

	template<model_config config> struct value_reader<config, core_base_creation_data> {
		NIHILUS_INLINE static core_base_creation_data gather_value(stream_iterator<config>& input) {
			core_base_creation_data value{};
			std::string_view name{ value_reader<config, std::string_view>::gather_value(input) };
			value.op_type					  = string_to_op_type<config.arch>::impl(name);
			value.n_dimensions				  = value_reader<config, uint32_t>::gather_value(input);
			value.layer_number				  = extract_layer_number(name);
			constexpr uint32_t MAX_DIMENSIONS = 4;
			if (value.n_dimensions > MAX_DIMENSIONS) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Tensor dimensions exceed maximum!", location>::impl();
			}
			for (uint64_t x = 0; x < value.n_dimensions; ++x) {
				uint64_t dim					= value_reader<config, uint64_t>::gather_value(input);
				constexpr uint64_t MAX_DIM_SIZE = 1ULL << 32;
				if (dim > MAX_DIM_SIZE) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Tensor dimensions size too large!", location>::impl();
				}
				value.dimensions[x] = dim;
			}
			value.type	 = static_cast<data_types>(value_reader<config, uint32_t>::gather_value(input));
			value.offset = value_reader<config, uint64_t>::gather_value(input);
			return value;
		}
	};

	NIHILUS_INLINE bool operator<(const core_base_creation_data& lhs, const core_base_creation_data& rhs) noexcept {
		return lhs.layer_number < rhs.layer_number;
	}

	NIHILUS_INLINE void sort_tensor_infos(aligned_vector<core_base_creation_data>& tensor_infos) noexcept {
		std::sort(tensor_infos.begin(), tensor_infos.end(), std::less<core_base_creation_data>{});
	}

	NIHILUS_INLINE uint64_t align_offset(uint64_t offset, uint64_t alignment) {
		alignment = alignment == 0 ? 1 : alignment;
		return offset + (alignment - (offset % alignment)) % alignment;
	}

	template<model_config config> struct model_parser_impl {};

	template<model_config config>
		requires((config.arch == model_arches::llama) && (config.format == model_format::gguf))
	struct model_parser_impl<config> {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		static_assert((std::endian::native == std::endian::little), "Sorry, but big-endian is not yet supported by the library");
		template<typename tokenizer_type> NIHILUS_INLINE static gguf_metadata<config> parse_model(array<array<void*, model_traits_type::block_count>, weight_types::count>& data,
			memory_mapped_file<config>* memory_file, tokenizer_type& tokenizer) {
			stream_iterator<config> ptr{ memory_file };
			gguf_metadata<config> gguf_file{ value_reader<config, gguf_metadata<config>>::gather_value(ptr) };
			tokenizer.tokens		= detail::move(gguf_file.ggml_tokens);
			tokenizer.merges		= detail::move(gguf_file.ggml_merges);
			tokenizer.token_types	= detail::move(gguf_file.ggml_token_type);
			tokenizer.chat_template = detail::move(gguf_file.chat_template);
			aligned_vector<core_base_creation_data> tensor_infos{};
			tensor_infos.reserve(gguf_file.tensor_count);
			for (uint64_t x = 0; x < gguf_file.tensor_count; ++x) {
				auto new_tensor{ value_reader<config, core_base_creation_data>::gather_value(ptr) };
				if (!core_traits_comparitor<config, model_arches::llama>::impl(new_tensor)) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "Tensor dimensions incorrect!", location>::impl();
				}
				tensor_infos.emplace_back(new_tensor);
			}
			uint64_t max_tensor_end = 0;
			for (const auto& tensor: tensor_infos) {
				uint64_t tensor_size = tensor.total_byte_size();
				uint64_t tensor_end	 = tensor.offset + tensor_size;
				max_tensor_end		 = detail::max(max_tensor_end, tensor_end);
			}

			uint64_t tensor_data_start = ptr.file->size() - max_tensor_end;

			sort_tensor_infos(tensor_infos);
			for (uint64_t x = 0; x < gguf_file.tensor_count; ++x) {
				uint64_t absolute_offset = align_offset(tensor_data_start + tensor_infos[x].offset, gguf_file.alignment);
				ptr.map_pointer(data[tensor_infos[x].op_type][tensor_infos[x].layer_number], absolute_offset, tensor_infos[x].total_byte_size());
			}
			return gguf_file;
		}
	};

	template<model_config config> struct model_parser {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

		template<typename tokenizer_type> NIHILUS_INLINE static gguf_metadata<config> parse_model(array<array<void*, model_traits_type::block_count>, weight_types::count>& data,
			memory_mapped_file<config>* memory_file, tokenizer_type& tokenizer) {
			return model_parser_impl<config>::parse_model(data, memory_file, tokenizer);
		}
	};
}
