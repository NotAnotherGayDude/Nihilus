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

#include <nihilus/common/memory_mapped_file.hpp>
#include <nihilus/common/parse_entity.hpp>
#include <nihilus/common/tokenizer.hpp>
#include <nihilus/common/core_traits.hpp>

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

	struct stream_iterator {
		memory_mapped_file* file{};
		uint64_t current_index = 0;
		uint64_t length		   = 0;
		bool valid			   = true;

		NIHILUS_FORCE_INLINE stream_iterator(memory_mapped_file* s) : file(s), length{ file->size() } {};

		template<typename value_type> NIHILUS_FORCE_INLINE value_type read() {
			value_type dst{};
			std::memcpy(&dst, static_cast<uint8_t*>(file->data()) + current_index, sizeof(value_type));
			current_index += sizeof(value_type);
			return dst;
		}

		NIHILUS_FORCE_INLINE bool read_bytes_to_pointer(void* dst, const uint64_t size) {
			std::memcpy(dst, static_cast<uint8_t*>(file->data()) + current_index, size);
			current_index += size;
			return true;
		}

		NIHILUS_FORCE_INLINE bool map_pointer(void* dst, const uint64_t offset) {
			*reinterpret_cast<void**>(dst) = reinterpret_cast<uint8_t*>(file->data()) + offset;
			return true;
		}

		template<typename value_type = uint8_t> NIHILUS_FORCE_INLINE bool has_bytes(uint64_t size = sizeof(value_type)) const {
			return (current_index + size <= length);
		}
	};

	template<model_config config, typename value_type, auto...> struct value_reader;

	template<model_config config, typename value_type>
		requires(std::is_pod_v<value_type> && !std::is_enum_v<value_type> || std::is_same_v<gguf_metadata_value_type, value_type>)
	struct value_reader<config, value_type> {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			if (input.has_bytes<value_type>()) {
				return input.read<value_type>();
			} else {
				static constexpr auto location = get_source_location();
				return status_handler<config, value_type>::template construct_status<"Sorry, but that index is out of range!", location, success_statuses::fail>();
			}
		}
	};

	template<typename enum_type> struct string_to_enum;

	template<> struct string_to_enum<tokenizer_types> {
		NIHILUS_FORCE_INLINE static tokenizer_types impl(std::string_view value) {
			if (string_literal_comparison<"gpt2">(value.data())) {
				return tokenizer_types::bpe;
			}

			return tokenizer_types::none;
		}
	};

	template<> struct string_to_enum<tokenizer_pre_types> {
		NIHILUS_FORCE_INLINE static tokenizer_pre_types impl(std::string_view value) {
			if (string_literal_comparison<"default">(value.data())) {
				return tokenizer_pre_types::default_pre;
			}
			if (string_literal_comparison<"llama3">(value.data()) || string_literal_comparison<"llama-v3">(value.data()) || string_literal_comparison<"llama-bpe">(value.data()) ||
				string_literal_comparison<"falcon3">(value.data())) {
				return tokenizer_pre_types::llama3;
			}
			if (string_literal_comparison<"deepseek-llm">(value.data())) {
				return tokenizer_pre_types::deepseek_llm;
			}
			if (string_literal_comparison<"deepseek-coder">(value.data())) {
				return tokenizer_pre_types::deepseek_coder;
			}
			if (string_literal_comparison<"deepseek-v3">(value.data())) {
				return tokenizer_pre_types::deepseek3_llm;
			}
			if (string_literal_comparison<"falcon">(value.data())) {
				return tokenizer_pre_types::falcon;
			}
			if (string_literal_comparison<"mpt">(value.data())) {
				return tokenizer_pre_types::mpt;
			}
			if (string_literal_comparison<"starcoder">(value.data())) {
				return tokenizer_pre_types::starcoder;
			}
			if (string_literal_comparison<"gpt-2">(value.data()) || string_literal_comparison<"phi-2">(value.data()) || string_literal_comparison<"jina-es">(value.data()) ||
				string_literal_comparison<"jina-de">(value.data()) || string_literal_comparison<"gigachat">(value.data()) ||
				string_literal_comparison<"jina-v1-en">(value.data()) || string_literal_comparison<"jina-v2-es">(value.data()) ||
				string_literal_comparison<"jina-v2-de">(value.data()) || string_literal_comparison<"jina-v2-code">(value.data()) ||
				string_literal_comparison<"roberta-bpe">(value.data())) {
				return tokenizer_pre_types::gpt2;
			}
			if (string_literal_comparison<"refact">(value.data())) {
				return tokenizer_pre_types::refact;
			}
			if (string_literal_comparison<"command-r">(value.data())) {
				return tokenizer_pre_types::command_r;
			}
			if (string_literal_comparison<"stablelm2">(value.data())) {
				return tokenizer_pre_types::stablelm2;
			}
			if (string_literal_comparison<"qwen2">(value.data()) || string_literal_comparison<"deepseek-r1-qwen">(value.data()) ||
				string_literal_comparison<"megrez">(value.data())) {
				return tokenizer_pre_types::qwen2;
			}
			if (string_literal_comparison<"olmo">(value.data())) {
				return tokenizer_pre_types::olmo;
			}
			if (string_literal_comparison<"dbrx">(value.data())) {
				return tokenizer_pre_types::dbrx;
			}
			if (string_literal_comparison<"smaug-bpe">(value.data())) {
				return tokenizer_pre_types::smaug;
			}
			if (string_literal_comparison<"poro-chat">(value.data())) {
				return tokenizer_pre_types::poro;
			}
			if (string_literal_comparison<"chatglm-bpe">(value.data())) {
				return tokenizer_pre_types::chatglm4;
			}
			if (string_literal_comparison<"viking">(value.data())) {
				return tokenizer_pre_types::viking;
			}
			if (string_literal_comparison<"jais">(value.data())) {
				return tokenizer_pre_types::jais;
			}
			if (string_literal_comparison<"tekken">(value.data())) {
				return tokenizer_pre_types::tekken;
			}
			if (string_literal_comparison<"smollm">(value.data())) {
				return tokenizer_pre_types::smollm;
			}
			if (string_literal_comparison<"codeshell">(value.data())) {
				return tokenizer_pre_types::codeshell;
			}
			if (string_literal_comparison<"bloom">(value.data())) {
				return tokenizer_pre_types::bloom;
			}
			if (string_literal_comparison<"gpt3-finnish">(value.data())) {
				return tokenizer_pre_types::gpt3_finnish;
			}
			if (string_literal_comparison<"exaone">(value.data())) {
				return tokenizer_pre_types::exaone;
			}
			if (string_literal_comparison<"chameleon">(value.data())) {
				return tokenizer_pre_types::chameleon;
			}
			if (string_literal_comparison<"minerva-7b">(value.data())) {
				return tokenizer_pre_types::minerva;
			}

			return tokenizer_pre_types::default_pre;
		}
	};

	template<model_config config, typename value_type>
		requires(std::is_enum_v<value_type> && !std::is_same_v<gguf_metadata_value_type, value_type>)
	struct value_reader<config, value_type> {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			uint64_t length = value_reader<config, uint64_t>::gather_value(input);
			if (!input.has_bytes<uint8_t>(length)) {
				static constexpr auto location = get_source_location();
				return status_handler<config, value_type>::template construct_status<"Sorry, but that index is out of range!", location, success_statuses::fail>();
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
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			gguf_metadata_value_type type{ value_reader<config, gguf_metadata_value_type>::gather_value(input) };
			uint64_t length{ value_reader<config, uint64_t>::gather_value(input) };
			constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
			if (length > MAX_ARRAY_LENGTH) {
				static constexpr auto location = get_source_location();
				return status_handler<config, value_type>::template construct_status<"Sorry, but that index is out of range!", location, success_statuses::fail>();
			}
			value_type value{};
			value.reserve(length);
			for (uint64_t x = 0; x < length; ++x) {
				value[value_reader<config, typename value_type::key_type>::gather_value(input)] = x;
			}
			return value;
		}
	};

	template<model_config config, typename value_type>
		requires(is_specialization_v<value_type, std::vector>)
	struct value_reader<config, value_type> {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			gguf_metadata_value_type type{ value_reader<config, gguf_metadata_value_type>::gather_value(input) };
			uint64_t length{ value_reader<config, uint64_t>::gather_value(input) };
			constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
			if (length > MAX_ARRAY_LENGTH) {
				static constexpr auto location = get_source_location();
				return status_handler<config, value_type>::template construct_status<"Array length exceeds maximum allowed size!", location, success_statuses::fail>();
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
		NIHILUS_FORCE_INLINE static std::string_view gather_value(stream_iterator& input) {
			uint64_t length = value_reader<config, uint64_t>::gather_value(input);
			if (!input.has_bytes<uint8_t>(length)) {
				static constexpr auto location = get_source_location();
				return status_handler<config, gguf_string_t>::template construct_status<"Sorry, but that index is out of range!", location, success_statuses::fail>();
			}
			const char* string_ptr{ static_cast<const char*>(input.file->data()) + input.current_index };
			input.current_index += length;
			std::string_view result(string_ptr, length);
			return result;
		}
	};

	struct metadata_base {
		std::string_view quantize_imatrix_dataset;
		std::vector<std::string_view> languages;
		int32_t quantize_imatrix_entries_count;
		std::string_view quantize_imatrix_file;
		int32_t quantize_imatrix_chunks_count;
		std::vector<std::string_view> tags;
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
		std::vector<std::string_view> ggml_merges;
		std::vector<int32_t> ggml_token_type;
		std::string_view chat_template;
		std::string_view ggml_model;
	};

	template<model_config config> struct gguf_metadata : public metadata_base,
														 public tokenizer_base<config.tokenizer_type>,
														 public model_traits<config.arch, config.model_size, config.model_generation>,
														 public tokenizer_traits<config.arch, config.tokenizer_type, config.tokenizer_pre_type> {};

	template<model_config config> struct parse_core<gguf_metadata<config>> {
		using value_type = gguf_metadata<config>;
		static constexpr auto parse_value =
			create_value<make_parse_entity<&value_type::architecture, "general.architecture">(), make_parse_entity<&value_type::finetune, "general.finetune">(),
				make_parse_entity<&value_type::size_label, "general.size_label">(), make_parse_entity<&value_type::license, "general.license">(),
				make_parse_entity<&value_type::tags, "general.tags">(), make_parse_entity<&value_type::languages, "general.languages">(),
				make_parse_entity<&value_type::block_count, "llama.block_count">(), make_parse_entity<&value_type::context_length, "llama.context_length">(),
				make_parse_entity<&value_type::embedding_length, "llama.embedding_length">(), make_parse_entity<&value_type::feed_forward_length, "llama.feed_forward_length">(),
				make_parse_entity<&value_type::attention_head_count, "llama.attention.attention_head_count">(),
				make_parse_entity<&value_type::attention_head_count_kv, "llama.attention.attention_head_count_kv">(),
				make_parse_entity<&value_type::rope_freq_base, "llama.rope.freq_base">(),
				make_parse_entity<&value_type::layer_norm_rms_epsilon, "llama.attention.layer_norm_rms_epsilon">(),
				make_parse_entity<&value_type::file_type, "general.file_type">(), make_parse_entity<&value_type::vocab_size, "llama.vocab_size">(),
				make_parse_entity<&value_type::rope_dimension_count, "llama.rope.dimension_count">(), make_parse_entity<&value_type::type, "tokenizer.ggml.model">(),
				make_parse_entity<&value_type::pre_type, "tokenizer.ggml.pre">(), make_parse_entity<&value_type::ggml_tokens, "tokenizer.ggml.tokens">(),
				make_parse_entity<&value_type::ggml_token_type, "tokenizer.ggml.token_type">(), make_parse_entity<&value_type::ggml_merges, "tokenizer.ggml.merges">(),
				make_parse_entity<&value_type::special_bos_id, "tokenizer.ggml.bos_token_id">(), make_parse_entity<&value_type::special_eos_id, "tokenizer.ggml.eos_token_id">(),
				make_parse_entity<&value_type::chat_template, "tokenizer.chat_template">(), make_parse_entity<&value_type::quantization_version, "general.quantization_version">(),
				make_parse_entity<&value_type::quantize_imatrix_file, "quantize.imatrix.file">(),
				make_parse_entity<&value_type::quantize_imatrix_dataset, "quantize.imatrix.dataset">(),
				make_parse_entity<&value_type::quantize_imatrix_entries_count, "quantize.imatrix.entries_count">(),
				make_parse_entity<&value_type::quantize_imatrix_chunks_count, "quantize.imatrix.chunks_count">()>();
	};

	template<model_config config, typename value_type> struct parse_types_impl {
		inline static constexpr auto memberCount = core_tuple_size<value_type>;

		template<uint64_t index> using member_type_t =
			std::remove_reference_t<decltype(get_member<value_type, get<index>(parse_core<value_type>::parse_value).member_ptr>(std::declval<value_type&>()))>;

		template<uint64_t index> NIHILUS_FORCE_INLINE static void processIndex(value_type& value, std::string_view string, stream_iterator& stream) {
			static constexpr auto tupleElem	 = get<index>(parse_core<value_type>::parse_value);
			static constexpr auto string_lit = tupleElem.name;
			static constexpr auto ptrNew	 = tupleElem.member_ptr;
			static constexpr auto keySize	 = string_lit.size();
			static constexpr auto keySizeNew = keySize + 1;
			if NIHILUS_LIKELY ((string.size() <= keySize) && string_literal_comparitor<decltype(string_lit), string_lit>::impl(string.data())) {
				auto& ref = get_member<value_type, ptrNew>(value);
				if constexpr (!std::is_const_v<std::remove_reference_t<decltype(ref)>>) {
					ref = value_reader<config, member_type_t<index>>::gather_value(stream);
				} else {
					member_type_t<index> value_new{ value_reader<config, std::remove_const_t<member_type_t<index>>>::gather_value(stream) };
					if (value_new != ref) {
						static constexpr string_literal sl_new{ "Sorry, but member of name: " + string_lit + " was not equal!" };
						static constexpr auto location = get_source_location();
						status_handler<config, void>::template construct_status<sl_new, location, success_statuses::fail>();
						return;
					}
				}
				( void )ref;
			}
		}
	};

	template<template<model_config, typename> typename parsing_type, model_config config, typename value_type, size_t... indices>
	inline static constexpr auto generateFunctionPtrs(std::index_sequence<indices...>) noexcept {
		using function_type = decltype(&parse_types_impl<config, value_type>::template processIndex<0>);
		return array<function_type, sizeof...(indices)>{ { &parsing_type<config, value_type>::template processIndex<indices>... } };
	}

	template<template<model_config, typename> typename parsing_type, model_config config, typename value_type>
	static constexpr auto function_ptrs{ generateFunctionPtrs<parsing_type, config, value_type>(std::make_index_sequence<core_tuple_size<value_type>>{}) };

	template<model_config config> NIHILUS_INLINE void calculate_and_skip_unknown_value(stream_iterator& input, gguf_metadata_value_type type) {
		switch (type) {
			case gguf_metadata_value_type::uint8:
			case gguf_metadata_value_type::int8:
			case gguf_metadata_value_type::boolean: {
				input.current_index += 1;
				break;
			}
			case gguf_metadata_value_type::uint16:
			case gguf_metadata_value_type::int16: {
				input.current_index += 2;
				break;
			}
			case gguf_metadata_value_type::uint32:
			case gguf_metadata_value_type::int32:
			case gguf_metadata_value_type::float32: {
				input.current_index += 4;
				break;
			}
			case gguf_metadata_value_type::uint64:
			case gguf_metadata_value_type::int64:
			case gguf_metadata_value_type::float64: {
				input.current_index += 8;
				break;
			}
			case gguf_metadata_value_type::string: {
				if (!input.has_bytes<uint64_t>()) {
					static constexpr auto location = get_source_location();
					status_handler<config, void>::template construct_status<"Insufficient bytes for string length!", location, success_statuses::fail>();
					return;
				}
				uint64_t string_length = input.read<uint64_t>();

				if (!input.has_bytes<uint8_t>(string_length)) {
					static constexpr auto location = get_source_location();
					status_handler<config, void>::template construct_status<"Insufficient bytes for string content!", location, success_statuses::fail>();
					return;
				}
				input.current_index += string_length;
				break;
			}
			case gguf_metadata_value_type::array: {
				if (!input.has_bytes<gguf_metadata_value_type>()) {
					static constexpr auto location = get_source_location();
					status_handler<config, void>::template construct_status<"Insufficient bytes for array type!", location, success_statuses::fail>();
					return;
				}
				gguf_metadata_value_type array_type = input.read<gguf_metadata_value_type>();

				if (!input.has_bytes<uint64_t>()) {
					static constexpr auto location = get_source_location();
					status_handler<config, void>::template construct_status<"Insufficient bytes for array length!", location, success_statuses::fail>();
					return;
				}
				uint64_t array_length = input.read<uint64_t>();

				constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
				if (array_length > MAX_ARRAY_LENGTH) {
					static constexpr auto location = get_source_location();
					status_handler<config, void>::template construct_status<"Array length exceeds maximum allowed size during skip!", location, success_statuses::fail>();
					return;
				}

				for (uint64_t i = 0; i < array_length; ++i) {
					calculate_and_skip_unknown_value<config>(input, array_type);
				}
				break;
			}
			case gguf_metadata_value_type::unset:
			default: {
				break;
			}
		}

		return;
	}

	template<model_config config> struct value_reader<config, gguf_metadata<config>> {
		NIHILUS_FORCE_INLINE static gguf_metadata<config> gather_value(stream_iterator& input) {
			gguf_metadata<config> value{};
			uint32_t magic = value_reader<config, uint32_t>::gather_value(input);
			if (magic != 0x46554747) {
				static constexpr auto location = get_source_location();
				return status_handler<config, gguf_metadata<config>>::template construct_status<"Sorry, but that magic value was incorrect!", location, success_statuses::fail>();
			}
			uint64_t version		= value_reader<config, uint32_t>::gather_value(input);
			value.tensor_count		= value_reader<config, uint64_t>::gather_value(input);
			value.metadata_kv_count = value_reader<config, uint64_t>::gather_value(input);

			static constexpr uint64_t MAX_TENSOR_COUNT	 = 100000;
			static constexpr uint64_t MAX_METADATA_COUNT = 10000;

			if (value.tensor_count > MAX_TENSOR_COUNT) {
				static constexpr auto location = get_source_location();
				return status_handler<config, gguf_metadata<config>>::template construct_status<"Tensor count exceeds reasonable maximum!", location, success_statuses::fail>();
			}
			if (value.metadata_kv_count > MAX_METADATA_COUNT) {
				static constexpr auto location = get_source_location();
				return status_handler<config, gguf_metadata<config>>::template construct_status<"Metadata count exceeds reasonable maximum!", location, success_statuses::fail>();
			}

			for (uint64_t x = 0; x < value.metadata_kv_count; ++x) {
				std::string_view new_string			= value_reader<config, gguf_string_t>::gather_value(input);
				gguf_metadata_value_type value_type = value_reader<config, gguf_metadata_value_type>::gather_value(input);
				auto index							= hash_map<gguf_metadata<config>, const char*>::findIndex(new_string.data(), new_string.data() + new_string.size());
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
		NIHILUS_FORCE_INLINE static op_types impl(std::string_view input) noexcept {
			if (string_literal_comparison<"token_embd.weight">(input.data())) {
				return op_types ::token_embd_weight;
			}
			if (string_literal_comparison<"rope_freqs.weight">(input.data())) {
				return op_types::rope_freqs_weight;
			}
			if (string_literal_comparison<"output_norm.weight">(input.data())) {
				return op_types::output_norm_weight;
			}
			if (string_literal_comparison<"output.weight">(input.data())) {
				return op_types::output_weight;
			}

			if (string_literal_comparison<".attn_q.weight">(input.data() + input.find(".attn_q.weight"))) {
				return op_types::attn_q_weight;
			}
			if (string_literal_comparison<".attn_norm.weight">(input.data() + input.find(".attn_norm.weight"))) {
				return op_types::attn_norm_weight;
			}

			if (string_literal_comparison<"blk.">(input.data()) && string_literal_comparison<".weight">(input.data() + input.size() - 7)) {
				auto second_dot = input.find('.', 4);
				if (second_dot != std::string_view::npos) {
					auto suffix = input.substr(second_dot + 1);

					if (string_literal_comparison<"attn_q.weight">(suffix.data())) {
						return op_types::attn_q_weight;
					}
					if (string_literal_comparison<"attn_norm.weight">(suffix.data())) {
						return op_types::attn_norm_weight;
					}
					if (string_literal_comparison<"attn_k.weight">(suffix.data())) {
						return op_types::attn_k_weight;
					}
					if (string_literal_comparison<"attn_v.weight">(suffix.data())) {
						return op_types::attn_v_weight;
					}
					if (string_literal_comparison<"attn_output.weight">(suffix.data())) {
						return op_types::attn_output_weight;
					}
					if (string_literal_comparison<"ffn_down.weight">(suffix.data())) {
						return op_types::ffn_down_weight;
					}
					if (string_literal_comparison<"ffn_gate.weight">(suffix.data())) {
						return op_types::ffn_gate_weight;
					}
					if (string_literal_comparison<"ffn_up.weight">(suffix.data())) {
						return op_types::ffn_up_weight;
					}
					if (string_literal_comparison<"ffn_norm.weight">(suffix.data())) {
						return op_types::ffn_norm_weight;
					}
				}
			}

			return op_types::count;
		}
	};

	struct core_base_creation_data {
		array<uint64_t, 4> dimensions{ { 1, 1, 1, 1 } };
		uint32_t n_dimensions{};
		uint64_t layer_number{};
		op_types op_type{};
		uint64_t offset{};
		data_types type{};

		NIHILUS_FORCE_INLINE uint64_t total_dims() const {
			return dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3];
		}

		NIHILUS_FORCE_INLINE uint64_t total_byte_size() const {
			uint64_t total_elements = total_dims();
			uint64_t block_size_val = block_size();
			uint64_t type_size_val	= type_size();
			uint64_t num_blocks		= (total_elements + block_size_val - 1) / block_size_val;
			return num_blocks * type_size_val;
		}

		template<core_traits_type core_traits> NIHILUS_FORCE_INLINE bool operator==(const core_traits& other) const {
			static constexpr auto other_dims = core_traits::get_array();
			return dimensions[0] == other_dims[0] && dimensions[1] == other_dims[1] && dimensions[2] == other_dims[2] && dimensions[3] == other_dims[3];
		}

		NIHILUS_FORCE_INLINE uint64_t block_size() const {
			return get_type_traits(type).block_size;
		}

		NIHILUS_FORCE_INLINE uint64_t type_size() const {
			return get_type_traits(type).type_size;
		}
	};

	template<model_config, model_arches> struct core_traits_comparitor;

	template<model_config config> struct core_traits_comparitor<config, model_arches::llama> {
		NIHILUS_FORCE_INLINE static bool impl(const core_base_creation_data& parse_core) noexcept {
			switch (parse_core.op_type) {
				case op_types::token_embd_weight: {
					return parse_core == core_traits<config, op_types::token_embd_weight>{};
				}
				case op_types::rope_freqs_weight: {
					return parse_core == core_traits<config, op_types::rope_freqs_weight>{};
				}
				case op_types::output_norm_weight: {
					return parse_core == core_traits<config, op_types::output_norm_weight>{};
				}
				case op_types::output_weight: {
					return parse_core == core_traits<config, op_types::output_weight>{};
				}
				case op_types::attn_q_weight: {
					return parse_core == core_traits<config, op_types::attn_q_weight>{};
				}
				case op_types::attn_norm_weight: {
					return parse_core == core_traits<config, op_types::attn_norm_weight>{};
				}
				case op_types::attn_k_weight: {
					return parse_core == core_traits<config, op_types::attn_k_weight>{};
				}
				case op_types::attn_v_weight: {
					return parse_core == core_traits<config, op_types::attn_v_weight>{};
				}
				case op_types::attn_output_weight: {
					return parse_core == core_traits<config, op_types::attn_output_weight>{};
				}
				case op_types::ffn_down_weight: {
					return parse_core == core_traits<config, op_types::ffn_down_weight>{};
				}
				case op_types::ffn_gate_weight: {
					return parse_core == core_traits<config, op_types::ffn_gate_weight>{};
				}
				case op_types::ffn_up_weight: {
					return parse_core == core_traits<config, op_types::ffn_up_weight>{};
				}
				case op_types::ffn_norm_weight: {
					return parse_core == core_traits<config, op_types::ffn_norm_weight>{};
				}
				default: {
					return false;
				}
			}
		}
	};

	NIHILUS_FORCE_INLINE constexpr uint64_t parse_number(std::string_view str) noexcept {
		uint64_t result = 0;
		for (char c: str) {
			if (c >= '0' && c <= '9') {
				result = result * 10 + (c - '0');
			} else {
				break;
			}
		}
		return result;
	}

	NIHILUS_FORCE_INLINE uint64_t extract_layer_number(std::string_view name) noexcept {
		if NIHILUS_LIKELY (name[0] == 'c' && name.starts_with("cache_")) {
			for (uint64_t i = 7; i < name.size(); ++i) {
				if (name[i] == 'l' && i + 1 < name.size()) {
					return parse_number(name.substr(i + 1));
				}
			}
			return 0;
		}
		if NIHILUS_LIKELY (name[0] == 'b' && name.starts_with("blk.")) {
			uint64_t start = 4;
			uint64_t end   = name.find('.', start);
			if (end != std::string_view::npos) {
				return parse_number(name.substr(start, end - start));
			}
		}

		return 0;
	}

	template<model_config config> struct value_reader<config, core_base_creation_data> {
		NIHILUS_FORCE_INLINE static core_base_creation_data gather_value(stream_iterator& input) {
			core_base_creation_data value{};
			std::string_view name{ value_reader<config, std::string_view>::gather_value(input) };
			value.op_type					  = string_to_op_type<config.arch>::impl(name);
			value.n_dimensions				  = value_reader<config, uint32_t>::gather_value(input);
			value.layer_number				  = extract_layer_number(name);
			constexpr uint32_t MAX_DIMENSIONS = 4;
			if (value.n_dimensions > MAX_DIMENSIONS) {
				static constexpr auto location = get_source_location();
				return status_handler<config, core_base_creation_data>::template construct_status<"Tensor dimensions exceed maximum!", location, success_statuses::fail>();
			}
			for (uint64_t x = 0; x < value.n_dimensions; ++x) {
				uint64_t dim					= value_reader<config, uint64_t>::gather_value(input);
				constexpr uint64_t MAX_DIM_SIZE = 1ULL << 32;
				if (dim > MAX_DIM_SIZE) {
					static constexpr auto location = get_source_location();
					return status_handler<config, core_base_creation_data>::template construct_status<"Tensor dimensions size too large!", location, success_statuses::fail>();
				}
				value.dimensions[x] = dim;
			}
			value.type	 = static_cast<data_types>(value_reader<config, uint32_t>::gather_value(input));
			value.offset = value_reader<config, uint64_t>::gather_value(input);
			return value;
		}
	};

	NIHILUS_FORCE_INLINE bool operator<(const core_base_creation_data& lhs, const core_base_creation_data& rhs) noexcept {
		return lhs.layer_number < rhs.layer_number;
	}

	NIHILUS_FORCE_INLINE void sort_tensor_infos(std::vector<core_base_creation_data>& tensor_infos) noexcept {
		std::sort(tensor_infos.begin(), tensor_infos.end(), std::less<core_base_creation_data>{});
	}

	NIHILUS_FORCE_INLINE uint64_t align_offset(uint64_t offset, uint64_t alignment) {
		alignment = alignment == 0 ? 1 : alignment;
		return offset + (alignment - (offset % alignment)) % alignment;
	}

	template<model_config config> struct model_parser_impl {};

	template<model_config config>
		requires((config.arch == model_arches::llama) && (config.format == model_format::gguf))
	struct model_parser_impl<config> {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		static_assert((std::endian::native == std::endian::little), "Sorry, but big-endian is not yet supported by the library");
		template<typename tokenizer_type> NIHILUS_FORCE_INLINE static gguf_metadata<config> parse_model(array<array<void*, model_traits_type::block_count>, op_types::count>& data,
			memory_mapped_file* memory_file, tokenizer_type& tokenizer) {
			stream_iterator ptr{ memory_file };
			gguf_metadata<config> gguf_file{ value_reader<config, gguf_metadata<config>>::gather_value(ptr) };
			tokenizer.tokens		= detail::move(gguf_file.ggml_tokens);
			tokenizer.merges		= detail::move(gguf_file.ggml_merges);
			tokenizer.token_types	= detail::move(gguf_file.ggml_token_type);
			tokenizer.chat_template = detail::move(gguf_file.chat_template);
			std::vector<core_base_creation_data> tensor_infos{};
			tensor_infos.reserve(gguf_file.tensor_count);
			for (uint64_t x = 0; x < gguf_file.tensor_count; ++x) {
				auto new_tensor{ value_reader<config, core_base_creation_data>::gather_value(ptr) };
				if (!core_traits_comparitor<config, model_arches::llama>::impl(new_tensor)) {
					static constexpr auto location = get_source_location();
					return status_handler<config, gguf_metadata<config>>::template construct_status<"Tensor dimensions incorrect!", location, success_statuses::fail>();
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
				uint64_t absolute_offset = tensor_data_start + tensor_infos[x].offset;
				ptr.map_pointer(data[tensor_infos[x].op_type][tensor_infos[x].layer_number], absolute_offset);
			};
			return gguf_file;
		}
	};

	template<model_config config> struct model_parser {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

		template<typename tokenizer_type> NIHILUS_FORCE_INLINE static gguf_metadata<config> parse_model(array<array<void*, model_traits_type::block_count>, op_types::count>& data,
			memory_mapped_file* memory_file, tokenizer_type& tokenizer) {
			return model_parser_impl<config>::parse_model(data, memory_file, tokenizer);
		}
	};
}
