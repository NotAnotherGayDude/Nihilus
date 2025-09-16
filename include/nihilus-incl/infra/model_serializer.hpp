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

#include <nihilus-incl/infra/model_parser.hpp>

namespace nihilus {

	struct serializer_params {
		const std::string_view out_file_path{};
		const std::string_view in_file_path{};
		uint64_t thread_count{};
	};

	template<model_config config> struct model_serializer_impl {};

	template<model_config config>
		requires((config.model_arch == model_arches::llama) && (config.model_format == model_formats::gguf))
	struct model_serializer_impl<config> {
		using model_traits_type = model_traits<config.model_arch, config.model_size, config.model_generation>;
		static_assert((std::endian::native == std::endian::little), "Sorry, but big-endian is not yet supported by the library");
		template<typename tokenizer_type> NIHILUS_INLINE static gguf_metadata<config> parse_model(array<array<void*, model_traits_type::block_count>, core_types::count>& data,
			memory_mapped_file<config>* memory_file, tokenizer_type& tokenizer) {
			stream_iterator<config> ptr{ memory_file };
			gguf_metadata<config> gguf_file{ value_reader<config, gguf_metadata<config>>::gather_value(ptr) };
			tokenizer.tokens		= detail::move(gguf_file.tokenizer_ggml_tokens);
			tokenizer.merges		= detail::move(gguf_file.tokenizer_ggml_merges);
			tokenizer.token_types	= detail::move(gguf_file.tokenizer_ggml_token_type);
			tokenizer.chat_template = detail::move(gguf_file.tokenizer_chat_template);
			aligned_vector<core_base_creation_data> tensor_infos{};
			tensor_infos.reserve(gguf_file.tensor_count);
			for (uint64_t x = 0; x < gguf_file.tensor_count; ++x) {
				auto new_tensor{ value_reader<config, core_base_creation_data, model_arches::llama>::gather_value(ptr) };
				if (!core_traits_comparitor<config, model_arches::llama>::impl(new_tensor)) {
					throw std::runtime_error{ "Tensor dimensions incorrect!" };
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
			}
			return gguf_file;
		}
	};

	template<model_config config> struct model_serializer {};

}
