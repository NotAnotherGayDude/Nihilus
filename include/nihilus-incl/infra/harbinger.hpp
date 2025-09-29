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

#include <nihilus-incl/infra/model.hpp>
#include <nihilus-incl/infra/model_parser.hpp>
#include <nihilus-incl/common/common.hpp>
#include <cstdint>

namespace nihilus {

	NIHILUS_HOST static consteval auto generate_model_config(model_generations model_generation = {}, model_sizes model_size = {}, kernel_type_profiles kernel_type_profile = {},
		model_arches model_arch = {}, device_types device_type = {}, bool exceptions = {}, uint64_t default_max_sequence_length =  1024 , uint64_t default_batch_size = {},
		kv_cache_strategies kv_cache_strategy = {}, user_input_types user_input_type = {}, rope_scaling_types rope_scaling_type = {},
		tokenizer_pre_types tokenizer_pre_type = tokenizer_pre_types::llama3, uint64_t kv_cache_block_size = {}, bool use_rotary_embeddings = {},
		rms_norm_types rms_norm_type = {}, tokenizer_types tokenizer_type = tokenizer_types::bpe, model_formats model_format = model_formats::gguf, float norm_epsilon = 0.0f,
		bool benchmark = {}, bool dev = {}) {
		return model_config{ .model_generation = model_generation,
			.model_size						   = model_size,
			.kernel_type_profile			   = kernel_type_profile,
			.model_arch						   = model_arch,
			.exceptions						   = exceptions,
			.default_max_sequence_length	   = default_max_sequence_length,
			.default_batch_size				   = default_batch_size,
			.kv_cache_strategy				   = kv_cache_strategy,
			.user_input_type				   = user_input_type,
			.rope_scaling_type				   = rope_scaling_type,
			.tokenizer_pre_type				   = tokenizer_pre_type,
			.kv_cache_block_size			   = kv_cache_block_size,
			.use_rotary_embeddings			   = use_rotary_embeddings,
			.rms_norm_type					   = rms_norm_type,
			.tokenizer_type					   = tokenizer_type,
			.device_type					   = device_type,
			.model_format					   = model_format,
			.norm_epsilon					   = norm_epsilon,
			.benchmark						   = benchmark,
			.dev							   = dev };
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::model_generation)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type model_generation) {
		config.model_generation = model_generation;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::model_size)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type model_size) {
		config.model_size = model_size;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::kernel_type_profile)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type kernel_type_profile) {
		config.kernel_type_profile = kernel_type_profile;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::model_arch)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type model_arch) {
		config.model_arch = model_arch;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::exceptions)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type exceptions) {
		config.exceptions = exceptions;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::default_max_sequence_length)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type default_max_sequence_length) {
		config.default_max_sequence_length = default_max_sequence_length;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::default_batch_size)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type default_batch_size) {
		config.default_batch_size = default_batch_size;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::kv_cache_strategy)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type kv_cache_strategy) {
		config.kv_cache_strategy = kv_cache_strategy;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::user_input_type)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type user_input_type) {
		config.user_input_type = user_input_type;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::rope_scaling_type)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type rope_scaling_type) {
		config.rope_scaling_type = rope_scaling_type;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::tokenizer_pre_type)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type tokenizer_pre_type) {
		config.tokenizer_pre_type = tokenizer_pre_type;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::kv_cache_block_size)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type kv_cache_block_size) {
		config.kv_cache_block_size = kv_cache_block_size;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::use_rotary_embeddings)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type use_rotary_embeddings) {
		config.use_rotary_embeddings = use_rotary_embeddings;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::rms_norm_type)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type rms_norm_type) {
		config.rms_norm_type = rms_norm_type;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::tokenizer_type)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type tokenizer_type) {
		config.tokenizer_type = tokenizer_type;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::device_type)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type device_type) {
		config.device_type = device_type;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::model_format)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type model_format) {
		config.model_format = model_format;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::norm_epsilon)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type norm_epsilon) {
		config.norm_epsilon = norm_epsilon;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::benchmark)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type benchmark) {
		config.benchmark = benchmark;
		return config;
	}

	template<model_config_types model_config_type, typename config_update_type>
		requires(model_config_type == model_config_types::dev)
	NIHILUS_HOST static consteval auto update_model_config(model_config config, config_update_type dev) {
		config.dev = dev;
		return config;
	}

	template<typename arg_type> NIHILUS_HOST static consteval auto model_config_updates(model_config config, arg_type&& arg) {
		return update_model_config(config, std::forward<arg_type>(arg));
	}

	template<typename arg_type, typename... arg_types> NIHILUS_HOST static consteval auto model_config_updates(model_config config, arg_type&& arg, arg_types&&... args) {
		auto updated_config = update_model_config(config, std::forward<arg_type>(arg));
		return model_config_updates(updated_config, std::forward<arg_types>(args)...);
	}

	template<const model_config& config> struct harbinger {
		using model_t	   = model<config>;
		using model_base_t = model_base;

		NIHILUS_HOST static auto parse_model_graph_data(cli_params params) {
			std::unique_ptr<model_base_t> return_value{ new model_t{ params } };
			return return_value;
		}

		NIHILUS_HOST static auto serialize_model(serializer_params) {
			//return model_serializer<config>::impl(params);
		}

		NIHILUS_HOST static auto get_model_graph() {
			std::unique_ptr<model_base_t> return_value{ new model_t{} };
			return return_value;
		}

		inline static cli_params parse_cli_arguments(int32_t argc, char** argv) {
			aligned_vector<std::string> cli_args{};
			for (int64_t x = 0; x < argc; ++x) {
				cli_args.emplace_back(argv[static_cast<uint64_t>(x)]);
			}
			return parse_cli_arguments(cli_args);
		}

		inline static cli_params parse_cli_arguments(const aligned_vector<std::string>& command_line) {
			cli_params result{};
			std::string current_flag{};
			bool expect_value = false;

			for (const auto& token_new: command_line) {
				if (token_new.empty())
					continue;

				if (token_new[0] == '-') {
					current_flag = token_new;
					if (token_new == "-m" || token_new == "-t" || token_new == "-p" || token_new == "-s" || token_new == "-n" || token_new == "-b") {
						expect_value = true;
					} else {
						expect_value = false;
					}
				} else if (expect_value) {
					if (current_flag == "-m") {
						result.model_file = token_new;
					} else if (current_flag == "-t") {
						try {
							result.thread_count = std::stoull(token_new);
						} catch (const std::exception&) {
							result.thread_count = 1;
						}
					} else if (current_flag == "-p") {
						result.prompt = token_new;
					} else if (current_flag == "-s") {
						try {
							result.seed = std::stoull(token_new);
						} catch (const std::exception&) {
							result.seed = 0;
						}
					} else if (current_flag == "-n") {
						try {
							result.n_tokens = std::stoull(token_new);
						} catch (const std::exception&) {
							result.n_tokens = 0;
						}
					} else if (current_flag == "-b") {
						try {
							result.batch_size = std::stoull(token_new);
						} catch (const std::exception&) {
							result.batch_size = 512;
						}
					}
					expect_value = false;
				}
			}

			return result;
		}
	};

}
