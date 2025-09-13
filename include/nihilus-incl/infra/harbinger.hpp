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

	NIHILUS_INLINE static consteval auto generate_model_config(model_generations model_generation, model_sizes model_size, kernel_type_profiles kernel_profile, model_arches arch,
		device_types device_type_new = device_types::cpu, bool benchmark = false, bool dev = false, uint64_t default_max_sequence_length = 1024, bool exceptions = false,
		uint64_t default_batch_size = 512, kv_cache_strategies cache_strategy = kv_cache_strategies::paged, bool use_gradient_checkpointing = false,
		rope_scaling_types rope_scaling = rope_scaling_types::linear, tokenizer_pre_types tokenizer_pre_type = tokenizer_pre_types::llama3, uint64_t kv_cache_block_size = 16,
		bool use_rotary_embeddings = true, bool use_flash_attention = true, norm_types rms_norm_type = norm_types::rms_standard,
		tokenizer_types tokenizer_type = tokenizer_types::bpe, model_format format = model_format::gguf, float norm_epsilon = 1e-6f) {
		model_config config{ model_generation, model_size, kernel_profile, arch, exceptions, default_max_sequence_length, default_batch_size, cache_strategy,
			use_gradient_checkpointing, rope_scaling, tokenizer_pre_type, kv_cache_block_size, use_rotary_embeddings, use_flash_attention, rms_norm_type, tokenizer_type,
			device_type_new, format, norm_epsilon, benchmark, dev };
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_generation(model_config config, model_generations model_generation) {
		config.model_generation = model_generation;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_size(model_config config, model_sizes model_size) {
		config.model_size = model_size;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_kernel_profile(model_config config, kernel_type_profiles kernel_profile) {
		config.kernel_profile = kernel_profile;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_arch(model_config config, model_arches arch) {
		config.arch = arch;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_exceptions(model_config config, bool exceptions) {
		config.exceptions = exceptions;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_cache_strategy(model_config config, kv_cache_strategies cache_strategy) {
		config.cache_strategy = cache_strategy;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_gradient_checkpointing(model_config config, bool use_gradient_checkpointing) {
		config.use_gradient_checkpointing = use_gradient_checkpointing;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_rope_scaling(model_config config, rope_scaling_types rope_scaling) {
		config.rope_scaling = rope_scaling;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_tokenizer_pre_type(model_config config, tokenizer_pre_types tokenizer_pre_type) {
		config.tokenizer_pre_type = tokenizer_pre_type;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_kv_cache_block_size(model_config config, uint64_t kv_cache_block_size) {
		config.kv_cache_block_size = kv_cache_block_size;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_rotary_embeddings(model_config config, bool use_rotary_embeddings) {
		config.use_rotary_embeddings = use_rotary_embeddings;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_flash_attention(model_config config, bool use_flash_attention) {
		config.use_flash_attention = use_flash_attention;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_rms_norm_type(model_config config, norm_types rms_norm_type) {
		config.rms_norm_type = rms_norm_type;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_tokenizer_type(model_config config, tokenizer_types tokenizer_type) {
		config.tokenizer_type = tokenizer_type;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_format(model_config config, model_format format) {
		config.format = format;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_norm_epsilon(model_config config, float norm_epsilon) {
		config.norm_epsilon = norm_epsilon;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_max_context_length(model_config config, uint64_t context_length) {
		config.default_max_sequence_length = context_length;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_benchmark(model_config config, bool benchmark) {
		config.benchmark = benchmark;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_config_dev(model_config config, bool dev) {
		config.dev = dev;
		return config;
	}

	NIHILUS_INLINE static consteval auto update_model_device_type(model_config config, device_types dev) {
		config.device_type = dev;
		return config;
	}

	template<typename... UpdateFuncs> NIHILUS_INLINE static consteval auto chain_model_config_updates(model_config config, UpdateFuncs... update_funcs) {
		return (update_funcs(config), ...);
	}

	template<const model_config& config> struct harbinger {
		using model_t	   = model<config>;
		using model_base_t = model_base;

		NIHILUS_INLINE static auto parse_model_graph_data(cli_params params) {
			std::unique_ptr<model_base_t> return_value{ new model_t{ params } };
			return return_value;
		}

		NIHILUS_INLINE static auto serialize_model(serializer_params) {
			//return model_serializer<config>::impl(params);
		}

		NIHILUS_INLINE static auto get_model_graph() {
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
