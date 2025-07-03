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

#include <nihilus/common/input_session.hpp>
#include <nihilus/common/model.hpp>
#include <nihilus/common/model_parser.hpp>
#include <nihilus/common/common.hpp>
#include <cstdint>

namespace nihilus {

	NIHILUS_FORCE_INLINE static consteval auto generate_model_config(model_generations model_generation, model_sizes model_size, kernel_type_profiles kernel_profile,
		model_arches arch, bool exceptions = false, kv_cache_strategies cache_strategy = kv_cache_strategies::paged, bool use_gradient_checkpointing = false,
		rope_scaling_types rope_scaling = rope_scaling_types::linear, vocab_pre_types vocab_pre_type = vocab_pre_types::llama3, uint64_t kv_cache_block_size = 16,
		bool use_rotary_embeddings = true, bool use_flash_attention = true, norm_types rms_norm_type = norm_types::rms_standard, vocab_types vocab_type = vocab_types::bpe,
		model_format format = model_format::gguf, float norm_epsilon = 1e-6f, bool benchmark = false) {
		model_config config{ model_generation, model_size, kernel_profile, arch, exceptions, cache_strategy, use_gradient_checkpointing, rope_scaling, vocab_pre_type,
			kv_cache_block_size, use_rotary_embeddings, use_flash_attention, rms_norm_type, vocab_type, format, norm_epsilon, benchmark };
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_generation(model_config config, model_generations model_generation) {
		config.model_generation = model_generation;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_size(model_config config, model_sizes model_size) {
		config.model_size = model_size;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_kernel_profile(model_config config, kernel_type_profiles kernel_profile) {
		config.kernel_profile = kernel_profile;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_arch(model_config config, model_arches arch) {
		config.arch = arch;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_exceptions(model_config config, bool exceptions) {
		config.exceptions = exceptions;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_cache_strategy(model_config config, kv_cache_strategies cache_strategy) {
		config.cache_strategy = cache_strategy;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_gradient_checkpointing(model_config config, bool use_gradient_checkpointing) {
		config.use_gradient_checkpointing = use_gradient_checkpointing;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_rope_scaling(model_config config, rope_scaling_types rope_scaling) {
		config.rope_scaling = rope_scaling;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_vocab_pre_type(model_config config, vocab_pre_types vocab_pre_type) {
		config.vocab_pre_type = vocab_pre_type;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_kv_cache_block_size(model_config config, uint64_t kv_cache_block_size) {
		config.kv_cache_block_size = kv_cache_block_size;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_rotary_embeddings(model_config config, bool use_rotary_embeddings) {
		config.use_rotary_embeddings = use_rotary_embeddings;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_flash_attention(model_config config, bool use_flash_attention) {
		config.use_flash_attention = use_flash_attention;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_rms_norm_type(model_config config, norm_types rms_norm_type) {
		config.rms_norm_type = rms_norm_type;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_vocab_type(model_config config, vocab_types vocab_type) {
		config.vocab_type = vocab_type;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_format(model_config config, model_format format) {
		config.format = format;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_norm_epsilon(model_config config, float norm_epsilon) {
		config.norm_epsilon = norm_epsilon;
		return config;
	}

	NIHILUS_FORCE_INLINE static consteval auto update_model_config_benchmark(model_config config, bool benchmark) {
		config.benchmark = benchmark;
		return config;
	}

	template<typename... UpdateFuncs> NIHILUS_FORCE_INLINE static consteval auto chain_model_config_updates(model_config config, UpdateFuncs... update_funcs) {
		return (update_funcs(config), ...);
	}

	template<auto config> struct harbinger {
		using model_type		 = model<config>;
		using model_base_type	 = typename model<config>::base_type;
		using input_session_type = input_session<config, model_type>;

		NIHILUS_FORCE_INLINE static auto parse_model_graph_data(cli_params params) {
			std::unique_ptr<model_base_type> return_value{};
			model_base_type* new_model{ new model_type{ params } };
			return_value.reset(new_model);
			return return_value;
		}

		NIHILUS_FORCE_INLINE static auto parse_model_graph_data() {
			std::unique_ptr<model_base_type> return_value{};
			model_base_type* new_model{ new model_type{} };
			return_value.reset(new_model);
			return return_value;
		}

		NIHILUS_FORCE_INLINE static cli_params parse_cli_arguments(uint32_t argc, char** argv) {
			std::vector<std::string> cli_args{};
			for (uint64_t x = 0; x < argc; ++x) {
				cli_args.emplace_back(argv[x]);
			}
			return parse_cli_arguments(cli_args);
		}

		NIHILUS_FORCE_INLINE static cli_params parse_cli_arguments(const std::vector<std::string>& command_line) {
			cli_params result{};
			std::string current_flag{};
			bool expect_value = false;

			for (const auto& token: command_line) {
				if (token.empty())
					continue;

				if (token[0] == '-') {
					current_flag = token;
					if (token == "-m" || token == "-t" || token == "-p" || token == "-s" || token == "-n" || token == "-b") {
						expect_value = true;
					} else {
						expect_value = false;
					}
				} else if (expect_value) {
					if (current_flag == "-m") {
						result.model_file = token;
					} else if (current_flag == "-t") {
						try {
							result.thread_count = std::stoull(token);
						} catch (const std::exception&) {
							result.thread_count = 1;
						}
					} else if (current_flag == "-p") {
						result.prompt = token;
					} else if (current_flag == "-s") {
						try {
							result.seed = std::stoull(token);
						} catch (const std::exception&) {
							result.seed = 0;
						}
					} else if (current_flag == "-n") {
						try {
							result.n_tokens = std::stoull(token);
						} catch (const std::exception&) {
							result.n_tokens = 0;
						}
					} else if (current_flag == "-b") {
						try {
							result.batch_size = std::stoull(token);
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
