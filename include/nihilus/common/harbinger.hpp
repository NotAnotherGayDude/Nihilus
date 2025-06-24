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

	NIHILUS_FORCE_INLINE static consteval auto generate_model_config(auto model_generation, auto model_size, kernel_type_profile kernel_profile, model_arch arch,
		bool exceptions = false, kv_cache_strategy cache_strategy = kv_cache_strategy::paged, bool use_gradient_checkpointing = false,
		rope_scaling_type rope_scaling = rope_scaling_type::linear, bool use_rotary_embeddings = true, uint64_t kv_cache_block_size = 16, bool use_flash_attention = true,
		norm_type rms_norm_type = norm_type::rms_standard, model_format format = model_format::gguf, float norm_epsilon = 1e-6f) {
		model_config<decltype(model_generation), decltype(model_size)> config{ model_generation, model_size, kernel_profile, arch, exceptions, cache_strategy,
			use_gradient_checkpointing, rope_scaling, use_rotary_embeddings, kv_cache_block_size, use_flash_attention, rms_norm_type, format, norm_epsilon };
		return config;
	};

	template<auto config> struct harbinger {
		using model_type			  = model<config>;
		using model_base_type		  = typename model<config>::base_type;
		using input_session_type	  = nihilus::input_session<model_type>;
		using input_session_base_type = input_session_type::base_type;

		NIHILUS_FORCE_INLINE static auto parse_model_graph_data(nihilus::cli_params params) {
			std::unique_ptr<model_base_type> return_value{};
			model_base_type* new_model{ new model_type{ params } };
			return_value.reset(new_model);
			return return_value;
		}
		
		NIHILUS_FORCE_INLINE static auto get_input_session(input_session_config& params, model_base_type& model) {
			std::unique_ptr<input_session_base_type> return_value{};
			input_session_type new_input_session{};
			input_session_base_type* new_model{ new input_session_type{ params, &static_cast<model_type&>(model) } };
			return_value.reset(new_model);
			return return_value;
		}
		
		NIHILUS_FORCE_INLINE static nihilus::cli_params parse_cli_arguments(uint32_t argc, char** argv) {
			std::vector<std::string> cli_args{};
			for (uint64_t x = 0; x < argc; ++x) {
				cli_args.emplace_back(argv[x]);
			}
			return parse_cli_arguments(cli_args);
		}

		NIHILUS_FORCE_INLINE static nihilus::cli_params parse_cli_arguments(const std::vector<std::string>& command_line) {
			nihilus::cli_params result{};
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
