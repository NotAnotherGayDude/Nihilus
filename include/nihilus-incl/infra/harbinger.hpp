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

	struct harbinger {

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
					if (token_new == "-m" || token_new == "-t" || token_new == "-p" || token_new == "-s" || token_new == "-n" || token_new == "-b" || token_new == "-i" ||
						token_new == "-r") {
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
					} else if (current_flag == "-i") {
						result.ip = token_new;
					} else if (current_flag == "-r") {
						result.port = static_cast<uint16_t>(std::stoi(token_new));
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
