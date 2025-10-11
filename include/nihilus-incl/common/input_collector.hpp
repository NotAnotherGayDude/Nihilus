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

#include <nihilus-incl/common/config.hpp>
#include <nihilus-incl/common/common.hpp>
#include <iterator>

namespace nihilus {

	template<const model_config& config> struct input_collector {
	  protected:
		aligned_vector<char> buffer;
		uint64_t current_length = 0;

	  public:
		NIHILUS_HOST input_collector() {
			buffer.resize(config.default_max_sequence_length);
			terminate();
		}

		NIHILUS_HOST std::string_view get_view() const noexcept {
			return std::string_view(buffer.data(), current_length);
		}

		NIHILUS_HOST void clear() noexcept {
			current_length = 0;
			terminate();
		}

		NIHILUS_HOST void terminate() noexcept {
			if (current_length < config.default_max_sequence_length) {
				buffer.at(current_length) = '\0';
			}
		}

		NIHILUS_HOST bool read_multiline() noexcept {
			int32_t c;
			bool last_was_newline = false;

			while ((c = getchar()) != EOF) {
				if (c == '\n') {
					if (last_was_newline) {
						terminate();
						return true;
					}
					if (current_length < config.default_max_sequence_length - 1) {
						buffer[current_length++] = '\n';
					}
					last_was_newline = true;
				} else {
					last_was_newline = false;
					if (current_length < config.default_max_sequence_length - 1) {
						buffer[current_length++] = static_cast<char>(c);
					}
				}
			}
			terminate();
			return current_length > 0;
		}
	};

	template<const model_config& config>
		requires(config.user_input_type == user_input_types::managed)
	struct input_collector<config> {
	  protected:
		aligned_vector<char> buffer;
		uint64_t current_length = 0;
		atomic_flag_wrapper<uint64_t> in_signal{};
		atomic_flag_wrapper<uint64_t> out_signal{};

	  public:
		NIHILUS_HOST input_collector() {
			buffer.resize(config.default_max_sequence_length);
			terminate();
		}

		NIHILUS_HOST void clear() noexcept {
			current_length = 0;
			terminate();
		}

		NIHILUS_HOST void terminate() noexcept {
			if (current_length < config.default_max_sequence_length) {
				buffer[current_length] = '\0';
			}
		}

		NIHILUS_HOST uint64_t remaining_length() noexcept {
			return (config.default_max_sequence_length - current_length) - 1;
		}

		NIHILUS_HOST uint64_t write_input(const char* string, uint64_t length) noexcept {
			if (in_signal.test() != 1) {
				return 0;
			}
			uint64_t remaining_length_val{ remaining_length() };
			uint64_t written_bytes = length > remaining_length_val ? remaining_length_val : length;
			std::memcpy(buffer.data() + current_length, string, written_bytes);
			current_length += written_bytes;
			terminate();
			in_signal.clear();
			in_signal.notify_one();
			return written_bytes;
		}

		NIHILUS_HOST std::string_view read_multiline() noexcept {
			if (in_signal.test() == 1) {
				in_signal.wait();
			}
			std::string_view string{ buffer.data(), current_length };
			clear();
			in_signal.test_and_set();
			return string;
		}
	};


}
