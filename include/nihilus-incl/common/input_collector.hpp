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

	template<model_config config> struct input_collector {
	  protected:
		aligned_vector<char> buffer;
		size_t current_length = 0;

	  public:
		NIHILUS_INLINE input_collector() {
			buffer.resize(config.default_max_sequence_length);
			terminate();
		}

		NIHILUS_INLINE std::string_view get_view() const noexcept {
			return std::string_view(buffer.data(), current_length);
		}

		NIHILUS_INLINE void clear() noexcept {
			current_length = 0;
			terminate();
		}

		NIHILUS_INLINE void terminate() noexcept {
			buffer[current_length] = '\0';
		}

		NIHILUS_INLINE bool read_multiline() noexcept {
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


}
