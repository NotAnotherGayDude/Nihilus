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

#include <nihilus-incl/common/string_literal.hpp>
#include <nihilus-incl/common/config.hpp>
#include <nihilus-incl/common/utility.hpp>
#include <type_traits>
#include <iostream>
#include <concepts>
#include <memory>
#include <mutex>

namespace nihilus {

	enum class log_levels {
		error,
		status,
	};

	template<log_levels level> NIHILUS_INLINE void log(const std::string_view string) {
		static std::mutex mutex_new{};
		std::unique_lock<std::mutex> lock{ mutex_new };
		if constexpr (level == log_levels::error) {
			std::cerr << string << std::endl;
		} else {
			std::cout << string << std::endl;
		}
	}

	template<auto config, string_literal error_type, const std::source_location& source_info> struct nihilus_exception {
		NIHILUS_INLINE static void impl() {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal return_value{ "Error: " + error_type + "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			log<log_levels::error>(return_value);
			std::exit(-1);
		}
		NIHILUS_INLINE static void impl(const std::string_view input_string) {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal return_value01{ "Error: " + error_type };
			static constexpr string_literal return_value02{ "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			std::string new_string{ return_value01.operator std::string() + static_cast<std::string>(input_string) + return_value02.operator std::string() };
			log<log_levels::error>(new_string);
			std::exit(-1);
		}
		nihilus_exception() = delete;	
	};

	template<auto config, string_literal error_type, const std::source_location& source_info>
		requires(config.exceptions)
	struct nihilus_exception<config, error_type, source_info> : public std::runtime_error {
		NIHILUS_INLINE static void impl() {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal return_value{ "Error: " + error_type + "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			throw nihilus_exception(static_cast<const std::string_view>(return_value));
		}
		NIHILUS_INLINE static void impl(const std::string_view input_string) {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal return_value01{ "Error: " + error_type };
			static constexpr string_literal return_value02{ "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			std::string new_string{ return_value01.operator std::string() + input_string + return_value02.operator std::string() };
			throw nihilus_exception(static_cast<const std::string_view>(new_string));
		}

	  protected:
		nihilus_exception(const std::string_view new_value) : std::runtime_error(static_cast<std::string>(new_value)) {
		}
	};
}
