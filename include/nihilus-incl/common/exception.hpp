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
#include <nihilus-incl/common/model_config.hpp>
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

	NIHILUS_HOST std::mutex& get_mutex() {
		static std::mutex mutex_new{};
		return mutex_new;
	}

	template<log_levels level> struct logger {
	  protected:
		inline static std::ostream& os{ level == log_levels::error ? std::cerr : std::cout };

	  public:
		template<typename... arg_types> NIHILUS_HOST static void log(arg_types&&... strings) {
			std::lock_guard<std::mutex> lock{ get_mutex() };
			((os << std::forward<arg_types>(strings)), ...);
			os << std::endl;
		}
	};

	template<bool exceptions, string_literal error_type, const std::source_location& source_info> struct nihilus_exception {
		NIHILUS_HOST static void impl() {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal new_string{ "Error: " + error_type + "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			logger<log_levels::error>::log(new_string.operator std::string_view());
			throw nihilus_exception{};
		}
		NIHILUS_HOST static void impl(const std::string_view input_string) {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal return_value01{ "Error: " + error_type };
			static constexpr string_literal return_value02{ "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			std::string new_string{ return_value01.operator std::string() + static_cast<std::string>(input_string) + return_value02.operator std::string() };
			logger<log_levels::error>::log(new_string.operator std::string_view());
			throw nihilus_exception{};
		}

	  protected:
		nihilus_exception() {
		}
	};

	template<bool exceptions, string_literal error_type, const std::source_location& source_info>
		requires(!exceptions)
	struct nihilus_exception<exceptions, error_type, source_info> {
		NIHILUS_HOST static void impl() {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal new_string{ "Error: " + error_type + "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			logger<log_levels::error>::log(new_string.operator std::string_view());
			std::exit(-1);
		}
		NIHILUS_HOST static void impl(const std::string_view input_string) {
			static constexpr uint64_t str_length{ str_len(source_info.file_name()) };
			static constexpr string_literal return_value01{ "Error: " + error_type };
			static constexpr string_literal return_value02{ "\nIn File: " + string_literal<str_length>{ source_info.file_name() } +
				"\nOn Line: " + to_string_literal<source_info.line()>() + "\n" };
			std::string new_string{ return_value01.operator std::string() + static_cast<std::string>(input_string) + return_value02.operator std::string() };
			logger<log_levels::error>::log(new_string.operator std::string_view());
			std::exit(-1);
		}
		nihilus_exception() = delete;
	};
}
