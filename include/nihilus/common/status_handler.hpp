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

#include <nihilus/common/string_literal.hpp>
#include <nihilus/common/config.hpp>
#include <nihilus/common/utility.hpp>
#include <type_traits>
#include <concepts>
#include <mutex>

namespace nihilus {

	struct source_location_wrapper {
		const char* file_name{};
		uint64_t line{};
	};

	NIHILUS_FORCE_INLINE consteval source_location_wrapper get_source_location(std::source_location location = std::source_location::current()) {
		return { location.file_name(), location.line() };
	}

	void test_function() {
		static constexpr auto new_value = get_source_location();
	}

	inline std::mutex mutex{};

	enum class log_levels {
		error,
		status,
	};

	template<log_levels level> NIHILUS_FORCE_INLINE void log(std::string_view string) {
		std::unique_lock lock{ mutex };
		if constexpr (level == log_levels::error) {
			std::cerr << string << std::endl;
		} else {
			std::cout << string << std::endl;
		}
	}

	template<auto config, string_literal error_type, source_location_wrapper source_info, typename return_type = void*> struct nihilus_exception;

	template<auto config, string_literal error_type, source_location_wrapper source_info, typename return_type>
		requires(!config.exceptions)
	struct nihilus_exception<config, error_type, source_info, return_type> {
		NIHILUS_FORCE_INLINE static return_type impl() {
			static constexpr uint64_t str_length{ strlen(source_info.file_name) };
			static constexpr string_literal return_value{ "Error: " + error_type + "\nIn File: " + string_literal<str_length>{ source_info.file_name } +
				"\nOn Line: " + to_string_literal<source_info.line>() + "\n" };
			log<log_levels::error>(return_value);
			test_function();
			if constexpr (!std::is_void_v<return_type>) {
				return return_type{};
			}
		}
		NIHILUS_FORCE_INLINE static return_type impl(const std::string& input_string) {
			static constexpr uint64_t str_length{ strlen(source_info.file_name) };
			static constexpr string_literal return_value01{ "Error: " + error_type };
			static constexpr string_literal return_value02{ "\nIn File: " + string_literal<str_length>{ source_info.file_name } +
				"\nOn Line: " + to_string_literal<source_info.line>() + "\n" };
			std::string new_string{ return_value01.operator std::string() + input_string + return_value02.operator std::string() };
			log<log_levels::error>(new_string);
			if constexpr (!std::is_void_v<return_type>) {
				return return_type{};
			}
		}
	};

	template<auto config, string_literal error_type, source_location_wrapper source_info, typename return_type>
		requires(config.exceptions)
	struct nihilus_exception<config, error_type, source_info, return_type> : public std::runtime_error {
		NIHILUS_FORCE_INLINE static return_type impl() {
			static constexpr uint64_t str_length{ strlen(source_info.file_name) };
			static constexpr string_literal return_value{ "Error: " + error_type + "\nIn File: " + string_literal<str_length>{ source_info.file_name } +
				"\nOn Line: " + to_string_literal<source_info.line>() + "\n" };
			throw nihilus_exception(static_cast<std::string_view>(return_value));
			std::unreachable();
		}
		NIHILUS_FORCE_INLINE static return_type impl(const std::string& input_string) {
			static constexpr uint64_t str_length{ strlen(source_info.file_name) };
			static constexpr string_literal return_value01{ "Error: " + error_type };
			static constexpr string_literal return_value02{ "\nIn File: " + string_literal<str_length>{ source_info.file_name } +
				"\nOn Line: " + to_string_literal<source_info.line>() + "\n" };
			std::string new_string{ return_value01.operator std::string() + input_string + return_value02.operator std::string() };
			throw nihilus_exception(static_cast<std::string_view>(new_string));
			std::unreachable();
		}

	  public:
		nihilus_exception(std::string_view new_value) : std::runtime_error(static_cast<std::string>(new_value)) {};
	};

	enum class success_statuses {
		unset	= 0,
		success = 1,
		fail	= 2,
	};

	template<auto config, typename value_type> struct status_handler {
		NIHILUS_FORCE_INLINE status_handler(success_statuses status_new) : status{ status_new } {};
		template<typename value_type_new> NIHILUS_FORCE_INLINE status_handler(value_type_new&& value_new)
			: status{ success_statuses::success }, value{ detail::forward<value_type_new>(value_new) } {};

		NIHILUS_FORCE_INLINE operator bool() {
			return status == success_statuses::success;
		}

		NIHILUS_FORCE_INLINE operator value_type&&() && {
			return std::move(value);
		}

		NIHILUS_FORCE_INLINE operator value_type&() & {
			return value;
		}

		template<string_literal error_type, source_location_wrapper source_info, success_statuses status_new>
		NIHILUS_FORCE_INLINE static status_handler construct_status(const std::string& input_string) {
			status_handler handler_new{ status_new };
			handler_new.value = nihilus_exception<config, error_type, source_info, value_type>::impl(input_string);
			return handler_new;
		}

		template<string_literal error_type, source_location_wrapper source_info, success_statuses status_new>
		NIHILUS_FORCE_INLINE static status_handler construct_status() {
			status_handler handler_new{ status_new };
			handler_new.value = nihilus_exception<config, error_type, source_info, value_type>::impl();
			return handler_new;
		}

	  protected:
		success_statuses status{};
		value_type value{};
	};

	template<auto config, typename value_type>
		requires(std::is_void_v<value_type>)
	struct status_handler<config, value_type> {
		NIHILUS_FORCE_INLINE status_handler(success_statuses status_new) : status{ status_new } {};
		template<typename value_type_new> NIHILUS_FORCE_INLINE status_handler(value_type_new&& value_new) : status{ success_statuses::success } {};

		NIHILUS_FORCE_INLINE operator bool() {
			return status == success_statuses::success;
		}

		template<string_literal error_type, source_location_wrapper source_info, success_statuses status_new>
		NIHILUS_FORCE_INLINE static status_handler construct_status(const std::string& input_string) {
			status_handler handler_new{ status_new };
			nihilus_exception<config, error_type, source_info, value_type>::impl(input_string);
			return handler_new;
		}

		template<string_literal error_type, source_location_wrapper source_info, success_statuses status_new>
		NIHILUS_FORCE_INLINE static status_handler construct_status() {
			status_handler handler_new{ status_new };
			nihilus_exception<config, error_type, source_info, value_type>::impl();
			return handler_new;
		}

	  protected:
		success_statuses status{};
	};
}