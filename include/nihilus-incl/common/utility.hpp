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
#include <new>

namespace nihilus::detail {

	template<typename value_type>
	concept r_value_reference_types = std::is_rvalue_reference_v<value_type>;

	template<typename value_01_type, typename value_02_type>
	concept convertible_to = std::is_convertible_v<value_02_type, value_01_type>;

	template<typename value_type> struct remove_const {
		using type = value_type;
	};

	template<typename value_type> struct remove_const<const value_type> {
		using type = value_type;
	};

	template<typename value_type> using remove_const_t = remove_const<value_type>::type;

	template<typename value_type> struct remove_volatile {
		using type = value_type;
	};

	template<typename value_type> struct remove_volatile<const value_type> {
		using type = value_type;
	};

	template<typename value_type> using remove_volatile_t = remove_volatile<value_type>::type;

	template<typename value_type> struct remove_reference {
		using type = value_type;
	};

	template<typename value_type> struct remove_reference<value_type&&> {
		using type = value_type;
	};

	template<typename value_type> struct remove_reference<value_type&> {
		using type = value_type;
	};

	template<typename value_type> using remove_reference_t = remove_reference<value_type>::type;

	template<typename value_type> struct remove_cvref {
		using type = value_type;
	};

	template<typename value_type> struct remove_cvref<value_type&> {
		using type = value_type;
	};

	template<typename value_type> using remove_cvref_t = remove_cvref<remove_const_t<value_type>>::type;

	template<typename value_01_type, convertible_to<value_01_type> value_02_type> NIHILUS_HOST constexpr value_01_type max(value_01_type val01, value_02_type val02) noexcept {
		return val01 > static_cast<remove_cvref_t<value_01_type>>(val02) ? val01 : static_cast<remove_cvref_t<value_01_type>>(val02);
	}

	template<typename value_01_type, convertible_to<value_01_type> value_02_type> NIHILUS_HOST constexpr value_01_type min(value_01_type val01, value_02_type val02) noexcept {
		return val01 < static_cast<remove_cvref_t<value_01_type>>(val02) ? val01 : static_cast<remove_cvref_t<value_01_type>>(val02);
	}

	template<typename value_type> NIHILUS_HOST constexpr decltype(auto) forward(value_type& arg) noexcept {
		return static_cast<value_type&&>(arg);
	}

	template<typename value_type> NIHILUS_HOST constexpr decltype(auto) forward(value_type&& arg) noexcept {
		return static_cast<value_type&&>(arg);
	}

	template<typename value_type> NIHILUS_HOST constexpr decltype(auto) move(value_type&& arg) noexcept {
		return static_cast<std::remove_reference_t<value_type>&&>(arg);
	}

}
