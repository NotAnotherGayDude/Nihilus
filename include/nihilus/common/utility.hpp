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

#include <nihilus/common/config.hpp>

namespace nihilus::detail {

	template<typename value_type01, typename value_type02> NIHILUS_FORCE_INLINE constexpr value_type01 max(value_type01 val01, value_type02 val02) noexcept {
		return val01 > static_cast<value_type01>(val02) ? val01 : static_cast<value_type01>(val02);
	}

	template<typename value_type01, typename value_type02> NIHILUS_FORCE_INLINE constexpr value_type01 min(value_type01 val01, value_type02 val02) noexcept {
		return val01 < static_cast<value_type01>(val02) ? val01 : static_cast<value_type01>(val02);
	}

	template<class value_type> NIHILUS_FORCE_INLINE constexpr value_type&& forward(std::remove_reference_t<value_type>& _Arg) noexcept {
		return static_cast<value_type&&>(_Arg);
	}

	template<class value_type> NIHILUS_FORCE_INLINE constexpr value_type&& forward(std::remove_reference_t<value_type>&& _Arg) noexcept {
		static_assert(!std::is_lvalue_reference_v<value_type>, "bad nihilus::detail::forward call");
		return static_cast<value_type&&>(_Arg);
	}

	template<class value_type> NIHILUS_FORCE_INLINE constexpr std::remove_reference_t<value_type>&& move(value_type&& _Arg) noexcept {
		return static_cast<std::remove_reference_t<value_type>&&>(_Arg);
	}

}