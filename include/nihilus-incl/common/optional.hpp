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

namespace nihilus {

	template<typename value_type_new> struct optional {
		using value_type	  = value_type_new;
		using reference		  = value_type&;
		using pointer		  = value_type*;
		using const_reference = const value_type&;
		using const_pointer	  = const value_type*;

		NIHILUS_INLINE optional() noexcept = default;

		NIHILUS_INLINE optional& operator=(optional&& other) noexcept {
			if (this != &other) {
				if (constructed) {
					destroy();
				}
				if (other.constructed) {
					value		= detail::move(other.value);
					constructed = true;
				}
			}
			return *this;
		}

		NIHILUS_INLINE optional(optional&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE optional& operator=(const optional& other) noexcept {
			if (this != &other) {
				if (constructed) {
					destroy();
				}
				if (other.constructed) {
					value		= other.value;
					constructed = true;
				}
			}
			return *this;
		}

		NIHILUS_INLINE optional(const optional& other) noexcept {
			*this = other;
		}

		template<typename... value_types> NIHILUS_INLINE optional& operator=(value_types&&... args) noexcept {
			if (constructed) {
				destroy();
			}
			new (&value) value_type(detail::forward<value_types>(args)...);
			constructed = true;
			return *this;
		}

		template<typename... value_types> NIHILUS_INLINE optional(value_types&&... args) noexcept : value{ std::forward<value_types>(args)... } {
			constructed = true;
		}

		NIHILUS_INLINE pointer operator->() noexcept {
			return &value;
		}

		NIHILUS_INLINE operator reference() noexcept {
			return value;
		}

		NIHILUS_INLINE operator const_reference() const noexcept {
			return value;
		}

		NIHILUS_INLINE reference operator*() noexcept {
			return value;
		}

		NIHILUS_INLINE const_pointer operator->() const noexcept {
			return &value;
		}

		NIHILUS_INLINE const_reference operator*() const noexcept {
			return value;
		}

		NIHILUS_INLINE void swap(optional& other) noexcept {
			std::swap(value, other.value);
		}

		NIHILUS_INLINE ~optional() noexcept = default;

	  protected:
		bool constructed{ false };
		value_type value;

		NIHILUS_INLINE void destroy() noexcept {
			if (constructed) {
				value.~value_type();
				constructed = false;
			}
		}
	};

}
