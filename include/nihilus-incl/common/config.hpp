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

#include <source_location>
#include <cstring>
#include <cstdint>
#include <utility>
#include <chrono>
#include <atomic>

#if NIHILUS_PLATFORM_WINDOWS
	#ifndef PATH_MAX
		#define PATH_MAX MAX_PATH
	#endif
	#include <io.h>
	#include <Windows.h>
#else
	#include <sys/mman.h>
	#include <sys/stat.h>
	#include <fcntl.h>
	#include <unistd.h>
	#if NIHILUS_PLATFORM_LINUX
		#include <sys/resource.h>
	#elif NIHILUS_PLATFORM_MAC
		#include <mach/mach.h>
		#include <TargetConditionals.h>
	#endif
#endif

#if NIHILUS_CUDA_ENABLED
	#define NIHILUS_ALIGN(x) __align__(x)
#else
	#define NIHILUS_ALIGN(x) alignas(x)
#endif

#if !defined(NIHILUS_LIKELY)
	#define NIHILUS_LIKELY(...) (__VA_ARGS__) [[likely]]
#endif

#if !defined(NIHILUS_UNLIKELY)
	#define NIHILUS_UNLIKELY(...) (__VA_ARGS__) [[unlikely]]
#endif

#if !defined(NIHILUS_ELSE_UNLIKELY)
	#define NIHILUS_ELSE_UNLIKELY(...) __VA_ARGS__ [[unlikely]]
#endif

#if NIHILUS_ARCH_X64
	#include <immintrin.h>
#elif NIHILUS_ARCH_ARM64
	#include <arm_sve.h>
	#include <arm_neon.h>
#else
	#error "Unsupported architecture"
#endif

namespace nihilus {

	using clock_type = std::conditional_t<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>;

	template<auto enum_error, typename... types> struct error_printer_impl;

	template<bool value, auto enum_error, typename... value_to_test> struct static_assert_printer {
		static constexpr bool impl{ [] {
			if constexpr (!value) {
				error_printer_impl<enum_error, value_to_test...>::failure_value;
				return false;
			} else {
				return true;
			}
		}() };
	};

	template<auto enum_error, auto... values> struct error_printer_impl_val;

	template<bool value, auto enum_error, auto... values> struct static_assert_printer_val {
		static constexpr bool impl{ [] {
			if constexpr (!value) {
				error_printer_impl_val<enum_error, values...>::failure_value;
				return false;
			} else {
				return true;
			}
		}() };
	};

	template<uint64_t byte_count, typename value_type01, typename value_type02> NIHILUS_INLINE void constexpr_memcpy(value_type02* dst, const value_type01* src) {
		std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), byte_count);
	}

	template<typename value_type01, typename value_type02> NIHILUS_INLINE void memcpy_wrapper(value_type02* dst, const value_type01* src, uint64_t byte_count) {
		std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), byte_count);
	}

	template<typename value_type> struct alignas(64) static_aligned_const {
		alignas(64) value_type value{};

		NIHILUS_INLINE constexpr operator const value_type&() const& {
			return value;
		}

		NIHILUS_INLINE operator value_type&() & {
			return value;
		}

		NIHILUS_INLINE operator value_type&&() && {
			return std::move(value);
		}

		NIHILUS_INLINE constexpr const value_type& operator*() const {
			return value;
		}

		NIHILUS_INLINE value_type& operator*() {
			return value;
		}

		NIHILUS_INLINE constexpr bool operator==(const static_aligned_const& other) const {
			return value == other.value;
		}

		NIHILUS_INLINE constexpr bool operator!=(const static_aligned_const& other) const {
			return value != other.value;
		}

		NIHILUS_INLINE constexpr bool operator<(const static_aligned_const& other) const {
			return value < other.value;
		}

		NIHILUS_INLINE constexpr bool operator>(const static_aligned_const& other) const {
			return value > other.value;
		}
	};

	template<typename value_type> static_aligned_const(value_type) -> static_aligned_const<value_type>;

}
