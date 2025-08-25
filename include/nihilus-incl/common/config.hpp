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

#include <nihilus-incl/cpu/nihilus_thread_count.hpp>
#include <nihilus-incl/cpu/nihilus_cpu_instructions.hpp>
#include <source_location>
#include <cstring>
#include <cstdint>
#include <utility>
#include <chrono>
#include <atomic>

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define NIHILUS_PLATFORM_WINDOWS 1
	#define NIHILUS_PLATFORM_MAC 0
	#define NIHILUS_PLATFORM_ANDROID 0
	#define NIHILUS_PLATFORM_LINUX 0
#elif defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__)) || defined(TARGET_OS_MAC)
	#include <mach/mach.h>
	#define NIHILUS_PLATFORM_WINDOWS 0
	#define NIHILUS_PLATFORM_MAC 1
	#define NIHILUS_PLATFORM_ANDROID 0
	#define NIHILUS_PLATFORM_LINUX 0
#elif defined(__ANDROID__)
	#define NIHILUS_PLATFORM_WINDOWS 0
	#define NIHILUS_PLATFORM_MAC 0
	#define NIHILUS_PLATFORM_ANDROID 1
	#define NIHILUS_PLATFORM_LINUX 0
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define NIHILUS_PLATFORM_WINDOWS 0
	#define NIHILUS_PLATFORM_MAC 0
	#define NIHILUS_PLATFORM_ANDROID 0
	#define NIHILUS_PLATFORM_LINUX 1
#else
	#error "Unsupported platform"
#endif

#if defined(_MSC_VER)
	#define NIHILUS_COMPILER_MSVC 1 
	#define NIHILUS_COMPILER_CLANG 0
	#define NIHILUS_COMPILER_GNUCXX 0
#elif defined(__clang__) || defined(__llvm__)
	#define NIHILUS_COMPILER_MSVC 0
	#define NIHILUS_COMPILER_CLANG 1
	#define NIHILUS_COMPILER_GNUCXX 0
#elif defined(__GNUC__) && !defined(__clang__)
	#define NIHILUS_COMPILER_MSVC 0
	#define NIHILUS_COMPILER_CLANG 0
	#define NIHILUS_COMPILER_GNUCXX 1
#else
	#error "Unsupported compiler"
#endif

#if defined(NDEBUG)
	#if NIHILUS_COMPILER_MSVC
		#define NIHILUS_INLINE [[msvc::forceinline]] inline
		#define NIHILUS_NON_MSVC_INLINE
	#elif NIHILUS_COMPILER_CLANG
		#define NIHILUS_INLINE inline __attribute__((always_inline))
		#define NIHILUS_NON_MSVC_INLINE inline __attribute__((always_inline))
	#elif NIHILUS_COMPILER_GNUCXX
		#define NIHILUS_INLINE inline __attribute__((always_inline))
		#define NIHILUS_NON_MSVC_INLINE inline __attribute__((always_inline))
	#endif
#else
	#if NIHILUS_COMPILER_MSVC
		#define NIHILUS_INLINE [[msvc::noinline]]
		#define NIHILUS_NON_MSVC_INLINE [[msvc::noinline]]
	#elif NIHILUS_COMPILER_CLANG
		#define NIHILUS_INLINE noinline
		#define NIHILUS_NON_MSVC_INLINE noinline
	#elif NIHILUS_COMPILER_GNUCXX
		#define NIHILUS_INLINE noinline
		#define NIHILUS_NON_MSVC_INLINE noinline
	#endif
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

	struct cpu_arch_index_holder {
		static constexpr static_aligned_const cpu_arch_index_raw{ arch_indices[NIHILUS_CPU_INSTRUCTION_INDEX] };
		static constexpr const uint64_t& cpu_arch_index{ *cpu_arch_index_raw };
	};

	struct cpu_alignment_holder {
		static constexpr static_aligned_const cpu_alignment_raw{ cpu_alignments[NIHILUS_CPU_INSTRUCTION_INDEX] };
		static constexpr const uint64_t& cpu_alignment{ *cpu_alignment_raw };
	};

}
