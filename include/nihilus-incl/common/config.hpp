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

#include <nihilus-incl/cpu/simd/nihilus_thread_count.hpp>
#include <nihilus-incl/cpu/simd/nihilus_cpu_instructions.hpp>
#include <source_location>
#include <cstring>
#include <cstdint>
#include <utility>
#include <atomic>

#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define NIHILUS_PLATFORM_WINDOWS 1
#elif defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__)) || defined(TARGET_OS_MAC)
	#include <mach/mach.h>
	#define NIHILUS_PLATFORM_MAC 1
#elif defined(__ANDROID__)
	#define NIHILUS_PLATFORM_ANDROID 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define NIHILUS_PLATFORM_LINUX 1
#else
	#error "Unsupported platform"
#endif

#if defined(_MSC_VER)
	#define NIHILUS_COMPILER_MSVC 1
#elif defined(__clang__) || defined(__llvm__)
	#define NIHILUS_COMPILER_CLANG 1
#elif defined(__GNUC__) && !defined(__clang__)
	#define NIHILUS_COMPILER_GNUCXX 1
#else
	#error "Unsupported compiler"
#endif

#if defined(NDEBUG)
	#if defined(NIHILUS_COMPILER_MSVC)
		#define NIHILUS_INLINE [[msvc::forceinline]] inline
		#define NIHILUS_NON_MSVC_INLINE 
	#elif defined(NIHILUS_COMPILER_CLANG)
		#define NIHILUS_INLINE inline __attribute__((always_inline))
		#define NIHILUS_NON_MSVC_INLINE inline __attribute__((always_inline))
	#elif defined(NIHILUS_COMPILER_GNUCXX)
		#define NIHILUS_INLINE inline __attribute__((always_inline))
#define NIHILUS_NON_MSVC_INLINE inline __attribute__((always_inline))
	#endif
#else
	#if defined(NIHILUS_COMPILER_MSVC)
		#define NIHILUS_INLINE [[msvc::noinline]]
		#define NIHILUS_NON_MSVC_INLINE [[msvc::noinline]]
	#elif defined(NIHILUS_COMPILER_CLANG)
		#define NIHILUS_INLINE noinline
		#define NIHILUS_NON_MSVC_INLINE noinline
	#elif defined(NIHILUS_COMPILER_GNUCXX)
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

#if !defined(NIHILUS_ALIGN)
	#define NIHILUS_ALIGN(N) alignas(N)
#endif

#if NIHILUS_ARCH_X64
	#include <immintrin.h>
#elif NIHILUS_ARCH_ARM64
	#include <arm_sve.h>
	#include <arm_neon.h>
#else
	#error "Unsupported architecture"
#endif

NIHILUS_INLINE void nihilus_pause() noexcept {
#if NIHILUS_ARCH_X64
	_mm_pause();
#elif NIHILUS_ARCH_ARM64
	__asm__ __volatile__("yield" ::: "memory");
#else
	__asm__ __volatile__("" ::: "memory");
#endif
}

#ifndef NDEBUG
	#define NIHILUS_ASSERT(x) \
		if (!(x)) \
		internal_abort(#x, std::source_location::current())
#else
	#define NIHILUS_ASSERT(x)
#endif

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


inline std::atomic_uint64_t current_count{};
