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

#include <nihilus-incl/cpu/simd/avx_2.hpp>
#include <nihilus-incl/cpu/simd/avx_512.hpp>
#include <nihilus-incl/cpu/simd/arm_neon.hpp>
#include <nihilus-incl/cpu/simd/arm_sve2.hpp>

namespace nihilus {

#if !defined(NIHILUS_AVX2) && !defined(NIHILUS_AVX512) && !defined(NIHILUS_NEON) && !defined(NIHILUS_SVE2)

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
		}
	};

#endif

}
