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

#include <nihilus/common/kernel_traits.hpp>
#include <nihilus/cpu/simd/common.hpp>

#if defined(NIHILUS_AVX512)

namespace nihilus {

	template<simd_int_512_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_values(const void* str) noexcept {
		return _mm512_load_si512(static_cast<const __m512i*>(str));
	}

	template<simd_int_512_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_valuesU(const void* str, void* str2) noexcept {
		std::memcpy(str2, str, sizeof(simd_int_type_new));
		return _mm512_load_si512(static_cast<const __m512i*>(str2));
	}

	template<simd_int_512_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 8)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(const char_t value) noexcept {
		return _mm512_set1_epi64(static_cast<const int64_t>(value));
	}

	template<simd_int_512_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 1)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(const char_t value) noexcept {
		return _mm512_set1_epi8(static_cast<const char>(value));
	}

	template<simd_int_512_type simd_int_type_new> NIHILUS_FORCE_INLINE static void store(const simd_int_type_new& value, void* storageLocation) noexcept {
		_mm512_store_si512(static_cast<__m512i*>(storageLocation), value);
	}

	template<simd_int_512_type simd_int_type_new> NIHILUS_FORCE_INLINE static void storeU(const simd_int_type_new& value, void* storageLocation, void* storageLocation02) noexcept {
		_mm512_store_si512(static_cast<__m512i*>(storageLocation), value);
		std::memcpy(storageLocation02, storageLocation, sizeof(simd_int_type_new));
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(value, other));
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint64_t>(_mm512_cmpgt_epi8_mask(other, value));
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_maskz_set1_epi8(_mm512_cmpeq_epi8_mask(value, other), 0xFF);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLtRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_maskz_set1_epi8(_mm512_cmpeq_epi8_mask(value, other), 0xFF);
	}

	template<simd_int_256_type simd_int_t01> NIHILUS_FORCE_INLINE static uint64_t opBitMask(const simd_int_t01& value) noexcept {
		return _mm512_movepi8_mask(value);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqBitMask(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(value, other));
	}

	template<simd_int_512_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opBitMaskRaw(const simd_int_t01& value) noexcept {
		return _mm512_movepi8_mask(value);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_shuffle_epi8(value, other);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opXor(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_xor_si512(value, other);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opAnd(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_and_si512(value, other);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opOr(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_or_si512(value, other);
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm512_andnot_si512(other, value);
	}

	template<simd_int_512_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opTest(const simd_int_t01& value) noexcept {
		return !_mm512_test_epi64_mask(value, value);
	}

	template<simd_int_512_type simd_type> NIHILUS_FORCE_INLINE static nihilus_simd_int_512 opSetLSB(const simd_type& value, bool value_new) noexcept {
		const nihilus_simd_int_512 mask{ _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 0x01u) };
		return value_new ? _mm512_or_si512(value, mask) : _mm512_andnot_si512(mask, value);
	}

	template<simd_int_512_type simd_type> NIHILUS_FORCE_INLINE static bool opGetMSB(const simd_type& value) noexcept {
		const nihilus_simd_int_512 result = _mm512_and_si512(value, _mm512_set_epi64(0x8000000000000000ll, 0x00ll, 0x00ll, 0x00ll, 0x00ll, 0x00ll, 0x00ll, 0x00ll));
		return !_mm512_test_epi64_mask(result, result);
	}

	template<simd_int_512_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opNot(const simd_int_t01& value) noexcept {
		return _mm512_xor_si512(value, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFll));
	}


	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02, const typename core_type::input_type03& input03) {
		}
	};

};

#endif
