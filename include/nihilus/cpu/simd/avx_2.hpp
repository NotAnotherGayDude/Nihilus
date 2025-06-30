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

#if defined(NIHILUS_AVX2)

namespace nihilus {

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_values(const void* str) noexcept {
		return _mm_load_si128(static_cast<const __m128i*>(str));
	}

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_valuesU(const void* str, void* str2) noexcept {
		std::memcpy(str2, str, sizeof(simd_int_type_new));
		return _mm_load_si128(static_cast<const __m128i*>(str2));
	}

	template<simd_int_128_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 8)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(const char_t str) noexcept {
		return _mm_set1_epi64x(static_cast<const int64_t>(str));
	}

	template<simd_int_128_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 1)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(const char_t str) noexcept {
		return _mm_set1_epi8(static_cast<const char>(str));
	}

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static void store(const simd_int_type_new& value, void* storageLocation) noexcept {
		_mm_store_si128(static_cast<__m128i*>(storageLocation), value);
	}

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static void storeU(const simd_int_type_new& value, void* storageLocation, void* storageLocation02) noexcept {
		_mm_store_si128(static_cast<__m128i*>(storageLocation), value);
		std::memcpy(storageLocation02, storageLocation, sizeof(simd_int_type_new));
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(value, other)));
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		const nihilus_simd_int_128 offset = _mm_set1_epi8(static_cast<char>(0x80));
		return static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_add_epi8(other, offset), _mm_add_epi8(value, offset))));
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm_cmpeq_epi8(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLtRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		const nihilus_simd_int_128 offset = _mm_set1_epi8(static_cast<char>(0x80));
		return _mm_cmpgt_epi8(_mm_add_epi8(other, offset), _mm_add_epi8(value, offset));
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static uint32_t opBitMask(const simd_int_t01& value) noexcept {
		return _mm_movemask_epi8(value);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqBitMask(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(value, other)));
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opBitMaskRaw(const simd_int_t01& value) noexcept {
		return _mm_movemask_epi8(value);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm_shuffle_epi8(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opXor(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm_xor_si128(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opAnd(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm_and_si128(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opOr(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm_or_si128(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm_andnot_si128(other, value);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opTest(const simd_int_t01& value) noexcept {
		return !_mm_testz_si128(value, value);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opNot(const simd_int_t01& value) noexcept {
		return _mm_xor_si128(value, _mm_set1_epi64x(0xFFFFFFFFFFFFFFFFll));
	}

	template<simd_int_128_type simd_type> NIHILUS_FORCE_INLINE static nihilus_simd_int_128 opSetLSB(const simd_type& value, bool value_new) noexcept {
		const nihilus_simd_int_128 mask{ _mm_set_epi64x(0, 0x01u) };
		return value_new ? _mm_or_si128(value, mask) : _mm_andnot_si128(mask, value);
	}

	template<simd_int_128_type simd_type> NIHILUS_FORCE_INLINE static bool opGetMSB(const simd_type& value) noexcept {
		const nihilus_simd_int_128 result = _mm_and_si128(value, _mm_set_epi64x(0x8000000000000000ll, 0x00ll));
		return !_mm_testz_si128(result, result);
	}

	template<simd_int_256_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_values(const void* str) noexcept {
		return _mm256_load_si256(static_cast<const __m256i*>(str));
	}

	template<simd_int_256_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_valuesU(const void* str, void* str2) noexcept {
		std::memcpy(str2, str, sizeof(simd_int_type_new));
		return _mm256_load_si256(static_cast<const __m256i*>(str2));
	}

	template<simd_int_256_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_valuesU(const void* str) noexcept {
		return _mm256_loadu_si256(static_cast<const __m256i*>(str));
	}

	template<simd_int_256_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 8)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(const char_t value) noexcept {
		return _mm256_set1_epi64x(static_cast<const int64_t>(value));
	}

	template<simd_int_256_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 1)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(const char_t value) noexcept {
		return _mm256_set1_epi8(static_cast<const char>(value));
	}

	template<simd_int_256_type simd_int_type_new> NIHILUS_FORCE_INLINE static void store(const simd_int_type_new& value, void* storageLocation) noexcept {
		_mm256_store_si256(static_cast<__m256i*>(storageLocation), value);
	}

	template<simd_int_256_type simd_int_type_new> NIHILUS_FORCE_INLINE static void storeU(const simd_int_type_new& value, void* storageLocation, void* storageLocation02) noexcept {
		_mm256_store_si256(static_cast<__m256i*>(storageLocation), value);
		std::memcpy(storageLocation02, storageLocation, sizeof(simd_int_type_new));
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, other)));
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		const nihilus_simd_int_256 offset = _mm256_set1_epi8(static_cast<char>(0x80));
		return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpgt_epi8(_mm256_add_epi8(other, offset), _mm256_add_epi8(value, offset))));
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm256_cmpeq_epi8(value, other);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLtRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		const nihilus_simd_int_256 offset = _mm256_set1_epi8(static_cast<char>(0x80));
		return _mm256_cmpgt_epi8(_mm256_add_epi8(other, offset), _mm256_add_epi8(value, offset));
	}

	template<simd_int_256_type simd_int_t01> NIHILUS_FORCE_INLINE static uint32_t opBitMask(const simd_int_t01& value) noexcept {
		return _mm256_movemask_epi8(value);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqBitMask(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, other)));
	}

	template<simd_int_256_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opBitMaskRaw(const simd_int_t01& value) noexcept {
		return _mm256_movemask_epi8(value);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm256_shuffle_epi8(value, other);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opXor(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm256_xor_si256(value, other);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opAnd(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm256_and_si256(value, other);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opOr(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm256_or_si256(value, other);
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return _mm256_andnot_si256(other, value);
	}

	template<simd_int_256_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opTest(const simd_int_t01& value) noexcept {
		return !_mm256_testz_si256(value, value);
	}

	template<simd_int_256_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opNot(const simd_int_t01& value) noexcept {
		return _mm256_xor_si256(value, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFll));
	}

	template<simd_int_256_type simd_type> NIHILUS_FORCE_INLINE static nihilus_simd_int_256 opSetLSB(const simd_type& value, bool value_new) noexcept {
		const nihilus_simd_int_256 mask{ _mm256_set_epi64x(0, 0, 0, 0x01u) };
		return value_new ? _mm256_or_si256(value, mask) : _mm256_andnot_si256(mask, value);
	}

	template<simd_int_256_type simd_type> NIHILUS_FORCE_INLINE static bool opGetMSB(const simd_type& value) noexcept {
		const nihilus_simd_int_256 result = _mm256_and_si256(value, _mm256_set_epi64x(0x8000000000000000ll, 0x00ll, 0x00ll, 0x00ll));
		return !_mm256_testz_si256(result, result);
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm_mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t, uint64_t, core_type&, const typename core_type::input_type01&,
			const typename core_type::input_type02&, const typename core_type::input_type03& ) {
		}
	};

};

#endif
