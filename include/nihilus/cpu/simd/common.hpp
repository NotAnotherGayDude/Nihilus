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

namespace nihilus {

#if defined(NIHILUS_AVX512)
	using nihilus_simd_int_128		  = __m128i;
	using nihilus_simd_int_256		  = __m256i;
	using nihilus_simd_int_512		  = __m512i;
	using nihilus_simd_int_t		  = __m512i;
	using nihilus_string_parsing_type = uint64_t;
	inline static constexpr uint64_t bitsPerStep{ 512 };
#elif defined(NIHILUS_AVX2)
	using nihilus_simd_int_128		  = __m128i;
	using nihilus_simd_int_256		  = __m256i;
	using nihilus_simd_int_512		  = __m512i;
	using nihilus_simd_int_t		  = __m256i;
	using nihilus_string_parsing_type = uint32_t;
	inline static constexpr uint64_t bitsPerStep{ 256 };
#elif defined(NIHILUS_NEON)
	using nihilus_simd_int_128		  = uint8x16_t;
	using nihilus_simd_int_256		  = uint32_t;
	using nihilus_simd_int_512		  = uint64_t;
	using nihilus_simd_int_t		  = uint8x16_t;
	using nihilus_string_parsing_type = uint16_t;
	inline static constexpr uint64_t bitsPerStep{ 128 };
#elif defined(NIHILUS_SVE)
	using nihilus_simd_int_128		  = uint8x16_t;
	using nihilus_simd_int_256		  = uint32_t;
	using nihilus_simd_int_512		  = uint64_t;
	using nihilus_simd_int_t		  = uint8x16_t;
	using nihilus_string_parsing_type = uint16_t;
	inline static constexpr uint64_t bitsPerStep{ 2048 };
#else
	struct __m128x;
	using nihilus_simd_int_128 = __m128x;
	using nihilus_simd_int_256 = uint32_t;
	using nihilus_simd_int_512 = uint64_t;

	using nihilus_simd_int_t		  = __m128x;
	using nihilus_string_parsing_type = uint16_t;
	inline constexpr uint64_t bitsPerStep{ 128 };

#endif

	template<typename value_type>
	concept simd_int_512_type = std::is_same_v<nihilus_simd_int_512, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_256_type = std::is_same_v<nihilus_simd_int_256, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_128_type = std::is_same_v<nihilus_simd_int_128, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_type = std::is_same_v<nihilus_simd_int_t, std::remove_cvref_t<value_type>>;

#if defined(NIHILUS_AVX512) || defined(NIHILUS_AVX2)

	#define blsr(value) _blsr_u64(value)

	template<uint16_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
	#if defined(NIHILUS_LINUX)
		return __tzcnt_u16(value);
	#else
		return _tzcnt_u16(value);
	#endif
	}

	template<uint32_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
		return _tzcnt_u32(value);
	}

	template<uint64_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
		return _tzcnt_u64(value);
	}

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

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(value, other)));
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, other)));
	}

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_256_type simd_int_t01, simd_int_256_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

#endif

#if defined(NIHILUS_AVX512)

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

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(value, other));
	}

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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

	template<simd_int_512_type simd_int_t01, simd_int_512_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
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
#endif

#if defined(NIHILUS_NEON)

	#define blsr(value) (value & (value - 1))

	template<uint16_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
		if (value != 0) {
	#if NIHILUS_REGULAR_VISUAL_STUDIO
			return _tzcnt_u16(value);
	#else
			return __builtin_ctz(value);
	#endif
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint32_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
		if (value != 0) {
	#if NIHILUS_REGULAR_VISUAL_STUDIO
			return _tzcnt_u32(value);
	#else
			return __builtin_ctz(value);
	#endif
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint64_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
		if (value != 0) {
	#if NIHILUS_REGULAR_VISUAL_STUDIO
			return _tzcnt_u64(value);
	#else
			return __builtin_ctzll(value);
	#endif
		} else {
			return sizeof(value_type) * 8;
		}
	}

#else

	#define blsr(value) (value & (value - 1))

	template<uint_type value_type> NIHILUS_FORCE_INLINE value_type tzcnt(value_type value) noexcept {
		if (value == 0) {
			return sizeof(value_type) * 8;
		}

		value_type count{};
		while ((value & 1) == 0) {
			value >>= 1;
			++count;
		}

		return count;
	}

#endif

	template<typename value_type> NIHILUS_FORCE_INLINE value_type post_cmp_tzcnt(value_type value) noexcept {
		return tzcnt(value);
	}

}