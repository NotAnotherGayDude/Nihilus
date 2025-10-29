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

#include <nihilus-incl/common/utility.hpp>
#include <nihilus-incl/common/config.hpp>

namespace nihilus {

	union m128x {
		NIHILUS_HOST m128x(uint64_t argOne, uint64_t argTwo) noexcept {
			m128x_uint64[0] = argOne;
			m128x_uint64[1] = argTwo;
		}

		NIHILUS_HOST m128x() noexcept {
			m128x_uint64[0] = 0;
			m128x_uint64[1] = 0;
		}

#if NIHILUS_PLATFORM_WINDOWS
		int8_t m128x_int8[16]{};
		int16_t m128x_int16[8];
		int32_t m128x_int32[4];
		int64_t m128x_int64[2];
		uint8_t m128x_uint8[16];
		uint16_t m128x_uint16[8];
		uint32_t m128x_uint32[4];
		uint64_t m128x_uint64[2];
#else
		int64_t m128x_int64[2];
		int32_t m128x_int32[4];
		int16_t m128x_int16[8];
		int8_t m128x_int8[16]{};
		uint64_t m128x_uint64[2];
		uint32_t m128x_uint32[4];
		uint16_t m128x_uint16[8];
		uint8_t m128x_uint8[16];
#endif
	};

#if NIHILUS_AVX512 | NIHILUS_AVX2
	using nihilus_simd_int_128 = __m128i;
	struct nihilus_simd_int_128_t {
		using type = nihilus_simd_int_128;
	};
	using nihilus_simd_int_256 = __m256i;
	struct nihilus_simd_int_256_t {
		using type = nihilus_simd_int_256;
	};
	using nihilus_simd_int_512 = __m512i;
	struct nihilus_simd_int_512_t {
		using type = nihilus_simd_int_512;
	};

#elif NIHILUS_NEON
	using nihilus_simd_int_128 = uint8x16_t;
	struct nihilus_simd_int_128_t {
		using type = nihilus_simd_int_128;
	};
	using nihilus_simd_int_256 = uint32_t;
	struct nihilus_simd_int_256_t {
		using type = nihilus_simd_int_256;
	};
	using nihilus_simd_int_512 = uint64_t;
	struct nihilus_simd_int_512_t {
		using type = nihilus_simd_int_512;
	};
#elif NIHILUS_SVE2
	using nihilus_simd_int_128 = svint8_t;
	struct nihilus_simd_int_128_t {
		using type = nihilus_simd_int_128;
	};
	using nihilus_simd_int_256 = svint16_t;
	struct nihilus_simd_int_256_t {
		using type = nihilus_simd_int_256;
	};
	using nihilus_simd_int_512 = svint32_t;
	struct nihilus_simd_int_512_t {
		using type = nihilus_simd_int_512;
	};
#else
	using nihilus_simd_int_128 = m128x;
	struct nihilus_simd_int_128_t {
		using type = nihilus_simd_int_128;
	};
	using nihilus_simd_int_256 = uint32_t;
	struct nihilus_simd_int_256_t {
		using type = nihilus_simd_int_256;
	};
	using nihilus_simd_int_512 = uint64_t;
	struct nihilus_simd_int_512_t {
		using type = nihilus_simd_int_512;
	};

#endif

#if NIHILUS_ARCH_ARM64
	using avx_list = detail::type_list<detail::type_holder<16, nihilus_simd_int_128_t, uint64_t, std::numeric_limits<uint64_t>::max()>,
		detail::type_holder<32, nihilus_simd_int_256_t, uint32_t, std::numeric_limits<uint32_t>::max()>,
		detail::type_holder<64, nihilus_simd_int_512_t, uint64_t, std::numeric_limits<uint64_t>::max()>>;

	template<uint16_types value_type> NIHILUS_HOST static value_type tzcnt(const value_type value) noexcept {
		if (value != 0) {
			return static_cast<value_type>(__builtin_ctz(value));
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint32_types value_type> NIHILUS_HOST static value_type tzcnt(const value_type value) noexcept {
		if (value != 0) {
			return __builtin_ctz(static_cast<int32_t>(value));
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint64_types value_type> NIHILUS_HOST static value_type tzcnt(const value_type value) noexcept {
		if (value != 0) {
			return __builtin_ctzll(static_cast<int64_t>(value));
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint16_types value_type> NIHILUS_HOST static value_type lzcnt(const value_type value) noexcept {
		return static_cast<uint16_t>(__builtin_clz(value));
	}

	template<uint32_types value_type> NIHILUS_HOST static value_type lzcnt(const value_type value) noexcept {
		return __builtin_clz(value);
	}

	template<uint64_types value_type> NIHILUS_HOST static value_type lzcnt(const value_type value) noexcept {
		return __builtin_clzll(value);
	}

#else
	using avx_list = detail::type_list<detail::type_holder<16, nihilus_simd_int_128_t, uint16_t, std::numeric_limits<uint16_t>::max()>,
		detail::type_holder<32, nihilus_simd_int_256_t, uint32_t, std::numeric_limits<uint32_t>::max()>,
		detail::type_holder<64, nihilus_simd_int_512_t, uint64_t, std::numeric_limits<uint64_t>::max()>>;

	template<uint16_types value_type> NIHILUS_HOST static value_type tzcnt(const value_type value) noexcept {
	#if NIHILUS_PLATFORM_LINUX
		return __tzcnt_u16(value);
	#else
		return _tzcnt_u16(value);
	#endif
	}

	template<uint32_types value_type> NIHILUS_HOST static value_type tzcnt(const value_type value) noexcept {
		return _tzcnt_u32(value);
	}

	template<uint64_types value_type> NIHILUS_HOST static value_type tzcnt(const value_type value) noexcept {
		return _tzcnt_u64(value);
	}

	template<uint16_types value_type> NIHILUS_HOST static value_type lzcnt(const value_type value) noexcept {
		return static_cast<uint16_t>(_lzcnt_u32(value));
	}

	template<uint32_types value_type> NIHILUS_HOST static value_type lzcnt(const value_type value) noexcept {
		return _lzcnt_u32(value);
	}

	template<uint64_types value_type> NIHILUS_HOST static value_type lzcnt(const value_type value) noexcept {
		return _lzcnt_u64(value);
	}

#endif

	template<typename value_type>
	concept nihilus_simd_512_types = detail::same_as<nihilus_simd_int_512_t, detail::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept nihilus_simd_256_types = detail::same_as<nihilus_simd_int_256_t, detail::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept nihilus_simd_128_types = detail::same_as<nihilus_simd_int_128_t, detail::remove_cvref_t<value_type>>;

	template<uint_types value_type> NIHILUS_HOST static constexpr value_type tzcnt_constexpr(value_type value) noexcept {
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

#if NIHILUS_AVX512 | NIHILUS_AVX2

	NIHILUS_HOST static half fp32_to_fp16(float f) {
		return static_cast<half>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
	}

	NIHILUS_HOST static float fp16_to_fp32(half h) {
		return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_shuffle(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm_shuffle_epi8(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01>
	NIHILUS_HOST static auto op_cmp_eq_bitmask(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(value, other)));
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_bitmask(const typename simd_int_t01::type& value) noexcept {
		return static_cast<uint16_t>(_mm_movemask_epi8(value));
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return _mm_load_si128(static_cast<const __m128i*>(str));
	}

	template<nihilus_simd_128_types simd_int_t01, typename char_type>
	NIHILUS_HOST static auto op_set1(char_type other) noexcept {
		return _mm_set1_epi8(other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm_xor_si128(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_sub(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm_sub_epi8(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_or(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm_or_si128(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_test(const typename simd_int_t01::type& value) noexcept {
		return !_mm_testz_si128(value, value);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST auto op_cmp_eq(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(value, other)));
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_shuffle(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm256_shuffle_epi8(value, other);
	}

	template<nihilus_simd_256_types simd_int_t01>
	NIHILUS_HOST static auto op_cmp_eq_bitmask(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, other)));
	}

	template<nihilus_simd_256_types simd_int_t01>
	NIHILUS_HOST static auto op_bitmask(const typename simd_int_t01::type& value) noexcept {
		return static_cast<uint32_t>(_mm256_movemask_epi8(value));
	}

	template<nihilus_simd_256_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return _mm256_load_si256(static_cast<const __m256i*>(str));
	}

	template<nihilus_simd_256_types simd_int_t01, typename char_type> NIHILUS_HOST static auto op_set1(char_type other) noexcept {
		return _mm256_set1_epi8(other);
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm256_xor_si256(value, other);
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_sub(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm256_sub_epi8(value, other);
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_or(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm256_or_si256(value, other);
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_test(const typename simd_int_t01::type& value) noexcept {
		return !_mm256_testz_si256(value, value);
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_cmp_eq(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, other)));
	}

#endif

#if NIHILUS_AVX512

	template<nihilus_simd_512_types simd_int_t01> NIHILUS_HOST static auto op_shuffle(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm512_shuffle_epi8(value, other);
	}

	template<nihilus_simd_512_types simd_int_t01>
	inline static auto op_cmp_eq_bitmask(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(value, other));
	}

	template<nihilus_simd_512_types simd_int_t01> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return _mm512_load_si512(static_cast<const __m512i*>(str));
	}

	template<nihilus_simd_512_types simd_int_t01> NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm512_xor_si512(value, other);
	}

	template<nihilus_simd_512_types simd_int_t01> NIHILUS_HOST static auto op_test(const typename simd_int_t01::type& value) noexcept {
		return !_mm512_testn_epi64_mask(value, value);
	}

	template<nihilus_simd_512_types simd_int_t01> NIHILUS_HOST static auto op_cmp_eq(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(value, other));
	}

#endif

#if NIHILUS_NEON

	NIHILUS_HOST static half fp32_to_fp16(float f) {
		return static_cast<half>(static_cast<__fp16>(f));
	}

	NIHILUS_HOST float sqrtf_fast(float x) {
		return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0);
	}

	NIHILUS_HOST static float fp16_to_fp32(half h) {
		return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_s16(vdup_n_s16(h))), 0);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_shuffle(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		const auto bit_mask{ vdupq_n_u8(0x0F) };
		return vqtbl1q_u8(value, vandq_u8(other, bit_mask));
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_bit_mask(const typename simd_int_t01::type& value) noexcept {
		constexpr uint8x16_t bit_mask{ 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80 };
		const auto minput = value & bit_mask;
		uint8x16_t tmp	  = vpaddq_u8(minput, minput);
		tmp				  = vpaddq_u8(tmp, tmp);
		tmp				  = vpaddq_u8(tmp, tmp);
		return static_cast<uint16_t>(vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0));
	}

	template<nihilus_simd_128_types simd_int_t01>
	NIHILUS_HOST static auto op_cmp_eq_bitmask(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return op_bit_mask<simd_int_t01>(vceqq_u8(value, other));
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_cmp_eq(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vceqq_u8(value, other), 4)), 0);
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return vld1q_u8(static_cast<const uint8_t*>(str));
	}

	template<nihilus_simd_128_types simd_int_t01>
	NIHILUS_HOST static nihilus_simd_int_128 op_xor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return veorq_u8(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static bool op_test(const typename simd_int_t01::type& value) noexcept {
		return vmaxvq_u8(value) != 0;
	}

#endif

#if NIHILUS_SVE2

	NIHILUS_HOST static half fp32_to_fp16_sve2(float f) {
		return static_cast<half>(svextract_f16(svcvt_f16_f32_z(svptrue_b32(), svdup_n_f32(f)), 0));
	}

	NIHILUS_HOST float sqrtf_fast_sve2(float x) {
		return svextract_f32(svsqrt_f32_z(svptrue_b32(), svdup_n_f32(x)), 0);
	}

	NIHILUS_HOST static float fp16_to_fp32(half h) {
		return svextract_f32(svcvt_f32_f16_z(svptrue_b16(), svreinterpret_f16_u16(svdup_n_u16(h))), 0);
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return svld1_s8(svptrue_b8(), static_cast<const int8_t*>(str));
	}
	template<nihilus_simd_128_types simd_int_t01, nihilus_simd_128_types simd_int_t02>
	NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return sveor_s8_z(svptrue_b8(), value, other);
	}
	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static auto op_test(const typename simd_int_t01::type& value) noexcept {
		return svptest_any(svptrue_b8(), svcmpne_s8(svptrue_b8(), value, svdup_s8(0)));
	}

	template<nihilus_simd_256_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return svld1_s16(svptrue_b16(), static_cast<const int16_t*>(str));
	}
	template<nihilus_simd_256_types simd_int_t01, nihilus_simd_256_types simd_int_t02>
	NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return sveor_s16_z(svptrue_b16(), value, other);
	}
	template<nihilus_simd_256_types simd_int_t01> NIHILUS_HOST static auto op_test(const typename simd_int_t01::type& value) noexcept {
		return svptest_any(svptrue_b16(), svcmpne_s16(svptrue_b16(), value, svdup_s16(0)));
	}

	template<nihilus_simd_512_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		return svld1_s32(svptrue_b32(), static_cast<const int32_t*>(str));
	}
	template<nihilus_simd_512_types simd_int_t01, nihilus_simd_512_types simd_int_t02>
	NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return sveor_s32_z(svptrue_b32(), value, other);
	}
	template<nihilus_simd_512_types simd_int_t01> NIHILUS_HOST static auto op_test(const typename simd_int_t01::type& value) noexcept {
		return svptest_any(svptrue_b32(), svcmpne_s32(svptrue_b32(), value, svdup_s32(0)));
	}

#endif

#if !NIHILUS_AVX512 & !NIHILUS_AVX2 & !NIHILUS_SVE2 & !NIHILUS_NEON

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_HOST static auto gather_values(const void* str) noexcept {
		m128x return_value{};
		return_value.m128x_int8[0]	= static_cast<const int8_t*>(str)[0];
		return_value.m128x_int8[1]	= static_cast<const int8_t*>(str)[1];
		return_value.m128x_int8[2]	= static_cast<const int8_t*>(str)[2];
		return_value.m128x_int8[3]	= static_cast<const int8_t*>(str)[3];
		return_value.m128x_int8[4]	= static_cast<const int8_t*>(str)[4];
		return_value.m128x_int8[5]	= static_cast<const int8_t*>(str)[5];
		return_value.m128x_int8[6]	= static_cast<const int8_t*>(str)[6];
		return_value.m128x_int8[7]	= static_cast<const int8_t*>(str)[7];
		return_value.m128x_int8[8]	= static_cast<const int8_t*>(str)[8];
		return_value.m128x_int8[9]	= static_cast<const int8_t*>(str)[9];
		return_value.m128x_int8[10] = static_cast<const int8_t*>(str)[10];
		return_value.m128x_int8[11] = static_cast<const int8_t*>(str)[11];
		return_value.m128x_int8[12] = static_cast<const int8_t*>(str)[12];
		return_value.m128x_int8[13] = static_cast<const int8_t*>(str)[13];
		return_value.m128x_int8[14] = static_cast<const int8_t*>(str)[14];
		return_value.m128x_int8[15] = static_cast<const int8_t*>(str)[15];
		return return_value;
	}

	template<nihilus_simd_128_types simd_int_t01, nihilus_simd_128_types simd_int_t02>
	NIHILUS_HOST static auto op_xor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		m128x result{};
		result.m128x_uint64[0] = value.m128x_uint64[0] ^ other.m128x_uint64[0];
		result.m128x_uint64[1] = value.m128x_uint64[1] ^ other.m128x_uint64[1];
		return result;
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_HOST static bool op_test(const typename simd_int_t01::type& value) noexcept {
		return (value.m128x_uint64[0] != 0) || (value.m128x_uint64[1] != 0);
	}

	NIHILUS_HOST static half fp32_to_fp16(float f) {
		const uint32_t b = std::bit_cast<uint32_t>(f) + 0x00001000;
		const uint32_t e = (b & 0x7F800000) >> 23;
		const uint32_t m = b & 0x007FFFFF;
		return static_cast<half>((b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | (m >> 13)) |
			((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF);
	}

	NIHILUS_HOST float sqrtf_fast(float x) {
		return std::bit_cast<float>(((std::bit_cast<uint32_t>(x) + 0x3f800000) >> 1) + 0x20000000);
	}

	NIHILUS_HOST static constexpr float fp32_from_bits(uint32_t w) noexcept {
		return std::bit_cast<float>(w);
	}

	NIHILUS_HOST static constexpr uint32_t fp32_to_bits(float f) noexcept {
		return std::bit_cast<uint32_t>(f);
	}

	NIHILUS_HOST static constexpr float compute_fp16_to_fp32(half h) noexcept {
		const uint32_t w	 = static_cast<uint32_t>(h) << 16;
		const uint32_t sign	 = w & 0x80000000u;
		const uint32_t two_w = w + w;

		constexpr uint32_t exp_offset = 0xE0u << 23;
		constexpr float exp_scale	  = fp32_from_bits(0x7800000u);
		const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

		constexpr uint32_t magic_mask  = 126u << 23;
		constexpr float magic_bias	   = 0.5f;
		const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

		constexpr uint32_t denormalized_cutoff = 1u << 27;
		const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
		return fp32_from_bits(result);
	}

	NIHILUS_ALIGN(64)
	static static_aligned_const<float>* __restrict fp16_to_fp32_array{ []() {
		NIHILUS_ALIGN(64) static array<static_aligned_const<float>, (1 << 16)> return_values_new{};
		for (uint64_t i = 0; i < (1 << 16); ++i) {
			return_values_new[i] = static_aligned_const<float>{ compute_fp16_to_fp32(static_cast<half>(i)) };
		}
		return return_values_new.data();
	}() };

	NIHILUS_HOST static float fp16_to_fp32(half f) {
		return fp16_to_fp32_array[f];
	}

#endif
}
