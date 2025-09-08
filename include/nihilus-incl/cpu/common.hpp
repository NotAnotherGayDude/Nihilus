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

#include <nihilus-incl/cpu/fallback.hpp>
#include <nihilus-incl/cpu/avx_2.hpp>
#include <nihilus-incl/cpu/avx_512.hpp>
#include <nihilus-incl/cpu/arm_neon.hpp>
#include <nihilus-incl/cpu/arm_sve2.hpp>

namespace nihilus {

	union m128x {
		NIHILUS_INLINE m128x(uint64_t argOne, uint64_t argTwo) noexcept {
			m128x_uint64[0] = argOne;
			m128x_uint64[1] = argTwo;
		}

		NIHILUS_INLINE m128x() noexcept {
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

#if NIHILUS_AVX512
	using nihilus_simd_int_128 = __m128i;
	struct nihilus_simd_int_128_t {
		using type = __m128i;
	};
	using nihilus_simd_int_256 = __m256i;
	struct nihilus_simd_int_256_t {
		using type = __m256i;
	};
	using nihilus_simd_int_512 = __m512i;
	struct nihilus_simd_int_512_t {
		using type = __m512i;
	};
	using nihilus_simd_int_t		  = __m256i;
	using nihilus_string_parsing_type = uint64_t;
#elif NIHILUS_AVX2
	using nihilus_simd_int_128 = __m128i;
	struct nihilus_simd_int_128_t {
		using type = __m128i;
	};
	using nihilus_simd_int_256 = __m256i;
	struct nihilus_simd_int_256_t {
		using type = __m256i;
	};
	using nihilus_simd_int_512 = __m512i;
	struct nihilus_simd_int_512_t {
		using type = __m512i;
	};
	using nihilus_simd_int			  = __m256i;
	using nihilus_string_parsing_type = uint32_t;
#elif NIHILUS_NEON
	using nihilus_simd_int_128 = uint8x16_t;
	struct nihilus_simd_int_128_t {
		using type = uint8x16_t;
	};
	using nihilus_simd_int_256 = uint32_t;
	struct nihilus_simd_int_256_t {
		using type = uint32_t;
	};
	using nihilus_simd_int_512 = uint64_t;
	struct nihilus_simd_int_512_t {
		using type = uint64_t;
	};
	using nihilus_simd_int_t		  = uint8x16_t;
	using nihilus_string_parsing_type = uint16_t;
#elif NIHILUS_SVE
	using nihilus_simd_int_128 = svint8_t;
	struct nihilus_simd_int_128_t {
		using type = svint8_t;
	};
	using nihilus_simd_int_256 = svint16_t;
	struct nihilus_simd_int_256_t {
		using type = svint16_t;
	};
	using nihilus_simd_int_512 = svint32_t;
	struct nihilus_simd_int_512_t {
		using type = svint32_t;
	};
	using nihilus_simd_int_t		  = svint16_t;
	using nihilus_string_parsing_type = uint64_t;
#else
	using nihilus_simd_int_128 = m128x;
	struct nihilus_simd_int_128_t {
		using type = m128x;
	};
	using nihilus_simd_int_256 = uint32_t;
	struct nihilus_simd_int_256_t {
		using type = uint32_t;
	};
	using nihilus_simd_int_512 = uint64_t;
	struct nihilus_simd_int_512_t {
		using type = uint64_t;
	};
	using nihilus_simd_int_t		  = m128x;
	using nihilus_string_parsing_type = uint16_t;

#endif

	template<typename value_type>
	concept nihilus_simd_512_types = std::same_as<nihilus_simd_int_512_t, detail::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept nihilus_simd_256_types = std::same_as<nihilus_simd_int_256_t, detail::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept nihilus_simd_128_types = std::same_as<nihilus_simd_int_128_t, detail::remove_cvref_t<value_type>>;

	template<uint_types value_type> NIHILUS_INLINE static constexpr value_type tzcnt_constexpr(value_type value) noexcept {
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

#if NIHILUS_AVX512 || NIHILUS_AVX2

	NIHILUS_INLINE static half fp32_to_fp16(float f) {
		return static_cast<half>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0));
	}

	NIHILUS_INLINE float sqrtf_fast(float x) {
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
	}

	NIHILUS_INLINE static float fp16_to_fp32(half h) {
		return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
	}

	#define blsr(value) _blsr_u64(value)

	NIHILUS_INLINE static uint32_t lzcnt(const uint32_t value) noexcept {
		return _lzcnt_u32(value);
	}

	NIHILUS_INLINE static uint64_t lzcnt(const uint64_t value) noexcept {
		return _lzcnt_u64(value);
	}

	template<uint16_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
	#if NIHILUS_PLATFORM_LINUX
		return __tzcnt_u16(value);
	#else
		return _tzcnt_u16(value);
	#endif
	}

	template<uint32_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
		return _tzcnt_u32(value);
	}

	template<uint64_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
		return _tzcnt_u64(value);
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return _mm_load_si128(static_cast<const __m128i*>(str));
	}

	template<nihilus_simd_128_types simd_int_t01, nihilus_simd_128_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return _mm_xor_si128(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return !_mm_testz_si128(value, value);
	}

	template<nihilus_simd_256_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return _mm256_load_si256(static_cast<const __m256i*>(str));
	}

	template<nihilus_simd_256_types simd_int_t01, nihilus_simd_256_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return _mm256_xor_si256(value, other);
	}

	template<nihilus_simd_256_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return !_mm256_testz_si256(value, value);
	}

#endif

#if NIHILUS_AVX512

	template<nihilus_simd_512_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return _mm512_load_si512(static_cast<const __m512i*>(str));
	}

	template<nihilus_simd_512_types simd_int_t01, nihilus_simd_512_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return _mm512_xor_si512(value, other);
	}

	template<nihilus_simd_512_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return !_mm512_test_epi64_mask(value, value);
	}

#endif

#if NIHILUS_NEON

	NIHILUS_INLINE static half fp32_to_fp16(float f) {
		return static_cast<half>(static_cast<__fp16>(f));
	}

	NIHILUS_INLINE float sqrtf_fast(float x) {
		return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0);
	}

	NIHILUS_INLINE static float fp16_to_fp32(half h) {
		return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
	}

	#define blsr(value) (value & (value - 1))

	template<uint16_types value_type> NIHILUS_INLINE value_type tzcnt(value_type value) noexcept {
		if (value != 0) {
			return __builtin_ctz(value);
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint32_types value_type> NIHILUS_INLINE value_type tzcnt(value_type value) noexcept {
		if (value != 0) {
			return __builtin_ctz(value);
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<uint64_types value_type> NIHILUS_INLINE value_type tzcnt(value_type value) noexcept {
		if (value != 0) {
			return __builtin_ctzll(value);
		} else {
			return sizeof(value_type) * 8;
		}
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return vld1q_u8(static_cast<const uint8_t*>(str));
	}

	template<nihilus_simd_128_types simd_int_t01, nihilus_simd_128_types simd_int_t02>
	NIHILUS_INLINE static nihilus_simd_int_128 opXor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return veorq_u8(value, other);
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_INLINE static bool opTest(const typename simd_int_t01::type& value) noexcept {
		return vmaxvq_u8(value) != 0;
	}

#endif

#if NIHILUS_SVE2

	NIHILUS_INLINE static half fp32_to_fp16_sve2(float f) {
		return static_cast<half>(svextract_f16(svcvt_f16_f32_z(svptrue_b32(), svdup_n_f32(f)), 0));
	}

	NIHILUS_INLINE float sqrtf_fast_sve2(float x) {
		return svextract_f32(svsqrt_f32_z(svptrue_b32(), svdup_n_f32(x)), 0);
	}

	NIHILUS_INLINE static float fp16_to_fp32(half h) {
		return svextract_f32(svcvt_f32_f16_z(svptrue_b16(), svreinterpret_f16_u16(svdup_n_u16(h))), 0);
	}

	#define blsr(value) __builtin_aarch64_rbitl(__builtin_clzl(__builtin_aarch64_rbitl(value) | 1))

	NIHILUS_INLINE static uint32_t lzcnt(const uint32_t value) noexcept {
		return __builtin_clz(value);
	}
	NIHILUS_INLINE static uint64_t lzcnt(const uint64_t value) noexcept {
		return __builtin_clzl(value);
	}

	template<uint16_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
		return __builtin_ctz(value);
	}
	template<uint32_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
		return __builtin_ctz(value);
	}
	template<uint64_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
		return __builtin_ctzl(value);
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return svld1_s8(svptrue_b8(), static_cast<const int8_t*>(str));
	}
	template<nihilus_simd_128_types simd_int_t01, nihilus_simd_128_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t01::type& other) noexcept {
		return sveor_s8_z(svptrue_b8(), value, other);
	}
	template<nihilus_simd_128_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return svptest_any(svptrue_b8(), svcmpne_s8(svptrue_b8(), value, svdup_s8(0)));
	}

	template<nihilus_simd_256_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return svld1_s16(svptrue_b16(), static_cast<const int16_t*>(str));
	}
	template<nihilus_simd_256_types simd_int_t01, nihilus_simd_256_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return sveor_s16_z(svptrue_b16(), value, other);
	}
	template<nihilus_simd_256_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return svptest_any(svptrue_b16(), svcmpne_s16(svptrue_b16(), value, svdup_s16(0)));
	}

	template<nihilus_simd_512_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return svld1_s32(svptrue_b32(), static_cast<const int32_t*>(str));
	}
	template<nihilus_simd_512_types simd_int_t01, nihilus_simd_512_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return sveor_s32_z(svptrue_b32(), value, other);
	}
	template<nihilus_simd_512_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return svptest_any(svptrue_b32(), svcmpne_s32(svptrue_b32(), value, svdup_s32(0)));
	}

#endif

#if !NIHILUS_AVX512 && !NIHILUS_AVX2 && !NIHILUS_NEON && !NIHILUS_SVE2

	template<uint_types value_type> NIHILUS_INLINE static constexpr value_type lzcnt(const value_type value) noexcept {
		if (value == 0) {
			return sizeof(value_type) * 8;
		}

		value_type count{};
		value_type mask{ static_cast<value_type>(1) << (std::numeric_limits<value_type>::digits - 1) };

		while ((value & mask) == 0) {
			++count;
			mask >>= 1;
		}

		return count;
	}

	template<typename value_type>
		requires(sizeof(value_type) == 8)
	NIHILUS_INLINE m128x mm128LoadUSi128(const value_type* ptr) noexcept {
		m128x returnValues{};
		returnValues.m128x_uint64[0] = ptr[0];
		returnValues.m128x_uint64[1] = ptr[1];
		return returnValues;
	}

	NIHILUS_INLINE m128x mm128LoadUSi128(const m128x* ptr) noexcept {
		m128x returnValues{ *ptr };
		return returnValues;
	}

	template<typename simd_int_t01, typename simd_int_t02> NIHILUS_INLINE m128x mm128XorSi128(const simd_int_t01& valOne, const simd_int_t02& valTwo) noexcept {
		m128x value{};
		std::copy(valOne.m128x_uint64, valOne.m128x_uint64 + 2, value.m128x_uint64);
		value.m128x_uint64[0] ^= valTwo.m128x_uint64[0];
		value.m128x_uint64[1] ^= valTwo.m128x_uint64[1];
		return value;
	}

	template<typename simd_int_t01, typename simd_int_t02> NIHILUS_INLINE bool mm128TestzSi128(simd_int_t01& valOneNew, simd_int_t02& valTwo) noexcept {
		detail::remove_const_t<simd_int_t01> valOne{ valOneNew };
		valOne.m128x_uint64[0] &= valTwo.m128x_uint64[0];
		valOne.m128x_uint64[1] &= valTwo.m128x_uint64[1];
		return valOne.m128x_uint64[0] == 0 && valOne.m128x_uint64[1] == 0;
	}

	template<nihilus_simd_128_types simd_int_t01> NIHILUS_INLINE static auto opTest(const typename simd_int_t01::type& value) noexcept {
		return !mm128TestzSi128(value, value);
	}

	template<nihilus_simd_128_types simd_int_t01, nihilus_simd_128_types simd_int_t02>
	NIHILUS_INLINE static auto opXor(const typename simd_int_t01::type& value, const typename simd_int_t02::type& other) noexcept {
		return mm128XorSi128(value, other);
	}

	template<nihilus_simd_128_types nihilus_simd_int_types_new> NIHILUS_INLINE static auto gather_values(const void* str) noexcept {
		return mm128LoadUSi128(static_cast<const m128x*>(str));
	}

	template<uint_types value_type> NIHILUS_INLINE static value_type tzcnt(value_type value) noexcept {
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

	NIHILUS_INLINE static half fp32_to_fp16(float f) {
		const uint32_t b = std::bit_cast<uint32_t>(f) + 0x00001000;
		const uint32_t e = (b & 0x7F800000) >> 23;
		const uint32_t m = b & 0x007FFFFF;
		return static_cast<half>((b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | (m >> 13)) |
			((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF);
	}

	NIHILUS_INLINE float sqrtf_fast(float x) {
		return std::bit_cast<float>(((std::bit_cast<uint32_t>(x) + 0x3f800000) >> 1) + 0x20000000);
	}

	NIHILUS_INLINE static constexpr float fp32_from_bits(uint32_t w) noexcept {
		return std::bit_cast<float>(w);
	}

	NIHILUS_INLINE static constexpr uint32_t fp32_to_bits(float f) noexcept {
		return std::bit_cast<uint32_t>(f);
	}

	NIHILUS_INLINE static constexpr float compute_fp16_to_fp32(half h) noexcept {
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

	alignas(64) static static_aligned_const<float>* __restrict fp16_to_fp32_array{ []() {
		alignas(64) static array<static_aligned_const<float>, (1 << 16)> return_values_new{};
		for (uint64_t i = 0; i < (1 << 16); ++i) {
			return_values_new[i] = static_aligned_const<float>{ compute_fp16_to_fp32(static_cast<half>(i)) };
		}
		return return_values_new.data();
	}() };

	NIHILUS_INLINE static float fp16_to_fp32(half f) {
		return fp16_to_fp32_array[f];
	}

#endif	

	NIHILUS_INLINE void dequantize_q8_0_to_f32(const block_q8_0<half>* src, float* dst, uint64_t count) {
		constexpr uint64_t block_size = 32;

		const uint64_t full_blocks = count / block_size;
		const uint64_t remainder   = count % block_size;

		for (uint64_t block_idx = 0; block_idx < full_blocks; ++block_idx) {
			const block_q8_0<half>& block = src[block_idx];
			const float scale			  = fp16_to_fp32(block.d);
			const int8_t* quantized		  = block.qs;
			const uint64_t base_offset	  = block_idx * block_size;

			for (uint64_t j = 0; j < block_size; ++j) {
				dst[base_offset + j] = scale * static_cast<float>(quantized[j]);
			}
		}
		if (remainder > 0) {
			const block_q8_0<half>& final_block = src[full_blocks];
			const float scale					= fp16_to_fp32(final_block.d);
			const int8_t* quantized				= final_block.qs;
			const uint64_t base_offset			= full_blocks * block_size;

			for (uint64_t j = 0; j < remainder; ++j) {
				dst[base_offset + j] = scale * static_cast<float>(quantized[j]);
			}
		}
	}
}
