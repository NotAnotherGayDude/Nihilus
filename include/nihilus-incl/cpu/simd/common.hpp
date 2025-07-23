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

	union m128x {
		template<typename value_type> constexpr m128x(value_type arg01, value_type arg02, value_type arg03, value_type arg04, value_type arg05, value_type arg06, value_type arg07,
			value_type arg08, value_type arg09, value_type arg10, value_type arg11, value_type arg12, value_type arg13, value_type arg14, value_type arg15,
			value_type arg16) noexcept {
			m128x_uint64[0] = static_cast<size_t>(arg01);
			m128x_uint64[0] |= static_cast<size_t>(arg02) << 8;
			m128x_uint64[0] |= static_cast<size_t>(arg03) << 16;
			m128x_uint64[0] |= static_cast<size_t>(arg04) << 24;
			m128x_uint64[0] |= static_cast<size_t>(arg05) << 32;
			m128x_uint64[0] |= static_cast<size_t>(arg06) << 40;
			m128x_uint64[0] |= static_cast<size_t>(arg07) << 48;
			m128x_uint64[0] |= static_cast<size_t>(arg08) << 56;
			m128x_uint64[1] = static_cast<size_t>(arg09);
			m128x_uint64[1] |= static_cast<size_t>(arg10) << 8;
			m128x_uint64[1] |= static_cast<size_t>(arg11) << 16;
			m128x_uint64[1] |= static_cast<size_t>(arg12) << 24;
			m128x_uint64[1] |= static_cast<size_t>(arg13) << 32;
			m128x_uint64[1] |= static_cast<size_t>(arg14) << 40;
			m128x_uint64[1] |= static_cast<size_t>(arg15) << 48;
			m128x_uint64[1] |= static_cast<size_t>(arg16) << 56;
		}

		constexpr m128x(size_t argOne, size_t argTwo) noexcept {
			m128x_uint64[0] = argOne;
			m128x_uint64[1] = argTwo;
		}

		constexpr m128x() noexcept {
			m128x_uint64[0] = 0;
			m128x_uint64[1] = 0;
		}

#if defined(NIHILUS_PLATFORM_WINDOWS)
		int8_t m128x_int8[16]{};
		int16_t m128x_int16[8];
		int32_t m128x_int32[4];
		int64_t m128x_int64[2];
		uint8_t m128x_uint8[16];
		int16_t m128x_uint16[8];
		int32_t m128x_uint32[4];
		size_t m128x_uint64[2];
#else
		int64_t m128x_int64[2];
		int32_t m128x_int32[4];
		int16_t m128x_int16[8];
		int8_t m128x_int8[16]{};
		size_t m128x_uint64[2];
		int32_t m128x_uint32[4];
		int16_t m128x_uint16[8];
		uint8_t m128x_uint8[16];
#endif
	};

#if defined(NIHILUS_AVX512)
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
#elif defined(NIHILUS_AVX2)
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
#elif defined(NIHILUS_NEON)
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
#elif defined(NIHILUS_SVE)
	using nihilus_simd_int_128		  = uint8x16_t;
	using nihilus_simd_int_256		  = uint32_t;
	using nihilus_simd_int_512		  = uint64_t;
	using nihilus_simd_int_t		  = uint8x16_t;
	using nihilus_string_parsing_type = uint16_t;
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
	concept nihilus_simd_512_types = std::same_as<nihilus_simd_int_512_t, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept nihilus_simd_256_types = std::same_as<nihilus_simd_int_256_t, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept nihilus_simd_128_types = std::same_as<nihilus_simd_int_128_t, std::remove_cvref_t<value_type>>;

#if defined(NIHILUS_AVX512) || defined(NIHILUS_AVX2)

	#define blsr(value) _blsr_u64(value)

	template<uint16_types value_type> NIHILUS_INLINE value_type tzcnt(value_type value) noexcept {
	#if defined(NIHILUS_LINUX)
		return __tzcnt_u16(value);
	#else
		return _tzcnt_u16(value);
	#endif
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

#if defined(NIHILUS_AVX512)

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

#if defined(NIHILUS_NEON)

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

	NIHILUS_INLINE static constexpr float fp32_from_bits(uint32_t w) {
		return std::bit_cast<float>(w);
	}

	NIHILUS_INLINE static constexpr uint32_t fp32_to_bits(float f) {
		return std::bit_cast<uint32_t>(f);
	}

	template<typename value_type> NIHILUS_INLINE value_type post_cmp_tzcnt(value_type value) noexcept {
		return tzcnt(value);
	}

	NIHILUS_INLINE static constexpr float fabsf_constexpr(float x) noexcept {
		return (x < 0.0f) ? -x : x;
	}

	NIHILUS_INLINE static constexpr float compute_fp16_to_fp32(half h) {
		const uint32_t w	 = static_cast<uint32_t>(h << 16);
		const uint32_t sign	 = w & 0x80000000;
		const uint32_t two_w = w + w;

		const uint32_t exp_offset	 = static_cast<uint32_t>(0xE0) << 23;
		constexpr float exp_scale	 = fp32_from_bits(static_cast<uint32_t>(0x7800000));
		const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

		constexpr const uint32_t magic_mask = static_cast<uint32_t>(126) << 23;
		constexpr const float magic_bias	= 0.5f;
		const float denormalized_value		= fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

		constexpr const uint32_t denormalized_cutoff = static_cast<uint32_t>(1) << 27;
		const uint32_t result						 = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
		return fp32_from_bits(result);
	}

	NIHILUS_INLINE const float* get_fp16_to_fp32_array() {
		static const float* fp16_to_fp32_array = []() {
			static float array[1 << 16];
			for (uint64_t i = 0; i < (1 << 16); ++i) {
				array[i] = compute_fp16_to_fp32(static_cast<half>(i));
			}
			return array;
		}();
		return fp16_to_fp32_array;
	}

	NIHILUS_INLINE static float fp16_to_fp32(half h) {
		static const float* reference = get_fp16_to_fp32_array();
		return reference[static_cast<uint64_t>(h)];
	}

}
