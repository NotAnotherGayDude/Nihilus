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
		NIHILUS_INLINE m128x(uint64_t argOne, uint64_t argTwo) noexcept {
			m128x_uint64[0] = argOne;
			m128x_uint64[1] = argTwo;
		}

		NIHILUS_INLINE m128x() noexcept {
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
		uint64_t m128x_uint64[2];
#else
		int64_t m128x_int64[2];
		int32_t m128x_int32[4];
		int16_t m128x_int16[8];
		int8_t m128x_int8[16]{};
		uint64_t m128x_uint64[2];
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

	NIHILUS_INLINE static uint32_t lzcnt(const uint32_t value) noexcept {
		return _lzcnt_u32(value);
	}

	NIHILUS_INLINE static uint64_t lzcnt(const uint64_t value) noexcept {
		return _lzcnt_u64(value);
	}

	template<uint16_types value_type> NIHILUS_INLINE static value_type tzcnt(const value_type value) noexcept {
	#if defined(NIHILUS_LINUX)
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

#if !defined(NIHILUS_AVX512) && !defined(NIHILUS_AVX2) && !defined(NIHILUS_NEON) && !defined(NIHILUS_SVE2)

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

	#define blsr(value) (value & (value - 1))

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
		std::remove_const_t<simd_int_t01> valOne{ valOneNew };
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

#endif

	NIHILUS_INLINE static constexpr float fp32_from_bits(uint32_t w) noexcept {
		return std::bit_cast<float>(w);
	}

	NIHILUS_INLINE static constexpr uint32_t fp32_to_bits(float f) noexcept {
		return std::bit_cast<uint32_t>(f);
	}

	struct alignas(64) float_holder {
		NIHILUS_INLINE float_holder() noexcept : value{} {
		}

		NIHILUS_INLINE float_holder(float new_value) noexcept : value{ new_value } {
		}

		NIHILUS_INLINE operator float() const noexcept {
			return value;
		}

		NIHILUS_INLINE operator float&() noexcept {
			return value;
		}

	  protected:
		alignas(64) float value{};
	};

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

	NIHILUS_INLINE const auto& get_fp16_to_fp32_array() noexcept {
		alignas(64) static auto fp16_to_fp32_array = []() {
			alignas(64) static std::unique_ptr<array<float_holder, (1 << 16)>> return_values_new{ std::make_unique<array<float_holder, (1 << 16)>>() };
			for (uint64_t i = 0; i < (1 << 16); ++i) {
				(*return_values_new)[i] = compute_fp16_to_fp32(static_cast<half>(i));
			}
			alignas(64) static array<float_holder, (1 << 16)>& fp16_to_fp32_array_ref{ *return_values_new };
			return fp16_to_fp32_array_ref;
		}();
		return fp16_to_fp32_array;
	}

	NIHILUS_INLINE static float fp16_to_fp32(uint16_t h) noexcept {
		return *(get_fp16_to_fp32_array().data() + static_cast<uint64_t>(h));
	}

}
