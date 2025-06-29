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
	using nihilus_string_parsing_type = size_t;
	inline static constexpr size_t bitsPerStep{ 512 };
#elif defined(NIHILUS_AVX2)
	using nihilus_simd_int_128		  = __m128i;
	using nihilus_simd_int_256		  = __m256i;
	using nihilus_simd_int_512		  = __m512i;
	using nihilus_simd_int_t		  = __m256i;
	using nihilus_string_parsing_type = uint32_t;
	inline static constexpr size_t bitsPerStep{ 256 };
#elif defined(NIHILUS_NEON)
	using nihilus_simd_int_128		  = uint8x16_t;
	using nihilus_simd_int_256		  = uint32_t;
	using nihilus_simd_int_512		  = size_t;
	using nihilus_simd_int_t		  = uint8x16_t;
	using nihilus_string_parsing_type = uint16_t;
	inline static constexpr size_t bitsPerStep{ 128 };
#elif defined(NIHILUS_SVE)
#else
	using nihilus_simd_int_128 = nihilus::__m128x;
	using nihilus_simd_int_256 = uint32_t;
	using nihilus_simd_int_512 = size_t;

	using nihilus_simd_int_t		  = nihilus::__m128x;
	using nihilus_string_parsing_type = uint16_t;
	inline constexpr size_t bitsPerStep{ 128 };

#endif

	template<typename value_type>
	concept simd_int_512_type = std::is_same_v<nihilus_simd_int_512, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_256_type = std::is_same_v<nihilus_simd_int_256, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_128_type = std::is_same_v<nihilus_simd_int_128, std::remove_cvref_t<value_type>>;
	template<typename value_type>
	concept simd_int_type = std::is_same_v<nihilus_simd_int_t, std::remove_cvref_t<value_type>>;

}