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

#include <nihilus/common/string_literal.hpp>
#include <type_traits>
#include <concepts>

namespace nihilus {

	template<uint64_t length> struct convert_length_to_int {
		static_assert(length <= 8, "Sorry, but that string is too long!");
		using type = std::conditional_t<length == 1, uint8_t,
			std::conditional_t<length <= 2, uint16_t, std::conditional_t<length <= 4, uint32_t, std::conditional_t<length <= 8, uint64_t, void>>>>;
	};

	template<uint64_t length> using convert_length_to_int_t = typename convert_length_to_int<length>::type;

	template<string_literal string>
		requires(string.length == 0)
	static constexpr auto packValues() {
		return uint8_t{};
	}

	template<string_literal string>
		requires(string.length > 0 && string.length <= 8)
	static constexpr auto packValues() {
		convert_length_to_int_t<string.length> returnValues{};
		for (size_t x = 0; x < string.length; ++x) {
			returnValues |= static_cast<convert_length_to_int_t<string.length>>(static_cast<uint64_t>(string[x]) << ((x % 8) * 8));
		}
		return returnValues;
	}

	template<size_t size> static constexpr size_t getPackingSize() {
		if constexpr (size >= 64) {
			return 64;
		} else if constexpr (size >= 32) {
			return 32;
		} else {
			return 16;
		}
	}

	template<string_literal string>
		requires(string.length != 0 && string.length > 8)
	static constexpr auto packValues() {
		NIHILUS_ALIGN(16) array<uint64_t, round_up_to_multiple<16>(getPackingSize<string.length>())> returnValues{};
		for (size_t x = 0; x < string.length; ++x) {
			if (x / 8 < (string.length / 8) + 1) {
				returnValues[x / 8] |= (static_cast<uint64_t>(string[x]) << ((x % 8) * 8));
			}
		}
		return returnValues;
	}

	template<typename value_type>
	concept equals_0 = value_type::length == 0;

	template<typename value_type>
	concept gt_0_lt_16 = value_type::length > 0 && value_type::length < 16;

	template<typename value_type>
	concept eq_16 = value_type::length == 16 && cpu_alignment >= 16;

	template<typename value_type>
	concept eq_32 = value_type::length == 32 && cpu_alignment >= 32;

	template<typename value_type>
	concept eq_64 = value_type::length == 64 && cpu_alignment >= 64;

	template<typename value_type>
	concept gt_16 = value_type::length > 16 && !eq_16<value_type> && !eq_32<value_type> && !eq_64<value_type>;

	template<size_t index, typename string_type> static constexpr auto stringLiteralFromView(string_type str) noexcept {
		string_literal<index + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		sl[index] = '\0';
		return sl;
	}

	template<string_literal string, size_t offset> static constexpr auto offSetNewLiteral() noexcept {
		constexpr size_t originalSize = string.length;
		constexpr size_t newSize	  = (offset >= originalSize) ? 0 : originalSize - offset;
		string_literal<newSize + 1> sl{};
		if constexpr (newSize > 0) {
			std::copy_n(string.data() + offset, newSize, sl.values);
			sl.values[newSize] = '\0';
		}
		return sl;
	}

	template<string_literal string, size_t offset> static constexpr auto offSetIntoLiteral() noexcept {
		constexpr size_t originalSize = string.length;
		constexpr size_t newSize	  = (offset >= originalSize) ? originalSize : offset;
		string_literal<newSize + 1> sl{};
		if constexpr (newSize > 0) {
			std::copy_n(string.data(), newSize, sl.values);
			sl.values[newSize] = '\0';
		}
		return sl;
	}

	template<typename sl_type, std::remove_cvref_t<sl_type> stringNew> struct string_literal_comparitor;

	template<equals_0 sl_type, std::remove_cvref_t<sl_type> stringNew> struct string_literal_comparitor<sl_type, stringNew> {
		NIHILUS_FORCE_INLINE static bool impl(const char*) noexcept {
			return true;
		}
	};

	template<gt_0_lt_16 sl_type, std::remove_cvref_t<sl_type> stringNew> struct string_literal_comparitor<sl_type, stringNew> {
		inline static constexpr auto stringLiteral{ stringNew };
		inline static constexpr auto newCount{ stringLiteral.size() };
		NIHILUS_FORCE_INLINE static bool impl(const char* str) noexcept {
			if constexpr (newCount > 8) {
				NIHILUS_ALIGN(16) static constexpr auto valuesNew{ packValues<stringLiteral>() };
				nihilus_simd_int_128 data1{};
				std::memcpy(&data1, str, newCount);
				const nihilus_simd_int_128 data2{ gatherValues<nihilus_simd_int_128>(valuesNew.data()) };
				return !opTest(opXor(data1, data2));
			} else if constexpr (newCount == 8) {
				static constexpr auto valuesNew{ packValues<stringLiteral>() };
				uint64_t l;
				std::memcpy(&l, str, 8);
				return !(l ^ valuesNew);
			} else if constexpr (newCount == 7) {
				static constexpr auto valuesNew{ packValues<stringLiteral>() };
				uint64_t l{};
				std::memcpy(&l, str, 7);
				return !(l ^ valuesNew);
			} else if constexpr (newCount == 6) {
				static constexpr auto valuesNew{ packValues<stringLiteral>() };
				uint64_t l{};
				std::memcpy(&l, str, 6);
				return !(l ^ valuesNew);
			} else if constexpr (newCount == 5) {
				static constexpr uint32_t valuesNew{ static_cast<uint32_t>(packValues<stringLiteral>()) };
				uint32_t l;
				std::memcpy(&l, str, 4);
				return !(l ^ valuesNew) && (str[4] == stringLiteral[4]);
			} else if constexpr (newCount == 4) {
				static constexpr uint32_t valuesNew{ static_cast<uint32_t>(packValues<stringLiteral>()) };
				uint32_t l;
				std::memcpy(&l, str, 4);
				return !(l ^ valuesNew);
			} else if constexpr (newCount == 3) {
				static constexpr uint16_t valuesNew{ static_cast<uint16_t>(packValues<stringLiteral>()) };
				uint16_t l;
				std::memcpy(&l, str, 2);
				return !(l ^ valuesNew) && (str[2] == stringLiteral[2]);
			} else if constexpr (newCount == 2) {
				static constexpr uint16_t valuesNew{ static_cast<uint16_t>(packValues<stringLiteral>()) };
				uint16_t l;
				std::memcpy(&l, str, 2);
				return !(l ^ valuesNew);
			} else if constexpr (newCount == 1) {
				return *str == stringLiteral[0];
			} else {
				return true;
			}
		};
	};

#if defined(NIHILUS_AVX2) || defined(NIHILUS_AVX512) || defined(NIHILUS_NEON) || defined(NIHILUS_SVE2)

	template<eq_16 sl_type, std::remove_cvref_t<sl_type> stringNew> struct string_literal_comparitor<sl_type, stringNew> {
		inline static constexpr auto newLiteral{ stringNew };
		NIHILUS_ALIGN(16) inline static constexpr auto valuesNew { packValues<newLiteral>() };
		NIHILUS_FORCE_INLINE static bool impl(const char* str) noexcept {
			NIHILUS_ALIGN(16) char valuesToLoad[16];
			std::memcpy(valuesToLoad, str, 16);
			const nihilus_simd_int_128 data1{ gatherValues<nihilus_simd_int_128>(valuesToLoad) };
			const nihilus_simd_int_128 data2{ gatherValues<nihilus_simd_int_128>(valuesNew.data()) };
			return !opTest(opXor(data1, data2));
		}
	};

#endif

#if defined(NIHILUS_AVX2) || defined(NIHILUS_AVX512) || defined(NIHILUS_SVE2)

	template<eq_32 sl_type, std::remove_cvref_t<sl_type> stringNew> struct string_literal_comparitor<sl_type, stringNew> {
		inline static constexpr auto newLiteral{ stringNew };
		NIHILUS_ALIGN(32) inline static constexpr auto valuesNew { packValues<newLiteral>() };
		NIHILUS_FORCE_INLINE static bool impl(const char* str) noexcept {
			NIHILUS_ALIGN(32) char valuesToLoad[32];
			std::memcpy(valuesToLoad, str, 32);
			const nihilus_simd_int_256 data1{ gatherValues<nihilus_simd_int_256>(valuesToLoad) };
			const nihilus_simd_int_256 data2{ gatherValues<nihilus_simd_int_256>(valuesNew.data()) };
			return !opTest(opXor(data1, data2));
		}
	};

#endif

#if defined(NIHILUS_AVX512) || defined(NIHILUS_SVE2)
	template<eq_64 sl_type, sl_type stringNew> struct string_literal_comparitor<sl_type, stringNew> {
		inline static constexpr auto newLiteral{ stringNew };
		NIHILUS_ALIGN(64) inline static constexpr auto valuesNew { packValues<newLiteral>() };
		NIHILUS_FORCE_INLINE static bool impl(const char* str) noexcept {
			NIHILUS_ALIGN(64) char valuesToLoad[64];
			std::memcpy(valuesToLoad, str, 64);
			const nihilus_simd_int_512 data1{ gatherValues<nihilus_simd_int_512>(valuesToLoad) };
			const nihilus_simd_int_512 data2{ gatherValues<nihilus_simd_int_512>(valuesNew.data()) };
			return !opTest(opXor(data1, data2));
		}
	};
#endif

	static constexpr auto getOffsetIntoLiteralSize(size_t inputSize) noexcept {
		if (inputSize >= 64 && cpu_alignment >= 64) {
			return 64;
		} else if (inputSize >= 32 && cpu_alignment >= 32) {
			return 32;
		} else {
			return 16;
		}
	}

	template<gt_16 sl_type, std::remove_cvref_t<sl_type> stringNew> struct string_literal_comparitor<sl_type, stringNew> {
		inline static constexpr auto string{ offSetIntoLiteral<stringNew, getOffsetIntoLiteralSize(stringNew.size())>() };
		inline static constexpr auto stringSize = string.size();
		inline static constexpr auto stringNewer{ offSetNewLiteral<stringNew, stringSize>() };
		NIHILUS_FORCE_INLINE static bool impl(const char* str) noexcept {
			if (!string_literal_comparitor<decltype(string), string>::impl(str)) {
				return false;
			} else {
				str += stringSize;
				return string_literal_comparitor<decltype(stringNewer), stringNewer>::impl(str);
			}
		}
	};

	template<string_literal string_literal> NIHILUS_FORCE_INLINE bool string_literal_comparison(const char* string) {
		using sl_type = decltype(string_literal);
		return string_literal_comparitor<sl_type, string_literal>::impl(string);
	}

}