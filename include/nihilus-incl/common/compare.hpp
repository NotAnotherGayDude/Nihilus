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

#include <nihilus-incl/common/string_literal.hpp>
#include <nihilus-incl/cpu/cpu_arch.hpp>
#include <nihilus-incl/common/allocator.hpp>
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
		requires(string.size() == 0)
	static constexpr auto pack_values() {
		return uint8_t{};
	}

	template<string_literal string>
		requires(string.size() > 0 && string.size() <= 8)
	static constexpr auto pack_values() {
		convert_length_to_int_t<string.size()> return_values{};
		for (uint64_t x = 0; x < string.size(); ++x) {
			return_values |= static_cast<convert_length_to_int_t<string.size()>>(static_cast<uint64_t>(string[x]) << ((x % 8) * 8));
		}
		return return_values;
	}

	template<uint64_t size> static constexpr uint64_t get_packing_size() {
		if constexpr (size >= 64) {
			return 64;
		} else if constexpr (size >= 32) {
			return 32;
		} else {
			return 16;
		}
	}

	template<string_literal string>
		requires(string.size() != 0 && string.size() > 8)
	static constexpr auto pack_values() {
		alignas(64) array<uint64_t, round_up_to_multiple<16>(get_packing_size<string.size()>())> return_values{};
		for (uint64_t x = 0; x < string.size(); ++x) {
			if (x / 8 < (string.size() / 8) + 1) {
				return_values[x / 8] |= (static_cast<uint64_t>(string[x]) << ((x % 8) * 8));
			}
		}
		return return_values;
	}

	template<typename value_type>
	concept equals_0 = value_type::size() == 0;

	template<typename value_type>
	concept gt_0_lt_16 = value_type::size() > 0 && value_type::size() < 16;

	template<typename value_type>
	concept eq_16 = value_type::size() == 16;

	template<typename value_type>
	concept eq_32 = value_type::size() == 32;

	template<typename value_type>
	concept eq_64 = value_type::size() == 64;

	template<typename value_type>
	concept gt_16 = value_type::size() > 16 && !eq_16<value_type> && !eq_32<value_type> && !eq_64<value_type>;

	template<uint64_t index, typename string_types> static constexpr auto string_literal_from_view(string_types str) noexcept {
		string_literal<index + 1> sl{};
		std::copy_n(str.data(), str.size(), sl.values);
		sl[index] = '\0';
		return sl;
	}

	template<string_literal string, uint64_t offset> static constexpr auto offset_new_literal() noexcept {
		constexpr uint64_t originalSize = string.size();
		constexpr uint64_t newSize		= (offset >= originalSize) ? 0 : originalSize - offset;
		string_literal<newSize + 1> sl{};
		if constexpr (newSize > 0) {
			std::copy_n(string.data() + offset, newSize, sl.values);
			sl.values[newSize] = '\0';
		}
		return sl;
	}

	template<string_literal string, uint64_t offset> static constexpr auto offset_into_literal() noexcept {
		constexpr uint64_t originalSize = string.size();
		constexpr uint64_t newSize		= (offset >= originalSize) ? originalSize : offset;
		string_literal<newSize + 1> sl{};
		if constexpr (newSize > 0) {
			std::copy_n(string.data(), newSize, sl.values);
			sl.values[newSize] = '\0';
		}
		return sl;
	}

	template<typename sl_type, detail::remove_cvref_t<sl_type> string_new> struct string_literal_comparitor;

	template<equals_0 sl_type, detail::remove_cvref_t<sl_type> string_new> struct string_literal_comparitor<sl_type, string_new> {
		NIHILUS_INLINE static bool impl(const char*) noexcept {
			return true;
		}
	};

	template<gt_0_lt_16 sl_type, detail::remove_cvref_t<sl_type> string_new> struct string_literal_comparitor<sl_type, string_new> {
		inline static constexpr auto string_lit{ string_new };
		inline static constexpr auto newCount{ string_lit.size() };
		NIHILUS_INLINE static bool impl(const char* str) noexcept {
			if constexpr (newCount > 8) {
				alignas(64) static constexpr auto values_new{ pack_values<string_lit>() };
				nihilus_simd_int_128 data1{};
				std::memcpy(&data1, str, newCount);
				const nihilus_simd_int_128 data2{ gather_values<nihilus_simd_int_128_t>(values_new.data()) };
				return !opTest<nihilus_simd_int_128_t>(opXor<nihilus_simd_int_128_t, nihilus_simd_int_128_t>(data1, data2));
			} else if constexpr (newCount == 8) {
				static constexpr static_aligned_const values_new{ pack_values<string_lit>() };
				static_aligned_const<uint64_t> l;
				std::memcpy(&l, str, 8);
				return !(l ^ values_new);
			} else if constexpr (newCount == 7) {
				static constexpr static_aligned_const values_new{ pack_values<string_lit>() };
				static_aligned_const<uint64_t> l{};
				std::memcpy(&l, str, 7);
				return !(l ^ values_new);
			} else if constexpr (newCount == 6) {
				static constexpr static_aligned_const values_new{ pack_values<string_lit>() };
				static_aligned_const<uint64_t> l{};
				std::memcpy(&l, str, 6);
				return !(l ^ values_new);
			} else if constexpr (newCount == 5) {
				static constexpr static_aligned_const values_new{ static_cast<uint32_t>(pack_values<string_lit>()) };
				static_aligned_const<uint32_t> l;
				std::memcpy(&l, str, 4);
				return !(l ^ values_new) && (str[4] == string_lit[4]);
			} else if constexpr (newCount == 4) {
				static constexpr static_aligned_const values_new{ static_cast<uint32_t>(pack_values<string_lit>()) };
				static_aligned_const<uint32_t> l;
				std::memcpy(&l, str, 4);
				return !(l ^ values_new);
			} else if constexpr (newCount == 3) {
				static constexpr static_aligned_const values_new{ static_cast<uint16_t>(pack_values<string_lit>()) };
				static_aligned_const<uint16_t> l;
				std::memcpy(&l, str, 2);
				return !(l ^ values_new) && (str[2] == string_lit[2]);
			} else if constexpr (newCount == 2) {
				static constexpr static_aligned_const values_new{ static_cast<uint16_t>(pack_values<string_lit>()) };
				static_aligned_const<uint16_t> l;
				std::memcpy(&l, str, 2);
				return !(l ^ values_new);
			} else if constexpr (newCount == 1) {
				return *str == string_lit[0];
			} else {
				return true;
			}
		}
	};

	template<eq_16 sl_type, detail::remove_cvref_t<sl_type> string_new> struct string_literal_comparitor<sl_type, string_new> {
		inline static constexpr auto new_literal{ string_new };
		alignas(64) inline static constexpr auto values_new{ pack_values<new_literal>() };
		NIHILUS_INLINE static bool impl(const char* str) noexcept {
			alignas(64) char values_to_load[16];
			std::memcpy(values_to_load, str, 16);
			const nihilus_simd_int_128 data1{ gather_values<nihilus_simd_int_128_t>(values_to_load) };
			const nihilus_simd_int_128 data2{ gather_values<nihilus_simd_int_128_t>(values_new.data()) };
			return !opTest<nihilus_simd_int_128_t>(opXor<nihilus_simd_int_128_t, nihilus_simd_int_128_t>(data1, data2));
		}
	};

#if NIHILUS_AVX2 || NIHILUS_AVX512 || NIHILUS_SVE2

	template<eq_32 sl_type, detail::remove_cvref_t<sl_type> string_new> struct string_literal_comparitor<sl_type, string_new> {
		inline static constexpr auto new_literal{ string_new };
		alignas(64) inline static constexpr auto values_new{ pack_values<new_literal>() };
		NIHILUS_INLINE static bool impl(const char* str) noexcept {
			alignas(64) char values_to_load[32];
			std::memcpy(values_to_load, str, 32);
			const nihilus_simd_int_256 data1{ gather_values<nihilus_simd_int_256_t>(values_to_load) };
			const nihilus_simd_int_256 data2{ gather_values<nihilus_simd_int_256_t>(values_new.data()) };
			return !opTest<nihilus_simd_int_256_t>(opXor<nihilus_simd_int_256_t, nihilus_simd_int_256_t>(data1, data2));
		}
	};

#endif

#if NIHILUS_AVX512 || NIHILUS_SVE2
	template<eq_64 sl_type, sl_type string_new> struct string_literal_comparitor<sl_type, string_new> {
		inline static constexpr auto new_literal{ string_new };
		alignas(64) inline static constexpr auto values_new{ pack_values<new_literal>() };
		NIHILUS_INLINE static bool impl(const char* str) noexcept {
			alignas(64) char values_to_load[64];
			std::memcpy(values_to_load, str, 64);
			const nihilus_simd_int_512 data1{ gather_values<nihilus_simd_int_512_t>(values_to_load) };
			const nihilus_simd_int_512 data2{ gather_values<nihilus_simd_int_512_t>(values_new.data()) };
			return !opTest(opXor(data1, data2));
		}
	};
#endif

	static constexpr auto get_offset_into_literal_size(uint64_t inputSize) noexcept {
		if (inputSize >= 64 && (NIHILUS_AVX512 || NIHILUS_SVE2)) {
			return 64;
		} else if (inputSize >= 32 && (NIHILUS_AVX2 || NIHILUS_AVX512 || NIHILUS_SVE2)) {
			return 32;
		} else {
			return 16;
		}
	}

	template<gt_16 sl_type, detail::remove_cvref_t<sl_type> string_new> struct string_literal_comparitor<sl_type, string_new> {
		inline static constexpr auto string{ offset_into_literal<string_new, get_offset_into_literal_size(string_new.size())>() };
		inline static constexpr auto string_size = string.size();
		inline static constexpr auto string_newer{ offset_new_literal<string_new, string_size>() };
		NIHILUS_INLINE static bool impl(const char* str) noexcept {
			if (!string_literal_comparitor<decltype(string), string>::impl(str)) {
				return false;
			} else {
				str += string_size;
				return string_literal_comparitor<decltype(string_newer), string_newer>::impl(str);
			}
		}
	};

	template<string_literal string_literal> NIHILUS_INLINE bool string_literal_comparison(const char* string) {
		using sl_type = decltype(string_literal);
		return string_literal_comparitor<sl_type, string_literal>::impl(string);
	}

}
