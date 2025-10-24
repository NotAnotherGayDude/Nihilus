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
#include <nihilus-incl/cpu/fallback.hpp>
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
		if constexpr (size >= 64 && (NIHILUS_AVX512 | NIHILUS_SVE2)) {
			return 64;
		} else if constexpr (size >= 32 && (NIHILUS_AVX2 | NIHILUS_AVX512 | NIHILUS_SVE2)) {
			return 32;
		} else {
			return 16;
		}
	}

	template<string_literal string>
		requires(string.size() > 8)
	static constexpr auto pack_values() {
		NIHILUS_ALIGN(64) array<uint64_t, get_packing_size<string.size()>() / sizeof(uint64_t) + 1> out{};
		for (uint64_t i = 0; i < string.size() && i < get_packing_size<string.size()>(); ++i) {
			out[i / 8] |= static_cast<uint64_t>(static_cast<unsigned char>(string[i])) << ((i % 8) * 8);
		}
		return out;
	}

	template<string_literal string, uint64_t offset> static constexpr auto offset_into_literal() noexcept {
		constexpr uint64_t originalSize = string.size();
		constexpr uint64_t newSize		= (offset >= originalSize) ? 0 : originalSize - offset;
		string_literal<newSize + 1> sl{};
		if constexpr (newSize > 0) {
			std::copy_n(string.data() + offset, newSize, sl.values);
			sl.values[newSize] = '\0';
		}
		return sl;
	}

	template<string_literal string, uint64_t offset> static constexpr auto offset_new_literal() noexcept {
		constexpr uint64_t originalSize = string.size();
		constexpr uint64_t newSize		= (offset >= originalSize) ? originalSize : offset;
		string_literal<newSize + 1> sl{};
		if constexpr (newSize > 0) {
			std::copy_n(string.data(), newSize, sl.values);
			sl.values[newSize] = '\0';
		}
		return sl;
	}

	static constexpr auto get_offset_into_literal_size(uint64_t inputSize) noexcept {
		if (inputSize >= 64 && (NIHILUS_AVX512 | NIHILUS_SVE2)) {
			return 64ull;
		} else if (inputSize >= 32 && (NIHILUS_AVX2 | NIHILUS_AVX512 | NIHILUS_SVE2)) {
			return 32ull;
		} else {
			return 16ull;
		}
	}

	template<string_literal string> struct string_literal_comparitor {
		NIHILUS_HOST static bool impl(const char*) noexcept {
			return false;
		}
	};

	template<string_literal string>
		requires(string.size() > 0 && string.size() < 16)
	struct string_literal_comparitor<string> {
		inline static constexpr auto string_lit{ string };
		inline static constexpr auto new_size{ string_lit.size() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			if constexpr (new_size > 8) {
				NIHILUS_ALIGN(64) static constexpr auto values_new{ pack_values<string_lit>() };
				nihilus_simd_int_128 data1{};
				constexpr_memcpy<new_size>(&data1, str);
				const nihilus_simd_int_128 data2{ gather_values<nihilus_simd_int_128_t>(values_new.data()) };
				return !op_test<nihilus_simd_int_128_t>(op_xor<nihilus_simd_int_128_t>(data1, data2));
			} else if constexpr (new_size == 8) {
				static constexpr static_aligned_const values_new{ pack_values<string_lit>() };
				static_aligned_const<uint64_t> l;
				constexpr_memcpy<8>(&l, str);
				return !(l ^ values_new);
			} else if constexpr (new_size == 7) {
				static constexpr static_aligned_const values_new{ pack_values<string_lit>() };
				static_aligned_const<uint64_t> l{};
				constexpr_memcpy<7>(&l, str);
				return !(l ^ values_new);
			} else if constexpr (new_size == 6) {
				static constexpr static_aligned_const values_new{ pack_values<string_lit>() };
				static_aligned_const<uint64_t> l{};
				constexpr_memcpy<6>(&l, str);
				return !(l ^ values_new);
			} else if constexpr (new_size == 5) {
				static constexpr static_aligned_const values_new{ static_cast<uint32_t>(pack_values<string_lit>()) };
				static_aligned_const<uint32_t> l;
				constexpr_memcpy<4>(&l, str);
				return !(l ^ values_new) && (str[4] == string_lit[4]);
			} else if constexpr (new_size == 4) {
				static constexpr static_aligned_const values_new{ static_cast<uint32_t>(pack_values<string_lit>()) };
				static_aligned_const<uint32_t> l;
				constexpr_memcpy<4>(&l, str);
				return !(l ^ values_new);
			} else if constexpr (new_size == 3) {
				static constexpr static_aligned_const values_new{ static_cast<uint16_t>(pack_values<string_lit>()) };
				static_aligned_const<uint16_t> l;
				constexpr_memcpy<2>(&l, str);
				return !(l ^ values_new) && (str[2] == string_lit[2]);
			} else if constexpr (new_size == 2) {
				static constexpr static_aligned_const values_new{ static_cast<uint16_t>(pack_values<string_lit>()) };
				static_aligned_const<uint16_t> l;
				constexpr_memcpy<2>(&l, str);
				return !(l ^ values_new);
			} else if constexpr (new_size == 1) {
				return *str == string_lit[0];
			} else {
				return true;
			}
		}
	};

	template<string_literal string>
		requires(string.size() == 16)
	struct string_literal_comparitor<string> {
		inline static constexpr auto new_literal{ string };
		NIHILUS_ALIGN(64) inline static constexpr auto values_new { pack_values<new_literal>() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			NIHILUS_ALIGN(64) char values_to_load[16];
			constexpr_memcpy<16>(values_to_load, str);
			const nihilus_simd_int_128 data1{ gather_values<nihilus_simd_int_128_t>(values_to_load) };
			const nihilus_simd_int_128 data2{ gather_values<nihilus_simd_int_128_t>(values_new.data()) };
			return !op_test<nihilus_simd_int_128_t>(op_xor<nihilus_simd_int_128_t>(data1, data2));
		}
	};

#if NIHILUS_AVX2 | NIHILUS_AVX512 | NIHILUS_SVE2

	template<string_literal string>
		requires(string.size() > 16 && string.size() < 32)
	struct string_literal_comparitor<string> {
		inline static constexpr auto string_new{ offset_new_literal<string, get_offset_into_literal_size(string.size())>() };
		inline static constexpr auto string_size = string_new.size();
		inline static constexpr auto string_newer{ offset_into_literal<string, string_size>() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			if (!string_literal_comparitor<string_new>::impl(str)) {
				return false;
			} else {
				str += string_size;
				return string_literal_comparitor<string_newer>::impl(str);
			}
		}
	};

	template<string_literal string>
		requires(string.size() == 32)
	struct string_literal_comparitor<string> {
		inline static constexpr auto new_literal{ string };
		NIHILUS_ALIGN(64) inline static constexpr auto values_new { pack_values<new_literal>() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			NIHILUS_ALIGN(64) char values_to_load[32];
			constexpr_memcpy<32>(values_to_load, str);
			const nihilus_simd_int_256 data1{ gather_values<nihilus_simd_int_256_t>(values_to_load) };
			const nihilus_simd_int_256 data2{ gather_values<nihilus_simd_int_256_t>(values_new.data()) };
			return !op_test<nihilus_simd_int_256_t>(op_xor<nihilus_simd_int_256_t>(data1, data2));
		}
	};

#endif

#if NIHILUS_AVX512 | NIHILUS_SVE2

	template<string_literal string>
		requires(string.size() > 32 && string.size() < 64)
	struct string_literal_comparitor<string> {
		inline static constexpr auto string_new{ offset_new_literal<string, get_offset_into_literal_size(string.size())>() };
		inline static constexpr auto string_size = string_new.size();
		inline static constexpr auto string_newer{ offset_into_literal<string, string_size>() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			if (!string_literal_comparitor<string_new>::impl(str)) {
				return false;
			} else {
				str += string_size;
				return string_literal_comparitor<string_newer>::impl(str);
			}
		}
	};

	template<string_literal string>
		requires(string.size() == 64)
	struct string_literal_comparitor<string> {
		inline static constexpr auto new_literal{ string };
		NIHILUS_ALIGN(64) inline static constexpr auto values_new { pack_values<new_literal>() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			NIHILUS_ALIGN(64) char values_to_load[64];
			constexpr_memcpy<64>(values_to_load, str);
			const nihilus_simd_int_512 data1{ gather_values<nihilus_simd_int_512_t>(values_to_load) };
			const nihilus_simd_int_512 data2{ gather_values<nihilus_simd_int_512_t>(values_new.data()) };
			return !op_test<nihilus_simd_int_512_t>(op_xor<nihilus_simd_int_512_t>(data1, data2));
		}
	};
#endif

	template<string_literal string>
		requires(string.size() > cpu_properties::cpu_alignment)
	struct string_literal_comparitor<string> {
		inline static constexpr auto string_new{ offset_new_literal<string, get_offset_into_literal_size(string.size())>() };
		inline static constexpr auto string_size = string_new.size();
		inline static constexpr auto string_newer{ offset_into_literal<string, string_size>() };
		NIHILUS_HOST static bool impl(const char* str) noexcept {
			if (!string_literal_comparitor<string_new>::impl(str)) {
				return false;
			} else {
				str += string_size;
				return string_literal_comparitor<string_newer>::impl(str);
			}
		}
	};

	struct comparison {
		template<typename char_type01, typename char_type02> NIHILUS_HOST static bool compare(const char_type01* lhs, char_type02* rhs, uint64_t lengthNew) noexcept {
#if NIHILUS_AVX512 || NIHILUS_SVE2
			if (lengthNew >= 64) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 2>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 2>::type::bytes_processed;
				static constexpr uint64_t mask		  = detail::get_type_at_index<avx_list, 2>::type::mask;
				NIHILUS_ALIGN(64) char values_to_load[64];
				simd_type::type value01, value02;
				while (lengthNew >= vector_size) {
					std::memcpy(values_to_load, lhs, 64);
					value01 = gather_values<simd_type>(values_to_load);
					std::memcpy(values_to_load, rhs, 64);
					value02 = gather_values<simd_type>(values_to_load);
					if (op_cmp_eq<simd_type>(value01, value02) != mask) {
						return false;
					}
					lengthNew -= vector_size;
					lhs += vector_size;
					rhs += vector_size;
				}
			}
#endif
#if NIHILUS_AVX512 || NIHILUS_SVE2 || NIHILUS_AVX2
			if (lengthNew >= 32) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 1>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 1>::type::bytes_processed;
				static constexpr uint64_t mask		  = detail::get_type_at_index<avx_list, 1>::type::mask;
				NIHILUS_ALIGN(32) char values_to_load[32];
				simd_type::type value01, value02;
				while (lengthNew >= vector_size) {
					std::memcpy(values_to_load, lhs, 32);
					value01 = gather_values<simd_type>(values_to_load);
					std::memcpy(values_to_load, rhs, 32);
					value02 = gather_values<simd_type>(values_to_load);
					if (op_cmp_eq<simd_type>(value01, value02) != mask) {
						return false;
					}
					lengthNew -= vector_size;
					lhs += vector_size;
					rhs += vector_size;
				}
			}
#endif
			if (lengthNew >= 16) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 0>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 0>::type::bytes_processed;
				static constexpr uint64_t mask		  = detail::get_type_at_index<avx_list, 0>::type::mask;
				NIHILUS_ALIGN(16) char values_to_load[16];
				simd_type::type value01, value02;
				while (lengthNew >= vector_size) {
					std::memcpy(values_to_load, lhs, 16);
					value01 = gather_values<simd_type>(values_to_load);
					std::memcpy(values_to_load, rhs, 16);
					value02 = gather_values<simd_type>(values_to_load);
					if (op_cmp_eq<simd_type>(value01, value02) != mask) {
						return false;
					}
					lengthNew -= vector_size;
					lhs += vector_size;
					rhs += vector_size;
				}
			}
			{
				static constexpr uint64_t n_bytes{ sizeof(uint64_t) };
				if (lengthNew >= n_bytes) {
					uint64_t v1, v2;
					std::memcpy(&v1, lhs, n_bytes);
					std::memcpy(&v2, rhs, n_bytes);
					if ((v1 ^ v2) != 0) {
						return false;
					}
					lengthNew -= n_bytes;
					lhs += n_bytes;
					rhs += n_bytes;
				}
			}
			{
				static constexpr uint64_t n_bytes{ sizeof(uint32_t) };
				if (lengthNew >= n_bytes) {
					uint32_t v1, v2;
					std::memcpy(&v1, lhs, n_bytes);
					std::memcpy(&v2, rhs, n_bytes);
					if ((v1 ^ v2) != 0) {
						return false;
					}
					lengthNew -= n_bytes;
					lhs += n_bytes;
					rhs += n_bytes;
				}
			}
			{
				static constexpr uint64_t n_bytes{ sizeof(uint16_t) };
				if (lengthNew >= n_bytes) {
					uint16_t v1, v2;
					std::memcpy(&v1, lhs, n_bytes);
					std::memcpy(&v2, rhs, n_bytes);
					if ((v1 ^ v2) != 0) {
						return false;
					}
					lengthNew -= n_bytes;
					lhs += n_bytes;
					rhs += n_bytes;
				}
			}
			if (lengthNew && *lhs != *rhs) {
				return false;
			}
			return true;
		}
	};

	template<size_t size> NIHILUS_ALIGN(64)
	inline constexpr array<uint8_t, size> whitespaceArray{ []() constexpr {
		constexpr const uint8_t values[]{ 0x20u, 0x64u, 0x64u, 0x64u, 0x11u, 0x64u, 0x71u, 0x02u, 0x64u, '\t', '\n', 0x70u, 0x64u, '\r', 0x64u, 0x64u };
		array<uint8_t, size> returnValues{};
		for (uint64_t x = 0; x < size; ++x) {
			returnValues[x] = values[x % 16];
		}
		return returnValues;
	}() };

	template<typename simd_type> NIHILUS_HOST static auto collect_whitespace_indices(const char* values) noexcept {
		static constexpr auto whiteSpaceArrayPtr{ whitespaceArray<sizeof(typename simd_type::type)>.data() };
		const typename simd_type::type simdValues{ gather_values<simd_type>(whiteSpaceArrayPtr) };
		const typename simd_type::type comparison_values{ gather_values<simd_type>(values) };
		return op_cmp_eq_bitmask<simd_type>(op_shuffle<simd_type>(simdValues, comparison_values), comparison_values);
	}

	struct whitespace_search {
		template<typename char_type> NIHILUS_HOST static uint64_t find_first_not_of(const char_type* text, uint64_t length) noexcept {
			int64_t i = 0;
			if NIHILUS_UNLIKELY (length < 16) {
				if NIHILUS_UNLIKELY (length == 0) {
					return std::string::npos;
				}
				while (i < static_cast<int64_t>(length)) {
					char c = *text;
					if NIHILUS_UNLIKELY (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
						return static_cast<uint64_t>(i);
					}
					++text;
					++i;
				}
				return std::string::npos;
			}

#if NIHILUS_AVX512 || NIHILUS_SVE2
			if (i + 64ll <= static_cast<int64_t>(length)) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 2>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 2>::type::bytes_processed;
				NIHILUS_ALIGN(vector_size) char values_to_load[vector_size];
				while (i + static_cast<int64_t>(vector_size) <= static_cast<int64_t>(length)) {
					std::memcpy(values_to_load, text, vector_size);
					uint64_t ws_mask	 = collect_whitespace_indices<simd_type>(values_to_load);
					uint64_t non_ws_mask = static_cast<uint64_t>(~ws_mask);
					if (non_ws_mask != 0) {
						return static_cast<uint64_t>(i) + static_cast<uint64_t>(std::countr_zero(non_ws_mask));
					}

					text += vector_size;
					i += vector_size;
				}
			}
#endif

#if NIHILUS_AVX2 || NIHILUS_AVX512 || NIHILUS_SVE2
			if (i + 32ll <= static_cast<int64_t>(length)) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 1>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 1>::type::bytes_processed;
				NIHILUS_ALIGN(vector_size) char values_to_load[vector_size];
				while (i + static_cast<int64_t>(vector_size) <= static_cast<int64_t>(length)) {
					std::memcpy(values_to_load, text, vector_size);
					uint32_t ws_mask	 = collect_whitespace_indices<simd_type>(values_to_load);
					uint32_t non_ws_mask = static_cast<uint32_t>(~ws_mask);
					if (non_ws_mask != 0) {
						return static_cast<uint64_t>(i) + static_cast<uint64_t>(std::countr_zero(non_ws_mask));
					}

					text += vector_size;
					i += vector_size;
				}
			}
#endif
			if (i + 16ll <= static_cast<int64_t>(length)) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 0>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 0>::type::bytes_processed;
				NIHILUS_ALIGN(vector_size) char values_to_load[vector_size];
				while (i + static_cast<int64_t>(vector_size) <= static_cast<int64_t>(length)) {
					std::memcpy(values_to_load, text, vector_size);
					uint16_t ws_mask	 = collect_whitespace_indices<simd_type>(values_to_load);
					uint16_t non_ws_mask = static_cast<uint16_t>(~ws_mask);
					if (non_ws_mask != 0) {
						return static_cast<uint64_t>(i) + static_cast<uint64_t>(std::countr_zero(non_ws_mask));
					}

					text += vector_size;
					i += vector_size;
				}
			}

			while (i < static_cast<int64_t>(length)) {
				char c = *text;
				if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
					return static_cast<uint64_t>(i);
				}
				++text;
				++i;
			}

			return std::string::npos;
		}

		template<typename char_type> NIHILUS_HOST static uint64_t find_last_not_of(const char_type* text, uint64_t length) noexcept {
			int64_t i = static_cast<int64_t>(length);

#if NIHILUS_AVX512 || NIHILUS_SVE2
			if (i >= 0 || length >= 64) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 2>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 2>::type::bytes_processed;
				NIHILUS_ALIGN(64) char values_to_load[64];

				if (i < 0) {
					i = static_cast<int64_t>(length) - static_cast<int64_t>(vector_size);
				}

				while (i >= 0) {
					std::memcpy(values_to_load, text + i, 64);

					uint64_t ws_mask	 = collect_whitespace_indices<simd_type>(values_to_load);
					uint64_t non_ws_mask = ~ws_mask;

					if (non_ws_mask != 0) {
						int64_t last_bit_pos = 63ll - static_cast<int64_t>(lzcnt(non_ws_mask));
						return static_cast<uint64_t>(i + last_bit_pos);
					}

					i -= vector_size;
				}
			}
#endif

#if NIHILUS_AVX2 || NIHILUS_AVX512 || NIHILUS_SVE2
			if (i >= 0 || length >= 32) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 1>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 1>::type::bytes_processed;
				NIHILUS_ALIGN(32) char values_to_load[32];

				if (i < 0) {
					i = static_cast<int64_t>(length) - static_cast<int64_t>(vector_size);
				}

				while (i >= 0) {
					std::memcpy(values_to_load, text + i, 32);

					uint32_t ws_mask	 = collect_whitespace_indices<simd_type>(values_to_load);
					uint32_t non_ws_mask = ~ws_mask;

					if (non_ws_mask != 0) {
						int64_t last_bit_pos = 31ll - static_cast<int64_t>(lzcnt(non_ws_mask));
						return static_cast<uint64_t>(i + last_bit_pos);
					}

					i -= static_cast<int64_t>(vector_size);
				}
			}
#endif

			if (i >= 0 || length >= 16) {
				using simd_type						  = typename detail::get_type_at_index<avx_list, 0>::type::type;
				static constexpr uint64_t vector_size = detail::get_type_at_index<avx_list, 0>::type::bytes_processed;
				NIHILUS_ALIGN(16) char values_to_load[16];

				if (i < 0) {
					i = static_cast<int64_t>(length) - static_cast<int64_t>(vector_size);
				}

				while (i >= 0) {
					std::memcpy(values_to_load, text + i, 16);

					uint16_t ws_mask	 = collect_whitespace_indices<simd_type>(values_to_load);
					uint16_t non_ws_mask = ~ws_mask;

					if (non_ws_mask != 0) {
						int64_t last_bit_pos = 15ll - static_cast<int64_t>(lzcnt(non_ws_mask));
						return static_cast<uint64_t>(i + last_bit_pos);
					}

					i -= static_cast<int64_t>(vector_size);
				}
			}

			return std::string::npos;
		}
	};

}
