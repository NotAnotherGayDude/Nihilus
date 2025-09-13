/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/jsonifier
/// Sep 1, 2024
#pragma once

#include <nihilus-incl/common/config.hpp>
#include <type_traits>
#include <algorithm>
#include <ostream>

namespace nihilus {

	template<uint64_t size_val> struct string_literal {
		using value_type	  = char;
		using const_reference = const value_type&;
		using reference		  = value_type&;
		using const_pointer	  = const value_type*;
		using pointer		  = value_type*;
		using size_type		  = uint64_t;

		inline static constexpr static_aligned_const length{ size_val > 0 ? size_val - 1 : 0 };

		NIHILUS_INLINE constexpr string_literal() noexcept = default;

		NIHILUS_INLINE constexpr string_literal(const char (&str)[size_val]) noexcept {
			for (uint64_t x = 0; x < length; ++x) {
				values[x] = str[x];
			}
			values[static_cast<int64_t>(length)] = '\0';
		}

		NIHILUS_INLINE explicit constexpr string_literal(const char* str) noexcept {
			for (uint64_t x = 0; x < length; ++x) {
				values[x] = str[x];
			}
			values[static_cast<int64_t>(length)] = '\0';
		}

		NIHILUS_INLINE constexpr const_pointer data() const noexcept {
			return values;
		}

		NIHILUS_INLINE constexpr pointer data() noexcept {
			return values;
		}

		template<size_type sizeNew> NIHILUS_INLINE constexpr auto operator+=(const string_literal<sizeNew>& str) const noexcept {
			string_literal<sizeNew + size_val - 1> new_literal{};
			std::copy(values, values + size(), new_literal.data());
			std::copy(str.data(), str.data() + sizeNew, new_literal.data() + size());
			return new_literal;
		}

		template<size_type sizeNew> NIHILUS_INLINE constexpr auto operator+=(const value_type (&str)[sizeNew]) const noexcept {
			string_literal<sizeNew + size_val - 1> new_literal{};
			std::copy(values, values + size(), new_literal.data());
			std::copy(str, str + sizeNew, new_literal.data() + size());
			return new_literal;
		}

		template<size_type sizeNew> NIHILUS_INLINE constexpr auto operator+(const string_literal<sizeNew>& str) const noexcept {
			string_literal<sizeNew + size_val - 1> new_literal{};
			std::copy(values, values + size(), new_literal.data());
			std::copy(str.data(), str.data() + sizeNew, new_literal.data() + size());
			return new_literal;
		}

		template<size_type sizeNew> NIHILUS_INLINE constexpr auto operator+(const value_type (&str)[sizeNew]) const noexcept {
			string_literal<sizeNew + size_val - 1> new_literal{};
			std::copy(values, values + size(), new_literal.data());
			std::copy(str, str + sizeNew, new_literal.data() + size());
			return new_literal;
		}

		template<size_type sizeNew> NIHILUS_INLINE constexpr friend auto operator+(const value_type (&lhs)[sizeNew], const string_literal<size_val>& str) noexcept {
			string_literal<sizeNew> sl_new{};
			std::copy_n(lhs, sizeNew, sl_new.data());
			return sl_new + str;
		}

		NIHILUS_INLINE constexpr reference operator[](size_type index) noexcept {
			return values[index];
		}

		NIHILUS_INLINE constexpr const_reference operator[](size_type index) const noexcept {
			return values[index];
		}

		NIHILUS_INLINE static constexpr size_type size() noexcept {
			return length;
		}

		template<typename string_types> NIHILUS_INLINE constexpr operator string_types() const noexcept {
			alignas(64) string_types return_values{ values, length };
			return return_values;
		}

		alignas(64) value_type values[size_val]{};
	};

	template<uint64_t size> string_literal(char (&)[size]) -> string_literal<size>;

	inline static constexpr uint64_t count_digits(int64_t number) noexcept {
		uint64_t count = 0;
		if (number < 0) {
			number *= -1;
			++count;
		}
		do {
			++count;
			number /= 10;
		} while (number != 0);
		return count;
	}

	template<int64_t number, uint64_t numDigits = count_digits(number)> inline static constexpr string_literal<numDigits + 1> to_string_literal() noexcept {
		char buffer[numDigits + 1]{};
		char* ptr = buffer + numDigits;
		*ptr	  = '\0';
		int64_t temp{};
		if constexpr (number < 0) {
			temp			   = number * -1;
			*(ptr - numDigits) = '-';
		} else {
			temp = number;
		}
		do {
			*--ptr = static_cast<char>(static_cast<int64_t>('0') + (temp % 10ll));
			temp /= 10;
		} while (temp != 0);
		return string_literal<numDigits + 1>{ buffer };
	}

	constexpr uint64_t str_len(const char* input) noexcept {
		uint64_t return_val{};
		while (input[return_val] != '\0') {
			++return_val;
		}
		return return_val;
	}

}
