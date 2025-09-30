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
#include <iterator>

namespace nihilus {

	template<typename value_type_new, uint64_t size> class array_iterator {
	  public:
		using iterator_concept	= std::contiguous_iterator_tag;
		using iterator_category = std::random_access_iterator_tag;
		using element_type		= value_type_new;
		using value_type		= value_type_new;
		using difference_type	= std::ptrdiff_t;
		using pointer			= value_type*;
		using reference			= value_type&;

		NIHILUS_HOST constexpr array_iterator() noexcept : ptr() {
		}

		NIHILUS_HOST constexpr array_iterator(pointer ptrNew) noexcept : ptr(ptrNew) {
		}

		NIHILUS_HOST constexpr reference operator*() const noexcept {
			return *ptr;
		}

		NIHILUS_HOST constexpr pointer operator->() const noexcept {
			return std::pointer_traits<pointer>::pointer_to(**this);
		}

		NIHILUS_HOST constexpr array_iterator& operator++() noexcept {
			++ptr;
			return *this;
		}

		NIHILUS_HOST constexpr array_iterator operator++(int32_t) noexcept {
			array_iterator temp = *this;
			++*this;
			return temp;
		}

		NIHILUS_HOST constexpr array_iterator& operator--() noexcept {
			--ptr;
			return *this;
		}

		NIHILUS_HOST constexpr array_iterator operator--(int32_t) noexcept {
			array_iterator temp = *this;
			--*this;
			return temp;
		}

		NIHILUS_HOST constexpr array_iterator& operator+=(const difference_type offSet) noexcept {
			ptr += offSet;
			return *this;
		}

		NIHILUS_HOST constexpr array_iterator operator+(const difference_type offSet) const noexcept {
			array_iterator temp = *this;
			temp += offSet;
			return temp;
		}

		NIHILUS_HOST friend constexpr array_iterator operator+(const difference_type offSet, array_iterator next) noexcept {
			next += offSet;
			return next;
		}

		NIHILUS_HOST constexpr array_iterator& operator-=(const difference_type offSet) noexcept {
			return *this += -offSet;
		}

		NIHILUS_HOST constexpr array_iterator operator-(const difference_type offSet) const noexcept {
			array_iterator temp = *this;
			temp -= offSet;
			return temp;
		}

		NIHILUS_HOST constexpr difference_type operator-(const array_iterator& other) const noexcept {
			return static_cast<difference_type>(ptr - other.ptr);
		}

		NIHILUS_HOST constexpr reference operator[](const difference_type offSet) const noexcept {
			return *(*this + offSet);
		}

		NIHILUS_HOST constexpr bool operator==(const array_iterator& other) const noexcept {
			return ptr == other.ptr;
		}

		NIHILUS_HOST constexpr std::strong_ordering operator<=>(const array_iterator& other) const noexcept {
			return ptr <=> other.ptr;
		}

		NIHILUS_HOST constexpr bool operator!=(const array_iterator& other) const noexcept {
			return !(*this == other);
		}

		NIHILUS_HOST constexpr bool operator<(const array_iterator& other) const noexcept {
			return ptr < other.ptr;
		}

		NIHILUS_HOST constexpr bool operator>(const array_iterator& other) const noexcept {
			return other < *this;
		}

		NIHILUS_HOST constexpr bool operator<=(const array_iterator& other) const noexcept {
			return !(other < *this);
		}

		NIHILUS_HOST constexpr bool operator>=(const array_iterator& other) const noexcept {
			return !(*this < other);
		}

		NIHILUS_ALIGN(64) pointer ptr;
	};

	template<typename value_type_new> class array_iterator<value_type_new, 0> {
	  public:
		using iterator_concept	= std::contiguous_iterator_tag;
		using iterator_category = std::random_access_iterator_tag;
		using element_type		= value_type_new;
		using value_type		= value_type_new;
		using difference_type	= std::ptrdiff_t;
		using pointer			= value_type*;
		using reference			= value_type&;

		NIHILUS_HOST constexpr array_iterator() noexcept {
		}

		NIHILUS_HOST constexpr array_iterator(std::nullptr_t ptrNew) noexcept {
			( void )ptrNew;
		}

		NIHILUS_HOST constexpr bool operator==(const array_iterator& other) const noexcept {
			( void )other;
			return true;
		}

		NIHILUS_HOST constexpr bool operator!=(const array_iterator& other) const noexcept {
			return !(*this == other);
		}

		NIHILUS_HOST constexpr bool operator>=(const array_iterator& other) const noexcept {
			return !(*this < other);
		}
	};

	template<typename value_type_new> class basic_iterator {
	  public:
		using iterator_concept	= std::contiguous_iterator_tag;
		using iterator_category = std::random_access_iterator_tag;
		using element_type		= value_type_new;
		using value_type		= value_type_new;
		using difference_type	= std::ptrdiff_t;
		using pointer			= value_type*;
		using reference			= value_type&;

		NIHILUS_HOST constexpr basic_iterator() noexcept : ptr() {
		}

		NIHILUS_HOST constexpr basic_iterator(pointer ptrNew) noexcept : ptr(ptrNew) {
		}

		NIHILUS_HOST constexpr reference operator*() const noexcept {
			return *ptr;
		}

		NIHILUS_HOST constexpr pointer operator->() const noexcept {
			return std::pointer_traits<pointer>::pointer_to(**this);
		}

		NIHILUS_HOST constexpr basic_iterator& operator++() noexcept {
			++ptr;
			return *this;
		}

		NIHILUS_HOST constexpr basic_iterator operator++(int32_t) noexcept {
			basic_iterator temp = *this;
			++*this;
			return temp;
		}

		NIHILUS_HOST constexpr basic_iterator& operator--() noexcept {
			--ptr;
			return *this;
		}

		NIHILUS_HOST constexpr basic_iterator operator--(int32_t) noexcept {
			basic_iterator temp = *this;
			--*this;
			return temp;
		}

		NIHILUS_HOST constexpr basic_iterator& operator+=(const difference_type offSet) noexcept {
			ptr += offSet;
			return *this;
		}

		NIHILUS_HOST constexpr basic_iterator operator+(const difference_type offSet) const noexcept {
			basic_iterator temp = *this;
			temp += offSet;
			return temp;
		}

		NIHILUS_HOST friend constexpr basic_iterator operator+(const difference_type offSet, basic_iterator next) noexcept {
			next += offSet;
			return next;
		}

		NIHILUS_HOST constexpr basic_iterator& operator-=(const difference_type offSet) noexcept {
			return *this += -offSet;
		}

		NIHILUS_HOST constexpr basic_iterator operator-(const difference_type offSet) const noexcept {
			basic_iterator temp = *this;
			temp -= offSet;
			return temp;
		}

		NIHILUS_HOST constexpr difference_type operator-(const basic_iterator& other) const noexcept {
			return static_cast<difference_type>(ptr - other.ptr);
		}

		NIHILUS_HOST constexpr reference operator[](const difference_type offSet) const noexcept {
			return *(*this + offSet);
		}

		NIHILUS_HOST constexpr bool operator==(const basic_iterator& other) const noexcept {
			return ptr == other.ptr;
		}

		NIHILUS_HOST constexpr std::strong_ordering operator<=>(const basic_iterator& other) const noexcept {
			return ptr <=> other.ptr;
		}

		NIHILUS_HOST constexpr bool operator!=(const basic_iterator& other) const noexcept {
			return !(*this == other);
		}

		NIHILUS_HOST constexpr bool operator<(const basic_iterator& other) const noexcept {
			return ptr < other.ptr;
		}

		NIHILUS_HOST constexpr bool operator>(const basic_iterator& other) const noexcept {
			return other < *this;
		}

		NIHILUS_HOST constexpr bool operator<=(const basic_iterator& other) const noexcept {
			return !(other < *this);
		}

		NIHILUS_HOST constexpr bool operator>=(const basic_iterator& other) const noexcept {
			return !(*this < other);
		}

		NIHILUS_ALIGN(64) pointer ptr;
	};

}
