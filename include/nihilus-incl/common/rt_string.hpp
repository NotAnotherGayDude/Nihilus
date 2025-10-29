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
// cuda_12.hpp

#pragma once

#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/common/compare.hpp>

namespace nihilus {

	struct rt_string_view {
		using value_type			 = char;
		using size_type				 = uint64_t;
		using difference_type		 = ptrdiff_t;
		using const_pointer			 = const value_type*;
		using const_reference		 = const value_type&;
		using const_iterator		 = basic_iterator<const value_type>;
		using iterator				 = basic_iterator<value_type>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		NIHILUS_HOST constexpr rt_string_view() {
		}

		NIHILUS_HOST constexpr rt_string_view(const_pointer string, size_type size) : data_val{ string }, size_val{ size } {
		}

		NIHILUS_HOST constexpr rt_string_view& operator=(rt_string_view&& other) noexcept {
			if (this != &other) {
				rt_string_view new_string_view{ std::move(other) };
				swap(new_string_view);
			}
			return *this;
		}

		NIHILUS_HOST constexpr rt_string_view(rt_string_view&& other) noexcept : data_val{ other.data_val }, size_val{ other.size_val } {
		}

		NIHILUS_HOST constexpr rt_string_view& operator=(const rt_string_view& other) noexcept {
			if (this != &other) {
				data_val = other.data_val;
				size_val = other.size_val;
			}
			return *this;
		}

		NIHILUS_HOST constexpr rt_string_view(const rt_string_view& other) noexcept : data_val{ other.data_val }, size_val{ other.size_val } {
		}

		NIHILUS_HOST constexpr rt_string_view(const std::string& other) noexcept : data_val{ other.data() }, size_val{ other.size() } {
		}

		NIHILUS_HOST constexpr rt_string_view(const std::string_view& other) noexcept : data_val{ other.data() }, size_val{ other.size() } {
		}

		NIHILUS_HOST rt_string_view substr(const size_type offset_new = 0, size_type count_new = std::string::npos) const {
			if NIHILUS_UNLIKELY (offset_new > size_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<true, "Substring position is out of range.", location>::impl();
			}

			count_new = detail::min(count_new, size_val - offset_new);
			return { data_val + offset_new, count_new };
		}

		NIHILUS_HOST auto begin() const {
			return const_iterator{ data_val };
		}

		NIHILUS_HOST auto end() const {
			return const_iterator{ data_val + size_val };
		}

		NIHILUS_HOST const_reference operator[](size_type index) const {
			return data_val[index];
		}

		NIHILUS_HOST bool empty() const {
			return size_val == 0;
		}

		NIHILUS_HOST void swap(rt_string_view& other) {
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
		}

		NIHILUS_HOST size_type size() const {
			return size_val;
		}

		NIHILUS_HOST const_pointer data() const {
			return data_val;
		}

		NIHILUS_HOST bool operator==(const rt_string_view& other) const {
			if (size_val == other.size_val) {
				return comparison::compare(data_val, other.data_val, size_val);
			} else {
				return false;
			}
		}

		NIHILUS_HOST uint64_t find_first_non_whitespace() const {
			return whitespace_search::find_first_not_of(data_val, size_val);
		}

		NIHILUS_HOST uint64_t find_last_non_whitespace() const {
			return whitespace_search::find_last_not_of(data_val, size_val);
		}

		NIHILUS_HOST uint64_t find_first_non_alpha() const {
			return alpha_search::find_first_not_of(data_val, size_val);
		}

		NIHILUS_HOST operator std::string_view() const {
			return { data_val, size_val };
		}

		NIHILUS_HOST constexpr ~rt_string_view() {
		}

	  protected:
		const_pointer data_val{};
		size_type size_val{};
	};

	struct rt_string : allocator<char> {
		using value_type			 = char;
		using size_type				 = uint64_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = basic_iterator<value_type>;
		using const_iterator		 = basic_iterator<const value_type>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;
		using allocator_type		 = allocator<value_type>;
		using allocator_traits		 = std::allocator_traits<allocator_type>;

		NIHILUS_HOST_DEVICE rt_string() {
		}

		NIHILUS_HOST rt_string(char value) {
			resize(1);
			if (data_val) {
				data_val[0] = value;
			}
		}

		NIHILUS_HOST rt_string& operator=(rt_string&& other) {
			if (this != &other) {
				std::swap(capacity_val, other.capacity_val);
				if (other.data_val) {
					std::swap(data_val, other.data_val);
				}
				std::swap(size_val, other.size_val);
			}
			return *this;
		}

		NIHILUS_HOST rt_string(rt_string&& other) noexcept : allocator_type{} {
			*this = std::move(other);
		}

		NIHILUS_HOST rt_string& operator=(const rt_string& other) {
			if (this != &other) {
				resize(other.size_val);
				std::memcpy(data_val, other.data_val, size_val);
			}
			return *this;
		}

		NIHILUS_HOST rt_string(const rt_string& other) noexcept : allocator_type{} {
			*this = other;
		}

		NIHILUS_HOST rt_string& operator=(const std::string& other) {
			resize(other.size());
			std::memcpy(data_val, other.data(), size_val);
			return *this;
		}

		NIHILUS_HOST rt_string(const std::string& other) noexcept : allocator_type{} {
			*this = other;
		}

		NIHILUS_HOST void resize(size_type size_new) noexcept {
			if NIHILUS_LIKELY (size_new > size_val) {
				reserve(size_new);
				if (data_val) {
					std::uninitialized_value_construct_n(data_val + size_val, size_new - size_val);
				}
				size_val = size_new;
			} else if NIHILUS_UNLIKELY (size_new < size_val) {
				std::destroy(data_val + size_new, data_val + size_val);
				size_val = size_new;
			}
			return;
		}

		NIHILUS_HOST void reserve(size_type size_new) noexcept {
			if NIHILUS_UNLIKELY (size_new > capacity_val) {
				size_type old_capacity = capacity_val;
				pointer old_data	   = data_val;
				data_val			   = allocator_traits::allocate(*this, size_new);
				capacity_val		   = size_new;
				if (old_data && data_val) {
					std::uninitialized_move_n(old_data, size_val, data_val);
				}
				allocator_traits::deallocate(*this, old_data, old_capacity);
			}
			return;
		}

		NIHILUS_HOST rt_string substr(size_type position, size_type count = std::numeric_limits<size_type>::max()) const {
			if NIHILUS_UNLIKELY (static_cast<int64_t>(position) >= static_cast<int64_t>(size_val)) {
				throw std::out_of_range("Substring position is out of range.");
			}

			count = detail::min(count, size_val - position);

			rt_string result{};
			if NIHILUS_LIKELY (count > 0) {
				result.resize(count);
				std::copy(data_val + position, data_val + position + count * sizeof(value_type), result.data_val);
			}
			return result;
		}

		NIHILUS_HOST iterator begin() noexcept {
			return iterator(data_val);
		}

		NIHILUS_HOST iterator end() noexcept {
			return iterator(data_val + size_val);
		}

		NIHILUS_HOST const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		NIHILUS_HOST const_iterator end() const noexcept {
			return const_iterator(data_val + size_val);
		}

		NIHILUS_HOST reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_HOST reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_HOST const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(cend());
		}

		NIHILUS_HOST const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(cbegin());
		}

		NIHILUS_HOST const_iterator cbegin() const noexcept {
			return begin();
		}

		NIHILUS_HOST const_iterator cend() const noexcept {
			return end();
		}

		NIHILUS_HOST const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		NIHILUS_HOST const_reverse_iterator crend() const noexcept {
			return rend();
		}

		NIHILUS_HOST reference operator[](size_type index) {
			return data_val[index];
		}

		NIHILUS_HOST const_reference operator[](size_type index) const {
			return data_val[index];
		}

		NIHILUS_HOST rt_string operator+(const rt_string& other) const {
			rt_string new_string{};
			new_string.resize(size_val + other.size_val);
			std::memcpy(new_string.data(), data_val, size_val);
			std::memcpy(new_string.data() + size_val, other.data_val, other.size_val);
			return new_string;
		}

		NIHILUS_HOST rt_string& operator+=(value_type other) {
			size_type old_size = size_val;
			resize(size_val + 1);
			if (data_val) {
				data_val[old_size] = other;
			}
			return *this;
		}

		NIHILUS_HOST rt_string& operator+=(const rt_string& other) {
			size_type old_size = size_val;
			resize(size_val + other.size_val);
			std::memcpy(data_val + old_size, other.data_val, other.size_val);
			return *this;
		}

		template<size_type size> NIHILUS_HOST rt_string& operator+=(const char (&other)[size]) {
			size_type old_size = size_val;
			resize(size_val + size - 1);
			std::memcpy(data_val + old_size, other, size - 1);
			return *this;
		}

		NIHILUS_HOST bool empty() const {
			return size_val == 0;
		}

		NIHILUS_HOST size_type size() const {
			return size_val;
		}

		NIHILUS_HOST const_pointer data() const {
			return data_val;
		}

		NIHILUS_HOST pointer data() {
			return data_val;
		}

		NIHILUS_HOST void set_values(const_pointer string, size_type size_new) {
			resize(size_new);
			std::memcpy(data_val, string, size_new);
		}

		NIHILUS_HOST void set_values(value_type value_new) {
			resize(1);
			data_val[0] = value_new;
		}

		NIHILUS_HOST void clear() {
			size_val = 0;
		}

		NIHILUS_HOST bool operator==(const rt_string& other) const {
			if (size_val == other.size_val) {
				return comparison::compare(data_val, other.data_val, size_val);
			} else {
				return false;
			}
		}

		NIHILUS_HOST uint64_t find_first_non_whitespace() const {
			return whitespace_search::find_first_not_of(data_val, size_val);
		}

		NIHILUS_HOST uint64_t find_last_non_whitespace() const {
			return whitespace_search::find_last_not_of(data_val, size_val);
		}

		NIHILUS_HOST uint64_t find_first_non_alpha() const {
			return alpha_search::find_first_not_of(data_val, size_val);
		}

		NIHILUS_HOST operator rt_string_view() const {
			return { data_val, size_val };
		}

		NIHILUS_HOST operator std::string_view() const {
			return { data_val, size_val };
		}

		NIHILUS_HOST ~rt_string() {
			if (data_val && size_val && capacity_val) {
				allocator_traits::deallocate(*this, data_val, capacity_val);
			}
		}

	  protected:
		size_type capacity_val{};
		size_type size_val{};
		pointer data_val{};
	};

}
