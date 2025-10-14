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

#include <nihilus-incl/common/allocator.hpp>
#include <nihilus-incl/common/exception.hpp>
#include <nihilus-incl/common/iterator.hpp>
#include <nihilus-incl/common/concepts.hpp>
#include <algorithm>
#include <stdexcept>

namespace nihilus {

	template<typename value_type_new> struct NIHILUS_ALIGN(64) aligned_vector : protected allocator<value_type_new> {
		using value_type			 = value_type_new;
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

		NIHILUS_HOST aligned_vector() noexcept {
		}

		NIHILUS_HOST aligned_vector& operator=(aligned_vector&& other) noexcept
			requires(std::is_move_assignable_v<value_type>)
		{
			if NIHILUS_LIKELY (this != &other) {
				aligned_vector vector_new{ std::move(other) };
				swap(vector_new);
			}
			return *this;
		}

		NIHILUS_HOST aligned_vector(aligned_vector&& other) noexcept
			requires(std::is_move_assignable_v<value_type>)
		{
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
			std::swap(capacity_val, other.capacity_val);
		}

		NIHILUS_HOST aligned_vector& operator=(const aligned_vector& other)
			requires(std::is_copy_assignable_v<value_type>)
		{
			if NIHILUS_LIKELY (this != &other) {
				aligned_vector vector_new{ other };
				swap(vector_new);
			}
			return *this;
		}

		NIHILUS_HOST aligned_vector(const aligned_vector& other)
			requires(std::is_copy_constructible_v<value_type>)
		{
			reserve(other.capacity_val);
			size_val = other.size_val;
			if (data_val) {
				std::uninitialized_copy_n(other.data(), other.size(), data_val);
			}
		}

		template<typename... arg_types> NIHILUS_HOST aligned_vector(arg_types&&... args)
			requires((std::is_constructible_v<value_type, arg_types> && ...) && std::is_move_constructible_v<value_type>)
		{
			static constexpr uint64_t size_new{ sizeof...(arg_types) };
			value_type values[size_new]{ detail::forward<arg_types>(args)... };
			reserve(size_new);
			for (uint64_t x = 0; x < size_new; ++x) {
				allocator_traits::construct(*this, data_val + x, std::move(values[x]));
			}
		}

		template<typename... arg_types> NIHILUS_HOST aligned_vector(const arg_types&... args)
			requires((std::is_constructible_v<value_type, arg_types> && ...) && std::is_copy_constructible_v<value_type> && !std::is_move_constructible_v<value_type>)
		{
			static constexpr uint64_t size_new{ sizeof...(arg_types) };
			value_type values[size_new]{ detail::forward<arg_types>(args)... };
			reserve(size_new);
			for (uint64_t x = 0; x < size_new; ++x) {
				allocator_traits::construct(*this, data_val + x, values[x]);
			}
		}

		NIHILUS_HOST aligned_vector& operator=(aligned_vector&& other)
			requires(!std::is_move_assignable_v<value_type>)
		= delete;

		NIHILUS_HOST aligned_vector(aligned_vector&& other)
			requires(!std::is_move_assignable_v<value_type>)
		= delete;

		NIHILUS_HOST aligned_vector& operator=(const aligned_vector& other)
			requires(!std::is_copy_assignable_v<value_type>)
		= delete;

		NIHILUS_HOST aligned_vector(const aligned_vector& other)
			requires(!std::is_copy_assignable_v<value_type>)
		= delete;

		NIHILUS_HOST iterator begin() noexcept {
			return iterator(data_val);
		}

		NIHILUS_HOST const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		NIHILUS_HOST iterator end() noexcept {
			return iterator(data_val + size_val);
		}

		NIHILUS_HOST const_iterator end() const noexcept {
			return const_iterator(data_val + size_val);
		}

		NIHILUS_HOST reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_HOST const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		NIHILUS_HOST reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_HOST const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
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

		template<typename... value_type_newer> NIHILUS_HOST iterator emplace_back(value_type_newer&&... value_new) {
			if NIHILUS_UNLIKELY (size_val + 1 > capacity_val) {
				reserve(detail::max(size_val * 2, size_val + 1));
			}
			if constexpr (sizeof...(value_type_newer) > 0) {
				if (data_val) {
					allocator_traits::construct(*this, data_val + size_val, std::forward<value_type_newer>(value_new)...);
				}
			} else {
				if (data_val) {
					allocator_traits::construct(*this, data_val + size_val, value_type{});
				}
			}
			++size_val;
			return iterator{ data_val + size_val - 1 };
		}

		template<typename iterator_type = iterator> NIHILUS_HOST iterator erase(iterator_type&& value_new) {
			size_type current_index = value_new - data_val;
			if (current_index >= size_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<true, "Invalid erasure index", location>::impl();
			}
			for (size_type x = current_index; x < size_val - 1; ++x) {
				data_val[x] = detail::move(data_val[x + 1]);
			}
			--size_val;
			return iterator{ data_val + current_index };
		}

		NIHILUS_HOST void swap(aligned_vector& other) noexcept {
			std::swap(capacity_val, other.capacity_val);
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
		}

		NIHILUS_HOST uint64_t size() const noexcept {
			return size_val;
		}

		NIHILUS_HOST uint64_t max_size() const noexcept {
			return allocator_traits::max_size(*this);
		}

		NIHILUS_HOST void clear() noexcept {
			resize(0);
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
				if constexpr (std::is_move_constructible_v<value_type>) {
					if (old_data && data_val) {
						std::uninitialized_move_n(old_data, size_val, data_val);
					}
				} else if constexpr (std::is_copy_constructible_v<value_type>) {
					if (old_data && data_val) {
						std::uninitialized_copy_n(old_data, size_val, data_val);
					}
				}
				allocator_traits::deallocate(*this, old_data, old_capacity);
			}
			return;
		}

		template<integral_or_enum_types index_type> NIHILUS_HOST reference at(index_type position) {
			if NIHILUS_UNLIKELY (size_val <= position) {
				throw std::runtime_error{ "invalid aligned_vector<value_type> subscript" };
			}
			if (!data_val) {
				throw std::runtime_error{ "invalid data_val value" };
			}
			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum_types index_type> NIHILUS_HOST const_reference at(index_type position) const {
			if NIHILUS_UNLIKELY (size_val <= position) {
				throw std::runtime_error{ "invalid aligned_vector<value_type> subscript" };
			}
			if (!data_val) {
				throw std::runtime_error{ "invalid data_val value" };
			}
			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum_types index_type> reference operator[](index_type position) noexcept {
			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum_types index_type> const_reference operator[](index_type position) const noexcept {
			return data_val[static_cast<uint64_t>(position)];
		}

		template<uint64_t index> NIHILUS_HOST reference operator[](tag<index> index_new) {
			return data_val[index_new];
		}

		template<uint64_t index> NIHILUS_HOST const_reference operator[](tag<index> index_new) const {
			return data_val[index_new];
		}

		NIHILUS_HOST reference front() noexcept {
			return data_val[0];
		}

		NIHILUS_HOST const_reference front() const noexcept {
			return data_val[0];
		}

		NIHILUS_HOST reference back() noexcept {
			return data_val[size_val - 1];
		}

		NIHILUS_HOST const_reference back() const noexcept {
			return data_val[size_val - 1];
		}

		NIHILUS_HOST value_type* data() noexcept {
			return data_val;
		}

		NIHILUS_HOST const value_type* data() const noexcept {
			return data_val;
		}

		NIHILUS_HOST bool empty() const noexcept {
			return size_val == 0;
		}

		NIHILUS_HOST friend bool operator==(const aligned_vector& lhs, const aligned_vector& rhs) {
			if NIHILUS_UNLIKELY (lhs.size_val != rhs.size_val) {
				return false;
			}
			for (uint64_t x = 0; x < lhs.size_val; ++x) {
				if (lhs[x] != rhs[x]) {
					return false;
				}
			}
			return true;
		}

		NIHILUS_HOST ~aligned_vector() {
			clear();
			if NIHILUS_LIKELY (capacity_val && data_val) {
				allocator_traits::deallocate(*this, data_val, capacity_val);
			}
		}

	  protected:
		NIHILUS_ALIGN(64) size_type capacity_val {};
		NIHILUS_ALIGN(64) size_type size_val {};
		NIHILUS_ALIGN(64) pointer data_val {};
	};

}
