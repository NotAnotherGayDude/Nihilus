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
#include <nihilus-incl/common/iterator.hpp>
#include <algorithm>
#include <stdexcept>

namespace nihilus {

	template<typename value_type_new> struct alignas(64) aligned_vector : protected allocator<value_type_new> {
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
		using allocator				 = allocator<value_type>;
		using allocator_traits		 = std::allocator_traits<allocator>;

		NIHILUS_INLINE aligned_vector() noexcept = default;

		NIHILUS_INLINE aligned_vector& operator=(aligned_vector&& other) noexcept
			requires(std::is_move_assignable_v<value_type>)
		{
			if (this != &other) {
				std::swap(data_val, other.data_val);
				std::swap(size_val, other.size_val);
				std::swap(capacity_val, other.capacity_val);
			}
			return *this;
		}

		NIHILUS_INLINE aligned_vector(aligned_vector&& other) noexcept
			requires(std::is_move_assignable_v<value_type>)
		{
			std::swap(data_val, other.data_val);
			std::swap(size_val, other.size_val);
			std::swap(capacity_val, other.capacity_val);
		}

		NIHILUS_INLINE constexpr aligned_vector& operator=(const aligned_vector& other)
			requires(std::is_copy_assignable_v<value_type>)
		{
			if (this != &other) {
				reserve(other.capacity_val);
				size_val = other.size_val;
				std::uninitialized_copy_n(other.data(), other.size(), data_val);
			}
			return *this;
		}

		NIHILUS_INLINE constexpr aligned_vector(const aligned_vector& other)
			requires(std::is_copy_constructible_v<value_type>)
		{
			reserve(other.capacity_val);
			size_val = other.size_val;
			std::uninitialized_copy_n(other.data(), other.size(), data_val);
		}

		NIHILUS_INLINE constexpr aligned_vector& operator=(const aligned_vector& other)
			requires(!std::is_copy_assignable_v<value_type>)
		= delete;

		NIHILUS_INLINE constexpr aligned_vector(const aligned_vector& other)
			requires(!std::is_copy_assignable_v<value_type>)
		= delete;

		NIHILUS_INLINE constexpr aligned_vector(std::initializer_list<value_type> values)
			requires(std::is_copy_assignable_v<value_type>)
		{
			reserve(values.size());
			size_val = values.size();
			std::uninitialized_move_n(values.begin(), size_val, data_val);
		}

		NIHILUS_INLINE constexpr iterator begin() noexcept {
			return iterator(data_val);
		}

		NIHILUS_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		NIHILUS_INLINE constexpr iterator end() noexcept {
			return iterator(data_val + size_val);
		}

		NIHILUS_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator(data_val + size_val);
		}

		NIHILUS_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		NIHILUS_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		NIHILUS_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		NIHILUS_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		NIHILUS_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		template<typename... value_type_newer> NIHILUS_INLINE iterator emplace_back(value_type_newer&&... value_new) {
			if (size_val + 1 > capacity_val) {
				reserve(detail::max(size_val * 2, size_val + 1));
			}
			if constexpr (sizeof...(value_type_newer) > 0) {
				allocator_traits::construct(*this, data_val + size_val, std::forward<value_type_newer>(value_new)...);
			} else {
				allocator_traits::construct(*this, data_val + size_val, value_type{});
			}
			++size_val;
			return iterator{ data_val + size_val - 1 };
		}

		NIHILUS_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		NIHILUS_INLINE constexpr uint64_t size() const noexcept {
			return size_val;
		}

		NIHILUS_INLINE constexpr uint64_t max_size() const noexcept {
			return allocator_traits::max_size(*this);
		}

		NIHILUS_INLINE constexpr void clear() noexcept {
			resize(0);
		}

		NIHILUS_INLINE void resize(size_type size_new) noexcept {
			reserve(size_new);
			if (size_new > size_val) {
				std::uninitialized_value_construct_n(data_val + size_val, size_new - size_val);
				size_val = size_new;
			} else if (size_new < size_val) {
				std::destroy(data_val + size_new, data_val + size_val);
				size_val = size_new;
			}
			return;
		}

		NIHILUS_INLINE void reserve(size_type size_new) noexcept {
			if (size_new > capacity_val) {
				size_type old_capacity = capacity_val;
				pointer old_data	   = data_val;
				data_val			   = allocator_traits::allocate(*this, size_new);
				capacity_val		   = size_new;
				if constexpr (std::is_move_constructible_v<value_type>) {
					std::uninitialized_move_n(old_data, size_val, data_val);
				} else if constexpr (std::is_copy_constructible_v<value_type>) {
					std::copy_n(old_data, size_val, data_val);
				}
				allocator_traits::deallocate(*this, old_data, old_capacity);
			}
			return;
		}

		template<integral_or_enum_types index_type> NIHILUS_INLINE constexpr reference at(index_type position) {
			if (size_val <= position) {
				throw std::runtime_error{ "invalid aligned_vector<value_type> subscript" };
			}

			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum_types index_type> NIHILUS_INLINE constexpr const_reference at(index_type position) const {
			if (size_val <= position) {
				throw std::runtime_error{ "invalid aligned_vector<value_type> subscript" };
			}

			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum_types index_type> NIHILUS_INLINE constexpr reference operator[](index_type position) noexcept {
			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum_types index_type> NIHILUS_INLINE constexpr const_reference operator[](index_type position) const noexcept {
			return data_val[static_cast<uint64_t>(position)];
		}

		template<uint64_t index> NIHILUS_INLINE constexpr uint64_t& operator[](tag<index> index_new) {
			return data_val[index_new];
		}

		template<uint64_t index> NIHILUS_INLINE constexpr uint64_t operator[](tag<index> index_new) const {
			return data_val[index_new];
		}

		NIHILUS_INLINE constexpr reference front() noexcept {
			return data_val[0];
		}

		NIHILUS_INLINE constexpr const_reference front() const noexcept {
			return data_val[0];
		}

		NIHILUS_INLINE constexpr reference back() noexcept {
			return data_val[size_val - 1];
		}

		NIHILUS_INLINE constexpr const_reference back() const noexcept {
			return data_val[size_val - 1];
		}

		NIHILUS_INLINE constexpr value_type* data() noexcept {
			return data_val;
		}

		NIHILUS_INLINE constexpr const value_type* data() const noexcept {
			return data_val;
		}

		NIHILUS_INLINE constexpr bool empty() const noexcept {
			return size_val == 0;
		}

		NIHILUS_INLINE constexpr friend bool operator==(const aligned_vector& lhs, const aligned_vector& rhs) {
			if (lhs.size_val != rhs.size_val)
				return false;
			for (uint64_t x = 0; x < lhs.size_val; ++x) {
				if (lhs[x] != rhs[x]) {
					return false;
				}
			}
			return true;
		}

		NIHILUS_INLINE ~aligned_vector() {
			clear();
			if (capacity_val && data_val) {
				allocator_traits::deallocate(*this, data_val, capacity_val);
			}
		}

	  protected:
		alignas(64) size_type capacity_val{};
		alignas(64) size_type size_val{};
		alignas(64) pointer data_val{};
	};

}
