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

#include <nihilus/common/iterator.hpp>
#include <nihilus/common/config.hpp>
#include <algorithm>
#include <stdexcept>

namespace nihilus {

	template<typename value_type>
	concept is_integral_constant = requires() {
		typename std::remove_cvref_t<value_type>::value_type;
		{ std::remove_cvref_t<value_type>::value } -> std::same_as<typename std::remove_cvref_t<value_type>::value_type>;
	};

	template<typename value_type01, typename value_type02>
	concept is_indexable = std::is_same_v<value_type01, value_type02> || std::integral<value_type01> || is_integral_constant<value_type01> || is_integral_constant<value_type02>;

	enum class array_static_assert_errors {
		invalid_index_type,
	};

	template<uint64_t index> using tag = std::integral_constant<uint64_t, index>;

	template<typename value_type_new, auto size_new> struct array {
	  public:
		static_assert(integral_or_enum<decltype(size_new)>, "Sorry, but the size val passed to array must be integral or enum!");
		static constexpr uint64_t size_val{ static_cast<uint64_t>(size_new) };
		using value_type			 = value_type_new;
		using uint64_type			 = decltype(size_new);
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = array_iterator<value_type, static_cast<uint64_t>(size_new)>;
		using const_iterator		 = const array_iterator<value_type, static_cast<uint64_t>(size_new)>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		NIHILUS_FORCE_INLINE constexpr array() = default;

		NIHILUS_FORCE_INLINE constexpr array(std::initializer_list<value_type> values) {
			for (uint64_t x = 0; x < values.size(); ++x) {
				data_val[x] = values.begin()[x];
			}
		};

		NIHILUS_FORCE_INLINE constexpr array(const array& other)
			requires(std::is_copy_constructible_v<value_type>)
		{
			for (uint64_t i = 0; i < size_val; ++i) {
				data_val[i] = other.data_val[i];
			}
		}

		NIHILUS_FORCE_INLINE array(const array& other)
			requires(!std::is_copy_constructible_v<value_type>)
		= delete;

		NIHILUS_FORCE_INLINE constexpr array& operator=(const array& other)
			requires(std::is_copy_assignable_v<value_type>)
		{
			if (this != &other) {
				for (uint64_t i = 0; i < size_val; ++i) {
					data_val[i] = other.data_val[i];
				}
			}
			return *this;
		}

		NIHILUS_FORCE_INLINE array& operator=(const array& other)
			requires(!std::is_copy_assignable_v<value_type>)
		= delete;

		NIHILUS_FORCE_INLINE constexpr array(array&& other) noexcept
			requires(std::is_move_constructible_v<value_type>)
		{
			for (uint64_t i = 0; i < size_val; ++i) {
				data_val[i] = std::move(other.data_val[i]);
			}
		}

		NIHILUS_FORCE_INLINE constexpr array& operator=(array&& other) noexcept
			requires(std::is_move_assignable_v<value_type>)
		{
			if (this != &other) {
				for (uint64_t i = 0; i < size_val; ++i) {
					data_val[i] = std::move(other.data_val[i]);
				}
			}
			return *this;
		}

		template<typename... Args> NIHILUS_FORCE_INLINE constexpr array(Args&&... args)
			requires(sizeof...(Args) == size_val && (std::is_constructible_v<value_type, Args> && ...) && std::is_copy_constructible_v<value_type>)
			: data_val{ static_cast<value_type>(std::forward<Args>(args))... } {
		}

		NIHILUS_FORCE_INLINE constexpr void fill(const value_type& _Value) {
			std::fill_n(data_val, size_new, _Value);
		}

		NIHILUS_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator(data_val);
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator(data_val);
		}

		NIHILUS_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator(data_val + size_val);
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator(data_val + size_val);
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		NIHILUS_FORCE_INLINE constexpr uint64_type size() const noexcept {
			return size_new;
		}

		NIHILUS_FORCE_INLINE constexpr uint64_type max_size() const noexcept {
			return size_new;
		}

		NIHILUS_FORCE_INLINE constexpr bool empty() const noexcept {
			return false;
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr reference at(index_type position) {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			if (size_new <= position) {
				throw std::runtime_error{ "invalid array<T, N> subscript" };
			}

			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr const_reference at(index_type position) const {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			if (size_new <= position) {
				throw std::runtime_error{ "invalid array<T, N> subscript" };
			}

			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr reference operator[](index_type position) noexcept {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			return data_val[static_cast<uint64_t>(position)];
		}

		template<integral_or_enum index_type> NIHILUS_FORCE_INLINE constexpr const_reference operator[](index_type position) const noexcept {
			static_assert(static_assert_printer<is_indexable<index_type, decltype(size_new)>, array_static_assert_errors::invalid_index_type, index_type>::impl,
				"Sorry, but please index into this array using the correct enum type!");
			return data_val[static_cast<uint64_t>(position)];
		}

		template<uint64_t index> NIHILUS_FORCE_INLINE constexpr uint64_t& operator[](tag<index> index_new) {
			return data_val[index_new];
		}

		template<uint64_t index> NIHILUS_FORCE_INLINE constexpr uint64_t operator[](tag<index> index_new) const {
			return data_val[index_new];
		}

		NIHILUS_FORCE_INLINE constexpr reference front() noexcept {
			return data_val[0];
		}

		NIHILUS_FORCE_INLINE constexpr const_reference front() const noexcept {
			return data_val[0];
		}

		NIHILUS_FORCE_INLINE constexpr reference back() noexcept {
			return data_val[size_new - 1];
		}

		NIHILUS_FORCE_INLINE constexpr const_reference back() const noexcept {
			return data_val[size_new - 1];
		}

		NIHILUS_FORCE_INLINE constexpr value_type* data() noexcept {
			return data_val;
		}

		NIHILUS_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return data_val;
		}

		NIHILUS_FORCE_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
			for (uint64_t x = 0; x < size_val; ++x) {
				if (lhs[x] != rhs[x]) {
					return false;
				}
			}
			return true;
		}

		value_type data_val[size_val]{};
	};

	template<typename T, typename... U> array(T, U...) -> array<T, 1 + sizeof...(U)>;

	struct empty_array_element {};

	template<class value_type_new> class array<value_type_new, 0> {
	  public:
		using value_type			 = value_type_new;
		using uint64_type			 = uint64_t;
		using difference_type		 = ptrdiff_t;
		using pointer				 = value_type*;
		using const_pointer			 = const value_type*;
		using reference				 = value_type&;
		using const_reference		 = const value_type&;
		using iterator				 = array_iterator<value_type, 0>;
		using const_iterator		 = const array_iterator<value_type, 0>;
		using reverse_iterator		 = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;

		NIHILUS_FORCE_INLINE constexpr void fill(const value_type&) {
		}

		NIHILUS_FORCE_INLINE constexpr void swap(array&) noexcept {
		}

		NIHILUS_FORCE_INLINE constexpr iterator begin() noexcept {
			return iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator begin() const noexcept {
			return const_iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr iterator end() noexcept {
			return iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator end() const noexcept {
			return const_iterator{};
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rbegin() noexcept {
			return reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rbegin() const noexcept {
			return const_reverse_iterator(end());
		}

		NIHILUS_FORCE_INLINE constexpr reverse_iterator rend() noexcept {
			return reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator rend() const noexcept {
			return const_reverse_iterator(begin());
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cbegin() const noexcept {
			return begin();
		}

		NIHILUS_FORCE_INLINE constexpr const_iterator cend() const noexcept {
			return end();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crbegin() const noexcept {
			return rbegin();
		}

		NIHILUS_FORCE_INLINE constexpr const_reverse_iterator crend() const noexcept {
			return rend();
		}

		NIHILUS_FORCE_INLINE constexpr uint64_type size() const noexcept {
			return 0;
		}

		NIHILUS_FORCE_INLINE constexpr uint64_type max_size() const noexcept {
			return 0;
		}

		NIHILUS_FORCE_INLINE constexpr bool empty() const noexcept {
			return true;
		}

		NIHILUS_FORCE_INLINE constexpr reference at(uint64_type) {
			throw std::runtime_error{ "invalid array<T, N> subscript" };
		}

		NIHILUS_FORCE_INLINE constexpr const_reference at(uint64_type) const {
			throw std::runtime_error{ "invalid array<T, N> subscript" };
		}

		NIHILUS_FORCE_INLINE constexpr reference operator[](uint64_type) noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr const_reference operator[](uint64_type) const noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr reference front() noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr const_reference front() const noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr reference back() noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr const_reference back() const noexcept {
			return *data();
		}

		NIHILUS_FORCE_INLINE constexpr value_type* data() noexcept {
			return nullptr;
		}

		NIHILUS_FORCE_INLINE constexpr const value_type* data() const noexcept {
			return nullptr;
		}

		NIHILUS_FORCE_INLINE constexpr friend bool operator==(const array& lhs, const array& rhs) {
			( void )lhs;
			( void )rhs;
			return true;
		}

	  protected:
		std::conditional_t<std::disjunction_v<std::is_default_constructible<value_type>, std::is_default_constructible<value_type>>, value_type, empty_array_element> data_val[1]{};
	};
}
