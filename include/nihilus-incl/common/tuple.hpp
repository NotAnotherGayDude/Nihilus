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
/// Taken mostly from: https://github.com/codeinred/tuplet
/// https://github.com/RealTimeChris/nihilus
/// Feb 3, 2023
#pragma once

#include <nihilus-incl/common/common.hpp>
#include <type_traits>
#include <cstddef>
#include <utility>

#if defined(NIHILUS_TUPLET_NO_UNIQUE_ADDRESS) && !NIHILUS_TUPLET_NO_UNIQUE_ADDRESS
	#define NIHILUS_TUPLET_NO_UNIQUE_ADDRESS
#else
	#if defined(_MSVC_LANG)
		#if _MSVC_LANG >= 202002L && (!defined __clang__)

			#define NIHILUS_TUPLET_HAS_NO_UNIQUE_ADDRESS 1
			#define NIHILUS_TUPLET_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]

		#elif _MSC_VER
			#define NIHILUS_TUPLET_HAS_NO_UNIQUE_ADDRESS 0
			#define NIHILUS_TUPLET_NO_UNIQUE_ADDRESS
		#endif
	#else
		#if __cplusplus > 201703L && (__has_cpp_attribute(no_unique_address))
			#define NIHILUS_TUPLET_HAS_NO_UNIQUE_ADDRESS 1
			#define NIHILUS_TUPLET_NO_UNIQUE_ADDRESS [[no_unique_address]]

		#else
			#define NIHILUS_TUPLET_HAS_NO_UNIQUE_ADDRESS 0
			#define NIHILUS_TUPLET_NO_UNIQUE_ADDRESS
		#endif
	#endif

#endif

namespace nihilus {

	template<typename tup, typename B> struct forward_as {
		using type = B&&;
	};

	template<typename tup, typename B> struct forward_as<tup&, B> {
		using type = B&;
	};

	template<typename tup, typename B> struct forward_as<tup const&, B> {
		using type = B const&;
	};

	template<typename tup, typename B> using forward_as_t = typename forward_as<tup, B>::type;

	template<typename value_type> using identity_t = value_type;

	template<typename first, typename...> using first_t = first;

	template<typename value_type> using type_t = typename value_type::type;

	template<typename tup> using base_list_t = typename std::decay_t<tup>::base_list;

	template<typename tuple>
	concept base_list_tuple = requires { typename std::decay_t<tuple>::base_list; };

	template<typename value_type>
	concept stateless = std::is_empty_v<std::decay_t<value_type>>;

	template<typename... value_type> struct type_list {};

	template<typename... Ls, typename... Rs> constexpr auto operator+(type_list<Ls...>, type_list<Rs...>) {
		return type_list<Ls..., Rs...>{};
	}

	template<typename value_type>
	concept indexable = stateless<value_type> || requires(value_type t) { t[tag<0>()]; };

	template<class... bases> struct type_map : bases... {
		using base_list = type_list<bases...>;
		using bases::operator[]...;
		using bases::decl_elem...;
	};

	template<uint64_t index, typename value_type> struct tuple_elem {
		static value_type decl_elem(tag<index>);
		using type = value_type;

		NIHILUS_TUPLET_NO_UNIQUE_ADDRESS value_type value;

		constexpr decltype(auto) operator[](tag<index>) & {
			return (value);
		}
		constexpr decltype(auto) operator[](tag<index>) const& {
			return (value);
		}
		constexpr decltype(auto) operator[](tag<index>) && {
			return (move(*this).value);
		}
	};

	template<typename index_sequence, typename... value_type> struct get_tuple_base;

	template<size_t... index, typename... value_type> struct get_tuple_base<std::index_sequence<index...>, value_type...> {
		using type = type_map<tuple_elem<index, value_type>...>;
	};

	template<typename... value_type> using tuple_base_t = typename get_tuple_base<std::make_index_sequence<sizeof...(value_type)>, value_type...>::type;

	template<typename... value_type> struct tuple : tuple_base_t<value_type...> {
		static constexpr uint64_t index = sizeof...(value_type);
		using super						= tuple_base_t<value_type...>;
		using super::operator[];
		using super::decl_elem;
	};

	template<> struct tuple<> : tuple_base_t<> {
		static constexpr uint64_t index = 0;
		using super						= tuple_base_t<>;
		using base_list					= type_list<>;
		using element_list				= type_list<>;
	};

	template<typename... types> tuple(types&&...) -> tuple<std::remove_cvref_t<types>...>;

	template<integral_or_enum_types auto index, indexable tup> static constexpr decltype(auto) get(tup&& tupleVal) {
		return static_cast<tup&&>(tupleVal)[tag<index>()];
	}

	template<typename... types> static constexpr auto make_tuple(types&&... args) {
		return tuple<std::remove_cvref_t<types>...>{ { { static_cast<types&&>(args) }... } };
	}

	template<typename value_type, typename... Q> static constexpr auto repeat_type(type_list<Q...>) {
		return type_list<first_t<value_type, Q>...>{};
	}

	template<typename... outer> static constexpr auto get_outer_bases(type_list<outer...>) {
		return (repeat_type<outer>(base_list_t<type_t<outer>>{}) + ...);
	}

	template<typename... inner> static constexpr auto get_inner_bases(type_list<inner...>) {
		return (base_list_t<type_t<inner>>{} + ...);
	}

	template<typename value_type, typename... outer, typename... inner> static constexpr auto tuple_cat_impl(value_type tupleVal, type_list<outer...>, type_list<inner...>)
		-> tuple<type_t<inner>...> {
		return { { { static_cast<forward_as_t<type_t<outer>&&, inner>>(tupleVal.identity_t<outer>::value).value }... } };
	}

	template<base_list_tuple... value_type> static constexpr auto tuple_cat(value_type&&... ts) {
		if constexpr (sizeof...(value_type) == 0) {
			return tuple<>{};
		} else {
#if !defined(NIHILUS_TUPLET_CAT_BY_FORWARDING_TUPLE)
	#if defined(__clang__)
		#define NIHILUS_TUPLET_CAT_BY_FORWARDING_TUPLE 0
	#else
		#define NIHILUS_TUPLET_CAT_BY_FORWARDING_TUPLE 1
	#endif
#endif
#if NIHILUS_TUPLET_CAT_BY_FORWARDING_TUPLE
			using big_tuple = tuple<value_type&&...>;
#else
			using big_tuple = tuple<std::decay_t<value_type>...>;
#endif
			using outer_bases	 = base_list_t<big_tuple>;
			constexpr auto outer = get_outer_bases(outer_bases{});
			constexpr auto inner = get_inner_bases(outer_bases{});
			return tuple_cat_impl(big_tuple{ { { static_cast<value_type&&>(ts) }... } }, outer, inner);
		}
	}

	template<typename... value_type> struct tuple_size;

	template<uint64_t index, typename... value_type> struct tuple_element;

	template<uint64_t index, typename... value_type> struct tuple_element<index, tuple<value_type...>> {
		using type = decltype(tuple<std::remove_cvref_t<value_type>...>::decl_elem(tag<index>()));
	};

	template<uint64_t index, typename tuple_type> using tuple_element_t = typename tuple_element<index, tuple_type>::type;

	template<typename... value_type> struct tuple_size<tuple<value_type...>> : std::integral_constant<uint64_t, sizeof...(value_type)> {};

	template<typename... value_type> struct tuple_size<std::tuple<value_type...>> : std::integral_constant<uint64_t, sizeof...(value_type)> {};

	template<typename... value_type> static constexpr auto tuple_size_v = tuple_size<std::remove_cvref_t<value_type>...>::value;
}
