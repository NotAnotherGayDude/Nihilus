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

#include <nihilus-incl/infra/behavioral_axes.hpp>

namespace nihilus {

	template<typename config_type, typename... bases> struct nihilus_cathedral : public bases... {
		using bases::operator[]...;
		NIHILUS_HOST nihilus_cathedral() {
		}
		nihilus_cathedral& operator=(nihilus_cathedral&&)	   = delete;
		nihilus_cathedral(nihilus_cathedral&&)				   = delete;
		nihilus_cathedral& operator=(const nihilus_cathedral&) = delete;
		nihilus_cathedral(const nihilus_cathedral&)			   = delete;

		static constexpr uint64_t size{ sizeof...(bases) };

		template<template<typename, typename> typename mixin_type, typename... arg_types> NIHILUS_HOST constexpr void impl(arg_types&&... args) noexcept {
			(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
		}

		template<template<typename, typename, auto...> typename mixin_type, auto... values, typename... arg_types>
		NIHILUS_HOST constexpr void impl_thread(arg_types&&... args) noexcept {
			(impl_internal_filtered_thread<mixin_type, bases, values...>(detail::forward<arg_types>(args)...), ...);
		}

		template<integral_or_enum_types auto enum_value> NIHILUS_HOST decltype(auto) get_core() noexcept {
			return (*this)[tag<enum_value>()];
		}

	  protected:
		template<template<typename, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_HOST constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) noexcept {
			if constexpr (mixin_type<config_type, base_type>::filter()) {
				mixin_type<config_type, base_type>::impl(*static_cast<typename base_type::derived_type*>(this), detail::forward<arg_types>(args)...);
			}
		}

		template<template<typename, typename, auto...> typename mixin_type, typename base_type, auto... values, typename... arg_types>
		NIHILUS_HOST constexpr void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) noexcept {
			if constexpr (mixin_type<config_type, base_type, values...>::filter()) {
				mixin_type<config_type, base_type, values...>::impl(*static_cast<typename base_type::derived_type*>(this), detail::forward<arg_types>(args)...);
			}
		}
	};

	template<typename config_type, typename enum_type, template<typename, enum_type> typename base_type, typename... value_type> struct get_nihilus_cathedral_enum;

	template<typename config_type, typename enum_type, template<typename, enum_type> typename base_type, size_t... index>
	struct get_nihilus_cathedral_enum<config_type, enum_type, base_type, std::index_sequence<index...>> {
		using type = nihilus_cathedral<config_type, base_type<config_type, static_cast<enum_type>(index)>...>;
	};

	template<typename config_type, typename enum_type, template<typename, enum_type> typename base_type> using get_nihilus_cathedral_enum_t =
		typename get_nihilus_cathedral_enum<config_type, enum_type, base_type, std::make_index_sequence<static_cast<uint64_t>(enum_type::count)>>::type;

	template<typename config_type, typename enum_type, template<typename, enum_type, typename> typename base_type, typename... value_type> struct get_nihilus_cathedral_array;

	template<typename config_type, typename enum_type, template<typename, enum_type, typename> typename base_type, size_t... index>
	struct get_nihilus_cathedral_array<config_type, enum_type, base_type, std::index_sequence<index...>> {
		using type = nihilus_cathedral<config_type, base_type<config_type, static_cast<enum_type>(sub_kernel_aggregator<config_type, enum_type>::values[index]),enum_type>...>;
	};

	template<typename config_type, typename enum_type, template<typename, enum_type, typename> typename base_type> using get_nihilus_cathedral_array_t =
		typename get_nihilus_cathedral_array<config_type, enum_type, base_type,
			std::make_index_sequence<static_cast<uint64_t>(sub_kernel_aggregator<config_type, enum_type>::values.size())>>::type;

	template<typename config_type, typename... value_types> struct get_nihilus_cathedral {
		using type = nihilus_cathedral<config_type, value_types...>;
	};

	template<typename config_type, typename... value_type> using get_nihilus_cathedral_t = typename get_nihilus_cathedral<config_type, value_type...>::type;

	template<typename config_type> static constexpr memory_plan nihilus_cathedral_memory_plan{ []() {
		return get_memory_plan<config_type>();
	}() };
}
