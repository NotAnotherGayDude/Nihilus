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

	template<const model_config& config, typename... bases> struct core_bases : public bases... {
		using bases::operator[]...;
		NIHILUS_HOST core_bases()				 = default;
		core_bases& operator=(core_bases&&)		 = delete;
		core_bases(core_bases&&)				 = delete;
		core_bases& operator=(const core_bases&) = delete;
		core_bases(const core_bases&)			 = delete;

		template<template<const model_config&, typename> typename mixin_type, typename... arg_types> NIHILUS_HOST constexpr void impl(arg_types&&... args) noexcept {
			(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
		}

		template<template<const model_config&, typename, auto...> typename mixin_type, auto... values, typename... arg_types>
		NIHILUS_HOST constexpr void impl_thread(arg_types&&... args) noexcept {
			(impl_internal_filtered_thread<mixin_type, bases, values...>(detail::forward<arg_types>(args)...), ...);
		}

		template<enum_types core_types, core_types enum_value> NIHILUS_HOST decltype(auto) get_core() noexcept {
			return (*this)[tag<enum_value>()];
		}

	  protected:
		template<template<const model_config&, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_HOST constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) noexcept {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<typename base_type::derived_type*>(this), detail::forward<arg_types>(args)...);
			}
		}

		template<template<const model_config&, typename, auto...> typename mixin_type, typename base_type, auto... values, typename... arg_types>
		NIHILUS_HOST constexpr void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) noexcept {
			if constexpr (mixin_type<config, base_type, values...>::filter()) {
				mixin_type<config, base_type, values...>::impl(*static_cast<typename base_type::derived_type*>(this), detail::forward<arg_types>(args)...);
			}
		}
	};

	template<const model_config& config, typename... value_type> struct get_core_bases {
		using type = core_bases<config, value_type...>;
	};

	template<const model_config& config, size_t... index> struct get_core_bases<config, std::index_sequence<index...>> {
		using type = core_bases<config, core_traits<config, static_cast<core_types>(index)>...>;
	};

	template<const model_config& config, typename... value_type> using get_core_base_t = typename get_core_bases<config, value_type...>::type;

	template<const model_config& config> using get_core_bases_t = typename get_core_bases<config, std::make_index_sequence<static_cast<uint64_t>(core_types::count)>>::type;

	template<const model_config& config> static constexpr memory_plan core_bases_memory_plan{ []() {
		return get_memory_plan<config>();
	}() };
}
