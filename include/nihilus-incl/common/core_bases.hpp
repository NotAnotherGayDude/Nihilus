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

#include <nihilus-incl/common/behavioral_axes.hpp>

namespace nihilus {

	template<model_config config, typename... bases> struct core_bases : public bases... {
		using bases::operator[]...;
		using bases::decl_elem...;
		NIHILUS_INLINE core_bases()				 = default;
		core_bases& operator=(core_bases&&)		 = delete;
		core_bases(core_bases&&)				 = delete;
		core_bases& operator=(const core_bases&) = delete;
		core_bases(const core_bases&)			 = delete;

		template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE static constexpr void impl_static(arg_types&&... args) {
			(impl_internal_filtered_static<mixin_type, bases>(args...), ...);
		}

		template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) const {
			(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
		}

		template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) {
			(impl_internal_filtered<mixin_type, bases>(args...), ...);
		}

		template<template<model_config, typename, processing_phases> typename mixin_type, processing_phases processing_phase, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_thread(arg_types&&... args) {
			(impl_internal_filtered_thread<mixin_type, processing_phase, bases>(args...), ...);
		}

		template<enum_types enum_type, enum_type enum_value> constexpr decltype(auto) get_core() const {
			return (*this)[tag<enum_value>()];
		}

		template<enum_types enum_type, enum_type enum_value> constexpr decltype(auto) get_core() {
			return (*this)[tag<enum_value>()];
		}

	  protected:
		template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_INLINE static constexpr void impl_internal_filtered_static([[maybe_unused]] arg_types&&... args) {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(detail::forward<arg_types>(args)...);
			}
		}

		template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) const {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<const base_type*>(this), detail::forward<arg_types>(args)...);
			}
		}

		template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), detail::forward<arg_types>(args)...);
			}
		}

		template<template<model_config, typename, processing_phases> typename mixin_type, processing_phases processing_phase, typename base_type, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) {
			if constexpr (mixin_type<config, base_type, processing_phase>::filter()) {
				mixin_type<config, base_type, processing_phase>::impl(*static_cast<base_type*>(this), detail::forward<arg_types>(args)...);
			}
		}
	};

	template<model_config config, typename enum_type, typename index_sequence> struct get_core_bases;

	template<model_config config, typename enum_type, size_t... index> struct get_core_bases<config, enum_type, std::index_sequence<index...>> {
		using type = core_bases<config, core_traits<config, static_cast<enum_type>(index)>...>;
	};

	template<model_config config, typename enum_type> using get_core_bases_t =
		typename get_core_bases<config, enum_type, std::make_index_sequence<static_cast<uint64_t>(enum_type::count)>>::type;

	template<uint64_t index, typename value_type> struct core_elem {
		static value_type decl_elem(tag<index>);
		using type = value_type;

		NIHILUS_TUPLET_NO_UNIQUE_ADDRESS value_type value;

		NIHILUS_INLINE constexpr decltype(auto) operator[](tag<index>) & {
			return (value);
		}

		NIHILUS_INLINE constexpr decltype(auto) operator[](tag<index>) const& {
			return (value);
		}

		NIHILUS_INLINE constexpr decltype(auto) operator[](tag<index>) && {
			return (move(*this).value);
		}
	};

	template<model_config config, typename enum_type, typename index_sequence, typename... value_type> struct get_core_base;

	template<model_config config, typename enum_type, size_t... index, typename... value_type>
	struct get_core_base<config, enum_type, std::index_sequence<index...>, value_type...> {
		using type = core_bases<config, core_elem<index, value_type>...>;
	};

	template<model_config config, typename enum_type, typename... value_type> using core_base_t =
		typename get_core_base<config, enum_type, std::make_index_sequence<static_cast<uint64_t>(enum_type::count)>, value_type...>::type;

	template<model_config config, typename enum_type> struct core_bases_traits {
		static constexpr memory_plan_new total_required_bytes{ []() {
			return get_memory_plan<config>();
		}() };
	};
}
