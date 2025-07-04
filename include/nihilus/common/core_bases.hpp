/*
Copyright (c) 2025 RealTimeChris (Chris model_traits_type.)

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
RealTimeChris (Chris model_traits_type.)
2025
*/

#pragma once

#include <nihilus/common/behavioral_axes.hpp>
#include <latch>

namespace nihilus {

	template<nihilus::model_config config, typename... bases> struct core_bases : public bases... {
		NIHILUS_FORCE_INLINE constexpr core_bases(){};
		template<template<nihilus::model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE constexpr void impl(arg_types&&... args) const {
			(impl_internal_filtered<mixin_type, bases>(std::forward<arg_types>(args)...), ...);
		}

		template<template<nihilus::model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_FORCE_INLINE constexpr void impl(arg_types&&... args) {
			(impl_internal_filtered<mixin_type, bases>(args...), ...);
		}

	  protected:
		template<template<nihilus::model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_FORCE_INLINE constexpr void impl_internal_filtered(arg_types&&... args) const {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<const base_type*>(this), std::forward<arg_types>(args)...);
			}
		}

	  protected:
		template<template<nihilus::model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_FORCE_INLINE constexpr void impl_internal_filtered(arg_types&&... args) {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), std::forward<arg_types>(args)...);
			}
		}
	};

	template<nihilus::model_config config, typename index_sequence> struct get_core_bases;

	template<nihilus::model_config config, size_t... index> struct get_core_bases<config, std::index_sequence<index...>> {
		using type = core_bases<config, nihilus::core_traits<config, static_cast<typename nihilus::model_traits_type<config>::op_type_type>(index)>...>;
	};

	template<nihilus::model_config config> using get_core_bases_t = typename get_core_bases<config, std::make_index_sequence<static_cast<uint64_t>(op_types::count)>>::type;

	template<model_config config_new> static constexpr get_core_bases_t<config_new> core_bases_val{};

	template<model_config config_new> struct core_bases_traits {
		static constexpr uint64_t total_required_bytes{ []() {
			uint64_t return_value{};
			core_bases_val<config_new>.template impl<memory_calculator>(return_value);
			return return_value;
		}() };
		static constexpr uint64_t max_depth{ []() {
			uint64_t return_value{};
			core_bases_val<config_new>.template impl<max_depth_calculator>(return_value);
			return return_value;
		}() };
		static constexpr array<uint64_t, max_depth+1> ops_per_depth{ []() {
			array<uint64_t, max_depth + 1> return_value{};
			core_bases_val<config_new>.template impl<ops_per_depth_calculator>(return_value);
			return return_value;
		}() };
	};
}