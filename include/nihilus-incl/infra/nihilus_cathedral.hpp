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
		using enum_type = typename get_first_type_t<bases...>::enum_type;

		template<template<typename, typename> typename mixin_type, typename... arg_types> NIHILUS_HOST constexpr void impl(arg_types&&... args) noexcept {
			(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
		}

		template<template<typename, typename, auto...> typename mixin_type, auto... values, typename... arg_types>
		NIHILUS_HOST constexpr void impl_thread(arg_types&&... args) noexcept {
			(impl_internal_filtered_thread<mixin_type, bases, values...>(detail::forward<arg_types>(args)...), ...);
		}

		template<integral_or_enum_types auto enum_value> NIHILUS_HOST decltype(auto) get_core() const noexcept {
			return (*this)[tag<static_cast<enum_type>(enum_value)>()];
		}

		template<integral_or_enum_types auto enum_value> NIHILUS_HOST decltype(auto) get_core() noexcept {
			return (*this)[tag<static_cast<enum_type>(enum_value)>()];
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

	template<integral_or_enum_types auto index, typename nihilus_cathedral_type> using get_cathedral_type =
		detail::remove_cvref_t<decltype(std::declval<nihilus_cathedral_type>().template get_core<index>())>;

	template<typename nihilus_cathedral_type> using last_cathedral_type =
		detail::remove_cvref_t<decltype(std::declval<nihilus_cathedral_type>().template get_core<nihilus_cathedral_type::size - 1>())>;

	template<typename config_type, typename enum_type, template<typename, typename> typename aggregator_type, template<enum_type, typename...> typename base_type,
		typename... value_type>
	struct get_nihilus_cathedral_array;

	template<typename config_type, typename enum_type, template<typename, typename> typename aggregator_type, template<enum_type, typename...> typename base_type, size_t... index>
	struct get_nihilus_cathedral_array<config_type, enum_type, aggregator_type, base_type, std::index_sequence<index...>> {
		using type = nihilus_cathedral<config_type, base_type<static_cast<enum_type>(aggregator_type<config_type, enum_type>::values[index]), config_type>...>;
	};

	template<typename config_type, typename enum_type, template<typename, typename> typename aggregator_type, template<enum_type, typename...> typename base_type>
	using get_nihilus_cathedral_array_t = detail::remove_cvref_t<typename get_nihilus_cathedral_array<config_type, enum_type, aggregator_type, base_type,
		std::make_index_sequence<static_cast<uint64_t>(aggregator_type<config_type, enum_type>::values.size())>>::type>;

	template<typename config_type> static constexpr memory_plan nihilus_cathedral_memory_plan{ []() {
		return get_memory_plan<config_type>();
	}() };

	template<typename config_type, uint64_t current_index> consteval memory_plan get_memory_plan(memory_plan values) {
		constexpr uint64_t max_index{ static_cast<uint64_t>(core_types::count) };
		using cathedral_type		 = get_nihilus_cathedral_array_t<config_type, tensor_types, data_holder_aggregator, indexed_data_holder>;
		using current_cathedral_type = get_cathedral_type<current_index, cathedral_type>;
		if constexpr (current_index == base_index<config_type::device_type>) {
			values.currently_allocated_bytes = 0;
			values.peak_allocated_bytes		 = 0;
		}
		if constexpr (current_index < max_index) {
			values.footprints[current_index].offset	   = values.currently_allocated_bytes;
			values.footprints[current_index].core_type = static_cast<core_types>(current_index);
			values.footprints[current_index].depth				  = current_index;
			values.footprints[current_index].is_active = true;
			values.footprints[current_index].total_required_bytes = current_cathedral_type::total_required_bytes;
			values.currently_allocated_bytes += current_cathedral_type::total_required_bytes;
			if (values.currently_allocated_bytes > values.peak_allocated_bytes) {
				values.peak_allocated_bytes = values.currently_allocated_bytes;
			}
			constexpr uint64_t cur_depth = current_index;
			if constexpr (cur_depth >= 2 && !weight_types<current_cathedral_type> && !global_input_types<current_cathedral_type>) {
				for (int64_t x = 0; x < static_cast<int64_t>(current_index); ++x) {
					if (is_valid_free_type(values.footprints[static_cast<uint64_t>(x)], cur_depth)) {
						values.footprints[static_cast<uint64_t>(x)].is_active = false;
						values.currently_allocated_bytes -= values.footprints[static_cast<uint64_t>(x)].total_required_bytes;
					}
				}
			}
			return get_memory_plan<config_type, current_index + 1>(values);
		} else {
			return values;
		}
	}
}
