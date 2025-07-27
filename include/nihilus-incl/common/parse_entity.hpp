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

#include <nihilus-incl/common/type_traits.hpp>
#include <nihilus-incl/common/compare.hpp>
#include <nihilus-incl/common/debugging_io.hpp>
#include <nihilus-incl/common/model_traits.hpp>

namespace nihilus {

	template<typename member_type_new> struct decompose_types {
		using member_type = member_type_new;
	};

	template<typename class_type_new, typename member_type_new> struct decompose_types<member_type_new class_type_new::*> {
		using member_type = member_type_new;
		using class_type  = class_type_new;
	};

	template<typename value_type> using remove_class_pointer_t = typename decompose_types<value_type>::member_type;

	template<typename value_type> using remove_member_pointer_t = typename decompose_types<value_type>::class_type;

	template<typename value_type_new, auto element> NIHILUS_INLINE decltype(auto) get_member(value_type_new& value) {
		using value_type = std::remove_cvref_t<decltype(element)>;
		if constexpr (std::is_member_object_pointer_v<value_type>) {
			return value.*element;
		} else if constexpr (std::is_member_function_pointer_v<value_type>) {
			return element;
		} else if constexpr (std::is_pointer_v<value_type>) {
			return *element;
		} else {
			return element;
		}
	}

	template<typename value_type> struct parse_core {};

	template<auto member_ptr_new, string_literal name_new> struct parse_entity {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type member_ptr{ member_ptr_new };
		inline static constexpr string_literal name{ name_new };
	};

	template<auto member_ptr_new, string_literal name_new>
		requires(std::is_member_pointer_v<decltype(member_ptr_new)>)
	struct parse_entity<member_ptr_new, name_new> {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		using class_type  = remove_member_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type class_type::* member_ptr{ member_ptr_new };
		inline static constexpr string_literal name{ name_new };
	};

	template<typename value_type>
	concept parse_entity_types = requires {
		std::remove_cvref_t<value_type>::name;
		std::remove_cvref_t<value_type>::member_ptr;
		typename std::remove_cvref_t<value_type>::member_type;
	} && std::is_member_pointer_v<decltype(std::remove_cvref_t<value_type>::member_ptr)>;

	template<auto... values, size_t... indices> inline static constexpr auto create_value_impl(std::index_sequence<indices...>) {
		static_assert((parse_entity_types<decltype(values)> + ...), "Sorry, but they must all be parse_entities passed to this function!");
		return make_tuple(values...);
	}

	template<auto member_ptr, string_literal name_new> inline static constexpr auto make_parse_entity() {
		return parse_entity<member_ptr, name_new>{};
	}

	template<auto... values> inline static constexpr auto create_value() noexcept {
		return create_value_impl<values...>(std::make_index_sequence<sizeof...(values)>{});
	}

	template<typename value_type> using core_tuple_type				 = decltype(parse_core<std::remove_cvref_t<value_type>>::parse_value);
	template<typename value_type> constexpr uint64_t core_tuple_size = tuple_size_v<core_tuple_type<value_type>>;

	template<typename value_type, uint64_t current_index = 0> NIHILUS_INLINE static constexpr uint64_t find_matching_element(const char* start, uint64_t length) noexcept {
		constexpr auto tuple_size = core_tuple_size<value_type>;

		if constexpr (current_index >= tuple_size) {
			return std::numeric_limits<uint64_t>::max();
		} else {
			constexpr auto element = get<current_index>(parse_core<value_type>::parse_value);

			constexpr auto element_name = element.name;
			if (length == element_name.size()) {
				if (string_literal_comparison<element_name>(start)) {
					return current_index;
				}
			}

			return find_matching_element<value_type, current_index + 1>(start, length);
		}
	}

	template<typename value_type, typename iterator_newer> struct hash_map {
		NIHILUS_INLINE static uint64_t find_index(iterator_newer iter, iterator_newer end) noexcept {
			return find_matching_element<value_type>(iter, static_cast<uint64_t>(end - iter));
		}
	};

}
