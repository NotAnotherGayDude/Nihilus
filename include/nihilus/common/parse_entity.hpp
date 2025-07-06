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

#include <nihilus/common/type_traits.hpp>
#include <nihilus/common/compare.hpp>
#include <nihilus/common/debugging_io.hpp>
#include <nihilus/common/model_traits.hpp>

namespace nihilus {

	template<typename member_type> struct remove_member_pointer {
		using type = member_type;
	};

	template<typename class_type, typename member_type> struct remove_member_pointer<member_type class_type::*> {
		using type = class_type;
	};

	template<typename value_type> using remove_member_pointer_t = typename remove_member_pointer<value_type>::type;

	template<typename member_type> struct remove_class_pointer {
		using type = member_type;
	};

	template<typename class_type, typename member_type> struct remove_class_pointer<member_type class_type::*> {
		using type = member_type;
	};

	template<typename value_type> using remove_class_pointer_t = typename remove_class_pointer<value_type>::type;

	template<typename value_type, auto element> NIHILUS_FORCE_INLINE decltype(auto) get_member(value_type& value) {
		using V = std::decay_t<decltype(element)>;
		if constexpr (std::is_member_object_pointer_v<V>) {
			return value.*element;
		} else if constexpr (std::is_member_function_pointer_v<V>) {
			return element;
		} else if constexpr (std::is_pointer_v<V>) {
			return *element;
		} else {
			return element;
		}
	}

	template<typename value_type> struct parse_core {};

	template<typename value_type> struct base_parse_entity {
		using class_type = value_type;
		inline static constexpr uint64_t index{ 0 };
	};

	template<auto member_ptr_new, string_literal name_new> struct parse_entity_temp {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type member_ptr{ member_ptr_new };
		inline static constexpr string_literal name{ name_new };
	};

	template<auto member_ptr_new, string_literal name_new>
		requires(std::is_member_pointer_v<decltype(member_ptr_new)>)
	struct parse_entity_temp<member_ptr_new, name_new> {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		using class_type  = remove_member_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type class_type::* member_ptr{ member_ptr_new };
		inline static constexpr string_literal name{ name_new };
	};

	template<auto member_ptr_new, string_literal name_new, uint64_t index_new, uint64_t max_index> struct parse_entity {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type member_ptr{ member_ptr_new };
		inline static constexpr bool isItLast{ index_new == max_index - 1 };
		inline static constexpr string_literal name{ name_new };
		inline static constexpr uint64_t index{ index_new };
	};

	template<auto member_ptr_new, string_literal name_new, uint64_t index_new, uint64_t max_index>
		requires(std::is_member_pointer_v<decltype(member_ptr_new)>)
	struct parse_entity<member_ptr_new, name_new, index_new, max_index> {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		using class_type  = remove_member_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type class_type::* member_ptr{ member_ptr_new };
		inline static constexpr bool isItLast{ index_new == max_index - 1 };
		inline static constexpr string_literal name{ name_new };
		inline static constexpr uint64_t index{ index_new };
	};

	template<uint64_t max_index, uint64_t index, auto value> inline static constexpr auto make_parse_entity_auto() noexcept {
		constexpr parse_entity<value.member_ptr, value.name, index, max_index> parseEntity{};
		return parseEntity;
	}

	template<auto... values, size_t... indices> inline static constexpr auto create_value_impl(std::index_sequence<indices...>) {
		return make_tuple(make_parse_entity_auto<sizeof...(values), indices, values>()...);
	}

	template<auto member_ptr, string_literal name_new> inline static constexpr auto make_parse_entity() {
		return parse_entity_temp<member_ptr, name_new>{};
	}

	template<auto... values> inline static constexpr auto create_value() noexcept {
		return create_value_impl<values...>(std::make_index_sequence<sizeof...(values)>{});
	}

	template<typename value_type> using core_tuple_type				 = decltype(parse_core<std::remove_cvref_t<value_type>>::parse_value);
	template<typename value_type> constexpr uint64_t core_tuple_size = tuple_size_v<core_tuple_type<value_type>>;

	template<typename value_type, uint64_t current_index = 0> NIHILUS_FORCE_INLINE static constexpr uint64_t find_matching_element(const char* start, uint64_t length) noexcept {
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
		NIHILUS_FORCE_INLINE static uint64_t findIndex(iterator_newer iter, iterator_newer end) noexcept {
			return find_matching_element<value_type>(iter, end - iter);
		}
	};

}