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

#include <nihilus-incl/db/format.hpp>

namespace nihilus {

	template<typename value_type> struct db_core;

	template<auto member_ptr_new> struct db_entity_regular {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type member_ptr{ member_ptr_new };
	};

	template<auto member_ptr_new> struct db_entity_member_ptr {
		using member_type = remove_class_pointer_t<decltype(member_ptr_new)>;
		using class_type  = remove_member_pointer_t<decltype(member_ptr_new)>;
		inline static constexpr member_type class_type::* member_ptr{ member_ptr_new };
	};

	template<auto member_ptr_new> using db_entity =
		std::conditional_t<std::is_member_pointer_v<decltype(member_ptr_new)>, db_entity_member_ptr<member_ptr_new>, db_entity_regular<member_ptr_new>>;

	template<typename value_type>
	concept db_entity_types = requires {
		detail::remove_cvref_t<value_type>::name;
		detail::remove_cvref_t<value_type>::member_ptr;
		typename detail::remove_cvref_t<value_type>::member_type;
	} && std::is_member_pointer_v<decltype(detail::remove_cvref_t<value_type>::member_ptr)>;

	template<auto member_ptr> inline static constexpr auto make_db_entity() {
		return db_entity<member_ptr>{};
	}

	template<typename value_type> struct serialize_impl;

	struct serialize {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			using value_type = detail::remove_cvref_t<value_type_new>;
			serialize_impl<value_type>::generate_metadata(detail::forward<value_type_new>(value), context, metadata);
		}
	};

	template<typename db_entity_type> struct db_entity_serialize : public db_entity_type {
		constexpr db_entity_serialize() noexcept = default;

		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void process_index(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			serialize::generate_metadata(value.*db_entity_type::member_ptr, context, metadata);
		}
	};

	template<typename... bases> struct serialize_map : public bases... {
		template<typename db_entity_type, typename... arg_types> NIHILUS_HOST static void iterate_valuesImpl([[maybe_unused]] arg_types&&... args) {
			db_entity_type::process_index(detail::forward<arg_types>(args)...);
		}

		template<typename... arg_types> static constexpr void iterate_values([[maybe_unused]] arg_types&&... args) {
			((iterate_valuesImpl<bases>(detail::forward<arg_types>(args)...)), ...);
		}
	};

	template<typename value_type> struct serialize_impl;

	template<typename value_type, typename index_sequence, typename... value_types> struct get_serialize_base;

	template<typename value_type> using db_tuple_type			   = decltype(db_core<detail::remove_cvref_t<value_type>>::db_value);
	template<typename value_type> constexpr uint64_t db_tuple_size = tuple_size_v<db_tuple_type<value_type>>;

	template<typename value_type, size_t... index> struct get_serialize_base<value_type, std::index_sequence<index...>> {
		using type = serialize_map<db_entity_serialize<std::remove_cvref_t<decltype(get<index>(db_core<value_type>::db_value))>>...>;
	};

	template<typename value_type> using serialize_base_t = typename get_serialize_base<value_type, std::make_index_sequence<db_tuple_size<value_type>>>::type;

	template<object_types value_type> struct serialize_impl<value_type> {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			uint64_t current_data_byte_length_start{ context.header.data_byte_length };
			context.header.metadata_byte_length += sizeof(nihilus_db_member_metadata);
			uint64_t index = metadata.data.size();
			metadata.data.emplace_back();
			serialize_base_t<value_type>::iterate_values(value, context, metadata);
			metadata.data[index].byte_offset  = current_data_byte_length_start;
			metadata.data[index].byte_length = context.header.data_byte_length - current_data_byte_length_start;
			metadata.data[index].db_data_type = db_data_types::object;
		}
	};

	template<vector_types value_type> struct serialize_impl<value_type> {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			uint64_t current_data_byte_length_start{ context.header.data_byte_length };
			context.header.metadata_byte_length += sizeof(nihilus_db_member_metadata);
			uint64_t index = metadata.data.size();
			metadata.data.emplace_back();
			const auto newSize = value.size();
			if NIHILUS_LIKELY (newSize > 0) {
				for (auto iter = value.begin(); iter != value.end(); ++iter) {
					serialize::generate_metadata(*iter, context, metadata);
				}
			}
			metadata.data[index].byte_offset = current_data_byte_length_start;
			metadata.data[index].byte_length  = context.header.data_byte_length - current_data_byte_length_start;
			metadata.data[index].db_data_type = db_data_types::vector;
		}
	};

	template<string_types value_type> struct serialize_impl<value_type> {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			uint64_t current_data_byte_length_start{ context.header.data_byte_length };
			context.header.metadata_byte_length += sizeof(nihilus_db_member_metadata);
			auto& new_metadata = metadata.data.emplace_back();
			context.header.data_byte_length += value.size();
			new_metadata.byte_offset = current_data_byte_length_start;
			new_metadata.byte_length  = context.header.data_byte_length - current_data_byte_length_start;
			new_metadata.db_data_type = db_data_types::string;
		}
	};

	template<enum_types value_type> struct serialize_impl<value_type> {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			uint64_t current_data_byte_length_start{ context.header.data_byte_length };
			context.header.metadata_byte_length += sizeof(nihilus_db_member_metadata);
			auto& new_metadata = metadata.data.emplace_back();
			context.header.data_byte_length += sizeof(value);
			new_metadata.byte_offset = current_data_byte_length_start;
			new_metadata.byte_length = context.header.data_byte_length - current_data_byte_length_start;
			new_metadata.db_data_type = db_data_types::number;
		}
	};

	template<num_types value_type> struct serialize_impl<value_type> {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			uint64_t current_data_byte_length_start{ context.header.data_byte_length };
			context.header.metadata_byte_length += sizeof(nihilus_db_member_metadata);
			auto& new_metadata = metadata.data.emplace_back();
			context.header.data_byte_length += sizeof(value);
			new_metadata.byte_offset = current_data_byte_length_start;
			new_metadata.byte_length  = context.header.data_byte_length - current_data_byte_length_start;
			new_metadata.db_data_type = db_data_types::number;
		}
	};

	template<bool_types value_type> struct serialize_impl<value_type> {
		template<typename value_type_new, typename context_type, typename metadata_type>
		NIHILUS_HOST static void generate_metadata(value_type_new&& value, context_type& context, metadata_type& metadata) noexcept {
			uint64_t current_data_byte_length_start{ context.header.data_byte_length };
			context.header.metadata_byte_length += sizeof(nihilus_db_member_metadata);
			auto& new_metadata = metadata.data.emplace_back();
			context.header.data_byte_length += sizeof(value);
			new_metadata.byte_offset = current_data_byte_length_start;
			new_metadata.byte_length  = context.header.data_byte_length - current_data_byte_length_start;
			new_metadata.db_data_type = db_data_types::boolean;
		}
	};
}
