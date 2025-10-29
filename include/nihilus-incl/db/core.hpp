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

#include <nihilus-incl/db/file_io.hpp>

namespace nihilus {

	struct snowflake;

};

namespace std {

	template<> struct hash<nihilus::snowflake> : hash<uint64_t> {
		NIHILUS_HOST uint64_t operator()(const nihilus::snowflake& other) const;
	};

}

namespace nihilus {

	inline static constexpr uint32_t pack_values_4(const char* values) {
		return static_cast<uint32_t>(
			static_cast<uint32_t>(values[0]) | static_cast<uint32_t>(values[1]) << 8 | static_cast<uint32_t>(values[2]) << 16 | static_cast<uint32_t>(values[3]) << 24);
	}

	struct snowflake {
		NIHILUS_HOST snowflake() {
		}

		NIHILUS_HOST snowflake& operator=(uint64_t other) {
			id = other;
			return *this;
		}

		NIHILUS_HOST snowflake(uint64_t other) {
			*this = other;
		}

		NIHILUS_HOST explicit operator const uint64_t&() const {
			return id;
		}

		NIHILUS_HOST explicit operator uint64_t&() {
			return id;
		}

		NIHILUS_HOST bool operator==(const snowflake& other) const {
			return id == other.id;
		}

		NIHILUS_HOST bool operator==(uint64_t other) const {
			return id == other;
		}

	  protected:
		uint64_t id{};
	};

	enum class db_data_types {
		null	= 1 << 0,
		object	= 1 << 1,
		vector	= 1 << 2,
		number	= 1 << 3,
		boolean = 1 << 4,
		string	= 1 << 5,
		count,
	};

	template<typename value_type> struct data_serializer;

	struct nihilus_db_member_metadata {
		db_data_types db_data_type{};
		uint64_t byte_offset{};
		uint64_t byte_length{};
	};

	struct nihilus_db_metadata_base {
		db_data_types db_data_type{};
		uint64_t byte_offset{};

	  protected:
		~nihilus_db_metadata_base() {
		}
	};

	template<typename value_type> struct nihilus_db_entity_old {
		nihilus_db_metadata_base* metadata{};
		uint64_t byte_offset{};
		uint64_t byte_length{};
		value_type data{};
		snowflake id{};
		bool in_use{};
	};

	struct NIHILUS_ALIGN(64) db_header {
		static constexpr uint32_t magic_val{ pack_values_4("void") };

		NIHILUS_HOST db_header() noexcept {
		}

		NIHILUS_HOST db_header(std::string_view db_name) noexcept
			: db_name_offset{ sizeof(magic) + sizeof(active_version) + sizeof(file_length) + sizeof(version) }, db_name_length{ db_name.size() },
			  record_metadata_offset{ db_name_offset + db_name_length }, metadata_byte_offset{ record_metadata_offset } {// Will be calculated later
		}

		uint32_t magic{};
		uint32_t active_version{ 0 };
		uint64_t file_length{};
		uint32_t version{ 1 };
		uint64_t db_name_offset{};
		uint64_t db_name_length{};

		uint64_t record_count{};
		uint64_t record_metadata_offset{};
		uint64_t metadata_byte_offset{};
		uint64_t metadata_byte_length{};
		uint64_t data_byte_length{};
		uint64_t tombstoned_byte_offset{};
		uint64_t tombstoned_byte_length{};
	};

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

	template<auto member_ptr> inline static constexpr auto make_db_entity() {
		return db_entity<member_ptr>{};
	}

	template<typename value_type> struct db_core;	

	template<> struct db_core<db_header> {
		using value_type			   = db_header;
		static constexpr auto db_value = create_value<make_db_entity<&value_type::magic>(), make_db_entity<&value_type::active_version>(),
			make_db_entity<&value_type::file_length>(), make_db_entity<&value_type::version>(), make_db_entity<&value_type::db_name_offset>(),
			make_db_entity<&value_type::db_name_length>(), make_db_entity<&value_type::record_count>(), make_db_entity<&value_type::record_metadata_offset>(),
			make_db_entity<&value_type::metadata_byte_offset>(), make_db_entity<&value_type::metadata_byte_length>(), make_db_entity<&value_type::data_byte_length>(),
			make_db_entity<&value_type::tombstoned_byte_offset>(), make_db_entity<&value_type::tombstoned_byte_length>()>();
	};

	template<> struct db_core<nihilus_db_metadata_base> {
		using value_type			   = nihilus_db_metadata_base;
		static constexpr auto db_value = create_value<make_db_entity<&value_type::db_data_type>()>();
	};

	template<> struct db_core<nihilus_db_member_metadata> {
		using value_type = nihilus_db_member_metadata;
		static constexpr auto db_value =
			create_value<make_db_entity<&value_type::byte_length>(), make_db_entity<&value_type::byte_offset>(), make_db_entity<&value_type::db_data_type>()>();
	};

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

	template<template<typename> typename function_type, typename... bases> struct serialize_map : public bases... {
		template<typename... arg_types> static constexpr void iterate_values([[maybe_unused]] arg_types&&... args) {
			((function_type<bases>::impl(detail::forward<arg_types>(args)...)), ...);
		}
	};

	template<template<typename> typename function_type, typename value_type, typename index_sequence, typename... value_types> struct get_db_cores;

	template<typename value_type> using db_tuple_type			   = decltype(db_core<detail::remove_cvref_t<value_type>>::db_value);
	template<typename value_type> constexpr uint64_t db_tuple_size = tuple_size_v<db_tuple_type<value_type>>;

	template<template<typename> typename function_type, typename value_type, size_t... index> struct get_db_cores<function_type, value_type, std::index_sequence<index...>> {
		using type = serialize_map<function_type, db_entity_serialize<std::remove_cvref_t<decltype(get<index>(db_core<value_type>::db_value))>>...>;
	};

	template<template<typename> typename function_type, typename value_type> using db_cores_t =
		typename get_db_cores<function_type, value_type, std::make_index_sequence<db_tuple_size<value_type>>>::type;

	struct metadata {
		db_data_types db_data_type{};
		uint64_t byte_offset{};
		uint64_t byte_length{};
		uint64_t element_count{};
	};

	struct db_metadata_size_data {
		uint64_t metadata_count{};
		uint64_t byte_size{};
	};

	struct db_data_size_data {
		uint64_t byte_size{};
	};

	template<typename value_type> struct metadata_size_collector;

	template<bool_types value_type> struct metadata_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type&, db_header& header) {
			header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<num_types value_type> struct metadata_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type&, db_header& header) {
			header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<string_types value_type> struct metadata_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type&, db_header& header) {
			header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<vector_types value_type> struct metadata_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type& value, db_header& header) {
			for (auto& value_new: value) {
				metadata_size_collector<decltype(value_new)>::impl(value_new, header);
			}
			header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<object_types value_type> struct metadata_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type& value, db_header& header) {
			db_cores_t<metadata_size_collector, value_type>::impl(value, header);
			header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<typename value_type> struct data_size_collector;

	template<bool_types value_type> struct data_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type&, db_data_size_data& data_size) {
			data_size.byte_size += sizeof(value_type);
		}
	};

	template<num_types value_type> struct data_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type&, db_data_size_data& data_size) {
			data_size.byte_size += sizeof(value_type);
		}
	};

	template<string_types value_type> struct data_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type& value, db_data_size_data& data_size) {
			data_size.byte_size += value.size();
		}
	};

	template<vector_types value_type> struct data_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type& value, db_data_size_data& data_size) {
			for (auto& value_new: value) {
				data_size_collector<decltype(value_new)>::impl(value_new, data_size);
			}
		}
	};

	template<object_types value_type> struct data_size_collector<value_type> {
		NIHILUS_HOST static void impl(value_type& value, db_data_size_data& data_size) {
			db_cores_t<data_size_collector, value_type>::impl(value, data_size);
		}
	};

	template<typename value_type> struct db_serializer;

	template<bool_types value_type> struct db_serializer<value_type> {
		NIHILUS_HOST static void impl(char* metadata_serialize_location, char* data_serialize_location, value_type& value, db_header& metadata_header) {
			metadata data;
			data.db_data_type  = db_data_types::boolean;
			data.byte_length   = sizeof(value_type);
			data.byte_offset   = metadata_header.data_byte_length;
			data.element_count = 1;
			std::memcpy(metadata_serialize_location + metadata_header.metadata_byte_length, &data, sizeof(metadata));
			std::memcpy(data_serialize_location + metadata_header.data_byte_length, &value, sizeof(value_type));
			metadata_header.metadata_byte_length += sizeof(metadata);
			metadata_header.data_byte_length += sizeof(value_type);
		}
	};

	template<num_types value_type> struct db_serializer<value_type> {
		NIHILUS_HOST static void impl(char* metadata_serialize_location, char* data_serialize_location, value_type& value, db_header& metadata_header) {
			metadata data;
			data.db_data_type  = db_data_types::number;
			data.byte_length   = sizeof(value_type);
			data.byte_offset   = metadata_header.data_byte_length;
			data.element_count = 1;
			std::memcpy(metadata_serialize_location + metadata_header.metadata_byte_length, &data, sizeof(metadata));
			std::memcpy(data_serialize_location + metadata_header.data_byte_length, &value, sizeof(value_type));
			metadata_header.metadata_byte_length += sizeof(metadata);
			metadata_header.data_byte_length += sizeof(value_type);
		}
	};

	template<string_types value_type> struct db_serializer<value_type> {
		NIHILUS_HOST static void impl(char* metadata_serialize_location, char* data_serialize_location, value_type& value, db_header& metadata_header) {
			metadata data;
			data.db_data_type  = db_data_types::string;
			data.byte_length   = value.size();
			data.byte_offset   = metadata_header.data_byte_length;
			data.element_count = 1;
			std::memcpy(metadata_serialize_location + metadata_header.metadata_byte_length, &data, sizeof(metadata));
			std::memcpy(data_serialize_location + metadata_header.data_byte_length, value.data(), value.size());
			metadata_header.metadata_byte_length += sizeof(metadata);
			metadata_header.data_byte_length += value.size();
		}
	};

	template<vector_types value_type> struct db_serializer<value_type> {
		NIHILUS_HOST static void impl(char* metadata_serialize_location, char* data_serialize_location, value_type& value, db_header& metadata_header) {
			metadata data;
			data.db_data_type = db_data_types::vector;
			data.byte_offset  = metadata_header.data_byte_length;
			for (auto& value_new: value) {
				db_serializer<decltype(value_new)>::impl(metadata_serialize_location, data_serialize_location, value_new, metadata_header);
				++data.element_count;
			}
			data.byte_length = metadata_header.data_byte_length - data.byte_offset;
			std::memcpy(metadata_serialize_location + metadata_header.metadata_byte_length, &data, sizeof(metadata));
			metadata_header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<object_types value_type> struct db_serializer<value_type> {
		NIHILUS_HOST static void impl(char* metadata_serialize_location, char* data_serialize_location, value_type& value, db_header& metadata_header) {
			metadata data;
			data.db_data_type = db_data_types::object;
			data.byte_offset  = metadata_header.data_byte_length;
			metadata_header.data_byte_length += sizeof(metadata);
			db_cores_t<db_serializer, value_type>::impl(metadata_serialize_location, data_serialize_location, value, metadata_header);
			data.byte_length = metadata_header.data_byte_length - data.byte_offset;
			std::memcpy(metadata_serialize_location + metadata_header.metadata_byte_length, &data, sizeof(metadata));
			metadata_header.metadata_byte_length += sizeof(metadata);
		}
	};

	template<typename value_type> struct db_parser;

	template<bool_types value_type> struct db_parser<value_type> {
		NIHILUS_HOST static void impl(char*& metadata_serialize_location, char*& data_serialize_location, value_type& value) {
			metadata data;
			std::memcpy(&data, metadata_serialize_location, sizeof(metadata));
			metadata_serialize_location += sizeof(metadata);
			std::memcpy(&value, data_serialize_location + data.byte_offset, sizeof(value_type));
			data_serialize_location += data.byte_length;
		}
	};

	template<num_types value_type> struct db_parser<value_type> {
		NIHILUS_HOST static void impl(char*& metadata_serialize_location, char*& data_serialize_location, value_type& value) {
			metadata data;
			std::memcpy(&data, metadata_serialize_location, sizeof(metadata));
			metadata_serialize_location += sizeof(metadata);
			std::memcpy(&value, data_serialize_location + data.byte_offset, sizeof(value_type));
			data_serialize_location += data.byte_length;
		}
	};

	template<string_types value_type> struct db_parser<value_type> {
		NIHILUS_HOST static void impl(char*& metadata_serialize_location, char*& data_serialize_location, value_type& value) {
			metadata data;
			std::memcpy(&data, metadata_serialize_location, sizeof(metadata));
			metadata_serialize_location += sizeof(metadata);
			value.resize(data.byte_length);
			std::memcpy(value.data(), data_serialize_location + data.byte_offset, sizeof(value_type));
			data_serialize_location += data.byte_length;
		}
	};

	template<vector_types value_type> struct db_parser<value_type> {
		NIHILUS_HOST static void impl(char*& metadata_serialize_location, char*& data_serialize_location, value_type& value) {
			metadata data;
			std::memcpy(&data, metadata_serialize_location, sizeof(metadata));
			metadata_serialize_location += sizeof(metadata);
			value.resize(data.element_count);
			for (uint64_t x = 0; x < data.element_count; ++x) {
				db_parser<decltype(value[x])>::impl(metadata_serialize_location, data_serialize_location, value[x]);
			}
		}
	};

	template<object_types value_type> struct db_parser<value_type> {
		NIHILUS_HOST static void impl(char*& metadata_serialize_location, char*& data_serialize_location, value_type& value) {
			metadata data;
			std::memcpy(&data, metadata_serialize_location, sizeof(metadata));
			metadata_serialize_location += sizeof(metadata);
			db_cores_t<db_parser, value_type>::impl(metadata_serialize_location, data_serialize_location, value);
		}
	};

}


namespace std {

	uint64_t hash<nihilus::snowflake>::operator()(const nihilus::snowflake& other) const {
		return hash<uint64_t>::operator()(other.operator const uint64_t&());
	}

}
