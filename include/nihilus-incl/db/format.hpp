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

#include <nihilus-incl/db/core.hpp>
#include <nihilus-incl/db/file_io.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <cstring>
#include <cstdint>

namespace nihilus {

	struct snowflake;

}

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
		null,
		object,
		vector,
		number,
		boolean,
		string,
	};

	struct nihilus_db_member_metadata {
		db_data_types db_data_type{};
		uint64_t byte_offset{};
		uint64_t byte_length{};
	};

	struct NIHILUS_ALIGN(64) nihilus_db_metadata_new {
		std::vector<nihilus_db_member_metadata> data{};
		uint64_t byte_offset{};
		uint64_t byte_length{};
		snowflake id{};
		bool in_use{};
	};

	struct NIHILUS_ALIGN(64) nihilus_db_metadata {
		uint64_t byte_offset{};
		uint64_t byte_length{};
		snowflake id{};
		bool in_use{};
	};

	struct NIHILUS_ALIGN(64) nihilus_db_header {
		static constexpr uint32_t magic_val{ pack_values_4("void") };
		static constexpr uint32_t metadata_size{ sizeof(nihilus_db_metadata) };
		NIHILUS_HOST nihilus_db_header() noexcept {}
		NIHILUS_HOST nihilus_db_header(std::string_view db_name) noexcept
			: db_name_offset{ sizeof(magic) + sizeof(active_version) + sizeof(file_length) + sizeof(version) }, db_name_length{ db_name.size() },
			  metadata_byte_offset{ db_name_offset + db_name_length } {}
		uint32_t magic{ magic_val };
		uint32_t active_version{ 0 };
		uint64_t file_length{};
		uint32_t version{ 1 };
		uint64_t db_name_offset{};
		uint64_t db_name_length{};
		uint64_t metadata_byte_offset{};
		uint64_t metadata_byte_length{};
		uint64_t data_byte_length{};
		uint64_t tombstoned_byte_offset{};
		uint64_t tombstoned_byte_length{};
	};

	void test_function();

	template<typename value_type_new> class nihilus_db_file {
	  public:
		using value_type = value_type_new;
		using reference	 = value_type&;
		using pointer	 = value_type*;
		friend void test_function();

		NIHILUS_HOST nihilus_db_file() noexcept {}

		NIHILUS_HOST static nihilus_db_file open(std::string_view path) {
			nihilus_db_file db;
			db.base_path = path;

			std::string meta_path = db.base_path + ".void_meta";

			if (!std::filesystem::exists(meta_path)) {
				return std::move(db);
			}

			auto meta_file = io_file<file_access_types::read>::open_file(meta_path, 0);

			nihilus_db_header header{};
			void* header_buffer = static_cast<void*>(&header);
			meta_file.read_data(0, sizeof(nihilus_db_header), header_buffer);

			if (header.magic == nihilus_db_header::magic_val && header.metadata_byte_length > 0) {
				std::string data_path = db.base_path + (header.active_version == 0 ? ".void_A" : ".void_B");

				db.metadata_cache.resize(header.metadata_byte_length);
				void* metadata_buffer  = static_cast<void*>(db.metadata_cache.data());
				uint64_t metadata_size = header.metadata_byte_length * sizeof(nihilus_db_metadata);
				meta_file.read_data(header.metadata_byte_offset, metadata_size, metadata_buffer);

				db.data_reader_file = io_file<file_access_types::read>::open_file(data_path, 0);
				db.tombstoned_count = header.tombstoned_byte_length / sizeof(uint64_t);
				db.tombstoned.resize(db.tombstoned_count);
				db.current_active_version = header.active_version;

				uint64_t tombstone_index{};
				for (uint64_t i = 0; i < header.metadata_byte_length; ++i) {
					if (db.metadata_cache[i].in_use) {
						db.snowflake_to_index[db.metadata_cache[i].id] = i;
					} else {
						db.tombstoned[tombstone_index] = i;
						++tombstone_index;
					}
				}
				if (tombstone_index != db.tombstoned_count) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<true, "Database corruption: incorrect tombstoned count!", location>::impl();
				}
			}

			return db;
		}

		NIHILUS_HOST uint64_t get_index() {
			uint64_t index = 0;
			if (tombstoned_count > 0) {
				index = tombstoned[tombstoned_count - 1];
				--tombstoned_count;
				return index;
			} else {
				index = metadata_cache.size();
				metadata_cache.emplace_back();
				return index;
			}
		}

		NIHILUS_HOST reference add_record(const snowflake& id) {
			uint64_t index{ get_index() };
			pending_writes[index]			  = {};
			metadata_cache[index].id		  = id;
			metadata_cache[index].in_use	  = true;
			metadata_cache[index].byte_offset = 0;
			snowflake_to_index[id]			  = index;
			return pending_writes[index];
		}

		NIHILUS_HOST reference operator[](const snowflake& id) {
			return get_record(id);
		}

		NIHILUS_HOST reference get_record(const snowflake& id) {
			auto it = snowflake_to_index.find(id);
			if (it == snowflake_to_index.end()) {
				return add_record(id);
			}

			uint64_t index					= it->second;
			const nihilus_db_metadata& meta = metadata_cache[index];

			if (!meta.in_use) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<true, "Database corruption: metadata marked inactive for existing record ID", location>::impl();
			}

			auto pending_it = pending_writes.find(index);
			if (pending_it != pending_writes.end()) {
				return pending_it->second;
			} else {
				data_reader_file.read_data(meta.byte_offset, sizeof(value_type), &pending_writes[index]);
				return pending_writes[index];
			}
		}

		NIHILUS_HOST bool delete_record(const snowflake& id) {
			auto it = snowflake_to_index.find(id);
			if (it == snowflake_to_index.end()) {
				return false;
			}

			uint64_t index = it->second;

			if (!metadata_cache[index].in_use) {
				return false;
			}

			metadata_cache[index].in_use = false;
			if (tombstoned_count >= tombstoned.size()) {
				tombstoned.emplace_back(index);
			} else {
				tombstoned[tombstoned_count] = index;
			}
			++tombstoned_count;
			pending_writes.erase(index);
			snowflake_to_index.erase(it);

			return true;
		}

		NIHILUS_HOST void flush() {
			std::string meta_path	   = base_path + ".void_meta";
			std::string meta_temp_path = base_path + ".void_meta_temp";

			uint8_t new_version		  = 1 - current_active_version;
			std::string old_data_path = base_path + (current_active_version == 0 ? ".void_A" : ".void_B");
			std::string new_data_path = base_path + (new_version == 0 ? ".void_A" : ".void_B");

			uint64_t current_offset = 0;
			for (uint64_t i = 0; i < metadata_cache.size(); ++i) {
				if (metadata_cache[i].in_use) {
					metadata_cache[i].byte_offset = current_offset;
					current_offset += sizeof(value_type);
				}
			}

			uint64_t data_file_size = current_offset + (tombstoned_count * sizeof(uint64_t));
			if (data_file_size == 0)
				data_file_size = 64;
			auto new_data_file = io_file<file_access_types::read_write>::open_file(new_data_path, data_file_size);

			value_type holder{};
			void* holder_buffer = static_cast<void*>(&holder);
			for (uint64_t i = 0; i < metadata_cache.size(); ++i) {
				if (metadata_cache[i].in_use) {
					auto pending_it = pending_writes.find(i);
					if (pending_it != pending_writes.end()) {
						void* data_buffer = static_cast<void*>(&pending_it->second);
						new_data_file.write_data(metadata_cache[i].byte_offset, sizeof(value_type), data_buffer);
					} else {
						data_reader_file.read_data(metadata_cache[i].byte_offset, sizeof(value_type), holder_buffer);
						new_data_file.write_data(metadata_cache[i].byte_offset, sizeof(value_type), holder_buffer);
					}
				}
			}
			new_data_file.write_data(current_offset, tombstoned_count * sizeof(uint64_t), tombstoned.data());
			new_data_file.cleanup();

			nihilus_db_header header{};
			header.active_version	 = new_version;
			header.file_length		 = sizeof(nihilus_db_header) + (metadata_cache.size() * sizeof(nihilus_db_metadata));
			header.version			 = 1;
			header.metadata_byte_offset	 = sizeof(nihilus_db_header);
			header.metadata_byte_length	 = metadata_cache.size();
			header.tombstoned_byte_offset = current_offset;
			header.tombstoned_byte_length = tombstoned_count * sizeof(uint64_t);

			uint64_t meta_file_size = sizeof(nihilus_db_header) + (metadata_cache.size() * sizeof(nihilus_db_metadata));
			auto meta_file_temp		= io_file<file_access_types::read_write>::open_file(meta_temp_path, meta_file_size);

			void* header_buffer = static_cast<void*>(&header);
			meta_file_temp.write_data(0, sizeof(nihilus_db_header), header_buffer);

			if (!metadata_cache.empty()) {
				void* metadata_buffer  = static_cast<void*>(metadata_cache.data());
				uint64_t metadata_size = metadata_cache.size() * sizeof(nihilus_db_metadata);
				meta_file_temp.write_data(header.metadata_byte_offset, metadata_size, metadata_buffer);
			}

			meta_file_temp.cleanup();

			std::error_code ec;
			std::filesystem::rename(meta_temp_path, meta_path, ec);
			if (ec) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<true, "CRITICAL: Atomic metadata commit failed! Error: ", location>::impl(ec.message());
			}
			data_reader_file.cleanup();
			if (std::filesystem::exists(old_data_path)) {
				std::filesystem::remove(old_data_path, ec);
			}

			pending_writes.clear();
			current_active_version = new_version;
			data_reader_file	   = io_file<file_access_types::read>::open_file(new_data_path, 0);
		}

		nihilus_db_file& operator=(nihilus_db_file&& other) noexcept = default;
		nihilus_db_file(nihilus_db_file&& other) noexcept			 = default;

		~nihilus_db_file() noexcept = default;

	  private:
		std::unordered_map<snowflake, uint64_t> snowflake_to_index{};
		std::unordered_map<uint64_t, value_type> pending_writes{};
		io_file<file_access_types::read> data_reader_file{};
		std::vector<nihilus_db_metadata> metadata_cache{};
		uint8_t current_active_version{ 0 };
		std::vector<uint64_t> tombstoned{};
		uint64_t tombstoned_count{};
		std::string base_path{};
	};

}

namespace std {

	uint64_t hash<nihilus::snowflake>::operator()(const nihilus::snowflake& other) const {
		return hash<uint64_t>::operator()(other.operator const uint64_t&());
	}

}
