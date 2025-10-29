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

namespace nihilus {	

	void test_function();

	struct nihilus_db_record_metadata {
		snowflake id{};
		bool in_use{};
		uint64_t metadata_offset{};
		uint64_t metadata_length{};
		uint64_t data_offset{};
		uint64_t data_length{};
	};

	template<typename value_type_new> class nihilus_db_file {
	  public:
		using value_type = value_type_new;
		using reference	 = value_type&;
		using pointer	 = value_type*;
		friend void test_function();

		NIHILUS_HOST nihilus_db_file() noexcept {
		}

		NIHILUS_HOST static nihilus_db_file open(std::string_view path) {
			nihilus_db_file db;
			db.base_path = path;

			std::string meta_path = db.base_path + ".void_meta";

			if (!std::filesystem::exists(meta_path)) {
				return std::move(db);
			}

			auto meta_file = io_file<file_access_types::read>::open_file(meta_path, 0);

			db_header header{};
			void* header_buffer = static_cast<void*>(&header);
			meta_file.read_data(0, sizeof(db_header), header_buffer);

			if (header.magic == db_header::magic_val && header.record_count > 0) {
				std::string data_path = db.base_path + (header.active_version == 0 ? ".void_A" : ".void_B");

				db.record_metadata.resize(header.record_count);
				void* record_meta_buffer  = static_cast<void*>(db.record_metadata.data());
				uint64_t record_meta_size = header.record_count * sizeof(nihilus_db_record_metadata);
				meta_file.read_data(header.record_metadata_offset, record_meta_size, record_meta_buffer);

				db.metadata_buffer.resize(header.metadata_byte_length);
				meta_file.read_data(header.metadata_byte_offset, header.metadata_byte_length, db.metadata_buffer.data());

				db.data_reader_file = io_file<file_access_types::read>::open_file(data_path, 0);

				db.data_buffer.resize(header.data_byte_length);
				db.data_reader_file.read_data(0, header.data_byte_length, db.data_buffer.data());

				db.tombstoned_count = header.tombstoned_byte_length / sizeof(uint64_t);
				db.tombstoned.resize(db.tombstoned_count);
				db.current_active_version = header.active_version;

				uint64_t tombstone_index{};
				for (uint64_t i = 0; i < header.record_count; ++i) {
					if (db.record_metadata[i].in_use) {
						db.snowflake_to_index[db.record_metadata[i].id] = i;
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

		NIHILUS_HOST uint64_t find_or_allocate_space(uint64_t needed_metadata_size, uint64_t needed_data_size) {
			for (uint64_t i = 0; i < tombstoned_count; ++i) {
				uint64_t candidate_index = tombstoned[i];
				const auto& meta		 = record_metadata[candidate_index];

				if (meta.metadata_length >= needed_metadata_size && meta.data_length >= needed_data_size) {
					tombstoned[i] = tombstoned[tombstoned_count - 1];
					--tombstoned_count;
					return candidate_index;
				}
			}

			uint64_t index = record_metadata.size();
			record_metadata.emplace_back();
			return index;
		}

		NIHILUS_HOST uint64_t get_index() {
			if (tombstoned_count > 0) {
				uint64_t index = tombstoned[tombstoned_count - 1];
				--tombstoned_count;
				return index;
			} else {
				uint64_t index = record_metadata.size();
				record_metadata.emplace_back();
				return index;
			}
		}

		inline reference add_record(const snowflake& id) {
			auto it = snowflake_to_index.find(id);
			if (it != snowflake_to_index.end()) {
				return get_record(id);
			}

			uint64_t index{ get_index() };
			pending_writes[index]		  = {};
			record_metadata[index].id	  = id;
			record_metadata[index].in_use = true;
			snowflake_to_index[id]		  = index;
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

			uint64_t index						   = it->second;
			const nihilus_db_record_metadata& meta = record_metadata[index];

			if (!meta.in_use) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<true, "Database corruption: metadata marked inactive for existing record ID", location>::impl();
			}

			auto pending_it = pending_writes.find(index);
			if (pending_it != pending_writes.end()) {
				return pending_it->second;
			} else {
				char* metadata_ptr = metadata_buffer.data() + meta.metadata_offset;
				char* data_ptr	   = data_buffer.data() + meta.data_offset;

				db_parser<value_type>::impl(metadata_ptr, data_ptr, pending_writes[index]);

				return pending_writes[index];
			}
		}

		NIHILUS_HOST bool delete_record(const snowflake& id) {
			auto it = snowflake_to_index.find(id);
			if (it == snowflake_to_index.end()) {
				return false;
			}

			uint64_t index = it->second;

			if (!record_metadata[index].in_use) {
				return false;
			}

			record_metadata[index].in_use = false;
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
			if (pending_writes.empty()) {
				return;
			}

			std::string meta_path	   = base_path + ".void_meta";
			std::string meta_temp_path = base_path + ".void_meta_temp";

			uint32_t new_version		  = 1 - current_active_version;
			std::string old_data_path = base_path + (current_active_version == 0 ? ".void_A" : ".void_B");
			std::string new_data_path = base_path + (new_version == 0 ? ".void_A" : ".void_B");

			struct record_size_info {
				uint64_t metadata_size;
				uint64_t data_size;
			};
			std::unordered_map<uint64_t, record_size_info> changed_sizes;

			for (const auto& [index, data]: pending_writes) {
				if (record_metadata[index].in_use) {
					db_header size_header{};
					db_data_size_data size_data{};

					metadata_size_collector<value_type>::impl(const_cast<value_type&>(data), size_header);
					data_size_collector<value_type>::impl(const_cast<value_type&>(data), size_data);

					changed_sizes[index] = { size_header.metadata_byte_length, size_data.byte_size };
				}
			}

			uint64_t max_metadata_offset = 0;
			uint64_t max_data_offset	 = 0;

			for (const auto& meta: record_metadata) {
				if (meta.in_use) {
					max_metadata_offset = std::max(max_metadata_offset, meta.metadata_offset + meta.metadata_length);
					max_data_offset		= std::max(max_data_offset, meta.data_offset + meta.data_length);
				}
			}

			if (max_metadata_offset > metadata_buffer.size()) {
				metadata_buffer.resize(max_metadata_offset);
			}
			if (max_data_offset > data_buffer.size()) {
				data_buffer.resize(max_data_offset);
			}

			for (const auto& [index, data]: pending_writes) {
				if (!record_metadata[index].in_use) {
					continue;
				}

				auto& meta		  = record_metadata[index];
				const auto& sizes = changed_sizes[index];

				bool fits_in_place = (meta.metadata_length >= sizes.metadata_size && meta.data_length >= sizes.data_size && meta.metadata_length > 0);

				if (!fits_in_place) {
					bool found_space = false;

					for (uint64_t i = 0; i < tombstoned_count; ++i) {
						uint64_t tomb_index	  = tombstoned[i];
						const auto& tomb_meta = record_metadata[tomb_index];

						if (tomb_meta.metadata_length >= sizes.metadata_size && tomb_meta.data_length >= sizes.data_size) {
							meta.metadata_offset = tomb_meta.metadata_offset;
							meta.metadata_length = tomb_meta.metadata_length;
							meta.data_offset	 = tomb_meta.data_offset;
							meta.data_length	 = tomb_meta.data_length;
							found_space			 = true;
							break;
						}
					}

					if (!found_space) {
						meta.metadata_offset = max_metadata_offset;
						meta.metadata_length = sizes.metadata_size;
						meta.data_offset	 = max_data_offset;
						meta.data_length	 = sizes.data_size;

						max_metadata_offset += sizes.metadata_size;
						max_data_offset += sizes.data_size;

						if (max_metadata_offset > metadata_buffer.size()) {
							metadata_buffer.resize(max_metadata_offset);
						}
						if (max_data_offset > data_buffer.size()) {
							data_buffer.resize(max_data_offset);
						}
					}
				}

				db_header serialize_header{};
				serialize_header.metadata_byte_length = 0;
				serialize_header.data_byte_length	  = 0;

				db_serializer<value_type>::impl(metadata_buffer.data() + meta.metadata_offset,
					data_buffer.data() + meta.data_offset,
					const_cast<value_type&>(data), serialize_header);

				meta.metadata_length = serialize_header.metadata_byte_length;
				meta.data_length	 = serialize_header.data_byte_length;
			}

			uint64_t data_file_size = max_data_offset + (tombstoned_count * sizeof(uint64_t));
			if (data_file_size == 0)
				data_file_size = 64;

			auto new_data_file = io_file<file_access_types::read_write>::open_file(new_data_path, data_file_size);
			new_data_file.write_data(0, max_data_offset, data_buffer.data());
			new_data_file.write_data(max_data_offset, tombstoned_count * sizeof(uint64_t), tombstoned.data());
			new_data_file.cleanup();

			db_header final_header{};
			final_header.magic					= db_header::magic_val;
			final_header.active_version			= new_version;
			final_header.version				= 1;
			final_header.record_count			= record_metadata.size();
			final_header.record_metadata_offset = sizeof(db_header);
			final_header.metadata_byte_offset	= final_header.record_metadata_offset + (record_metadata.size() * sizeof(nihilus_db_record_metadata));
			final_header.metadata_byte_length	= max_metadata_offset;
			final_header.data_byte_length		= max_data_offset;
			final_header.tombstoned_byte_offset = max_data_offset;
			final_header.tombstoned_byte_length = tombstoned_count * sizeof(uint64_t);
			final_header.file_length			= final_header.metadata_byte_offset + final_header.metadata_byte_length;

			uint64_t meta_file_size = final_header.file_length;
			auto meta_file_temp		= io_file<file_access_types::read_write>::open_file(meta_temp_path, meta_file_size);

			void* header_buffer = static_cast<void*>(&final_header);
			meta_file_temp.write_data(0, sizeof(db_header), header_buffer);

			if (!record_metadata.empty()) {
				void* record_meta_buffer  = static_cast<void*>(record_metadata.data());
				uint64_t record_meta_size = record_metadata.size() * sizeof(nihilus_db_record_metadata);
				meta_file_temp.write_data(final_header.record_metadata_offset, record_meta_size, record_meta_buffer);
			}

			if (max_metadata_offset > 0) {
				meta_file_temp.write_data(final_header.metadata_byte_offset, max_metadata_offset, metadata_buffer.data());
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

		std::vector<nihilus_db_record_metadata> record_metadata{};
		std::vector<char> metadata_buffer{};
		std::vector<char> data_buffer{};

		uint32_t current_active_version{ 0 };
		std::vector<uint64_t> tombstoned{};
		uint64_t tombstoned_count{};
		std::string base_path{};
	};

}
