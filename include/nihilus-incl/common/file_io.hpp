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

#include <nihilus-incl/common/model_config.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <thread>
#include <mutex>
#include <latch>
#include <cmath>

namespace nihilus {

	template<const model_config& config_new> struct config_holder {
		static constexpr const model_config& config{ config_new };
	};

	template<const model_config& config> class file_loader;

	template<const model_config& config>
		requires(config.exceptions)
	class file_loader<config> {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_loader - Path does not exist", location>::impl(filePath.string());
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_loader - Failed to open file", location>::impl();
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					static constexpr auto location = std::source_location::current();
					nihilus_exception<config, "file_loader - Failed to read file", location>::impl();
				}
			}
		}

		operator const std::string&() const noexcept {
			return contents;
		}

		uint64_t size() const noexcept {
			return contents.size();
		}

	  protected:
		std::string contents;
	};

	template<const model_config& config>
		requires(!config.exceptions)
	class file_loader<config> {
	  public:
		explicit file_loader(const std::filesystem::path& filePath) {
			if (!std::filesystem::exists(filePath)) {
				log<log_levels::error>("file_loader - Path does not exist");
			}

			std::ifstream file(filePath, std::ios::binary | std::ios::ate);
			if (!file) {
				log<log_levels::error>("file_loader - Failed to open file");
			}

			const std::streamsize size = file.tellg();
			file.seekg(0, std::ios::beg);
			if (size != -1) {
				contents.resize(static_cast<uint64_t>(size));
				if (!file.read(contents.data(), size)) {
					log<log_levels::error>("file_loader - Failed to read file");
				}
			}
		}

		operator const std::string&() const noexcept {
			return contents;
		}

		uint64_t size() const noexcept {
			return contents.size();
		}

	  protected:
		std::string contents;
	};

	template<const model_config& config> class file_saver {
	  public:
		file_saver(const std::filesystem::path& path, const void* data, uint64_t size) {
			if (!data || size == 0) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_saver - Cannot save null or empty data to file: ", location>::impl(path.string());
			}

			std::ofstream file(path, std::ios::binary | std::ios::trunc);
			if (!file) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_saver - Cannot save null or empty data to file: ", location>::impl(path.string());
			}

			file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
			if (!file) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "file_saver - Cannot save null or empty data to file: ", location>::impl(path.string());
			}
		}
	};
}
