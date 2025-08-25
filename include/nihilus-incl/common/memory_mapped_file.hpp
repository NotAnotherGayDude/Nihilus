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

#include <nihilus-incl/common/common.hpp>

#if NIHILUS_PLATFORM_WINDOWS
	#ifndef PATH_MAX
		#define PATH_MAX MAX_PATH
	#endif
	#include <Windows.h>
	#include <io.h>
#else
	#include <sys/mman.h>
	#include <sys/stat.h>
	#include <fcntl.h>
	#include <unistd.h>
	#if NIHIULUS_PLATFORM_LINUX
		#include <sys/resource.h>
	#endif
	#if NIHIULUS_PLATFORM_MACOS
		#include <TargetConditionals.h>
	#endif
#endif
#include <unordered_set>
#include <variant>
#include <fstream>
#include <regex>
#include <map>
#include <bit>

namespace nihilus {

#if NIHILUS_PLATFORM_WINDOWS
	NIHILUS_INLINE static std::string format_win_error(DWORD error_code) {
		LPSTR buffer = nullptr;
		DWORD size	 = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, error_code,
			  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&buffer), 0, nullptr);

		if (!size || !buffer) {
			return "Unknown Win32 error: " + std::to_string(error_code);
		}

		std::string result(buffer, size);
		LocalFree(buffer);

		while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
			result.pop_back();
		}

		return result;
	}
#endif

	class memory_mapped_file {
	  public:
		NIHILUS_INLINE explicit memory_mapped_file() noexcept = default;

		NIHILUS_INLINE explicit memory_mapped_file(std::string_view file_path, uint64_t prefetch_bytes = 0, bool numa_aware = false) : file_path(file_path) {
			map_file(file_path, prefetch_bytes, numa_aware);
			lock_memory();
		}

		NIHILUS_INLINE ~memory_mapped_file() {
			unmap_file();
		}

		NIHILUS_INLINE memory_mapped_file(memory_mapped_file&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE memory_mapped_file& operator=(memory_mapped_file&& other) noexcept {
			if (this != &other) {
				unmap_file();
				std::swap(mapped_data, other.mapped_data);
				std::swap(file_path, other.file_path);
				std::swap(file_size, other.file_size);
#if NIHILUS_PLATFORM_WINDOWS
				std::swap(mapping_handle, other.mapping_handle);
				std::swap(file_handle, other.file_handle);
#else
				std::swap(mapped_fragments, other.mapped_fragments);
				std::swap(file_descriptor, other.file_descriptor);
#endif
			}
			return *this;
		}

		memory_mapped_file(const memory_mapped_file&)			 = delete;
		memory_mapped_file& operator=(const memory_mapped_file&) = delete;

		NIHILUS_INLINE void* data() const noexcept {
			return mapped_data;
		}

		NIHILUS_INLINE uint64_t size() const noexcept {
			return file_size;
		}

	  protected:
		std::string_view file_path{};
		uint64_t file_size{};
		void* mapped_data{};

#if NIHILUS_PLATFORM_WINDOWS
		HANDLE mapping_handle{};
		HANDLE file_handle{};
#else
		aligned_vector<std::pair<uint64_t, uint64_t>> mapped_fragments{};
		int32_t file_descriptor{};
#endif

		NIHILUS_INLINE void map_file(std::string_view file_path, uint64_t prefetch_bytes, bool numa_aware) {
#if NIHILUS_PLATFORM_WINDOWS
			( void )numa_aware;
			( void )prefetch_bytes;
			std::string_view file_pathstr(file_path);
			if (file_pathstr.empty()) {
				return;
			}
			file_handle = CreateFileA(file_pathstr.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (file_handle == INVALID_HANDLE_VALUE) {
				throw std::runtime_error(std::string{ "Failed to open file: " } + format_win_error(GetLastError()));
			}
			LARGE_INTEGER file_size_new;
			if (!GetFileSizeEx(file_handle, &file_size_new)) {
				CloseHandle(file_handle);
				throw std::runtime_error(std::string{ "Failed to get file size: " } + format_win_error(GetLastError()));
			}
			file_size = static_cast<uint64_t>(file_size_new.QuadPart);
			if (file_size == 0) {
				CloseHandle(file_handle);
				throw std::runtime_error("Cannot map empty file");
			}
			mapping_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
			if (mapping_handle == nullptr) {
				CloseHandle(file_handle);
				throw std::runtime_error("Failed to create file mapping: " + format_win_error(GetLastError()));
			}
			mapped_data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
			if (mapped_data == nullptr) {
				CloseHandle(mapping_handle);
				CloseHandle(file_handle);
				throw std::runtime_error("Failed to map view of file: " + format_win_error(GetLastError()));
			}
			if (reinterpret_cast<std::uintptr_t>(mapped_data) % 64 != 0) {
				UnmapViewOfFile(mapped_data);
				CloseHandle(mapping_handle);
				CloseHandle(file_handle);
				throw std::runtime_error("Memory mapping failed to achieve required SIMD alignment");
			}
#else
			( void )prefetch_bytes;
			std::string_view file_pathstr(file_path);
			file_descriptor = open(file_pathstr.data(), O_RDONLY);
			if (file_descriptor == -1) {
				throw std::runtime_error("Failed to open file: " + std::string(std::strerror(errno)));
			}
			struct stat file_stat;
			if (fstat(file_descriptor, &file_stat) == -1) {
				close(file_descriptor);
				throw std::runtime_error("Failed to get file statistics: " + std::string(std::strerror(errno)));
			}
			file_size = static_cast<uint64_t>(file_stat.st_size);
			if (file_size == 0) {
				close(file_descriptor);
				throw std::runtime_error("Cannot map empty file");
			}
			int32_t flags = MAP_SHARED;

	#ifdef __APPLE__
			fcntl(file_descriptor, F_RDAHEAD, 1);
	#else
			posix_fadvise(file_descriptor, 0, 0, POSIX_FADV_SEQUENTIAL);
	#endif

			uint64_t aligned_size = ((file_size + 64 - 1) / 64) * 64;
			mapped_data			  = mmap(nullptr, aligned_size, PROT_READ, flags, file_descriptor, 0);
			if (mapped_data == MAP_FAILED) {
				close(file_descriptor);
				mapped_data = nullptr;
				throw std::runtime_error("Failed to memory map file: " + std::string(std::strerror(errno)));
			}
			if (reinterpret_cast<std::uintptr_t>(mapped_data) % 64 != 0) {
				munmap(mapped_data, aligned_size);
				close(file_descriptor);
				throw std::runtime_error("Memory mapping failed to achieve required SIMD alignment");
			}

	#ifdef __APPLE__
			madvise(mapped_data, aligned_size, MADV_SEQUENTIAL);
	#endif

			mapped_fragments.emplace_back(0, file_size);
#endif
		}

		NIHILUS_INLINE void unmap_file() {
#if NIHILUS_PLATFORM_WINDOWS
			if (mapped_data) {
				UnmapViewOfFile(mapped_data);
				mapped_data = nullptr;
			}

			if (mapping_handle) {
				CloseHandle(mapping_handle);
				mapping_handle = nullptr;
			}

			if (file_handle != INVALID_HANDLE_VALUE) {
				CloseHandle(file_handle);
				file_handle = INVALID_HANDLE_VALUE;
			}
#else
			for (const auto& frag: mapped_fragments) {
				if (munmap(static_cast<uint8_t*>(mapped_data) + frag.first, frag.second - frag.first) != 0) {
				}
			}
			mapped_fragments.clear();

			if (file_descriptor != -1) {
				close(file_descriptor);
				file_descriptor = -1;
			}
#endif
			file_size = 0;
		}

		NIHILUS_INLINE void lock_memory() {
#if NIHILUS_PLATFORM_WINDOWS
			if (mapped_data && file_size > 0) {
				VirtualLock(mapped_data, file_size);
				SetProcessWorkingSetSize(GetCurrentProcess(), file_size * 2, file_size * 3);
			}
#else
			if (mapped_data && file_size > 0) {
				mlock(mapped_data, file_size);
				posix_madvise(mapped_data, file_size, POSIX_MADV_WILLNEED);
			}
#endif
		}
	};

}
