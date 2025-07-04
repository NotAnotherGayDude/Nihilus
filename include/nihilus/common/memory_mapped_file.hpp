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

#include <nihilus/common/config.hpp>

#if defined(NIHILUS_PLATFORM_WINDOWS)
	#if !defined(NOMINMAX)
		#define NOMINMAX
	#endif
	#include <Windows.h>
	#include <io.h>
	#ifndef PATH_MAX
		#define PATH_MAX MAX_PATH
	#endif
#else
	#include <sys/mman.h>
	#include <sys/stat.h>
	#include <fcntl.h>
	#include <unistd.h>
	#if defined(NIHIULUS_PLATFORM_LINUX)
		#include <sys/resource.h>
	#endif
	#if defined(NIHIULUS_PLATFORM_MACOS)
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

#ifdef NIHILUS_PLATFORM_WINDOWS
	static std::string format_win_error(DWORD error_code) {
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
		NIHILUS_FORCE_INLINE explicit memory_mapped_file() noexcept = default;

		NIHILUS_FORCE_INLINE explicit memory_mapped_file(std::string_view file_path, uint64_t prefetch_bytes = 0, bool numa_aware = false) : file_path_(file_path) {
			map_file(file_path, prefetch_bytes, numa_aware);
			lock_memory();
		}

		NIHILUS_FORCE_INLINE ~memory_mapped_file() {
			unmap_file();
		}

		NIHILUS_FORCE_INLINE memory_mapped_file(memory_mapped_file&& other) noexcept {
			*this = std::move(other);
		}

		NIHILUS_FORCE_INLINE memory_mapped_file& operator=(memory_mapped_file&& other) noexcept {
			if (this != &other) {
				unmap_file();

				file_path_	 = other.file_path_;
				mapped_data_ = other.mapped_data_;
				file_size_	 = other.file_size_;
#ifdef NIHILUS_PLATFORM_WINDOWS
				file_handle_	= other.file_handle_;
				mapping_handle_ = other.mapping_handle_;
#else
				file_descriptor_  = other.file_descriptor_;
				mapped_fragments_ = std::move(other.mapped_fragments_);
#endif

				other.mapped_data_ = nullptr;
				other.file_size_   = 0;
#ifdef NIHILUS_PLATFORM_WINDOWS
				other.file_handle_	  = INVALID_HANDLE_VALUE;
				other.mapping_handle_ = nullptr;
#else
				other.file_descriptor_ = -1;
				other.mapped_fragments_.clear();
#endif
			}
			return *this;
		}

		memory_mapped_file(const memory_mapped_file&)			 = delete;
		memory_mapped_file& operator=(const memory_mapped_file&) = delete;

		NIHILUS_FORCE_INLINE void* data() const noexcept {
			return mapped_data_;
		}

		NIHILUS_FORCE_INLINE uint64_t size() const noexcept {
			return file_size_;
		}

		NIHILUS_FORCE_INLINE static bool memory_mapping_supported() noexcept {
#if defined(_POSIX_MAPPED_FILES) || defined(NIHILUS_PLATFORM_WINDOWS)
			return true;
#else
			return false;
#endif
		}

	  protected:
		std::string_view file_path_;
		void* mapped_data_	= nullptr;
		uint64_t file_size_ = 0;

#ifdef NIHILUS_PLATFORM_WINDOWS
		HANDLE file_handle_	   = INVALID_HANDLE_VALUE;
		HANDLE mapping_handle_ = nullptr;
#else
		int32_t file_descriptor_ = -1;
		std::vector<std::pair<uint64_t, uint64_t>> mapped_fragments_;
#endif

		NIHILUS_FORCE_INLINE void map_file(std::string_view file_path, uint64_t prefetch_bytes, bool numa_aware) {
#ifdef NIHILUS_PLATFORM_WINDOWS
			( void )numa_aware;
			std::string_view file_path_str(file_path);

			file_handle_ = CreateFileA(file_path_str.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

			if (file_handle_ == INVALID_HANDLE_VALUE) {
				throw std::runtime_error(std::string{ "Failed to open file: " } + format_win_error(GetLastError()));
			}

			LARGE_INTEGER file_size;
			if (!GetFileSizeEx(file_handle_, &file_size)) {
				CloseHandle(file_handle_);
				throw std::runtime_error(std::string{ "Failed to get file size: " } + format_win_error(GetLastError()));
			}

			file_size_ = static_cast<uint64_t>(file_size.QuadPart);

			if (file_size_ == 0) {
				CloseHandle(file_handle_);
				throw std::runtime_error("Cannot map empty file");
			}

			mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);

			if (mapping_handle_ == nullptr) {
				CloseHandle(file_handle_);
				throw std::runtime_error("Failed to create file mapping: " + format_win_error(GetLastError()));
			}

			mapped_data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);

			if (mapped_data_ == nullptr) {
				CloseHandle(mapping_handle_);
				CloseHandle(file_handle_);
				throw std::runtime_error("Failed to map view of file: " + format_win_error(GetLastError()));
			}

			if (reinterpret_cast<std::uintptr_t>(mapped_data_) % cpu_alignment != 0) {
				UnmapViewOfFile(mapped_data_);
				CloseHandle(mapping_handle_);
				CloseHandle(file_handle_);
				throw std::runtime_error("Memory mapping failed to achieve required SIMD alignment");
			}

			if (prefetch_bytes > 0) {
	#if _WIN32_WINNT >= 0x602
				HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
				if (kernel32) {
					using PrefetchVirtualMemoryFunc = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
					auto prefetch_func				= reinterpret_cast<PrefetchVirtualMemoryFunc>(GetProcAddress(kernel32, "PrefetchVirtualMemory"));

					if (prefetch_func) {
						WIN32_MEMORY_RANGE_ENTRY range;
						range.VirtualAddress = mapped_data_;
						range.NumberOfBytes	 = std::min(file_size_, prefetch_bytes);

						if (!prefetch_func(GetCurrentProcess(), 1, &range, 0)) {
						}
					}
				}
	#endif
			}

#else
			std::string_view file_path_str(file_path);

			file_descriptor_ = open(file_path_str.data(), O_RDONLY);
			if (file_descriptor_ == -1) {
				throw std::runtime_error("Failed to open file: " + std::string(std::strerror(errno)));
			}

			struct stat file_stat;
			if (fstat(file_descriptor_, &file_stat) == -1) {
				close(file_descriptor_);
				throw std::runtime_error("Failed to get file statistics: " + std::string(std::strerror(errno)));
			}

			file_size_ = static_cast<uint64_t>(file_stat.st_size);

			if (file_size_ == 0) {
				close(file_descriptor_);
				throw std::runtime_error("Cannot map empty file");
			}

			int32_t flags = MAP_SHARED;
			if (numa_aware) {
				prefetch_bytes = 0;
			}

	#ifdef __linux__
			if (posix_fadvise(file_descriptor_, 0, 0, POSIX_FADV_SEQUENTIAL) != 0) {
			}

			if (prefetch_bytes > 0) {
				flags |= MAP_POPULATE;
			}
	#endif

			uint64_t aligned_size = ((file_size_ + cpu_alignment - 1) / cpu_alignment) * cpu_alignment;

			mapped_data_ = mmap(nullptr, aligned_size, PROT_READ, flags, file_descriptor_, 0);

			if (mapped_data_ == MAP_FAILED) {
				close(file_descriptor_);
				mapped_data_ = nullptr;
				throw std::runtime_error("Failed to memory map file: " + std::string(std::strerror(errno)));
			}

			if (reinterpret_cast<std::uintptr_t>(mapped_data_) % cpu_alignment != 0) {
				munmap(mapped_data_, aligned_size);
				close(file_descriptor_);
				throw std::runtime_error("Memory mapping failed to achieve required SIMD alignment");
			}

			mapped_fragments_.emplace_back(0, file_size_);

			if (prefetch_bytes > 0) {
				uint64_t prefetch_size = std::min(file_size_, prefetch_bytes);
				if (posix_madvise(mapped_data_, prefetch_size, POSIX_MADV_WILLNEED) != 0) {
				}
			}

			if (numa_aware) {
				if (posix_madvise(mapped_data_, file_size_, POSIX_MADV_RANDOM) != 0) {
				}
			} else {
				if (posix_madvise(mapped_data_, file_size_, POSIX_MADV_SEQUENTIAL) != 0) {
				}
			}
#endif
		}

		NIHILUS_FORCE_INLINE void unmap_file() {
#ifdef NIHILUS_PLATFORM_WINDOWS
			if (mapped_data_) {
				UnmapViewOfFile(mapped_data_);
				mapped_data_ = nullptr;
			}

			if (mapping_handle_) {
				CloseHandle(mapping_handle_);
				mapping_handle_ = nullptr;
			}

			if (file_handle_ != INVALID_HANDLE_VALUE) {
				CloseHandle(file_handle_);
				file_handle_ = INVALID_HANDLE_VALUE;
			}
#else
			for (const auto& frag: mapped_fragments_) {
				if (munmap(static_cast<uint8_t*>(mapped_data_) + frag.first, frag.second - frag.first) != 0) {
				}
			}
			mapped_fragments_.clear();

			if (file_descriptor_ != -1) {
				close(file_descriptor_);
				file_descriptor_ = -1;
			}
#endif
			file_size_ = 0;
		}

		NIHILUS_FORCE_INLINE void unmap_fragment(uint64_t first, uint64_t last) {
#ifdef NIHILUS_PLATFORM_WINDOWS
			( void )first;
			( void )last;
#else
			if (!mapped_data_)
				return;

			long page_size = sysconf(_SC_PAGESIZE);
			if (page_size <= 0)
				return;

			uint64_t page_uint64_t = static_cast<uint64_t>(page_size);

			uint64_t offset_in_page = first & (page_uint64_t - 1);
			if (offset_in_page != 0) {
				first += page_uint64_t - offset_in_page;
			}

			last = last & ~(page_uint64_t - 1);

			if (last <= first)
				return;

			void* unmap_addr	= static_cast<uint8_t*>(mapped_data_) + first;
			uint64_t unmap_size = last - first;

			if (munmap(unmap_addr, unmap_size) != 0) {
				return;
			}

			std::vector<std::pair<uint64_t, uint64_t>> new_fragments;
			for (const auto& frag: mapped_fragments_) {
				if (frag.first < first && frag.second > last) {
					new_fragments.emplace_back(frag.first, first);
					new_fragments.emplace_back(last, frag.second);
				} else if (frag.first < first && frag.second > first) {
					new_fragments.emplace_back(frag.first, first);
				} else if (frag.first < last && frag.second > last) {
					new_fragments.emplace_back(last, frag.second);
				} else if (frag.first >= first && frag.second <= last) {
				} else {
					new_fragments.push_back(frag);
				}
			}
			mapped_fragments_ = std::move(new_fragments);
#endif
		}

		NIHILUS_FORCE_INLINE void lock_memory() {
#ifdef NIHILUS_PLATFORM_WINDOWS
			if (mapped_data_ && file_size_ > 0) {
				VirtualLock(mapped_data_, file_size_);
			}
#else
	#ifdef _POSIX_MEMLOCK_RANGE
			if (mapped_data_ && file_size_ > 0) {
				mlock(mapped_data_, file_size_);
			}
	#endif
#endif
		}
	};

}
