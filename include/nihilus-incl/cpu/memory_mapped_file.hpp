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
#include <nihilus-incl/common/model_config.hpp>
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

	template<model_config config> class memory_mapped_file {
	  public:
		NIHILUS_INLINE explicit memory_mapped_file() noexcept = default;

		NIHILUS_INLINE explicit memory_mapped_file(const std::string_view file_path_new, uint64_t file_offset = 0) {
			const std::string_view file_pathstr(file_path_new);
#if NIHILUS_PLATFORM_WINDOWS
			if (file_pathstr.empty()) {
				return;
			}
			file_handle = CreateFileA(file_pathstr.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
			if (file_handle == INVALID_HANDLE_VALUE) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to open file", location>::impl(format_win_error(GetLastError()));
			}
			LARGE_INTEGER file_size_new;
			if (!GetFileSizeEx(file_handle, &file_size_new)) {
				CloseHandle(file_handle);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to get file size", location>::impl(format_win_error(GetLastError()));
			}
			uint64_t file_size = static_cast<uint64_t>(file_size_new.QuadPart);
			if (file_size == 0) {
				CloseHandle(file_handle);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Cannot map empty file", location>::impl("");
			}
			if (file_offset >= file_size) {
				CloseHandle(file_handle);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Offset exceeds file size", location>::impl("");
			}
			SYSTEM_INFO sys_info;
			GetSystemInfo(&sys_info);
			uint64_t aligned_offset	   = (file_offset / sys_info.dwAllocationGranularity) * sys_info.dwAllocationGranularity;
			uint64_t offset_adjustment = file_offset - aligned_offset;
			mapped_size				   = file_size - file_offset;
			mapping_handle			   = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
			if (mapping_handle == nullptr) {
				CloseHandle(file_handle);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to create file mapping", location>::impl(format_win_error(GetLastError()));
			}
			void* raw_mapped_data = MapViewOfFile(mapping_handle, FILE_MAP_READ, static_cast<DWORD>(aligned_offset >> 32), static_cast<DWORD>(aligned_offset & 0xFFFFFFFF),
				mapped_size + offset_adjustment);
			if (raw_mapped_data == nullptr) {
				CloseHandle(mapping_handle);
				CloseHandle(file_handle);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to map view of file", location>::impl(format_win_error(GetLastError()));
			}
			mapped_data = static_cast<uint8_t*>(raw_mapped_data) + offset_adjustment;
			if (reinterpret_cast<std::uintptr_t>(mapped_data) % 32 != 0) {
				UnmapViewOfFile(raw_mapped_data);
				CloseHandle(mapping_handle);
				CloseHandle(file_handle);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Memory mapping failed to achieve required SIMD alignment", location>::impl("");
			}
#else
			file_descriptor = open(file_pathstr.data(), O_RDONLY);
			if (file_descriptor == -1) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to open file", location>::impl(std::string(std::strerror(errno)));
			}
			struct stat file_stat;
			if (fstat(file_descriptor, &file_stat) == -1) {
				close(file_descriptor);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to get file statistics", location>::impl(std::string(std::strerror(errno)));
			}
			uint64_t file_size = static_cast<uint64_t>(file_stat.st_size);
			if (file_size == 0) {
				close(file_descriptor);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Cannot map empty file", location>::impl("");
			}
			if (file_offset >= file_size) {
				close(file_descriptor);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Offset exceeds file size", location>::impl("");
			}
			uint64_t page_size		   = static_cast<uint64_t>(getpagesize());
			uint64_t aligned_offset	   = (file_offset / page_size) * page_size;
			uint64_t offset_adjustment = file_offset - aligned_offset;
			mapped_size				   = file_size - file_offset;
			int32_t flags			   = MAP_SHARED;
	#if NIHILUS_PLATFORM_MAC
			fcntl(file_descriptor, F_RDAHEAD, 1);
	#else
			posix_fadvise(file_descriptor, static_cast<int64_t>(aligned_offset), static_cast<int64_t>(mapped_size + offset_adjustment), POSIX_FADV_SEQUENTIAL);
	#endif
			uint64_t aligned_map_size = ((mapped_size + offset_adjustment + 32ull - 1ull) / 32ull) * 32ull;
			void* raw_mapped_data	  = mmap(nullptr, aligned_map_size, PROT_READ, flags, file_descriptor, static_cast<int64_t>(aligned_offset));
			if (raw_mapped_data == MAP_FAILED) {
				close(file_descriptor);
				mapped_data					   = nullptr;
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Failed to memory map file", location>::impl(std::string(std::strerror(errno)));
			}
			mapped_data = static_cast<uint8_t*>(raw_mapped_data) + offset_adjustment;
			if (reinterpret_cast<std::uintptr_t>(mapped_data) % 32 != 0) {
				munmap(raw_mapped_data, aligned_map_size);
				close(file_descriptor);
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Memory mapping failed to achieve required SIMD alignment", location>::impl("");
			}
	#if NIHILUS_PLATFORM_MAC
			madvise(raw_mapped_data, aligned_map_size, MADV_SEQUENTIAL);
	#endif
			mapped_fragments.emplace_back(0, mapped_size);
#endif
#if NIHILUS_PLATFORM_WINDOWS
			if (mapped_data && mapped_size > 0) {
	#if NIHILUS_CUDA_ENABLED
				cudaError_t result = cudaHostRegister(mapped_data, mapped_size, cudaHostRegisterReadOnly);
				if (result == cudaSuccess) {
					if constexpr (config.device_type == device_types::gpu) {
						cudaStreamCreateWithPriority(&transfer_stream, cudaStreamNonBlocking, 0);
					}
				} else {
					VirtualLock(mapped_data, mapped_size);
				}
	#else
				VirtualLock(mapped_data, mapped_size);
	#endif
			}
#else
			if (mapped_data && mapped_size > 0) {
	#if NIHILUS_CUDA_ENABLED
				cudaError_t result = cudaHostRegister(mapped_data, mapped_size, cudaHostRegisterReadOnly);
				if (result == cudaSuccess) {
					if constexpr (config.device_type == device_types::gpu) {
						cudaStreamCreateWithPriority(&transfer_stream, cudaStreamNonBlocking, 0);
					}
				} else {
					mlock(mapped_data, mapped_size);
				}
	#else
				mlock(mapped_data, mapped_size);
	#endif
				posix_madvise(mapped_data, mapped_size, POSIX_MADV_WILLNEED);
			}
#endif
		}

		NIHILUS_INLINE memory_mapped_file(memory_mapped_file&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE memory_mapped_file& operator=(memory_mapped_file&& other) noexcept {
			if (this != &other) {
				std::swap(mapped_data, other.mapped_data);
				std::swap(mapped_size, other.mapped_size);
#if NIHILUS_CUDA_ENABLED
				std::swap(transfer_stream, other.transfer_stream);
#endif
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
			return mapped_size;
		}

		NIHILUS_INLINE ~memory_mapped_file() {
#if NIHILUS_CUDA_ENABLED
			if (mapped_data) {
				cudaHostUnregister(mapped_data);
			}
			if (transfer_stream) {
				cudaStreamDestroy(transfer_stream);
				transfer_stream = nullptr;
			}
#endif

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
			mapped_size = 0;
		}

	  protected:
		uint64_t mapped_size{};
		void* mapped_data{};

#if NIHILUS_CUDA_ENABLED
		[[no_unique_address]] std::conditional_t<config.device_type == device_types::gpu, cudaStream_t, int8_t> transfer_stream{};
#endif

#if NIHILUS_PLATFORM_WINDOWS
		HANDLE mapping_handle{};
		HANDLE file_handle{};
#else
		aligned_vector<std::pair<uint64_t, uint64_t>> mapped_fragments{};
		int32_t file_descriptor{};
#endif
	};

}
