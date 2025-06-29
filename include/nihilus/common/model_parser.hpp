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
#include <nihilus/common/model_graph_data.hpp>
#include <nihilus/common/tokenizer.hpp>
#include <nihilus/common/debugging_io.hpp>
#include <nihilus/common/model_traits.hpp>

#if defined(NIHILUS_PLATFORM_WINDOWS)
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

	enum class gguf_metadata_value_type : uint32_t {
		GGUF_METADATA_VALUE_TYPE_UINT8	 = 0,
		GGUF_METADATA_VALUE_TYPE_INT8	 = 1,
		GGUF_METADATA_VALUE_TYPE_UINT16	 = 2,
		GGUF_METADATA_VALUE_TYPE_INT16	 = 3,
		GGUF_METADATA_VALUE_TYPE_UINT32	 = 4,
		GGUF_METADATA_VALUE_TYPE_INT32	 = 5,
		GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
		GGUF_METADATA_VALUE_TYPE_BOOL	 = 7,
		GGUF_METADATA_VALUE_TYPE_STRING	 = 8,
		GGUF_METADATA_VALUE_TYPE_ARRAY	 = 9,
		GGUF_METADATA_VALUE_TYPE_UINT64	 = 10,
		GGUF_METADATA_VALUE_TYPE_INT64	 = 11,
		GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
		GGUF_METADATA_VALUE_TYPE_UNSET	 = 13,
	};

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
		}

		NIHILUS_FORCE_INLINE ~memory_mapped_file() {
			unmap_file();
		}

		NIHILUS_FORCE_INLINE memory_mapped_file(memory_mapped_file&& other) noexcept
			: file_path_(other.file_path_), mapped_data_(other.mapped_data_), file_size_(other.file_size_)
#ifdef NIHILUS_PLATFORM_WINDOWS
			  ,
			  file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
			  ,
			  file_descriptor_(other.file_descriptor_), mapped_fragments_(std::move(other.mapped_fragments_))
#endif
		{
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

		NIHILUS_FORCE_INLINE bool is_valid() const noexcept {
			return mapped_data_ != nullptr;
		}

		NIHILUS_FORCE_INLINE std::string_view file_path() const noexcept {
			return file_path_;
		}

		NIHILUS_FORCE_INLINE void unmap_fragment(uint64_t first_byte, uint64_t last_byte) {
			unmap_fragment_impl(first_byte, last_byte);
		}

		NIHILUS_FORCE_INLINE void lock_memory() {
			lock_memory_impl();
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
			std::string file_path_str(file_path);

			file_handle_ = CreateFileA(file_path_str.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

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
				throw std::runtime_error(std::string{ "Failed to create file mapping: " } + format_win_error(GetLastError()));
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
			std::string file_path_str(file_path);

			file_descriptor_ = open(file_path_str.c_str(), O_RDONLY);
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
				if (munmap(static_cast<char*>(mapped_data_) + frag.first, frag.second - frag.first) != 0) {
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

		NIHILUS_FORCE_INLINE void unmap_fragment_impl(uint64_t first, uint64_t last) {
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

			void* unmap_addr	= static_cast<char*>(mapped_data_) + first;
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

		NIHILUS_FORCE_INLINE void lock_memory_impl() {
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

	struct stream_iterator {
		memory_mapped_file* file{};
		uint64_t current_index = 0;
		uint64_t length		   = 0;
		bool valid			   = true;

		NIHILUS_FORCE_INLINE stream_iterator(memory_mapped_file* s) : file(s), length{ file->size() } {};

		template<typename value_type> NIHILUS_FORCE_INLINE value_type read() {
			value_type dst{};
			std::memcpy(&dst, static_cast<uint8_t*>(file->data()) + current_index, sizeof(value_type));
			current_index += sizeof(value_type);
			return dst;
		}

		NIHILUS_FORCE_INLINE bool read_bytes_to_pointer(void* dst, const uint64_t size) {
			std::memcpy(dst, static_cast<uint8_t*>(file->data()) + current_index, size);
			current_index += size;
			return true;
		}

		NIHILUS_FORCE_INLINE bool read_bytes_to_pointer(void* dst, const uint64_t size, uint64_t offset) {
			std::memcpy(dst, static_cast<uint8_t*>(file->data()) + offset, size);
			return true;
		}

		NIHILUS_FORCE_INLINE bool map_pointer(void* dst, const uint64_t offset) {
			*reinterpret_cast<void**>(dst) = reinterpret_cast<uint8_t*>(file->data()) + offset;
			return true;
		}

		template<typename value_type = uint8_t> NIHILUS_FORCE_INLINE bool has_bytes(uint64_t size = sizeof(value_type)) const {
			return (current_index + size <= length);
		}
	};

	template<typename value_type, auto...> struct value_reader {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			if (input.has_bytes<value_type>()) {
				return input.read<value_type>();
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
		}
	};

	using gguf_string_t = std::string;

	struct gguf_array_t;

	using gguf_metadata_value_variant = std::variant<float, uint64_t, int64_t, double, bool, gguf_string_t, gguf_array_t*>;

	struct gguf_metadata_value_t {
		gguf_metadata_value_t() noexcept = default;
		gguf_metadata_value_t& operator=(const gguf_metadata_value_t& other) noexcept;
		gguf_metadata_value_t(const gguf_metadata_value_t& other) noexcept {
			*this = other;
		};
		gguf_metadata_value_t(const gguf_metadata_value_variant& other) noexcept;
		gguf_metadata_value_variant value{};
		~gguf_metadata_value_t();
	};

	struct gguf_array_t {
		std::vector<gguf_metadata_value_t> array{};
		gguf_metadata_value_type type{};
	};

	gguf_metadata_value_t::~gguf_metadata_value_t() {
		if (std::holds_alternative<gguf_array_t*>(value)) {
			if (std::get<gguf_array_t*>(value)) {
				delete std::get<gguf_array_t*>(value);
			}
		}
	}

	gguf_metadata_value_t& gguf_metadata_value_t::operator=(const gguf_metadata_value_t& other) noexcept {
		if (std::holds_alternative<float>(other.value)) {
			value.emplace<float>(std::get<float>(other.value));
		} else if (std::holds_alternative<uint64_t>(other.value)) {
			value.emplace<uint64_t>(std::get<uint64_t>(other.value));
		} else if (std::holds_alternative<int64_t>(other.value)) {
			value.emplace<int64_t>(std::get<int64_t>(other.value));
		} else if (std::holds_alternative<double>(other.value)) {
			value.emplace<double>(std::get<double>(other.value));
		} else if (std::holds_alternative<bool>(other.value)) {
			value.emplace<bool>(std::get<bool>(other.value));
		} else if (std::holds_alternative<gguf_string_t>(other.value)) {
			value.emplace<gguf_string_t>(std::get<gguf_string_t>(other.value));
		} else if (std::holds_alternative<gguf_array_t*>(other.value)) {
			if (std::holds_alternative<gguf_array_t*>(value)) {
				if (std::get<gguf_array_t*>(value)) {
					delete std::get<gguf_array_t*>(value);
				}
			}
			value.emplace<gguf_array_t*>(new gguf_array_t{ *std::get<gguf_array_t*>(other.value) });
		}
		return *this;
	};

	gguf_metadata_value_t::gguf_metadata_value_t(const gguf_metadata_value_variant& other) noexcept {
		if (std::holds_alternative<float>(other)) {
			value.emplace<float>(std::get<float>(other));
		} else if (std::holds_alternative<uint64_t>(other)) {
			value.emplace<uint64_t>(std::get<uint64_t>(other));
		} else if (std::holds_alternative<int64_t>(other)) {
			value.emplace<int64_t>(std::get<int64_t>(other));
		} else if (std::holds_alternative<double>(other)) {
			value.emplace<double>(std::get<double>(other));
		} else if (std::holds_alternative<bool>(other)) {
			value.emplace<bool>(std::get<bool>(other));
		} else if (std::holds_alternative<gguf_string_t>(other)) {
			value.emplace<gguf_string_t>(std::get<gguf_string_t>(other));
		} else if (std::holds_alternative<gguf_array_t*>(other)) {
			value.emplace<gguf_array_t*>(new gguf_array_t{ *std::get<gguf_array_t*>(other) });
		}
	};

	template<> struct value_reader<gguf_string_t> {
		NIHILUS_FORCE_INLINE static std::string gather_value(stream_iterator& input) {
			uint64_t length = value_reader<uint64_t>::gather_value(input);
			if (!input.has_bytes<char>(length)) {
				throw std::runtime_error("Sorry, but that index is out of range!");
			}
			std::string result(length, '\0');
			result.resize(length);
			input.read_bytes_to_pointer(result.data(), length);
			return result;
		}
	};

	template<> struct value_reader<gguf_array_t> {
		NIHILUS_FORCE_INLINE static gguf_array_t gather_value(stream_iterator& input);
	};

	template<> struct value_reader<gguf_metadata_value_variant> {
		NIHILUS_INLINE static gguf_metadata_value_variant gather_value(stream_iterator& input, gguf_metadata_value_type type) {
			gguf_metadata_value_variant value{};
			switch (type) {
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT8: {
					value.emplace<int64_t>(value_reader<int8_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT16: {
					value.emplace<int64_t>(value_reader<int16_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT32: {
					value.emplace<int64_t>(value_reader<int32_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT64: {
					value.emplace<int64_t>(value_reader<int64_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT8: {
					value.emplace<uint64_t>(value_reader<uint8_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT16: {
					value.emplace<uint64_t>(value_reader<uint16_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32: {
					value.emplace<uint64_t>(value_reader<uint32_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT64: {
					value.emplace<uint64_t>(value_reader<uint64_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_BOOL: {
					value.emplace<bool>(value_reader<bool>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_FLOAT32: {
					value.emplace<float>(value_reader<float>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_FLOAT64: {
					value.emplace<double>(value_reader<double>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING: {
					value.emplace<gguf_string_t>(value_reader<gguf_string_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_ARRAY: {
					value.emplace<gguf_array_t*>(new gguf_array_t{ value_reader<gguf_array_t>::gather_value(input) });
					break;
				}
				case gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UNSET: {
					break;
				}
			}
			return value;
		}
	};

	gguf_array_t value_reader<gguf_array_t>::gather_value(stream_iterator& input) {
		gguf_metadata_value_type type{ value_reader<gguf_metadata_value_type>::gather_value(input) };
		uint64_t length{ value_reader<uint64_t>::gather_value(input) };
		constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
		if (length > MAX_ARRAY_LENGTH) {
			throw std::runtime_error{ "Array length exceeds maximum allowed size!" };
		}
		gguf_array_t value{};
		value.type = type;
		value.array.reserve(length);
		for (uint64_t x = 0; x < length; ++x) {
			value.array.emplace_back(value_reader<gguf_metadata_value_variant>::gather_value(input, type));
		}
		return value;
	}

	struct gguf_metadata_kv_t;

	struct gguf_metadata_kv_t {
		gguf_metadata_value_type value_type{};

		gguf_metadata_value_t value{};

		NIHILUS_FORCE_INLINE operator bool() const {
			return std::get<bool>(value.value);
		}

		NIHILUS_FORCE_INLINE operator int64_t() const {
			return std::get<int64_t>(value.value);
		}

		NIHILUS_FORCE_INLINE operator uint64_t() const {
			return std::get<uint64_t>(value.value);
		}

		NIHILUS_FORCE_INLINE operator gguf_string_t() const {
			return std::get<gguf_string_t>(value.value);
		}

		NIHILUS_FORCE_INLINE operator gguf_array_t() const {
			return *std::get<gguf_array_t*>(value.value);
		}

		NIHILUS_FORCE_INLINE operator float() const {
			return std::get<float>(value.value);
		}

		NIHILUS_FORCE_INLINE operator double() const {
			return std::get<double>(value.value);
		}
	};

	template<> struct value_reader<gguf_metadata_kv_t> {
		NIHILUS_FORCE_INLINE static gguf_metadata_kv_t gather_value(stream_iterator& input) {
			gguf_metadata_kv_t value{};
			value.value_type  = value_reader<gguf_metadata_value_type>::gather_value(input);
			value.value.value = value_reader<gguf_metadata_value_variant>::gather_value(input, value.value_type);
			return value;
		}
	};

	struct gguf_header_t {
		std::map<std::string, gguf_metadata_kv_t> metadata_kv{};
		uint64_t metadata_kv_count{};
		uint64_t tensor_count{};
		uint32_t version{};
		uint32_t magic{};
	};

	template<typename value_type> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, value_type& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<value_type>(v)) {
			out = std::get<value_type>(v);
		}
	};

	template<> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, uint32_t& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<uint64_t>(v)) {
			out = static_cast<uint32_t>(std::get<uint64_t>(v));
		}
	};

	template<> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, uint16_t& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<uint64_t>(v)) {
			out = static_cast<uint16_t>(std::get<uint64_t>(v));
		}
	};

	template<> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, uint8_t& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<uint64_t>(v)) {
			out = static_cast<uint8_t>(std::get<uint64_t>(v));
		}
	};
	template<> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, int32_t& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<int64_t>(v)) {
			out = static_cast<int32_t>(std::get<uint64_t>(v));
		}
	};

	template<> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, int16_t& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<int64_t>(v)) {
			out = static_cast<int16_t>(std::get<uint64_t>(v));
		}
	};

	template<> NIHILUS_FORCE_INLINE void gather_scalar(const std::string& key, int8_t& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<int64_t>(v)) {
			out = static_cast<int8_t>(std::get<uint64_t>(v));
		}
	};

	template<typename value_type>
	NIHILUS_FORCE_INLINE void gather_array(const std::string& key, std::vector<value_type>& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv);

	template<int_type value_type>
	NIHILUS_FORCE_INLINE void gather_array(const std::string& key, std::vector<value_type>& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(static_cast<value_type>(std::get<int64_t>(value.value)));
			}
		}
	};

	template<uint_type value_type>
	NIHILUS_FORCE_INLINE void gather_array(const std::string& key, std::vector<value_type>& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(static_cast<value_type>(std::get<uint64_t>(value.value)));
			}
		}
	};

	template<typename value_type>
	NIHILUS_FORCE_INLINE void gather_array(const std::string& key, std::vector<value_type>& out, const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(std::get<value_type>(value.value));
			}
		}
	};

	NIHILUS_FORCE_INLINE void print_variant(auto variant) {
		if (std::holds_alternative<float>(variant)) {
			std::cout << "Value: " << std::get<float>(variant) << std::endl;
		} else if (std::holds_alternative<uint64_t>(variant)) {
			std::cout << "Value: " << std::get<uint64_t>(variant) << std::endl;
		} else if (std::holds_alternative<int64_t>(variant)) {
			std::cout << "Value: " << std::get<int64_t>(variant) << std::endl;
		} else if (std::holds_alternative<double>(variant)) {
			std::cout << "Value: " << std::get<double>(variant) << std::endl;
		} else if (std::holds_alternative<bool>(variant)) {
			std::cout << "Value: " << std::get<bool>(variant) << std::endl;
		} else if (std::holds_alternative<gguf_string_t>(variant)) {
			std::cout << "Value: " << std::get<gguf_string_t>(variant) << std::endl;
		} else if (std::holds_alternative<gguf_array_t*>(variant)) {
		}
	}

	NIHILUS_FORCE_INLINE vocab_types string_to_vocab_type(const std::string& vocab_type_str) {
		if (vocab_type_str == "llama")
			return vocab_types::spm;
		if (vocab_type_str == "gpt2")
			return vocab_types::bpe;
		if (vocab_type_str == "bert")
			return vocab_types::wpm;
		if (vocab_type_str == "t5")
			return vocab_types::spm;
		if (vocab_type_str == "gpt-neox")
			return vocab_types::bpe;
		if (vocab_type_str == "falcon")
			return vocab_types::bpe;
		if (vocab_type_str == "mpt")
			return vocab_types::bpe;
		if (vocab_type_str == "starcoder")
			return vocab_types::bpe;
		if (vocab_type_str == "gpt-j")
			return vocab_types::bpe;
		if (vocab_type_str == "refact")
			return vocab_types::bpe;
		if (vocab_type_str == "command-r")
			return vocab_types::bpe;
		if (vocab_type_str == "qwen2")
			return vocab_types::bpe;
		if (vocab_type_str == "olmo")
			return vocab_types::bpe;
		if (vocab_type_str == "dbrx")
			return vocab_types::bpe;
		if (vocab_type_str == "minicp")
			return vocab_types::bpe;
		if (vocab_type_str == "tekken")
			return vocab_types::bpe;
		if (vocab_type_str == "smollm")
			return vocab_types::bpe;
		if (vocab_type_str == "viking")
			return vocab_types::bpe;
		if (vocab_type_str == "jais")
			return vocab_types::bpe;
		if (vocab_type_str == "chatglm")
			return vocab_types::spm;
		if (vocab_type_str == "baichuan")
			return vocab_types::spm;
		if (vocab_type_str == "xverse")
			return vocab_types::spm;
		if (vocab_type_str == "internlm2")
			return vocab_types::spm;
		if (vocab_type_str == "minicpm")
			return vocab_types::spm;
		if (vocab_type_str == "gemma")
			return vocab_types::spm;
		if (vocab_type_str == "gemma2")
			return vocab_types::spm;
		if (vocab_type_str == "phi3")
			return vocab_types::spm;
		if (vocab_type_str == "paligemma")
			return vocab_types::spm;
		if (vocab_type_str == "bloom")
			return vocab_types::bpe;
		if (vocab_type_str == "stablelm")
			return vocab_types::bpe;
		if (vocab_type_str == "qwen")
			return vocab_types::bpe;
		if (vocab_type_str == "plamo")
			return vocab_types::spm;
		if (vocab_type_str == "codeshell")
			return vocab_types::bpe;
		if (vocab_type_str == "orion")
			return vocab_types::spm;
		if (vocab_type_str == "internlm")
			return vocab_types::spm;
		if (vocab_type_str == "minicpm3")
			return vocab_types::spm;
		if (vocab_type_str == "rwkv")
			return vocab_types::rwkv;
		if (vocab_type_str == "command")
			return vocab_types::bpe;
		if (vocab_type_str == "deci")
			return vocab_types::bpe;
		if (vocab_type_str == "granite")
			return vocab_types::bpe;
		if (vocab_type_str == "chameleon")
			return vocab_types::bpe;

		return vocab_types::none;
	}

	template<> struct value_reader<construction_parameters<model_arches::llama>, model_arches::llama> {
		NIHILUS_FORCE_INLINE static construction_parameters<model_arches::llama> gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv) {
			construction_parameters<model_arches::llama> value{};
			std::string architecture{};
			if (metadata_kv.contains("general.architecture")) {
				architecture = metadata_kv.at("general.architecture").operator gguf_string_t();
			}
			gather_scalar(architecture + ".rope.dimension_count", value.rope_dimension_count, metadata_kv);
			gather_scalar(architecture + ".feed_forward_length", value.feed_forward_length, metadata_kv);
			gather_scalar(architecture + ".embedding_length", value.embedding_length, metadata_kv);
			gather_scalar(architecture + ".context_length", value.context_length, metadata_kv);
			gather_scalar(architecture + ".attention.head_count_kv", value.head_count_kv, metadata_kv);
			gather_scalar(architecture + ".block_count", value.block_count, metadata_kv);
			gather_scalar(architecture + ".attention.head_count", value.head_count, metadata_kv);
			gather_scalar(architecture + ".vocab_size", value.vocab_size, metadata_kv);
			gather_scalar(architecture + ".rope.type", value.rope_type, metadata_kv);
			gather_scalar(architecture + ".expert_count", value.n_expert, metadata_kv);
			gather_scalar(architecture + ".expert_used_count", value.n_expert_used, metadata_kv);
			gather_scalar(architecture + ".rope.freq_base", value.rope_freq_base, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.factor", value.rope_freq_scale, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.attn_factor", value.rope_attn_factor, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.beta_fast", value.rope_beta_fast, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.beta_slow", value.rope_beta_slow, metadata_kv);
			gather_scalar(architecture + ".attention.layer_norm_rms_epsilon", value.rms_norm_epsilon, metadata_kv);
			gather_scalar(architecture + ".attention.scale", value.f_attention_scale, metadata_kv);
			gather_scalar(architecture + ".rope.scaling.ext_factor", value.rope_ext_factor, metadata_kv);

			return value;
		}
	};

	template<typename derived_type> struct value_reader<vocab<model_arches::llama, vocab_types::bpe, derived_type>> {
		NIHILUS_FORCE_INLINE static void gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv,
			vocab<model_arches::llama, vocab_types::bpe, derived_type>& tokenizer) {
			std::string tokenizer_model;
			std::string tokenizer_pre;

			gather_scalar("tokenizer.ggml.model", tokenizer_model, metadata_kv);
			gather_scalar("tokenizer.ggml.pre", tokenizer_pre, metadata_kv);
			gather_scalar("tokenizer.ggml.token_type_count", tokenizer.n_token_types, metadata_kv);

			if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
				tokenizer.special_bos_id  = token_null;
				tokenizer.special_eos_id  = token_null;
				tokenizer.special_unk_id  = token_null;
				tokenizer.special_sep_id  = token_null;
				tokenizer.special_pad_id  = token_null;
				tokenizer.special_mask_id = token_null;
				tokenizer.linefeed_id	  = token_null;
				uint32_t n_tokens		  = 0;
				gather_scalar("general.vocab_size", n_tokens, metadata_kv);
				if (n_tokens != 0) {
					tokenizer.id_to_token.resize(n_tokens);
				}
				return;
			}

			if (tokenizer_model == "llama") {
				tokenizer.special_bos_id  = 1;
				tokenizer.special_eos_id  = 2;
				tokenizer.special_unk_id  = 0;
				tokenizer.special_sep_id  = token_null;
				tokenizer.special_pad_id  = token_null;
				tokenizer.special_mask_id = token_null;
			} else if (tokenizer_model == "bert") {
				tokenizer.special_bos_id  = 101;
				tokenizer.special_eos_id  = token_null;
				tokenizer.special_unk_id  = 100;
				tokenizer.special_sep_id  = 102;
				tokenizer.special_pad_id  = 0;
				tokenizer.special_mask_id = 103;
			} else if (tokenizer_model == "gpt2") {
				std::vector<std::string> merges;
				gather_array("tokenizer.ggml.merges", merges, metadata_kv);

				for (size_t i = 0; i < merges.size(); ++i) {
					const std::string& merge_str = merges[i];
					size_t space_pos			 = merge_str.find(' ', 1);
					if (space_pos != std::string::npos) {
						std::string first  = merge_str.substr(0, space_pos);
						std::string second = merge_str.substr(space_pos + 1);

						tokenizer.bpe_ranks.emplace(std::make_pair(first, second), static_cast<int32_t>(i));
					}
				}

				tokenizer.special_bos_id  = 11;
				tokenizer.special_eos_id  = 11;
				tokenizer.special_unk_id  = token_null;
				tokenizer.special_sep_id  = token_null;
				tokenizer.special_pad_id  = token_null;
				tokenizer.special_mask_id = token_null;
			} else if (tokenizer_model == "t5") {
				tokenizer.special_bos_id  = token_null;
				tokenizer.special_eos_id  = 1;
				tokenizer.special_unk_id  = 2;
				tokenizer.special_sep_id  = token_null;
				tokenizer.special_pad_id  = 0;
				tokenizer.special_mask_id = token_null;

				std::vector<uint8_t> precompiled_data;
				gather_array("tokenizer.ggml.precompiled_charsmap", precompiled_data, metadata_kv);
				if (!precompiled_data.empty()) {
					tokenizer.precompiled_charsmap.assign(precompiled_data.begin(), precompiled_data.end());
				}
			} else if (tokenizer_model == "rwkv") {
				tokenizer.special_bos_id = token_null;
				tokenizer.special_eos_id = token_null;
				tokenizer.special_unk_id = token_null;
				tokenizer.special_sep_id = token_null;
				tokenizer.special_pad_id = token_null;
			} else {
				throw std::runtime_error("Unknown tokenizer model: " + tokenizer_model);
			}

			if (tokenizer_model == "gpt2" || tokenizer_model == "bert" || tokenizer_model == "t5") {
				tokenizer.add_space_prefix = false;
				tokenizer.clean_spaces	   = true;

				if (tokenizer_pre.empty()) {
					tokenizer.pre_type = vocab_pre_types::default_pre;
				} else if (tokenizer_pre == "default") {
					tokenizer.pre_type = vocab_pre_types::default_pre;
				} else if (tokenizer_pre == "llama3" || tokenizer_pre == "llama-v3" || tokenizer_pre == "llama-bpe" || tokenizer_pre == "falcon3") {
					tokenizer.pre_type		= vocab_pre_types::llama3;
					tokenizer.ignore_merges = true;
					tokenizer.add_bos		= true;
				} else if (tokenizer_pre == "deepseek-llm") {
					tokenizer.pre_type	   = vocab_pre_types::deepseek_llm;
					tokenizer.clean_spaces = false;
				} else if (tokenizer_pre == "deepseek-coder") {
					tokenizer.pre_type	   = vocab_pre_types::deepseek_coder;
					tokenizer.clean_spaces = false;
				} else if (tokenizer_pre == "deepseek-v3") {
					tokenizer.pre_type	   = vocab_pre_types::deepseek3_llm;
					tokenizer.clean_spaces = false;
				} else if (tokenizer_pre == "falcon") {
					tokenizer.pre_type = vocab_pre_types::falcon;
				} else if (tokenizer_pre == "mpt") {
					tokenizer.pre_type = vocab_pre_types::mpt;
				} else if (tokenizer_pre == "starcoder") {
					tokenizer.pre_type = vocab_pre_types::starcoder;
				} else if (tokenizer_pre == "gpt-2" || tokenizer_pre == "phi-2" || tokenizer_pre == "jina-es" || tokenizer_pre == "jina-de" || tokenizer_pre == "gigachat" ||
					tokenizer_pre == "roberta-bpe") {
					tokenizer.pre_type = vocab_pre_types::gpt2;
				} else if (tokenizer_pre == "refact") {
					tokenizer.pre_type = vocab_pre_types::refact;
				} else if (tokenizer_pre == "command-r") {
					tokenizer.pre_type	   = vocab_pre_types::command_r;
					tokenizer.clean_spaces = false;
				} else if (tokenizer_pre == "qwen2") {
					tokenizer.pre_type	   = vocab_pre_types::qwen2;
					tokenizer.clean_spaces = false;
				} else if (tokenizer_pre == "tekken") {
					tokenizer.pre_type		= vocab_pre_types::tekken;
					tokenizer.clean_spaces	= false;
					tokenizer.ignore_merges = true;
					tokenizer.add_bos		= true;
				} else if (tokenizer_pre == "smollm") {
					tokenizer.pre_type	   = vocab_pre_types::smollm;
					tokenizer.clean_spaces = false;
				} else if (tokenizer_pre == "chameleon") {
					tokenizer.pre_type	   = vocab_pre_types::chameleon;
					tokenizer.add_bos	   = true;
					tokenizer.clean_spaces = false;
				} else {
					throw std::runtime_error("Unknown pre-tokenizer type: " + tokenizer_pre);
				}
			} else {
				tokenizer.pre_type = vocab_pre_types::default_pre;
				if (tokenizer_model == "llama") {
					tokenizer.add_space_prefix = true;
					tokenizer.clean_spaces	   = false;
					tokenizer.add_bos		   = true;
					tokenizer.add_eos		   = false;
				} else if (tokenizer_model == "bert") {
					tokenizer.add_space_prefix = false;
					tokenizer.clean_spaces	   = true;
					tokenizer.add_bos		   = true;
					tokenizer.add_eos		   = false;
				} else if (tokenizer_model == "t5") {
					tokenizer.add_bos = false;
					tokenizer.add_eos = true;
				} else if (tokenizer_model == "rwkv") {
					tokenizer.add_space_prefix = false;
					tokenizer.clean_spaces	   = false;
					tokenizer.add_bos		   = false;
					tokenizer.add_eos		   = false;
				}
			}

			gather_scalar("tokenizer.ggml.add_space_prefix", tokenizer.add_space_prefix, metadata_kv);
			gather_scalar("tokenizer.ggml.remove_extra_whitespace", tokenizer.remove_extra_whitespaces, metadata_kv);
			gather_scalar("tokenizer.ggml.add_bos", tokenizer.add_bos, metadata_kv);
			gather_scalar("tokenizer.ggml.add_eos", tokenizer.add_eos, metadata_kv);

			std::vector<std::string> tokens;
			gather_array("tokenizer.ggml.tokens", tokens, metadata_kv);

			std::vector<float> scores;
			gather_array("tokenizer.ggml.scores", scores, metadata_kv);

			std::vector<int32_t> token_types;
			gather_array("tokenizer.ggml.token_type", token_types, metadata_kv);

			uint32_t n_tokens = static_cast<uint32_t>(tokens.size());
			tokenizer.id_to_token.resize(n_tokens);

			for (uint32_t i = 0; i < n_tokens; i++) {
				std::string word = tokens[i];
				if (word.empty()) {
					word = "[EMPTY_" + std::to_string(i) + "]";
				}

				tokenizer.token_to_id[word] = i;
				tokenizer.max_token_len		= std::max(tokenizer.max_token_len, static_cast<int32_t>(word.size()));

				auto& token_data = tokenizer.id_to_token[i];
				token_data.text	 = std::move(word);
				token_data.score = (i < scores.size()) ? scores[i] : 0.0f;
				token_data.att	 = tokens::normal;

				if (i < token_types.size()) {
					switch (token_types[i]) {
						case 0:
							token_data.att = tokens::unknown;
							break;
						case 1:
							token_data.att = tokens::unused;
							break;
						case 2:
							token_data.att = tokens::normal;
							break;
						case 3:
							token_data.att = tokens::control;
							break;
						case 4:
							token_data.att = tokens::user_defined;
							break;
						case 5:
							token_data.att = tokens::byte;
							break;
						default:
							token_data.att = tokens::undefined;
							break;
					}
				}
			}

			gather_scalar("tokenizer.ggml.bos_token_id", tokenizer.special_bos_id, metadata_kv);
			gather_scalar("tokenizer.ggml.eos_token_id", tokenizer.special_eos_id, metadata_kv);
			gather_scalar("tokenizer.ggml.unknown_token_id", tokenizer.special_unk_id, metadata_kv);
			gather_scalar("tokenizer.ggml.separator_token_id", tokenizer.special_sep_id, metadata_kv);
			gather_scalar("tokenizer.ggml.padding_token_id", tokenizer.special_pad_id, metadata_kv);

			for (const auto& [text, id]: tokenizer.token_to_id) {
				if (tokenizer.special_eot_id == token_null) {
					if (text == "<|eot_id|>" || text == "<|im_end|>" || text == "<|end|>" || text == "<end_of_turn>" || text == "<|endoftext|>" || text == "< EOT >" ||
						text == "<｜end▁of▁sentence｜>") {
						tokenizer.special_eot_id	  = id;
						tokenizer.id_to_token[id].att = tokens::control;
					}
				}

				if (tokenizer.special_eom_id == token_null && text == "<|eom_id|>") {
					tokenizer.special_eom_id	  = id;
					tokenizer.id_to_token[id].att = tokens::control;
				}

				if (tokenizer.special_fim_pre_id == token_null) {
					if (text == "<|fim_prefix|>" || text == "<fim-prefix>" || text == "<｜fim▁begin｜>" || text == "<PRE>") {
						tokenizer.special_fim_pre_id  = id;
						tokenizer.id_to_token[id].att = tokens::control;
					}
				}

				if (tokenizer.special_fim_suf_id == token_null) {
					if (text == "<|fim_suffix|>" || text == "<fim-suffix>" || text == "<｜fim▁hole｜>" || text == "<SUF>") {
						tokenizer.special_fim_suf_id  = id;
						tokenizer.id_to_token[id].att = tokens::control;
					}
				}

				if (tokenizer.special_fim_mid_id == token_null) {
					if (text == "<|fim_middle|>" || text == "<fim-middle>" || text == "<｜fim▁end｜>" || text == "<MID>") {
						tokenizer.special_fim_mid_id  = id;
						tokenizer.id_to_token[id].att = tokens::control;
					}
				}
			}

			for (token id = 0; id < static_cast<token>(n_tokens); ++id) {
				if (static_cast<size_t>(tokenizer.id_to_token[id].att) &
					(static_cast<size_t>(tokens::control) | static_cast<size_t>(tokens::user_defined) | static_cast<size_t>(tokens::unused))) {
					tokenizer.cache_special_tokens.push_back(id);
				}
			}

			std::sort(tokenizer.cache_special_tokens.begin(), tokenizer.cache_special_tokens.end(), [&](token a, token b) {
				return tokenizer.id_to_token[a].text.size() > tokenizer.id_to_token[b].text.size();
			});

			tokenizer.special_eog_ids.clear();
			for (const auto& [text, id]: tokenizer.token_to_id) {
				if (text == "<|eot_id|>" || text == "<|im_end|>" || text == "<|end|>" || text == "<end_of_turn>" || text == "<|endoftext|>" || text == "<|eom_id|>" ||
					text == "< EOT >") {
					tokenizer.special_eog_ids.insert(id);
				}
			}

			if (tokenizer.special_eos_id != token_null) {
				tokenizer.special_eog_ids.insert(tokenizer.special_eos_id);
			}
			if (tokenizer.special_eot_id != token_null) {
				tokenizer.special_eog_ids.insert(tokenizer.special_eot_id);
			}
			if (tokenizer.special_eom_id != token_null) {
				tokenizer.special_eog_ids.insert(tokenizer.special_eom_id);
			}

			if (tokenizer_model == "llama") {
				tokenizer.init_tokenizer(vocab_types::spm);
			} else if (tokenizer_model == "gpt2") {
				tokenizer.init_tokenizer(vocab_types::bpe);
			} else if (tokenizer_model == "bert") {
				tokenizer.init_tokenizer(vocab_types::wpm);
			} else if (tokenizer_model == "t5") {
				tokenizer.init_tokenizer(vocab_types::ugm);
			} else if (tokenizer_model == "rwkv") {
				tokenizer.init_tokenizer(vocab_types::rwkv);
			}
		}
	};

	template<model_config config, typename derived_type, vocab_types vocab_type>
	struct value_reader<tokenizer<config, derived_type, model_arches::llama, vocab_type>, model_arches::llama> {
		NIHILUS_FORCE_INLINE static void gather_value(const std::map<std::string, gguf_metadata_kv_t>& metadata_kv,
			tokenizer<config, derived_type, model_arches::llama, vocab_type>& tokenizer) {
			gather_scalar("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id, metadata_kv);
			gather_scalar("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id, metadata_kv);
			gather_scalar("tokenizer.chat_template", tokenizer.chat_template, metadata_kv);
			std::string vocab_type_str{};
			gather_scalar("tokenizer.ggml.model", vocab_type_str, metadata_kv);
			//tokenizer.vocab_type = string_to_vocab_type(vocab_type_str);
			gather_array("tokenizer.ggml.merges", tokenizer.merges, metadata_kv);
			gather_scalar("tokenizer.ggml.pre", tokenizer.pre, metadata_kv);
			gather_array("tokenizer.ggml.tokens", tokenizer.tokens, metadata_kv);
			gather_array("tokenizer.ggml.token_type", tokenizer.token_types, metadata_kv);
			return;
		}
	};

	template<> struct value_reader<gguf_header_t> {
		NIHILUS_FORCE_INLINE static gguf_header_t gather_value(stream_iterator& input) {
			gguf_header_t value{};
			value.magic = value_reader<uint32_t>::gather_value(input);
			if (value.magic != 0x46554747) {
				throw std::runtime_error{ "Sorry, but that magic value was incorrect!" };
			}
			value.version						  = value_reader<uint32_t>::gather_value(input);
			value.tensor_count					  = value_reader<uint64_t>::gather_value(input);
			value.metadata_kv_count				  = value_reader<uint64_t>::gather_value(input);
			constexpr uint64_t MAX_TENSOR_COUNT	  = 100000;
			constexpr uint64_t MAX_METADATA_COUNT = 10000;
			if (value.tensor_count > MAX_TENSOR_COUNT) {
				throw std::runtime_error{ "Tensor count exceeds reasonable maximum!" };
			}
			if (value.metadata_kv_count > MAX_METADATA_COUNT) {
				throw std::runtime_error{ "Metadata count exceeds reasonable maximum!" };
			}
			for (uint64_t x = 0; x < value.metadata_kv_count; ++x) {
				std::string new_string		  = value_reader<gguf_string_t>::gather_value(input);
				value.metadata_kv[new_string] = value_reader<gguf_metadata_kv_t>::gather_value(input);
			}
			return value;
		}
	};

	template<model_arches arch> struct string_to_op_type;

	template<> struct string_to_op_type<model_arches::llama> {
		NIHILUS_FORCE_INLINE static op_types impl(std::string_view input) noexcept {
			if (input == "token_embd.weight")
				return op_types::token_embd_weight;
			if (input == "rope_freqs.weight")
				return op_types::rope_freqs_weight;
			if (input == "output_norm.weight")
				return op_types::output_norm_weight;
			if (input == "output.weight")
				return op_types::output_weight;

			if (input.find(".attn_q.weight") != std::string_view::npos)
				return op_types::attn_q_weight;
			if (input.find(".attn_norm.weight") != std::string_view::npos)
				return op_types::attn_norm_weight;

			if (input.starts_with("blk.") && input.ends_with(".weight")) {
				auto second_dot = input.find('.', 4);
				if (second_dot != std::string_view::npos) {
					auto suffix = input.substr(second_dot + 1);

					if (suffix == "attn_q.weight")
						return op_types::attn_q_weight;
					if (suffix == "attn_norm.weight")
						return op_types::attn_norm_weight;
					if (suffix == "attn_k.weight")
						return op_types::attn_k_weight;
					if (suffix == "attn_v.weight")
						return op_types::attn_v_weight;
					if (suffix == "ffn_down.weight")
						return op_types::ffn_down_weight;
					if (suffix == "ffn_gate.weight")
						return op_types::ffn_gate_weight;
					if (suffix == "attn_output.weight")
						return op_types::attn_output_weight;
					if (suffix == "ffn_norm.weight")
						return op_types::ffn_norm_weight;
					if (suffix == "ffn_up.weight")
						return op_types::ffn_up_weight;
				}
			}

			return op_types::count;
		}
	};

	template<> struct value_reader<core_base_creation_data> {
		NIHILUS_FORCE_INLINE static core_base_creation_data gather_value(stream_iterator& input) {
			core_base_creation_data value{};
			value.name						  = value_reader<gguf_string_t>::gather_value(input);
			value.n_dimensions				  = value_reader<uint32_t>::gather_value(input);
			constexpr uint32_t MAX_DIMENSIONS = 8;
			if (value.n_dimensions > MAX_DIMENSIONS) {
				throw std::runtime_error{ "Tensor dimensions exceed maximum!" };
			}
			for (uint64_t x = 0; x < value.n_dimensions; ++x) {
				uint64_t dim					= value_reader<uint64_t>::gather_value(input);
				constexpr uint64_t MAX_DIM_SIZE = 1ULL << 32;
				if (dim > MAX_DIM_SIZE) {
					throw std::runtime_error{ "Tensor dimension size too large!" };
				}
				value.dimensions[x] = dim;
			}
			value.type	 = static_cast<data_types>(value_reader<uint32_t>::gather_value(input));
			value.offset = value_reader<uint64_t>::gather_value(input);
			return value;
		}
	};

	NIHILUS_FORCE_INLINE constexpr uint64_t parse_number(std::string_view str) noexcept {
		uint64_t result = 0;
		for (char c: str) {
			if (c >= '0' && c <= '9') {
				result = result * 10 + (c - '0');
			} else {
				break;
			}
		}
		return result;
	}

	NIHILUS_FORCE_INLINE uint64_t extract_layer_number(std::string_view name) noexcept {
		if NIHILUS_LIKELY (name[0] == 'c' && name.starts_with("cache_")) {
			for (uint64_t i = 7; i < name.size(); ++i) {
				if (name[i] == 'l' && i + 1 < name.size()) {
					return parse_number(name.substr(i + 1));
				}
			}
			return 0;
		}
		if NIHILUS_LIKELY (name[0] == 'b' && name.starts_with("blk.")) {
			uint64_t start = 4;
			uint64_t end   = name.find('.', start);
			if (end != std::string_view::npos) {
				return parse_number(name.substr(start, end - start));
			}
		}

		return 0;
	}

	NIHILUS_FORCE_INLINE bool operator<(const core_base_creation_data& lhs, const core_base_creation_data& rhs) noexcept {
		const uint64_t lhs_number{ extract_layer_number(lhs.name) };
		const uint64_t rhs_number{ extract_layer_number(rhs.name) };
		return lhs_number < rhs_number;
	}
	NIHILUS_FORCE_INLINE void sort_tensor_infos(std::vector<core_base_creation_data>& tensor_infos) noexcept {
		std::sort(tensor_infos.begin(), tensor_infos.end(), std::less<core_base_creation_data>{});
	}

	struct gguf_file_t {
		std::vector<core_base_creation_data> tensor_infos{};
		gguf_header_t header{};
	};

	NIHILUS_FORCE_INLINE uint64_t align_offset(uint64_t offset, uint64_t alignment) {
		alignment = alignment == 0 ? 1 : alignment;
		return offset + (alignment - (offset % alignment)) % alignment;
	}

	template<model_config config> struct model_parser_impl {};

	template<model_config config>
		requires((config.arch == model_arches::llama) && (config.format == model_format::gguf))
	struct model_parser_impl<config> {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;
		static_assert((std::endian::native == std::endian::little), "Sorry, but big-endian is not yet supported by the library");
		template<typename tokenizer_type> NIHILUS_FORCE_INLINE static model_graph_data<config> parse_model(
			array<array<void*, model_traits_type::block_count>, op_types::count>& data, memory_mapped_file* memory_file, tokenizer_type& tokenizer) {
			model_graph_data<config> return_value{};
			gguf_file_t gguf_file{};
			stream_iterator ptr{ memory_file };
			gguf_file.header = value_reader<gguf_header_t>::gather_value(ptr);
			for (uint64_t x = 0; x < gguf_file.header.tensor_count; ++x) {
				gguf_file.tensor_infos.emplace_back(value_reader<core_base_creation_data>::gather_value(ptr));
			}
			uint64_t max_tensor_end = 0;
			for (const auto& tensor: gguf_file.tensor_infos) {
				uint64_t tensor_size = tensor.core_total_byte_size();
				uint64_t tensor_end	 = tensor.offset + tensor_size;
				max_tensor_end		 = std::max(max_tensor_end, tensor_end);
			}

			uint64_t tensor_data_start = ptr.file->size() - max_tensor_end;
			uint64_t alignment{ 0 };
			gather_scalar("alignment", alignment, gguf_file.header.metadata_kv);
			return_value.cparams = value_reader<construction_parameters<model_arches::llama>, model_arches::llama>::gather_value(gguf_file.header.metadata_kv);
			value_reader<tokenizer_type, model_arches::llama>::gather_value(gguf_file.header.metadata_kv, tokenizer);
			value_reader<typename tokenizer_type::vocab_type>::gather_value(gguf_file.header.metadata_kv, *static_cast<typename tokenizer_type::vocab_type*>(&tokenizer));
			sort_tensor_infos(gguf_file.tensor_infos);
			for (uint64_t x = 0; x < gguf_file.header.tensor_count; ++x) {
				uint64_t absolute_offset = tensor_data_start + gguf_file.tensor_infos[x].offset;
				ptr.map_pointer(data[string_to_op_type<model_arches::llama>::impl(gguf_file.tensor_infos[x].name)][extract_layer_number(gguf_file.tensor_infos[x].name)],
					align_offset(absolute_offset, alignment));
			};
			return return_value;
		}
	};

	template<model_config config> struct model_parser {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

		template<typename tokenizer_type> NIHILUS_FORCE_INLINE static model_graph_data<config> parse_model(
			array<array<void*, model_traits_type::block_count>, op_types::count>& data, memory_mapped_file* memory_file, tokenizer_type& tokenizer) {
			return model_parser_impl<config>::parse_model(data, memory_file, tokenizer);
		}
	};
}
