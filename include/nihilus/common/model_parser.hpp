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
#include <nihilus/common/compare.hpp>
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
		uint8	= 0,
		int8	= 1,
		uint16	= 2,
		int16	= 3,
		uint32	= 4,
		int32	= 5,
		float32 = 6,
		boolean = 7,
		string	= 8,
		array	= 9,
		uint64	= 10,
		int64	= 11,
		float64 = 12,
		unset	= 13,
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
			( void )numa_aware;
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
			if (!input.has_bytes<uint8_t>(length)) {
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
				case gguf_metadata_value_type::int8: {
					value.emplace<int64_t>(value_reader<int8_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::int16: {
					value.emplace<int64_t>(value_reader<int16_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::int32: {
					value.emplace<int64_t>(value_reader<int32_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::int64: {
					value.emplace<int64_t>(value_reader<int64_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::uint8: {
					value.emplace<uint64_t>(value_reader<uint8_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::uint16: {
					value.emplace<uint64_t>(value_reader<uint16_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::uint32: {
					value.emplace<uint64_t>(value_reader<uint32_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::uint64: {
					value.emplace<uint64_t>(value_reader<uint64_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::boolean: {
					value.emplace<bool>(value_reader<bool>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::float32: {
					value.emplace<float>(value_reader<float>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::float64: {
					value.emplace<double>(value_reader<double>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::string: {
					value.emplace<gguf_string_t>(value_reader<gguf_string_t>::gather_value(input));
					break;
				}
				case gguf_metadata_value_type::array: {
					value.emplace<gguf_array_t*>(new gguf_array_t{ value_reader<gguf_array_t>::gather_value(input) });
					break;
				}
				case gguf_metadata_value_type::unset: {
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
		std::unordered_map<std::string, gguf_metadata_kv_t> metadata_kv{};
		uint64_t metadata_kv_count{};
		uint64_t tensor_count{};
		uint32_t version{};
		uint32_t magic{};
	};

	template<typename value_type>
	NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, value_type& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<value_type>(v)) {
			out = std::get<value_type>(v);
		}
		return true;
	};

	template<> NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, uint32_t& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<uint64_t>(v)) {
			out = static_cast<uint32_t>(std::get<uint64_t>(v));
			return true;
		}
		return false;
	};

	template<> NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, uint16_t& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<uint64_t>(v)) {
			out = static_cast<uint16_t>(std::get<uint64_t>(v));
			return true;
		}
		return false;
	};

	template<> NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, uint8_t& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<uint64_t>(v)) {
			out = static_cast<uint8_t>(std::get<uint64_t>(v));
			return true;
		}
		return false;
	};

	template<> NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, int32_t& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<int64_t>(v)) {
			out = static_cast<int32_t>(std::get<uint64_t>(v));
			return true;
		}
		return false;
	};

	template<> NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, int16_t& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<int64_t>(v)) {
			out = static_cast<int16_t>(std::get<uint64_t>(v));
			return true;
		}
		return false;
	};

	template<> NIHILUS_FORCE_INLINE bool gather_scalar(const std::string& key, int8_t& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<int64_t>(v)) {
			out = static_cast<int8_t>(std::get<uint64_t>(v));
			return true;
		}
		return false;
	};

	template<typename value_type>
	NIHILUS_FORCE_INLINE bool gather_array(const std::string& key, std::vector<value_type>& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv);

	template<int_type value_type>
	NIHILUS_FORCE_INLINE bool gather_array(const std::string& key, std::vector<value_type>& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(static_cast<value_type>(std::get<int64_t>(value.value)));
			}
			return true;
		}
		return false;
	};

	template<uint_type value_type>
	NIHILUS_FORCE_INLINE bool gather_array(const std::string& key, std::vector<value_type>& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(static_cast<value_type>(std::get<uint64_t>(value.value)));
			}
			return true;
		}
		return false;
	};

	template<typename value_type>
	NIHILUS_FORCE_INLINE bool gather_array(const std::string& key, std::vector<value_type>& out, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it == metadata_kv.end())
			return false;
		const auto& v = it->second.value.value;
		if (std::holds_alternative<gguf_array_t*>(v)) {
			gguf_array_t& new_array{ *std::get<gguf_array_t*>(v) };
			for (auto& value: new_array.array) {
				out.emplace_back(std::get<value_type>(value.value));
			}
			return true;
		}
		return false;
	};

	template<typename map_type>
	NIHILUS_FORCE_INLINE bool gather_map(const std::string& key, map_type& bpe_ranks, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if NIHILUS_UNLIKELY (it == metadata_kv.end())
			return false;

		const auto& v = it->second.value.value;
		if NIHILUS_UNLIKELY (!std::holds_alternative<gguf_array_t*>(v))
			return false;

		gguf_array_t& array = *std::get<gguf_array_t*>(v);

		bpe_ranks.clear();
		bpe_ranks.reserve(array.array.size());

		for (size_t i = 0; i < array.array.size(); ++i) {
			const std::string& merge_str = std::get<std::string>(array.array[i].value);
			const std::string_view merge_view{ merge_str };
			const size_t space_pos = merge_view.find(' ', 1);
			if NIHILUS_LIKELY (space_pos != std::string_view::npos) {
				bpe_ranks.emplace(std::make_pair(std::string{ merge_view.substr(0, space_pos) }, std::string{ merge_view.substr(space_pos + 1) }), static_cast<int32_t>(i));
			}
			return true;
		}
		return false;
	}

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

	static constexpr string_literal vocab_type_none{ "no vocab" };
	static constexpr string_literal vocab_type_spm{ "llama" };
	static constexpr string_literal vocab_type_bpe{ "gpt2" };
	static constexpr string_literal vocab_type_wpm{ "bert" };
	static constexpr string_literal vocab_type_ugm{ "t5" };
	static constexpr string_literal vocab_type_rwkv{ "rwkv" };

	template<vocab_types vocab_type> NIHILUS_FORCE_INLINE bool compare_vocab_type(std::string_view string) {
		if constexpr (vocab_type == vocab_types::none) {
			return (vocab_type_none.size() <= string.size()) ? string_literal_comparison<vocab_type_none>(string.data()) : false;
		} else if constexpr (vocab_type == vocab_types::spm) {
			return (vocab_type_spm.size() <= string.size()) ? string_literal_comparison<vocab_type_spm>(string.data()) : false;
		} else if constexpr (vocab_type == vocab_types::bpe) {
			return (vocab_type_bpe.size() <= string.size()) ? string_literal_comparison<vocab_type_bpe>(string.data()) : false;
		} else if constexpr (vocab_type == vocab_types::wpm) {
			return (vocab_type_wpm.size() <= string.size()) ? string_literal_comparison<vocab_type_wpm>(string.data()) : false;
		} else if constexpr (vocab_type == vocab_types::ugm) {
			return (vocab_type_ugm.size() <= string.size()) ? string_literal_comparison<vocab_type_ugm>(string.data()) : false;
		} else if constexpr (vocab_type == vocab_types::rwkv) {
			return (vocab_type_rwkv.size() <= string.size()) ? string_literal_comparison<vocab_type_rwkv>(string.data()) : false;
		}
	}

	static constexpr string_literal vocab_pre_type_llama_01{ "llama-bpe" };

	template<vocab_pre_types vocab_type> NIHILUS_FORCE_INLINE bool compare_vocab_pre_type(std::string_view string) {
		if constexpr (vocab_type == vocab_pre_types::llama3) {
			return (vocab_pre_type_llama_01.size() <= string.size()) ? string_literal_comparison<vocab_pre_type_llama_01>(string.data()) : false;
		}
	}

	static constexpr array<const char*, 6> type_names{ vocab_type_none.data(), vocab_type_spm.data(), vocab_type_bpe.data(), vocab_type_wpm.data(), vocab_type_ugm.data(),
		vocab_type_rwkv.data() };

	template<> struct value_reader<construction_parameters<model_arches::llama>, model_arches::llama> {
		NIHILUS_FORCE_INLINE static construction_parameters<model_arches::llama> gather_value(const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
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

	template<model_config config, typename derived_type> struct value_reader<vocab<config, vocab_types::bpe, derived_type>> {
		NIHILUS_FORCE_INLINE static bool gather_value(const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv,
			vocab<config, vocab_types::bpe, derived_type>& tokenizer) {
			using vocab_traits_type = vocab_traits<config.arch, config.vocab_type, config.vocab_pre_type>;

			// Validate tokenizer model type
			std::string tokenizer_model;
			gather_scalar("tokenizer.ggml.model", tokenizer_model, metadata_kv);
			if (!compare_vocab_type<vocab_types::bpe>(tokenizer_model)) {
				throw std::runtime_error{ "Sorry, but you have selected an incorrect vocab type." };
			}

			// Validate vocab pre-processing type
			std::string tokenizer_pre;
			gather_scalar("tokenizer.ggml.pre", tokenizer_pre, metadata_kv);
			if (!compare_vocab_pre_type<vocab_traits_type::pre_type>(tokenizer_pre)) {
				throw std::runtime_error{ "Sorry, but you have selected an incorrect vocab-pre type." };
			}

			// Validate special token IDs against constexpr traits
			auto validate_special_token = [&](const std::string& key, token constexpr_value, const std::string& token_name) {
				if (constexpr_value != token_null) {
					token parsed_value = token_null;
					if (gather_scalar(key, parsed_value, metadata_kv)) {
						if (parsed_value != constexpr_value) {
							throw std::runtime_error{ "Mismatch in " + token_name + " token ID: expected " + std::to_string(constexpr_value) + ", got " +
								std::to_string(parsed_value) };
						}
					} else {
						throw std::runtime_error{ "Missing required " + token_name + " token ID in model metadata" };
					}
				}
			};

			// Validate all special token IDs
			validate_special_token("tokenizer.ggml.bos_token_id", vocab_traits_type::special_bos_id, "BOS");
			validate_special_token("tokenizer.ggml.eos_token_id", vocab_traits_type::special_eos_id, "EOS");
			validate_special_token("tokenizer.ggml.eot_token_id", vocab_traits_type::special_eot_id, "EOT");
			validate_special_token("tokenizer.ggml.eom_token_id", vocab_traits_type::special_eom_id, "EOM");
			validate_special_token("tokenizer.ggml.unk_token_id", vocab_traits_type::special_unk_id, "UNK");
			validate_special_token("tokenizer.ggml.sep_token_id", vocab_traits_type::special_sep_id, "SEP");
			validate_special_token("tokenizer.ggml.pad_token_id", vocab_traits_type::special_pad_id, "PAD");
			validate_special_token("tokenizer.ggml.mask_token_id", vocab_traits_type::special_mask_id, "MASK");
			validate_special_token("tokenizer.ggml.linefeed_id", vocab_traits_type::linefeed_id, "LINEFEED");
			validate_special_token("tokenizer.ggml.fim_pre_token_id", vocab_traits_type::special_fim_pre_id, "FIM_PRE");
			validate_special_token("tokenizer.ggml.fim_suf_token_id", vocab_traits_type::special_fim_suf_id, "FIM_SUF");
			validate_special_token("tokenizer.ggml.fim_mid_token_id", vocab_traits_type::special_fim_mid_id, "FIM_MID");
			validate_special_token("tokenizer.ggml.fim_pad_token_id", vocab_traits_type::special_fim_pad_id, "FIM_PAD");
			validate_special_token("tokenizer.ggml.fim_rep_token_id", vocab_traits_type::special_fim_rep_id, "FIM_REP");
			validate_special_token("tokenizer.ggml.fim_sep_token_id", vocab_traits_type::special_fim_sep_id, "FIM_SEP");

			// Validate boolean configuration flags
			auto validate_bool_config = [&](const std::string& key, bool constexpr_value, const std::string& config_name) {
				bool parsed_value = false;
				if (gather_scalar(key, parsed_value, metadata_kv)) {
					if (parsed_value != constexpr_value) {
						throw std::runtime_error{ "Mismatch in " + config_name + " configuration: expected " + (constexpr_value ? "true" : "false") + ", got " +
							(parsed_value ? "true" : "false") };
					}
				}
				// Note: Optional validation - some models may not specify these flags
			};

			validate_bool_config("tokenizer.ggml.add_space_prefix", vocab_traits_type::add_space_prefix, "add_space_prefix");
			validate_bool_config("tokenizer.ggml.add_bos_token", vocab_traits_type::add_bos, "add_bos");
			validate_bool_config("tokenizer.ggml.add_eos_token", vocab_traits_type::add_eos, "add_eos");
			validate_bool_config("tokenizer.ggml.ignore_merges", vocab_traits_type::ignore_merges, "ignore_merges");
			validate_bool_config("tokenizer.ggml.clean_spaces", vocab_traits_type::clean_spaces, "clean_spaces");
			validate_bool_config("tokenizer.ggml.remove_extra_whitespaces", vocab_traits_type::remove_extra_whitespaces, "remove_extra_whitespaces");
			validate_bool_config("tokenizer.ggml.escape_whitespaces", vocab_traits_type::escape_whitespaces, "escape_whitespaces");
			validate_bool_config("tokenizer.ggml.treat_whitespace_as_suffix", vocab_traits_type::treat_whitespace_as_suffix, "treat_whitespace_as_suffix");

			// Gather BPE merges and validate
			gather_map("tokenizer.ggml.merges", tokenizer.bpe_ranks, metadata_kv);

			// Gather token arrays
			std::vector<std::string> tokens;
			gather_array("tokenizer.ggml.tokens", tokens, metadata_kv);

			std::vector<float> scores;
			gather_array("tokenizer.ggml.scores", scores, metadata_kv);

			std::vector<int32_t> token_types;
			gather_array("tokenizer.ggml.token_type", token_types, metadata_kv);

			// Validate token count against max_token_len if specified
			uint32_t n_tokens = static_cast<uint32_t>(tokens.size());

			// Process tokens with validation
			tokenizer.id_to_token.resize(n_tokens);
			for (uint32_t i = 0; i < n_tokens; i++) {
				std::string word = tokens[i];
				if (word.empty()) {
					word = "[EMPTY_" + std::to_string(i) + "]";
				}

				// Validate token length against max_token_len
				if constexpr (vocab_traits_type::max_token_len > 0) {
					if (word.length() > static_cast<size_t>(vocab_traits_type::max_token_len)) {
						throw std::runtime_error{ "Token length exceeds maximum: token '" + word + "' has length " + std::to_string(word.length()) + ", max allowed is " +
							std::to_string(vocab_traits_type::max_token_len) };
					}
				}

				tokenizer.token_to_id[word] = static_cast<int32_t>(i);

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

			// Cross-validate special tokens found in vocabulary against constexpr traits
			auto cross_validate_special_token = [&](token expected_id, const std::vector<std::string>& possible_texts, const std::string& token_name) {
				if (expected_id != token_null) {
					bool found = false;
					for (const auto& text: possible_texts) {
						auto it = tokenizer.token_to_id.find(text);
						if (it != tokenizer.token_to_id.end()) {
							if (it->second != expected_id) {
								throw std::runtime_error{ "Cross-validation failed for " + token_name + ": found text '" + text + "' with ID " + std::to_string(it->second) +
									", but constexpr traits specify ID " + std::to_string(expected_id) };
							}
							found = true;
							break;
						}
					}
					if (!found) {
						// Warning: Could add logging here if needed
					}
				}
			};

			// Cross-validate special tokens with their text representations
			cross_validate_special_token(vocab_traits_type::special_eot_id,
				{ "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "< EOT >", "<｜end▁of▁sentence｜>" }, "EOT");
			cross_validate_special_token(vocab_traits_type::special_eom_id, { "<|eom_id|>" }, "EOM");
			cross_validate_special_token(vocab_traits_type::special_fim_pre_id, { "<|fim_prefix|>", "<fim-prefix>", "<｜fim▁begin｜>", "<PRE>" }, "FIM_PRE");
			cross_validate_special_token(vocab_traits_type::special_fim_suf_id, { "<|fim_suffix|>", "<fim-suffix>", "<｜fim▁hole｜>", "<SUF>" }, "FIM_SUF");
			cross_validate_special_token(vocab_traits_type::special_fim_mid_id, { "<|fim_middle|>", "<fim-middle>", "<｜fim▁end｜>", "<MID>" }, "FIM_MID");

			// Mark special tokens based on constexpr traits
			for (const auto& [text, id]: tokenizer.token_to_id) {
				if constexpr (vocab_traits_type::special_eot_id == token_null) {
					if (text == "<|eot_id|>" || text == "<|im_end|>" || text == "<|end|>" || text == "<end_of_turn>" || text == "<|endoftext|>" || text == "< EOT >" ||
						text == "<｜end▁of▁sentence｜>") {
						tokenizer.id_to_token[static_cast<uint64_t>(id)].att = tokens::control;
					}
				}

				if constexpr (vocab_traits_type::special_eom_id == token_null && text == "<|eom_id|>") {
					tokenizer.id_to_token[static_cast<uint64_t>(id)].att = tokens::control;
				}

				if constexpr (vocab_traits_type::special_fim_pre_id == token_null) {
					if (text == "<|fim_prefix|>" || text == "<fim-prefix>" || text == "<｜fim▁begin｜>" || text == "<PRE>") {
						tokenizer.id_to_token[static_cast<uint64_t>(id)].att = tokens::control;
					}
				}

				if constexpr (vocab_traits_type::special_fim_suf_id == token_null) {
					if (text == "<|fim_suffix|>" || text == "<fim-suffix>" || text == "<｜fim▁hole｜>" || text == "<SUF>") {
						tokenizer.id_to_token[static_cast<uint64_t>(id)].att = tokens::control;
					}
				}

				if constexpr (vocab_traits_type::special_fim_mid_id == token_null) {
					if (text == "<|fim_middle|>" || text == "<fim-middle>" || text == "<｜fim▁end｜>" || text == "<MID>") {
						tokenizer.id_to_token[static_cast<uint64_t>(id)].att = tokens::control;
					}
				}
			}

			// Build special token cache
			tokenizer.cache_special_tokens.reserve(n_tokens);
			for (token id = 0; id < static_cast<token>(n_tokens); ++id) {
				if (static_cast<size_t>(tokenizer.id_to_token[static_cast<uint64_t>(id)].att) &
					(static_cast<size_t>(tokens::control) | static_cast<size_t>(tokens::user_defined) | static_cast<size_t>(tokens::unused))) {
					tokenizer.cache_special_tokens.emplace_back(id);
				}
			}

			std::sort(tokenizer.cache_special_tokens.begin(), tokenizer.cache_special_tokens.end(), [&](token a, token b) {
				return tokenizer.id_to_token[static_cast<uint64_t>(a)].text.size() > tokenizer.id_to_token[static_cast<uint64_t>(b)].text.size();
			});

			// Build EOG token set with validation
			tokenizer.special_eog_ids.clear();
			for (const auto& [text, id]: tokenizer.token_to_id) {
				if (text == "<|eot_id|>" || text == "<|im_end|>" || text == "<|end|>" || text == "<end_of_turn>" || text == "<|endoftext|>" || text == "<|eom_id|>" ||
					text == "< EOT >") {
					tokenizer.special_eog_ids.insert(id);
				}
			}

			// Add constexpr special tokens to EOG set
			if constexpr (vocab_traits_type::special_eos_id != token_null) {
				tokenizer.special_eog_ids.insert(vocab_traits_type::special_eos_id);
			}
			if constexpr (vocab_traits_type::special_eot_id != token_null) {
				tokenizer.special_eog_ids.insert(vocab_traits_type::special_eot_id);
			}
			if constexpr (vocab_traits_type::special_eom_id != token_null) {
				tokenizer.special_eog_ids.insert(vocab_traits_type::special_eom_id);
			}

			// Final validation summary
			if constexpr (config.exceptions) {
				// Could add comprehensive validation summary logging here
			}
		}
	};

	// Helper function for optional scalar gathering (you may need to implement this)
	template<typename T>
	NIHILUS_FORCE_INLINE bool gather_scalar_optional(const std::string& key, T& value, const std::unordered_map<std::string, gguf_metadata_kv_t>& metadata_kv) {
		auto it = metadata_kv.find(key);
		if (it != metadata_kv.end()) {
			// Implementation depends on your gguf_metadata_kv_t structure
			// This should extract the value and return true if successful
			return extract_value(it->second, value);
		}
		return false;
	}

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
			if (string_literal_comparison<"token_embd.weight">(input.data())) {
				return op_types::token_embd_weight;
			}
			if (string_literal_comparison<"rope_freqs.weight">(input.data())) {
				return op_types::rope_freqs_weight;
			}
			if (string_literal_comparison<"output_norm.weight">(input.data())) {
				return op_types::output_norm_weight;
			}
			if (string_literal_comparison<"output.weight">(input.data())) {
				return op_types::output_weight;
			}

			if (string_literal_comparison<".attn_q.weight">(input.data() + input.find(".attn_q.weight"))) {
				return op_types::attn_q_weight;
			}
			if (string_literal_comparison<".attn_norm.weight">(input.data() + input.find(".attn_norm.weight"))) {
				return op_types::attn_norm_weight;
			}

			if (string_literal_comparison<"blk.">(input.data()) && string_literal_comparison<".weight">(input.data() + input.size() - 7)) {
				auto second_dot = input.find('.', 4);
				if (second_dot != std::string_view::npos) {
					auto suffix = input.substr(second_dot + 1);

					if (string_literal_comparison<"attn_q.weight">(suffix.data())) {
						return op_types::attn_q_weight;
					}
					if (string_literal_comparison<"attn_norm.weight">(suffix.data())) {
						return op_types::attn_norm_weight;
					}
					if (string_literal_comparison<"attn_k.weight">(suffix.data())) {
						return op_types::attn_k_weight;
					}
					if (string_literal_comparison<"attn_v.weight">(suffix.data())) {
						return op_types::attn_v_weight;
					}
					if (string_literal_comparison<"attn_output.weight">(suffix.data())) {
						return op_types::attn_output_weight;
					}
					if (string_literal_comparison<"ffn_down.weight">(suffix.data())) {
						return op_types::ffn_down_weight;
					}
					if (string_literal_comparison<"ffn_gate.weight">(suffix.data())) {
						return op_types::ffn_gate_weight;
					}
					if (string_literal_comparison<"ffn_up.weight">(suffix.data())) {
						return op_types::ffn_up_weight;
					}
					if (string_literal_comparison<"ffn_norm.weight">(suffix.data())) {
						return op_types::ffn_norm_weight;
					}
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
