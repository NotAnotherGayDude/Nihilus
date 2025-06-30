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

#include <nihilus/common/parse_entity.hpp>
#include <nihilus/common/tokenizer.hpp>

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
			: file_path_(other.file_path_), mapped_data_(other.mapped_data_), file_size_(other.file_size_),
#ifdef NIHILUS_PLATFORM_WINDOWS
			  file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
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

		NIHILUS_FORCE_INLINE bool map_pointer(void* dst, const uint64_t offset) {
			*reinterpret_cast<void**>(dst) = reinterpret_cast<uint8_t*>(file->data()) + offset;
			return true;
		}

		template<typename value_type = uint8_t> NIHILUS_FORCE_INLINE bool has_bytes(uint64_t size = sizeof(value_type)) const {
			return (current_index + size <= length);
		}
	};

	template<typename value_type, auto...> struct value_reader;

	template<typename value_type>
		requires(std::is_pod_v<value_type>)
	struct value_reader<value_type> {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			if (input.has_bytes<value_type>()) {
				return input.read<value_type>();
			} else {
				throw std::runtime_error{ "Sorry, but that index is out of range!" };
			}
		}
	};

	template<typename value_type>
		requires(is_specialization_v<value_type, std::vector>)
	struct value_reader<value_type> {
		NIHILUS_FORCE_INLINE static value_type gather_value(stream_iterator& input) {
			gguf_metadata_value_type type{ value_reader<gguf_metadata_value_type>::gather_value(input) };
			uint64_t length{ value_reader<uint64_t>::gather_value(input) };
			constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
			if (length > MAX_ARRAY_LENGTH) {
				throw std::runtime_error{ "Array length exceeds maximum allowed size!" };
			}
			value_type value{};
			value.reserve(length);
			for (uint64_t x = 0; x < length; ++x) {
				value.emplace_back(value_reader<typename value_type::value_type>::gather_value(input));
			}
			return value;
		}
	};

	using gguf_string_t = std::string_view;

	template<> struct value_reader<gguf_string_t> {
		NIHILUS_FORCE_INLINE static std::string_view gather_value(stream_iterator& input) {
			uint64_t length = value_reader<uint64_t>::gather_value(input);
			if (!input.has_bytes<uint8_t>(length)) {
				throw std::runtime_error("Sorry, but that index is out of range!");
			}
			const char* string_ptr{ static_cast<const char*>(input.file->data()) + input.current_index };
			input.current_index += length;
			std::string_view result(string_ptr, length);
			return result;
		}
	};

	struct gguf_metadata_base {
		uint64_t tensor_count;
		std::string_view general_name;
		std::string_view general_architecture;
		std::string_view general_type;
		std::string_view general_basename;
		std::string_view general_finetune;
		std::string_view general_size_label;
		uint64_t metadata_kv_count;
		std::string_view general_license;
		std::vector<std::string_view> general_tags;
		std::vector<std::string_view> general_languages;
		uint32_t alignment{};
		uint32_t general_file_type;
		uint32_t general_quantization_version;
		std::string_view tokenizer_ggml_model;
		std::string_view tokenizer_ggml_pre;
		std::string_view tokenizer_chat_template;
		std::vector<std::string_view> tokenizer_ggml_tokens;
		std::vector<int32_t> tokenizer_ggml_token_type;
		std::vector<std::string_view> tokenizer_ggml_merges;
		int32_t quantize_imatrix_entries_count;
		int32_t quantize_imatrix_chunks_count;
		std::string_view quantize_imatrix_file;
		std::string_view quantize_imatrix_dataset;
	};

	template<model_arches arch, vocab_types type, vocab_pre_types pre> struct gguf_metadata;

	template<vocab_types type, vocab_pre_types pre> struct gguf_metadata<model_arches::llama, type, pre> : public gguf_metadata_base,
																										   public vocab_traits<model_arches::llama, type, pre> {
		uint32_t llama_rope_dimension_count;
		uint32_t llama_block_count;
		uint32_t llama_context_length;
		uint32_t llama_embedding_length;
		uint32_t llama_feed_forward_length;
		uint32_t llama_attention_head_count;
		uint32_t llama_attention_head_count_kv;
		uint32_t llama_vocab_size;
		float llama_rope_freq_base;
		float llama_attention_layer_norm_rms_epsilon;
	};

	template<model_arches arch, vocab_types type, vocab_pre_types pre> struct core<gguf_metadata<arch, type, pre>> {
		using value_type				  = gguf_metadata<arch, type, pre>;
		static constexpr auto parse_value = create_value<make_parse_entity<&value_type::general_basename, "general.basename">(),
			make_parse_entity<&value_type::alignment, "general.alignment">(), make_parse_entity<&value_type::general_architecture, "general.architecture">(),
			make_parse_entity<&value_type::general_file_type, "general.filetype">(), make_parse_entity<&value_type::general_finetune, "general.finetune">(),
			make_parse_entity<&value_type::general_languages, "general.languages">(), make_parse_entity<&value_type::general_license, "general.license">(),
			make_parse_entity<&value_type::general_name, "general.name">(), make_parse_entity<&value_type::general_quantization_version, "general.quantization_version">(),
			make_parse_entity<&value_type::general_size_label, "general.size_label">(), make_parse_entity<&value_type::general_tags, "general.tags">(),
			make_parse_entity<&value_type::general_type, "general.type">(), make_parse_entity<&value_type::special_bos_id, "tokenizer.ggml.bos_token_id">(),
			make_parse_entity<&value_type::special_eos_id, "tokenizer.ggml.eos_token_id">(), make_parse_entity<&value_type::special_eot_id, "tokenizer.ggml.eot_token_id">(),
			make_parse_entity<&value_type::special_eom_id, "tokenizer.ggml.eom_token_id">(), make_parse_entity<&value_type::special_unk_id, "tokenizer.ggml.unknown_token_id">(),
			make_parse_entity<&value_type::special_sep_id, "tokenizer.ggml.separator_token_id">(),
			make_parse_entity<&value_type::special_pad_id, "tokenizer.ggml.padding_token_id">(), make_parse_entity<&value_type::special_mask_id, "tokenizer.ggml.mask_token_id">(),
			make_parse_entity<&value_type::linefeed_id, "tokenizer.ggml.linefeed_id">(), make_parse_entity<&value_type::special_fim_pre_id, "tokenizer.ggml.fim_prefix_token_id">(),
			make_parse_entity<&value_type::special_fim_suf_id, "tokenizer.ggml.fim_suffix_token_id">(),
			make_parse_entity<&value_type::special_fim_mid_id, "tokenizer.ggml.fim_middle_token_id">(),
			make_parse_entity<&value_type::special_fim_pad_id, "tokenizer.ggml.fim_pad_token_id">(),
			make_parse_entity<&value_type::special_fim_rep_id, "tokenizer.ggml.fim_repo_token_id">(),
			make_parse_entity<&value_type::special_fim_sep_id, "tokenizer.ggml.fim_sep_token_id">(),
			make_parse_entity<&value_type::max_token_len, "tokenizer.ggml.max_token_length">(),
			make_parse_entity<&value_type::add_space_prefix, "tokenizer.ggml.add_space_prefix">(), make_parse_entity<&value_type::add_bos, "tokenizer.ggml.add_bos_token">(),
			make_parse_entity<&value_type::add_eos, "tokenizer.ggml.add_eos_token">(), make_parse_entity<&value_type::ignore_merges, "tokenizer.ggml.ignore_merges">(),
			make_parse_entity<&value_type::clean_spaces, "tokenizer.ggml.clean_up_tokenization_spaces">(),
			make_parse_entity<&value_type::remove_extra_whitespaces, "tokenizer.ggml.remove_extra_whitespaces">(),
			make_parse_entity<&value_type::escape_whitespaces, "tokenizer.ggml.escape_whitespaces">(),
			make_parse_entity<&value_type::treat_whitespace_as_suffix, "tokenizer.ggml.treat_whitespace_as_suffix">(),
			make_parse_entity<&value_type::quantize_imatrix_entries_count, "quantize.imatrix.entries_count">(),
			make_parse_entity<&value_type::quantize_imatrix_chunks_count, "quantize.imatrix.chunks_count">(),
			make_parse_entity<&value_type::quantize_imatrix_file, "quantize.imatrix.file">(),
			make_parse_entity<&value_type::quantize_imatrix_dataset, "quantize.imatrix.dataset">(),
			make_parse_entity<&value_type::llama_rope_dimension_count, "llama.rope.dimension_count">(), make_parse_entity<&value_type::llama_block_count, "llama.block_count">(),
			make_parse_entity<&value_type::llama_context_length, "llama.context_length">(), make_parse_entity<&value_type::llama_embedding_length, "llama.embedding_length">(),
			make_parse_entity<&value_type::llama_feed_forward_length, "llama.feed_forward_length">(),
			make_parse_entity<&value_type::llama_attention_head_count, "llama.attention.head_count">(),
			make_parse_entity<&value_type::llama_attention_head_count_kv, "llama.attention.head_count_kv">(),
			make_parse_entity<&value_type::llama_vocab_size, "llama.vocab_size">(), make_parse_entity<&value_type::llama_rope_freq_base, "llama.rope.freq_base">(),
			make_parse_entity<&value_type::llama_attention_layer_norm_rms_epsilon, "llama.attention.layer_norm_rms_epsilon">()>();
	};

	template<typename stream_type, typename value_type> struct parse_types_impl {
		inline static constexpr auto memberCount = core_tuple_size<value_type>;

		template<uint64_t index> using member_type_t =
			std::remove_reference_t<decltype(get_member<value_type>(get<index>(core<value_type>::parse_value).member_ptr, std::declval<value_type&>()))>;

		template<uint64_t index> NIHILUS_FORCE_INLINE static bool processIndex(value_type& value, std::string_view string, stream_type& stream) {
			static constexpr auto tupleElem	 = get<index>(core<value_type>::parse_value);
			static constexpr auto string_lit = tupleElem.name;
			static constexpr auto ptrNew	 = tupleElem.member_ptr;
			static constexpr auto keySize	 = string_lit.size();
			static constexpr auto keySizeNew = keySize + 1;
			if NIHILUS_LIKELY ((string.size() <= keySize) && string_literal_comparitor<decltype(string_lit), string_lit>::impl(string.data())) {
				auto& ref = get_member<value_type>(ptrNew, value);
				if constexpr (!std::is_const_v<std::remove_reference_t<decltype(ref)>>) {
					ref = value_reader<member_type_t<index>>::gather_value(stream);
				} else {
					member_type_t<index> value_new{ value_reader<std::remove_const_t<member_type_t<index>>>::gather_value(stream) };
					if (value_new != ref) {
						std::string error_string{ std::string{ "Sorry, but member of name: " } + std::string{ string_lit.data(), string_lit.size() } + " was not equal!" };
						throw std::runtime_error{ error_string };
					}
				}
			}
			return false;
		}
	};

	template<template<typename, typename> typename parsing_type, typename stream_type, typename value_type, size_t... indices>
	inline static constexpr auto generateFunctionPtrs(std::index_sequence<indices...>) noexcept {
		using function_type = decltype(&parse_types_impl<stream_type, value_type>::template processIndex<0>);
		return array<function_type, sizeof...(indices)>{ { &parsing_type<stream_type, value_type>::template processIndex<indices>... } };
	}

	template<template<typename, typename> typename parsing_type, typename stream_type, typename value_type>
	static constexpr auto function_ptrs{ generateFunctionPtrs<parsing_type, stream_type, value_type>(std::make_index_sequence<core_tuple_size<value_type>>{}) };

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

	NIHILUS_INLINE uint64_t calculate_and_skip_unknown_value(stream_iterator& input, gguf_metadata_value_type type) {
		uint64_t bytes_skipped = 0;

		switch (type) {
			case gguf_metadata_value_type::uint8:
			case gguf_metadata_value_type::int8:
			case gguf_metadata_value_type::boolean: {
				bytes_skipped = 1;
				input.current_index += bytes_skipped;
				break;
			}
			case gguf_metadata_value_type::uint16:
			case gguf_metadata_value_type::int16: {
				bytes_skipped = 2;
				input.current_index += bytes_skipped;
				break;
			}
			case gguf_metadata_value_type::uint32:
			case gguf_metadata_value_type::int32:
			case gguf_metadata_value_type::float32: {
				bytes_skipped = 4;
				input.current_index += bytes_skipped;
				break;
			}
			case gguf_metadata_value_type::uint64:
			case gguf_metadata_value_type::int64:
			case gguf_metadata_value_type::float64: {
				bytes_skipped = 8;
				input.current_index += bytes_skipped;
				break;
			}
			case gguf_metadata_value_type::string: {
				if (!input.has_bytes<uint64_t>()) {
					throw std::runtime_error("Insufficient bytes for string length!");
				}
				uint64_t string_length = input.read<uint64_t>();
				bytes_skipped		   = sizeof(uint64_t) + string_length;

				if (!input.has_bytes<uint8_t>(string_length)) {
					throw std::runtime_error("Insufficient bytes for string content!");
				}
				input.current_index += string_length;
				break;
			}
			case gguf_metadata_value_type::array: {
				if (!input.has_bytes<gguf_metadata_value_type>()) {
					throw std::runtime_error("Insufficient bytes for array type!");
				}
				gguf_metadata_value_type array_type = input.read<gguf_metadata_value_type>();
				bytes_skipped += sizeof(gguf_metadata_value_type);

				if (!input.has_bytes<uint64_t>()) {
					throw std::runtime_error("Insufficient bytes for array length!");
				}
				uint64_t array_length = input.read<uint64_t>();
				bytes_skipped += sizeof(uint64_t);

				constexpr uint64_t MAX_ARRAY_LENGTH = 1024 * 1024;
				if (array_length > MAX_ARRAY_LENGTH) {
					throw std::runtime_error("Array length exceeds maximum allowed size during skip!");
				}

				for (uint64_t i = 0; i < array_length; ++i) {
					uint64_t element_bytes = calculate_and_skip_unknown_value(input, array_type);
					bytes_skipped += element_bytes;
				}
				break;
			}
			case gguf_metadata_value_type::unset:
			default: {
				break;
			}
		}

		return bytes_skipped;
	}

	template<model_arches arch, vocab_types type, vocab_pre_types pre> struct value_reader<gguf_metadata<arch, type, pre>> {
		NIHILUS_FORCE_INLINE static gguf_metadata<arch, type, pre> gather_value(stream_iterator& input) {
			gguf_metadata<arch, type, pre> value{};
			uint32_t magic = value_reader<uint32_t>::gather_value(input);
			if (magic != 0x46554747) {
				throw std::runtime_error{ "Sorry, but that magic value was incorrect!" };
			}
			uint64_t version		= value_reader<uint32_t>::gather_value(input);
			value.tensor_count		= value_reader<uint64_t>::gather_value(input);
			value.metadata_kv_count = value_reader<uint64_t>::gather_value(input);

			static constexpr uint64_t MAX_TENSOR_COUNT	 = 100000;
			static constexpr uint64_t MAX_METADATA_COUNT = 10000;

			if (value.tensor_count > MAX_TENSOR_COUNT) {
				throw std::runtime_error{ "Tensor count exceeds reasonable maximum!" };
			}
			if (value.metadata_kv_count > MAX_METADATA_COUNT) {
				throw std::runtime_error{ "Metadata count exceeds reasonable maximum!" };
			}

			for (uint64_t x = 0; x < value.metadata_kv_count; ++x) {
				std::string_view new_string			= value_reader<gguf_string_t>::gather_value(input);
				gguf_metadata_value_type value_type = value_reader<gguf_metadata_value_type>::gather_value(input);
				auto index							= hash_map<gguf_metadata<arch, type, pre>, const char*>::findIndex(new_string.data(), new_string.data() + new_string.size());
				if (index < function_ptrs<parse_types_impl, stream_iterator, gguf_metadata<arch, type, pre>>.size()) {
					function_ptrs<parse_types_impl, stream_iterator, gguf_metadata<arch, type, pre>>[index](value, new_string, input);
				} else {
					calculate_and_skip_unknown_value(input, value_type);
				}
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

	struct core_base_creation_data {
		array<uint64_t, 4> dimensions{ { 1, 1, 1, 1 } };
		uint32_t n_dimensions{};
		uint64_t layer_number{};
		op_types op_type{};
		uint64_t offset{};
		data_types type{};

		NIHILUS_FORCE_INLINE uint64_t core_total_dims() const {
			return dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3];
		}

		NIHILUS_FORCE_INLINE uint64_t core_total_byte_size() const {
			uint64_t total_elements = core_total_dims();
			uint64_t block_size		= core_block_size();
			uint64_t type_size		= core_type_size();
			uint64_t num_blocks		= (total_elements + block_size - 1) / block_size;
			return num_blocks * type_size;
		}

		NIHILUS_FORCE_INLINE uint64_t core_block_size() const {
			return get_type_traits(type).block_size;
		}

		NIHILUS_FORCE_INLINE uint64_t core_type_size() const {
			return get_type_traits(type).type_size;
		}

		NIHILUS_FORCE_INLINE uint64_t core_row_size(int64_t dims_new) const {
			return core_type_size() * dims_new / core_block_size();
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

	template<model_arches arch> struct value_reader<core_base_creation_data, arch> {
		NIHILUS_FORCE_INLINE static core_base_creation_data gather_value(stream_iterator& input) {
			core_base_creation_data value{};
			std::string_view name{ value_reader<std::string_view>::gather_value(input) };
			value.op_type					  = string_to_op_type<arch>::impl(name);
			value.n_dimensions				  = value_reader<uint32_t>::gather_value(input);
			value.layer_number				  = extract_layer_number(name);
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

	NIHILUS_FORCE_INLINE bool operator<(const core_base_creation_data& lhs, const core_base_creation_data& rhs) noexcept {
		return lhs.layer_number < rhs.layer_number;
	}

	NIHILUS_FORCE_INLINE void sort_tensor_infos(std::vector<core_base_creation_data>& tensor_infos) noexcept {
		std::sort(tensor_infos.begin(), tensor_infos.end(), std::less<core_base_creation_data>{});
	}

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
		template<typename tokenizer_type> NIHILUS_FORCE_INLINE static gguf_metadata<config.arch, config.vocab_type, config.vocab_pre_type> parse_model(
			array<array<void*, model_traits_type::block_count>, op_types::count>& data, memory_mapped_file* memory_file, tokenizer_type& tokenizer) {
			stream_iterator ptr{ memory_file };
			gguf_metadata<config.arch, config.vocab_type, config.vocab_pre_type> gguf_file{
				value_reader<gguf_metadata<config.arch, config.vocab_type, config.vocab_pre_type>>::gather_value(ptr)
			};
			tokenizer.tokens		= std::move(gguf_file.tokenizer_ggml_tokens);
			tokenizer.merges		= std::move(gguf_file.tokenizer_ggml_merges);
			tokenizer.token_types	= std::move(gguf_file.tokenizer_ggml_token_type);
			tokenizer.chat_template = std::move(gguf_file.tokenizer_chat_template);
			tokenizer.pre			= std::move(gguf_file.tokenizer_ggml_pre);
			std::vector<core_base_creation_data> tensor_infos{};
			tensor_infos.reserve(gguf_file.tensor_count);
			for (uint64_t x = 0; x < gguf_file.tensor_count; ++x) {
				auto new_tensor{ value_reader<core_base_creation_data, model_arches::llama>::gather_value(ptr) };
				tensor_infos.emplace_back(new_tensor);
			}
			uint64_t max_tensor_end = 0;
			for (const auto& tensor: tensor_infos) {
				uint64_t tensor_size = tensor.core_total_byte_size();
				uint64_t tensor_end	 = tensor.offset + tensor_size;
				max_tensor_end		 = std::max(max_tensor_end, tensor_end);
			}

			uint64_t tensor_data_start = ptr.file->size() - max_tensor_end;

			sort_tensor_infos(tensor_infos);
			for (uint64_t x = 0; x < gguf_file.tensor_count; ++x) {
				uint64_t absolute_offset = tensor_data_start + tensor_infos[x].offset;
				ptr.map_pointer(data[tensor_infos[x].op_type][tensor_infos[x].layer_number], absolute_offset);
			};
			return gguf_file;
		}
	};

	template<model_config config> struct model_parser {
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

		template<typename tokenizer_type> NIHILUS_FORCE_INLINE static gguf_metadata<config.arch, config.vocab_type, config.vocab_pre_type> parse_model(
			array<array<void*, model_traits_type::block_count>, op_types::count>& data, memory_mapped_file* memory_file, tokenizer_type& tokenizer) {
			return model_parser_impl<config>::parse_model(data, memory_file, tokenizer);
		}
	};
}
