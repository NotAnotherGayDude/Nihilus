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

#include <nihilus-incl/common/allocator.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/config.hpp>
#include <stdexcept>
#include <iterator>

namespace nihilus {

	template<model_config config> struct memory_buffer : public allocator<uint8_t, cpu_alignment> {
		using value_type   = uint8_t;
		using alloc		   = allocator<value_type, cpu_alignment>;
		using pointer	   = value_type*;
		using uint64_types = uint64_t;
		using size_type	   = uint64_t;

		NIHILUS_INLINE memory_buffer() noexcept = default;

		NIHILUS_INLINE memory_buffer& operator=(const memory_buffer&) noexcept = delete;
		NIHILUS_INLINE memory_buffer(const memory_buffer&) noexcept			   = delete;

		NIHILUS_INLINE memory_buffer& operator=(memory_buffer&& other) noexcept {
			if (this != &other) {
				std::swap(data_val, other.data_val);
				std::swap(size_val, other.size_val);
				std::swap(head, other.head);
				std::swap(tail, other.tail);
			}
			return *this;
		}

		NIHILUS_INLINE memory_buffer(memory_buffer&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_INLINE void init(uint64_t size) noexcept {
			std::cout << "SIZE: " << size << std::endl;
			if (data_val) {
				clear();
			}
			data_val = alloc::allocate(size);
			std::fill_n(data_val, size, value_type{});
			if (!data_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "memory_buffer - failed to allocate memory", location>::impl();
			}
			size_val = size;
		}

		NIHILUS_INLINE void deinit() noexcept {
			clear();
		}

		NIHILUS_INLINE uint64_types size() noexcept {
			return size_val;
		}

		NIHILUS_INLINE pointer data() noexcept {
			return data_val;
		}

		NIHILUS_INLINE void* claim_memory(uint64_t offset_to_claim) noexcept {
			uint64_t aligned_amount = round_up_to_multiple<cpu_alignment>(offset_to_claim);
			if (aligned_amount > size_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "memory_buffer - not enough memory allocated!", location>::impl();
			}
			pointer return_value = data_val + aligned_amount;
			return return_value;
		}

		NIHILUS_INLINE ~memory_buffer() noexcept {
			clear();
		}

	  protected:
		value_type* data_val{};
		size_type size_val{};
		size_type tail{};
		size_type head{};

		NIHILUS_INLINE void clear() noexcept {
			if (data_val) {
				alloc::deallocate(data_val);
				data_val = nullptr;
				size_val = 0;
			}
		}
	};

}
