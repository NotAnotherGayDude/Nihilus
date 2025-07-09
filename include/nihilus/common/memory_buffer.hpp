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

#include <nihilus/common/allocator.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/config.hpp>
#include <stdexcept>
#include <iterator>

namespace nihilus {

	template<model_config config> struct memory_buffer : public allocator<uint8_t> {
		using value_type  = uint8_t;
		using alloc		  = allocator<value_type>;
		using pointer	  = value_type*;
		using uint64_type = uint64_t;

		NIHILUS_FORCE_INLINE memory_buffer() noexcept = default;

		NIHILUS_FORCE_INLINE memory_buffer& operator=(const memory_buffer&) noexcept = delete;
		NIHILUS_FORCE_INLINE memory_buffer(const memory_buffer&) noexcept			 = delete;

		NIHILUS_FORCE_INLINE memory_buffer& operator=(memory_buffer&& other) noexcept {
			if (this != &other) {
				std::swap(current_offset, other.current_offset);
				std::swap(data_val, other.data_val);
				std::swap(size_val, other.size_val);
			}
			return *this;
		}

		NIHILUS_FORCE_INLINE memory_buffer(memory_buffer&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_FORCE_INLINE void init(uint64_t size) noexcept {
			if (data_val) {
				clear();
			}
			data_val = alloc::allocate(size);
			size_val = size;
		}

		NIHILUS_FORCE_INLINE void deinit() noexcept {
			clear();
		}

		NIHILUS_FORCE_INLINE uint64_type size() noexcept {
			return size_val;
		}

		NIHILUS_FORCE_INLINE pointer data() noexcept {
			return data_val;
		}

		NIHILUS_FORCE_INLINE void* claim_memory(uint64_t amount_to_claim) noexcept {
			static constexpr uint64_t alignment = cpu_alignment;

			uint64_t aligned_amount = round_up_to_multiple<alignment>(amount_to_claim);

			if (current_offset + aligned_amount > size_val) {
				if constexpr (config.exceptions) {
					throw std::runtime_error{ "Sorry, but this memory_buffer is out of memory!" };
				} else {
					return nullptr;
				}
			}

			pointer return_value = data_val + current_offset;
			current_offset += aligned_amount;
			return return_value;
		}

		NIHILUS_FORCE_INLINE ~memory_buffer() noexcept {
			clear();
		}

	  protected:
		uint64_type current_offset{};
		value_type* data_val{};
		uint64_type size_val{};

		NIHILUS_FORCE_INLINE void clear() noexcept {
			if (data_val) {
				alloc::deallocate(data_val);
				data_val = nullptr;
				size_val = 0;
			}
		}
	};

}
