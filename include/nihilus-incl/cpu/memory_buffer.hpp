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

	template<typename config_type> struct memory_transfer;

	template<typename config_type> struct memory_transfer {
		template<pointer_types value_type> NIHILUS_HOST static void host_to_device(const value_type& src, value_type* dst, uint64_t count) noexcept {
			memcpy_wrapper(dst, &src, sizeof(value_type) * count);
		}

		template<not_pointer_types value_type> NIHILUS_HOST static void host_to_device(const value_type& src, value_type* dst) noexcept {
			*dst = src;
		}

		template<pointer_types value_type> NIHILUS_HOST static void device_to_host(const value_type* src, value_type& dst, uint64_t count) noexcept {
			memcpy_wrapper(&dst, src, sizeof(value_type) * count);
		}

		template<not_pointer_types value_type> NIHILUS_HOST static void device_to_host(const value_type* src, value_type& dst) noexcept {
			dst = *src;
		}
	};

	template<typename config_type> struct memory_buffer : public allocator<uint8_t> {
		using value_type = uint8_t;
		using alloc		 = allocator<value_type>;
		using pointer	 = value_type*;
		using size_type	 = uint64_t;

		NIHILUS_HOST memory_buffer() noexcept {
		}

		NIHILUS_HOST memory_buffer& operator=(const memory_buffer&) noexcept = delete;
		NIHILUS_HOST memory_buffer(const memory_buffer&) noexcept			 = delete;

		NIHILUS_HOST memory_buffer& operator=(memory_buffer&& other) noexcept {
			if (this != &other) {
				std::swap(data_val, other.data_val);
				std::swap(size_val, other.size_val);
			}
			return *this;
		}

		NIHILUS_HOST memory_buffer(memory_buffer&& other) noexcept {
			*this = detail::move(other);
		}

		NIHILUS_HOST void init(uint64_t size) noexcept {
			if (data_val) {
				clear();
			}
			data_val = alloc::allocate(size);
			if (!data_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config_type::exceptions, "memory_buffer - failed to allocate memory", location>::impl();
			}
			size_val = size;
		}

		NIHILUS_HOST void deinit() noexcept {
			clear();
		}

		NIHILUS_HOST size_type size() noexcept {
			return size_val;
		}

		NIHILUS_HOST pointer data() noexcept {
			return data_val;
		}

		NIHILUS_HOST void* claim_memory(uint64_t offset_to_claim) noexcept {
			uint64_t aligned_amount = round_up_to_multiple<64>(offset_to_claim);
			if (aligned_amount > size_val) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config_type::exceptions, "memory_buffer - not enough memory allocated!", location>::impl();
			}
			pointer return_value = data_val + aligned_amount;
			return return_value;
		}

		NIHILUS_HOST ~memory_buffer() noexcept {
			clear();
		}

	  protected:
		size_type size_val{};
		pointer data_val{};

		NIHILUS_HOST void clear() noexcept {
			if (data_val) {
				alloc::deallocate(data_val);
				data_val = nullptr;
				size_val = 0;
			}
		}
	};

}
