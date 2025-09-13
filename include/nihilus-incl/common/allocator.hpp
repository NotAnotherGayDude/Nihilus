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

#include <nihilus-incl/common/config.hpp>
#include <nihilus-incl/common/utility.hpp>
#include <memory_resource>

namespace nihilus {

	template<auto multiple, typename value_01_type = decltype(multiple)> NIHILUS_HOST constexpr value_01_type round_up_to_multiple(value_01_type value) noexcept {
		if constexpr ((multiple & (multiple - 1)) == 0) {
			constexpr value_01_type mulSub1{ multiple - 1 };
			constexpr value_01_type notMulSub1{ static_cast<value_01_type>(~mulSub1) };
			return (value + (mulSub1)) & notMulSub1;
		} else {
			const value_01_type remainder = value % multiple;
			return remainder == 0 ? value : value + (multiple - remainder);
		}
	}

	template<typename value_type_new> class allocator {
	  public:
		using value_type	   = value_type_new;
		using pointer		   = value_type*;
		using const_pointer	   = const value_type*;
		using reference		   = value_type&;
		using const_reference  = const value_type&;
		using size_type		   = uint64_t;
		using difference_type  = std::ptrdiff_t;
		using allocator_traits = std::allocator_traits<allocator<value_type>>;

		template<typename U> struct rebind {
			using other = allocator<U>;
		};

		NIHILUS_HOST allocator() noexcept {
		}

		template<typename U> NIHILUS_HOST allocator(const allocator<U>&) noexcept {
		}

		NIHILUS_HOST static pointer allocate(size_type count_new) noexcept {
			if NIHILUS_UNLIKELY (count_new == 0) {
				return nullptr;
			}
#if NIHILUS_PLATFORM_WINDOWS || NIHILUS_PLATFORM_LINUX
			return static_cast<value_type*>(_mm_malloc(round_up_to_multiple<64>(count_new * sizeof(value_type)), 64));
#else
			return static_cast<value_type*>(aligned_alloc(64, round_up_to_multiple<64>(count_new * sizeof(value_type))));
#endif
		}

		NIHILUS_HOST static void deallocate(pointer ptr, uint64_t = 0) noexcept {
			if NIHILUS_LIKELY (ptr) {
#if NIHILUS_PLATFORM_WINDOWS || NIHILUS_PLATFORM_LINUX
				_mm_free(ptr);
#else
				free(ptr);
#endif
			}
		}

		template<typename... arg_types> NIHILUS_HOST static void construct(pointer ptr, arg_types&&... args) noexcept {
			new (ptr) value_type(detail::forward<arg_types>(args)...);
		}

		NIHILUS_HOST static void destroy(pointer ptr) noexcept {
			ptr->~value_type();
		}

		NIHILUS_HOST constexpr bool operator==(const allocator&) const noexcept {
			return true;
		}
	};

}// namespace internal
