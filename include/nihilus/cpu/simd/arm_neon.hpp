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

#include <nihilus/common/kernel_traits.hpp>
#include <nihilus/cpu/simd/common.hpp>

#if defined(NIHILUS_NEON)

#include <arm_neon.h>

namespace nihilus {

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_values(const void* str) noexcept {
		return vld1q_u8(static_cast<const uint8_t*>(str));
	}

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static simd_int_type_new gather_valuesU(const void* str, void*) noexcept {
		return vld1q_u8(static_cast<const uint8_t*>(str));
	}

	template<simd_int_128_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 8)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(char_t str) noexcept {
		return vdupq_n_u64(str);
	}

	template<simd_int_128_type simd_int_type_new, typename char_t>
		requires(sizeof(char_t) == 1)
	NIHILUS_FORCE_INLINE static simd_int_type_new gather_value(char_t str) noexcept {
		return vdupq_n_u8(str);
	}

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static void store(const simd_int_type_new& value, void* storageLocation) noexcept {
		vst1q_u8(static_cast<uint8_t*>(storageLocation), value);
	}

	template<simd_int_128_type simd_int_type_new> NIHILUS_FORCE_INLINE static void storeU(const simd_int_type_new& value, void* storageLocation, void*) noexcept {
		vst1q_u8(static_cast<uint8_t*>(storageLocation), value);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto op_cmp_eq(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vceqq_u8(value, other), 4)), 0);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opCmpLt(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vcgtq_u8(other, value), 4)), 0);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vceqq_u8(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto opCmpLtRaw(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vcgtq_u8(other, value);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static uint64_t opBitMaskRaw(const simd_int_t01& value) noexcept {
		return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(value, 4)), 0);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static uint32_t opBitMask(const simd_int_t01& value) noexcept {
		constexpr uint8x16_t bit_mask{ 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80 };
		const auto minput = value & bit_mask;
		uint8x16_t tmp	  = vpaddq_u8(minput, minput);
		tmp				  = vpaddq_u8(tmp, tmp);
		tmp				  = vpaddq_u8(tmp, tmp);
		return vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static auto op_cmp_eqBitMask(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return opBitMask(vceqq_u8(value, other));
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02> NIHILUS_FORCE_INLINE static auto opShuffle(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		const auto bitMask{ vdupq_n_u8(0x0F) };
		return vqtbl1q_u8(value, vandq_u8(other, bitMask));
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static nihilus_simd_int_128 opXor(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return veorq_u8(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static nihilus_simd_int_128 opAnd(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vandq_u8(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static nihilus_simd_int_128 opOr(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vorrq_u8(value, other);
	}

	template<simd_int_128_type simd_int_t01, simd_int_128_type simd_int_t02>
	NIHILUS_FORCE_INLINE static nihilus_simd_int_128 opAndNot(const simd_int_t01& value, const simd_int_t02& other) noexcept {
		return vbicq_u8(value, other);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static bool opTest(const simd_int_t01& value) noexcept {
		return vmaxvq_u8(value) != 0;
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static auto opNot(const simd_int_t01& value) noexcept {
		return vmvnq_u8(value);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static nihilus_simd_int_128 opSetLSB(const simd_int_t01& value, bool value_new) noexcept {
		constexpr uint8x16_t mask{ 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
		return value_new ? vorrq_u8(value, mask) : vbicq_u8(value, mask);
	}

	template<simd_int_128_type simd_int_t01> NIHILUS_FORCE_INLINE static bool opGetMSB(const simd_int_t01& value) noexcept {
		return (vgetq_lane_u8(value, 15) & 0x80) != 0;
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm_mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02, const typename core_type::input_type03& input03) {
		}
	};

};

#endif
