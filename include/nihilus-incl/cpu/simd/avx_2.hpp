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

#include <nihilus-incl/common/kernel_traits.hpp>
#include <nihilus-incl/cpu/simd/common.hpp>
#include <assert.h>

#if defined(NIHILUS_AVX2)

namespace nihilus {

	NIHILUS_INLINE static uint64_t largest_pow2(uint64_t x) {
		if (x == 0) {
			return 0;
		}
		return 1ULL << (63 - lzcnt(x));
	};

	NIHILUS_INLINE static half fp32_to_fp16(float f) {
		static constexpr float scale_to_inf	 = fp32_from_bits(UINT32_C(0x77800000));
		static constexpr float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
		float base							 = (fabsf(f) * scale_to_inf) * scale_to_zero;

		const uint32_t w	  = fp32_to_bits(f);
		const uint32_t shl1_w = w + w;
		const uint32_t sign	  = w & UINT32_C(0x80000000);
		uint32_t bias		  = shl1_w & UINT32_C(0xFF000000);
		if (bias < UINT32_C(0x71000000)) {
			bias = UINT32_C(0x71000000);
		}

		base						 = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
		const uint32_t bits			 = fp32_to_bits(base);
		const uint32_t exp_bits		 = (bits >> 13) & UINT32_C(0x00007C00);
		const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
		const uint32_t nonsign		 = exp_bits + mantissa_bits;
		return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
	}

	NIHILUS_INLINE static void dequantize_row_q8_0(const block_q8_0<half>* __restrict x, float* __restrict y, uint64_t k) {
		static constexpr int64_t qk = Q_SIZE;

		const uint64_t nb = k & ~(Q_SIZE - 1);

		for (uint64_t i = 0; i < nb; i++) {
			const float d = fp16_to_fp32(x[i].d);

			for (uint64_t j = 0; j < qk; ++j) {
				y[i * qk + j] = x[i].qs[j] * d;
			}
		}
	}

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::embedding_lookup, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::embedding_lookup, core_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			const uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01 = input01[1];
			const uint64_t ne02 = input_type01::get_array()[2];
			const uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11 = input02[1];

			const uint64_t nr  = count_elements(input02);
			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + nth - 1ull) / nth;
			const uint64_t ir0 = dr * ith;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const block_q8_0<half>* __restrict input01_data = input01.data;
			const int32_t* __restrict input02_data			= input02.data;
			float* __restrict output_data					= output.data;

			const uint64_t blocks_per_row	   = ne00 / Q_SIZE;
			const uint64_t input01_stride_dim1 = blocks_per_row;
			const uint64_t input01_stride_dim2 = ne01 * blocks_per_row;
			const uint64_t input01_stride_dim3 = ne01 * ne02 * blocks_per_row;

			const uint64_t input02_stride_dim1 = 1;
			const uint64_t input02_stride_dim2 = ne10;
			const uint64_t input02_stride_dim3 = ne10 * ne11;

			const uint64_t output_stride_dim1 = ne00;
			const uint64_t output_stride_dim2 = output[1] * ne00;
			const uint64_t output_stride_dim3 = output[1] * output[2] * ne00;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12 = i / (ne11 * ne10);
				const uint64_t i11 = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10 = i - i12 * ne11 * ne10 - i11 * ne10;

				const uint64_t input02_idx = i10 * input02_stride_dim1 + i11 * input02_stride_dim2 + i12 * input02_stride_dim3;
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * input01_stride_dim1 + i11 * input01_stride_dim2 + i12 * input01_stride_dim3;

				const uint64_t output_idx = i10 * output_stride_dim1 + i11 * output_stride_dim2 + i12 * output_stride_dim3;

				dequantize_row_q8_0(&input01_data[input01_block_idx], &output_data[output_idx], ne00);
			}
		}
	};

	NIHILUS_INLINE static void vec_cpy_f32(const uint64_t n, float* __restrict y, const float* __restrict x) {
		const uint64_t np = n & ~(Q_SIZE - 1);
		for (uint64_t i = 0; i < np; i += Q_SIZE) {
			__m256 ax0 = _mm256_load_ps(x + i);
			__m256 ax1 = _mm256_load_ps(x + i + 8);
			__m256 ax2 = _mm256_load_ps(x + i + 16);
			__m256 ax3 = _mm256_load_ps(x + i + 24);
			_mm256_store_ps(y + i, ax0);
			_mm256_store_ps(y + i + 8, ax1);
			_mm256_store_ps(y + i + 16, ax2);
			_mm256_store_ps(y + i + 24, ax3);
		}
		for (uint64_t i = np; i < n; ++i) {
			y[i] = x[i];
		}
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::embedding_lookup, transform_type, core_type, float, float, int32_t>
		: public kernel_base<kernel_types::embedding_lookup, core_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			const uint64_t ne00 = input01[0];
			const uint64_t ne01 = input01[1];
			const uint64_t ne02 = input01[2];
			const uint64_t ne10 = input02[0];
			const uint64_t ne11 = input02[1];

			const uint64_t nr  = count_elements(input02);
			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + nth - 1ull) / nth;
			const uint64_t ir0 = dr * ith;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const float* __restrict input01_data   = input01.data;
			const int32_t* __restrict input02_data = input02.data;
			float* __restrict output_data		   = output.data;

			const uint64_t blocks_per_row		   = ne00 / Q_SIZE;
			const uint64_t output_elements_per_row = ne00;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12		   = i / (ne11 * ne10);
				const uint64_t i11		   = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10		   = (i - i12 * ne11 * ne10 - i11 * ne10);
				const uint64_t input02_idx = i10 + i11 * ne10 + i12 * (ne10 * ne11);
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * blocks_per_row + i11 * (ne01 * blocks_per_row) + i12 * (ne01 * ne02 * blocks_per_row);
				const float* __restrict src_ptr	 = &input01_data[input01_block_idx];

				const uint64_t output_idx = i10 * output_elements_per_row + i11 * (output[1] * output_elements_per_row) + i12 * (output[1] * output[2] * output_elements_per_row);
				float* __restrict dst_ptr = &output_data[output_idx];

				vec_cpy_f32(ne00, dst_ptr, src_ptr);
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::fused_qkv_rope, transform_type, core_type, float, float, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::fused_attention, transform_type, core_type, float, float, float> {
		using output_type	= typename core_type::output_type;
		using input_01_type = typename core_type::input_01_type::output_type;
		using input_02_type = typename core_type::input_02_type::output_type;
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
			//std::cout << "CURRENT TYPE: " << typeid(output_type).name() << std::endl;
			//std::cout << "CURRENT 00 TYPE: " << typeid(input_01_type).name() << std::endl;
			//std::cout << "CURRENT 01 TYPE: " << typeid(input_02_type).name() << std::endl;
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::fused_attn_out, transform_type, core_type, float, float, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::fused_swiglu, transform_type, core_type, float, float, block_q8_0<half>, block_q8_0<half>, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&, const typename core_type::input_04_type&, const typename core_type::input_05_type&) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::fused_qkv_rope, transform_type, core_type, float, float, float, float, block_q8_0<half>, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03, const typename core_type::input_04_type& input04,
			const typename core_type::input_05_type& input05, const typename core_type::input_06_type& input06) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::fused_swiglu, transform_type, core_type, float, float, block_q8_0<half>, block_q8_0<half>, block_q8_0<half>, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&, const typename core_type::input_04_type&, const typename core_type::input_05_type&) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::fused_final_proj, transform_type, core_type, float, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::fused_final_proj, core_type, float, float, float, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&) {
		}
	};

};

#endif