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

#include <nihilus-incl/cpu/simd/avx_2.hpp>
#include <nihilus-incl/cpu/simd/avx_512.hpp>
#include <nihilus-incl/cpu/simd/arm_neon.hpp>
#include <nihilus-incl/cpu/simd/arm_sve2.hpp>

namespace nihilus {

#if !defined(NIHILUS_AVX2) && !defined(NIHILUS_AVX512) && !defined(NIHILUS_NEON) && !defined(NIHILUS_SVE2)

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm_mul, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	NIHILUS_INLINE static void dequantize_row_q8_0(const block_q8_0<half>* __restrict x, float* __restrict y, uint64_t k) {
		static constexpr int64_t qk = Q_SIZE;

		const uint64_t nb = k / qk;

		for (uint64_t i = 0; i < nb; i++) {
			const float d = fp16_to_fp32(x[i].d);

			for (uint64_t j = 0; j < qk; ++j) {
				y[i * qk + j] = x[i].qs[j] * d;
			}
		}
	}

	NIHILUS_INLINE static void vec_cpy_f32(const uint64_t n, float* y, const float* x) {
		for (uint64_t i = 0; i < n; ++i) {
			y[i] = x[i];
		}
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
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

			const block_q8_0<half>* input01_data = input01.data;
			const int32_t* input02_data			 = input02.data;
			float* output_data					 = output.data;

			const uint64_t blocks_per_row		   = ne00 / Q_SIZE;
			const uint64_t output_elements_per_row = ne00;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12 = i / (ne11 * ne10);
				const uint64_t i11 = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);

				const uint64_t input02_idx = i10 + i11 * ne10 + i12 * (ne10 * ne11);
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * blocks_per_row + i11 * (ne01 * blocks_per_row) + i12 * (ne01 * ne02 * blocks_per_row);
				const block_q8_0<half>* src_ptr	 = &input01_data[input01_block_idx];

				const uint64_t output_idx = i10 * output_elements_per_row + i11 * (output[1] * output_elements_per_row) + i12 * (output[1] * output[2] * output_elements_per_row);
				float* dst_ptr			  = &output_data[output_idx];

				dequantize_row_q8_0(src_ptr, dst_ptr, ne00);
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, float, int32_t> {
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

			const float* input01_data	= input01.data;
			const int32_t* input02_data = input02.data;
			float* output_data			= output.data;

			const uint64_t blocks_per_row		   = ne00 / Q_SIZE;
			const uint64_t output_elements_per_row = ne00;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12 = i / (ne11 * ne10);
				const uint64_t i11 = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);

				const uint64_t input02_idx = i10 + i11 * ne10 + i12 * (ne10 * ne11);
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * blocks_per_row + i11 * (ne01 * blocks_per_row) + i12 * (ne01 * ne02 * blocks_per_row);
				const float* src_ptr			 = &input01_data[input01_block_idx];

				const uint64_t output_idx = i10 * output_elements_per_row + i11 * (output[1] * output_elements_per_row) + i12 * (output[1] * output[2] * output_elements_per_row);
				float* dst_ptr			  = &output_data[output_idx];

				vec_cpy_f32(ne00, dst_ptr, src_ptr);
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&) {
		}
	};

#endif

}
