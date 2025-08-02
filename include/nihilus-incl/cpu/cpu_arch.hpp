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

	NIHILUS_INLINE static void vec_scale_f32(const uint64_t n, float* __restrict y, const float v) {
		const uint64_t np = (n & ~(Q_SIZE - 1));

		__m256 vx = _mm256_set1_ps(v);

		__m256 ay[(Q_SIZE / 8)];

		for (uint64_t i = 0; i < np; i += Q_SIZE) {
			for (uint64_t j = 0; j < (Q_SIZE / 8); j++) {
				ay[j] = _mm256_loadu_ps(y + i + j * 8);
				ay[j] = _mm256_mul_ps(ay[j], vx);

				_mm256_storeu_ps(y + i + j * 8, ay[j]);
			}
		}

		for (uint64_t i = np; i < n; ++i) {
			y[i] *= v;
		}
	}

	NIHILUS_INLINE float sqrtf(float x) {
		return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type&) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			static constexpr uint64_t ne0 = core_type::get_array()[0];
			const uint64_t ne1			  = output[1];
			static constexpr uint64_t ne2 = core_type::get_array()[2];
			static constexpr uint64_t ne3 = core_type::get_array()[3];

			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);

			const float* __restrict input01_data = input01.data;
			float* __restrict output_data		 = output.data;

			const int64_t src_row_elements	  = ne00;
			const int64_t src_plane_elements  = ne00 * ne01;
			const int64_t src_volume_elements = ne00 * ne01 * ne02;

			const int64_t dst_row_elements	  = ne0;
			const int64_t dst_plane_elements  = ne0 * ne1;
			const int64_t dst_volume_elements = ne0 * ne1 * ne2;

			static constexpr float eps = core_type::model_traits_type::layer_norm_rms_epsilon;

			for (int64_t i03 = 0; i03 < ne03; i03++) {
				for (int64_t i02 = 0; i02 < ne02; i02++) {
					for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
						int64_t src_offset		  = i03 * src_volume_elements + i02 * src_plane_elements + i01 * src_row_elements;
						const float* __restrict x = &input01_data[src_offset];

						float sum = 0.0;
						for (int64_t i00 = 0; i00 < ne00; i00++) {
							sum += x[i00] * x[i00];
						}
						const float mean = sum / ne00;

						int64_t dst_offset	= i03 * dst_volume_elements + i02 * dst_plane_elements + i01 * dst_row_elements;
						float* __restrict y = &output_data[dst_offset];

						memcpy(y, x, ne00 * sizeof(float));
						const float scale = 1.0f / sqrtf(mean + eps);
						vec_scale_f32(ne00, y, scale);
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::rms_norm_mul, core_type, float, float, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type&) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			static constexpr uint64_t ne0 = core_type::get_array()[0];
			const uint64_t ne1			  = output[1];
			static constexpr uint64_t ne2 = core_type::get_array()[2];
			static constexpr uint64_t ne3 = core_type::get_array()[3];

			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);

			const float* __restrict input01_data = input01.data;
			float* __restrict output_data		 = output.data;

			const int64_t src_row_elements	  = ne00;
			const int64_t src_plane_elements  = ne00 * ne01;
			const int64_t src_volume_elements = ne00 * ne01 * ne02;

			const int64_t dst_row_elements	  = ne0;
			const int64_t dst_plane_elements  = ne0 * ne1;
			const int64_t dst_volume_elements = ne0 * ne1 * ne2;

			static constexpr float eps = core_type::model_traits_type::layer_norm_rms_epsilon;

			for (int64_t i03 = 0; i03 < ne03; i03++) {
				for (int64_t i02 = 0; i02 < ne02; i02++) {
					for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
						int64_t src_offset		  = i03 * src_volume_elements + i02 * src_plane_elements + i01 * src_row_elements;
						const float* __restrict x = &input01_data[src_offset];

						float sum = 0.0;
						for (int64_t i00 = 0; i00 < ne00; i00++) {
							sum += x[i00] * x[i00];
						}
						const float mean = sum / ne00;

						int64_t dst_offset	= i03 * dst_volume_elements + i02 * dst_plane_elements + i01 * dst_row_elements;
						float* __restrict y = &output_data[dst_offset];

						memcpy(y, x, ne00 * sizeof(float));
						const float scale = 1.0f / sqrtf(mean + eps);
						vec_scale_f32(ne00, y, scale);
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::rms_norm, core_type, float, float> {
		using input_type01 = core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			static constexpr uint64_t ne0 = core_type::get_array()[0];
			const uint64_t ne1			  = output[1];
			static constexpr uint64_t ne2 = core_type::get_array()[2];
			static constexpr uint64_t ne3 = core_type::get_array()[3];

			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);

			const float* __restrict input01_data = input01.data;
			float* __restrict output_data		 = output.data;

			const int64_t src_row_elements	  = ne00;
			const int64_t src_plane_elements  = ne00 * ne01;
			const int64_t src_volume_elements = ne00 * ne01 * ne02;

			const int64_t dst_row_elements	  = ne0;
			const int64_t dst_plane_elements  = ne0 * ne1;
			const int64_t dst_volume_elements = ne0 * ne1 * ne2;

			static constexpr float eps = core_type::model_traits_type::layer_norm_rms_epsilon;

			for (int64_t i03 = 0; i03 < ne03; i03++) {
				for (int64_t i02 = 0; i02 < ne02; i02++) {
					for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
						int64_t src_offset		  = i03 * src_volume_elements + i02 * src_plane_elements + i01 * src_row_elements;
						const float* __restrict x = &input01_data[src_offset];

						float sum = 0.0;
						for (int64_t i00 = 0; i00 < ne00; i00++) {
							sum += x[i00] * x[i00];
						}
						const float mean = sum / ne00;

						int64_t dst_offset	= i03 * dst_volume_elements + i02 * dst_plane_elements + i01 * dst_row_elements;
						float* __restrict y = &output_data[dst_offset];

						memcpy(y, x, ne00 * sizeof(float));
						const float scale = 1.0f / sqrtf(mean + eps);
						vec_scale_f32(ne00, y, scale);
					}
				}
			}
		}
	};

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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
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
			__m256 ax0 = _mm256_loadu_ps(x + i);
			__m256 ax1 = _mm256_loadu_ps(x + i + 8);
			__m256 ax2 = _mm256_loadu_ps(x + i + 16);
			__m256 ax3 = _mm256_loadu_ps(x + i + 24);
			_mm256_storeu_ps(y + i, ax0);
			_mm256_storeu_ps(y + i + 8, ax1);
			_mm256_storeu_ps(y + i + 16, ax2);
			_mm256_storeu_ps(y + i + 24, ax3);
		}
		for (uint64_t i = np; i < n; ++i) {
			y[i] = x[i];
		}
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, float, int32_t> {
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t nr  = ne01 * ne02 * ne03;
			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + nth - 1ull) / nth;
			const uint64_t ir0 = dr * ith;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const float* __restrict input01_data = input01.data;
			const float* __restrict input02_data = input02.data;
			float* __restrict output_data		  = output.data;

			static constexpr uint64_t input01_stride_dim1 = ne00;
			const uint64_t input01_stride_dim2			  = ne00 * ne01;
			const uint64_t input01_stride_dim3			  = ne00 * ne01 * ne02;
			static constexpr uint64_t input02_stride_dim1 = ne10;
			const uint64_t input02_stride_dim2			  = ne10 * ne11;
			const uint64_t input02_stride_dim3			  = ne10 * ne11 * ne12;
			static constexpr uint64_t output_stride_dim1  = ne00;
			const uint64_t output_stride_dim2			  = output[1] * ne00;
			const uint64_t output_stride_dim3			  = output[1] * output[2] * ne00;

			const bool is_broadcasting = (ne11 == 1);

			static constexpr uint64_t simd_width = 8;
			const uint64_t ne00_simd			 = ne00 & ~(simd_width - 1);
			const uint64_t ne00_remainder		 = ne00 - ne00_simd;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i13 = i / (ne02 * ne01);
				const uint64_t i12 = (i - i13 * ne02 * ne01) / ne01;
				const uint64_t i11 = i - i13 * ne02 * ne01 - i12 * ne01;

				const uint64_t input01_base = i11 * input01_stride_dim1 + i12 * input01_stride_dim2 + i13 * input01_stride_dim3;
				const uint64_t output_base	= i11 * output_stride_dim1 + i12 * output_stride_dim2 + i13 * output_stride_dim3;

				uint64_t input02_base;
				if (is_broadcasting) {
					input02_base = 0 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				} else {
					input02_base = i11 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				}

				uint64_t i10 = 0;
				for (; i10 < ne00_simd; i10 += simd_width) {
					__m256 input01_vec = _mm256_loadu_ps(&input01_data[input01_base + i10]);
					__m256 input02_vec = _mm256_loadu_ps(&input02_data[input02_base + i10]);
					__m256 result_vec  = _mm256_mul_ps(input01_vec, input02_vec);
					_mm256_storeu_ps(&output_data[output_base + i10], result_vec);
				}

				for (; i10 < ne00; ++i10) {
					const uint64_t input01_idx = input01_base + i10;
					const uint64_t input02_idx = input02_base + i10;
					const uint64_t output_idx  = output_base + i10;

					output_data[output_idx] = input01_data[input01_idx] * input02_data[input02_idx];
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			const uint64_t ne10			   = input02[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];
			static constexpr uint64_t ne0  = ne01;
			const uint64_t ne1			   = ne11;
			static constexpr uint64_t ne2  = ne12;
			static constexpr uint64_t ne3  = ne13;

			block_q8_0<half>* src0_data;
			if constexpr (array_types<decltype(input01.data)>) {
				src0_data = input01.data[0];
			} else {
				src0_data = input01.data;
			}
			const float* __restrict src1_data = input02.data;
			float* __restrict dst_data		   = output.data;

			static constexpr uint64_t nb01 = ne00 / Q_SIZE;
			static constexpr uint64_t nb02 = nb01 * ne01;
			static constexpr uint64_t nb03 = nb02 * ne02;
			const uint64_t nb11			   = ne10;
			const uint64_t nb12			   = nb11 * ne11;
			const uint64_t nb13			   = nb12 * ne12;
			const uint64_t nb1			   = ne0;
			const uint64_t nb2			   = nb1 * ne1;
			const uint64_t nb3			   = nb2 * ne2;

			// Calculate work distribution for this thread
			const uint64_t total_work	   = ne13 * ne12 * ne11;
			const uint64_t work_per_thread = (total_work + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = std::min(work_start + work_per_thread, total_work);

			for (uint64_t work_idx = work_start; work_idx < work_end; ++work_idx) {
				// Convert linear work index back to 3D coordinates
				const uint64_t i11 = work_idx % ne11;
				const uint64_t i12 = (work_idx / ne11) % ne12;
				const uint64_t i13 = work_idx / (ne11 * ne12);

				static constexpr uint64_t r2	 = ne12 / ne02;
				static constexpr uint64_t r3	 = ne13 / ne03;
				const uint64_t i03				 = i13 / r3;
				const uint64_t i02				 = i12 / r2;
				const block_q8_0<half>* src0_row = src0_data + (i02 * nb02 + i03 * nb03);
				const float* __restrict src1_col			 = src1_data + (i11 + i12 * ne11 + i13 * ne12 * ne11);
				float* __restrict dst_col					 = dst_data + (i11 * nb1 + i12 * nb2 + i13 * nb3);

				// Process multiple output rows simultaneously for better cache usage
				static constexpr uint64_t UNROLL_ROWS = 4;
				uint64_t ir0						  = 0;

				// Process UNROLL_ROWS rows at once
				for (; ir0 + UNROLL_ROWS <= ne0; ir0 += UNROLL_ROWS) {
					__m256 results[UNROLL_ROWS];
					for (int i = 0; i < UNROLL_ROWS; i++) {
						results[i] = _mm256_setzero_ps();
					}

					const uint64_t num_blocks			   = ne00 / Q_SIZE;
					static constexpr uint64_t BLOCK_UNROLL = 2;
					uint64_t block_idx					   = 0;

					// Process blocks in pairs for better instruction-level parallelism
					for (; block_idx + BLOCK_UNROLL <= num_blocks; block_idx += BLOCK_UNROLL) {
						// Prefetch next blocks
						if (block_idx + BLOCK_UNROLL + 4 < num_blocks) {
							_mm_prefetch(( const char* )&src0_row[(ir0 * nb01) + block_idx + BLOCK_UNROLL + 4], _MM_HINT_T0);
							_mm_prefetch(( const char* )&src1_col[(block_idx + BLOCK_UNROLL + 4) * Q_SIZE], _MM_HINT_T0);
						}

						for (int row = 0; row < UNROLL_ROWS; row++) {
							const block_q8_0<half>* weight_blocks = src0_row + (ir0 + row) * nb01;

							for (int b = 0; b < BLOCK_UNROLL; b++) {
								const block_q8_0<half>& block = weight_blocks[block_idx + b];
								const __m256 scale_vec		  = _mm256_set1_ps(fp16_to_fp32(block.d));
								const float* __restrict input_ptr		  = &src1_col[(block_idx + b) * Q_SIZE];
								const int8_t* weight_ptr	  = block.qs;

								// Process 32 elements in 4 groups of 8
								__m256 acc0 = _mm256_setzero_ps();
								__m256 acc1 = _mm256_setzero_ps();
								__m256 acc2 = _mm256_setzero_ps();
								__m256 acc3 = _mm256_setzero_ps();

								// Load and convert weights (8 elements at a time)
								__m256i w0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr)));
								__m256i w1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 8)));
								__m256i w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 16)));
								__m256i w3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 24)));

								__m256 wf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(w0), scale_vec);
								__m256 wf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(w1), scale_vec);
								__m256 wf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(w2), scale_vec);
								__m256 wf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(w3), scale_vec);

								// Load input values
								__m256 v0 = _mm256_loadu_ps(input_ptr);
								__m256 v1 = _mm256_loadu_ps(input_ptr + 8);
								__m256 v2 = _mm256_loadu_ps(input_ptr + 16);
								__m256 v3 = _mm256_loadu_ps(input_ptr + 24);

								// Fused multiply-add
								acc0 = _mm256_fmadd_ps(wf0, v0, acc0);
								acc1 = _mm256_fmadd_ps(wf1, v1, acc1);
								acc2 = _mm256_fmadd_ps(wf2, v2, acc2);
								acc3 = _mm256_fmadd_ps(wf3, v3, acc3);

								// Sum all accumulators
								__m256 sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
								results[row]   = _mm256_add_ps(results[row], sum_vec);
							}
						}
					}

					// Handle remaining blocks
					for (; block_idx < num_blocks; ++block_idx) {
						for (int row = 0; row < UNROLL_ROWS; row++) {
							const block_q8_0<half>* weight_blocks = src0_row + (ir0 + row) * nb01;
							const block_q8_0<half>& block		  = weight_blocks[block_idx];
							const __m256 scale_vec				  = _mm256_set1_ps(fp16_to_fp32(block.d));
							const float* __restrict input_ptr				  = &src1_col[block_idx * Q_SIZE];
							const int8_t* weight_ptr			  = block.qs;

							__m256 acc0 = _mm256_setzero_ps();
							__m256 acc1 = _mm256_setzero_ps();
							__m256 acc2 = _mm256_setzero_ps();
							__m256 acc3 = _mm256_setzero_ps();

							__m256i w0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr)));
							__m256i w1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 8)));
							__m256i w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 16)));
							__m256i w3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 24)));

							__m256 wf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(w0), scale_vec);
							__m256 wf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(w1), scale_vec);
							__m256 wf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(w2), scale_vec);
							__m256 wf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(w3), scale_vec);

							__m256 v0 = _mm256_loadu_ps(input_ptr);
							__m256 v1 = _mm256_loadu_ps(input_ptr + 8);
							__m256 v2 = _mm256_loadu_ps(input_ptr + 16);
							__m256 v3 = _mm256_loadu_ps(input_ptr + 24);

							acc0 = _mm256_fmadd_ps(wf0, v0, acc0);
							acc1 = _mm256_fmadd_ps(wf1, v1, acc1);
							acc2 = _mm256_fmadd_ps(wf2, v2, acc2);
							acc3 = _mm256_fmadd_ps(wf3, v3, acc3);

							__m256 sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
							results[row]   = _mm256_add_ps(results[row], sum_vec);
						}
					}

					// Efficient horizontal reduction and store results
					for (int row = 0; row < UNROLL_ROWS; row++) {
						// Optimized horizontal sum
						__m128 low		   = _mm256_castps256_ps128(results[row]);
						__m128 high		   = _mm256_extractf128_ps(results[row], 1);
						__m128 sum128	   = _mm_add_ps(low, high);
						sum128			   = _mm_hadd_ps(sum128, sum128);
						sum128			   = _mm_hadd_ps(sum128, sum128);
						dst_col[ir0 + row] = _mm_cvtss_f32(sum128);
					}
				}

				// Handle remaining rows
				for (; ir0 < ne0; ++ir0) {
					const block_q8_0<half>* weight_blocks = src0_row + ir0 * nb01;
					const uint64_t num_blocks			  = ne00 / Q_SIZE;

					__m256 acc0 = _mm256_setzero_ps();
					__m256 acc1 = _mm256_setzero_ps();
					__m256 acc2 = _mm256_setzero_ps();
					__m256 acc3 = _mm256_setzero_ps();

					for (uint64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
						const block_q8_0<half>& block = weight_blocks[block_idx];
						const __m256 scale_vec		  = _mm256_set1_ps(fp16_to_fp32(block.d));
						const float* __restrict input_ptr		  = &src1_col[block_idx * Q_SIZE];
						const int8_t* weight_ptr	  = block.qs;

						__m256i w0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr)));
						__m256i w1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 8)));
						__m256i w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 16)));
						__m256i w3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(( __m128i* )(weight_ptr + 24)));

						__m256 wf0 = _mm256_mul_ps(_mm256_cvtepi32_ps(w0), scale_vec);
						__m256 wf1 = _mm256_mul_ps(_mm256_cvtepi32_ps(w1), scale_vec);
						__m256 wf2 = _mm256_mul_ps(_mm256_cvtepi32_ps(w2), scale_vec);
						__m256 wf3 = _mm256_mul_ps(_mm256_cvtepi32_ps(w3), scale_vec);

						__m256 v0 = _mm256_loadu_ps(input_ptr);
						__m256 v1 = _mm256_loadu_ps(input_ptr + 8);
						__m256 v2 = _mm256_loadu_ps(input_ptr + 16);
						__m256 v3 = _mm256_loadu_ps(input_ptr + 24);

						acc0 = _mm256_fmadd_ps(wf0, v0, acc0);
						acc1 = _mm256_fmadd_ps(wf1, v1, acc1);
						acc2 = _mm256_fmadd_ps(wf2, v2, acc2);
						acc3 = _mm256_fmadd_ps(wf3, v3, acc3);
					}

					__m256 sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
					// Optimized horizontal reduction
					__m128 low	  = _mm256_castps256_ps128(sum_vec);
					__m128 high	  = _mm256_extractf128_ps(sum_vec, 1);
					__m128 sum128 = _mm_add_ps(low, high);
					sum128		  = _mm_hadd_ps(sum128, sum128);
					sum128		  = _mm_hadd_ps(sum128, sum128);
					dst_col[ir0]  = _mm_cvtss_f32(sum128);
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			std::cout << "MUL-MATTING TYPE: " << core_type::op_type << std::endl;
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];
			const uint64_t ne10			   = input02[0];
			const uint64_t ne11			   = input02[1];
			uint64_t element_count01	   = count_elements(input01);
			for (uint64_t x = 0; x < element_count01; ++x) {
				std::cout << input01.data[x] << ", ";
			}
			uint64_t element_count02 = count_elements(input02);
			std::cout << "MUL-MATTING TYPE: " << core_type::op_type << std::endl;
			for (uint64_t x = 0; x < element_count02; ++x) {
				std::cout << input02.data[x] << ", ";
			}
			const half* src0_data  = input01.data;
			const float* __restrict src1_data = input02.data;
			float* __restrict dst_data		   = output.data;

			static constexpr uint64_t r2 = ne12 / ne02;
			static constexpr uint64_t r3 = ne13 / ne03;

			for (uint64_t i13 = 0; i13 < ne13; ++i13) {
				for (uint64_t i12 = 0; i12 < ne12; ++i12) {
					for (uint64_t i11 = 0; i11 < ne11; ++i11) {
						const uint64_t i03 = i13 / r3;
						const uint64_t i02 = i12 / r2;

						for (uint64_t i01 = 0; i01 < ne01; ++i01) {
							float sum = 0.0f;

							for (uint64_t i00 = 0; i00 < ne00; ++i00) {
								const uint64_t src0_idx = i00 + i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;
								const uint64_t src1_idx = i00 + i11 * ne10 + i12 * ne10 * ne11 + i13 * ne10 * ne11 * ne12;

								sum += fp16_to_fp32(src0_data[src0_idx]) * src1_data[src1_idx];
							}

							const uint64_t dst_idx = i01 + i11 * ne01 + i12 * ne01 * ne11 + i13 * ne01 * ne11 * ne12;
							dst_data[dst_idx]	   = sum;
						}
					}
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::add, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];

			const uint64_t nr  = ne01 * ne02 * ne03;
			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + nth - 1ull) / nth;
			const uint64_t ir0 = dr * ith;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const float* __restrict input01_data = input01.data;
			const float* __restrict input02_data = input02.data;
			float* __restrict output_data		  = output.data;

			static constexpr uint64_t input01_stride_dim1 = ne00;
			const uint64_t input01_stride_dim2			  = ne00 * ne01;
			const uint64_t input01_stride_dim3			  = ne00 * ne01 * ne02;
			static constexpr uint64_t input02_stride_dim1 = ne10;
			const uint64_t input02_stride_dim2			  = ne10 * ne11;
			const uint64_t input02_stride_dim3			  = ne10 * ne11 * ne12;
			static constexpr uint64_t output_stride_dim1  = ne00;
			const uint64_t output_stride_dim2			  = output[1] * ne00;
			const uint64_t output_stride_dim3			  = output[1] * output[2] * ne00;

			const bool is_broadcasting = (ne11 == 1);

			static constexpr uint64_t simd_width = 8;
			const uint64_t ne00_simd			 = ne00 & ~(simd_width - 1);
			const uint64_t ne00_remainder		 = ne00 - ne00_simd;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i13 = i / (ne02 * ne01);
				const uint64_t i12 = (i - i13 * ne02 * ne01) / ne01;
				const uint64_t i11 = i - i13 * ne02 * ne01 - i12 * ne01;

				const uint64_t input01_base = i11 * input01_stride_dim1 + i12 * input01_stride_dim2 + i13 * input01_stride_dim3;
				const uint64_t output_base	= i11 * output_stride_dim1 + i12 * output_stride_dim2 + i13 * output_stride_dim3;

				uint64_t input02_base;
				if (is_broadcasting) {
					input02_base = 0 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				} else {
					input02_base = i11 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				}

				uint64_t i10 = 0;
				for (; i10 < ne00_simd; i10 += simd_width) {
					__m256 input01_vec = _mm256_loadu_ps(&input01_data[input01_base + i10]);

					__m256 input02_vec = _mm256_loadu_ps(&input02_data[input02_base + i10]);

					__m256 result_vec = _mm256_add_ps(input01_vec, input02_vec);

					_mm256_storeu_ps(&output_data[output_base + i10], result_vec);
				}

				for (; i10 < ne00; ++i10) {
					const uint64_t input01_idx = input01_base + i10;
					const uint64_t input02_idx = input02_base + i10;
					const uint64_t output_idx  = output_base + i10;

					output_data[output_idx] = input01_data[input01_idx] + input02_data[input02_idx];
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data  = input01.data;
			const float* __restrict mask_data = input02.data;
			float* __restrict dst_data		   = output.data;

			for (uint64_t i03 = 0; i03 < ne03; ++i03) {
				for (uint64_t i02 = 0; i02 < ne02; ++i02) {
					for (uint64_t i01 = 0; i01 < ne01; ++i01) {
						const uint64_t row_offset = i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;

						float max_val = -INFINITY;
						for (uint64_t i00 = 0; i00 < ne00; ++i00) {
							const uint64_t src_idx	= row_offset + i00;
							const uint64_t mask_idx = i00 + i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;

							float val = src_data[src_idx];
							if (mask_data != nullptr) {
								val += mask_data[mask_idx];
							}

							if (val > max_val) {
								max_val = val;
							}
						}

						float sum = 0.0f;
						for (uint64_t i00 = 0; i00 < ne00; ++i00) {
							const uint64_t src_idx	= row_offset + i00;
							const uint64_t mask_idx = i00 + i01 * ne00 + i02 * ne00 * ne01 + i03 * ne00 * ne01 * ne02;

							float val = src_data[src_idx];
							if (mask_data != nullptr) {
								val += mask_data[mask_idx];
							}

							float exp_val	  = expf(val - max_val);
							dst_data[src_idx] = exp_val;
							sum += exp_val;
						}

						const float inv_sum = 1.0f / sum;
						for (uint64_t i00 = 0; i00 < ne00; ++i00) {
							const uint64_t dst_idx = row_offset + i00;
							dst_data[dst_idx] *= inv_sum;
						}
					}
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			std::cout << "COPYING TYPE: " << core_type::op_type << std::endl;
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const half* src_data = input01.data;
			half* dst_data		 = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = ne00 * ne01 * ne02_runtime * ne03;

			for (uint64_t i = 0; i < total_elements; ++i) {
				dst_data[i] = src_data[i];
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::rope, core_type, float, float, int32_t, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		using input_type03 = typename core_type::input_03_type;

		static constexpr float constexpr_pow(float base, float exp) {
			if (exp == 0.0f)
				return 1.0f;
			if (exp == 1.0f)
				return base;

			float result		= 1.0f;
			int int_exp			= static_cast<int>(exp);
			float current_power = base;

			while (int_exp > 0) {
				if (int_exp & 1)
					result *= current_power;
				current_power *= current_power;
				int_exp >>= 1;
			}

			float frac = exp - static_cast<int>(exp);
			if (frac > 0.0f) {
				result *= (1.0f + frac * (base - 1.0f) / base);
			}

			return result;
		}

		template<size_t N> static constexpr auto make_freq_table() {
			array<float, N> freqs{};
			constexpr float rope_freq_base = core_type::model_traits_type::rope_freq_base;
			constexpr uint32_t rope_dim	   = core_type::model_traits_type::rope_dimension_count;

			for (size_t i = 0; i < N; ++i) {
				const float freq_exponent = (2.0f * static_cast<float>(i)) / static_cast<float>(rope_dim);
				const float theta_power	  = constexpr_pow(rope_freq_base, freq_exponent);
				freqs[i]				  = 1.0f / theta_power;
			}
			return freqs;
		}

		static constexpr float rope_freq_base		   = core_type::model_traits_type::rope_freq_base;
		static constexpr uint32_t rope_dimension_count = core_type::model_traits_type::rope_dimension_count;
		static constexpr uint64_t head_dim			   = core_type::model_traits_type::head_dim;
		static constexpr uint32_t attention_head_count = core_type::model_traits_type::attention_head_count;

		static constexpr uint64_t batch_size	  = input_type01::get_array()[0];
		static constexpr uint64_t num_heads		  = input_type01::get_array()[2];
		static constexpr uint64_t tensor_head_dim = input_type01::get_array()[3];

		static constexpr uint64_t rope_dim		= rope_dimension_count;
		static constexpr uint64_t half_rope_dim = rope_dim / 2;

		static constexpr auto freq_table = make_freq_table<half_rope_dim>();

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
			/*
			
			const float* __restrict src_data		   = input01.data;
			const int32_t* pos_data		   = input02.data;
			const float* __restrict freq_scaling_data = input03.data;
			float* __restrict dst_data				   = output.data;

			const uint64_t seq_len = input01[1];

			const uint64_t total_work_items = batch_size * seq_len * num_heads;
			const uint64_t total_elements	= batch_size * seq_len * num_heads * head_dim;

			const uint64_t work_per_thread = (total_work_items + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = (work_start + work_per_thread < total_work_items) ? work_start + work_per_thread : total_work_items;

			const int nr = total_work_items;
			int ir		 = 0;

			for (int64_t i3 = 0; i3 < batch_size; i3++) {
				for (int64_t i2 = 0; i2 < seq_len; i2++) {
					const int64_t position = pos_data[i2];

					for (int64_t i1 = 0; i1 < num_heads; i1++) {
						if (ir++ < work_start)
							continue;
						if (ir > work_end)
							break;

						const uint64_t src_offset = i3 * seq_len * num_heads * head_dim + i2 * num_heads * head_dim + i1 * head_dim;
						const uint64_t dst_offset = i3 * seq_len * num_heads * head_dim + i2 * num_heads * head_dim + i1 * head_dim;

						for (int64_t i0 = 0; i0 < rope_dim; i0 += 2) {
							const uint64_t dim_pair = i0 / 2;
							float freq				= freq_table[dim_pair];

							if (freq_scaling_data != nullptr) {
								const uint64_t scaling_idx = (dim_pair < input_type02::get_array()[0]) ? dim_pair : 0;
								freq *= freq_scaling_data[scaling_idx];
							}

							const float angle	  = static_cast<float>(position) * freq;
							const float cos_theta = cosf(angle);
							const float sin_theta = sinf(angle);

							const float x0 = src_data[src_offset + i0];
							const float x1 = src_data[src_offset + i0 + 1];

							dst_data[dst_offset + i0]	  = x0 * cos_theta - x1 * sin_theta;
							dst_data[dst_offset + i0 + 1] = x0 * sin_theta + x1 * cos_theta;
						}

						for (int64_t i0 = rope_dim; i0 < head_dim; i0++) {
							dst_data[dst_offset + i0] = src_data[src_offset + i0];
						}
					}
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::copy, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			/*
			std::cout << "COPYING TYPE: " << core_type::op_type << std::endl;
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			float* __restrict dst_data		  = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = count_elements(output);

			for (uint64_t i = 0; i < total_elements; ++i) {
				dst_data[i] = src_data[i];
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			/*
			std::cout << "COPYING TYPE: " << core_type::op_type << std::endl;
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			half* dst_data		  = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = count_elements(output);

			for (uint64_t i = 0; i < total_elements; ++i) {
				dst_data[i] = fp32_to_fp16(src_data[i]);
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::cont, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			/*
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			float* __restrict dst_data		  = output.data;

			const uint64_t ne02			  = input01[2];
			const uint64_t total_elements = ne00 * ne01 * ne02 * ne03;

			const uint64_t work_per_thread = (total_elements + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = detail::min(work_start + work_per_thread, total_elements);

			const uint64_t src_stride0 = input01.strides[0];
			const uint64_t src_stride1 = input01.strides[1];
			const uint64_t src_stride2 = input01.strides[2];
			const uint64_t src_stride3 = input01.strides[3];

			for (uint64_t linear_idx = work_start; linear_idx < work_end; ++linear_idx) {
				const uint64_t i3		  = linear_idx / (ne00 * ne01 * ne02);
				const uint64_t remaining3 = linear_idx % (ne00 * ne01 * ne02);
				const uint64_t i2		  = remaining3 / (ne00 * ne01);
				const uint64_t remaining2 = remaining3 % (ne00 * ne01);
				const uint64_t i1		  = remaining2 / ne00;
				const uint64_t i0		  = remaining2 % ne00;

				const uint64_t src_idx = i3 * src_stride3 + i2 * src_stride2 + i1 * src_stride1 + i0 * src_stride0;
				dst_data[linear_idx]   = src_data[src_idx];
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

#endif

}
