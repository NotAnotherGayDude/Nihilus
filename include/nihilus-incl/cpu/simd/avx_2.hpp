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

#if defined(NIHILUS_AVX2)

namespace nihilus {

	NIHILUS_INLINE static void vec_scale_f32(const uint64_t n, float* y, const float v) {
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type&) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);

			const float* input01_data = input01.data;
			float* output_data		  = output.data;

			static constexpr float eps = core_type::model_traits_type::layer_norm_rms_epsilon;
			for (uint64_t i03 = 0; i03 < ne03; i03++) {
				for (uint64_t i02 = 0; i02 < ne02; i02++) {
					for (uint64_t i01 = ith; i01 < ne01; i01 += nth) {
						const float* x = ( float* )(( char* )input01_data + i01 + i02 + i03);

						double sum = 0.0;
						for (int64_t i00 = 0; i00 < ne00; i00++) {
							sum += static_cast<double>(x[i00] * x[i00]);
						}

						const float mean = sum / ne00;

						float* y = ( float* )(( char* )output_data + i01 + i02 + i03);

						memcpy(y, x, ne00 * sizeof(float));

						const float scale = 1.0f / sqrtf(mean + eps);
						vec_scale_f32(static_cast<int>(ne00), y, scale);
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm_mul, core_type, float, float, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type&) {
			static constexpr uint64_t ne00	= input_type01::get_array()[0];
			const uint64_t ne01				= input01[1];
			static constexpr uint64_t ne02	= input_type01::get_array()[2];
			static constexpr uint64_t ne03	= input_type01::get_array()[3];
			const uint64_t ith				= static_cast<uint64_t>(thread_index);
			const uint64_t nth				= static_cast<uint64_t>(thread_count);
			const float* input01_data		= input01.data;
			float* output_data				= output.data;
			static constexpr float epsilon	= core_type::model_traits_type::layer_norm_rms_epsilon;
			static constexpr float inv_ne00 = 1.0f / static_cast<float>(ne00);
			for (uint64_t i03 = 0; i03 < ne03; i03++) {
				for (uint64_t i02 = 0; i02 < ne02; i02++) {
					for (uint64_t i01 = ith; i01 < ne01; i01 += nth) {
						const uint64_t offset = (i03 * ne02 * ne01 + i02 * ne01 + i01) * ne00;
						const float* x		  = &input01_data[offset];
						float* y			  = &output_data[offset];

						float sum		 = 0.0f;
						const int64_t np = static_cast<int64_t>(ne00) & ~(Q_SIZE - 1);
						__m256 sum_vec	 = _mm256_setzero_ps();
						for (uint64_t i = 0; i < np; i += Q_SIZE) {
							for (uint64_t j = 0; j < 4; j++) {
								__m256 ax = _mm256_loadu_ps(x + i + j * 8);
								ax		  = _mm256_mul_ps(ax, ax);
								sum_vec	  = _mm256_add_ps(sum_vec, ax);
							}
						}
						alignas(Q_SIZE) float sum_array[8];
						_mm256_store_ps(sum_array, sum_vec);
						for (uint64_t i = 0; i < 8; i++) {
							sum += sum_array[i];
						}
						for (uint64_t i = np; i < ne00; ++i) {
							const float xi = x[i];
							sum += xi * xi;
						}
						const float mean  = sum * inv_ne00;
						const float scale = 1.0f / sqrtf(mean + epsilon);
						std::memcpy(y, x, ne00 * sizeof(float));
						vec_scale_f32(static_cast<uint64_t>(ne00), y, scale);
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		using input_type01 = core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			static constexpr uint64_t ne00	= input_type01::get_array()[0];
			const uint64_t ne01				= input01[1];
			static constexpr uint64_t ne02	= input_type01::get_array()[2];
			static constexpr uint64_t ne03	= input_type01::get_array()[3];
			const uint64_t ith				= static_cast<uint64_t>(thread_index);
			const uint64_t nth				= static_cast<uint64_t>(thread_count);
			const float* input01_data		= input01.data;
			float* output_data				= output.data;
			static constexpr float epsilon	= core_type::model_traits_type::layer_norm_rms_epsilon;
			static constexpr float inv_ne00 = 1.0f / static_cast<float>(ne00);
			for (uint64_t i03 = 0; i03 < ne03; i03++) {
				for (uint64_t i02 = 0; i02 < ne02; i02++) {
					for (uint64_t i01 = ith; i01 < ne01; i01 += nth) {
						const uint64_t offset = (i03 * ne02 * ne01 + i02 * ne01 + i01) * ne00;
						const float* x		  = &input01_data[offset];
						float* y			  = &output_data[offset];

						float sum		 = 0.0f;
						const int64_t np = static_cast<int64_t>(ne00) & ~(Q_SIZE - 1);
						__m256 sum_vec	 = _mm256_setzero_ps();
						for (uint64_t i = 0; i < np; i += Q_SIZE) {
							for (uint64_t j = 0; j < 4; j++) {
								__m256 ax = _mm256_loadu_ps(x + i + j * 8);
								ax		  = _mm256_mul_ps(ax, ax);
								sum_vec	  = _mm256_add_ps(sum_vec, ax);
							}
						}
						alignas(Q_SIZE) float sum_array[8];
						_mm256_store_ps(sum_array, sum_vec);
						for (uint64_t i = 0; i < 8; i++) {
							sum += sum_array[i];
						}
						for (uint64_t i = np; i < ne00; ++i) {
							const float xi = x[i];
							sum += xi * xi;
						}
						const float mean  = sum * inv_ne00;
						const float scale = 1.0f / sqrtf(mean + epsilon);
						std::memcpy(y, x, ne00 * sizeof(float));
						vec_scale_f32(static_cast<uint64_t>(ne00), y, scale);
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne10 = input_type02::get_array()[0];
			const uint64_t ne11			   = input02[1];

			const uint64_t nr  = count_elements(input02);
			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);
			const uint64_t dr  = (nr + nth - 1ull) / nth;
			const uint64_t ir0 = dr * ith;
			const uint64_t ir1 = detail::min(ir0 + dr, nr);

			const block_q8_0<half>* input01_data = input01.data;
			const int32_t* input02_data			 = input02.data;
			float* output_data					 = output.data;

			static constexpr uint64_t blocks_per_row		  = ne00 / Q_SIZE;
			static constexpr uint64_t output_elements_per_row = ne00;

			static constexpr uint64_t log2_ne10		= 5;
			static constexpr uint64_t log2_ne11		= 6;
			static constexpr uint64_t log2_ne10ne11 = log2_ne10 + log2_ne11;

			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12 = i >> log2_ne10ne11;
				const uint64_t i11 = (i >> log2_ne10) & ((1ull << log2_ne11) - 1);
				const uint64_t i10 = i & ((1ull << log2_ne10) - 1);

				const uint64_t input02_idx = i10 + i11 * ne10 + i12 * (ne10 * ne11);
				const uint64_t token_id	   = static_cast<uint64_t>(input02_data[input02_idx]);

				const uint64_t input01_block_idx = token_id * blocks_per_row + i11 * (ne01 * blocks_per_row) + i12 * (ne01 * ne02 * blocks_per_row);

				const uint64_t output_idx = i10 * output_elements_per_row + i11 * (output[1] * output_elements_per_row) + i12 * (output[1] * output[2] * output_elements_per_row);
				dequantize_row_q8_0(&input01_data[input01_block_idx], &output_data[output_idx], ne00);
			}
		}
	};

	NIHILUS_INLINE static void vec_cpy_f32(const uint64_t n, float* y, const float* x) {
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
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
				const uint64_t i12		   = i / (ne11 * ne10);
				const uint64_t i11		   = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10		   = (i - i12 * ne11 * ne10 - i11 * ne10);
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;

		NIHILUS_INLINE static void nihilus_vec_dot_q8_0_f32(const uint64_t n, float* result, const block_q8_0<half>* x, const float* y) {
			uint64_t blocks						 = n / Q_SIZE;
			static constexpr uint64_t simd_width = 8;

			float sum	   = 0.0f;
			__m256 sum_vec = _mm256_setzero_ps();

			for (uint64_t block_idx = 0; block_idx < blocks; ++block_idx) {
				const block_q8_0<half>& block = x[block_idx];
				const float scale			  = fp16_to_fp32(block.d);
				const __m256 scale_vec		  = _mm256_set1_ps(scale);

				const float* y_block = y + block_idx * Q_SIZE;

				uint64_t q = 0;
				for (; q + simd_width <= Q_SIZE; q += simd_width) {
					__m128i q8_vals		= _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&block.qs[q]));
					__m256i q8_expanded = _mm256_cvtepi8_epi32(q8_vals);
					__m256 weight_vec	= _mm256_cvtepi32_ps(q8_expanded);
					weight_vec			= _mm256_mul_ps(weight_vec, scale_vec);
					__m256 input_vec	= _mm256_loadu_ps(y_block + q);
					sum_vec				= _mm256_fmadd_ps(weight_vec, input_vec, sum_vec);
				}

				for (; q < Q_SIZE; ++q) {
					const float weight = scale * static_cast<float>(block.qs[q]);
					const float input  = y_block[q];
					sum += weight * input;
				}
			}

			alignas(32) float sum_array[8];
			_mm256_store_ps(sum_array, sum_vec);
			for (uint64_t i = 0; i < 8; i++) {
				sum += sum_array[i];
			}

			*result = sum;
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
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
			const float* src1_data = input02.data;
			float* dst_data		   = output.data;

			static constexpr uint64_t nb00 = 1;
			static constexpr uint64_t nb01 = ne00 / Q_SIZE;
			static constexpr uint64_t nb02 = nb01 * ne01;
			static constexpr uint64_t nb03 = nb02 * ne02;

			static constexpr uint64_t nb10 = 1;
			const uint64_t nb11			   = ne10;
			const uint64_t nb12			   = nb11 * ne11;
			const uint64_t nb13			   = nb12 * ne12;

			static constexpr uint64_t nb0 = 1;
			const uint64_t nb1			  = ne0;
			const uint64_t nb2			  = nb1 * ne1;
			const uint64_t nb3			  = nb2 * ne2;

			const int64_t nr0 = static_cast<int64_t>(ne0);
			const int64_t nr1 = static_cast<int64_t>(ne1 * ne2 * ne3);

			if (thread_count == 1) {
				const int64_t ir0_start = 0;
				const int64_t ir0_end	= nr0;
				const int64_t ir1_start = 0;
				const int64_t ir1_end	= nr1;

				for (int64_t ir1 = ir1_start; ir1 < ir1_end; ++ir1) {
					const int64_t i13				 = ir1 / (ne12 * ne1);
					const int64_t i12				 = (ir1 - i13 * ne12 * ne1) / ne1;
					const int64_t i11				 = ir1 - i13 * ne12 * ne1 - i12 * ne1;
					static constexpr int64_t r2		 = ne12 / ne02;
					static constexpr int64_t r3		 = ne13 / ne03;
					const int64_t i03				 = i13 / r3;
					const int64_t i02				 = i12 / r2;
					const block_q8_0<half>* src0_row = src0_data + (i02 * nb02 + i03 * nb03);
					const float* src1_col			 = src1_data + (i11 + i12 * ne11 + i13 * ne12 * ne11);
					float* dst_col					 = dst_data + (i11 * nb1 + i12 * nb2 + i13 * nb3);

					for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
						nihilus_vec_dot_q8_0_f32(ne00, &dst_col[ir0], src0_row + ir0 * nb01, src1_col);
					}
				}
				return;
			}

			static constexpr int64_t blck_0 = 16;
			static constexpr int64_t blck_1 = 16;

			const int64_t ith = thread_index;
			const int64_t nth = thread_count;

			uint64_t chunk_size = 16;
			if (nr0 == 1 || nr1 == 1) {
				chunk_size = 64;
			}

			int64_t nchunk0 = static_cast<int64_t>((nr0 + chunk_size - 1) / chunk_size);
			int64_t nchunk1 = static_cast<int64_t>((nr1 + chunk_size - 1) / chunk_size);
			if (nchunk0 * nchunk1 < nth * 4) {
				nchunk0 = nr0 > nr1 ? nth : 1;
				nchunk1 = nr0 > nr1 ? 1 : nth;
			}

			const int64_t dr0  = (nr0 + nchunk0 - 1) / nchunk0;
			const int64_t dr1  = (nr1 + nchunk1 - 1) / nchunk1;
			const int64_t ith0 = ith % nchunk0;
			const int64_t ith1 = ith / nchunk0;

			const int64_t ir0_start = dr0 * ith0;
			const int64_t ir0_end	= detail::min(ir0_start + dr0, nr0);
			const int64_t ir1_start = dr1 * ith1;
			const int64_t ir1_end	= detail::min(ir1_start + dr1, nr1);

			if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
				return;
			}

			for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
				for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
					for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
						const int64_t i13				 = ir1 / (ne12 * ne1);
						const int64_t i12				 = (ir1 - i13 * ne12 * ne1) / ne1;
						const int64_t i11				 = ir1 - i13 * ne12 * ne1 - i12 * ne1;
						static constexpr int64_t r2		 = ne12 / ne02;
						static constexpr int64_t r3		 = ne13 / ne03;
						const int64_t i03				 = i13 / r3;
						const int64_t i02				 = i12 / r2;
						const block_q8_0<half>* src0_row = src0_data + (i02 * nb02 + i03 * nb03);
						const float* src1_col			 = src1_data + (i11 + i12 * ne11 + i13 * ne12 * ne11);
						float* dst_col					 = dst_data + (i11 * nb1 + i12 * nb2 + i13 * nb3);

						for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
							nihilus_vec_dot_q8_0_f32(ne00, &dst_col[ir0], src0_row + ir0 * nb01, src1_col);
						}
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void nihilus_vec_dot_f16_f32(const half* x, float* y, int64_t n) {
			int64_t i = 0;
			for (; i + 7 < n; i += 8) {
				__m128i x_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
				__m256 y_vec  = _mm256_cvtph_ps(x_vec);
				_mm256_storeu_ps(y + i, y_vec);
			}
			for (; i + 3 < n; i += 4) {
				__m128i x_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x + i));
				__m128 y_vec  = _mm_cvtph_ps(x_vec);
				_mm_storeu_ps(y + i, y_vec);
			}
			for (; i < n; ++i) {
				y[i] = fp16_to_fp32(x[i]);
			}
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];
			const uint64_t ne10			   = input02[0];
			const uint64_t ne11			   = input02[1];
			static constexpr uint64_t ne0  = ne01;
			const uint64_t ne1			   = ne11;
			static constexpr uint64_t ne2  = ne12;
			static constexpr uint64_t ne3  = ne13;
			const half* src0_data		   = input01.data;
			const float* src1_data		   = input02.data;
			float* dst_data				   = output.data;
			static constexpr uint64_t nb00 = 1;
			static constexpr uint64_t nb01 = ne00;
			static constexpr uint64_t nb02 = nb01 * ne01;
			static constexpr uint64_t nb03 = nb02 * ne02;

			static constexpr uint64_t nb10 = 1;
			const uint64_t nb11			   = ne10;
			const uint64_t nb12			   = nb11 * ne11;
			const uint64_t nb13			   = nb12 * ne12;

			static constexpr uint64_t nb0 = 1;
			const uint64_t nb1			  = ne0;
			const uint64_t nb2			  = nb1 * ne1;
			const uint64_t nb3			  = nb2 * ne2;

			static constexpr int64_t blck_0 = 16;
			static constexpr int64_t blck_1 = 16;

			const int64_t ith = thread_index;
			const int64_t nth = thread_count;

			static constexpr int64_t nr0 = ne0;
			const int64_t nr1			 = ne1 * ne2 * ne3;

			static constexpr int64_t base_chunk_size = 16;
			static constexpr int64_t alt_chunk_size	 = 64;

			const uint64_t chunk_size = (nr0 == 1 || nr1 == 1) ? alt_chunk_size : base_chunk_size;

			int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
			int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

			static constexpr int64_t thread_threshold = 4;
			if (nchunk0 * nchunk1 < nth * thread_threshold) {
				nchunk0 = nr0 > nr1 ? nth : 1;
				nchunk1 = nr0 > nr1 ? 1 : nth;
			}

			const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
			const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

			const int64_t ith0 = ith % nchunk0;
			const int64_t ith1 = ith / nchunk0;

			const int64_t ir0_start = dr0 * ith0;
			const int64_t ir0_end	= detail::min(ir0_start + dr0, nr0);
			const int64_t ir1_start = dr1 * ith1;
			const int64_t ir1_end	= detail::min(ir1_start + dr1, nr1);

			if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
				return;
			}

			static constexpr int64_t ne12_times_ne1_divisor = ne12;
			static constexpr int64_t r2						= ne12 / ne02;
			static constexpr int64_t r3						= ne13 / ne03;

			for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
				for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
					for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
						const int64_t i13 = ir1 / (ne12 * ne1);
						const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
						const int64_t i11 = ir1 - i13 * ne12 * ne1 - i12 * ne1;

						const int64_t i03 = i13 / r3;
						const int64_t i02 = i12 / r2;

						const half* src0_row  = src0_data + (i02 * nb02 + i03 * nb03);
						const float* src1_col = src1_data + (i11 + i12 * ne11 + i13 * ne12 * ne11);
						float* dst_col		  = dst_data + (i11 * nb1 + i12 * nb2 + i13 * nb3);

						for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
							nihilus_vec_dot_f16_f32(src0_row + ir0 * nb01, &dst_col[ir0], ne00);
						}
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&, const typename core_type::input_02_type&,
			const typename core_type::input_03_type&) {
		}
	};

};

#endif
