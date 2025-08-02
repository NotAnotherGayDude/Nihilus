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

	NIHILUS_INLINE static float unhalf(uint16_t d) {
		return fp16_to_fp32(d);
	}

	NIHILUS_INLINE void quantize_row_q8_0(const float* __restrict x, block_q8_0<half>* __restrict vy, int64_t k) {
		const int64_t nb = k / Q_SIZE;

		block_q8_0<half>* __restrict y = vy;

		for (int64_t i = 0; i < nb; i++) {
			__m256 v0 = _mm256_loadu_ps(x);
			__m256 v1 = _mm256_loadu_ps(x + 8);
			__m256 v2 = _mm256_loadu_ps(x + 16);
			__m256 v3 = _mm256_loadu_ps(x + 24);
			x += 32;

			const __m256 signBit = _mm256_set1_ps(-0.0f);
			__m256 maxAbs		 = _mm256_andnot_ps(signBit, v0);
			maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
			maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
			maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

			__m128 max4			  = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
			max4				  = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
			max4				  = _mm_max_ss(max4, _mm_movehdup_ps(max4));
			const float maxScalar = _mm_cvtss_f32(max4);

			const float d	 = maxScalar / 127.f;
			y[i].d			 = fp32_to_fp16(d);
			const float id	 = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
			const __m256 mul = _mm256_set1_ps(id);

			v0 = _mm256_mul_ps(v0, mul);
			v1 = _mm256_mul_ps(v1, mul);
			v2 = _mm256_mul_ps(v2, mul);
			v3 = _mm256_mul_ps(v3, mul);

			v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
			v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
			v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
			v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

			__m256i i0 = _mm256_cvtps_epi32(v0);
			__m256i i1 = _mm256_cvtps_epi32(v1);
			__m256i i2 = _mm256_cvtps_epi32(v2);
			__m256i i3 = _mm256_cvtps_epi32(v3);

			i0				   = _mm256_packs_epi32(i0, i1);
			i2				   = _mm256_packs_epi32(i2, i3);
			i0				   = _mm256_packs_epi16(i0, i2);
			const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
			i0				   = _mm256_permutevar8x32_epi32(i0, perm);

			_mm256_storeu_si256(( __m256i* )y[i].qs, i0);
		}
	}

	template<typename TA, typename TB, typename TC> class tinyBLAS_Q0_AVX2_2 {
	  public:
		static constexpr int8_t kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

		NIHILUS_INLINE tinyBLAS_Q0_AVX2_2(int64_t k, const TA* A, int64_t lda, const TB* B, int64_t ldb, TC* C, int64_t ldc, int ith, int nth)
			: A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
			iq4nlt = _mm_loadu_si128(( const __m128i* )kvalues_iq4nl);
		}

		NIHILUS_INLINE void matmul(int64_t m, int64_t n) {
			mnpack(0, m, 0, n);
		}

	  private:
		inline void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
			int64_t mc, nc, mp, np;
			switch ((detail::min(m - m0, 4L) << 4) | detail::min(n - n0, 4L)) {
				case 0x44:
					mc = 4;
					nc = 4;
					gemm<4, 4>(m0, m, n0, n);
					break;
				case 0x43:
					mc = 4;
					nc = 3;
					gemm<4, 3>(m0, m, n0, n);
					break;
				case 0x42:
					mc = 4;
					nc = 2;
					gemm<4, 2>(m0, m, n0, n);
					break;
				case 0x41:
					mc = 4;
					nc = 1;
					gemm<4, 1>(m0, m, n0, n);
					break;
				case 0x34:
					mc = 3;
					nc = 4;
					gemm<3, 4>(m0, m, n0, n);
					break;
				case 0x33:
					mc = 3;
					nc = 3;
					gemm<3, 3>(m0, m, n0, n);
					break;
				case 0x32:
					mc = 3;
					nc = 2;
					gemm<3, 2>(m0, m, n0, n);
					break;
				case 0x31:
					mc = 3;
					nc = 1;
					gemm<3, 1>(m0, m, n0, n);
					break;
				case 0x24:
					mc = 2;
					nc = 4;
					gemm<2, 4>(m0, m, n0, n);
					break;
				case 0x23:
					mc = 2;
					nc = 3;
					gemm<2, 3>(m0, m, n0, n);
					break;
				case 0x22:
					mc = 2;
					nc = 2;
					gemm<2, 2>(m0, m, n0, n);
					break;
				case 0x21:
					mc = 2;
					nc = 1;
					gemm<2, 1>(m0, m, n0, n);
					break;
				case 0x14:
					mc = 1;
					nc = 4;
					gemm<1, 4>(m0, m, n0, n);
					break;
				case 0x13:
					mc = 1;
					nc = 3;
					gemm<1, 3>(m0, m, n0, n);
					break;
				case 0x12:
					mc = 1;
					nc = 2;
					gemm<1, 2>(m0, m, n0, n);
					break;
				case 0x11:
					mc = 1;
					nc = 1;
					gemm<1, 1>(m0, m, n0, n);
					break;
				default:
					return;
			}
			mp = m0 + (m - m0) / mc * mc;
			np = n0 + (n - n0) / nc * nc;
			mnpack(mp, m, n0, np);
			mnpack(m0, m, np, n);
		}

		template<int RM, int RN> NIHILUS_INLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
			int64_t max_a_idx = (m - 1) * lda + (k - 1);
			int64_t max_b_idx = (k - 1) * ldb + (n - 1);

			int64_t ytiles = (m - m0) / RM;
			int64_t xtiles = (n - n0) / RN;
			int64_t tiles  = xtiles * ytiles;
			int64_t duty   = (tiles + nth - 1) / nth;
			int64_t start  = duty * ith;
			int64_t end	   = start + duty;
			if (end > tiles)
				end = tiles;

			for (int64_t job = start; job < end; ++job) {
				int64_t ii		  = m0 + job / xtiles * RM;
				int64_t jj		  = n0 + job % xtiles * RN;
				__m256 Cv[RN][RM] = {};

				for (int64_t l = 0; l < k; ++l) {
					for (int64_t j = 0; j < RN; ++j) {
						for (int64_t i = 0; i < RM; ++i) {
							int64_t a_idx = (ii + i) * k + l;
							int64_t b_idx = l * n + (jj + j);
							int64_t c_idx = (jj + j) * ldc + (ii + i);
							int64_t max_a_elements = m * k;
							int64_t max_b_elements = k * n;

							__m256i a_vec = load_avx2(A + a_idx);
							__m256i b_vec = load_avx2(B + b_idx);

							__m256 udTmp = updot_avx2(_mm256_sign_epi8(a_vec, a_vec), _mm256_sign_epi8(b_vec, a_vec));

							float scale = get_scale(A[a_idx]) * get_scale(B[b_idx]);

							Cv[j][i] = madd_avx2(_mm256_set1_ps(scale), udTmp, Cv[j][i]);
						}
					}
				}

				for (int64_t j = 0; j < RN; ++j) {
					for (int64_t i = 0; i < RM; ++i) {
						int64_t c_idx = (jj + j) * ldc + (ii + i);
						C[c_idx] = hsum_avx2(Cv[j][i]);
					}
				}
			}
		}

		template<typename T> NIHILUS_INLINE __m256i load_avx2(const T* b) {
			if constexpr (std::is_same_v<T, block_q8_0<half>>) {
				const void* block_start = static_cast<const void*>(b);
				const void* qs_ptr		= static_cast<const void*>(b->qs);
				const void* qs_end		= static_cast<const void*>(b->qs + 32);
				uintptr_t block_addr  = reinterpret_cast<uintptr_t>(block_start);
				uintptr_t qs_addr	  = reinterpret_cast<uintptr_t>(qs_ptr);
				uintptr_t qs_end_addr = reinterpret_cast<uintptr_t>(qs_end);
				return _mm256_loadu_si256(( const __m256i* )b->qs);
			} else {
				static_assert(std::is_same_v<T, block_q8_0<half>>, "Unsupported type for load_avx2!");
				return _mm256_setzero_si256();
			}
		}

		template<typename T> NIHILUS_INLINE float get_scale(const T& block) {
			if constexpr (std::is_same_v<T, block_q8_0<half>>) {
				return unhalf(block.d);
			} else {
				static_assert(std::is_same_v<T, block_q8_0<half>>, "Unsupported type for get_scale!");
				return 1.0f;
			}
		}

		NIHILUS_INLINE __m256 updot_avx2(__m256i u, __m256i s) {
			__m256i res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
			return _mm256_cvtepi32_ps(res);
		}

		NIHILUS_INLINE __m256 madd_avx2(__m256 a, __m256 b, __m256 c) {
			return _mm256_fmadd_ps(a, b, c);
		}

		NIHILUS_INLINE float hsum_avx2(__m256 v) {
			__m128 lo	= _mm256_castps256_ps128(v);
			__m128 hi	= _mm256_extractf128_ps(v, 1);
			lo			= _mm_add_ps(lo, hi);
			__m128 shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1));
			lo			= _mm_add_ps(lo, shuf);
			shuf		= _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2));
			lo			= _mm_add_ps(lo, shuf);
			return _mm_cvtss_f32(lo);
		}

		NIHILUS_INLINE static __m256i denibble_avx2(const uint8_t* p) {
			__m128i x = _mm_loadu_si128(( const __m128i* )p);
			return _mm256_and_si256(_mm256_set1_epi8(15), _mm256_set_m128i(_mm_srli_epi16(x, 4), x));
		}

		NIHILUS_INLINE static __m256i bittobyte_avx2(const uint8_t* p) {
			uint32_t x32;
			memcpy(&x32, p, sizeof(uint32_t));
			__m256i bytes = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
				_mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
					_mm256_shuffle_epi8(_mm256_set1_epi32(x32), _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000))));
			return _mm256_andnot_si256(bytes, _mm256_set1_epi8(( char )0xF0));
		}

		const TA* const A;
		const TB* const B;
		TC* const C;
		const int64_t k;
		const int64_t lda;
		const int64_t ldb;
		const int64_t ldc;
		const int ith;
		const int nth;
		__m128i iq4nlt;
	};

	template<typename TA, typename TB, typename TC>
	NIHILUS_INLINE void llamafile_sgemm(int64_t ith, int64_t nth, int64_t m, int64_t n, int64_t k, const TA* A, int64_t lda, const TB* B, int64_t ldb, TC* C, int64_t ldc) {
		tinyBLAS_Q0_AVX2_2<TA, TB, TC> tb{ k, A, lda, B, ldb, C, ldc, static_cast<int>(ith), static_cast<int>(nth) };
		tb.matmul(m, n);
	}

	NIHILUS_INLINE static void vec_scale_f32(const uint64_t n, float* __restrict y, const float v) {
		const uint64_t np = (n & ~(Q_SIZE - 1));

		__m256 vx = _mm256_set1_ps(v);

		__m256 ay[(Q_SIZE / 8)];

		for (uint64_t i = 0; i < np; i += Q_SIZE) {
			for (uint64_t j = 0; j < (Q_SIZE / 8); j++) {
				ay[j] = _mm256_load_ps(y + i + j * 8);
				ay[j] = _mm256_mul_ps(ay[j], vx);

				_mm256_store_ps(y + i + j * 8, ay[j]);
			}
		}

		for (uint64_t i = np; i < n; ++i) {
			y[i] *= v;
		}
	}

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::rms_norm_mul, core_type, float, float, float> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
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
			float* __restrict input02_data;
			if constexpr (array_types<decltype(input02.data)>) {
				input02_data = input02.data[0];
			}
			float* __restrict output_data = output.data;

			const int64_t src_row_elements	  = ne00;
			const int64_t src_plane_elements  = ne00 * ne01;
			const int64_t src_volume_elements = ne00 * ne01 * ne02;

			const int64_t dst_row_elements	  = ne0;
			const int64_t dst_plane_elements  = ne0 * ne1;
			const int64_t dst_volume_elements = ne0 * ne1 * ne2;

			static constexpr float eps			= core_type::model_traits_type::layer_norm_rms_epsilon;
			static constexpr int64_t simd_width = 8;

			for (int64_t i03 = 0; i03 < ne03; i03++) {
				for (int64_t i02 = 0; i02 < ne02; i02++) {
					for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
						int64_t src_offset		  = i03 * src_volume_elements + i02 * src_plane_elements + i01 * src_row_elements;
						const float* __restrict x = &input01_data[src_offset];

						const float* __restrict w = &input02_data[0];

						__m256 sum_vec = _mm256_setzero_ps();
						int64_t i00	   = 0;

						for (; i00 <= ne00 - simd_width; i00 += simd_width) {
							__m256 x_vec   = _mm256_load_ps(&x[i00]);
							__m256 squared = _mm256_mul_ps(x_vec, x_vec);
							sum_vec		   = _mm256_add_ps(sum_vec, squared);
						}

						__m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
						__m128 sum_low	= _mm256_castps256_ps128(sum_vec);
						__m128 sum_128	= _mm_add_ps(sum_low, sum_high);
						sum_128			= _mm_hadd_ps(sum_128, sum_128);
						sum_128			= _mm_hadd_ps(sum_128, sum_128);
						float sum		= _mm_cvtss_f32(sum_128);

						for (; i00 < ne00; i00++) {
							sum += x[i00] * x[i00];
						}

						const float mean  = sum / ne00;
						const float scale = 1.0f / sqrtf(mean + eps);

						int64_t dst_offset	= i03 * dst_volume_elements + i02 * dst_plane_elements + i01 * dst_row_elements;
						float* __restrict y = &output_data[dst_offset];

						__m256 scale_vec = _mm256_set1_ps(scale);
						i00				 = 0;

						for (; i00 <= ne00 - simd_width; i00 += simd_width) {
							__m256 x_vec	  = _mm256_load_ps(&x[i00]);
							__m256 w_vec	  = _mm256_load_ps(&w[i00]);
							__m256 multiplied = _mm256_mul_ps(x_vec, w_vec);
							__m256 result	  = _mm256_mul_ps(multiplied, scale_vec);
							_mm256_store_ps(&y[i00], result);
						}

						for (; i00 < ne00; i00++) {
							y[i00] = (x[i00] * w[i00]) * scale;
						}
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rms_norm, transform_type, core_type, float, float>
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_broadcasting> NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
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

			const uint64_t combined_dims = ne01 * ne02;
			const uint64_t combined_pow2 = largest_pow2(combined_dims);
			const uint64_t combined_log2 = tzcnt(combined_pow2);
			const uint64_t combined_mask = combined_pow2 - 1;

			const uint64_t ne01_pow2 = largest_pow2(ne01);
			const uint64_t ne01_log2 = tzcnt(ne01_pow2);
			const uint64_t ne01_mask = ne01_pow2 - 1;

			const uint64_t pow2_section_size = combined_pow2 * ne03;

			const float* __restrict input01_data = input01.data;
			const float* __restrict input02_data = input02.data;
			float* __restrict output_data		 = output.data;

			static constexpr uint64_t input01_stride_dim1 = ne00;
			const uint64_t input01_stride_dim2			  = ne00 * ne01;
			const uint64_t input01_stride_dim3			  = ne00 * ne01 * ne02;
			static constexpr uint64_t input02_stride_dim1 = ne10;
			const uint64_t input02_stride_dim2			  = ne10 * ne11;
			const uint64_t input02_stride_dim3			  = ne10 * ne11 * ne12;
			static constexpr uint64_t output_stride_dim1  = ne00;
			const uint64_t output_stride_dim2			  = output[1] * ne00;
			const uint64_t output_stride_dim3			  = output[1] * output[2] * ne00;

			static constexpr uint64_t simd_width = 8;
			const uint64_t ne00_simd			 = ne00 & ~(simd_width - 1);

			const uint64_t ir1_fast = detail::min(ir1, ir0 + pow2_section_size);

			for (uint64_t i = ir0; i < ir1_fast; ++i) {
				const uint64_t i13		= i / combined_dims;
				const uint64_t flat_idx = i & combined_mask;
				const uint64_t i12		= flat_idx >> ne01_log2;
				const uint64_t i11		= flat_idx & ne01_mask;

				const uint64_t input01_base = i11 * input01_stride_dim1 + i12 * input01_stride_dim2 + i13 * input01_stride_dim3;
				const uint64_t output_base	= i11 * output_stride_dim1 + i12 * output_stride_dim2 + i13 * output_stride_dim3;

				uint64_t input02_base;
				if constexpr (is_broadcasting) {
					input02_base = 0 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				} else {
					input02_base = i11 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				}

				uint64_t i10 = 0;
				for (; i10 < ne00_simd; i10 += simd_width) {
					__m256 input01_vec = _mm256_load_ps(&input01_data[input01_base + i10]);
					__m256 input02_vec = _mm256_load_ps(&input02_data[input02_base + i10]);
					__m256 result_vec  = _mm256_mul_ps(input01_vec, input02_vec);
					_mm256_store_ps(&output_data[output_base + i10], result_vec);
				}

				for (; i10 < ne00; ++i10) {
					const uint64_t input01_idx = input01_base + i10;
					const uint64_t input02_idx = input02_base + i10;
					const uint64_t output_idx  = output_base + i10;
					output_data[output_idx]	   = input01_data[input01_idx] * input02_data[input02_idx];
				}
			}

			for (uint64_t i = ir1_fast; i < ir1; ++i) {
				const uint64_t i13 = i / (ne02 * ne01);
				const uint64_t i12 = (i - i13 * ne02 * ne01) / ne01;
				const uint64_t i11 = i - i13 * ne02 * ne01 - i12 * ne01;

				const uint64_t input01_base = i11 * input01_stride_dim1 + i12 * input01_stride_dim2 + i13 * input01_stride_dim3;
				const uint64_t output_base	= i11 * output_stride_dim1 + i12 * output_stride_dim2 + i13 * output_stride_dim3;

				uint64_t input02_base;
				if constexpr (is_broadcasting) {
					input02_base = 0 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				} else {
					input02_base = i11 * input02_stride_dim1 + i12 * input02_stride_dim2 + i13 * input02_stride_dim3;
				}

				uint64_t i10 = 0;
				for (; i10 < ne00_simd; i10 += simd_width) {
					__m256 input01_vec = _mm256_load_ps(&input01_data[input01_base + i10]);
					__m256 input02_vec = _mm256_load_ps(&input02_data[input02_base + i10]);
					__m256 result_vec  = _mm256_mul_ps(input01_vec, input02_vec);
					_mm256_store_ps(&output_data[output_base + i10], result_vec);
				}

				for (; i10 < ne00; ++i10) {
					const uint64_t input01_idx = input01_base + i10;
					const uint64_t input02_idx = input02_base + i10;
					const uint64_t output_idx  = output_base + i10;
					output_data[output_idx]	   = input01_data[input01_idx] * input02_data[input02_idx];
				}
			}
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			const uint64_t ne11		   = input02[1];
			const bool is_broadcasting	   = (ne11 == 1);
			if (is_broadcasting) {
				impl<true>(thread_index, thread_count, output, input01, input02);
			} else {
				impl<false>(thread_index, thread_count, output, input01, input02);
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
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

			static constexpr uint64_t ne0 = core_type::get_array()[0];
			const uint64_t ne1			  = output[1];
			static constexpr uint64_t ne2 = core_type::get_array()[2];
			static constexpr uint64_t ne3 = core_type::get_array()[3];

			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);

			static constexpr int64_t vec_dot_num_rows = type_traits<typename core_type::input_01_type::output_type>::n_rows;
			static constexpr int64_t block_size		  = type_traits<typename core_type::input_01_type::output_type>::block_size;

			static constexpr uint64_t r2 = ne12 / ne02;
			static constexpr uint64_t r3 = ne13 / ne03;

			const block_q8_0<half>* __restrict src0_data;
			if constexpr (array_types<decltype(input01.data)>) {
				src0_data = input01.data[0];
			} else {
				src0_data = input01.data;
			}

			const float* __restrict src1_data = input02.data;
			float* __restrict dst_data		  = output.data;

			static constexpr uint64_t src0_row_elements = ne00;
			const uint64_t src0_plane_elements			= ne00 * ne01;
			const uint64_t src0_volume_elements			= ne00 * ne01 * ne02;

			static constexpr uint64_t src1_row_elements = ne10;
			const uint64_t src1_plane_elements			= ne10 * ne11;
			const uint64_t src1_volume_elements			= ne10 * ne11 * ne12;

			static constexpr uint64_t dst_row_elements = ne0;
			const uint64_t dst_plane_elements		   = ne0 * ne1;
			const uint64_t dst_volume_elements		   = ne0 * ne1 * ne2;

			// Use second half of output allocation as workspace for quantization
			block_q8_0<half>* wdata = reinterpret_cast<block_q8_0<half>*>(dst_data + dst_volume_elements * ne3);

			const uint64_t work_row_elements	= ne10;
			const uint64_t work_plane_elements	= ne10 * ne11;
			const uint64_t work_volume_elements = ne10 * ne11 * ne12;

			// Quantize src1 data to Q8_0 format
			for (uint64_t i13 = 0; i13 < ne13; ++i13) {
				for (uint64_t i12 = 0; i12 < ne12; ++i12) {
					for (uint64_t i11 = 0; i11 < ne11; ++i11) {
						constexpr size_t bs		  = block_size;
						uint64_t ne10_block_start = (ith * ne10 / bs) / nth;
						uint64_t ne10_block_end	  = ((ith + 1) * ne10 / bs) / nth;

						quantize_row_q8_0(src1_data + i13 * src1_volume_elements + i12 * src1_plane_elements + i11 * src1_row_elements + ne10_block_start * bs,
							wdata + i13 * work_volume_elements + i12 * work_plane_elements + i11 * work_row_elements + ne10_block_start, (ne10_block_end - ne10_block_start) * bs);
					}
				}
			}

			const block_q8_0<half>* working_src1_data = wdata;

			for (uint64_t i13 = 0; i13 < ne13; i13++) {
				for (uint64_t i12 = 0; i12 < ne12; i12++) {
					int64_t actual_m = detail::min(static_cast<int64_t>(ne01), 4096L);
					llamafile_sgemm(static_cast<int64_t>(ith), static_cast<int64_t>(nth), static_cast<int64_t>(actual_m), static_cast<int64_t>(ne11),
						static_cast<int64_t>(ne00 / block_size), src0_data + (i12 / r2) * src0_volume_elements + (i13 / r3) * src0_volume_elements * ne02,
						static_cast<int64_t>(ne00 / block_size),// FIXED: was src0_plane_elements / block_size
						working_src1_data + (i12 * ne11 + i13 * ne12 * ne11) * work_row_elements,
						static_cast<int64_t>(ne11),// FIXED: was work_row_elements
						dst_data + i12 * dst_plane_elements + i13 * dst_volume_elements,
						static_cast<int64_t>(ne1));// FIXED: was dst_plane_elements
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			/*
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];
			static constexpr uint64_t ne12 = input_type02::get_array()[2];
			static constexpr uint64_t ne13 = input_type02::get_array()[3];
			const uint64_t ne10 = input02[0];
			const uint64_t ne11 = input02[1];
			const half* __restrict src0_data  = input01.data;
			const float* __restrict src1_data = input02.data;
			float* __restrict dst_data		  = output.data;
			static constexpr uint64_t r2 = ne12 / ne02;
			static constexpr uint64_t r3 = ne13 / ne03;
			static constexpr uint64_t src0_stride_01 = ne00;
			static constexpr uint64_t src0_stride_02 = ne00 * ne01;
			static constexpr uint64_t src0_stride_03 = ne00 * ne01 * ne02;

			const uint64_t src1_stride_11 = ne10;
			const uint64_t src1_stride_12 = ne10 * ne11;
			const uint64_t src1_stride_13 = ne10 * ne11 * ne12;

			const uint64_t dst_stride_11 = ne01;
			const uint64_t dst_stride_12 = ne01 * ne11;
			const uint64_t dst_stride_13 = ne01 * ne11 * ne12;

			for (uint64_t i13 = 0; i13 < ne13; ++i13) {
				const uint64_t i03			 = i13 / r3;
				const uint64_t base_src0_i03 = i03 * src0_stride_03;
				const uint64_t base_src1_i13 = i13 * src1_stride_13;
				const uint64_t base_dst_i13	 = i13 * dst_stride_13;

				for (uint64_t i12 = 0; i12 < ne12; ++i12) {
					const uint64_t i02			 = i12 / r2;
					const uint64_t base_src0_i02 = base_src0_i03 + i02 * src0_stride_02;
					const uint64_t base_src1_i12 = base_src1_i13 + i12 * src1_stride_12;
					const uint64_t base_dst_i12	 = base_dst_i13 + i12 * dst_stride_12;

					for (uint64_t i11 = 0; i11 < ne11; ++i11) {
						const uint64_t base_src1_i11 = base_src1_i12 + i11 * src1_stride_11;
						const uint64_t base_dst_i11	 = base_dst_i12 + i11 * dst_stride_11;

						for (uint64_t i01 = 0; i01 < ne01; ++i01) {
							const uint64_t base_src0_i01 = base_src0_i02 + i01 * src0_stride_01;

							float sum = 0.0f;

							for (uint64_t i00 = 0; i00 < ne00; ++i00) {
								const uint64_t src0_idx = base_src0_i01 + i00;
								const uint64_t src1_idx = base_src1_i11 + i00;
								sum += fp16_to_fp32(src0_data[src0_idx]) * src1_data[src1_idx];
							}

							const uint64_t dst_idx = base_dst_i11 + i01;
							dst_data[dst_idx]	   = sum;
						}
					}
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::add, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
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
			float* __restrict output_data		 = output.data;

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
					__m256 input01_vec = _mm256_load_ps(&input01_data[input01_base + i10]);

					__m256 input02_vec = _mm256_load_ps(&input02_data[input02_base + i10]);

					__m256 result_vec = _mm256_add_ps(input01_vec, input02_vec);

					_mm256_store_ps(&output_data[output_base + i10], result_vec);
				}

				for (; i10 < ne00; ++i10) {
					const uint64_t input01_idx = input01_base + i10;
					const uint64_t input02_idx = input02_base + i10;
					const uint64_t output_idx  = output_base + i10;

					output_data[output_idx] = input01_data[input01_idx] + input02_data[input02_idx];
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			const uint64_t ne01			   = input01[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data  = input01.data;
			const float* __restrict mask_data = input02.data;
			float* __restrict dst_data		  = output.data;

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
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
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
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
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
			int64_t int_exp		= static_cast<int64_t>(exp);
			float current_power = base;

			while (int_exp > 0) {
				if (int_exp & 1)
					result *= current_power;
				current_power *= current_power;
				int_exp >>= 1;
			}

			float frac = exp - static_cast<int64_t>(exp);
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
			const auto array_new					   = input_type02::get_array();
			const uint64_t seq_len = input01[1];

			const uint64_t total_work_items = batch_size * seq_len * num_heads;
			const uint64_t total_elements	= batch_size * seq_len * num_heads * head_dim;

			const uint64_t work_per_thread = (total_work_items + thread_count - 1) / thread_count;
			const uint64_t work_start	   = thread_index * work_per_thread;
			const uint64_t work_end		   = (work_start + work_per_thread < total_work_items) ? work_start + work_per_thread : total_work_items;

			const int64_t nr = total_work_items;
			int64_t ir		 = 0;

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
								const uint64_t scaling_idx = (dim_pair < array_new[0]) ? dim_pair : 0;
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
					}
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::copy, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			//std::cout << "COPYING TYPE: " << core_type::op_type << std::endl;
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			float* __restrict dst_data		 = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = count_elements(output);

			for (uint64_t i = 0; i < total_elements; ++i) {
				dst_data[i] = src_data[i];
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne02 = input_type01::get_array()[2];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			half* dst_data					 = output.data;

			const uint64_t ne02_runtime	  = input01[2];
			const uint64_t total_elements = count_elements(output);

			for (uint64_t i = 0; i < total_elements; ++i) {
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::cont, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01) {
			static constexpr uint64_t ne00 = input_type01::get_array()[0];
			static constexpr uint64_t ne01 = input_type01::get_array()[1];
			static constexpr uint64_t ne03 = input_type01::get_array()[3];

			const float* __restrict src_data = input01.data;
			float* __restrict dst_data		 = output.data;

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
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

};

#endif