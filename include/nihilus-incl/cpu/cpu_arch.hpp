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
	struct kernel_dispatcher_impl<0, kernel_types::embedding_lookup, transform_type, core_type, float, block_q8_0<half>, int32_t>
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

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::embedding_lookup, transform_type, core_type, float, float, int32_t>
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

	NIHILUS_INLINE std::ostream& operator<<(std::ostream& os, const array<uint64_t, 4>& values) {
		os << "[";
		for (size_t x = 0; x < 4; ++x) {
			os << values[x];
			if (x < 3) {
				os << ",";
			}
		}
		os << "]" << std::endl;
		return os;
	}

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<0, kernel_types::fused_qkv_rope, transform_type, core_type, float, float, float, float, block_q8_0<half>, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03, const typename core_type::input_04_type& input04,
			const typename core_type::input_05_type& input05, const typename core_type::input_06_type& input06) {
			std::cout << core_type::op_type << " Dims: " << core_type::get_array() << std::endl;
			std::cout << core_type::op_type << " Bytes: " << core_type::total_required_bytes << std::endl;
			using model_traits_type				   = typename core_type::model_traits_type;
			static constexpr uint64_t head_dim	   = model_traits_type::head_dim;
			static constexpr uint64_t num_heads	   = model_traits_type::attention_head_count;
			static constexpr uint64_t num_kv_heads = model_traits_type::attention_head_count_kv;
			static constexpr uint64_t hidden_dim   = model_traits_type::embedding_length;

			const int64_t seq_len	   = input01[0];
			const int64_t block_id	   = thread_index / thread_count;
			const int64_t local_thread = thread_index % thread_count;

			const float* embeddings	 = input01.data;
			const float* norm_weight = input02.data[0];
			const float* rope_freqs	 = input03.data;

			const block_q8_0<half>* q_weights = input04.data[0];
			const block_q8_0<half>* k_weights = input05.data[0];
			const block_q8_0<half>* v_weights = input06.data[0];

			float* q_output = output.data;
			float* k_output = q_output + (num_heads * head_dim);
			float* v_output = k_output + (num_kv_heads * head_dim);

			const int64_t heads_per_thread = (num_heads + thread_count - 1) / thread_count;
			const int64_t head_start	   = local_thread * heads_per_thread;
			const int64_t head_end		   = std::min(head_start + heads_per_thread, ( int64_t )num_heads);

			const __m256 zero = _mm256_setzero_ps();

			__m256 sum	  = zero;
			__m256 sum_sq = zero;

			const int64_t simd_limit = (hidden_dim / 8) * 8;
			for (int64_t i = 0; i < simd_limit; i += 8) {
				__m256 x = _mm256_loadu_ps(&embeddings[i]);
				sum		 = _mm256_add_ps(sum, x);
				sum_sq	 = _mm256_fmadd_ps(x, x, sum_sq);
			}

			float mean_scalar = 0.0f, var_scalar = 0.0f;
			for (int64_t i = simd_limit; i < hidden_dim; ++i) {
				float x = embeddings[i];
				mean_scalar += x;
				var_scalar += x * x;
			}

			alignas(32) float mean_vals[8], var_vals[8];
			_mm256_storeu_ps(mean_vals, sum);
			_mm256_storeu_ps(var_vals, sum_sq);

			float mean	   = mean_scalar;
			float variance = var_scalar;
			for (int i = 0; i < 8; ++i) {
				mean += mean_vals[i];
				variance += var_vals[i];
			}

			mean /= hidden_dim;
			variance			= variance / hidden_dim - mean * mean;
			const float inv_std = 1.0f / sqrtf(variance + 1e-6f);

			const __m256 mean_vec	 = _mm256_set1_ps(mean);
			const __m256 inv_std_vec = _mm256_set1_ps(inv_std);

			alignas(32) float normalized_embeddings[hidden_dim];

			for (int64_t i = 0; i < simd_limit; i += 8) {
				__m256 x		  = _mm256_loadu_ps(&embeddings[i]);
				__m256 norm_w	  = _mm256_loadu_ps(&norm_weight[i]);
				__m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
				normalized		  = _mm256_mul_ps(normalized, norm_w);
				_mm256_storeu_ps(&normalized_embeddings[i], normalized);
			}

			for (int64_t i = simd_limit; i < hidden_dim; ++i) {
				normalized_embeddings[i] = (embeddings[i] - mean) * inv_std * norm_weight[i];
			}

			for (int64_t head = head_start; head < head_end; ++head) {
				const int64_t head_offset	= head * head_dim;
				const int64_t k_head		= std::min(head, ( int64_t )num_kv_heads - 1);
				const int64_t k_head_offset = k_head * head_dim;
				const int64_t v_head		= k_head;
				const int64_t v_head_offset = v_head * head_dim;

				for (int64_t dim = 0; dim < head_dim; dim += 2) {
					__m256 q_accum1 = zero, q_accum2 = zero;
					__m256 k_accum1 = zero, k_accum2 = zero;
					__m256 v_accum1 = zero, v_accum2 = zero;

					const int64_t q_weight_row1 = head_offset + dim;
					const int64_t q_weight_row2 = head_offset + dim + 1;
					const int64_t k_weight_row1 = k_head_offset + dim;
					const int64_t k_weight_row2 = k_head_offset + dim + 1;
					const int64_t v_weight_row1 = v_head_offset + dim;
					const int64_t v_weight_row2 = v_head_offset + dim + 1;

					for (int64_t i = 0; i < simd_limit; i += 8) {
						__m256 normalized = _mm256_loadu_ps(&normalized_embeddings[i]);

						const int64_t q_block_idx1 = (q_weight_row1 * hidden_dim + i) / 32;
						const int64_t q_elem_idx1  = (q_weight_row1 * hidden_dim + i) % 32;
						const int64_t q_block_idx2 = (q_weight_row2 * hidden_dim + i) / 32;
						const int64_t q_elem_idx2  = (q_weight_row2 * hidden_dim + i) % 32;

						const block_q8_0<short>& q_block1 = q_weights[q_block_idx1];
						const block_q8_0<short>& q_block2 = q_weights[q_block_idx2];
						const float q_scale1			  = static_cast<float>(q_block1.d);
						const float q_scale2			  = static_cast<float>(q_block2.d);

						__m128i q_quant1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q_block1.qs[q_elem_idx1]));
						__m128i q_quant2_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&q_block2.qs[q_elem_idx2]));
						__m256i q_quant1_32 = _mm256_cvtepi16_epi32(q_quant1_16);
						__m256i q_quant2_32 = _mm256_cvtepi16_epi32(q_quant2_16);
						__m256 q_weights1_f = _mm256_mul_ps(_mm256_cvtepi32_ps(q_quant1_32), _mm256_set1_ps(q_scale1));
						__m256 q_weights2_f = _mm256_mul_ps(_mm256_cvtepi32_ps(q_quant2_32), _mm256_set1_ps(q_scale2));

						q_accum1 = _mm256_fmadd_ps(normalized, q_weights1_f, q_accum1);
						q_accum2 = _mm256_fmadd_ps(normalized, q_weights2_f, q_accum2);

						const int64_t k_block_idx1 = (k_weight_row1 * hidden_dim + i) / 32;
						const int64_t k_elem_idx1  = (k_weight_row1 * hidden_dim + i) % 32;
						const int64_t k_block_idx2 = (k_weight_row2 * hidden_dim + i) / 32;
						const int64_t k_elem_idx2  = (k_weight_row2 * hidden_dim + i) % 32;

						const block_q8_0<short>& k_block1 = k_weights[k_block_idx1];
						const block_q8_0<short>& k_block2 = k_weights[k_block_idx2];
						const float k_scale1			  = static_cast<float>(k_block1.d);
						const float k_scale2			  = static_cast<float>(k_block2.d);

						__m128i k_quant1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&k_block1.qs[k_elem_idx1]));
						__m128i k_quant2_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&k_block2.qs[k_elem_idx2]));
						__m256i k_quant1_32 = _mm256_cvtepi16_epi32(k_quant1_16);
						__m256i k_quant2_32 = _mm256_cvtepi16_epi32(k_quant2_16);
						__m256 k_weights1_f = _mm256_mul_ps(_mm256_cvtepi32_ps(k_quant1_32), _mm256_set1_ps(k_scale1));
						__m256 k_weights2_f = _mm256_mul_ps(_mm256_cvtepi32_ps(k_quant2_32), _mm256_set1_ps(k_scale2));

						k_accum1 = _mm256_fmadd_ps(normalized, k_weights1_f, k_accum1);
						k_accum2 = _mm256_fmadd_ps(normalized, k_weights2_f, k_accum2);

						const int64_t v_block_idx1 = (v_weight_row1 * hidden_dim + i) / 32;
						const int64_t v_elem_idx1  = (v_weight_row1 * hidden_dim + i) % 32;
						const int64_t v_block_idx2 = (v_weight_row2 * hidden_dim + i) / 32;
						const int64_t v_elem_idx2  = (v_weight_row2 * hidden_dim + i) % 32;

						const block_q8_0<short>& v_block1 = v_weights[v_block_idx1];
						const block_q8_0<short>& v_block2 = v_weights[v_block_idx2];
						const float v_scale1			  = static_cast<float>(v_block1.d);
						const float v_scale2			  = static_cast<float>(v_block2.d);

						__m128i v_quant1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&v_block1.qs[v_elem_idx1]));
						__m128i v_quant2_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&v_block2.qs[v_elem_idx2]));
						__m256i v_quant1_32 = _mm256_cvtepi16_epi32(v_quant1_16);
						__m256i v_quant2_32 = _mm256_cvtepi16_epi32(v_quant2_16);
						__m256 v_weights1_f = _mm256_mul_ps(_mm256_cvtepi32_ps(v_quant1_32), _mm256_set1_ps(v_scale1));
						__m256 v_weights2_f = _mm256_mul_ps(_mm256_cvtepi32_ps(v_quant2_32), _mm256_set1_ps(v_scale2));

						v_accum1 = _mm256_fmadd_ps(normalized, v_weights1_f, v_accum1);
						v_accum2 = _mm256_fmadd_ps(normalized, v_weights2_f, v_accum2);
					}

					float q_remainder1 = 0.0f, q_remainder2 = 0.0f;
					float k_remainder1 = 0.0f, k_remainder2 = 0.0f;
					float v_remainder1 = 0.0f, v_remainder2 = 0.0f;

					for (int64_t i = simd_limit; i < hidden_dim; ++i) {
						float normalized = normalized_embeddings[i];

						const int64_t q_block_idx1 = (q_weight_row1 * hidden_dim + i) / 32;
						const int64_t q_elem_idx1  = (q_weight_row1 * hidden_dim + i) % 32;
						const int64_t q_block_idx2 = (q_weight_row2 * hidden_dim + i) / 32;
						const int64_t q_elem_idx2  = (q_weight_row2 * hidden_dim + i) % 32;

						const block_q8_0<short>& q_block1 = q_weights[q_block_idx1];
						const block_q8_0<short>& q_block2 = q_weights[q_block_idx2];
						const float q_weight1			  = static_cast<float>(q_block1.qs[q_elem_idx1]) * static_cast<float>(q_block1.d);
						const float q_weight2			  = static_cast<float>(q_block2.qs[q_elem_idx2]) * static_cast<float>(q_block2.d);

						q_remainder1 += normalized * q_weight1;
						q_remainder2 += normalized * q_weight2;

						const int64_t k_block_idx1 = (k_weight_row1 * hidden_dim + i) / 32;
						const int64_t k_elem_idx1  = (k_weight_row1 * hidden_dim + i) % 32;
						const int64_t k_block_idx2 = (k_weight_row2 * hidden_dim + i) / 32;
						const int64_t k_elem_idx2  = (k_weight_row2 * hidden_dim + i) % 32;

						const block_q8_0<short>& k_block1 = k_weights[k_block_idx1];
						const block_q8_0<short>& k_block2 = k_weights[k_block_idx2];
						const float k_weight1			  = static_cast<float>(k_block1.qs[k_elem_idx1]) * static_cast<float>(k_block1.d);
						const float k_weight2			  = static_cast<float>(k_block2.qs[k_elem_idx2]) * static_cast<float>(k_block2.d);

						k_remainder1 += normalized * k_weight1;
						k_remainder2 += normalized * k_weight2;

						const int64_t v_block_idx1 = (v_weight_row1 * hidden_dim + i) / 32;
						const int64_t v_elem_idx1  = (v_weight_row1 * hidden_dim + i) % 32;
						const int64_t v_block_idx2 = (v_weight_row2 * hidden_dim + i) / 32;
						const int64_t v_elem_idx2  = (v_weight_row2 * hidden_dim + i) % 32;

						const block_q8_0<short>& v_block1 = v_weights[v_block_idx1];
						const block_q8_0<short>& v_block2 = v_weights[v_block_idx2];
						const float v_weight1			  = static_cast<float>(v_block1.qs[v_elem_idx1]) * static_cast<float>(v_block1.d);
						const float v_weight2			  = static_cast<float>(v_block2.qs[v_elem_idx2]) * static_cast<float>(v_block2.d);

						v_remainder1 += normalized * v_weight1;
						v_remainder2 += normalized * v_weight2;
					}

					alignas(32) float q_vals1[8], q_vals2[8];
					alignas(32) float k_vals1[8], k_vals2[8];
					alignas(32) float v_vals1[8], v_vals2[8];

					_mm256_storeu_ps(q_vals1, q_accum1);
					_mm256_storeu_ps(q_vals2, q_accum2);
					_mm256_storeu_ps(k_vals1, k_accum1);
					_mm256_storeu_ps(k_vals2, k_accum2);
					_mm256_storeu_ps(v_vals1, v_accum1);
					_mm256_storeu_ps(v_vals2, v_accum2);

					float q_final1 = q_remainder1;
					float q_final2 = q_remainder2;
					float k_final1 = k_remainder1;
					float k_final2 = k_remainder2;
					float v_final1 = v_remainder1;
					float v_final2 = v_remainder2;

					for (int i = 0; i < 8; ++i) {
						q_final1 += q_vals1[i];
						q_final2 += q_vals2[i];
						k_final1 += k_vals1[i];
						k_final2 += k_vals2[i];
						v_final1 += v_vals1[i];
						v_final2 += v_vals2[i];
					}

					const int64_t seq_pos = block_id;
					const float freq	  = rope_freqs[dim / 2];
					const float cos_val	  = cosf(seq_pos * freq);
					const float sin_val	  = sinf(seq_pos * freq);

					q_output[head_offset + dim]		= q_final1 * cos_val - q_final2 * sin_val;
					q_output[head_offset + dim + 1] = q_final1 * sin_val + q_final2 * cos_val;

					k_output[k_head_offset + dim]	  = k_final1 * cos_val - k_final2 * sin_val;
					k_output[k_head_offset + dim + 1] = k_final1 * sin_val + k_final2 * cos_val;

					v_output[v_head_offset + dim]	  = v_final1;
					v_output[v_head_offset + dim + 1] = v_final2;
				}
			}
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<0, kernel_types::fused_attention, transform_type, core_type, float, float, float> {
		using output_type	= typename core_type::output_type;
		using input_01_type = typename core_type::input_01_type::output_type;
		using input_02_type = typename core_type::input_02_type::output_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			std::cout << core_type::op_type << " Dims: " << core_type::get_array() << std::endl;
			std::cout << core_type::op_type << " Bytes: " << core_type::total_required_bytes << std::endl;
			using model_traits_type				   = typename core_type::model_traits_type;
			static constexpr uint64_t head_dim	   = model_traits_type::head_dim;
			static constexpr uint64_t num_heads	   = model_traits_type::attention_head_count;
			static constexpr uint64_t num_kv_heads = model_traits_type::attention_head_count_kv;

			const int64_t seq_len	   = input01[0];
			const int64_t block_id	   = thread_index / thread_count;
			const int64_t local_thread = thread_index % thread_count;

			const float* qkv_data  = input01.data;
			const float* mask_data = input02.data;

			const float* q_data = qkv_data;
			const float* k_data = q_data + (num_heads * head_dim);
			const float* v_data = k_data + (num_kv_heads * head_dim);

			float* attention_output = output.data;

			const int64_t heads_per_thread = (num_heads + thread_count - 1) / thread_count;
			const int64_t head_start	   = local_thread * heads_per_thread;
			const int64_t head_end		   = std::min(head_start + heads_per_thread, ( int64_t )num_heads);

			const float scale	   = 1.0f / sqrtf(static_cast<float>(head_dim));
			const __m256 scale_vec = _mm256_set1_ps(scale);
			const __m256 zero	   = _mm256_setzero_ps();
			const __m256 neg_inf   = _mm256_set1_ps(-1e9f);

			for (int64_t head = head_start; head < head_end; ++head) {
				const int64_t q_offset	 = head * head_dim;
				const int64_t kv_head	 = std::min(head, ( int64_t )num_kv_heads - 1);
				const int64_t k_offset	 = kv_head * head_dim;
				const int64_t v_offset	 = kv_head * head_dim;
				const int64_t out_offset = head * head_dim;

				float attention_scores[131072];
				float max_score = -1e9f;

				for (int64_t pos = 0; pos <= block_id; ++pos) {
					__m256 score_accum = zero;

					const int64_t simd_limit = (head_dim / 8) * 8;
					for (int64_t dim = 0; dim < simd_limit; dim += 8) {
						__m256 q_vec = _mm256_loadu_ps(&q_data[q_offset + dim]);
						__m256 k_vec = _mm256_loadu_ps(&k_data[k_offset + dim + pos * num_kv_heads * head_dim]);
						score_accum	 = _mm256_fmadd_ps(q_vec, k_vec, score_accum);
					}

					float score_remainder = 0.0f;
					for (int64_t dim = simd_limit; dim < head_dim; ++dim) {
						score_remainder += q_data[q_offset + dim] * k_data[k_offset + dim + pos * num_kv_heads * head_dim];
					}

					alignas(32) float score_vals[8];
					_mm256_store_ps(score_vals, score_accum);
					float total_score = score_remainder;
					for (int i = 0; i < 8; ++i) {
						total_score += score_vals[i];
					}

					total_score *= scale;

					if (mask_data && mask_data[pos] == 0.0f) {
						total_score = -1e9f;
					}

					attention_scores[pos] = total_score;
					if (total_score > max_score) {
						max_score = total_score;
					}
				}

				for (int64_t pos = seq_len; pos < 131072; ++pos) {
					attention_scores[pos] = -1e9f;
				}

				const __m256 max_vec = _mm256_set1_ps(max_score);
				float exp_sum		 = 0.0f;

				const int64_t valid_positions = block_id + 1;
				const int64_t pos_simd_limit  = (valid_positions / 8) * 8;

				__m256 exp_sum_vec = zero;
				for (int64_t pos = 0; pos < pos_simd_limit; pos += 8) {
					__m256 scores	= _mm256_loadu_ps(&attention_scores[pos]);
					__m256 shifted	= _mm256_sub_ps(scores, max_vec);
					__m256 exp_vals = _mm256_exp_ps(shifted);
					_mm256_storeu_ps(&attention_scores[pos], exp_vals);
					exp_sum_vec = _mm256_add_ps(exp_sum_vec, exp_vals);
				}

				for (int64_t pos = pos_simd_limit; pos < valid_positions; ++pos) {
					float exp_val		  = expf(attention_scores[pos] - max_score);
					attention_scores[pos] = exp_val;
					exp_sum += exp_val;
				}

				alignas(32) float exp_sum_vals[8];
				_mm256_store_ps(exp_sum_vals, exp_sum_vec);
				for (int i = 0; i < 8; ++i) {
					exp_sum += exp_sum_vals[i];
				}

				const float inv_sum		 = 1.0f / exp_sum;
				const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

				for (int64_t pos = 0; pos < pos_simd_limit; pos += 8) {
					__m256 probs = _mm256_loadu_ps(&attention_scores[pos]);
					probs		 = _mm256_mul_ps(probs, inv_sum_vec);
					_mm256_storeu_ps(&attention_scores[pos], probs);
				}

				for (int64_t pos = pos_simd_limit; pos < valid_positions; ++pos) {
					attention_scores[pos] *= inv_sum;
				}

				for (int64_t dim = 0; dim < head_dim; ++dim) {
					__m256 output_accum = zero;

					for (int64_t pos = 0; pos < pos_simd_limit; pos += 8) {
						__m256 prob_vec = _mm256_loadu_ps(&attention_scores[pos]);

						alignas(32) float prob_vals[8];
						_mm256_store_ps(prob_vals, prob_vec);

						for (int i = 0; i < 8; ++i) {
							if (pos + i < valid_positions) {
								float prob	 = prob_vals[i];
								float v_val	 = v_data[v_offset + dim + (pos + i) * num_kv_heads * head_dim];
								output_accum = _mm256_fmadd_ps(_mm256_set1_ps(prob), _mm256_set1_ps(v_val), output_accum);
							}
						}
					}

					float output_remainder = 0.0f;
					for (int64_t pos = pos_simd_limit; pos < valid_positions; ++pos) {
						float prob	= attention_scores[pos];
						float v_val = v_data[v_offset + dim + pos * num_kv_heads * head_dim];
						output_remainder += prob * v_val;
					}

					alignas(32) float output_vals[8];
					_mm256_store_ps(output_vals, output_accum);
					float final_output = output_remainder;
					for (int i = 0; i < 8; ++i) {
						final_output += output_vals[i];
					}

					attention_output[out_offset + dim] = final_output;
				}
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<0, kernel_types::fused_attn_out, transform_type, core_type, float, float, block_q8_0<half>, float> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
			std::cout << core_type::op_type << " Dims: " << core_type::get_array() << std::endl;
			std::cout << core_type::op_type << " Bytes: " << core_type::total_required_bytes << std::endl;
			using model_traits_type				 = typename core_type::model_traits_type;
			static constexpr uint64_t num_heads	 = model_traits_type::attention_head_count;
			static constexpr uint64_t head_dim	 = model_traits_type::head_dim;
			static constexpr uint64_t hidden_dim = model_traits_type::embedding_length;

			const int64_t seq_len	   = input01[0];
			const int64_t block_id	   = thread_index / thread_count;
			const int64_t local_thread = thread_index % thread_count;

			const float* attention_output		   = input01.data;
			const block_q8_0<half>* output_weights = input02.data[0];
			const float* residual_input			   = input03.data;

			float* final_output = output.data;

			const int64_t elements_per_thread = (hidden_dim + thread_count - 1) / thread_count;
			const int64_t elem_start		  = local_thread * elements_per_thread;
			const int64_t elem_end			  = std::min(elem_start + elements_per_thread, ( int64_t )hidden_dim);

			const __m256 zero = _mm256_setzero_ps();

			for (int64_t out_dim = elem_start; out_dim < elem_end; ++out_dim) {
				__m256 output_accum = zero;

				const int64_t total_attn_dim = num_heads * head_dim;
				const int64_t simd_limit	 = (total_attn_dim / 8) * 8;

				for (int64_t attn_dim = 0; attn_dim < simd_limit; attn_dim += 8) {
					__m256 attn_vec = _mm256_loadu_ps(&attention_output[attn_dim]);

					const int64_t weight_row = out_dim;
					const int64_t block_idx	 = (weight_row * total_attn_dim + attn_dim) / 32;
					const int64_t elem_idx	 = (weight_row * total_attn_dim + attn_dim) % 32;

					const block_q8_0<short>& weight_block = output_weights[block_idx];
					const float scale					  = static_cast<float>(weight_block.d);

					__m128i weight_quant_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&weight_block.qs[elem_idx]));
					__m256i weight_quant_32 = _mm256_cvtepi16_epi32(weight_quant_16);
					__m256 weight_float		= _mm256_cvtepi32_ps(weight_quant_32);
					weight_float			= _mm256_mul_ps(weight_float, _mm256_set1_ps(scale));

					output_accum = _mm256_fmadd_ps(attn_vec, weight_float, output_accum);
				}

				float output_remainder = 0.0f;
				for (int64_t attn_dim = simd_limit; attn_dim < total_attn_dim; ++attn_dim) {
					const float attn_val = attention_output[attn_dim];

					const int64_t weight_row = out_dim;
					const int64_t block_idx	 = (weight_row * total_attn_dim + attn_dim) / 32;
					const int64_t elem_idx	 = (weight_row * total_attn_dim + attn_dim) % 32;

					const block_q8_0<short>& weight_block = output_weights[block_idx];
					const float scale					  = static_cast<float>(weight_block.d);
					const float weight_val				  = static_cast<float>(weight_block.qs[elem_idx]) * scale;

					output_remainder += attn_val * weight_val;
				}

				alignas(32) float output_vals[8];
				_mm256_store_ps(output_vals, output_accum);
				float projection_result = output_remainder;
				for (int i = 0; i < 8; ++i) {
					projection_result += output_vals[i];
				}

				final_output[out_dim] = projection_result + residual_input[out_dim];
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<0, kernel_types::fused_swiglu, transform_type, core_type, float, float, block_q8_0<half>, block_q8_0<half>, block_q8_0<half>, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03, const typename core_type::input_04_type& input04,
			const typename core_type::input_05_type& input05) {
			std::cout << core_type::op_type << " Dims: " << core_type::get_array() << std::endl;
			std::cout << core_type::op_type << " Bytes: " << core_type::total_required_bytes << std::endl;
			using model_traits_type				 = typename core_type::model_traits_type;
			static constexpr uint64_t hidden_dim = model_traits_type::embedding_length;
			static constexpr uint64_t ffn_dim	 = model_traits_type::feed_forward_length;

			const int64_t seq_len	   = input01[0];
			const int64_t block_id	   = thread_index / thread_count;
			const int64_t local_thread = thread_index % thread_count;

			const float* attention_output			 = input01.data;
			const block_q8_0<short>* ffn_norm_weight = input02.data[0];
			const block_q8_0<short>* gate_weights	 = input03.data[0];
			const block_q8_0<short>* up_weights		 = input04.data[0];
			const block_q8_0<short>* down_weights	 = input05.data[0];

			float* final_output = output.data;

			const int64_t elements_per_thread = (hidden_dim + thread_count - 1) / thread_count;
			const int64_t elem_start		  = local_thread * elements_per_thread;
			const int64_t elem_end			  = std::min(elem_start + elements_per_thread, ( int64_t )hidden_dim);

			const __m256 zero = _mm256_setzero_ps();

			__m256 sum	  = zero;
			__m256 sum_sq = zero;

			const int64_t simd_limit = (hidden_dim / 8) * 8;
			for (int64_t i = 0; i < simd_limit; i += 8) {
				__m256 x = _mm256_loadu_ps(&attention_output[i]);
				sum		 = _mm256_add_ps(sum, x);
				sum_sq	 = _mm256_fmadd_ps(x, x, sum_sq);
			}

			float mean_scalar = 0.0f, var_scalar = 0.0f;
			for (int64_t i = simd_limit; i < hidden_dim; ++i) {
				float x = attention_output[i];
				mean_scalar += x;
				var_scalar += x * x;
			}

			alignas(32) float mean_vals[8], var_vals[8];
			_mm256_store_ps(mean_vals, sum);
			_mm256_store_ps(var_vals, sum_sq);

			float mean	   = mean_scalar;
			float variance = var_scalar;
			for (int i = 0; i < 8; ++i) {
				mean += mean_vals[i];
				variance += var_vals[i];
			}

			mean /= hidden_dim;
			variance			= variance / hidden_dim - mean * mean;
			const float inv_std = 1.0f / sqrtf(variance + 1e-5f);

			const __m256 mean_vec	 = _mm256_set1_ps(mean);
			const __m256 inv_std_vec = _mm256_set1_ps(inv_std);

			alignas(32) static thread_local float gate_intermediate[14336];
			alignas(32) static thread_local float up_intermediate[14336];

			for (int64_t ffn_dim_idx = 0; ffn_dim_idx < ffn_dim; ++ffn_dim_idx) {
				__m256 gate_accum = zero;
				__m256 up_accum	  = zero;

				for (int64_t i = 0; i < simd_limit; i += 8) {
					__m256 x = _mm256_loadu_ps(&attention_output[i]);

					const int64_t norm_block_idx		= i / 32;
					const int64_t norm_elem_idx			= i % 32;
					const block_q8_0<short>& norm_block = ffn_norm_weight[norm_block_idx];
					const float norm_scale				= static_cast<float>(norm_block.d);

					__m128i norm_quant_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&norm_block.qs[norm_elem_idx]));
					__m256i norm_quant_32 = _mm256_cvtepi16_epi32(norm_quant_16);
					__m256 norm_w		  = _mm256_cvtepi32_ps(norm_quant_32);
					norm_w				  = _mm256_mul_ps(norm_w, _mm256_set1_ps(norm_scale));

					__m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
					normalized		  = _mm256_mul_ps(normalized, norm_w);

					const int64_t gate_block_idx		= (ffn_dim_idx * hidden_dim + i) / 32;
					const int64_t gate_elem_idx			= (ffn_dim_idx * hidden_dim + i) % 32;
					const block_q8_0<short>& gate_block = gate_weights[gate_block_idx];
					const float gate_scale				= static_cast<float>(gate_block.d);

					__m128i gate_quant_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&gate_block.qs[gate_elem_idx]));
					__m256i gate_quant_32 = _mm256_cvtepi16_epi32(gate_quant_16);
					__m256 gate_weights_f = _mm256_cvtepi32_ps(gate_quant_32);
					gate_weights_f		  = _mm256_mul_ps(gate_weights_f, _mm256_set1_ps(gate_scale));

					gate_accum = _mm256_fmadd_ps(normalized, gate_weights_f, gate_accum);

					const int64_t up_block_idx		  = (ffn_dim_idx * hidden_dim + i) / 32;
					const int64_t up_elem_idx		  = (ffn_dim_idx * hidden_dim + i) % 32;
					const block_q8_0<short>& up_block = up_weights[up_block_idx];
					const float up_scale			  = static_cast<float>(up_block.d);

					__m128i up_quant_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&up_block.qs[up_elem_idx]));
					__m256i up_quant_32 = _mm256_cvtepi16_epi32(up_quant_16);
					__m256 up_weights_f = _mm256_cvtepi32_ps(up_quant_32);
					up_weights_f		= _mm256_mul_ps(up_weights_f, _mm256_set1_ps(up_scale));

					up_accum = _mm256_fmadd_ps(normalized, up_weights_f, up_accum);
				}

				float gate_remainder = 0.0f, up_remainder = 0.0f;
				for (int64_t i = simd_limit; i < hidden_dim; ++i) {
					const int64_t norm_block_idx		= i / 32;
					const int64_t norm_elem_idx			= i % 32;
					const block_q8_0<short>& norm_block = ffn_norm_weight[norm_block_idx];
					const float norm_scale				= static_cast<float>(norm_block.d);
					const float norm_weight				= static_cast<float>(norm_block.qs[norm_elem_idx]) * norm_scale;

					float normalized = (attention_output[i] - mean) * inv_std * norm_weight;

					const int64_t gate_block_idx		= (ffn_dim_idx * hidden_dim + i) / 32;
					const int64_t gate_elem_idx			= (ffn_dim_idx * hidden_dim + i) % 32;
					const block_q8_0<short>& gate_block = gate_weights[gate_block_idx];
					const float gate_scale				= static_cast<float>(gate_block.d);
					const float gate_weight				= static_cast<float>(gate_block.qs[gate_elem_idx]) * gate_scale;
					gate_remainder += normalized * gate_weight;

					const int64_t up_block_idx		  = (ffn_dim_idx * hidden_dim + i) / 32;
					const int64_t up_elem_idx		  = (ffn_dim_idx * hidden_dim + i) % 32;
					const block_q8_0<short>& up_block = up_weights[up_block_idx];
					const float up_scale			  = static_cast<float>(up_block.d);
					const float up_weight			  = static_cast<float>(up_block.qs[up_elem_idx]) * up_scale;
					up_remainder += normalized * up_weight;
				}

				alignas(32) float gate_vals[8], up_vals[8];
				_mm256_store_ps(gate_vals, gate_accum);
				_mm256_store_ps(up_vals, up_accum);

				float gate_result = gate_remainder, up_result = up_remainder;
				for (int i = 0; i < 8; ++i) {
					gate_result += gate_vals[i];
					up_result += up_vals[i];
				}

				gate_intermediate[ffn_dim_idx] = gate_result;
				up_intermediate[ffn_dim_idx]   = up_result;
			}

			for (int64_t ffn_dim_idx = 0; ffn_dim_idx < ffn_dim; ++ffn_dim_idx) {
				float gate_val = gate_intermediate[ffn_dim_idx];
				float up_val   = up_intermediate[ffn_dim_idx];

				float silu_gate				   = gate_val / (1.0f + expf(-gate_val));
				gate_intermediate[ffn_dim_idx] = silu_gate * up_val;
			}

			for (int64_t out_dim = elem_start; out_dim < elem_end; ++out_dim) {
				__m256 down_accum = zero;

				const int64_t ffn_simd_limit = (ffn_dim / 8) * 8;
				for (int64_t ffn_idx = 0; ffn_idx < ffn_simd_limit; ffn_idx += 8) {
					__m256 swiglu_vec = _mm256_loadu_ps(&gate_intermediate[ffn_idx]);

					const int64_t down_block_idx = (out_dim * ffn_dim + ffn_idx) / 32;
					const int64_t down_elem_idx	 = (out_dim * ffn_dim + ffn_idx) % 32;

					const block_q8_0<short>& down_block = down_weights[down_block_idx];
					const float down_scale				= static_cast<float>(down_block.d);

					__m128i down_quant_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&down_block.qs[down_elem_idx]));
					__m256i down_quant_32 = _mm256_cvtepi16_epi32(down_quant_16);
					__m256 down_weights_f = _mm256_cvtepi32_ps(down_quant_32);
					down_weights_f		  = _mm256_mul_ps(down_weights_f, _mm256_set1_ps(down_scale));

					down_accum = _mm256_fmadd_ps(swiglu_vec, down_weights_f, down_accum);
				}

				float down_remainder = 0.0f;
				for (int64_t ffn_idx = ffn_simd_limit; ffn_idx < ffn_dim; ++ffn_idx) {
					float swiglu_val = gate_intermediate[ffn_idx];

					const int64_t down_block_idx = (out_dim * ffn_dim + ffn_idx) / 32;
					const int64_t down_elem_idx	 = (out_dim * ffn_dim + ffn_idx) % 32;

					const block_q8_0<short>& down_block = down_weights[down_block_idx];
					const float down_scale				= static_cast<float>(down_block.d);
					const float down_weight				= static_cast<float>(down_block.qs[down_elem_idx]) * down_scale;

					down_remainder += swiglu_val * down_weight;
				}

				alignas(32) float down_vals[8];
				_mm256_store_ps(down_vals, down_accum);
				float final_result = down_remainder;
				for (int i = 0; i < 8; ++i) {
					final_result += down_vals[i];
				}

				final_output[out_dim] = final_result + attention_output[out_dim];
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<0, kernel_types::fused_final_proj, transform_type, core_type, float, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::fused_final_proj, core_type, float, float, float, block_q8_0<half>> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
			using model_traits_type				 = typename core_type::model_traits_type;
			static constexpr uint64_t hidden_dim = model_traits_type::embedding_length;
			static constexpr uint64_t vocab_size = model_traits_type::vocab_size;

			const int64_t seq_len	   = input01[0];
			const int64_t block_id	   = thread_index >> tzcnt(static_cast<uint64_t>(thread_count));
			const int64_t local_thread = thread_index & (thread_count - 1);

			const float* swiglu_output				= input01.data;
			const float* output_norm_weight			= input02.data;
			const block_q8_0<short>* output_weights = input03.data;

			float* logits = output.data;

			const int64_t vocab_per_thread = (vocab_size + thread_count - 1) / thread_count;
			const int64_t vocab_start	   = local_thread * vocab_per_thread;
			const int64_t vocab_end		   = std::min(vocab_start + vocab_per_thread, ( int64_t )vocab_size);

			const __m256 zero		 = _mm256_setzero_ps();
			const int64_t simd_limit = (hidden_dim / 8) * 8;

			__m256 sum	  = zero;
			__m256 sum_sq = zero;

			for (int64_t i = 0; i < simd_limit; i += 8) {
				__m256 x = _mm256_loadu_ps(&swiglu_output[i]);
				sum		 = _mm256_add_ps(sum, x);
				sum_sq	 = _mm256_fmadd_ps(x, x, sum_sq);
			}

			float mean_scalar = 0.0f, var_scalar = 0.0f;
			for (int64_t i = simd_limit; i < hidden_dim; ++i) {
				float x = swiglu_output[i];
				mean_scalar += x;
				var_scalar += x * x;
			}

			alignas(32) float mean_vals[8], var_vals[8];
			_mm256_store_ps(mean_vals, sum);
			_mm256_store_ps(var_vals, sum_sq);

			float mean	   = mean_scalar;
			float variance = var_scalar;
			for (int i = 0; i < 8; ++i) {
				mean += mean_vals[i];
				variance += var_vals[i];
			}

			mean /= hidden_dim;
			variance			= variance / hidden_dim - mean * mean;
			const float inv_std = 1.0f / sqrtf(variance + 1e-5f);

			const __m256 mean_vec	 = _mm256_set1_ps(mean);
			const __m256 inv_std_vec = _mm256_set1_ps(inv_std);

			alignas(32) float normalized_activations[hidden_dim];

			for (int64_t i = 0; i < simd_limit; i += 8) {
				__m256 x		  = _mm256_loadu_ps(&swiglu_output[i]);
				__m256 norm_w	  = _mm256_loadu_ps(&output_norm_weight[i]);
				__m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x, mean_vec), inv_std_vec);
				normalized		  = _mm256_mul_ps(normalized, norm_w);
				_mm256_store_ps(&normalized_activations[i], normalized);
			}

			for (int64_t i = simd_limit; i < hidden_dim; ++i) {
				normalized_activations[i] = (swiglu_output[i] - mean) * inv_std * output_norm_weight[i];
			}

			for (int64_t vocab_idx = vocab_start; vocab_idx < vocab_end; ++vocab_idx) {
				__m256 logit_accum = zero;

				const int64_t weight_base = vocab_idx * hidden_dim;

				for (int64_t i = 0; i < simd_limit; i += 8) {
					__m256 normalized = _mm256_load_ps(&normalized_activations[i]);

					const int64_t global_idx = weight_base + i;
					const int64_t block_idx	 = global_idx >> 5;
					const int64_t elem_idx	 = global_idx & 31;

					const block_q8_0<short>& weight_block = output_weights[block_idx];
					const __m256 scale_vec				  = _mm256_set1_ps(static_cast<float>(weight_block.d));

					__m128i weight_quant_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&weight_block.qs[elem_idx]));
					__m256i weight_quant_32 = _mm256_cvtepi16_epi32(weight_quant_16);
					__m256 weight_float		= _mm256_cvtepi32_ps(weight_quant_32);
					weight_float			= _mm256_mul_ps(weight_float, scale_vec);

					logit_accum = _mm256_fmadd_ps(normalized, weight_float, logit_accum);
				}

				float logit_remainder = 0.0f;
				for (int64_t i = simd_limit; i < hidden_dim; ++i) {
					const int64_t global_idx = weight_base + i;
					const int64_t block_idx	 = global_idx >> 5;
					const int64_t elem_idx	 = global_idx & 31;

					const block_q8_0<short>& weight_block = output_weights[block_idx];
					const float scale					  = static_cast<float>(weight_block.d);
					const float weight_val				  = static_cast<float>(weight_block.qs[elem_idx]) * scale;

					logit_remainder += normalized_activations[i] * weight_val;
				}

				alignas(32) float logit_vals[8];
				_mm256_store_ps(logit_vals, logit_accum);
				float final_logit = logit_remainder;
				for (int i = 0; i < 8; ++i) {
					final_logit += logit_vals[i];
				}

				logits[vocab_idx] = final_logit;
			}
		}
	};

#endif

}
