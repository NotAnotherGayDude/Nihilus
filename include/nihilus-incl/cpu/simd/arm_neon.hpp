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

#if NIHILUS_NEON

	#include <arm_neon.h>

namespace nihilus {

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::none, processing_phase::prompt_eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::none, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::none, processing_phase::prompt_eval_time, transform_type, core_type, float, float, float, float>
		: public kernel_base<kernel_types::none, core_type, float, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		using input_type03 = typename core_type::input_03_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t ne01, uint64_t ne11, uint64_t ne21,
			const float* __restrict src0_data, const float* __restrict src1_data, const block_q8_0<half>* __restrict src2_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::none, processing_phase::prompt_eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::none, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			static constexpr float eps			 = core_type::model_traits_type::layer_norm_rms_epsilon;
			static constexpr uint64_t ne00		 = input_type01::get_array()[0];
			const uint64_t ne01					 = input01[1];
			static constexpr uint64_t ne10		 = input_type02::get_array()[0];
			const uint64_t ith					 = static_cast<uint64_t>(thread_index);
			const uint64_t nth					 = static_cast<uint64_t>(thread_count);
			const float* __restrict input01_data = input01.data;
			const float* __restrict input02_data = input02.data[current_block];
			float* __restrict output_data		 = output.data;
			const uint64_t total_elements		 = ne00 * ne01;
			if (ith == 0) {
				for (uint64_t i = 0; i < ne01; ++i) {
					const uint64_t input_offset	 = i * ne00;
					const uint64_t output_offset = i * ne00;
					float sum_squares			 = 0.0f;
					for (uint64_t j = 0; j < ne00; ++j) {
						const float val = input01_data[input_offset + j];
						sum_squares += val * val;
					}
					const float mean_square = sum_squares / static_cast<float>(ne00);
					const float rms_norm	= 1.0f / sqrtf(mean_square + eps);

					for (uint64_t j = 0; j < ne00; ++j) {
						output_data[output_offset + j] = input01_data[input_offset + j] * rms_norm * input02_data[j];
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::mul, processing_phase::prompt_eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t nr, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::get_rows, processing_phase::prompt_eval_time, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			static constexpr uint64_t ne00					= input_type01::get_array()[0];
			const uint64_t ne01								= input01[1];
			static constexpr uint64_t ne02					= input_type01::get_array()[2];
			static constexpr uint64_t ne10					= input_type02::get_array()[0];
			const uint64_t ne11								= input02[1];
			const uint64_t nr								= count_elements(input02);
			const uint64_t ith								= static_cast<uint64_t>(thread_index);
			const uint64_t nth								= static_cast<uint64_t>(thread_count);
			const uint64_t dr								= (nr + nth - 1ull) / nth;
			const uint64_t ir0								= dr * ith;
			const uint64_t ir1								= detail::min(ir0 + dr, nr);
			const block_q8_0<half>* __restrict input01_data = input01.data;
			const int32_t* __restrict input02_data			= input02.data;
			float* __restrict output_data					= output.data;
			static constexpr uint64_t blocks_per_row		= ne00 / Q_SIZE;
			static constexpr uint64_t input01_stride_dim1	= blocks_per_row;
			const uint64_t input01_stride_dim2				= ne01 * blocks_per_row;
			const uint64_t input01_stride_dim3				= ne01 * ne02 * blocks_per_row;
			static constexpr uint64_t input02_stride_dim1	= 1;
			static constexpr uint64_t input02_stride_dim2	= ne10;
			const uint64_t input02_stride_dim3				= ne10 * ne11;
			static constexpr uint64_t output_stride_dim1	= ne00;
			const uint64_t output_stride_dim2				= output[1] * ne00;
			const uint64_t output_stride_dim3				= output[1] * output[2] * ne00;
			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t i12				 = i / (ne11 * ne10);
				const uint64_t i11				 = (i - i12 * ne11 * ne10) / ne10;
				const uint64_t i10				 = i - i12 * ne11 * ne10 - i11 * ne10;
				const uint64_t input02_idx		 = i10 * input02_stride_dim1 + i11 * input02_stride_dim2 + i12 * input02_stride_dim3;
				const uint64_t token_id			 = static_cast<uint64_t>(input02_data[input02_idx]);
				const uint64_t input01_block_idx = token_id * input01_stride_dim1 + i11 * input01_stride_dim2 + i12 * input01_stride_dim3;
				const uint64_t output_idx		 = i10 * output_stride_dim1 + i11 * output_stride_dim2 + i12 * output_stride_dim3;

				for (uint64_t j = 0; j < blocks_per_row; ++j) {
					const uint64_t final_block_idx = input01_block_idx + j;

					const block_q8_0<half>& block = input01_data[final_block_idx];
					const float scale			  = fp16_to_fp32(block.d);

					for (uint64_t k = 0; k < Q_SIZE; ++k) {
						int8_t quant_val						 = static_cast<int8_t>(block.qs[k]);
						float result							 = scale * static_cast<float>(quant_val);
						output_data[output_idx + j * Q_SIZE + k] = result;
					}
				}
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::get_rows, processing_phase::eval_time, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			static constexpr uint64_t ne00					= input_type01::get_array()[0];
			const uint64_t nr								= count_elements(input02);
			const uint64_t ith								= static_cast<uint64_t>(thread_index);
			const uint64_t nth								= static_cast<uint64_t>(thread_count);
			const uint64_t dr								= (nr + nth - 1ull) / nth;
			const uint64_t ir0								= dr * ith;
			const uint64_t ir1								= detail::min(ir0 + dr, nr);
			const block_q8_0<half>* __restrict input01_data = input01.data;
			const int32_t* __restrict input02_data			= input02.data;
			float* __restrict output_data					= output.data;
			static constexpr uint64_t blocks_per_row		= ne00 / Q_SIZE;
			for (uint64_t i = ir0; i < ir1; ++i) {
				const uint64_t token_id			 = static_cast<uint64_t>(input02_data[i]);
				const uint64_t input01_block_idx = token_id * blocks_per_row;
				const uint64_t output_idx		 = i * ne00;
				for (uint64_t j = 0; j < blocks_per_row; ++j) {
					const block_q8_0<half>& block = input01_data[input01_block_idx + j];
					const float scale			  = fp16_to_fp32(block.d);
					for (uint64_t k = 0; k < Q_SIZE; ++k) {
						output_data[output_idx + j * Q_SIZE + k] = scale * static_cast<float>(static_cast<int8_t>(block.qs[k]));
					}
				}
			}
		}
	};
	NIHILUS_INLINE static half fp32_to_fp16(float f) {
		static constexpr float scale_to_inf	 = fp32_from_bits(0x77800000);
		static constexpr float scale_to_zero = fp32_from_bits(0x08800000);
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

	NIHILUS_INLINE void quantize_row_q8_0_scalar(const float* __restrict x, block_q8_0<half>* __restrict vy, int64_t k) {
		const int64_t nb			   = k / Q_SIZE;
		block_q8_0<half>* __restrict y = vy;

		for (int64_t i = 0; i < nb; i++) {
			float maxAbs = 0.0f;
			for (int j = 0; j < 32; j++) {
				float absVal = fabsf(x[j]);
				if (absVal > maxAbs) {
					maxAbs = absVal;
				}
			}

			const float d = maxAbs / 127.0f;
			y[i].d		  = fp32_to_fp16(d);

			const float id = (maxAbs != 0.0f) ? 127.0f / maxAbs : 0.0f;

			for (int j = 0; j < 32; j++) {
				float scaled	  = x[j] * id;
				float rounded	  = roundf(scaled);
				int32_t quantized = ( int32_t )rounded;
				if (quantized > 127)
					quantized = 127;
				if (quantized < -128)
					quantized = -128;

				y[i].qs[j] = ( int8_t )quantized;
			}

			x += 32;
		}
	}

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::mul_mat, processing_phase::prompt_eval_time, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		using input_type01			   = typename core_type::input_01_type;
		using input_type02			   = typename core_type::input_02_type;
		static constexpr uint64_t ne00 = input_type01::get_array()[0];
		static constexpr uint64_t ne02 = input_type01::get_array()[2];
		static constexpr uint64_t ne03 = input_type01::get_array()[3];
		static constexpr uint64_t ne10 = input_type02::get_array()[0];
		static constexpr uint64_t ne12 = input_type02::get_array()[2];
		static constexpr uint64_t ne13 = input_type02::get_array()[3];
		static constexpr uint64_t ne0  = core_type::get_array()[0];
		static constexpr uint64_t ne2  = core_type::get_array()[2];
		static constexpr uint64_t ne3  = core_type::get_array()[3];

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
			const uint64_t ne01	   = input01[1];
			const uint64_t ne11	   = input02[1];
			const uint64_t ne1	   = output[1];
			using output_transform = typename core_type::transform_type;
			const block_q8_0<half>* src01;
			if constexpr (array_types<decltype(input01.data)>) {
				src01 = input01.data[current_block];
			} else {
				src01 = input01.data;
			}
			const float* src02 = input02.data;
			float* dst		   = output.data;

			const uint64_t output_size		  = ne01 * ne11;
			const uint64_t input_size		  = ne10 * ne11;
			const uint64_t quantized_blocks	  = (input_size + Q_SIZE - 1) / Q_SIZE;
			block_q8_0<half>* quantized_input = reinterpret_cast<block_q8_0<half>*>(dst + output_size);

			const uint64_t ith = static_cast<uint64_t>(thread_index);
			const uint64_t nth = static_cast<uint64_t>(thread_count);
			if (ith == 0) {
				for (uint64_t col = 0; col < ne11; ++col) {
					quantize_row_q8_0_scalar(src02 + col * ne10, quantized_input + col * (ne10 / Q_SIZE), ne10);
				}
			}

			const uint64_t dr  = (ne01 + nth - 1ull) / nth;
			const uint64_t ir0 = dr * ith;
			const uint64_t ir1 = detail::min(ir0 + dr, ne01);

			static constexpr uint64_t blocks_per_row = ne00 / Q_SIZE;

			for (uint64_t i1 = ir0; i1 < ir1; ++i1) {
				for (uint64_t i0 = 0; i0 < ne11; ++i0) {
					float sum = 0.0f;

					for (uint64_t block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
						const uint64_t weight_block_idx = i1 * blocks_per_row + block_idx;
						const uint64_t input_block_idx	= i0 * (ne10 / Q_SIZE) + block_idx;

						const block_q8_0<half>& weight_block = src01[weight_block_idx];
						const block_q8_0<half>& input_block	 = quantized_input[input_block_idx];

						const float weight_scale   = fp16_to_fp32(weight_block.d);
						const float input_scale	   = fp16_to_fp32(input_block.d);
						const float combined_scale = weight_scale * input_scale;

						for (uint64_t k = 0; k < Q_SIZE; ++k) {
							const int8_t weight_val = static_cast<int8_t>(weight_block.qs[k]);
							const int8_t input_val	= static_cast<int8_t>(input_block.qs[k]);
							sum += combined_scale * static_cast<float>(weight_val) * static_cast<float>(input_val);
						}
					}
					dst[output_transform::impl(i1, i0, ne01, ne11)] = sum;
				}
			}
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::mul_mat, processing_phase::prompt_eval_time, transform_type, core_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::softmax, processing_phase::prompt_eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::copy, processing_phase::prompt_eval_time, transform_type, core_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::rope, processing_phase::prompt_eval_time, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<kernel_types::rope, core_type, float, float, int32_t, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		using input_type03 = typename core_type::input_03_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
			const float* __restrict src_data		  = input01.data;
			const int32_t* pos_data					  = input02.data;
			const float* __restrict freq_scaling_data = input03.data;
			float* __restrict dst_data				  = output.data;
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::copy, processing_phase::prompt_eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::copy, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::cont, processing_phase::prompt_eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::cont, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::silu, processing_phase::prompt_eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::none, processing_phase::eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::none, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::none, processing_phase::eval_time, transform_type, core_type, float, float, float, float>
		: public kernel_base<kernel_types::none, core_type, float, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;
		using input_type03 = typename core_type::input_03_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t ne01, uint64_t ne11, uint64_t ne21,
			const float* __restrict src0_data, const float* __restrict src1_data, const block_q8_0<half>* __restrict src2_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::none, processing_phase::eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::none, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::mul, processing_phase::eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::mul, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_power_of_2able> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, uint64_t nr, uint64_t ne01, uint64_t ne11,
			const float* __restrict src0_data, const float* __restrict src1_data, float* __restrict dst_data) {
		}

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::get_rows, processing_phase::prompt_eval_time, transform_type, core_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::get_rows, processing_phase::eval_time, transform_type, core_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::mul_mat, processing_phase::eval_time, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		template<bool is_power_of_2abl> NIHILUS_INLINE static void process_tensor_elements(uint64_t ith, uint64_t nth, int64_t current_block, core_type& output,
			const typename core_type::input_01_type& input01, const typename core_type::input_02_type& input02) {
		}

	  public:
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::mul_mat, processing_phase::eval_time, transform_type, core_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::softmax, processing_phase::eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::copy, processing_phase::eval_time, transform_type, core_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<1, kernel_types::rope, processing_phase::eval_time, transform_type, core_type, float, float, int32_t, float>
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
		static constexpr uint64_t rope_dimension_count			   = core_type::model_traits_type::rope_dimension_count;
		static constexpr uint32_t attention_head_count = core_type::model_traits_type::attention_head_count;

		static constexpr uint64_t batch_size	  = input_type01::get_array()[0];
		static constexpr uint64_t num_heads		  = input_type01::get_array()[2];
		static constexpr uint64_t tensor_rope_dimension_count = input_type01::get_array()[3];

		static constexpr uint64_t rope_dim		= rope_dimension_count;
		static constexpr uint64_t half_rope_dim = rope_dim / 2;

		static constexpr auto freq_table = make_freq_table<half_rope_dim>();

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02, const typename core_type::input_03_type& input03) {
			/*
			const float* __restrict src_data		  = input01.data;
			const int32_t* pos_data					  = input02.data;
			const float* __restrict freq_scaling_data = input03.data;
			float* __restrict dst_data				  = output.data;

			const uint64_t seq_len = input01[1];

			const uint64_t total_work_items = batch_size * seq_len * num_heads;
			const uint64_t total_elements	= batch_size * seq_len * num_heads * rope_dimension_count;

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

						const uint64_t src_offset = i3 * seq_len * num_heads * rope_dimension_count + i2 * num_heads * rope_dimension_count + i1 * rope_dimension_count;
						const uint64_t dst_offset = i3 * seq_len * num_heads * rope_dimension_count + i2 * num_heads * rope_dimension_count + i1 * rope_dimension_count;

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

						for (int64_t i0 = rope_dim; i0 < rope_dimension_count; i0++) {
							dst_data[dst_offset + i0] = src_data[src_offset + i0];
						}
					}
				}
			}*/
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::copy, processing_phase::eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::copy, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::cont, processing_phase::eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::cont, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<1, kernel_types::silu, processing_phase::eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

};

#endif
