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

#if NIHILUS_AVX512

namespace nihilus {

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::none, processing_phases::prompt_eval_time, transform_type, core_type, float, float, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::none, processing_phases::prompt_eval_time, transform_type, core_type, float, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_type, float, float, float, block_q8_0<half>> {
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
	struct kernel_dispatcher_impl<2, kernel_types::none, processing_phases::prompt_eval_time, transform_type, core_type, float, float, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::mul, processing_phases::prompt_eval_time, transform_type, core_type, float, float, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::get_rows, processing_phases::prompt_eval_time, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::get_rows, processing_phases::prompt_eval_time, transform_type, core_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::mul_mat, processing_phases::prompt_eval_time, transform_type, core_type, float, block_q8_0<half>, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::mul_mat, processing_phases::prompt_eval_time, transform_type, core_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::softmax, processing_phases::prompt_eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::copy, processing_phases::prompt_eval_time, transform_type, core_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::rope, processing_phases::prompt_eval_time, transform_type, core_type, float, float, int32_t, float>
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
		static constexpr uint64_t rope_dimension_count = core_type::model_traits_type::rope_dimension_count;
		static constexpr uint32_t attention_head_count = core_type::model_traits_type::attention_head_count;

		static constexpr uint64_t batch_size				  = input_type01::get_array()[0];
		static constexpr uint64_t num_heads					  = input_type01::get_array()[2];
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

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::copy, processing_phases::prompt_eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::copy, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::cont, processing_phases::prompt_eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::cont, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::silu, processing_phases::prompt_eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::none, processing_phases::eval_time, transform_type, core_type, float, float, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::none, processing_phases::eval_time, transform_type, core_type, float, float, float, block_q8_0<half>>
		: public kernel_base<kernel_types::none, core_type, float, float, float, block_q8_0<half>> {
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
	struct kernel_dispatcher_impl<2, kernel_types::none, processing_phases::eval_time, transform_type, core_type, float, float, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::mul, processing_phases::eval_time, transform_type, core_type, float, float, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::get_rows, processing_phases::eval_time, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		using input_type01 = core_type::input_01_type;
		using input_type02 = core_type::input_02_type;
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::get_rows, processing_phases::eval_time, transform_type, core_type, float, float, int32_t>
		: public kernel_base<kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::mul_mat, processing_phases::eval_time, transform_type, core_type, float, block_q8_0<half>, float>
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
	struct kernel_dispatcher_impl<2, kernel_types::mul_mat, processing_phases::eval_time, transform_type, core_type, float, half, float>
		: public kernel_base<kernel_types::mul_mat, core_type, float, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::softmax, processing_phases::eval_time, transform_type, core_type, float, float, float>
		: public kernel_base<kernel_types::softmax, core_type, float, float, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::copy, processing_phases::eval_time, transform_type, core_type, half, half, float>
		: public kernel_base<kernel_types::copy, core_type, half, half, float> {
		using input_type01 = typename core_type::input_01_type;
		using input_type02 = typename core_type::input_02_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01,
			const typename core_type::input_02_type& input02) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::rope, processing_phases::eval_time, transform_type, core_type, float, float, int32_t, float>
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
		static constexpr uint64_t rope_dimension_count = core_type::model_traits_type::rope_dimension_count;
		static constexpr uint32_t attention_head_count = core_type::model_traits_type::attention_head_count;

		static constexpr uint64_t batch_size				  = input_type01::get_array()[0];
		static constexpr uint64_t num_heads					  = input_type01::get_array()[2];
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

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::copy, processing_phases::eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::copy, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::cont, processing_phases::eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::cont, core_type, float, float> {
		using input_type01 = typename core_type::input_01_type;

		NIHILUS_INLINE static void impl(int64_t thread_index, int64_t thread_count, int64_t current_block, core_type& output, const typename core_type::input_01_type& input01) {
		}
	};

	template<typename transform_type, typename core_type>
	struct kernel_dispatcher_impl<2, kernel_types::silu, processing_phases::eval_time, transform_type, core_type, float, float>
		: public kernel_base<kernel_types::silu, core_type, float, float> {
		NIHILUS_INLINE static void impl(int64_t, int64_t, int64_t, core_type&, const typename core_type::input_01_type&) {
		}
	};

};

#endif
