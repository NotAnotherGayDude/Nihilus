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

#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/file_io.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/infra/model_traits.hpp>
#include <latch>

namespace nihilus {

	enum class kernel_trait_static_assert_errors {
		silu_input_and_output_dimensions_must_match,
		silu_input_total_elements_must_match_output_total_elements,
		softmax_mask_dimension_1_must_match_sequence_length,
		softmax_mask_dimension_0_must_match_attention_heads,
		softmax_input_dimension_3_must_match_output,
		softmax_input_dimension_2_must_match_output,
		softmax_input_dimension_1_must_match_output,
		softmax_input_dimension_0_must_match_output,
		mul_mat_inner_dimensions_must_match_for_matrix_multiplication,
		mul_mat_output_rows_must_equal_weight_matrix_columns,
		mul_mat_output_batch_3_must_match_input_batch,
		mul_mat_output_batch_2_must_match_input_batch,
		mul_mat_output_columns_must_equal_second_input_columns,
		mul_mat_output_rows_must_equal_first_input_columns,
		mul_element_wise_dimension_0_must_match,
		mul_element_wise_dimension_1_must_match_or_be_broadcast,
		mul_element_wise_dimension_2_must_match_or_be_broadcast,
		mul_element_wise_dimension_3_must_match_or_be_broadcast,
		mul_output_dimension_0_must_match_input,
		mul_output_dimension_1_must_match_input,
		mul_output_dimension_2_must_match_input,
		mul_output_dimension_3_must_match_input,
		rms_norm_single_input_dimension_3_must_match_output,
		rms_norm_single_input_dimension_2_must_match_output,
		rms_norm_single_input_dimension_1_must_match_output,
		rms_norm_single_input_dimension_0_must_match_output,
		rms_norm_input_dimension_0_must_match_output,
		rms_norm_input_dimension_1_must_match_output,
		rms_norm_input_dimension_2_must_match_output,
		rms_norm_input_dimension_3_must_match_output,
		rms_norm_weight_dimension_0_must_match_input_dimension_0,
		rms_norm_weight_dimension_1_must_be_one_for_broadcast,
		rms_norm_weight_dimension_2_must_be_one_for_broadcast,
		rms_norm_weight_dimension_3_must_be_one_for_broadcast,
		get_rows_indices_dimension_0_must_match_output_dimension_1,
		get_rows_indices_dimension_1_must_be_one,
		get_rows_indices_dimension_2_must_be_one,
		get_rows_indices_dimension_3_must_be_one,
		get_rows_output_dimension_0_must_match_embedding_dimension_0,
		get_rows_output_dimension_2_must_be_one,
		get_rows_output_dimension_3_must_be_one,
		transpose_dimension_0_must_swap_with_dimension_1,
		transpose_dimension_1_must_swap_with_dimension_0,
		transpose_dimension_2_must_match,
		transpose_dimension_3_must_match,
		copy_input_total_elements_must_match_output_total_elements,
		copy_allows_type_conversion,
		logit_sample_input_dimension_0_must_be_vocab_size,
		logit_sample_input_dimension_1_must_match_output_dimension_0,
		logit_sample_output_dimension_1_must_be_one,
		logit_sample_output_dimension_2_must_be_one,
		logit_sample_output_dimension_3_must_be_one,
		permute_input_dimension_0_must_match_output_dimension_0,
		permute_input_dimension_1_must_match_output_dimension_2,
		permute_input_dimension_2_must_match_output_dimension_1,
		permute_input_dimension_3_must_match_output_dimension_3,
		reshape_input_total_elements_must_match_output_total_elements,
		reshape_dimension_product_must_be_preserved,
		rope_input_dimension_0_must_match_output_dimension_0,
		rope_input_dimension_1_must_match_output_dimension_1,
		rope_input_dimension_2_must_match_output_dimension_2,
		rope_input_dimension_3_must_match_output_dimension_3,
		rope_freq_dimension_0_must_be_half_of_rope_dimension_count,
		rope_freq_dimension_1_must_be_one,
		rope_freq_dimension_2_must_be_one,
		rope_freq_dimension_3_must_be_one,
		rope_position_dimension_0_must_equal_one_or_match_sequence_length,
		rope_position_dimension_1_must_be_one,
		rope_position_dimension_2_must_be_one,
		rope_position_dimension_3_must_be_one,
		unary_dimension_mismatch,
		repetition_penalty_logits_dim_mismatch,
		presence_penalty_logits_dim_mismatch,
		frequency_penalty_logits_dim_mismatch,
		temperature_scale_logits_dim_mismatch,
		top_k_filter_logits_dim_mismatch,
		top_p_filter_logits_dim_mismatch,
		vocab_mask_dimension_mismatch,
		vocab_mask_logits_dim_mismatch,
		sample_output_must_be_single_token,
	};

	enum class processing_phases {
		prompt_eval_time,
		eval_time,
	};

	template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

	template<size_t... indices> struct core_trait_dims;

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new> {
		static constexpr uint64_t runtime_dims{ 5 };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 5> {
		static constexpr uint64_t runtime_dims{ 5 };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 0, 1> {
		static constexpr uint64_t runtime_dims{ 0 };
		mutable uint64_t dim00{ dim00_new };
		mutable uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}

		NIHILUS_HOST uint64_t& get_mutable_dim(uint64_t index) const {
			if (index == 0) {
				return dim00;
			} else {
				return dim01;
			}
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 1, 2> {
		static constexpr uint64_t runtime_dims{ 0 };
		static constexpr uint64_t dim00{ dim00_new };
		mutable uint64_t dim01{ dim01_new };
		mutable uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}

		NIHILUS_HOST uint64_t& get_mutable_dim(uint64_t index) const {
			if (index == 1) {
				return dim01;
			} else {
				return dim02;
			}
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 2, 3> {
		static constexpr uint64_t runtime_dims{ 0 };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		mutable uint64_t dim02{ dim02_new };
		mutable uint64_t dim03{ dim03_new };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}

		NIHILUS_HOST uint64_t& get_mutable_dim(uint64_t index) const {
			if (index == 2) {
				return dim02;
			} else {
				return dim03;
			}
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 0> {
		static constexpr uint64_t runtime_dims{ 0 };
		mutable uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}

		NIHILUS_HOST array<uint64_t, 4> get_array_rt() {
			return { *dims[0], *dims[1], *dims[2], *dims[3] };
		}

		NIHILUS_HOST uint64_t& get_mutable_dim() const {
			return dim00;
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 1> {
		static constexpr uint64_t runtime_dims{ 1 };
		static constexpr uint64_t dim00{ dim00_new };
		mutable uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}

		NIHILUS_HOST array<uint64_t, 4> get_array_rt() {
			return { *dims[0], *dims[1], *dims[2], *dims[3] };
		}

		NIHILUS_HOST uint64_t& get_mutable_dim() const {
			return dim01;
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 2> {
		static constexpr uint64_t runtime_dims{ 2 };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		mutable uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_HOST static constexpr array<uint64_t, 4> get_array() {
			return { dim00_new, dim01_new, dim02_new, dim03_new };
		}

		NIHILUS_HOST array<uint64_t, 4> get_array_rt() {
			return { *dims[0], *dims[1], *dims[2], *dims[3] };
		}

		NIHILUS_HOST uint64_t& get_mutable_dim() const {
			return dim02;
		}
	};

	template<typename value_type>
	concept runtime_dims_t = detail::remove_cvref_t<value_type>::runtime_dims != 5;

	enum class get_new_dims_errors { unknown_kernel_type };

	template<typename config_type, typename core_traits_type, device_types devive_type, uint64_t, core_types core_type, processing_phases processing_phase>
	struct kernel_dispatcher_impl;

	template<typename config_type, typename dims_type, kernel_types kernel_type, typename output_type_new, typename... operand_types> struct kernel_traits;

	template<typename config_type, typename dims_type_new, typename output_type_new> struct kernel_traits<config_type, dims_type_new, kernel_types::weights, output_type_new>
		: public dims_type_new {
		using output_type = output_type_new;
		using dims_type	  = dims_type_new;
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::mul_mat, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();

		static_assert(static_assert_printer_val<(input_01_dims[0] == input_02_dims[0]),
			kernel_trait_static_assert_errors::mul_mat_inner_dimensions_must_match_for_matrix_multiplication, input_01_dims[0], input_02_dims[0]>::impl);

		static_assert(static_assert_printer_val<(output_dims[0] == input_01_dims[1]), kernel_trait_static_assert_errors::mul_mat_output_rows_must_equal_first_input_columns,
			output_dims[0], input_01_dims[1]>::impl);

		static_assert(static_assert_printer_val<(output_dims[1] == input_02_dims[1]), kernel_trait_static_assert_errors::mul_mat_output_columns_must_equal_second_input_columns,
			output_dims[1], input_02_dims[1]>::impl);

		static_assert(static_assert_printer_val<(output_dims[2] == input_01_dims[2] || output_dims[2] == input_02_dims[2]),
			kernel_trait_static_assert_errors::mul_mat_output_batch_2_must_match_input_batch, output_dims[2], input_01_dims[2]>::impl);

		static_assert(static_assert_printer_val<(output_dims[3] == input_01_dims[3] || output_dims[3] == input_02_dims[3]),
			kernel_trait_static_assert_errors::mul_mat_output_batch_3_must_match_input_batch, output_dims[3], input_01_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::add, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == input_02_dims[0]), kernel_trait_static_assert_errors::mul_element_wise_dimension_0_must_match,
			input_01_dims[0], input_02_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == input_02_dims[1] || input_02_dims[1] == 1),
			kernel_trait_static_assert_errors::mul_element_wise_dimension_1_must_match_or_be_broadcast, input_01_dims[1], input_02_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == input_02_dims[2] || input_02_dims[2] == 1),
			kernel_trait_static_assert_errors::mul_element_wise_dimension_2_must_match_or_be_broadcast, input_01_dims[2], input_02_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == input_02_dims[3] || input_02_dims[3] == 1),
			kernel_trait_static_assert_errors::mul_element_wise_dimension_3_must_match_or_be_broadcast, input_01_dims[3], input_02_dims[3]>::impl);
		static_assert(static_assert_printer_val<(output_dims[0] == input_01_dims[0]), kernel_trait_static_assert_errors::mul_output_dimension_0_must_match_input, output_dims[0],
			input_01_dims[0]>::impl);
		static_assert(static_assert_printer_val<(output_dims[1] == input_01_dims[1]), kernel_trait_static_assert_errors::mul_output_dimension_1_must_match_input, output_dims[1],
			input_01_dims[1]>::impl);
		static_assert(static_assert_printer_val<(output_dims[2] == input_01_dims[2]), kernel_trait_static_assert_errors::mul_output_dimension_2_must_match_input, output_dims[2],
			input_01_dims[2]>::impl);
		static_assert(static_assert_printer_val<(output_dims[3] == input_01_dims[3]), kernel_trait_static_assert_errors::mul_output_dimension_3_must_match_input, output_dims[3],
			input_01_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::mul, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == input_02_dims[0]), kernel_trait_static_assert_errors::mul_element_wise_dimension_0_must_match,
			input_01_dims[0], input_02_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == input_02_dims[1] || input_02_dims[1] == 1),
			kernel_trait_static_assert_errors::mul_element_wise_dimension_1_must_match_or_be_broadcast, input_01_dims[1], input_02_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == input_02_dims[2] || input_02_dims[2] == 1),
			kernel_trait_static_assert_errors::mul_element_wise_dimension_2_must_match_or_be_broadcast, input_01_dims[2], input_02_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == input_02_dims[3] || input_02_dims[3] == 1),
			kernel_trait_static_assert_errors::mul_element_wise_dimension_3_must_match_or_be_broadcast, input_01_dims[3], input_02_dims[3]>::impl);
		static_assert(static_assert_printer_val<(output_dims[0] == input_01_dims[0]), kernel_trait_static_assert_errors::mul_output_dimension_0_must_match_input, output_dims[0],
			input_01_dims[0]>::impl);
		static_assert(static_assert_printer_val<(output_dims[1] == input_01_dims[1]), kernel_trait_static_assert_errors::mul_output_dimension_1_must_match_input, output_dims[1],
			input_01_dims[1]>::impl);
		static_assert(static_assert_printer_val<(output_dims[2] == input_01_dims[2]), kernel_trait_static_assert_errors::mul_output_dimension_2_must_match_input, output_dims[2],
			input_01_dims[2]>::impl);
		static_assert(static_assert_printer_val<(output_dims[3] == input_01_dims[3]), kernel_trait_static_assert_errors::mul_output_dimension_3_must_match_input, output_dims[3],
			input_01_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::rms_norm, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::rms_norm_input_dimension_0_must_match_output,
			input_01_dims[0], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[1]), kernel_trait_static_assert_errors::rms_norm_input_dimension_1_must_match_output,
			input_01_dims[1], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == output_dims[2]), kernel_trait_static_assert_errors::rms_norm_input_dimension_2_must_match_output,
			input_01_dims[2], output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == output_dims[3]), kernel_trait_static_assert_errors::rms_norm_input_dimension_3_must_match_output,
			input_01_dims[3], output_dims[3]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[0] == input_01_dims[0]), kernel_trait_static_assert_errors::rms_norm_weight_dimension_0_must_match_input_dimension_0,
			input_02_dims[0], input_01_dims[0]>::impl);
		static_assert(
			static_assert_printer_val<(input_02_dims[1] == 1), kernel_trait_static_assert_errors::rms_norm_weight_dimension_1_must_be_one_for_broadcast, input_02_dims[1]>::impl);
		static_assert(
			static_assert_printer_val<(input_02_dims[2] == 1), kernel_trait_static_assert_errors::rms_norm_weight_dimension_2_must_be_one_for_broadcast, input_02_dims[2]>::impl);
		static_assert(
			static_assert_printer_val<(input_02_dims[3] == 1), kernel_trait_static_assert_errors::rms_norm_weight_dimension_3_must_be_one_for_broadcast, input_02_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::copy, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static constexpr auto input_total	= input_02_dims[0] * input_02_dims[1] * input_02_dims[2] * input_02_dims[3];
		static constexpr auto output_total	= output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer_val<(input_total == output_total), kernel_trait_static_assert_errors::copy_input_total_elements_must_match_output_total_elements,
			input_total, output_total>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::silu, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static constexpr auto input_total	= input_01_dims[0] * input_01_dims[1] * input_01_dims[2] * input_01_dims[3];
		static constexpr auto output_total	= output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer_val<(input_total == output_total), kernel_trait_static_assert_errors::silu_input_total_elements_must_match_output_total_elements,
			input_total, output_total>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::silu_input_and_output_dimensions_must_match,
			input_01_dims[0], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[1]), kernel_trait_static_assert_errors::silu_input_and_output_dimensions_must_match,
			input_01_dims[1], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == output_dims[2]), kernel_trait_static_assert_errors::silu_input_and_output_dimensions_must_match,
			input_01_dims[2], output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == output_dims[3]), kernel_trait_static_assert_errors::silu_input_and_output_dimensions_must_match,
			input_01_dims[3], output_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::transpose, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(output_dims[0] == input_01_dims[1]), kernel_trait_static_assert_errors::transpose_dimension_0_must_swap_with_dimension_1,
			output_dims[0], input_01_dims[1]>::impl);
		static_assert(static_assert_printer_val<(output_dims[1] == input_01_dims[0]), kernel_trait_static_assert_errors::transpose_dimension_1_must_swap_with_dimension_0,
			output_dims[1], input_01_dims[0]>::impl);
		static_assert(static_assert_printer_val<(output_dims[2] == input_01_dims[2]), kernel_trait_static_assert_errors::transpose_dimension_2_must_match, output_dims[2],
			input_01_dims[2]>::impl);
		static_assert(static_assert_printer_val<(output_dims[3] == input_01_dims[3]), kernel_trait_static_assert_errors::transpose_dimension_3_must_match, output_dims[3],
			input_01_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::view, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static constexpr auto input_total	= input_01_dims[0] * input_01_dims[1] * input_01_dims[2] * input_01_dims[3];
		static constexpr auto output_total	= output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer_val<(input_total >= output_total), kernel_trait_static_assert_errors::copy_input_total_elements_must_match_output_total_elements,
			input_total, output_total>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::sample_logits, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == model_traits_type<config_type>::vocab_size),
			kernel_trait_static_assert_errors::logit_sample_input_dimension_0_must_be_vocab_size, input_01_dims[0], model_traits_type<config_type>::vocab_size>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[0]),
			kernel_trait_static_assert_errors::logit_sample_input_dimension_1_must_match_output_dimension_0, input_01_dims[1], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(output_dims[1] == 1), kernel_trait_static_assert_errors::logit_sample_output_dimension_1_must_be_one, output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(output_dims[2] == 1), kernel_trait_static_assert_errors::logit_sample_output_dimension_2_must_be_one, output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(output_dims[3] == 1), kernel_trait_static_assert_errors::logit_sample_output_dimension_3_must_be_one, output_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::permute, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::permute_input_dimension_0_must_match_output_dimension_0,
			input_01_dims[0], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[2]), kernel_trait_static_assert_errors::permute_input_dimension_1_must_match_output_dimension_2,
			input_01_dims[1], output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == output_dims[1]), kernel_trait_static_assert_errors::permute_input_dimension_2_must_match_output_dimension_1,
			input_01_dims[2], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == output_dims[3]), kernel_trait_static_assert_errors::permute_input_dimension_3_must_match_output_dimension_3,
			input_01_dims[3], output_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::reshape, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static constexpr auto input_total	= input_01_dims[0] * input_01_dims[1] * input_01_dims[2] * input_01_dims[3];
		static constexpr auto output_total	= output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer_val<(input_total == output_total), kernel_trait_static_assert_errors::reshape_input_total_elements_must_match_output_total_elements,
			input_total, output_total>::impl);
		static_assert(static_assert_printer_val<(input_total == output_total), kernel_trait_static_assert_errors::reshape_dimension_product_must_be_preserved, input_total,
			output_total>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new, typename input_03_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::rope, output_type_new, input_01_type_new, input_02_type_new, input_03_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using input_03_type					= input_03_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto input_03_dims = input_03_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::rope_input_dimension_0_must_match_output_dimension_0,
			input_01_dims[0], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[1]), kernel_trait_static_assert_errors::rope_input_dimension_1_must_match_output_dimension_1,
			input_01_dims[1], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == output_dims[2]), kernel_trait_static_assert_errors::rope_input_dimension_2_must_match_output_dimension_2,
			input_01_dims[2], output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == output_dims[3]), kernel_trait_static_assert_errors::rope_input_dimension_3_must_match_output_dimension_3,
			input_01_dims[3], output_dims[3]>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[0] == model_traits_type<config_type>::rope_dimension_count / 2),
			kernel_trait_static_assert_errors::rope_freq_dimension_0_must_be_half_of_rope_dimension_count, input_03_dims[0],
			model_traits_type<config_type>::rope_dimension_count / 2>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[1] == 1), kernel_trait_static_assert_errors::rope_freq_dimension_1_must_be_one, input_03_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[2] == 1), kernel_trait_static_assert_errors::rope_freq_dimension_2_must_be_one, input_03_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[3] == 1), kernel_trait_static_assert_errors::rope_freq_dimension_3_must_be_one, input_03_dims[3]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[0] == 1 || input_02_dims[0] == input_01_dims[2]),
			kernel_trait_static_assert_errors::rope_position_dimension_0_must_equal_one_or_match_sequence_length, input_02_dims[0], input_01_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[1] == 1), kernel_trait_static_assert_errors::rope_position_dimension_1_must_be_one, input_02_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[2] == 1), kernel_trait_static_assert_errors::rope_position_dimension_2_must_be_one, input_02_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[3] == 1), kernel_trait_static_assert_errors::rope_position_dimension_3_must_be_one, input_02_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::get_rows, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_02_dims[0] == output_dims[1]), kernel_trait_static_assert_errors::get_rows_indices_dimension_0_must_match_output_dimension_1,
			input_02_dims[0], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[1] == 1), kernel_trait_static_assert_errors::get_rows_indices_dimension_1_must_be_one, input_02_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[2] == 1), kernel_trait_static_assert_errors::get_rows_indices_dimension_2_must_be_one, input_02_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[3] == 1), kernel_trait_static_assert_errors::get_rows_indices_dimension_3_must_be_one, input_02_dims[3]>::impl);
		static_assert(static_assert_printer_val<(output_dims[0] == input_01_dims[0]),
			kernel_trait_static_assert_errors::get_rows_output_dimension_0_must_match_embedding_dimension_0, output_dims[0], input_01_dims[0]>::impl);
		static_assert(static_assert_printer_val<(output_dims[2] == 1), kernel_trait_static_assert_errors::get_rows_output_dimension_2_must_be_one, output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(output_dims[3] == 1), kernel_trait_static_assert_errors::get_rows_output_dimension_3_must_be_one, output_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::cont, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static constexpr auto input_total	= input_01_dims[0] * input_01_dims[1] * input_01_dims[2] * input_01_dims[3];
		static constexpr auto output_total	= output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
		static_assert(static_assert_printer_val<(input_total == output_total), kernel_trait_static_assert_errors::reshape_input_total_elements_must_match_output_total_elements,
			input_total, output_total>::impl);
		static_assert(static_assert_printer_val<(input_total == output_total), kernel_trait_static_assert_errors::reshape_dimension_product_must_be_preserved, input_total,
			output_total>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::rms_norm, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::rms_norm_single_input_dimension_0_must_match_output,
			input_01_dims[0], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[1]), kernel_trait_static_assert_errors::rms_norm_single_input_dimension_1_must_match_output,
			input_01_dims[1], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == output_dims[2]), kernel_trait_static_assert_errors::rms_norm_single_input_dimension_2_must_match_output,
			input_01_dims[2], output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == output_dims[3]), kernel_trait_static_assert_errors::rms_norm_single_input_dimension_3_must_match_output,
			input_01_dims[3], output_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::softmax, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::softmax_input_dimension_0_must_match_output,
			input_01_dims[0], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[1]), kernel_trait_static_assert_errors::softmax_input_dimension_1_must_match_output,
			input_01_dims[1], output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[2] == output_dims[2]), kernel_trait_static_assert_errors::softmax_input_dimension_2_must_match_output,
			input_01_dims[2], output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[3] == output_dims[3]), kernel_trait_static_assert_errors::softmax_input_dimension_3_must_match_output,
			input_01_dims[3], output_dims[3]>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new, typename input_03_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::repetition_penalty, output_type_new, input_01_type_new, input_02_type_new, input_03_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using input_03_type					= input_03_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto input_03_dims = input_03_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::repetition_penalty_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new, typename input_03_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::presence_penalty, output_type_new, input_01_type_new, input_02_type_new, input_03_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using input_03_type					= input_03_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto input_03_dims = input_03_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::presence_penalty_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new, typename input_03_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::frequency_penalty, output_type_new, input_01_type_new, input_02_type_new, input_03_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using input_03_type					= input_03_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto input_03_dims = input_03_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::frequency_penalty_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::temperature_scale, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::temperature_scale_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::top_k_filter, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::top_k_filter_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::top_p_filter, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::top_p_filter_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::vocab_mask, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using input_02_type					= input_02_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto input_02_dims = input_02_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == input_02_dims[0]), kernel_trait_static_assert_errors::vocab_mask_dimension_mismatch>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[0] == output_dims[0]), kernel_trait_static_assert_errors::vocab_mask_logits_dim_mismatch>::impl);
	};

	template<typename config_type, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config_type, dims_type_new, kernel_types::sample_logits, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
		using input_01_type				  = input_01_type_new;
		using input_02_type				  = input_02_type_new;
		using output_type				  = output_type_new;
		using dims_type					  = dims_type_new;
		static constexpr auto output_dims = dims_type::get_array();
		static_assert(static_assert_printer_val<(output_dims[0] == 1 && output_dims[1] == 1 && output_dims[2] == 1 && output_dims[3] == 1),
			kernel_trait_static_assert_errors::sample_output_must_be_single_token>::impl);
	};

	enum class get_new_dims_new_errors { unknown_kernel_type, not_allowed_for_automatic_generation };

	template<kernel_types kernel_type, typename dims_01_type> struct get_new_dims_new_1;

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type> struct get_new_dims_new_2;

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type, typename dims_03_type> struct get_new_dims_new_3;

	template<kernel_types kernel_type, typename dims_01_type> struct get_new_dims_new_1 {
		static constexpr auto dims01 = dims_01_type::get_array();
		static constexpr auto get_new_dims_new_fn() {
			if constexpr (kernel_type == kernel_types::silu) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::rms_norm) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::cont) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::silu) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else {
				if constexpr (kernel_type == kernel_types::reshape || kernel_type == kernel_types::transpose || kernel_type == kernel_types::permute ||
					kernel_type == kernel_types::view || kernel_type == kernel_types::cont) {
					static_assert(static_assert_printer_val<false, get_new_dims_new_errors::not_allowed_for_automatic_generation, kernel_type>::impl);
				} else {
					static_assert(static_assert_printer_val<false, get_new_dims_new_errors::unknown_kernel_type, kernel_type>::impl);
				}
			}
		}
		using type = decltype(get_new_dims_new_fn());
	};

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type> struct get_new_dims_new_2 {
		static constexpr auto dims01 = dims_01_type::get_array();
		static constexpr auto dims02 = dims_02_type::get_array();
		static constexpr auto get_new_dims_new_fn() {
			if constexpr (kernel_type == kernel_types::get_rows) {
				return core_trait_dims<dims01[0], dims02[0], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims02[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::copy) {
				return core_trait_dims<dims02[0], dims02[1], dims02[2], dims02[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::temperature_scale) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::top_k_filter) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::top_p_filter) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::vocab_mask) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::sample_logits) {
				return core_trait_dims<1, 1, 1, 1, 0>{};
			} else if constexpr (kernel_type == kernel_types::weights) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_new_errors::unknown_kernel_type, dims_01_type, dims_02_type>::impl);
			}
		}
		using type = decltype(get_new_dims_new_fn());
	};

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type, typename dims_03_type> struct get_new_dims_new_3 {
		static constexpr auto dims01 = dims_01_type::get_array();
		static constexpr auto dims02 = dims_02_type::get_array();
		static constexpr auto dims03 = dims_03_type::get_array();
		static constexpr auto get_new_dims_new_fn() {
			if constexpr (kernel_type == kernel_types::rope) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::repetition_penalty) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::presence_penalty) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::frequency_penalty) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::weights) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_new_errors::unknown_kernel_type, dims_01_type, dims_02_type, dims_03_type>::impl);
			}
		}
		using type = decltype(get_new_dims_new_fn());
	};

	template<kernel_types kernel_type, typename dims_01_type> using get_new_dims_new_1_t = typename get_new_dims_new_1<kernel_type, dims_01_type>::type;

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type> using get_new_dims_new_2_t =
		typename get_new_dims_new_2<kernel_type, dims_01_type, dims_02_type>::type;

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type, typename dims_03_type> using get_new_dims_new_3_t =
		typename get_new_dims_new_3<kernel_type, dims_01_type, dims_02_type, dims_03_type>::type;

	template<typename config_type, core_types core_type> struct core_traits {};

	template<typename config_type, composite_kernel_types kernel_type_new, typename output_type_new, typename... input_kernel_traits_types> struct composite_kernel_traits
		: public std::tuple_element_t<sizeof...(input_kernel_traits_types) - 1, std::tuple<input_kernel_traits_types...>>::dims_type {
		using output_type									= output_type_new;
		static constexpr composite_kernel_types kernel_type = kernel_type_new;

		using dims_type = typename std::tuple_element_t<sizeof...(input_kernel_traits_types) - 1, std::tuple<input_kernel_traits_types...>>::dims_type;

		using input_types_tuple = std::tuple<input_kernel_traits_types...>;

		template<uint64_t N> using input_type = std::conditional_t<(N < sizeof...(input_kernel_traits_types)), std::tuple_element_t<N, input_types_tuple>, void>;

		static constexpr uint64_t input_count = sizeof...(input_kernel_traits_types);
	};

}
