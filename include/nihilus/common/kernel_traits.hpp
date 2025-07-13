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

#include <nihilus/common/common.hpp>
#include <nihilus/common/array.hpp>
#include <latch>

namespace nihilus {

	enum class kernel_trait_static_assert_errors {
		Sorry_but_these_output_types_are_not_the_same,
		Sorry_but_these_input_type01_types_are_not_the_same,
		Sorry_but_these_input_type02_types_are_not_the_same,
		Sorry_but_these_input_type03_types_are_not_the_same,
		MUL_Input_dimensions_0_must_match,
		MUL_Output_dimensions_0_must_match_inputs,
		MUL_Broadcasting_requires_input02_1_equals_1_or_matching_dimensions_1,
		MUL_Output_dimensions_1_must_match_input01,
		MUL_Batch_dimensions_2_must_match,
		MUL_Batch_dimensions_3_must_match,
		MUL_Total_element_count_must_match_between_inputs,
		MUL_Total_element_count_must_match_between_input_and_output,
		RMS_NORM_Output_dimensions_0_must_match_input_dimensions,
		RMS_NORM_Output_dimensions_1_must_match_input_dimensions,
		RMS_NORM_Output_dimensions_2_must_match_input_dimensions,
		RMS_NORM_Output_dimensions_3_must_match_input_dimensions,
		RMS_NORM_Input_type_must_be_valid_activation_type,
		RMS_NORM_Output_type_must_be_valid_activation_type,
		RMS_NORM_Total_element_count_must_match_between_input_and_output,
		SILU_Output_dimensions_0_must_match_input_dimensions,
		SILU_Output_dimensions_1_must_match_input_dimensions,
		SILU_Output_dimensions_2_must_match_input_dimensions,
		SILU_Output_dimensions_3_must_match_input_dimensions,
		SILU_Input_type_must_be_valid_activation_type,
		SILU_Output_type_must_be_valid_activation_type,
		SOFTMAX_Output_dimensions_0_must_match_input01,
		SOFTMAX_Output_dimensions_1_must_match_input01,
		SOFTMAX_Output_dimensions_2_must_match_input01,
		SOFTMAX_Output_dimensions_3_must_match_input01,
		SOFTMAX_Mask_dimensions_0_must_match_scores,
		RESHAPE_Input_type_must_be_valid_tensor_type,
		RESHAPE_Output_type_must_be_valid_tensor_type,
		RESHAPE_Total_element_count_must_be_preserved,
		TRANSPOSE_Input_type_must_be_valid_tensor_type,
		TRANSPOSE_Output_type_must_be_valid_tensor_type,
		TRANSPOSE_Total_element_count_must_be_preserved,
		PERMUTE_Input_type_must_be_valid_tensor_type,
		PERMUTE_Output_type_must_be_valid_tensor_type,
		PERMUTE_Total_element_count_must_be_preserved,
		CONT_Input_type_must_be_valid_tensor_type,
		CONT_Output_type_must_be_valid_tensor_type,
		CONT_Total_element_count_must_match_between_input_and_output,
		VIEW_Input_type_must_be_valid_tensor_type,
		VIEW_Output_type_must_be_valid_tensor_type,
		VIEW_Output_cannot_have_more_elements_than_input,
		MUL_MAT_Weight_rows_must_match_input_vector_size,
		MUL_MAT_Output_size_must_match_weight_columns,
		MUL_MAT_Batch_dimension_2_must_match_or_support_GQA_broadcasting,
		MUL_MAT_Output_head_count_must_match_attention_head_count,
		MUL_MAT_Batch_dimension_3_must_match,
		MUL_MAT_Output_batch_dimension_3_must_match_attention_dimensions,
		MUL_MAT_Input1_type_must_be_valid_tensor_type,
		MUL_MAT_Input2_type_must_be_valid_tensor_type,
		MUL_MAT_Output_type_must_be_valid_tensor_type,
		GET_ROWS_Output_rows_must_match_number_of_indices,
		GET_ROWS_Output_sequence_length_must_match_input_token_count,
		GET_ROWS_Output_dimension_2_must_be_1,
		GET_ROWS_Output_dimension_3_must_be_1,
		GET_ROWS_Index_tensor_dimension_1_must_be_1,
		GET_ROWS_Index_tensor_dimension_2_must_be_1,
		GET_ROWS_Index_tensor_dimension_3_must_be_1,
		GET_ROWS_Embedding_matrix_type_must_be_valid_tensor_type,
		GET_ROWS_Index_type_must_be_integer_type,
		GET_ROWS_Output_type_must_be_valid_tensor_type,
		ROPE_Output_dimensions_must_match_input_tensor,
		ROPE_Sequence_length_must_match,
		ROPE_Number_of_heads_must_match,
		ROPE_Batch_dimension_must_match,
		ROPE_Position_count_must_match_sequence_length,
		ROPE_Position_indices_must_be_1D,
		ROPE_Frequency_count_must_be_head_dim_div_2,
		ROPE_Frequencies_must_be_1D,
		ROPE_Head_dimension_must_be_even,
		ADD_Input_dimensions_0_must_match,
		ADD_Input_dimensions_1_must_match,
		ADD_Input_dimensions_2_must_match,
		ADD_Input_dimensions_3_must_match,
		ADD_Output_dimensions_0_must_match_input_dimensions,
		ADD_Output_dimensions_1_must_match_input_dimensions,
		ADD_Output_dimensions_2_must_match_input_dimensions,
		ADD_Output_dimensions_3_must_match_input_dimensions,
		ADD_Input1_type_must_be_valid_tensor_type,
		ADD_Input2_type_must_be_valid_tensor_type,
		ADD_Output_type_must_be_valid_tensor_type,
		SUB_Input_dimensions_0_must_match,
		SUB_Input_dimensions_1_must_match,
		SUB_Input_dimensions_2_must_match,
		SUB_Input_dimensions_3_must_match,
		SUB_Output_dimensions_0_must_match_input_dimensions,
		SUB_Output_dimensions_1_must_match_input_dimensions,
		SUB_Output_dimensions_2_must_match_input_dimensions,
		SUB_Output_dimensions_3_must_match_input_dimensions,
		SUB_Input1_type_must_be_valid_tensor_type,
		SUB_Input2_type_must_be_valid_tensor_type,
		SUB_Output_type_must_be_valid_tensor_type,
		COPY_Source_type_must_be_valid_tensor_type,
		COPY_Destination_type_must_be_valid_tensor_type,
		COPY_Source_and_destination_must_have_same_total_element_count,
		NONE_Type_must_be_valid_tensor_type,
	};

	template<model_config config_new, op_types op_type> struct core_traits;

	template<size_t... indices> struct core_trait_dims;

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new> {
		static constexpr array<bool, 4> runtime_dims{ false, false, false, false };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		constexpr core_trait_dims() noexcept {};

		static constexpr uint64_t runtime_dim{ std::numeric_limits<uint64_t>::max() };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 0> {
		static constexpr array<bool, 4> runtime_dims{ true, false, false, false };
		uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00_new, dim01_new, dim02_new, dim03_new } };
		}

		NIHILUS_FORCE_INLINE uint64_t& get_mutable_dim() {
			return dim00;
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 1> {
		static constexpr array<bool, 4> runtime_dims{ true, false, false, false };
		static constexpr uint64_t dim00{ dim00_new };
		uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00_new, dim01_new, dim02_new, dim03_new } };
		}

		NIHILUS_FORCE_INLINE uint64_t& get_mutable_dim() {
			return dim01;
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 2> {
		static constexpr array<bool, 4> runtime_dims{ true, false, false, false };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_FORCE_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00_new, dim01_new, dim02_new, dim03_new } };
		}

		NIHILUS_FORCE_INLINE uint64_t& get_mutable_dim() {
			return dim02;
		}

		NIHILUS_FORCE_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dimension_new> struct runtime_dims {
		static constexpr uint64_t dimension{ dimension_new };
	};

	template<typename value_type>
	concept runtime_dims_t = requires() { std::remove_cvref_t<value_type>::dimension; };

	enum class get_new_dims_errors { unknown_kernel_type };

	template<uint64_t, auto op_type, typename... operand_types> struct kernel_dispatcher_impl;

	template<auto op_type, kernel_types kernel_type, typename core_type, typename... operand_types> struct kernel_base;

	template<kernel_types kernel_type, typename... operand_types> struct kernel_base_new;

	template<kernel_types kernel_type, typename... core_traits01> struct get_new_dims;

	template<model_config config, kernel_types kernel_type, op_types op_type, typename... dimensions_type> using get_new_dims_1_t =
		typename get_new_dims<kernel_type, core_traits<config, op_type>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, typename... dimensions_type> using get_new_dims_2_t =
		typename get_new_dims<kernel_type, core_traits<config, op_type01>, core_traits<config, op_type02>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, op_types op_type03, typename... dimensions_type> using get_new_dims_3_t =
		typename get_new_dims<kernel_type, core_traits<config, op_type01>, core_traits<config, op_type02>, core_traits<config, op_type03>, dimensions_type...>::type;

	template<auto op_type, kernel_types kernel_type, single_input core_type, typename output_type, typename input_type01>
	struct kernel_base<op_type, kernel_type, core_type, output_type, input_type01> {
		using input01									 = input_type01;
		using output									 = output_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_type01::get_array();
		static constexpr auto strides01					 = core_type::strides;
		static constexpr auto strides02					 = core_type::input_type01::strides;
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_type01>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_type01, typename core_type::input_type01::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_type01_types_are_not_the_same, kernel_base, core_type, output_type, input_type01>::impl);
	};

	template<kernel_types kernel_type, typename input_type01> struct kernel_base_new<kernel_type, input_type01> {
		using input01									 = input_type01;
		static constexpr auto dims01					 = input01::get_array();
		static constexpr auto strides01					 = input01::strides;
		static constexpr uint64_t input01_total_elements = dims01[0] * dims01[1] * dims01[2] * dims01[3];
	};

	template<kernel_types kernel_type, typename input_type01, typename input_type02> struct kernel_base_new<kernel_type, input_type01, input_type02> {
		using input01									 = input_type01;
		using input02									 = input_type02;
		static constexpr auto dims01					 = input01::get_array();
		static constexpr auto dims02					 = input02::get_array();
		static constexpr auto strides01					 = input01::strides;
		static constexpr auto strides02					 = input02::strides;
		static constexpr uint64_t input01_total_elements = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input02_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
	};

	template<auto op_type, kernel_types kernel_type, double_input core_type, typename output_type, typename input_type01, typename input_type02>
	struct kernel_base<op_type, kernel_type, core_type, output_type, input_type01, input_type02> {
		using input01									 = typename core_type::input_type01;
		using input02									 = typename core_type::input_type02;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_type01::get_array();
		static constexpr auto dims03					 = core_type::input_type02::get_array();
		static constexpr auto strides01					 = core_type::strides;
		static constexpr auto strides02					 = core_type::input_type01::strides;
		static constexpr auto strides03					 = core_type::input_type02::strides;
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_type01, typename core_type::input_type01::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_type01_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_type02, typename core_type::input_type02::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_type02_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02>::impl);
	};

	template<auto op_type, kernel_types kernel_type, triple_input core_type, typename output_type, typename input_type01, typename input_type02, typename input_type03>
	struct kernel_base<op_type, kernel_type, core_type, output_type, input_type01, input_type02, input_type03> {
		using input01									 = typename core_type::input_type01;
		using input02									 = typename core_type::input_type02;
		using input03									 = typename core_type::input_type03;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_type01::get_array();
		static constexpr auto dims03					 = core_type::input_type02::get_array();
		static constexpr auto dims04					 = core_type::input_type03::get_array();
		static constexpr auto strides01					 = core_type::strides;
		static constexpr auto strides02					 = core_type::input_type01::strides;
		static constexpr auto strides03					 = core_type::input_type02::strides;
		static constexpr auto strides04					 = core_type::input_type03::strides;
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static constexpr uint64_t input03_total_elements = dims04[0] * dims04[1] * dims04[2] * dims04[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_type01, typename core_type::input_type01::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_type01_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02,
			input_type03>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_type02, typename core_type::input_type02::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_type02_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02,
			input_type03>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_type03, typename core_type::input_type03::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_type03_types_are_not_the_same, kernel_base, core_type, output_type, input_type01, input_type02,
			input_type03>::impl);
	};

	template<auto op_type, kernel_types kernel_type, typename core_type, typename... operand_types> struct kernel_traits;

	template<kernel_types kernel_type, typename... operand_types> struct kernel_traits_new;

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::mul, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::mul, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::mul, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		static_assert(
			static_assert_printer<(dims02[0] == dims03[0]), kernel_trait_static_assert_errors::MUL_Input_dimensions_0_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<(dims02[0] == dims01[0]), kernel_trait_static_assert_errors::MUL_Output_dimensions_0_must_match_inputs, kernel_traits, output, input01,
			input02>::impl);
		static_assert(static_assert_printer<(dims03[1] == 1 || dims02[1] == dims03[1]),
			kernel_trait_static_assert_errors::MUL_Broadcasting_requires_input02_1_equals_1_or_matching_dimensions_1, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::MUL_Output_dimensions_1_must_match_input01, kernel_traits, output, input01,
			input02>::impl);
		static_assert(
			static_assert_printer<(dims02[2] == dims03[2]), kernel_trait_static_assert_errors::MUL_Batch_dimensions_2_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[3] == dims03[3]), kernel_trait_static_assert_errors::MUL_Batch_dimensions_3_must_match, kernel_traits, output, input01, input02>::impl);
		static constexpr bool is_broadcasting = (dims03[1] == 1 && dims02[1] > 1);
		static_assert(static_assert_printer<(base_type::total_elements == base_type::input01_total_elements),
			kernel_trait_static_assert_errors::MUL_Total_element_count_must_match_between_inputs, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer < (base_type::total_elements == base_type::input02_total_elements) || is_broadcasting,
			kernel_trait_static_assert_errors::MUL_Total_element_count_must_match_between_input_and_output, kernel_traits, output, input01, input02 > ::impl);
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::rms_norm, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::rms_norm, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::rms_norm, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01				 = base_type::input01;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static_assert(static_assert_printer<(dims01[0] == dims02[std::integral_constant<uint64_t, 0>{}]),
			kernel_trait_static_assert_errors::RMS_NORM_Output_dimensions_0_must_match_input_dimensions, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::RMS_NORM_Output_dimensions_1_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<(dims01[2] == dims02[2]), kernel_trait_static_assert_errors::RMS_NORM_Output_dimensions_2_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<(dims01[3] == dims02[3]), kernel_trait_static_assert_errors::RMS_NORM_Output_dimensions_3_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<is_valid_activation_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::RMS_NORM_Input_type_must_be_valid_activation_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_activation_type<typename core_type::output_type>,
			kernel_trait_static_assert_errors::RMS_NORM_Output_type_must_be_valid_activation_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(base_type::input01_total_elements == base_type::total_elements),
			kernel_trait_static_assert_errors::RMS_NORM_Total_element_count_must_match_between_input_and_output, kernel_traits, output, input01>::impl);
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::silu, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::silu, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::silu, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01				 = base_type::input01;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::SILU_Output_dimensions_0_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::SILU_Output_dimensions_1_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<(dims01[2] == dims02[2]), kernel_trait_static_assert_errors::SILU_Output_dimensions_2_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<(dims01[3] == dims02[3]), kernel_trait_static_assert_errors::SILU_Output_dimensions_3_must_match_input_dimensions, kernel_traits,
			output, input01>::impl);
		static_assert(static_assert_printer<is_valid_activation_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::SILU_Input_type_must_be_valid_activation_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_activation_type<typename core_type::output_type>,
			kernel_trait_static_assert_errors::SILU_Output_type_must_be_valid_activation_type, kernel_traits, output, input01>::impl);
	};

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::softmax, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::softmax, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::softmax, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03; /*
		static_assert(static_assert_printer<(dims02[0] == dims01[0]), kernel_trait_static_assert_errors::SOFTMAX_Output_dimensions_0_must_match_input01, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims02[1] == dims01[1]), kernel_trait_static_assert_errors::SOFTMAX_Output_dimensions_1_must_match_input01, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims02[2] == dims01[2]), kernel_trait_static_assert_errors::SOFTMAX_Output_dimensions_2_must_match_input01, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims02[3] == dims01[3]), kernel_trait_static_assert_errors::SOFTMAX_Output_dimensions_3_must_match_input01, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims03[0] == dims02[0]), kernel_trait_static_assert_errors::SOFTMAX_Mask_dimensions_0_must_match_scores, kernel_traits, output,
			input01, input02>::impl);*/
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::reshape, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::reshape, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type = kernel_base<op_type, kernel_types::reshape, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output;
		/*
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::RESHAPE_Input_type_must_be_valid_tensor_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::RESHAPE_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(base_type::input01_total_elements == base_type::total_elements),
			kernel_trait_static_assert_errors::RESHAPE_Total_element_count_must_be_preserved, kernel_traits, output, input01>::impl);*/
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::transpose, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::transpose, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type = kernel_base<op_type, kernel_types::transpose, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output; /*
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::TRANSPOSE_Input_type_must_be_valid_tensor_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>,
			kernel_trait_static_assert_errors::TRANSPOSE_Output_type_must_be_valid_tensor_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(base_type::input01_total_elements == base_type::total_elements),
			kernel_trait_static_assert_errors::TRANSPOSE_Total_element_count_must_be_preserved, kernel_traits, output, input01>::impl);*/
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::permute, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::permute, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type = kernel_base<op_type, kernel_types::permute, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output; /*
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::PERMUTE_Input_type_must_be_valid_tensor_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::PERMUTE_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(base_type::input01_total_elements == base_type::total_elements),
			kernel_trait_static_assert_errors::PERMUTE_Total_element_count_must_be_preserved, kernel_traits, output, input01>::impl);*/
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::cont, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::cont, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type = kernel_base<op_type, kernel_types::cont, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output; /*
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::CONT_Input_type_must_be_valid_tensor_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::CONT_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(base_type::input01_total_elements == base_type::total_elements),
			kernel_trait_static_assert_errors::CONT_Total_element_count_must_match_between_input_and_output, kernel_traits, output, input01>::impl);*/
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::view, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::view, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type = kernel_base<op_type, kernel_types::view, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01	= base_type::input01;
		using output	= base_type::output; /*
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::VIEW_Input_type_must_be_valid_tensor_type, kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::VIEW_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01>::impl);
		static_assert(static_assert_printer<(base_type::input01_total_elements >= base_type::total_elements),
			kernel_trait_static_assert_errors::VIEW_Output_cannot_have_more_elements_than_input, kernel_traits, output, input01>::impl);*/
	};

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::mul_mat, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::mul_mat, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::mul_mat, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		/*
		static_assert(static_assert_printer<(dims02[0] == dims03[0]), kernel_trait_static_assert_errors::MUL_MAT_Weight_rows_must_match_input_vector_size, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims03[2] % dims02[2] == 0), kernel_trait_static_assert_errors::MUL_MAT_Batch_dimension_2_must_match_or_support_GQA_broadcasting,
			kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<(dims03[3] % dims02[3] == 0), kernel_trait_static_assert_errors::MUL_MAT_Batch_dimension_3_must_match, kernel_traits, output, input01,
			input02>::impl);

		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::MUL_MAT_Input1_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type02::output_type>,
			kernel_trait_static_assert_errors::MUL_MAT_Input2_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::MUL_MAT_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01, input02>::impl);

		static constexpr uint64_t M						   = dims02[0];
		static constexpr uint64_t K						   = dims02[1];
		static constexpr uint64_t N						   = dims03[1];
		static constexpr uint64_t batch_size			   = dims02[2] * dims02[3];
		static constexpr uint64_t expected_output_elements = M * (dims03[1] == 1 ? 1 : N) * batch_size;*/
	};

	template<typename input_type01, typename input_type02> struct kernel_traits_new<kernel_types::get_rows, input_type01, input_type02>
		: public kernel_base_new<kernel_types::get_rows, input_type01, input_type02> {
		using base_type				 = kernel_base_new<kernel_types::get_rows, input_type01, input_type02>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		//static constexpr auto dims03{core_trait_dims<};
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::GET_ROWS_Output_rows_must_match_number_of_indices, kernel_traits_new,
			output, input01, input02>::impl);
		//static_assert(static_assert_printer<(dims01[1] == dims03[0]), kernel_trait_static_assert_errors::GET_ROWS_Output_sequence_length_must_match_input_token_count,
		//kernel_traits_new, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims01[2] == 1), kernel_trait_static_assert_errors::GET_ROWS_Output_dimension_2_must_be_1, kernel_traits_new, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims01[3] == 1), kernel_trait_static_assert_errors::GET_ROWS_Output_dimension_3_must_be_1, kernel_traits_new, output, input01, input02>::impl);
		//static_assert(
		//			static_assert_printer<(dims03[1] == 1), kernel_trait_static_assert_errors::GET_ROWS_Index_tensor_dimension_1_must_be_1, kernel_traits_new, output, input01, input02>::impl);
		//static_assert(
		//static_assert_printer<(dims03[2] == 1), kernel_trait_static_assert_errors::GET_ROWS_Index_tensor_dimension_2_must_be_1, kernel_traits_new, output, input01, input02>::impl);
		//static_assert(
		//			static_assert_printer<(dims03[3] == 1), kernel_trait_static_assert_errors::GET_ROWS_Index_tensor_dimension_3_must_be_1, kernel_traits_new, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename input_type01::output_type>,
			kernel_trait_static_assert_errors::GET_ROWS_Embedding_matrix_type_must_be_valid_tensor_type, kernel_traits_new, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_integral_type<typename input_type02::output_type>, kernel_trait_static_assert_errors::GET_ROWS_Index_type_must_be_integer_type,
			kernel_traits_new, output, input01, input02>::impl);
		//static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>,
		//kernel_trait_static_assert_errors::GET_ROWS_Output_type_must_be_valid_tensor_type, kernel_traits_new, output, input01, input02>::impl);
		static constexpr uint64_t vocab_size	   = dims02[0];
		static constexpr uint32_t embedding_length = dims02[1];
		//static constexpr uint64_t sequence_length = dims03[0];
	};

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::get_rows, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::get_rows, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::get_rows, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		/*
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::GET_ROWS_Output_rows_must_match_number_of_indices, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims03[0]), kernel_trait_static_assert_errors::GET_ROWS_Output_sequence_length_must_match_input_token_count,
			kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims01[2] == 1), kernel_trait_static_assert_errors::GET_ROWS_Output_dimension_2_must_be_1, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims01[3] == 1), kernel_trait_static_assert_errors::GET_ROWS_Output_dimension_3_must_be_1, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims03[1] == 1), kernel_trait_static_assert_errors::GET_ROWS_Index_tensor_dimension_1_must_be_1, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims03[2] == 1), kernel_trait_static_assert_errors::GET_ROWS_Index_tensor_dimension_2_must_be_1, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims03[3] == 1), kernel_trait_static_assert_errors::GET_ROWS_Index_tensor_dimension_3_must_be_1, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::GET_ROWS_Embedding_matrix_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_integral_type<typename core_type::input_type02::output_type>,
			kernel_trait_static_assert_errors::GET_ROWS_Index_type_must_be_integer_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>,
			kernel_trait_static_assert_errors::GET_ROWS_Output_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static constexpr uint64_t vocab_size	  = dims02[0];
		static constexpr uint32_t embedding_length	  = dims02[1];
		static constexpr uint64_t sequence_length = dims03[0];*/
	};

	template<auto op_type, triple_input core_type> struct kernel_traits<op_type, kernel_types::rope, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type, typename core_type::input_type03::output_type>
		: public kernel_base<op_type, kernel_types::rope, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type, typename core_type::input_type03::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::rope, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type, typename core_type::input_type03::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using input03				 = base_type::input03;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		static constexpr auto dims04 = base_type::dims04;
		/*
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::ROPE_Output_dimensions_must_match_input_tensor, kernel_traits, output,
			input01, input02, input03>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::ROPE_Sequence_length_must_match, kernel_traits, output, input01, input02,
			input03>::impl);
		static_assert(static_assert_printer<(dims01[2] == dims02[2]), kernel_trait_static_assert_errors::ROPE_Number_of_heads_must_match, kernel_traits, output, input01, input02,
			input03>::impl);
		static_assert(static_assert_printer<(dims01[3] == dims02[3]), kernel_trait_static_assert_errors::ROPE_Batch_dimension_must_match, kernel_traits, output, input01, input02,
			input03>::impl);
		static_assert(static_assert_printer<(dims03[0] == dims02[1]), kernel_trait_static_assert_errors::ROPE_Position_count_must_match_sequence_length, kernel_traits, output,
			input01, input02, input03>::impl);
		static_assert(static_assert_printer<(dims03[1] == 1 && dims03[2] == 1 && dims03[3] == 1), kernel_trait_static_assert_errors::ROPE_Position_indices_must_be_1D,
			kernel_traits, output, input01, input02, input03>::impl);
		static_assert(static_assert_printer<(dims04[0] == dims02[0] / 2), kernel_trait_static_assert_errors::ROPE_Frequency_count_must_be_head_dim_div_2, kernel_traits, output,
			input01, input02, input03>::impl);
		static_assert(static_assert_printer<(dims04[1] == 1 && dims04[2] == 1 && dims04[3] == 1), kernel_trait_static_assert_errors::ROPE_Frequencies_must_be_1D, kernel_traits,
			output, input01, input02, input03>::impl);
		static_assert(static_assert_printer<(dims02[0] % 2 == 0), kernel_trait_static_assert_errors::ROPE_Head_dimension_must_be_even, kernel_traits, output, input01, input02,
			input03>::impl);
		static constexpr uint64_t head_dim		  = dims02[0];
		static constexpr uint64_t sequence_length = dims02[1];
		static constexpr uint64_t num_heads		  = dims02[2];*/
	};

	template<auto op_type, typename core_type01, typename core_type02> struct kernel_traits<op_type, kernel_types::add, core_type01, core_type02>
		: public kernel_base<op_type, kernel_types::add, core_type01, core_type02> {};

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::add, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::add, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::add, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		static_assert(
			static_assert_printer<(dims02[0] == dims03[0]), kernel_trait_static_assert_errors::ADD_Input_dimensions_0_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[1] == dims03[1]), kernel_trait_static_assert_errors::ADD_Input_dimensions_1_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[2] == dims03[2]), kernel_trait_static_assert_errors::ADD_Input_dimensions_2_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[3] == dims03[3]), kernel_trait_static_assert_errors::ADD_Input_dimensions_3_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::ADD_Output_dimensions_0_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::ADD_Output_dimensions_1_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[2] == dims02[2]), kernel_trait_static_assert_errors::ADD_Output_dimensions_2_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[3] == dims02[3]), kernel_trait_static_assert_errors::ADD_Output_dimensions_3_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::ADD_Input1_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type02::output_type>,
			kernel_trait_static_assert_errors::ADD_Input2_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::ADD_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01, input02>::impl);
		static constexpr auto get_dims() {
			return dims01;
		}
	};

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::sub, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::sub, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::sub, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
						 typename core_type::input_type02::output_type>;
		using input01				 = base_type::input01;
		using input02				 = base_type::input02;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		/*
		static_assert(
			static_assert_printer<(dims02[0] == dims03[0]), kernel_trait_static_assert_errors::SUB_Input_dimensions_0_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[1] == dims03[1]), kernel_trait_static_assert_errors::SUB_Input_dimensions_1_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[2] == dims03[2]), kernel_trait_static_assert_errors::SUB_Input_dimensions_2_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[3] == dims03[3]), kernel_trait_static_assert_errors::SUB_Input_dimensions_3_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::SUB_Output_dimensions_0_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::SUB_Output_dimensions_1_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[2] == dims02[2]), kernel_trait_static_assert_errors::SUB_Output_dimensions_2_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[3] == dims02[3]), kernel_trait_static_assert_errors::SUB_Output_dimensions_3_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::SUB_Input1_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type02::output_type>,
			kernel_trait_static_assert_errors::SUB_Input2_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::SUB_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01, input02>::impl);*/
	};

	template<auto op_type, double_input core_type> struct kernel_traits<op_type, kernel_types::add_rms_norm_mul, core_type, typename core_type::output_type,
		typename core_type::input_type01::output_type, typename core_type::input_type02::output_type>
		: public kernel_base<op_type, kernel_types::add_rms_norm_mul, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			  typename core_type::input_type02::output_type> {
		using base_type = kernel_base<op_type, kernel_types::add_rms_norm_mul, core_type, typename core_type::output_type, typename core_type::input_type01::output_type,
			typename core_type::input_type02::output_type>;
		using input01	= base_type::input01;
		using input02	= base_type::input02;
		using output	= base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		static constexpr auto dims03 = base_type::dims03;
		/*
		static_assert(
			static_assert_printer<(dims02[0] == dims03[0]), kernel_trait_static_assert_errors::SUB_Input_dimensions_0_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[1] == dims03[1]), kernel_trait_static_assert_errors::SUB_Input_dimensions_1_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[2] == dims03[2]), kernel_trait_static_assert_errors::SUB_Input_dimensions_2_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(
			static_assert_printer<(dims02[3] == dims03[3]), kernel_trait_static_assert_errors::SUB_Input_dimensions_3_must_match, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[0] == dims02[0]), kernel_trait_static_assert_errors::SUB_Output_dimensions_0_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[1] == dims02[1]), kernel_trait_static_assert_errors::SUB_Output_dimensions_1_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[2] == dims02[2]), kernel_trait_static_assert_errors::SUB_Output_dimensions_2_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<(dims01[3] == dims02[3]), kernel_trait_static_assert_errors::SUB_Output_dimensions_3_must_match_input_dimensions, kernel_traits, output,
			input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::SUB_Input1_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type02::output_type>,
			kernel_trait_static_assert_errors::SUB_Input2_type_must_be_valid_tensor_type, kernel_traits, output, input01, input02>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>, kernel_trait_static_assert_errors::SUB_Output_type_must_be_valid_tensor_type,
			kernel_traits, output, input01, input02>::impl);*/
	};

	template<auto op_type, single_input core_type>
	struct kernel_traits<op_type, kernel_types::copy, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::copy, core_type, typename core_type::output_type, typename core_type::input_type01::output_type> {
		using base_type				 = kernel_base<op_type, kernel_types::copy, core_type, typename core_type::output_type, typename core_type::input_type01::output_type>;
		using input01				 = base_type::input01;
		using output				 = base_type::output;
		static constexpr auto dims01 = base_type::dims01;
		static constexpr auto dims02 = base_type::dims02;
		/*
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::COPY_Source_type_must_be_valid_tensor_type, kernel_traits, input01>::impl);
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::output_type>,
			kernel_trait_static_assert_errors::COPY_Destination_type_must_be_valid_tensor_type, kernel_traits, input01>::impl);
		static constexpr uint64_t source_elements = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t dest_elements	  = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static_assert(static_assert_printer<(source_elements == dest_elements), kernel_trait_static_assert_errors::COPY_Source_and_destination_must_have_same_total_element_count,
			kernel_traits, input01>::impl);*/
	};

	template<auto op_type, single_input core_type> struct kernel_traits<op_type, kernel_types::none, core_type, typename core_type::input_type01::output_type>
		: public kernel_base<op_type, kernel_types::none, core_type, typename core_type::input_type01::output_type> {
		static_assert(static_assert_printer<is_valid_tensor_type<typename core_type::input_type01::output_type>,
			kernel_trait_static_assert_errors::NONE_Type_must_be_valid_tensor_type, kernel_traits>::impl);
	};

}
