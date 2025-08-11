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
#include <nihilus-incl/common/array.hpp>
#include <latch>

namespace nihilus {

	enum class kernel_trait_static_assert_errors {
		Sorry_but_these_output_types_are_not_the_same,
		Sorry_but_these_input_types01_types_are_not_the_same,
		Sorry_but_these_input_types02_types_are_not_the_same,
		Sorry_but_these_input_types03_types_are_not_the_same,
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
		ROPE_Frequency_count_must_be_rope_dimension_count_div_2,
		ROPE_Frequencies_must_be_1D,
		ROPE_rope_dimension_countension_must_be_even,
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

	enum class processing_phase {
		prompt_eval_time,
		eval_time,
	};

	template<model_config config_new, op_types op_type> struct core_traits;

	template<size_t... indices> struct core_trait_dims;

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new> {
		static constexpr uint64_t runtime_dims{ 5 };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		constexpr core_trait_dims() noexcept {
		}

		static constexpr uint64_t runtime_dim{ std::numeric_limits<uint64_t>::max() };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_INLINE operator array<uint64_t, 4>() const {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 0> {
		static constexpr uint64_t runtime_dims{ 0 };
		mutable uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00_new, dim01_new, dim02_new, dim03_new } };
		}

		NIHILUS_INLINE operator array<uint64_t, 4>() const {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_INLINE uint64_t& get_mutable_dim() const {
			return dim00;
		}

		NIHILUS_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 1> {
		static constexpr uint64_t runtime_dims{ 1 };
		static constexpr uint64_t dim00{ dim00_new };
		mutable uint64_t dim01{ dim01_new };
		static constexpr uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00_new, dim01_new, dim02_new, dim03_new } };
		}

		NIHILUS_INLINE operator array<uint64_t, 4>() const {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_INLINE uint64_t& get_mutable_dim() const {
			return dim01;
		}

		NIHILUS_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 2> {
		static constexpr uint64_t runtime_dims{ 2 };
		static constexpr uint64_t dim00{ dim00_new };
		static constexpr uint64_t dim01{ dim01_new };
		mutable uint64_t dim02{ dim02_new };
		static constexpr uint64_t dim03{ dim03_new };

		const uint64_t* dims[4]{ &dim00, &dim01, &dim02, &dim03 };

		NIHILUS_INLINE static constexpr array<uint64_t, 4> get_array() {
			return { { dim00_new, dim01_new, dim02_new, dim03_new } };
		}

		NIHILUS_INLINE operator array<uint64_t, 4>() const {
			return { { dim00, dim01, dim02, dim03 } };
		}

		NIHILUS_INLINE uint64_t& get_mutable_dim() const {
			return dim02;
		}

		NIHILUS_INLINE uint64_t operator[](uint64_t index) const {
			return *dims[index];
		}
	};

	template<uint64_t dimension_new> struct runtime_dims {
		static constexpr uint64_t dimension{ dimension_new };
	};

	template<typename value_type>
	concept runtime_dims_t = requires() { std::remove_cvref_t<value_type>::dimension; };

	enum class get_new_dims_errors { unknown_kernel_type };

	template<uint64_t, kernel_types kernel_type, processing_phase phase, typename... operand_types> struct kernel_dispatcher_impl;

	template<kernel_types kernel_type, typename core_type, typename... operand_types> struct kernel_base;

	template<kernel_types kernel_type, typename core_traits01, typename... operand_types> struct get_new_dims_1;

	template<kernel_types kernel_type, typename core_traits01, typename core_traits02, typename... operand_types> struct get_new_dims_2;

	template<kernel_types kernel_type, typename core_traits01, typename core_traits02, typename core_traits03, typename... operand_types> struct get_new_dims_3;

	template<kernel_types kernel_type, typename core_traits01, typename core_traits02, typename core_traits03, typename core_traits04, typename... operand_types>
	struct get_new_dims_4;

	template<kernel_types kernel_type, core_traits_types core_traits01, runtime_dims_t... dimension_type> struct get_new_dims_1<kernel_type, core_traits01, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::add_rms_norm) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::silu) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01> struct get_new_dims_1<kernel_type, core_traits01> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::add_rms_norm) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::silu) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02> struct get_new_dims_2<kernel_type, core_traits01, core_traits02> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::get_rows) {
				return core_trait_dims<dims01[1], dims02[0], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::rms_norm_mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims02[2], dims02[3]>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::add_rms_norm_mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::view) {
				return core_trait_dims<dims02[0], dims02[1], dims02[2], dims02[3]>{};
			}else if constexpr (kernel_type == kernel_types::mul_mat_reshape) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02, runtime_dims_t... dimension_type>
	struct get_new_dims_2<kernel_type, core_traits01, core_traits02, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::get_rows) {
				return core_trait_dims<dims01[0], dims02[0], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::rms_norm_mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims02[2], dims02[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::add_rms_norm_mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::view) {
				return core_trait_dims<dims02[0], dims02[1], dims02[2], dims02[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat_reshape) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02, core_traits_types core_traits03, runtime_dims_t... dimension_type>
	struct get_new_dims_3<kernel_type, core_traits01, core_traits02, core_traits03, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		using dim_traits03			 = typename core_traits03::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();
		static constexpr auto dims03 = dim_traits03::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::rope) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::rope_permute) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02, core_traits03, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<kernel_types kernel_type, core_traits_types core_traits01, core_traits_types core_traits02, core_traits_types core_traits03, core_traits_types core_traits04,
		runtime_dims_t... dimension_type>
	struct get_new_dims_4<kernel_type, core_traits01, core_traits02, core_traits03, core_traits04, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		using dim_traits02			 = typename core_traits02::core_traits_dims_type;
		using dim_traits03			 = typename core_traits03::core_traits_dims_type;
		using dim_traits04			 = typename core_traits04::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();
		static constexpr auto dims02 = dim_traits02::get_array();
		static constexpr auto dims03 = dim_traits03::get_array();
		static constexpr auto dims04 = dim_traits04::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::rope_copy) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02, core_traits03, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<model_config config, kernel_types kernel_type, op_types op_type, typename... dimensions_type> using get_new_dims_1_t =
		typename get_new_dims_1<kernel_type, core_traits<config, op_type>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, typename... dimensions_type> using get_new_dims_2_t =
		typename get_new_dims_2<kernel_type, core_traits<config, op_type01>, core_traits<config, op_type02>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, op_types op_type03, typename... dimensions_type> using get_new_dims_3_t =
		typename get_new_dims_3<kernel_type, core_traits<config, op_type01>, core_traits<config, op_type02>, core_traits<config, op_type03>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type> using get_new_dims_1_2_t = typename get_new_dims_1<kernel_type, core_traits<config, op_type>>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02> using get_new_dims_2_2_t =
		typename get_new_dims_2<kernel_type, core_traits<config, op_type01>, core_traits<config, op_type02>>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, op_types op_type03> using get_new_dims_3_2_t =
		typename get_new_dims_3<kernel_type, core_traits<config, op_type01>, core_traits<config, op_type02>, core_traits<config, op_type03>>::type;

	template<kernel_types kernel_type, single_input_types core_type, typename output_type, typename input_01_type>
	struct kernel_base<kernel_type, core_type, output_type, input_01_type> {
		using input01									 = input_01_type;
		using output									 = output_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_01_type::get_array();
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_01_type, typename core_type::input_01_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types01_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type>::impl);
	};

	template<kernel_types kernel_type, double_input_types core_type, typename output_type, typename input_01_type, typename input_02_type>
	struct kernel_base<kernel_type, core_type, output_type, input_01_type, input_02_type> {
		using input01									 = typename core_type::input_01_type;
		using input02									 = typename core_type::input_02_type;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_01_type::get_array();
		static constexpr auto dims03					 = core_type::input_02_type::get_array();
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_01_type, typename core_type::input_01_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types01_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_02_type, typename core_type::input_02_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types02_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type>::impl);
	};

	template<kernel_types kernel_type, triple_input_types core_type, typename output_type, typename input_01_type, typename input_02_type, typename input_03_type>
	struct kernel_base<kernel_type, core_type, output_type, input_01_type, input_02_type, input_03_type> {
		using input01									 = typename core_type::input_01_type;
		using input02									 = typename core_type::input_02_type;
		using input03									 = typename core_type::input_03_type;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_01_type::get_array();
		static constexpr auto dims03					 = core_type::input_02_type::get_array();
		static constexpr auto dims04					 = core_type::input_03_type::get_array();
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static constexpr uint64_t input03_total_elements = dims04[0] * dims04[1] * dims04[2] * dims04[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_01_type, typename core_type::input_01_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types01_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type,
			input_03_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_02_type, typename core_type::input_02_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types02_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type,
			input_03_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_03_type, typename core_type::input_03_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types03_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type,
			input_03_type>::impl);
	};

	template<kernel_types kernel_type, quadruple_input_types core_type, typename output_type, typename input_01_type, typename input_02_type, typename input_03_type,
		typename input_04_type>
	struct kernel_base<kernel_type, core_type, output_type, input_01_type, input_02_type, input_03_type, input_04_type> {
		using input01									 = typename core_type::input_01_type;
		using input02									 = typename core_type::input_02_type;
		using input03									 = typename core_type::input_03_type;
		using input04									 = typename core_type::input_04_type;
		using output									 = core_type;
		static constexpr auto dims01					 = core_type::get_array();
		static constexpr auto dims02					 = core_type::input_01_type::get_array();
		static constexpr auto dims03					 = core_type::input_02_type::get_array();
		static constexpr auto dims04					 = core_type::input_03_type::get_array();
		static constexpr auto dims05					 = core_type::input_04_type::get_array();
		static constexpr uint64_t total_elements		 = dims01[0] * dims01[1] * dims01[2] * dims01[3];
		static constexpr uint64_t input01_total_elements = dims02[0] * dims02[1] * dims02[2] * dims02[3];
		static constexpr uint64_t input02_total_elements = dims03[0] * dims03[1] * dims03[2] * dims03[3];
		static constexpr uint64_t input03_total_elements = dims04[0] * dims04[1] * dims04[2] * dims04[3];
		static constexpr uint64_t input04_total_elements = dims05[0] * dims05[1] * dims05[2] * dims05[3];
		static_assert(static_assert_printer<( std::is_same_v<output_type, typename core_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_output_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_01_type, typename core_type::input_01_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types01_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type,
			input_03_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_02_type, typename core_type::input_02_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types02_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type,
			input_03_type>::impl);
		static_assert(static_assert_printer<( std::is_same_v<input_03_type, typename core_type::input_03_type::output_type> ),
			kernel_trait_static_assert_errors::Sorry_but_these_input_types03_types_are_not_the_same, kernel_base, core_type, output_type, input_01_type, input_02_type,
			input_03_type>::impl);
	};

	template<kernel_types kernel_type, typename core_type, typename... operand_types> struct kernel_traits;

	template<kernel_types kernel_type, typename... operand_types> struct kernel_traits_new;

}