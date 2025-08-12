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
	};

	enum class processing_phase {
		prompt_eval_time,
		eval_time,
	};

	template<model_config config_new, op_types op_type> struct op_traits;

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

	template<uint64_t dim00_new, uint64_t dim01_new, uint64_t dim02_new, uint64_t dim03_new> struct core_trait_dims<dim00_new, dim01_new, dim02_new, dim03_new, 5> {
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

	template<kernel_types kernel_type, typename core_traits01, typename... operand_types> struct get_new_dims_1;

	template<kernel_types kernel_type, typename core_traits01, typename core_traits02, typename... operand_types> struct get_new_dims_2;

	template<kernel_types kernel_type, typename core_traits01, typename core_traits02, typename core_traits03, typename... operand_types> struct get_new_dims_3;

	template<kernel_types kernel_type, core_traits_types core_traits01, runtime_dims_t... dimension_type> struct get_new_dims_1<kernel_type, core_traits01, dimension_type...> {
		using dim_traits01			 = typename core_traits01::core_traits_dims_type;
		static constexpr auto dims01 = dim_traits01::get_array();

		static constexpr auto get_new_dims_fn() {
			if constexpr (kernel_type == kernel_types::silu) {
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
			if constexpr (kernel_type == kernel_types::silu) {
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
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims02[2], dims02[3]>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3]>{};
			} else if constexpr (kernel_type == kernel_types::view) {
				return core_trait_dims<dims02[0], dims02[1], dims02[2], dims02[3]>{};
			} else if constexpr (kernel_type == kernel_types::none) {
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
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims02[2], dims02[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::view) {
				return core_trait_dims<dims02[0], dims02[1], dims02[2], dims02[3], (dimension_type::dimension, ...)>{};
			} else if constexpr (kernel_type == kernel_types::none) {
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
			} else if constexpr (kernel_type == kernel_types::none) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], (dimension_type::dimension, ...)>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_errors::unknown_kernel_type, core_traits01, core_traits02, core_traits03, dimension_type...>::impl);
			}
		}

		using type = decltype(get_new_dims_fn());
	};

	template<model_config config, kernel_types kernel_type, op_types op_type, typename... dimensions_type> using get_new_dims_1_t =
		typename get_new_dims_1<kernel_type, op_traits<config, op_type>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, typename... dimensions_type> using get_new_dims_2_t =
		typename get_new_dims_2<kernel_type, op_traits<config, op_type01>, op_traits<config, op_type02>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, op_types op_type03, typename... dimensions_type> using get_new_dims_3_t =
		typename get_new_dims_3<kernel_type, op_traits<config, op_type01>, op_traits<config, op_type02>, op_traits<config, op_type03>, dimensions_type...>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type> using get_new_dims_1_2_t = typename get_new_dims_1<kernel_type, op_traits<config, op_type>>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02> using get_new_dims_2_2_t =
		typename get_new_dims_2<kernel_type, op_traits<config, op_type01>, op_traits<config, op_type02>>::type;

	template<model_config config, kernel_types kernel_type, op_types op_type01, op_types op_type02, op_types op_type03> using get_new_dims_3_2_t =
		typename get_new_dims_3<kernel_type, op_traits<config, op_type01>, op_traits<config, op_type02>, op_traits<config, op_type03>>::type;

	template<model_config config_new> using model_traits_type = model_traits<config_new.arch, config_new.model_size, config_new.model_generation>;

	template<model_config config, typename dims_type, kernel_types kernel_type, typename output_type_new, typename... operand_types> struct kernel_traits;

	template<model_config config, typename dims_type_new, typename output_type_new> struct kernel_traits<config, dims_type_new, kernel_types::none, output_type_new>
		: public dims_type_new {
		using output_type = output_type_new;
		using dims_type	  = dims_type_new;
	};

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::mul_mat, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::add, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::mul, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::rms_norm, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::copy, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::silu, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::transpose, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::view, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::logit_sample, output_type_new, input_01_type_new> : public dims_type_new {
		using input_01_type					= input_01_type_new;
		using output_type					= output_type_new;
		using dims_type						= dims_type_new;
		static constexpr auto input_01_dims = input_01_type::get_array();
		static constexpr auto output_dims	= dims_type::get_array();
		static_assert(static_assert_printer_val<(input_01_dims[0] == model_traits_type<config>::vocab_size),
			kernel_trait_static_assert_errors::logit_sample_input_dimension_0_must_be_vocab_size, input_01_dims[0], model_traits_type<config>::vocab_size>::impl);
		static_assert(static_assert_printer_val<(input_01_dims[1] == output_dims[0]),
			kernel_trait_static_assert_errors::logit_sample_input_dimension_1_must_match_output_dimension_0, input_01_dims[1], output_dims[0]>::impl);
		static_assert(static_assert_printer_val<(output_dims[1] == 1), kernel_trait_static_assert_errors::logit_sample_output_dimension_1_must_be_one, output_dims[1]>::impl);
		static_assert(static_assert_printer_val<(output_dims[2] == 1), kernel_trait_static_assert_errors::logit_sample_output_dimension_2_must_be_one, output_dims[2]>::impl);
		static_assert(static_assert_printer_val<(output_dims[3] == 1), kernel_trait_static_assert_errors::logit_sample_output_dimension_3_must_be_one, output_dims[3]>::impl);
	};

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::permute, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::reshape, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new, typename input_03_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::rope, output_type_new, input_01_type_new, input_02_type_new, input_03_type_new> : public dims_type_new {
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
		static_assert(static_assert_printer_val<(input_03_dims[0] == model_traits_type<config>::rope_dimension_count / 2),
			kernel_trait_static_assert_errors::rope_freq_dimension_0_must_be_half_of_rope_dimension_count, input_03_dims[0],
			model_traits_type<config>::rope_dimension_count / 2>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[1] == 1), kernel_trait_static_assert_errors::rope_freq_dimension_1_must_be_one, input_03_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[2] == 1), kernel_trait_static_assert_errors::rope_freq_dimension_2_must_be_one, input_03_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_03_dims[3] == 1), kernel_trait_static_assert_errors::rope_freq_dimension_3_must_be_one, input_03_dims[3]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[0] == 1 || input_02_dims[0] == input_01_dims[2]),
			kernel_trait_static_assert_errors::rope_position_dimension_0_must_equal_one_or_match_sequence_length, input_02_dims[0], input_01_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[1] == 1), kernel_trait_static_assert_errors::rope_position_dimension_1_must_be_one, input_02_dims[1]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[2] == 1), kernel_trait_static_assert_errors::rope_position_dimension_2_must_be_one, input_02_dims[2]>::impl);
		static_assert(static_assert_printer_val<(input_02_dims[3] == 1), kernel_trait_static_assert_errors::rope_position_dimension_3_must_be_one, input_02_dims[3]>::impl);
	};

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::get_rows, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::cont, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::rms_norm, output_type_new, input_01_type_new> : public dims_type_new {
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

	template<model_config config, typename dims_type_new, typename output_type_new, typename input_01_type_new, typename input_02_type_new>
	struct kernel_traits<config, dims_type_new, kernel_types::softmax, output_type_new, input_01_type_new, input_02_type_new> : public dims_type_new {
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
				return core_trait_dims<dims01[0], dims02[0], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::mul_mat) {
				return core_trait_dims<dims01[1], dims02[1], dims02[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::sub) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::copy) {
				return core_trait_dims<dims02[0], dims02[1], dims02[2], dims02[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::add) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::mul) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::softmax) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_02_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::none) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_new_errors::unknown_kernel_type, dims_01_type, dims_02_type>::impl);
			}
		}

		using type = decltype(get_new_dims_new_fn());
	};

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type, typename dims_03_type> struct get_new_dims_new_3 {
		static constexpr auto dims01 = dims_01_type::get_array();
		static constexpr auto dims02 = dims_02_type::get_array();
		static constexpr auto dims03 = dims_02_type::get_array();

		static constexpr auto get_new_dims_new_fn() {
			if constexpr (kernel_type == kernel_types::rope) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else if constexpr (kernel_type == kernel_types::none) {
				return core_trait_dims<dims01[0], dims01[1], dims01[2], dims01[3], dims_01_type::runtime_dims>{};
			} else {
				static_assert(static_assert_printer<false, get_new_dims_new_errors::unknown_kernel_type, dims_01_type, dims_02_type, dims_03_type>::impl);
			}
		}

		using type = decltype(get_new_dims_new_fn());
	};

	template<kernel_types kernel_type, typename dims_01_type> using get_new_dims_new_1_t = typename get_new_dims_new_1<kernel_type, dims_01_type>::type;

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type> using get_new_dims_new_2_t = typename get_new_dims_new_2<kernel_type, dims_01_type, dims_02_type>::type;

	template<kernel_types kernel_type, typename dims_01_type, typename dims_02_type, typename dims_03_type> using get_new_dims_new_3_t = typename get_new_dims_new_3<kernel_type, dims_01_type, dims_02_type, dims_03_type>::type;

	template<model_config config, core_types core_type> struct core_traits {};

	template<model_config config, composite_kernel_types composite_kernel_type, typename output_type_new, typename... input_kernel_traits_types> struct composite_kernel_traits {};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new> : public input_type_01_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using dims_type						   = typename input_type_01_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new>
		: public input_type_02_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using dims_type						   = typename input_type_02_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new, typename input_type_03_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new,
		input_type_03_kernel_traits_type_new> : public input_type_03_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using input_type_03_kernel_traits_type = input_type_03_kernel_traits_type_new;
		using dims_type						   = typename input_type_03_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new, typename input_type_03_kernel_traits_type_new, typename input_type_04_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new,
		input_type_03_kernel_traits_type_new, input_type_04_kernel_traits_type_new> : public input_type_04_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using input_type_03_kernel_traits_type = input_type_03_kernel_traits_type_new;
		using input_type_04_kernel_traits_type = input_type_04_kernel_traits_type_new;
		using dims_type						   = typename input_type_04_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new, typename input_type_03_kernel_traits_type_new, typename input_type_04_kernel_traits_type_new,
		typename input_type_05_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new,
		input_type_03_kernel_traits_type_new, input_type_04_kernel_traits_type_new, input_type_05_kernel_traits_type_new> : public input_type_05_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using input_type_03_kernel_traits_type = input_type_03_kernel_traits_type_new;
		using input_type_04_kernel_traits_type = input_type_04_kernel_traits_type_new;
		using input_type_05_kernel_traits_type = input_type_05_kernel_traits_type_new;
		using dims_type						   = typename input_type_05_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new, typename input_type_03_kernel_traits_type_new, typename input_type_04_kernel_traits_type_new,
		typename input_type_05_kernel_traits_type_new, typename input_type_06_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new,
		input_type_03_kernel_traits_type_new, input_type_04_kernel_traits_type_new, input_type_05_kernel_traits_type_new, input_type_06_kernel_traits_type_new>
		: public input_type_06_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using input_type_03_kernel_traits_type = input_type_03_kernel_traits_type_new;
		using input_type_04_kernel_traits_type = input_type_04_kernel_traits_type_new;
		using input_type_05_kernel_traits_type = input_type_05_kernel_traits_type_new;
		using input_type_06_kernel_traits_type = input_type_06_kernel_traits_type_new;
		using dims_type						   = typename input_type_06_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new, typename input_type_03_kernel_traits_type_new, typename input_type_04_kernel_traits_type_new,
		typename input_type_05_kernel_traits_type_new, typename input_type_06_kernel_traits_type_new, typename input_type_07_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new,
		input_type_03_kernel_traits_type_new, input_type_04_kernel_traits_type_new, input_type_05_kernel_traits_type_new, input_type_06_kernel_traits_type_new,
		input_type_07_kernel_traits_type_new> : public input_type_07_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using input_type_03_kernel_traits_type = input_type_03_kernel_traits_type_new;
		using input_type_04_kernel_traits_type = input_type_04_kernel_traits_type_new;
		using input_type_05_kernel_traits_type = input_type_05_kernel_traits_type_new;
		using input_type_06_kernel_traits_type = input_type_06_kernel_traits_type_new;
		using input_type_07_kernel_traits_type = input_type_07_kernel_traits_type_new;
		using dims_type						   = typename input_type_07_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

	template<model_config config, composite_kernel_types kernel_type_new, typename output_type_new, typename input_type_01_kernel_traits_type_new,
		typename input_type_02_kernel_traits_type_new, typename input_type_03_kernel_traits_type_new, typename input_type_04_kernel_traits_type_new,
		typename input_type_05_kernel_traits_type_new, typename input_type_06_kernel_traits_type_new, typename input_type_07_kernel_traits_type_new,
		typename input_type_08_kernel_traits_type_new>
	struct composite_kernel_traits<config, kernel_type_new, output_type_new, input_type_01_kernel_traits_type_new, input_type_02_kernel_traits_type_new,
		input_type_03_kernel_traits_type_new, input_type_04_kernel_traits_type_new, input_type_05_kernel_traits_type_new, input_type_06_kernel_traits_type_new,
		input_type_07_kernel_traits_type_new, input_type_08_kernel_traits_type_new> : public input_type_08_kernel_traits_type_new::dims_type {
		using output_type					   = output_type_new;
		using input_type_01_kernel_traits_type = input_type_01_kernel_traits_type_new;
		using input_type_02_kernel_traits_type = input_type_02_kernel_traits_type_new;
		using input_type_03_kernel_traits_type = input_type_03_kernel_traits_type_new;
		using input_type_04_kernel_traits_type = input_type_04_kernel_traits_type_new;
		using input_type_05_kernel_traits_type = input_type_05_kernel_traits_type_new;
		using input_type_06_kernel_traits_type = input_type_06_kernel_traits_type_new;
		using input_type_07_kernel_traits_type = input_type_07_kernel_traits_type_new;
		using input_type_08_kernel_traits_type = input_type_08_kernel_traits_type_new;
		using dims_type						   = typename input_type_08_kernel_traits_type::dims_type;
		static constexpr composite_kernel_types kernel_type{ kernel_type_new };
	};

}
