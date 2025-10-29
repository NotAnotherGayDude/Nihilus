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
#include <nihilus-incl/db/file_io.hpp>
#include <nihilus-incl/common/array.hpp>
#include <nihilus-incl/infra/model_traits.hpp>
#include <nihilus-incl/common/type_traits.hpp>
#include <latch>

namespace nihilus {

	enum class processing_phases {
		prompt_eval_time,
		eval_time,
	};

	template<typename config_type> using model_traits_type = model_traits<config_type::model_arch, config_type::model_size, config_type::model_generation>;

	template<typename config_type_new, data_strategy_types data_strategy_type, typename output_type> struct data_mixin;

	template<typename config_type_new, core_types> struct core_traits;

	template<typename config_type_new, typename output_type_new> struct data_mixin<config_type_new, data_strategy_types::global, output_type_new> {
		using output_type = output_type_new;
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };

		template<typename output_type_newer = output_type> NIHILUS_HOST output_type_newer* get_data() {
			return static_cast<output_type_newer*>(data);
		}

		NIHILUS_HOST void** get_data_ptr() {
			return &data;
		}

		NIHILUS_HOST void set_data(void* data_new) {
			data = data_new;
		}

	  protected:
		void* data{};
	};

	template<typename config_type_new, typename output_type_new> struct data_mixin<config_type_new, data_strategy_types::per_block, output_type_new> {
		using output_type = output_type_new;
		static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };

		template<typename output_type_newer = output_type> NIHILUS_HOST output_type_newer* get_data(uint64_t index) {
			return static_cast<output_type_newer*>(data[index]);
		}

		NIHILUS_HOST void** get_data_ptr(uint64_t index) {
			return &data[index];
		}

		NIHILUS_HOST void set_data(void* data_new, uint64_t index) {
			data[index] = data_new;
		}

	  protected:
		array<void*, model_traits_type<config_type_new>::block_count> data{};
	};

	template<typename config_type_new, core_types core_type> struct sync_base {};

	template<typename config_type_new, core_types core_type>
		requires(config_type_new::device_type == device_types::cpu && core_type != core_types::token_embeddings && core_type != core_types::final_norm_and_sampling)
	struct sync_base<config_type_new, core_type> {
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_type_new>::block_count> current_chunk_prompt_eval{};
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_type_new>::block_count> current_chunk_eval{};
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_type_new>::block_count> latch_prompt_eval{};
		array<atomic_flag_wrapper<int64_t>, model_traits_type<config_type_new>::block_count> latch_eval{};
	};

	template<typename config_type_new, core_types core_type>
		requires(config_type_new::device_type == device_types::cpu && (core_type == core_types::token_embeddings || core_type == core_types::final_norm_and_sampling))
	struct sync_base<config_type_new, core_type> {
		atomic_flag_wrapper<int64_t> current_chunk_prompt_eval{};
		atomic_flag_wrapper<int64_t> current_chunk_eval{};
		atomic_flag_wrapper<int64_t> latch_prompt_eval{};
		atomic_flag_wrapper<int64_t> latch_eval{};
	};

	template<integral_or_enum_types auto index, typename derived_type_new> struct core_elem_base {
		using derived_type = derived_type_new;
		mutable uint64_t total_required_bytes_rt{};
		NIHILUS_HOST constexpr decltype(auto) operator[](tag<index>) & noexcept {
			return *static_cast<derived_type*>(this);
		}
	};

	template<typename value_type> constexpr uint64_t popcount(value_type value) noexcept {
		uint64_t count = 0;
		value &= 0xF;
		while (value != 0) {
			value &= (value - 1);
			count++;
		}
		return count;
	}

	constexpr bool has_at_most_two_bits_set(uint64_t value) noexcept {
		return popcount(value & 0xF) <= 2;
	}

	enum class dim_trait_static_assert_errors : uint8_t {
		reshape_total_element_count_mismatch,
		view_total_element_count_mismatch,
		transpose_total_element_count_mismatch,
		transpose_dimension_0_mismatch,
		transpose_dimension_1_mismatch,
		permute_total_element_count_mismatch,
		cont_total_element_count_mismatch
	};

	template<uint64_t... runtime_mask> constexpr uint64_t get_runtime_mask() {
		constexpr uint64_t result = ((1ULL << runtime_mask) | ...);
		static_assert(((runtime_mask < 4) && ...), "Sorry, but you can only define a dimension within the first 4 dimensions as runtime mutable!");
		static_assert(has_at_most_two_bits_set(result), "Sorry, but you can only define one or two of the first 4 dimensions as runtime mutable!");
		return result;
	}

	template<kernel_types kernel_type_new> struct kernel_types_type {
		static constexpr kernel_types kernel_type{ kernel_type_new };
	};

	template<typename value_type>
	concept preserved_dimensions_kernel_types = detail::remove_cvref_t<value_type>::kernel_type == kernel_types::add ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::mul || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::sub ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::rms_norm || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::silu ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::softmax || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::rope ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::copy || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::top_k_filter ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::top_p_filter || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::repetition_penalty ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::presence_penalty || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::frequency_penalty ||
		detail::remove_cvref_t<value_type>::kernel_type == kernel_types::temperature_scale || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::vocab_mask;

	template<typename value_type>
	concept kernel_types_types = requires() { detail::remove_cvref_t<value_type>::kernel_type; };

	template<typename value_type>
	concept weights_kernel_types = detail::remove_cvref_t<value_type>::kernel_type == kernel_types::weights && kernel_types_types<value_type>;

	template<integral_or_enum_types auto index> struct tag_new : public detail::integral_constant<uint64_t, static_cast<uint64_t>(index)> {};

	enum class incorrect_runtime_dims {
		incorrect_runtime_dim,
	};

	template<uint64_t runtime_mask_new, uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> struct kernel_dims {
		static constexpr array<uint64_t, 4> dims{ dim_00, dim_01, dim_02, dim_03 };
		static_assert(has_at_most_two_bits_set(runtime_mask_new), "Sorry, but you can only define one or two of the first 4 dimensions as runtime mutable!");
		static constexpr uint64_t runtime_mask{ runtime_mask_new & 0xF };
		mutable uint64_t rt_dims[4]{ dims[0], dims[1], dims[2], dims[3] };

		static constexpr auto get_array() {
			return array<uint64_t, 4>{ dims[0], dims[1], dims[2], dims[3] };
		}

		NIHILUS_HOST auto get_array_rt() {
			return array<uint64_t, 4>{ rt_dims[0], rt_dims[1], rt_dims[2], rt_dims[3] };
		}

		template<typename index_tag> NIHILUS_HOST uint64_t& get_dims(index_tag index) {
			static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
			static_assert(static_assert_printer_val<((runtime_mask & (1ULL << index)) != 0), incorrect_runtime_dims::incorrect_runtime_dim, index>::impl);
			return rt_dims[index];
		}

		template<typename index_tag> NIHILUS_HOST uint64_t get_dims(index_tag index) const {
			static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
			static_assert(static_assert_printer_val<((runtime_mask & (1ULL << index)) != 0), incorrect_runtime_dims::incorrect_runtime_dim, index>::impl);
			return rt_dims[index];
		}
	};

	static constexpr uint64_t compute_elements(const auto& elems) {
		uint64_t return_value{ 1 };
		for (uint64_t x = 0; x < elems.size(); ++x) {
			return_value *= elems[x];
		}
		return return_value;
	}

	template<typename value_type>
	concept kernel_dims_types = requires() {
		detail::remove_cvref_t<value_type>::runtime_mask;
		detail::remove_cvref_t<value_type>::rt_dims;
		detail::remove_cvref_t<value_type>::dims;
	};

	template<typename value_type, typename... value_types> struct get_first_type {
		using type = value_type;
	};

	template<typename... value_types> using get_first_type_t = get_first_type<value_types...>::type;

	template<bool batched, typename kernel_type, typename... dims_types> struct dim_traits;

	template<bool batched, kernel_dims_types input_dims_01> struct dim_traits<batched, kernel_types_type<kernel_types::weights>, input_dims_01> {
		using dims_type = kernel_dims<input_dims_01::runtime_mask, input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>;
	};

	template<bool batched, preserved_dimensions_kernel_types kernel_type, kernel_dims_types... dims_types> struct dim_traits<batched, kernel_type, dims_types...> {
		using first_type = get_first_type_t<dims_types...>;
		using dims_type	 = kernel_dims<first_type::runtime_mask, first_type::dims[0], first_type::dims[1], first_type::dims[2], first_type::dims[3]>;
	};

	template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims>
	struct dim_traits<batched, kernel_types_type<kernel_types::reshape>, output_dims, input_dims> {
		static constexpr auto dims01			= input_dims::dims;
		static constexpr auto dims02			= output_dims::dims;
		static constexpr size_t input_elements	= compute_elements(dims01);
		static constexpr size_t output_elements = compute_elements(dims02);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
			dim_trait_static_assert_errors::reshape_total_element_count_mismatch, input_elements, output_elements>::impl);
		using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
	};

	template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims> struct dim_traits<batched, kernel_types_type<kernel_types::view>, output_dims, input_dims> {
		static constexpr auto dims01			= input_dims::dims;
		static constexpr auto dims02			= output_dims::dims;
		static constexpr size_t input_elements	= compute_elements(dims01);
		static constexpr size_t output_elements = compute_elements(dims02);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
			dim_trait_static_assert_errors::view_total_element_count_mismatch, input_elements, output_elements>::impl);
		using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
	};

	template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims>
	struct dim_traits<batched, kernel_types_type<kernel_types::transpose>, output_dims, input_dims> {
		static constexpr auto dims01			= input_dims::dims;
		static constexpr auto dims02			= output_dims::dims;
		static constexpr size_t input_elements	= compute_elements(dims01);
		static constexpr size_t output_elements = compute_elements(dims02);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
			dim_trait_static_assert_errors::transpose_total_element_count_mismatch, input_elements, output_elements>::impl);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || dims01[0] == dims02[1]),
			dim_trait_static_assert_errors::transpose_dimension_0_mismatch, dims01[0], dims02[1]>::impl);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || dims01[1] == dims02[0]),
			dim_trait_static_assert_errors::transpose_dimension_1_mismatch, dims01[1], dims02[0]>::impl);
		using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
	};

	template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims>
	struct dim_traits<batched, kernel_types_type<kernel_types::permute>, output_dims, input_dims> {
		static constexpr auto dims01			= input_dims::dims;
		static constexpr auto dims02			= output_dims::dims;
		static constexpr size_t input_elements	= compute_elements(dims01);
		static constexpr size_t output_elements = compute_elements(dims02);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
			dim_trait_static_assert_errors::permute_total_element_count_mismatch, input_elements, output_elements>::impl);
		using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
	};

	template<bool batched, kernel_dims_types input_dims_01, kernel_dims_types input_dims_02>
	struct dim_traits<batched, kernel_types_type<kernel_types::mul_mat>, input_dims_01, input_dims_02> {
		static constexpr auto dims01 = input_dims_01::dims;
		static constexpr auto dims02 = input_dims_02::dims;
		using dims_type				 = kernel_dims<input_dims_02::runtime_mask, batched ? dims02[0] : dims01[1], batched ? dims01[1] : dims02[1], dims02[2], dims01[3]>;
	};

	template<bool batched, kernel_dims_types input_dims_01, kernel_dims_types input_dims_02>
	struct dim_traits<batched, kernel_types_type<kernel_types::get_rows>, input_dims_01, input_dims_02> {
		static constexpr auto dims01 = input_dims_01::dims;
		static constexpr auto dims02 = input_dims_02::dims;
		using dims_type = kernel_dims<(batched ? 5 : input_dims_02::runtime_mask), batched ? dims02[0] : dims01[0], dims01[0], batched ? dims02[1] : dims02[0], dims01[3]>;
	};

	template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims> struct dim_traits<batched, kernel_types_type<kernel_types::cont>, output_dims, input_dims> {
		static constexpr auto dims01			= input_dims::dims;
		static constexpr auto dims02			= output_dims::dims;
		static constexpr size_t input_elements	= compute_elements(dims01);
		static constexpr size_t output_elements = compute_elements(dims02);
		static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
			dim_trait_static_assert_errors::cont_total_element_count_mismatch, input_elements, output_elements>::impl);
		using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
	};

	template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims_01, kernel_dims_types input_dims_02>
	struct dim_traits<batched, kernel_types_type<kernel_types::sample_logits>, output_dims, input_dims_01, input_dims_02> {
		using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
	};

	template<typename value_type>
	concept kernel_traits_types = requires() {
		detail::remove_cvref_t<value_type>::kernel_type;
		typename detail::remove_cvref_t<value_type>::output_type;
	};

	template<bool batched, kernel_types_types kernel_type_new, typename... input_types_new> struct kernel_traits;

	template<bool batched, weights_kernel_types kernel_type_new, typename output_type_new, kernel_dims_types dims_type_new>
	struct kernel_traits<batched, kernel_type_new, output_type_new, dims_type_new> : public dim_traits<batched, kernel_type_new, dims_type_new>::dims_type {
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		using output_type = output_type_new;
		static constexpr bool raw_kernel_type{ false };
	};

	template<bool batched, kernel_types_types kernel_type_new, typename output_type_new, kernel_traits_types input_type_01>
	struct kernel_traits<batched, kernel_type_new, output_type_new, input_type_01> : public dim_traits<batched, kernel_type_new, input_type_01>::dims_type {
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		using output_type = output_type_new;
		static constexpr bool raw_kernel_type{ false };
	};

	template<bool batched, kernel_types_types kernel_type_new, typename output_type_new, typename input_type_01, kernel_traits_types input_type_02>
	struct kernel_traits<batched, kernel_type_new, output_type_new, input_type_01, input_type_02>
		: public dim_traits<batched, kernel_type_new, input_type_01, input_type_02>::dims_type {
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		using output_type = output_type_new;
		static constexpr bool raw_kernel_type{ false };
	};

	template<bool batched, kernel_types_types kernel_type_new, typename output_type_new, typename input_type_01, kernel_traits_types input_type_02,
		kernel_traits_types input_type_03>
	struct kernel_traits<batched, kernel_type_new, output_type_new, input_type_01, input_type_02, input_type_03>
		: public dim_traits<batched, kernel_type_new, input_type_01, input_type_02, input_type_03>::dims_type {
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		using output_type = output_type_new;
		static constexpr bool raw_kernel_type{ false };
	};

	template<typename value_type>
	concept one_input_kernel_traits_types = kernel_traits_types<value_type> &&
		(detail::remove_cvref_t<value_type>::kernel_type == kernel_types::rms_norm || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::reshape ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::transpose || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::permute ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::view || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::silu ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::cont || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::weights ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::count || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::copy);

	template<typename value_type>
	concept two_input_kernel_traits_types = kernel_traits_types<value_type> &&
		(detail::remove_cvref_t<value_type>::kernel_type == kernel_types::get_rows || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::mul ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::mul_mat || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::add ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::sub || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::softmax ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::top_k_filter || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::top_p_filter ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::repetition_penalty ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::presence_penalty ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::frequency_penalty ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::temperature_scale || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::vocab_mask ||
			detail::remove_cvref_t<value_type>::kernel_type == kernel_types::sample_logits);

	template<typename value_type>
	concept three_input_kernel_traits_types = kernel_traits_types<value_type> && detail::remove_cvref_t<value_type>::kernel_type == kernel_types::rope;

	template<typename value_type>
	concept not_kernel_traits_types = !kernel_traits_types<value_type>;

	template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
		enum_types auto enum_value_new, kernel_types_types kernel_type_new, typename output_type_new, typename... sub_kernel_types>
	struct op_traits;

	template<uint64_t required_bytes, uint64_t block_count, data_strategy_types data_strategy_type> constexpr uint64_t get_total_required_bytes{ [] {
		if constexpr (data_strategy_type == data_strategy_types::per_block) {
			return required_bytes * block_count;
		} else {
			return required_bytes;
		}
	}() };

	template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
		enum_types auto enum_value_new, kernel_types_types kernel_type_new, typename output_type_new, kernel_traits_types... sub_kernel_types>
	struct op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, sub_kernel_types...>
		: data_mixin<config_type_new, data_strategy_type, output_type_new>,
		  get_last_tuple_type<tuple<sub_kernel_types...>>,
		  core_elem_base<enum_value_new,
			  op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, sub_kernel_types...>> {
		static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		static constexpr core_types core_type{ core_type_new };
		using enum_type		   = decltype(enum_value_new);
		using output_type	   = output_type_new;
		using last_kernel_type = get_last_tuple_type<tuple<sub_kernel_types...>>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(last_kernel_type::get_array())),
			model_traits_type<config_type_new>::block_count, data_strategy_type> };
	};

	template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
		enum_types auto enum_value_new, kernel_types_types kernel_type_new, typename output_type_new, not_kernel_traits_types input_type_01,
		kernel_traits_types... sub_kernel_types>
	struct op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, input_type_01,
		sub_kernel_types...>
		: data_mixin<config_type_new, data_strategy_type, output_type_new>,
		  get_last_tuple_type<tuple<sub_kernel_types...>>,
		  core_elem_base<enum_value_new,
			  op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, sub_kernel_types...>> {
		static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		static constexpr core_types core_type{ core_type_new };
		using enum_type		   = decltype(enum_value_new);
		using output_type	   = output_type_new;
		using last_kernel_type = get_last_tuple_type<tuple<sub_kernel_types...>>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(last_kernel_type::get_array())),
			model_traits_type<config_type_new>::block_count, data_strategy_type> };
	};

	template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
		enum_types auto enum_value_new, kernel_types_types kernel_type_new, typename output_type_new, not_kernel_traits_types input_type_01, not_kernel_traits_types input_type_02,
		kernel_traits_types... sub_kernel_types>
	struct op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, input_type_01, input_type_02,
		sub_kernel_types...>
		: data_mixin<config_type_new, data_strategy_type, output_type_new>,
		  get_last_tuple_type<tuple<sub_kernel_types...>>,
		  core_elem_base<enum_value_new,
			  op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, sub_kernel_types...>> {
		static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		static constexpr core_types core_type{ core_type_new };
		using enum_type		   = decltype(enum_value_new);
		using output_type	   = output_type_new;
		using last_kernel_type = get_last_tuple_type<tuple<sub_kernel_types...>>;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(last_kernel_type::get_array())),
			model_traits_type<config_type_new>::block_count, data_strategy_type> };
	};

	template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
		enum_types auto enum_value_new, kernel_types_types kernel_type_new, typename output_type_new, not_kernel_traits_types input_type_01, not_kernel_traits_types input_type_02,
		not_kernel_traits_types input_type_03, kernel_traits_types... sub_kernel_types>
	struct op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, input_type_01, input_type_02,
		input_type_03, sub_kernel_types...>
		: data_mixin<config_type_new, data_strategy_type, output_type_new>,
		  get_last_tuple_type<tuple<sub_kernel_types...>>,
		  core_elem_base<enum_value_new,
			  op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, sub_kernel_types...>> {
		static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		static constexpr core_types core_type{ core_type_new };
		using enum_type		   = decltype(enum_value_new);
		using last_kernel_type = get_last_tuple_type<tuple<sub_kernel_types...>>;
		using output_type	   = output_type_new;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(last_kernel_type::get_array())),
			model_traits_type<config_type_new>::block_count, data_strategy_type> };
	};

	template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
		enum_types auto enum_value_new, weights_kernel_types kernel_type_new, typename output_type_new, kernel_dims_types dims_type>
	struct op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, dims_type>
		: data_mixin<config_type_new, data_strategy_type, output_type_new>,
		  dims_type,
		  core_elem_base<enum_value_new,
			  op_traits<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, kernel_type_new, output_type_new, dims_type>> {
		static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
		static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
		static constexpr core_types core_type{ core_type_new };
		using enum_type	  = decltype(enum_value_new);
		using output_type = output_type_new;
		static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::get_array())),
			model_traits_type<config_type_new>::block_count, data_strategy_type> };
	};

	template<typename config_type, typename core_traits_type, device_types devive_type, uint64_t, core_types core_type, processing_phases processing_phase>
	struct kernel_dispatcher_impl;

	template<typename value_type>
	concept runtime_dims_types = requires() { detail::remove_cvref_t<value_type>::runtime_dimension_count; } && (detail::remove_cvref_t<value_type>::runtime_dimension_count >= 1);

	template<typename value_type>
	concept double_runtime_dims_types =
		requires() { detail::remove_cvref_t<value_type>::runtime_dimension_count; } && (detail::remove_cvref_t<value_type>::runtime_dimension_count == 2);

}
