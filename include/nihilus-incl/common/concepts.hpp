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

#include <nihilus-incl/common/utility.hpp>
#include <type_traits>
#include <concepts>

namespace nihilus {

	template<typename value_type>
	concept integral_types = std::is_integral_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept uint_types = std::is_unsigned_v<detail::remove_cvref_t<value_type>> && std::is_integral_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept int_types = std::is_signed_v<detail::remove_cvref_t<value_type>> && std::is_integral_v<detail::remove_cvref_t<value_type>> && !uint_types<value_type>;

	template<typename value_type>
	concept int8_types = int_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 1;

	template<typename value_type>
	concept int16_types = int_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 2;

	template<typename value_type>
	concept int32_types = int_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept int64_types = int_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept integral8_types = integral_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 1;

	template<typename value_type>
	concept integral16_types = integral_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 2;

	template<typename value_type>
	concept integral32_types = integral_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept integral64_types = integral_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept uint8_types = uint_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 1;

	template<typename value_type>
	concept uint16_types = uint_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 2;

	template<typename value_type>
	concept uint32_types = uint_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept uint64_types = uint_types<detail::remove_cvref_t<value_type>> && sizeof(detail::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept float_types = std::floating_point<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept float32_types = float_types<value_type> && sizeof(detail::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept float64_types = float_types<value_type> && sizeof(detail::remove_cvref_t<value_type>) == 8;

	template<typename value_type> using base_type = std::remove_cvref_t<value_type>;

	template<typename value_type> using x_type = decltype(base_type<value_type>::x);

	template<typename value_type>
	concept uint_cuda_types = std::is_unsigned_v<x_type<value_type>> && std::is_integral_v<x_type<value_type>>;

	template<typename value_type>
	concept int_cuda_types = std::is_signed_v<x_type<value_type>> && std::is_integral_v<x_type<value_type>> && !uint_cuda_types<value_type>;

	template<typename value_type>
	concept int8_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

	template<typename value_type>
	concept int16_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

	template<typename value_type>
	concept int32_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

	template<typename value_type>
	concept int64_cuda_types = int_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

	template<typename value_type>
	concept uint8_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

	template<typename value_type>
	concept uint16_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

	template<typename value_type>
	concept uint32_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

	template<typename value_type>
	concept uint64_cuda_types = uint_cuda_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

	template<typename value_type>
	concept float_cuda_types = std::floating_point<x_type<value_type>>;

	template<typename value_type>
	concept float32_cuda_types = float_cuda_types<value_type> && sizeof(x_type<value_type>) == 4;

	template<typename value_type>
	concept float64_cuda_types = float_cuda_types<value_type> && sizeof(x_type<value_type>) == 8;

	template<typename value_type>
	concept has_size_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value.size() } -> std::same_as<typename detail::remove_cvref_t<value_type>::size_type>;
	};

	template<typename value_type>
	concept has_data_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value.data() } -> std::same_as<typename detail::remove_cvref_t<value_type>::pointer>;
	};

	template<typename value_type>
	concept has_find_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value.find(std::declval<typename detail::remove_cvref_t<value_type>::value_type>()) } -> std::same_as<typename detail::remove_cvref_t<value_type>::size_type>;
	};

	template<typename value_type>
	concept vector_subscriptable_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value[std::declval<typename detail::remove_cvref_t<value_type>::size_type>()] } -> std::same_as<typename detail::remove_cvref_t<value_type>::reference>;
	};

	template<typename value_type>
	concept string_types = vector_subscriptable_types<value_type> && has_data_types<value_type> && has_size_types<value_type> && has_find_types<value_type>;

	template<typename value_type>
	concept array_types = vector_subscriptable_types<value_type> && has_data_types<value_type> && has_size_types<value_type>;

	template<typename value_type>
	concept core_traits_types = requires(detail::remove_cvref_t<value_type> value) {
		typename detail::remove_cvref_t<value_type>::output_type;
	};

	template<typename value_type>
	concept blocking_types = requires(detail::remove_cvref_t<value_type> value) {
		detail::remove_cvref_t<value_type>::sync_flag_start;
		detail::remove_cvref_t<value_type>::sync_flag_end;
	};

	template<typename value_type>
	concept arithmetic_types = std::is_arithmetic_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept quantized_types = requires {
		sizeof(value_type);
		!std::is_arithmetic_v<value_type>;
	};

	template<typename value_type>
	concept fp_types = std::is_floating_point_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept valid_activation_types = fp_types<value_type> || quantized_types<value_type>;

	template<typename value_type>
	concept valid_tensor_types = arithmetic_types<value_type> || quantized_types<value_type>;

	template<typename value_type>
	concept integral_or_enum_types = std::integral<value_type> || std::is_enum_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept has_latch_types = requires() { detail::remove_cvref_t<value_type>::latch_eval; };

	template<typename value_type>
	concept has_total_required_bytes_types = requires() { detail::remove_cvref_t<value_type>::total_required_bytes; };

	template<typename value_type>
	concept has_chunk_types = requires() { detail::remove_cvref_t<value_type>::current_chunk_eval; };

	template<typename value_type>
	concept active_input_types = requires() { detail::remove_cvref_t<value_type>::runtime_dims; };

	template<typename value_type>
	concept has_return_type = requires() { typename detail::remove_cvref_t<value_type>::return_type; };

	template<typename value_type>
	concept is_integral_constant = requires() {
		typename detail::remove_cvref_t<value_type>::value_type;
		{ detail::remove_cvref_t<value_type>::value } -> std::same_as<typename detail::remove_cvref_t<value_type>::value_type>;
	};

	template<typename value_01_type, typename value_02_type>
	concept is_indexable =
		std::is_same_v<value_01_type, value_02_type> || std::integral<value_01_type> || is_integral_constant<value_01_type> || is_integral_constant<value_02_type>;

	// from
	// https://stackoverflow.com/questions/16337610/how-to-know-if-a-type-is-a-specialization-of-stdvector
	template<typename, template<typename...> typename> constexpr bool is_specialization_v = false;

	template<template<typename...> typename value_type, typename... arg_types> constexpr bool is_specialization_v<value_type<arg_types...>, value_type> = true;

	template<typename, template<auto...> typename> constexpr bool is_specialization_val_v = false;

	template<template<auto> typename value_type, auto... arg_types> constexpr bool is_specialization_val_v<value_type<arg_types...>, value_type> = true;

	enum class input_types : uint8_t {
		none  = 1 << 0,
		one	  = 1 << 1,
		two	  = 1 << 2,
		three = 1 << 3,
		four  = 1 << 4,
		five  = 1 << 5,
		six	  = 1 << 6,
	};

	template<typename value_type>
	concept single_input_types = detail::remove_cvref_t<value_type>::input_type == input_types::one;

	template<typename value_type>
	concept double_input_types = detail::remove_cvref_t<value_type>::input_type == input_types::two && !single_input_types<value_type>;

	template<typename value_type>
	concept triple_input_types = detail::remove_cvref_t<value_type>::input_type == input_types::three && !double_input_types<value_type> && !single_input_types<value_type>;

	template<typename value_type>
	concept quadruple_input_types = detail::remove_cvref_t<value_type>::input_type == input_types::four && !triple_input_types<value_type> && !double_input_types<value_type> &&
		!single_input_types<value_type>;

	template<typename value_type>
	concept quintuple_input_types = detail::remove_cvref_t<value_type>::input_type == input_types::five && !quadruple_input_types<value_type> && !triple_input_types<value_type> &&
		!double_input_types<value_type> && !single_input_types<value_type>;

	template<typename value_type>
	concept sextuple_input_types = detail::remove_cvref_t<value_type>::input_type == input_types::six && !quintuple_input_types<value_type> && !quadruple_input_types<value_type> &&
		!triple_input_types<value_type> && !double_input_types<value_type> && !single_input_types<value_type>;

	template<typename value_type>
	concept has_count = requires() { detail::remove_cvref_t<value_type>::count; } && static_cast<uint64_t>(detail::remove_cvref_t<value_type>::count) < 128;

	template<typename value_type>
	concept enum_types = std::is_enum_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept printable_enum_types = enum_types<value_type> && has_count<value_type>;

	template<typename value_type>
	concept pointer_types = std::is_pointer_v<value_type>;

	template<typename value_type>
	concept not_pointer_types = !std::is_pointer_v<value_type>;

	template<typename value_type>
	concept dim04_types = requires() { base_type<value_type>::w; };

	template<typename value_type>
	concept dim03_types = requires() { base_type<value_type>::z; } && !dim04_types<value_type>;

	template<typename value_type>
	concept dim02_types = requires() { base_type<value_type>::y; } && !dim03_types<value_type> && !dim04_types<value_type>;

	template<typename value_type>
	concept dim01_types = requires() { base_type<value_type>::x; } && !dim02_types<value_type> && !dim03_types<value_type> && !dim04_types<value_type>;

	template<typename value_type>
	concept dim_types = requires() { base_type<value_type>::x; };

	template<integral_or_enum_types auto index> using tag = std::integral_constant<uint64_t, static_cast<uint64_t>(index)>;

}
