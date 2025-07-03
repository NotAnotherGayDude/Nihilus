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

#include <type_traits>
#include <concepts>

namespace nihilus {

	template<typename value_type>
	concept uint_type = std::is_unsigned_v<std::remove_cvref_t<value_type>> && std::is_integral_v<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept int_type = std::is_signed_v<std::remove_cvref_t<value_type>> && std::is_integral_v<std::remove_cvref_t<value_type>> && !uint_type<value_type>;

	template<typename value_type>
	concept int8_type = int_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 1;

	template<typename value_type>
	concept int16_type = int_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 2;

	template<typename value_type>
	concept int32_type = int_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept int64_type = int_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept uint8_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 1;

	template<typename value_type>
	concept uint16_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 2;

	template<typename value_type>
	concept uint32_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept uint64_type = uint_type<std::remove_cvref_t<value_type>> && sizeof(std::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept float_type = std::floating_point<std::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept float32_type = float_type<value_type> && sizeof(std::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept float64_type = float_type<value_type> && sizeof(std::remove_cvref_t<value_type>) == 8;

	template<typename value_type>
	concept has_size = requires(std::remove_cvref_t<value_type> value) {
		{ value.size() } -> std::same_as<typename value_type::uint64_type>;
	};

	template<typename value_type>
	concept has_data = requires(std::remove_cvref_t<value_type> value) {
		{ value.data() } -> std::same_as<typename value_type::pointer>;
	};

	template<typename value_type>
	concept has_find = requires(std::remove_cvref_t<value_type> value) {
		{ value.find(std::declval<typename std::remove_cvref_t<value_type>::value_type>()) } -> std::same_as<typename value_type::size_type>;
	};

	template<typename value_type>
	concept vector_subscriptable = requires(std::remove_cvref_t<value_type> value) {
		{ value[std::declval<typename value_type::uint64_type>()] } -> std::same_as<typename value_type::reference>;
	};

	template<typename value_type>
	concept string_type = vector_subscriptable<value_type> && has_data<value_type> && has_size<value_type> && has_find<value_type>;

	template<typename value_type>
	concept array_type = vector_subscriptable<value_type> && has_data<value_type> && has_size<value_type>;

	template<typename value_type>
	concept has_array_data_type = requires(std::remove_cvref_t<value_type>) { array_type<decltype(value_type::data)>; };

	template<typename value_type>
	concept core_traits_type = requires(std::remove_cvref_t<value_type>) {
		typename value_type::output_type;
		value_type::data;
	};

	template<typename value_type>
	concept core_traits_type_new = requires(std::remove_cvref_t<value_type>) {
		typename value_type::output_type;
		value_type::data;
		value_type::dims;
	};

	template<typename value_type>
	concept blocking = requires(std::remove_cvref_t<value_type> value) {
		std::remove_cvref_t<value_type>::sync_flag_start;
		std::remove_cvref_t<value_type>::sync_flag_end;
	};

	template<typename value_type>
	concept no_input = requires(std::remove_cvref_t<value_type>) { typename std::remove_cvref_t<value_type>::output_type; };

	template<typename value_type>
	concept single_input = requires(std::remove_cvref_t<value_type>) { typename std::remove_cvref_t<value_type>::input_type01; } && no_input<value_type>;

	template<typename value_type>
	concept double_input = requires(std::remove_cvref_t<value_type>) { typename std::remove_cvref_t<value_type>::input_type02; } && single_input<value_type>;

	template<typename value_type>
	concept triple_input = requires(std::remove_cvref_t<value_type>) { typename std::remove_cvref_t<value_type>::input_type03; } && double_input<value_type>;

	template<typename value_type>
	concept single_input_blocking = single_input<value_type> && blocking<value_type>;

	template<typename value_type>
	concept double_input_blocking = double_input<value_type> && blocking<value_type>;

	template<typename value_type>
	concept triple_input_blocking = triple_input<value_type> && blocking<value_type>;

	template<typename value_type>
	concept active_thread = single_input<value_type> || double_input<value_type> || triple_input<value_type> || single_input_blocking<value_type> ||
		double_input_blocking<value_type> || triple_input_blocking<value_type>;

	template<typename T>
	concept is_arithmetic_type = std::is_arithmetic_v<T>;

	template<typename T>
	concept is_quantized_type = requires {
		sizeof(T);
		!std::is_arithmetic_v<T>;
	};

	template<typename T>
	concept is_fp_type = std::is_floating_point_v<T>;

	template<typename T>
	concept is_integral_type = std::is_integral_v<T>;

	template<typename T>
	concept is_valid_tensor_type = is_arithmetic_type<T> || is_quantized_type<T>;

	template<typename T>
	concept is_valid_weight_type = is_arithmetic_type<T> || is_quantized_type<T>;

	template<typename T>
	concept is_valid_activation_type = is_fp_type<T> || is_quantized_type<T>;

	template<typename value_type>
	concept integral_or_enum = std::integral<value_type> || std::is_enum_v<value_type>;

	// from
	// https://stackoverflow.com/questions/16337610/how-to-know-if-a-type-is-a-specialization-of-stdvector
	template<typename, template<typename...> typename> constexpr bool is_specialization_v = false;

	template<template<typename...> typename value_type, typename... arg_types> constexpr bool is_specialization_v<value_type<arg_types...>, value_type> = true;

}
