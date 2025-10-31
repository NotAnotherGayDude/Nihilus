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

#include <nihilus-incl/common/model_config.hpp>
#include <type_traits>
#include <concepts>

namespace nihilus {

	template<typename value_type>
	concept integral_types = std::is_integral_v<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept uint_types = std::is_unsigned_v<detail::remove_cvref_t<value_type>> && integral_types<detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept int_types = std::is_signed_v<detail::remove_cvref_t<value_type>> && integral_types<detail::remove_cvref_t<value_type>> && !uint_types<value_type>;

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
	concept float16_types = detail::is_same_v<detail::remove_cvref_t<value_type>, fp16_t> || detail::is_same_v<detail::remove_cvref_t<value_type>, bf16_t>;

	template<typename value_type>
	concept float32_types = float_types<value_type> && sizeof(detail::remove_cvref_t<value_type>) == 4;

	template<typename value_type>
	concept float64_types = float_types<value_type> && sizeof(detail::remove_cvref_t<value_type>) == 8;

	template<typename value_type> using base_type = detail::remove_cvref_t<value_type>;

#if NIHILUS_COMPILER_CUDA

	template<typename value_type> using x_type = decltype(base_type<value_type>::x);

	template<typename value_type>
	concept half_cuda_types = detail::is_same_v<__half, detail::remove_cvref_t<value_type>>;

	template<typename value_type>
	concept int8_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

	template<typename value_type>
	concept int16_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

	template<typename value_type>
	concept int32_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

	template<typename value_type>
	concept int64_cuda_types = int_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

	template<typename value_type>
	concept uint8_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 1;

	template<typename value_type>
	concept uint16_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 2;

	template<typename value_type>
	concept uint32_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 4;

	template<typename value_type>
	concept uint64_cuda_types = uint_types<x_type<value_type>> && sizeof(x_type<value_type>) == 8;

	template<typename value_type>
	concept float32_cuda_types = float32_types<value_type> && sizeof(x_type<value_type>) == 4;

	template<typename value_type>
	concept float64_cuda_types = float64_types<value_type> && sizeof(x_type<value_type>) == 8;

#endif

	template<typename value_type>
	concept has_size_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value.size() } -> detail::same_as<typename detail::remove_cvref_t<value_type>::size_type>;
	};

	template<typename value_type>
	concept has_data_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value.data() } -> detail::same_as<typename detail::remove_cvref_t<value_type>::pointer>;
	};

	template<typename value_type>
	concept has_find_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value.find(std::declval<typename detail::remove_cvref_t<value_type>::value_type>()) } -> detail::same_as<typename detail::remove_cvref_t<value_type>::size_type>;
	};

	template<typename value_type>
	concept vector_subscriptable_types = requires(detail::remove_cvref_t<value_type> value) {
		{ value[std::declval<typename detail::remove_cvref_t<value_type>::size_type>()] } -> detail::same_as<typename detail::remove_cvref_t<value_type>::reference>;
	};

	template<typename value_type>
	concept string_types = vector_subscriptable_types<value_type> && has_data_types<value_type> && has_size_types<value_type> && has_find_types<value_type>;

	template<typename value_type>
	concept array_types = vector_subscriptable_types<value_type> && has_data_types<value_type> && has_size_types<value_type>;

	template<typename value_type>
	concept core_traits_types = requires(detail::remove_cvref_t<value_type> value) { typename detail::remove_cvref_t<value_type>::output_type; };

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
	concept has_find = requires(detail::remove_cvref_t<value_type> value) {
		{ value.find(typename detail::remove_cvref_t<value_type>::value_type{}) } -> detail::same_as<typename detail::remove_cvref_t<value_type>::size_type>;
	} || requires(detail::remove_cvref_t<value_type> value) {
		{ value.find(typename detail::remove_cvref_t<value_type>::key_type{}) } -> detail::same_as<typename detail::remove_cvref_t<value_type>::iterator>;
	} || requires(detail::remove_cvref_t<value_type> value) {
		{ value.find(typename detail::remove_cvref_t<value_type>::key_type{}) } -> detail::same_as<typename detail::remove_cvref_t<value_type>::const_iterator>;
	};

	template<typename value_type>
	concept has_range = requires(detail::remove_cvref_t<value_type> value) {
		{ value.begin() };
		{ value.end() };
	};

	template<typename value_type>
	concept map_subscriptable = requires(detail::remove_cvref_t<value_type> value) {
		{ value[typename detail::remove_cvref_t<value_type>::key_type{}] } -> detail::same_as<const typename detail::remove_cvref_t<value_type>::mapped_type&>;
	} || requires(detail::remove_cvref_t<value_type> value) {
		{ value[typename detail::remove_cvref_t<value_type>::key_type{}] } -> detail::same_as<typename detail::remove_cvref_t<value_type>::mapped_type&>;
	};

	template<typename value_type>
	concept vector_subscriptable = requires(detail::remove_cvref_t<value_type> value) {
		{ value[typename detail::remove_cvref_t<value_type>::size_type{}] } -> detail::same_as<typename detail::remove_cvref_t<value_type>::const_reference>;
	} || requires(detail::remove_cvref_t<value_type> value) {
		{ value[typename detail::remove_cvref_t<value_type>::size_type{}] } -> detail::same_as<typename detail::remove_cvref_t<value_type>::reference>;
	};

	template<typename value_type>
	concept has_size = requires(detail::remove_cvref_t<value_type> value) {
		{ value.size() } -> detail::same_as<typename detail::remove_cvref_t<value_type>::size_type>;
	};

	template<typename value_type>
	concept has_empty = requires(detail::remove_cvref_t<value_type> value) {
		{ value.empty() } -> detail::same_as<bool>;
	};

	template<typename value_type>
	concept bool_types =
		detail::same_as<detail::remove_cvref_t<value_type>, bool> || detail::same_as<detail::remove_cvref_t<value_type>, std::vector<bool>::reference> ||
		detail::same_as<detail::remove_cvref_t<value_type>, std::vector<bool>::const_reference>;

	template<typename value_type>
	concept num_types = (float_types<value_type> || uint_types<value_type> || int_types<value_type>);

	template<typename value_type>
	concept map_types = map_subscriptable<value_type> && has_range<value_type> && has_size<value_type> && has_find<value_type> && has_empty<value_type>;

	template<typename value_type> struct db_core;

	template<typename value_type>
	concept object_types = requires { db_core<detail::remove_cvref_t<value_type>>::db_value; };

	template<typename value_type>
	concept has_emplace_back = requires(detail::remove_cvref_t<value_type> value) {
		{ value.emplace_back(typename detail::remove_cvref_t<value_type>::value_type{}) } -> detail::same_as<typename detail::remove_cvref_t<value_type>::reference>;
	};

	template<typename value_type>
	concept has_resize = requires(detail::remove_cvref_t<value_type> value) { value.resize(typename detail::remove_cvref_t<value_type>::size_type{}); };

	template<typename value_type>
	concept vector_types = vector_subscriptable<value_type> && has_resize<value_type> && has_emplace_back<value_type>;

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
	concept active_input_types = requires() { detail::remove_cvref_t<value_type>::runtime_dim; };

	template<typename value_type>
	concept has_return_type = requires() { typename detail::remove_cvref_t<value_type>::return_type; };

	template<typename value_type>
	concept is_integral_constant = requires() {
		typename detail::remove_cvref_t<value_type>::value_type;
		{ detail::remove_cvref_t<value_type>::value } -> detail::same_as<typename detail::remove_cvref_t<value_type>::value_type>;
	};

	template<typename value_type_01, typename value_type_02>
	concept is_indexable =
		detail::is_same_v<value_type_01, value_type_02> || std::integral<value_type_01> || is_integral_constant<value_type_01> || is_integral_constant<value_type_02>;

	// from
	// https://stackoverflow.com/questions/16337610/how-to-know-if-a-type-is-a-specialization-of-stdvector
	template<typename, template<typename...> typename> constexpr bool is_specialization_v = false;

	template<template<typename...> typename value_type, typename... arg_types> constexpr bool is_specialization_v<value_type<arg_types...>, value_type> = true;

	template<typename, template<auto...> typename> constexpr bool is_specialization_val_v = false;

	template<template<auto> typename value_type, auto... arg_types> constexpr bool is_specialization_val_v<value_type<arg_types...>, value_type> = true;

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

	template<integral_or_enum_types auto index> struct tag : detail::integral_constant<uint64_t, static_cast<uint64_t>(index)> {};

	template<typename value_type>
	concept managed_user_input_config_types = detail::remove_cvref_t<value_type>::user_input_type == user_input_types::managed;

	template<typename value_type>
	concept gpu_device_config_types = detail::remove_cvref_t<value_type>::device_type == device_types::gpu;

	template<typename value_type>
	concept cpu_device_config_types = detail::remove_cvref_t<value_type>::device_type == device_types::cpu;

	template<typename value_type>
	concept batched_processing_config_types = detail::remove_cvref_t<value_type>::batched_processing == true;

	template<typename value_type>
	concept llama_arch_config_types = detail::remove_cvref_t<value_type>::model_arch == model_arches::llama;

	template<typename value_type>
	concept dev_or_benchmark_config_types = detail::remove_cvref_t<value_type>::benchmark || detail::remove_cvref_t<value_type>::dev;

	template<typename value_type>
	concept weight_types =
		static_cast<uint64_t>(detail::remove_cvref_t<value_type>::tensor_type) >= 0 && static_cast<uint64_t>(detail::remove_cvref_t<value_type>::tensor_type) < 13;

	template<typename value_type>
	concept global_input_types =
		static_cast<uint64_t>(detail::remove_cvref_t<value_type>::tensor_type) >= 13 && static_cast<uint64_t>(detail::remove_cvref_t<value_type>::tensor_type) < 30;

}
