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

#include <nihilus-incl/common/data_types.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/array.hpp>

namespace nihilus {

	NIHILUS_DEVICE constexpr uint64_t count_elements(const array<uint64_t, 4>& dims) {
		return dims[0] * dims[1] * dims[2] * dims[3];
	}

	template<typename derived_type> struct type_traits_base {
		NIHILUS_HOST static constexpr uint64_t total_byte_size(const array<uint64_t, 4>& dims_new) {
			uint64_t element_count{ count_elements(dims_new) };
			if constexpr (derived_type::block_size == 1) {
				return element_count * derived_type::type_size;
			} else {
				return (element_count + derived_type::block_size - 1) / derived_type::block_size * derived_type::type_size;
			}
		}
	};

	struct type_traits_dynamic {
		uint64_t block_size{};
		uint64_t type_size{};
		bool is_quantized{};
		uint64_t n_rows{};
		data_types data_type{};
	};

	template<typename data_types> struct type_traits;

	template<typename derived_type> struct get_dynamic_type_traits {
		NIHILUS_HOST_DEVICE consteval static type_traits_dynamic get_dynamic_type_traits_impl() {
			type_traits_dynamic return_values{};
			return_values.block_size   = derived_type::block_size;
			return_values.is_quantized = derived_type::is_quantized;
			return_values.n_rows	   = derived_type::n_rows;
			return_values.data_type	   = derived_type::data_type;
			return_values.type_size	   = derived_type::type_size;
			return return_values;
		}
	};

	template<integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<detail::remove_cvref_t<value_type_new>>>,
																				  public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types data_type{ data_types::i8 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<detail::remove_cvref_t<value_type_new>>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types data_type{ data_types::i16 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<detail::remove_cvref_t<value_type_new>>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types data_type{ data_types::i32 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<detail::remove_cvref_t<value_type_new>>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types data_type{ data_types::i64 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<float> : public type_traits_base<type_traits<float>>, public get_dynamic_type_traits<type_traits<float>> {
		using value_type = float;
		using quant_type = float;
		inline static constexpr data_types data_type{ data_types::f32 };
		inline static constexpr uint64_t type_size{ sizeof(float) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<bf16_t> : public type_traits_base<type_traits<bf16_t>>, public get_dynamic_type_traits<type_traits<bf16_t>> {
		using value_type = bf16_t;
		using quant_type = bf16_t;
		inline static constexpr data_types data_type{ data_types::bf16 };
		inline static constexpr uint64_t type_size{ sizeof(bf16_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

#if NIHILUS_COMPILER_CUDA
	template<> struct type_traits<fp16_t> : public type_traits_base<type_traits<fp16_t>>, public get_dynamic_type_traits<type_traits<fp16_t>> {
		using value_type = fp16_t;
		using quant_type = fp16_t;
		inline static constexpr data_types data_type{ data_types::f16 };
		inline static constexpr uint64_t type_size{ sizeof(fp16_t) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};
#endif

	template<> struct type_traits<double> : public type_traits_base<type_traits<double>>, public get_dynamic_type_traits<type_traits<double>> {
		using value_type = double;
		using quant_type = double;
		inline static constexpr data_types data_type{ data_types::f64 };
		inline static constexpr uint64_t type_size{ sizeof(double) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<block_q8_0<half>> : public type_traits_base<type_traits<block_q8_0<half>>>, public get_dynamic_type_traits<type_traits<block_q8_0<half>>> {
		using value_type = block_q8_0<half>;
		using quant_type = block_q8_0<half>;
		inline static constexpr data_types data_type{ data_types::q8_0 };
		inline static constexpr uint64_t type_size{ sizeof(block_q8_0<half>) };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ Q_SIZE };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<void> : public type_traits_base<type_traits<void>> {
		inline static constexpr data_types data_type{ data_types::count };
		inline static constexpr uint64_t type_size{ 0 };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ 0 };
		inline static constexpr uint64_t n_rows{ 0 };
	};

	template<typename enum_type> NIHILUS_HOST_DEVICE constexpr type_traits_dynamic get_type_traits(enum_type data_type) {
		switch (static_cast<uint64_t>(data_type)) {
			case static_cast<uint64_t>(enum_type::f64): {
				return type_traits<double>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::f32): {
				return type_traits<float>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::f16): {
				return type_traits<int16_t>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::q8_0): {
				return type_traits<block_q8_0<half>>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::i64): {
				return type_traits<int64_t>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::i32): {
				return type_traits<int32_t>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::i16): {
				return type_traits<int16_t>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::i8): {
				return type_traits<int8_t>::get_dynamic_type_traits_impl();
			}
			case static_cast<uint64_t>(enum_type::bf16): {
				return type_traits<bf16_t>::get_dynamic_type_traits_impl();
			}
			default: {
				return {};
			}
		}
	}

}
