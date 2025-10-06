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
		NIHILUS_HOST static constexpr uint64_t row_size(uint64_t ne) {
			return derived_type::type_size * ne / derived_type::block_size;
		}

		NIHILUS_HOST static constexpr uint64_t total_byte_size(const array<uint64_t, 4>& dims) {
			array<uint64_t, 4> strides{};
			strides[0] = derived_type::type_size;
			strides[1] = strides[0] * (dims[0] / derived_type::block_size);
			for (int32_t i = 2; i < 4; i++) {
				strides[i] = strides[i - 1] * dims[i - 1];
			}
			uint64_t nbytes{};
			uint64_t blck_size = derived_type::block_size;
			if (blck_size == 1) {
				nbytes = derived_type::type_size;
				for (uint64_t i = 0; i < 4; ++i) {
					nbytes += (dims[i] - 1) * strides[i];
				}
			} else {
				nbytes = dims[0] * strides[0] / blck_size;
				for (uint64_t i = 1; i < 4; ++i) {
					nbytes += (dims[i] - 1) * strides[i];
				}
			}
			return nbytes;
		}

		NIHILUS_HOST constexpr static array<uint64_t, 4> get_strides(const array<uint64_t, 4>& dims) {
			array<uint64_t, 4> return_values{};
			return_values[0] = derived_type::type_size;
			return_values[1] = return_values[0] * (dims[0] / derived_type::block_size);
			for (int32_t i = 2; i < 4; i++) {
				return_values[i] = return_values[i - 1] * dims[i - 1];
			}
			return return_values;
		}
	};

	struct type_traits_dynamic {
		uint64_t block_size{};
		uint64_t type_size{};
		bool is_quantized{};
		uint64_t n_rows{};
		data_types type{};

		NIHILUS_HOST constexpr uint64_t row_size(uint64_t ne) const {
			return type_size * ne / block_size;
		}

		NIHILUS_DEVICE constexpr uint64_t total_byte_size(const array<uint64_t, 4>& dims) const {
			array<uint64_t, 4> strides{};
			strides[0] = type_size;
			strides[1] = strides[0] * (dims[0] / block_size);
			for (int32_t i = 2; i < 4; i++) {
				strides[i] = strides[i - 1] * dims[i - 1];
			}
			uint64_t nbytes{};
			uint64_t blck_size = block_size;
			if (blck_size == 1) {
				nbytes = type_size;
				for (uint64_t i = 0; i < 4; ++i) {
					nbytes += (dims[i] - 1) * strides[i];
				}
			} else {
				nbytes = dims[0] * strides[0] / blck_size;
				for (uint64_t i = 1; i < 4; ++i) {
					nbytes += (dims[i] - 1) * strides[i];
				}
			}
			return nbytes;
		}
	};

	template<typename data_types> struct type_traits;

	template<typename derived_type> struct get_dynamic_type_traits {
		NIHILUS_HOST_DEVICE consteval static type_traits_dynamic get_dynamic_type_traits_impl() {
			type_traits_dynamic return_values{};
			return_values.block_size   = derived_type::block_size;
			return_values.is_quantized = derived_type::is_quantized;
			return_values.n_rows	   = derived_type::n_rows;
			return_values.type		   = derived_type::type;
			return_values.type_size	   = derived_type::type_size;
			return return_values;
		}
	};

	template<integral8_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				  public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types type{ data_types::i8 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<integral16_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types type{ data_types::i16 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<integral32_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types type{ data_types::i32 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<integral64_types value_type_new> struct type_traits<value_type_new> : public type_traits_base<type_traits<value_type_new>>,
																				   public get_dynamic_type_traits<type_traits<value_type_new>> {
		using value_type = value_type_new;
		using quant_type = value_type;
		inline static constexpr data_types type{ data_types::i64 };
		inline static constexpr uint64_t type_size{ sizeof(value_type) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<float> : public type_traits_base<type_traits<float>>, public get_dynamic_type_traits<type_traits<float>> {
		using value_type = float;
		using quant_type = float;
		inline static constexpr data_types type{ data_types::f32 };
		inline static constexpr uint64_t type_size{ sizeof(float) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<double> : public type_traits_base<type_traits<double>>, public get_dynamic_type_traits<type_traits<double>> {
		using value_type = double;
		using quant_type = double;
		inline static constexpr data_types type{ data_types::f64 };
		inline static constexpr uint64_t type_size{ sizeof(double) };
		inline static constexpr bool is_quantized{ false };
		inline static constexpr uint64_t block_size{ 1 };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<block_q8_0<half>> : public type_traits_base<type_traits<block_q8_0<half>>>, public get_dynamic_type_traits<type_traits<block_q8_0<half>>> {
		using value_type = block_q8_0<half>;
		using quant_type = block_q8_0<half>;
		inline static constexpr data_types type{ data_types::q8_0 };
		inline static constexpr uint64_t type_size{ sizeof(block_q8_0<half>) };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ Q_SIZE };
		inline static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<void> : public type_traits_base<type_traits<void>> {
		inline static constexpr data_types type{ data_types::count };
		inline static constexpr uint64_t type_size{ 0 };
		inline static constexpr bool is_quantized{ true };
		inline static constexpr uint64_t block_size{ 0 };
		inline static constexpr uint64_t n_rows{ 0 };
	};

	template<typename value_type> NIHILUS_HOST uint64_t get_runtime_byte_size(value_type& core) {
		array<uint64_t, 4> dims{};
		dims[0] = core[0];
		dims[1] = core[1];
		dims[2] = core[2];
		dims[3] = core[3];
		if constexpr (value_type::runtime_dims != 5) {
			dims[value_type::runtime_dims] = core.get_mutable_dim();
		}
		return type_traits<typename value_type::output_type>::total_byte_size(dims);
	}

	template<typename enum_type> NIHILUS_HOST_DEVICE constexpr type_traits_dynamic get_type_traits(enum_type type) {
		switch (static_cast<uint64_t>(type)) {
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
			default: {
				return {};
			}
		}
	}

}
