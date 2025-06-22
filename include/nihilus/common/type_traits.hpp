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

#include <nihilus/common/data_types.hpp>
#include <nihilus/common/common.hpp>
#include <nihilus/common/array.hpp>

namespace nihilus {

	struct type_traits_dynamic {
		uint64_t block_size{};
		uint64_t type_size{};
		bool is_quantized{};
		uint64_t n_rows{};
		data_type type{};
	};

	template<typename data_type> struct type_traits;

	template<typename derived_type> struct total_bytes_size {
		NIHILUS_FORCE_INLINE constexpr static uint64_t total_byte_size(const array<uint64_t, 4>& dims) {
			uint64_t total_elements = dims[0] * dims[1] * dims[2] * dims[3];
			uint64_t num_blocks		= (total_elements + derived_type::block_size - 1) / derived_type::block_size;
			return num_blocks * derived_type::type_size;
		}
	};

	template<typename derived_type> struct get_strides {
		NIHILUS_FORCE_INLINE constexpr static array<size_t, 4> impl(const array<uint64_t, 4>& dims) {
			array<size_t, 4> return_values{};
			return_values[0] = derived_type::type_size;
			return_values[1] = return_values[0] * (dims[0] / derived_type::block_size);
			for (int i = 2; i < 4; i++) {
				return_values[i] = return_values[i - 1] * dims[i - 1];
			}
			return return_values;
		}
	};

	template<typename derived_type> struct get_dynamic_type_traits {
		NIHILUS_FORCE_INLINE constexpr static type_traits_dynamic get_dynamic_type_traits_impl() {
			type_traits_dynamic return_values{};
			return_values.block_size   = derived_type::block_size;
			return_values.is_quantized = derived_type::is_quantized;
			return_values.n_rows	   = derived_type::n_rows;
			return_values.type		   = derived_type::type;
			return_values.type_size	   = derived_type::type_size;
			return return_values;
		}
	};

	template<> struct type_traits<int8_t>
		: public total_bytes_size<type_traits<int8_t>>, public get_strides<type_traits<int8_t>>, public get_dynamic_type_traits<type_traits<int8_t>> {
		using value_type = int8_t;
		using quant_type = int8_t;
		static constexpr data_type type{ data_type::i8 };
		static constexpr uint64_t type_size{ sizeof(int8_t) };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 1 };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<int32_t>
		: public total_bytes_size<type_traits<int32_t>>, public get_strides<type_traits<int32_t>>, public get_dynamic_type_traits<type_traits<int32_t>> {
		using value_type = int32_t;
		using quant_type = int32_t;
		static constexpr data_type type{ data_type::i32 };
		static constexpr uint64_t type_size{ sizeof(int32_t) };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 1 };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<int64_t>
		: public total_bytes_size<type_traits<int64_t>>, public get_strides<type_traits<int64_t>>, public get_dynamic_type_traits<type_traits<int64_t>> {
		using value_type = int64_t;
		using quant_type = int64_t;
		static constexpr data_type type{ data_type::i64 };
		static constexpr uint64_t type_size{ sizeof(int64_t) };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 1 };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<float> : public total_bytes_size<type_traits<float>>, public get_strides<type_traits<float>>, public get_dynamic_type_traits<type_traits<float>> {
		using value_type = float;
		using quant_type = float;
		static constexpr data_type type{ data_type::f32 };
		static constexpr uint64_t type_size{ sizeof(float) };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 1 };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<double>
		: public total_bytes_size<type_traits<double>>, public get_strides<type_traits<double>>, public get_dynamic_type_traits<type_traits<double>> {
		using value_type = double;
		using quant_type = double;
		static constexpr data_type type{ data_type::f32 };
		static constexpr uint64_t type_size{ sizeof(double) };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 1 };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<int16_t>
		: public total_bytes_size<type_traits<int16_t>>, public get_strides<type_traits<int16_t>>, public get_dynamic_type_traits<type_traits<int16_t>> {
		using value_type = fp16_t;
		using quant_type = fp16_t;
		static constexpr data_type type{ data_type::f16 };
		static constexpr uint64_t type_size{ sizeof(fp16_t) };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 1 };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<block_q8_0<half>>
		: public total_bytes_size<type_traits<block_q8_0<half>>>, public get_strides<type_traits<block_q8_0<half>>>, public get_dynamic_type_traits<type_traits<block_q8_0<half>>> {
		using value_type = block_q8_0<half>;
		using quant_type = block_q8_0<half>;
		static constexpr data_type type{ data_type::q8_0 };
		static constexpr uint64_t type_size{ sizeof(block_q8_0<half>) };
		static constexpr bool is_quantized{ true };
		static constexpr uint64_t block_size{ Q_SIZE };
		static constexpr uint64_t n_rows{ 1 };
	};

	template<> struct type_traits<void> : public total_bytes_size<type_traits<void>>, public get_strides<type_traits<void>> {
		static constexpr data_type type{ data_type::count };
		static constexpr uint64_t type_size{ 0 };
		static constexpr bool is_quantized{ false };
		static constexpr uint64_t block_size{ 0 };
		static constexpr uint64_t n_rows{ 0 };
	};

	NIHILUS_FORCE_INLINE type_traits_dynamic get_type_traits(data_type type) {
		switch (type) {
			case data_type::f64: {
				return type_traits<double>::get_dynamic_type_traits_impl();
			}
			case data_type::f32: {
				return type_traits<float>::get_dynamic_type_traits_impl();
			}
			case data_type::f16: {
				return type_traits<int16_t>::get_dynamic_type_traits_impl();
			}
			case data_type::q8_0: {
				return type_traits<block_q8_0<half>>::get_dynamic_type_traits_impl();
			}
			case data_type::i64: {
				return type_traits<int64_t>::get_dynamic_type_traits_impl();
			}
			case data_type::i32: {
				return type_traits<int32_t>::get_dynamic_type_traits_impl();
			}
			case data_type::i16: {
				return type_traits<int16_t>::get_dynamic_type_traits_impl();
			}
			case data_type::i8: {
				return type_traits<int8_t>::get_dynamic_type_traits_impl();
			}
			case data_type::count: {
				return {};
			}
		}
	}

}
