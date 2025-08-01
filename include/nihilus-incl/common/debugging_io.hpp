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
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <string>

namespace nihilus {

	float f16_to_f32(uint16_t f16_bits) {
		uint32_t sign = (f16_bits & 0x8000) << 16;
		uint32_t exp  = (f16_bits & 0x7C00);
		uint32_t frac = (f16_bits & 0x03FF);

		if (exp == 0x7C00) {
			exp = 0xFF << 23;
			frac <<= 13;
		} else if (exp != 0) {
			exp = ((exp >> 10) - 15 + 127) << 23;
			frac <<= 13;
		} else if (frac != 0) {
			exp = 127 - 14;
			while ((frac & 0x0400) == 0) {
				frac <<= 1;
				exp--;
			}
			frac = (frac & 0x03FF) << 13;
			exp <<= 23;
		}

		uint32_t result = sign | exp | frac;
		return *reinterpret_cast<float*>(&result);
	}

	void print_typed_data(std::ostream& stream, const uint8_t* data, uint64_t size, data_types type, int64_t offending_index = 0) {
		if (!size) {
			stream << "[empty]\n";
			return;
		}

		stream << "[";

		switch (type) {
			case data_types::f32: {
				const float* values = reinterpret_cast<const float*>(data);
				size_t count		= detail::min(size / sizeof(float), 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << std::fixed << std::setprecision(6) << values[i];
				}
				break;
			}

			case data_types::f16: {
				const uint16_t* values = reinterpret_cast<const uint16_t*>(data);
				size_t count		   = detail::min(size / sizeof(uint16_t), 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << std::fixed << std::setprecision(6) << f16_to_f32(values[i]);
				}
				break;
			}

			case data_types::q8_0: {
				const uint8_t* values = reinterpret_cast<const uint8_t*>(data);
				size_t count		  = detail::min(size, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << static_cast<int>(values[i]);
				}
				break;
			}

			case data_types::i8: {
				const int8_t* values = reinterpret_cast<const int8_t*>(data);
				size_t count		 = detail::min(size, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << static_cast<int>(values[i]);
				}
				break;
			}

			case data_types::i16: {
				const int16_t* values = reinterpret_cast<const int16_t*>(data);
				size_t count		  = detail::min(size / 2, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << values[i];
				}
				break;
			}

			case data_types::i32: {
				const int32_t* values = reinterpret_cast<const int32_t*>(data);
				size_t count		  = detail::min(size / 4, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << values[i];
				}
				break;
			}

			case data_types::i64: {
				const int64_t* values = reinterpret_cast<const int64_t*>(data);
				size_t count		  = detail::min(size / 8, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << values[i];
				}
				break;
			}

			case data_types::f64: {
				const double* values = reinterpret_cast<const double*>(data);
				size_t count		 = detail::min(size / 8, 8);
				for (size_t i = offending_index; i < count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << std::fixed << std::setprecision(10) << values[i];
				}
				break;
			}

			default:
				stream << "unknown_type";
				break;
		}

		stream << "]\n";
	}

	void print_typed_data(std::ostream& stream, const std::vector<uint8_t>& data, data_types type, int64_t offending_index = 0) {
		if (data.empty()) {
			stream << "[empty]\n";
			return;
		}

		stream << "[";

		switch (type) {
			case data_types::f32: {
				const float* values = reinterpret_cast<const float*>(data.data());
				size_t count		= detail::min(data.size() / sizeof(float), 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << std::fixed << std::setprecision(6) << values[i];
				}
				break;
			}

			case data_types::f16: {
				const uint16_t* values = reinterpret_cast<const uint16_t*>(data.data());
				size_t count		   = detail::min(data.size() / sizeof(uint16_t), 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << std::fixed << std::setprecision(6) << f16_to_f32(values[i]);
				}
				break;
			}

			case data_types::q8_0: {
				const uint8_t* values = reinterpret_cast<const uint8_t*>(data.data());
				size_t count		  = detail::min(data.size(), 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << static_cast<int>(values[i]);
				}
				break;
			}

			case data_types::i8: {
				const int8_t* values = reinterpret_cast<const int8_t*>(data.data());
				size_t count		 = detail::min(data.size(), 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << static_cast<int>(values[i]);
				}
				break;
			}

			case data_types::i16: {
				const int16_t* values = reinterpret_cast<const int16_t*>(data.data());
				size_t count		  = detail::min(data.size() / 2, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << values[i];
				}
				break;
			}

			case data_types::i32: {
				const int32_t* values = reinterpret_cast<const int32_t*>(data.data());
				size_t count		  = detail::min(data.size() / 4, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << values[i];
				}
				break;
			}

			case data_types::i64: {
				const int64_t* values = reinterpret_cast<const int64_t*>(data.data());
				size_t count		  = detail::min(data.size() / 8, 8);
				for (size_t i = offending_index; i < offending_index + count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << values[i];
				}
				break;
			}

			case data_types::f64: {
				const double* values = reinterpret_cast<const double*>(data.data());
				size_t count		 = detail::min(data.size() / 8, 8);
				for (size_t i = offending_index; i < count; ++i) {
					if (i > 0)
						stream << ", ";
					stream << std::fixed << std::setprecision(10) << values[i];
				}
				break;
			}

			default:
				stream << "unknown_type";
				break;
		}

		stream << "]\n";
	}

	template<typename value_type> void print_typed_data(std::stringstream& stream, const value_type* data, uint64_t count) {
		if (!data) {
			stream << "[empty]\n";
			return;
		}

		stream << "[";
		for (size_t i = 0; i < count; ++i) {
			if (i % (count / 10) == 0) {
				if (i > 0)
					stream << ", ";
				stream << std::fixed << std::setprecision(6) << data[i];
			}
		}

		stream << "]\n";
	}
}

#if defined(NIHILUS_DEV)

	#include <jsonifier/Index.hpp>

namespace nihilus {

	enum ggml_op {
		GGML_OP_NONE = 0,
		GGML_OP_DUP,
		GGML_OP_ADD,
		GGML_OP_ADD1,
		GGML_OP_ACC,
		GGML_OP_SUB,
		GGML_OP_MUL,
		GGML_OP_DIV,
		GGML_OP_SQR,
		GGML_OP_SQRT,
		GGML_OP_LOG,
		GGML_OP_SIN,
		GGML_OP_COS,
		GGML_OP_SUM,
		GGML_OP_SUM_ROWS,
		GGML_OP_MEAN,
		GGML_OP_ARGMAX,
		GGML_OP_COUNT_EQUAL,
		GGML_OP_REPEAT,
		GGML_OP_REPEAT_BACK,
		GGML_OP_CONCAT,
		GGML_OP_SILU_BACK,
		GGML_OP_NORM,
		GGML_OP_RMS_NORM,
		GGML_OP_RMS_NORM_BACK,
		GGML_OP_GROUP_NORM,
		GGML_OP_MUL_MAT,
		GGML_OP_MUL_MAT_ID,
		GGML_OP_OUT_PROD,
		GGML_OP_SCALE,
		GGML_OP_SET,
		GGML_OP_CPY,
		GGML_OP_CONT,
		GGML_OP_RESHAPE,
		GGML_OP_VIEW,
		GGML_OP_PERMUTE,
		GGML_OP_TRANSPOSE,
		GGML_OP_GET_ROWS,
		GGML_OP_GET_ROWS_BACK,
		GGML_OP_DIAG,
		GGML_OP_DIAG_MASK_INF,
		GGML_OP_DIAG_MASK_ZERO,
		GGML_OP_SOFT_MAX,
		GGML_OP_SOFT_MAX_BACK,
		GGML_OP_ROPE,
		GGML_OP_ROPE_BACK,
		GGML_OP_CLAMP,
		GGML_OP_CONV_TRANSPOSE_1D,
		GGML_OP_IM2COL,
		GGML_OP_IM2COL_BACK,
		GGML_OP_CONV_TRANSPOSE_2D,
		GGML_OP_POOL_1D,
		GGML_OP_POOL_2D,
		GGML_OP_POOL_2D_BACK,
		GGML_OP_UPSCALE,
		GGML_OP_PAD,
		GGML_OP_PAD_REFLECT_1D,
		GGML_OP_ARANGE,
		GGML_OP_TIMESTEP_EMBEDDING,
		GGML_OP_ARGSORT,
		GGML_OP_LEAKY_RELU,
		GGML_OP_FLASH_ATTN_EXT,
		GGML_OP_FLASH_ATTN_BACK,
		GGML_OP_SSM_CONV,
		GGML_OP_SSM_SCAN,
		GGML_OP_WIN_PART,
		GGML_OP_WIN_UNPART,
		GGML_OP_GET_REL_POS,
		GGML_OP_ADD_REL_POS,
		GGML_OP_RWKV_WKV6,
		GGML_OP_GATED_LINEAR_ATTN,
		GGML_OP_UNARY,
		GGML_OP_MAP_UNARY,
		GGML_OP_MAP_BINARY,
		GGML_OP_MAP_CUSTOM1_F32,
		GGML_OP_MAP_CUSTOM2_F32,
		GGML_OP_MAP_CUSTOM3_F32,
		GGML_OP_MAP_CUSTOM1,
		GGML_OP_MAP_CUSTOM2,
		GGML_OP_MAP_CUSTOM3,
		GGML_OP_CROSS_ENTROPY_LOSS,
		GGML_OP_CROSS_ENTROPY_LOSS_BACK,
		GGML_OP_OPT_STEP_ADAMW,
		GGML_OP_COUNT,
	};

	enum class source_types {
		ggml,
		nihilus,
	};

	struct intermediary_ggml_tensor {
		array<uint64_t, 4> dims{};
		vector<std::string> dims_string{ [] {
			vector<std::string> return_values{};
			return_values.resize(4);
			return return_values;
		}() };
		std::string name{};
		vector<uint8_t> data{};
		data_types type{};
		ggml_op op{};
	};

	constexpr double get_tolerance_for_type(data_types type) {
		switch (type) {
			case data_types::f32:
				return 1e-3;
			case data_types::f16:
				return 1e-3;
			case data_types::f64:
				return 1e-12;
			case data_types::i8:
			case data_types::i16:
			case data_types::i32:
			case data_types::i64:
			case data_types::q8_0:
				return 0.0;
			default:
				return 1e-4;
		}
	}

	constexpr kernel_types convert_ggml_op_to_nihilus_kernel(ggml_op op) noexcept {
		switch (op) {
			case GGML_OP_GET_ROWS:
				return kernel_types::get_rows;
			case GGML_OP_RMS_NORM:
				return kernel_types::rms_norm;
			case GGML_OP_MUL:
				return kernel_types::mul;
			case GGML_OP_MUL_MAT:
			case GGML_OP_MUL_MAT_ID:
				return kernel_types::mul_mat;
			case GGML_OP_RESHAPE:
				return kernel_types::reshape;
			case GGML_OP_PERMUTE:
				return kernel_types::permute;
			case GGML_OP_TRANSPOSE:
				return kernel_types::transpose;
			case GGML_OP_VIEW:
				return kernel_types::view;
			case GGML_OP_CONT:
				return kernel_types::cont;
			case GGML_OP_CPY:
			case GGML_OP_DUP:
				return kernel_types::copy;
			case GGML_OP_ROPE:
				return kernel_types::rope;
			case GGML_OP_SOFT_MAX:
				return kernel_types::softmax;
			case GGML_OP_ADD:
			case GGML_OP_ADD1:
				return kernel_types::add;
			case GGML_OP_SUB:
				return kernel_types::sub;
			case GGML_OP_SILU_BACK:
				return kernel_types::silu;
			case GGML_OP_NONE:
			case GGML_OP_ACC:
			case GGML_OP_DIV:
			case GGML_OP_SQR:
			case GGML_OP_SQRT:
			case GGML_OP_LOG:
			case GGML_OP_SIN:
			case GGML_OP_COS:
			case GGML_OP_SUM:
			case GGML_OP_SUM_ROWS:
			case GGML_OP_MEAN:
			case GGML_OP_ARGMAX:
			case GGML_OP_COUNT_EQUAL:
			case GGML_OP_REPEAT:
			case GGML_OP_REPEAT_BACK:// Repeat backward - not implemented
			case GGML_OP_CONCAT:// Concatenation - not implemented
			case GGML_OP_NORM:// Layer norm - could potentially map to rms_norm
			case GGML_OP_RMS_NORM_BACK:// RMS norm backward - not implemented
			case GGML_OP_GROUP_NORM:// Group normalization - not implemented
			case GGML_OP_OUT_PROD:// Outer product - not implemented
			case GGML_OP_SCALE:// Scaling - could potentially map to mul
			case GGML_OP_SET:// Set values - not implemented
			case GGML_OP_GET_ROWS_BACK:// Get rows backward - not implemented
			case GGML_OP_DIAG:// Diagonal - not implemented
			case GGML_OP_DIAG_MASK_INF:// Diagonal mask with infinity - not implemented
			case GGML_OP_DIAG_MASK_ZERO:// Diagonal mask with zero - not implemented
			case GGML_OP_SOFT_MAX_BACK:// Softmax backward - not implemented
			case GGML_OP_ROPE_BACK:// ROPE backward - not implemented
			case GGML_OP_CLAMP:// Clamp values - not implemented
			case GGML_OP_CONV_TRANSPOSE_1D:// 1D transposed convolution - not implemented
			case GGML_OP_IM2COL:// Image to column - not implemented
			case GGML_OP_IM2COL_BACK:// Image to column backward - not implemented
			case GGML_OP_CONV_TRANSPOSE_2D:// 2D transposed convolution - not implemented
			case GGML_OP_POOL_1D:// 1D pooling - not implemented
			case GGML_OP_POOL_2D:// 2D pooling - not implemented
			case GGML_OP_POOL_2D_BACK:// 2D pooling backward - not implemented
			case GGML_OP_UPSCALE:// Upscaling - not implemented
			case GGML_OP_PAD:// Padding - not implemented
			case GGML_OP_PAD_REFLECT_1D:// 1D reflection padding - not implemented
			case GGML_OP_ARANGE:// Range generation - not implemented
			case GGML_OP_TIMESTEP_EMBEDDING:// Timestep embedding - not implemented
			case GGML_OP_ARGSORT:// Argument sort - not implemented
			case GGML_OP_LEAKY_RELU:// Leaky ReLU - not implemented
			case GGML_OP_FLASH_ATTN_EXT:// Flash attention - not implemented
			case GGML_OP_FLASH_ATTN_BACK:// Flash attention backward - not implemented
			case GGML_OP_SSM_CONV:// State space model convolution - not implemented
			case GGML_OP_SSM_SCAN:// State space model scan - not implemented
			case GGML_OP_WIN_PART:// Window partition - not implemented
			case GGML_OP_WIN_UNPART:// Window unpartition - not implemented
			case GGML_OP_GET_REL_POS:// Get relative position - not implemented
			case GGML_OP_ADD_REL_POS:// Add relative position - not implemented
			case GGML_OP_RWKV_WKV6:// RWKV WKV6 - not implemented
			case GGML_OP_GATED_LINEAR_ATTN:// Gated linear attention - not implemented
			case GGML_OP_UNARY:// Unary operation - not implemented
			case GGML_OP_MAP_CUSTOM1:// Custom operation 1 - not implemented
			case GGML_OP_MAP_CUSTOM2:// Custom operation 2 - not implemented
			case GGML_OP_MAP_CUSTOM3:// Custom operation 3 - not implemented
			case GGML_OP_CROSS_ENTROPY_LOSS:// Cross entropy loss - not implemented
			case GGML_OP_CROSS_ENTROPY_LOSS_BACK:// Cross entropy loss backward - not implemented
			case GGML_OP_OPT_STEP_ADAMW:// AdamW optimizer step - not implemented
			case GGML_OP_COUNT:// Count sentinel - not a real operation
			default:
				return kernel_types::none;
		}
	}

	template<typename value_type> std::ostream& operator<<(std::ostream& os, const vector<value_type>& tensor) {
		os << "[";
		for (uint64_t x = 0; x < tensor.size(); ++x) {
			os << +tensor[x];
			if (x < tensor.size() - 1) {
				os << ",";
			}
		}
		os << "]" << std::endl;
		return os;
	}

	std::ostream& operator<<(std::ostream& os, const array<uint64_t, 4>& tensor) {
		os << "[";
		for (uint64_t x = 0; x < tensor.size(); ++x) {
			os << +tensor[x];
			if (x < tensor.size() - 1) {
				os << ",";
			}
		}
		os << "]" << std::endl;
		return os;
	}

	struct comparison_result {
		std::string result_output{};
		bool result{ true };
		NIHILUS_INLINE operator bool() {
			return result;
		}
	};

	struct intermediary_tensor {
		array<uint64_t, 4> dims{};

		NIHILUS_INLINE intermediary_tensor() noexcept = default;

		NIHILUS_INLINE intermediary_tensor(const intermediary_tensor& other) {
			dims = other.dims;
			data = other.data;
			name = other.name;
			type = other.type;
			op	 = other.op;
		}

		NIHILUS_INLINE intermediary_tensor(intermediary_tensor&& other) noexcept {
			dims = detail::move(other.dims);
			data = detail::move(other.data);
			name = detail::move(other.name);
			type = other.type;
			op	 = other.op;
		}

		NIHILUS_INLINE intermediary_tensor& operator=(const intermediary_tensor& other) {
			if (this != &other) {
				dims = other.dims;
				data = other.data;
				name = other.name;
				type = other.type;
				op	 = other.op;
			}
			return *this;
		}

		NIHILUS_INLINE intermediary_tensor& operator=(intermediary_tensor&& other) noexcept {
			if (this != &other) {
				dims = detail::move(other.dims);
				data = detail::move(other.data);
				name = detail::move(other.name);
				type = other.type;
				op	 = other.op;
			}
			return *this;
		}

		NIHILUS_INLINE intermediary_tensor(const intermediary_ggml_tensor& other) {
			dims = other.dims;
			data = other.data;
			name = other.name;
			type = other.type;
			op	 = convert_ggml_op_to_nihilus_kernel(other.op);
		}

		template<core_traits_types tensor_type> NIHILUS_INLINE intermediary_tensor(tensor_type& other, const std::string& name_new, uint64_t current_block) {
			using output_type = typename tensor_type::output_type;
			dims[0]			  = other[0];
			dims[1]			  = other[1];
			dims[2]			  = other[2];
			dims[3]			  = other[3];

			if constexpr (tensor_type::runtime_dims != 5) {
				dims[tensor_type::runtime_dims] = other.get_mutable_dim();
			}
			source_type			   = source_types::nihilus;
			uint64_t element_count = get_runtime_byte_size(other);
			data.resize(element_count);

			op	 = other.kernel_type;
			name = name_new;
			type = type_traits<output_type>::type;

			if constexpr (array_types<decltype(other.data)>) {
				if (other.data[current_block]) {
					const output_type* src = other.data[current_block];
					std::memcpy(data.data(), src, element_count);
				} else {
					std::cout << "Sorry, but no data for op: " << op << std::endl;
				}
			} else {
				if (other.data) {
					const output_type* src = other.data;
					std::memcpy(data.data(), src, element_count);
				} else {
					std::cout << "Sorry, but no data for op: " << op << std::endl;
				}
			}
		}
		std::string name{};
		vector<uint8_t> data{};
		source_types source_type{ source_types::ggml };
		data_types type{};
		kernel_types op{};
		/*
		bool compare_tensor_data_smart(const intermediary_tensor& data1, const intermediary_tensor& data2, data_types type, std::stringstream& stream,
			size_t max_differences = 5) const {
			if (data1.byte_size != data2.data.size()) {
				stream << "Size mismatch: " << data1.byte_size << " vs " << data2.data.size() << std::endl;
				return false;
			}

			if (!data1.data) {
				stream << "Empty Tensor!" << std::endl;
				return false;
			}

			double tolerance		 = get_tolerance_for_type(type);
			size_t differences_found = 0;
			bool has_differences	 = false;

			switch (type) {
				case data_types::f32: {
					const float* vals1 = reinterpret_cast<const float*>(data1.data);
					const float* vals2 = reinterpret_cast<const float*>(data2.data.data());
					size_t count	   = data1.byte_size / sizeof(float);

					for (size_t i = 0; i < count; ++i) {
						double diff = fabs(static_cast<double>(vals1[i]) - static_cast<double>(vals2[i]));

						bool both_nan = std::isnan(vals1[i]) && std::isnan(vals2[i]);
						bool both_inf = std::isinf(vals1[i]) && std::isinf(vals2[i]) && (std::signbit(vals1[i]) == std::signbit(vals2[i]));

						if (both_nan || both_inf) {
							continue;
						}

						if (diff > tolerance) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "f32 difference at index " << i << ": " << std::scientific << std::setprecision(10) << vals1[i] << " vs " << vals2[i]
									   << " (diff: " << diff << ", tolerance: " << tolerance << ")" << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}

				case data_types::f16: {
					const uint16_t* vals1 = reinterpret_cast<const uint16_t*>(data1.data);
					const uint16_t* vals2 = reinterpret_cast<const uint16_t*>(data2.data.data());
					size_t count		  = data1.byte_size / sizeof(uint16_t);

					for (size_t i = 0; i < count; ++i) {
						float f1	= f16_to_f32(vals1[i]);
						float f2	= f16_to_f32(vals2[i]);
						double diff = fabs(static_cast<double>(f1) - static_cast<double>(f2));

						if (diff > tolerance) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "f16 difference at index " << i << ": " << std::fixed << std::setprecision(6) << f1 << " vs " << f2 << " (diff: " << std::scientific
									   << diff << ")" << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}

				case data_types::f64: {
					const double* vals1 = reinterpret_cast<const double*>(data1.data);
					const double* vals2 = reinterpret_cast<const double*>(data2.data.data());
					size_t count		= data1.byte_size / sizeof(double);

					for (size_t i = 0; i < count; ++i) {
						double diff = fabs(vals1[i] - vals2[i]);

						if (diff > tolerance) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "f64 difference at index " << i << ": " << std::scientific << std::setprecision(15) << vals1[i] << " vs " << vals2[i]
									   << " (diff: " << diff << ")" << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}

				case data_types::i8:
				case data_types::i16:
				case data_types::i32:
				case data_types::i64:
				case data_types::q8_0:
				default: {
					for (size_t i = 0; i < data1.byte_size; ++i) {
						if (static_cast<uint8_t*>(data1.data)[i] != data2.data[i]) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "Byte difference at index " << i << ": " << static_cast<int>(static_cast<uint8_t*>(data1.data)[i]) << " vs "
									   << static_cast<int>(data2.data[i]) << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}
			}

			return !has_differences;
		}*/

		NIHILUS_INLINE comparison_result operator==(const intermediary_tensor& other) const {
			comparison_result return_value{};
			std::stringstream stream{};
			if (op != other.op) {
				//stream << "Incorrect op-types:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
				//<< ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				//stream << "LHS OP: " << op << std::endl;
				//stream << "RHS OP: " << other.op << std::endl;
				//return_value.result		   = false;
				//return_value.result_output = stream.str();
				//return return_value;
			}
			if (type != other.type) {
				stream << "Incorrect Types:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type << ", RHS of source type: " << ( int32_t )other.source_type
					   << std::endl;
				stream << "LHS TYPE: " << ( int32_t )type << std::endl;
				stream << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				return_value.result = false;
				stream << "LHS Byte-Size: " << data.size() << std::endl;
				stream << "RHS Byte-Size: " << other.data.size() << std::endl;
				stream << "LHS Dims: " << dims << std::endl;
				stream << "RHS Dims: " << other.dims << std::endl;
				return_value.result_output = stream.str();
				return return_value;
			}

			if (data.size() != other.data.size()) {
				stream << "Incorrect Byte-Sizes:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
					   << ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				stream << "LHS Byte-Size: " << data.size() << std::endl;
				stream << "RHS Byte-Size: " << other.data.size() << std::endl;
				stream << "LHS TYPE: " << ( int32_t )type << std::endl;
				stream << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				stream << "LHS Dims: " << dims << std::endl;
				stream << "RHS Dims: " << other.dims << std::endl;
				return_value.result		   = false;
				return_value.result_output = stream.str();
				return return_value;
			}

			if (dims != other.dims) {
				stream << "Incorrect Dims:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type << ", RHS of source type: " << ( int32_t )other.source_type
					   << std::endl;
				stream << "LHS Dims: " << dims << std::endl;
				stream << "RHS Dims: " << other.dims << std::endl;
				return_value.result = false;
				stream << "LHS TYPE: " << ( int32_t )type << std::endl;
				stream << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				stream << "LHS Byte-Size: " << data.size() << std::endl;
				stream << "RHS Byte-Size: " << other.data.size() << std::endl;
				return_value.result_output = stream.str();
				return return_value;
			}

			//bool data_equal = compare_tensor_data_smart(*this, other, type, stream);

			//if (!data_equal) {
			//return_value.result		   = false;
			//				return_value.result_output = stream.str();
			//				return return_value;
			//}

			bool result{ dims == other.dims && name == other.name };
			return_value.result = result;
			return return_value;
		}
	};

	struct tensor_wrapper {
		NIHILUS_INLINE tensor_wrapper() noexcept = default;

		NIHILUS_INLINE tensor_wrapper(const tensor_wrapper& other) {
			dims = other.dims;
			data = other.data;
			name = other.name;
			type = other.type;
			op	 = other.op;
		}

		NIHILUS_INLINE tensor_wrapper(tensor_wrapper&& other) noexcept {
			dims = detail::move(other.dims);
			data = detail::move(other.data);
			name = detail::move(other.name);
			type = other.type;
			op	 = other.op;
		}

		NIHILUS_INLINE tensor_wrapper& operator=(const tensor_wrapper& other) {
			if (this != &other) {
				dims = other.dims;
				data = other.data;
				name = other.name;
				type = other.type;
				op	 = other.op;
			}
			return *this;
		}

		NIHILUS_INLINE tensor_wrapper& operator=(tensor_wrapper&& other) noexcept {
			if (this != &other) {
				dims = detail::move(other.dims);
				data = detail::move(other.data);
				name = detail::move(other.name);
				type = other.type;
				op	 = other.op;
			}
			return *this;
		}

		template<core_traits_types tensor_type> NIHILUS_INLINE tensor_wrapper(tensor_type& other, const std::string& name_new, uint64_t current_block) {
			using output_type = typename tensor_type::output_type;
			dims[0]			  = other[0];
			dims[1]			  = other[1];
			dims[2]			  = other[2];
			dims[3]			  = other[3];

			if constexpr (tensor_type::runtime_dims != 5) {
				dims[tensor_type::runtime_dims] = other.get_mutable_dim();
			}
			source_type			   = source_types::nihilus;
			uint64_t element_count = get_runtime_byte_size(other);
			byte_size			   = element_count;
			op					   = other.kernel_type;
			name				   = name_new;
			type				   = type_traits<output_type>::type;

			if constexpr (array_types<decltype(other.data)>) {
				if (other.data[current_block]) {
					data = other.data[current_block];
				} else {
					std::cout << "Sorry, but no data for op: " << op << std::endl;
				}
			} else {
				if (other.data) {
					data = other.data;
				} else {
					std::cout << "Sorry, but no data for op: " << op << std::endl;
				}
			}
		}
		array<uint64_t, 4> dims{};
		std::string name{};
		void* data{};
		uint64_t byte_size{};
		source_types source_type{ source_types::ggml };
		data_types type{};
		kernel_types op{};

		bool compare_tensor_data_smart(const tensor_wrapper& data1, const intermediary_tensor& data2, data_types type, std::stringstream& stream,
			size_t max_differences = 5) const {
			if (data1.byte_size != data2.data.size()) {
				stream << "Size mismatch: " << data1.byte_size << " vs " << data2.data.size() << std::endl;
				return false;
			}

			if (!data1.data) {
				stream << "Empty Tensor!" << std::endl;
				return false;
			}

			double tolerance		 = get_tolerance_for_type(type);
			size_t differences_found = 0;
			bool has_differences	 = false;

			switch (type) {
				case data_types::f32: {
					const float* vals1 = reinterpret_cast<const float*>(data1.data);
					const float* vals2 = reinterpret_cast<const float*>(data2.data.data());
					size_t count	   = data1.byte_size / sizeof(float);

					for (size_t i = 0; i < count; ++i) {
						double diff = fabs(static_cast<double>(vals1[i]) - static_cast<double>(vals2[i]));

						bool both_nan = std::isnan(vals1[i]) && std::isnan(vals2[i]);
						bool both_inf = std::isinf(vals1[i]) && std::isinf(vals2[i]) && (std::signbit(vals1[i]) == std::signbit(vals2[i]));

						if (both_nan || both_inf) {
							continue;
						}

						if (diff > tolerance) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "f32 difference at index " << i << ": " << std::scientific << std::setprecision(10) << vals1[i] << " vs " << vals2[i]
									   << " (diff: " << diff << ", tolerance: " << tolerance << ")" << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}

				case data_types::f16: {
					const uint16_t* vals1 = reinterpret_cast<const uint16_t*>(data1.data);
					const uint16_t* vals2 = reinterpret_cast<const uint16_t*>(data2.data.data());
					size_t count		  = data1.byte_size / sizeof(uint16_t);

					for (size_t i = 0; i < count; ++i) {
						float f1	= f16_to_f32(vals1[i]);
						float f2	= f16_to_f32(vals2[i]);
						double diff = fabs(static_cast<double>(f1) - static_cast<double>(f2));

						if (diff > tolerance) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "f16 difference at index " << i << ": " << std::fixed << std::setprecision(6) << f1 << " vs " << f2 << " (diff: " << std::scientific
									   << diff << ")" << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}

				case data_types::f64: {
					const double* vals1 = reinterpret_cast<const double*>(data1.data);
					const double* vals2 = reinterpret_cast<const double*>(data2.data.data());
					size_t count		= data1.byte_size / sizeof(double);

					for (size_t i = 0; i < count; ++i) {
						double diff = fabs(vals1[i] - vals2[i]);

						if (diff > tolerance) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "f64 difference at index " << i << ": " << std::scientific << std::setprecision(15) << vals1[i] << " vs " << vals2[i]
									   << " (diff: " << diff << ")" << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}

				case data_types::i8:
				case data_types::i16:
				case data_types::i32:
				case data_types::i64:
				case data_types::q8_0:
				default: {
					for (size_t i = 0; i < data1.byte_size; ++i) {
						if (static_cast<uint8_t*>(data1.data)[i] != data2.data[i]) {
							has_differences = true;
							if (differences_found < max_differences) {
								stream << "Incorrect Data:, For Tensor: " << name << std::endl;
								stream << "Byte difference at index " << i << ": " << static_cast<int>(static_cast<uint8_t*>(data1.data)[i]) << " vs "
									   << static_cast<int>(data2.data[i]) << std::endl;
								differences_found++;
								stream << "LHS Data: " << std::endl;
								print_typed_data(stream, static_cast<uint8_t*>(data1.data), data1.byte_size, type, i);
								stream << "RHS Data: " << std::endl;
								print_typed_data(stream, data2.data, type, i);
								break;
							}
						}
					}
					break;
				}
			}

			return !has_differences;
		}

		NIHILUS_INLINE comparison_result operator==(const intermediary_tensor& other) const {
			comparison_result return_value{};
			std::stringstream stream{};
			if (op != other.op) {
				//stream << "Incorrect op-types:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
				//<< ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				//stream << "LHS OP: " << op << std::endl;
				//stream << "RHS OP: " << other.op << std::endl;
				//return_value.result		   = false;
				//return_value.result_output = stream.str();
				//return return_value;
			}
			if (type != other.type) {
				stream << "Incorrect Types:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type << ", RHS of source type: " << ( int32_t )other.source_type
					   << std::endl;
				stream << "LHS TYPE: " << ( int32_t )type << std::endl;
				stream << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				return_value.result = false;
				stream << "LHS Byte-Size: " << byte_size << std::endl;
				stream << "RHS Byte-Size: " << other.data.size() << std::endl;
				stream << "LHS Dims: " << dims << std::endl;
				stream << "RHS Dims: " << other.dims << std::endl;
				return_value.result_output = stream.str();
				return return_value;
			}

			if (byte_size != other.data.size()) {
				stream << "Incorrect Byte-Sizes:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
					   << ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				stream << "LHS Byte-Size: " << byte_size << std::endl;
				stream << "RHS Byte-Size: " << other.data.size() << std::endl;
				stream << "LHS TYPE: " << ( int32_t )type << std::endl;
				stream << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				stream << "LHS Dims: " << dims << std::endl;
				stream << "RHS Dims: " << other.dims << std::endl;
				return_value.result		   = false;
				return_value.result_output = stream.str();
				return return_value;
			}

			if (dims != other.dims) {
				stream << "Incorrect Dims:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type << ", RHS of source type: " << ( int32_t )other.source_type
					   << std::endl;
				stream << "LHS Dims: " << dims << std::endl;
				stream << "RHS Dims: " << other.dims << std::endl;
				return_value.result = false;
				stream << "LHS TYPE: " << ( int32_t )type << std::endl;
				stream << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				stream << "LHS Byte-Size: " << byte_size << std::endl;
				stream << "RHS Byte-Size: " << other.data.size() << std::endl;
				return_value.result_output = stream.str();
				return return_value;
			}

			bool data_equal = compare_tensor_data_smart(*this, other, type, stream);

			if (!data_equal) {
				return_value.result		   = false;
				return_value.result_output = stream.str();
				return return_value;
			}

			bool result{ dims == other.dims && name == other.name };
			return_value.result = result;
			return return_value;
		}
	};
}

namespace jsonifier {
	template<> struct core<nihilus::intermediary_tensor> {
		using value_type				 = nihilus::intermediary_tensor;
		static constexpr auto parseValue = jsonifier::createValue<&value_type::dims, &value_type::name, &value_type::op, &value_type::type, &value_type::data>();
	};
}

namespace nihilus {

	NIHILUS_INLINE std::string convert_op_to_string(op_types type, uint64_t current_block) {
		std::string block{ std::to_string(current_block) };
		switch (type) {
			case op_types::norm_attn_norm: {
				return "attn_norm-" + block;
			}
			case op_types::inp_embd: {
				return "inp_embd";
			}
			case op_types::qcur_reshaped: {
				return "Qcur-" + block + " (reshaped)-02";
			}
			case op_types::qcur_rope: {
				return "Qcur-" + block + "-02";
			}
			case op_types::kcur_rope: {
				return "Kcur-" + block + "-02";
			}
			case op_types::kcur_reshaped: {
				return "Kcur-" + block + " (reshaped)-02";
			}
			case op_types::k_cache_view: {
				return "k_cache_view-" + block;
			}
			case op_types::kq_mask: {
				return "KQ_mask";
			}
			case op_types::inp_pos: {
				return "inp_pos";
			}
			case op_types::inp_tokens: {
				return "inp_tokens";
			}
			case op_types::inp_out_ids: {
				return "inp_out_ids";
			}
			case op_types::attn_k_weight: {
				return "blk." + block + ".attn_k.weight";
			}
			case op_types::attn_q_weight: {
				return "blk." + block + ".attn_q.weight";
			}
			case op_types::attn_v_weight: {
				return "blk." + block + ".attn_v.weight";
			}
			case op_types::attn_norm_weight: {
				return "blk." + block + ".attn_norm.weight";
			}
			case op_types::attn_output_weight: {
				return "blk." + block + ".attn_output.weight";
			}
			case op_types::ffn_down_weight: {
				return "blk." + block + ".ffn_down.weight";
			}
			case op_types::ffn_gate_weight: {
				return "blk." + block + ".ffn_gate.weight";
			}
			case op_types::ffn_norm_weight: {
				return "blk." + block + ".ffn_norm.weight";
			}
			case op_types::ffn_up_weight: {
				return "blk." + block + ".ffn_up.weight";
			}
			case op_types::output_norm_weight: {
				return "output_norm.weight";
			}
			case op_types::output_weight: {
				return "output.weight";
			}
			case op_types::rope_freqs_weight: {
				return "rope_freqs.weight";
			}
			case op_types::token_embd_weight: {
				return "token_embd.weight";
			}
			case op_types::cache_k: {
				return "cache_k_l" + block;
			}
			case op_types::cache_v: {
				return "cache_v_l" + block;
			}
			case op_types::qcur_mul_mat: {
				return "Qcur-" + block;
			}
			case op_types::kcur_mul_mat: {
				return "Kcur-" + block;
			}
			case op_types::vcur_mul_mat: {
				return "Vcur-" + block;
			}
			case op_types::k_cache_view_copy: {
				return "k_cache_view-" + block + " (copy of Kcur-" + block + ")";
			}
			case op_types::vcur_transposed: {
				return "Vcur-" + block + " (transposed)-02";
			}
			case op_types::v_cache_view: {
				return "v_cache_view-" + block;
			}
			case op_types::v_cache_view_copy: {
				return "v_cache_view-" + block + " (copy of Vcur-" + block + " (transposed))";
			}
			case op_types::v: {
				return "v-" + block;
			}
			case op_types::k: {
				return "k-" + block;
			}
			case op_types::q: {
				return "q-" + block;
			}
			case op_types::kq: {
				return "kq-" + block;
			}
			case op_types::kq_soft_max: {
				return "kq_soft_max_ext-" + block;
			}
			case op_types::kqv: {
				return "kqv-" + block;
			}
			case op_types::kqv_merged: {
				return "kqv_merged-" + block;
			}
			case op_types::kqv_merged_cont: {
				return "kqv_merged_cont-" + block;
			}
			case op_types::kqv_out: {
				return "kqv_out-" + block;
			}
			case op_types::ffn_inp_norm_out_ffn_norm: {
				return "ffn_norm-" + block;
			}
			case op_types::ffn_inp: {
				return "ffn_inp-" + block;
			}
			case op_types::ffn_gate: {
				return "ffn_gate-" + block;
			}
			case op_types::ffn_silu: {
				return "ffn_silu-" + block;
			}
			case op_types::ffn_up: {
				return "ffn_up-" + block;
			}
			case op_types::ffn_gate_par: {
				return "ffn_gate_par-" + block;
			}
			case op_types::ffn_out: {
				return "ffn_out-" + block;
			}
			case op_types::l_out: {
				return "l_out-" + block;
			}
			case op_types::final_norm: {
				return "norm";
			}
			case op_types::result_norm: {
				return "result_norm";
			}
			case op_types::attn_residual: {
				return "node_1016";
			}
			case op_types::prev_residual: {
				return "node_1017";
			}
			case op_types::result_output: {
				return "result_output";
			}
			case op_types::count: {
				return "count";
			}
			default: {
				return {};
			}
		}
	}

	template<typename value_type>
	concept has_mutable_dims = requires(std::remove_cvref_t<value_type> value) { value.get_mutable_dim(); };
	template<model_config config> struct tensor_debugger {
		inline static jsonifier::jsonifier_core parser{};

		template<typename tensor_type> static bool compare_tensor_data(tensor_type& tensor, uint64_t current_block, uint64_t iteration, uint64_t runtime_dim) {
			std::string file_name{ convert_op_to_string(tensor.type, current_block) + "-" + std::to_string(iteration) + ".json" };
			file_loader<config> file{ file_name };
			std::string new_string = file.operator const std::string&();
			tensor_wrapper tensor_newer{ tensor, convert_op_to_string(tensor.type, current_block), current_block };
			intermediary_tensor tensor_new{};
			parser.parseJson(tensor_new, new_string);
			std::stringstream stream{};
			if (!new_string.empty()) {
				auto return_value{ tensor_newer == tensor_new };
				if (!return_value) {
					stream << "Not Equal: Tensor of name: " << convert_op_to_string(tensor.type, current_block) << ", OF TYPE: " << tensor.type << " (iteration " << iteration
						   << ")" << std::endl;
					stream << return_value.result_output;
					log<log_levels::status>(stream.str());
				} else {
					stream << "Found an equal op of name: " << convert_op_to_string(tensor.type, current_block) << ", OF TYPE: " << tensor.type << " (iteration " << iteration
						   << ")" << std::endl;
					log<log_levels::status>(stream.str());
				}
				return return_value;
			} else {
				stream << "Not Found: Tensor of name: " << convert_op_to_string(tensor.type, current_block) << ", OF TYPE: " << tensor.type << " (iteration " << iteration << ")"
					   << std::endl;
			}
			log<log_levels::status>(stream.str());
			return false;
		}
	};

}
#else

namespace nihilus {
	template<typename value_type>
	concept has_mutable_dims = requires(std::remove_cvref_t<value_type> value) { value.get_mutable_dim(); };
	template<model_config config> struct tensor_debugger {

		template<typename tensor_type> static bool compare_tensor_data(tensor_type& tensor, uint64_t current_block, uint64_t iteration, uint64_t runtime_dim) {
			return false;
		}
	};
}

#endif
