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

#include <jsonifier/Index.hpp>
#include <nihilus/common/arch_traits.hpp>
#include <nihilus/common/common.hpp>
#include <filesystem>
#include <stdexcept>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <string>

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
		GGML_OP_NORM,// normalize
		GGML_OP_RMS_NORM,
		GGML_OP_RMS_NORM_BACK,
		GGML_OP_GROUP_NORM,
		GGML_OP_L2_NORM,

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
		GGML_OP_CONV_2D_DW,
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
		GGML_OP_RWKV_WKV7,

		GGML_OP_UNARY,

		GGML_OP_MAP_CUSTOM1,
		GGML_OP_MAP_CUSTOM2,
		GGML_OP_MAP_CUSTOM3,

		GGML_OP_CUSTOM,

		GGML_OP_CROSS_ENTROPY_LOSS,
		GGML_OP_CROSS_ENTROPY_LOSS_BACK,
		GGML_OP_OPT_STEP_ADAMW,

		GGML_OP_COUNT,
	};

	struct intermediary_ggml_tensor {
		std::vector<uint64_t> dims{ [] {
			std::vector<uint64_t> return_values{};
			return_values.resize(4);
			return return_values;
		}() };
		std::string name{};
		std::vector<uint8_t> data{};
		data_type type{};
		ggml_op op{};
	};

	constexpr kernel_type convert_ggml_op_to_nihilus_kernel(ggml_op op) noexcept {
		switch (op) {
			// Direct mappings - perfect 1:1 correspondence
			case GGML_OP_GET_ROWS:
				return kernel_type::get_rows;

			case GGML_OP_RMS_NORM:
				return kernel_type::rms_norm;

			case GGML_OP_MUL:
				return kernel_type::mul;

			case GGML_OP_MUL_MAT:
			case GGML_OP_MUL_MAT_ID:// Matrix multiplication with ID - maps to mul_mat
				return kernel_type::mul_mat;

			case GGML_OP_RESHAPE:
				return kernel_type::reshape;

			case GGML_OP_PERMUTE:
				return kernel_type::permute;

			case GGML_OP_TRANSPOSE:
				return kernel_type::transpose;

			case GGML_OP_VIEW:
				return kernel_type::view;

			case GGML_OP_CONT:
				return kernel_type::cont;

			case GGML_OP_CPY:
			case GGML_OP_DUP:// Duplicate is essentially a copy operation
				return kernel_type::copy;

			case GGML_OP_ROPE:
				return kernel_type::rope;

			case GGML_OP_SOFT_MAX:
				return kernel_type::softmax;

			case GGML_OP_ADD:
			case GGML_OP_ADD1:// Add scalar - can be handled by add kernel
				return kernel_type::add;

			case GGML_OP_SUB:
				return kernel_type::sub;

			// SILU-related operations
			case GGML_OP_SILU_BACK:// SILU backward pass - maps to silu kernel
				return kernel_type::silu;

			// Operations that don't have direct Nihilus equivalents or are unsupported
			case GGML_OP_NONE:
			case GGML_OP_ACC:// Accumulate - could potentially map to add
			case GGML_OP_DIV:// Division - not implemented in Nihilus yet
			case GGML_OP_SQR:// Square - not implemented
			case GGML_OP_SQRT:// Square root - not implemented
			case GGML_OP_LOG:// Logarithm - not implemented
			case GGML_OP_SIN:// Sine - not implemented
			case GGML_OP_COS:// Cosine - not implemented
			case GGML_OP_SUM:// Sum reduction - not implemented
			case GGML_OP_SUM_ROWS:// Row-wise sum - not implemented
			case GGML_OP_MEAN:// Mean - not implemented
			case GGML_OP_ARGMAX:// Argmax - not implemented
			case GGML_OP_COUNT_EQUAL:// Count equal elements - not implemented
			case GGML_OP_REPEAT:// Repeat tensor - not implemented
			case GGML_OP_REPEAT_BACK:// Repeat backward - not implemented
			case GGML_OP_CONCAT:// Concatenation - not implemented
			case GGML_OP_NORM:// Layer norm - could potentially map to rms_norm
			case GGML_OP_RMS_NORM_BACK:// RMS norm backward - not implemented
			case GGML_OP_GROUP_NORM:// Group normalization - not implemented
			case GGML_OP_L2_NORM:// L2 normalization - not implemented
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
			case GGML_OP_CONV_2D_DW:// 2D depthwise convolution - not implemented
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
			case GGML_OP_RWKV_WKV7:// RWKV WKV7 - not implemented
			case GGML_OP_UNARY:// Unary operation - not implemented
			case GGML_OP_MAP_CUSTOM1:// Custom operation 1 - not implemented
			case GGML_OP_MAP_CUSTOM2:// Custom operation 2 - not implemented
			case GGML_OP_MAP_CUSTOM3:// Custom operation 3 - not implemented
			case GGML_OP_CUSTOM:// Custom operation - not implemented
			case GGML_OP_CROSS_ENTROPY_LOSS:// Cross entropy loss - not implemented
			case GGML_OP_CROSS_ENTROPY_LOSS_BACK:// Cross entropy loss backward - not implemented
			case GGML_OP_OPT_STEP_ADAMW:// AdamW optimizer step - not implemented
			case GGML_OP_COUNT:// Count sentinel - not a real operation
			default:
				return kernel_type::none;
		}
	}

	template<typename value_type> std::ostream& operator<<(std::ostream& os, const std::vector<value_type>& tensor) {
		os << "[";
		for (size_t x = 0; x < tensor.size(); ++x) {
			os << +tensor[x];
			if (x < tensor.size() - 1) {
				os << ",";
			}
		}
		os << "]" << std::endl;
		return os;
	}

	struct intermediary_tensor {
		std::vector<uint64_t> dims{ [] {
			std::vector<uint64_t> return_values{};
			return_values.resize(4);
			return return_values;
		}() };
		NIHILUS_FORCE_INLINE intermediary_tensor() noexcept = default;
		NIHILUS_FORCE_INLINE intermediary_tensor(const intermediary_ggml_tensor& other) {
			dims = other.dims;
			data = other.data;
			name = other.name;
			type = other.type;
			op	 = convert_ggml_op_to_nihilus_kernel(other.op);
		}

		template<core_traits_type tensor_type> NIHILUS_FORCE_INLINE intermediary_tensor(const tensor_type& other, const std::string& name_new, size_t current_block) {
			using output_type = typename tensor_type::output_type;
			dims[0]			  = other.dims[tag<0>{}];
			dims[1]			  = other.dims[tag<1>{}];
			dims[2]			  = other.dims[tag<2>{}];
			dims[3]			  = other.dims[tag<3>{}];
			data.resize(128);
			if constexpr (array_type<decltype(other.data)>) {
				if (other.data[current_block]) {
					std::memcpy(data.data(), other.data[current_block], 128);
				}
			} else {
				if (other.data) {
					std::memcpy(data.data(), other.data, 128);
				}
			}
			name = name_new;
			type = type_traits<output_type>::type;
			op	 = other.krn_type;
		}
		std::string name{};
		std::vector<uint8_t> data{};
		data_type type{};
		kernel_type op{};
		NIHILUS_FORCE_INLINE bool operator==(const intermediary_tensor& other) const {
			if (op != other.op) {
				std::cout << "Incorret op-types:, For Tensor: " << name << std::endl;
				std::cout << "LHS OP: " << ( int32_t )op << std::endl;
				std::cout << "RHS OP: " << ( int32_t )other.op << std::endl;
			}
			if (type != other.type) {
				std::cout << "Incorret Types:, For Tensor: " << name << std::endl;
				std::cout << "LHS TYPE: " << ( int32_t )type << std::endl;
				std::cout << "RHS TYPE: " << ( int32_t )other.type << std::endl;
			}

			if (dims != other.dims) {
				std::cout << "Incorret Dims:, For Tensor: " << name << std::endl;
				std::cout << "LHS Dims: " << dims << std::endl;
				std::cout << "RHS Dims: " << other.dims << std::endl;
			}
			if (data != other.data) {
				std::cout << "Incorret Data:, For Tensor: " << name << std::endl;
				std::cout << "LHS Data: " << data << std::endl;
				std::cout << "RHS Data: " << other.data << std::endl;
			}

			return data == other.data && dims == other.dims && name == other.name;
		}
	};

	std::ostream& operator<<(std::ostream& os, const std::array<uint64_t, 4>& tensor) {
		os << "[";
		os << tensor[0];
		os << ",";
		os << tensor[1];
		os << ",";
		os << tensor[2];
		os << ",";
		os << tensor[3];
		os << "]" << std::endl;
		return os;
	}

	std::ostream& operator<<(std::ostream& os, const intermediary_ggml_tensor& tensor) {
		os << "Name: ";
		os << tensor.name << std::endl;
		os << "Dims: ";
		os << tensor.dims << std::endl;
		os << "Type: ";
		os << get_type_name(tensor.type) << std::endl;
		os << "Op-Type: ";
		os << kernel_names[convert_ggml_op_to_nihilus_kernel(tensor.op)] << std::endl;
		os << std::endl;
		//os << tensor.data.data() << std::endl;
		return os;
	}
}

namespace jsonifier {
	template<> struct core<nihilus::intermediary_ggml_tensor> {
		using value_type				 = nihilus::intermediary_ggml_tensor;
		static constexpr auto parseValue = createValue<&value_type::dims, &value_type::name, &value_type::op, &value_type::type, &value_type::data>();
	};
}

namespace nihilus {

	NIHILUS_FORCE_INLINE std::string convert_op_to_string(llama_op_types type, size_t current_block) {
		std::string block{ std::to_string(current_block) };
		switch (type) {
			case llama_op_types::inp_embd: {
				return "inp_embd";
			}
			case llama_op_types::inp_tokens: {
				return "leaf_2";
			}
			case llama_op_types::attn_k_weight: {
				return "blk." + block + ".attn_k.weight";
			}
			case llama_op_types::attn_q_weight: {
				return "blk." + block + ".attn_q.weight";
			}
			case llama_op_types::attn_v_weight: {
				return "blk." + block + ".attn_v.weight";
			}
			case llama_op_types::attn_norm_weight: {
				return "blk." + block + ".attn_norm.weight";
			}
			case llama_op_types::attn_output_weight: {
				return "blk." + block + ".attn_output.weight";
			}
			case llama_op_types::ffn_down_weight: {
				return "blk." + block + ".ffn_down.weight";
			}
			case llama_op_types::ffn_gate_weight: {
				return "blk." + block + ".ffn_gate.weight";
			}
			case llama_op_types::ffn_norm_weight: {
				return "blk." + block + ".ffn_norm.weight";
			}
			case llama_op_types::ffn_up_weight: {
				return "blk." + block + ".ffn_up.weight";
			}
			case llama_op_types::output_norm_weight: {
				return "output_norm.weight";
			}
			case llama_op_types::output_weight: {
				return "output.weight";
			}
			case llama_op_types::rope_freqs_weight: {
				return "rope_freqs.weight";
			}
			case llama_op_types::token_embd_weight: {
				return "token_embd.weight";
			}
			case llama_op_types::cache_k: {
				return "cache_k_l" + block;
			}
			case llama_op_types::cache_v: {
				return "cache_v_l" + block;
			}
			default: {
				return {};
			}
		}
	}

	std::map<std::string, intermediary_tensor> get_tensors(std::string_view path) {
		std::map<std::string, intermediary_ggml_tensor> return_values_ggml{};
		std::map<std::string, intermediary_tensor> return_values{};
		file_loader<false> file_loader{ path };
		jsonifier::jsonifier_core parser{};
		parser.parseJson<jsonifier::parse_options{ .minified = true }>(return_values_ggml, file_loader.operator const std::string&());
		for (auto& [key, value]: return_values_ggml) {
			return_values[key] = value;
			std::cout << key << std::endl;
			std::cout << value << std::endl;
		}
		for (auto& value: parser.getErrors()) {
			std::cout << value << std::endl;
		}
		return return_values;
	}

	struct tensor_debugger {
		inline static std::map<std::string, intermediary_tensor> leafs{ get_tensors("C:/users/chris/source/repos/ft-tl/Leaf_Data.json") };
		inline static std::map<std::string, intermediary_tensor> nodes{ get_tensors("C:/users/chris/source/repos/ft-tl/Node_Data.json") };
		template<core_traits_type tensor_type> static bool compare_tensor_data(const tensor_type& tensor, size_t current_block) {
			std::string tensor_name{ convert_op_to_string(tensor.type, current_block) };
			if (leafs.contains(tensor_name)) {
				intermediary_tensor tensor_new{ tensor, tensor_name, current_block };
				std::cout << "Found an op of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << std::endl;
				return tensor_new == leafs[tensor_name];
			} else if (nodes.contains(tensor_name)) {
				intermediary_tensor tensor_new{ tensor, tensor_name, current_block };
				std::cout << "Found an op of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << std::endl;
				return tensor_new == nodes[tensor_name];
			} else {
				return false;
			}
		}

		static bool compare_tensor_data(const intermediary_tensor& tensor) {
			std::string tensor_name{ tensor.name };
			if (leafs.contains(tensor_name)) {
				intermediary_tensor tensor_new{ tensor };
				std::cout << "Found an op of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << std::endl;
				return tensor_new == leafs[tensor_name];
			} else if (nodes.contains(tensor_name)) {
				intermediary_tensor tensor_new{ tensor };
				std::cout << "Found an op of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << std::endl;
				return tensor_new == nodes[tensor_name];
			} else {
				return false;
			}
		}
	};

}
