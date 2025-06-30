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

#if defined(NIHILUS_DEBUG)
	#include <jsonifier/Index.hpp>

#include <nihilus/common/common.hpp>
#include <filesystem>
#include <iostream>
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
		GGML_OP_UPSCALE,// nearest interpolate
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
		std::vector<uint64_t> dims{ [] {
			std::vector<uint64_t> return_values{};
			return_values.resize(4);
			return return_values;
		}() };
		std::string name{};
		std::vector<uint8_t> data{};
		data_types type{};
		ggml_op op{};
	};

	constexpr kernel_types convert_ggml_op_to_nihilus_kernel(ggml_op op) noexcept {
		switch (op) {
			// Direct mappings - perfect 1:1 correspondence
			case GGML_OP_GET_ROWS:
				return kernel_types::get_rows;

			case GGML_OP_RMS_NORM:
				return kernel_types::rms_norm;

			case GGML_OP_MUL:
				return kernel_types::mul;

			case GGML_OP_MUL_MAT:
			case GGML_OP_MUL_MAT_ID:// Matrix multiplication with ID - maps to mul_mat
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
			case GGML_OP_DUP:// Duplicate is essentially a copy operation
				return kernel_types::copy;

			case GGML_OP_ROPE:
				return kernel_types::rope;

			case GGML_OP_SOFT_MAX:
				return kernel_types::softmax;

			case GGML_OP_ADD:
			case GGML_OP_ADD1:// Add scalar - can be handled by add kernel
				return kernel_types::add;

			case GGML_OP_SUB:
				return kernel_types::sub;

			// SILU-related operations
			case GGML_OP_SILU_BACK:// SILU backward pass - maps to silu kernel
				return kernel_types::silu;

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

	template<typename value_type> std::ostream& operator<<(std::ostream& os, const std::vector<value_type>& tensor) {
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

	struct intermediary_tensor {
		std::vector<uint64_t> dims{ [] {
			std::vector<uint64_t> return_values{};
			return_values.resize(4);
			return return_values;
		}() };

		NIHILUS_FORCE_INLINE intermediary_tensor() noexcept = default;

		NIHILUS_FORCE_INLINE intermediary_tensor(const intermediary_tensor& other) {
			dims = other.dims;
			data = other.data;
			name = other.name;
			type = other.type;
			op	 = other.op;
		}

		NIHILUS_FORCE_INLINE intermediary_tensor(intermediary_tensor&& other) noexcept {
			dims = std::move(other.dims);
			data = std::move(other.data);
			name = std::move(other.name);
			type = other.type;
			op	 = other.op;
		}

		NIHILUS_FORCE_INLINE intermediary_tensor& operator=(const intermediary_tensor& other) {
			if (this != &other) {
				dims = other.dims;
				data = other.data;
				name = other.name;
				type = other.type;
				op	 = other.op;
			}
			return *this;
		}

		NIHILUS_FORCE_INLINE intermediary_tensor& operator=(intermediary_tensor&& other) noexcept {
			if (this != &other) {
				dims = std::move(other.dims);
				data = std::move(other.data);
				name = std::move(other.name);
				type = other.type;
				op	 = other.op;
			}
			return *this;
		}

		// YOUR EXISTING CONSTRUCTORS...
		NIHILUS_FORCE_INLINE intermediary_tensor(const intermediary_ggml_tensor& other) {
			dims = other.dims;
			data = other.data;
			name = other.name;
			type = other.type;
			op	 = convert_ggml_op_to_nihilus_kernel(other.op);
		}

		template<core_traits_type tensor_type> NIHILUS_FORCE_INLINE intermediary_tensor(tensor_type& other, const std::string& name_new, uint64_t current_block) {
			using output_type = typename tensor_type::output_type;
			dims[0]			  = other[0];
			dims[1]			  = other[1];
			dims[2]			  = other[2];
			dims[3]			  = other[3];
			source_type		  = source_types::nihilus;
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
		source_types source_type{ source_types::ggml };
		data_types type{};
		kernel_types op{};
		NIHILUS_FORCE_INLINE bool operator==(intermediary_tensor& other) const {
			if (op != other.op) {
				std::cout << "Incorret op-types:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
						  << ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				std::cout << "LHS OP: " << ( int32_t )op << std::endl;
				std::cout << "RHS OP: " << ( int32_t )other.op << std::endl;
				return false;
			}
			if (type != other.type) {
				std::cout << "Incorret Types:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
						  << ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				std::cout << "LHS TYPE: " << ( int32_t )type << std::endl;
				std::cout << "RHS TYPE: " << ( int32_t )other.type << std::endl;
				return false;
			}

			if (dims != other.dims) {
				std::cout << "Incorret Dims:, For Tensor: " << name << ", LHS of source type: " << ( int32_t )source_type
						  << ", RHS of source type: " << ( int32_t )other.source_type << std::endl;
				std::cout << "LHS Dims: " << dims << std::endl;
				std::cout << "RHS Dims: " << other.dims << std::endl;
				return false;
			}
			size_t this_dims  = dims[0] * dims[1] * dims[2] * dims[3];
			size_t other_dims = other.dims[0] * other.dims[1] * other.dims[2] * other.dims[3];
			size_t this_size  = get_type_traits(type).type_size * this_dims;
			size_t other_size = get_type_traits(other.type).type_size * other_dims;
			size_t final_size = std::min(this_size, other_size);
			final_size		  = std::min(final_size, static_cast<size_t>(128ull));

			int64_t equal_data = std ::memcmp(data.data(), other.data.data(), final_size);
			for (int32_t i = 0; i < final_size; i++) {
				if (data.data()[i] != other.data.data()[i]) {
					printf("Difference at %d: %d vs %d\n", i, data[i], other.data[i]);
					equal_data = 1;
					break;
				}
			}

			if (equal_data) {
				std::cout << "Incorret Data:, For Tensor: " << name << std::endl;
				std::cout << "At Index: " << equal_data << std::endl;
				std::cout << "LHS Data: " << data << std::endl;
				std::cout << "RHS Data: " << other.data << std::endl;
				return false;
			}

			return dims == other.dims && name == other.name;
		}
	};
}

namespace jsonifier {
	template<> struct core<nihilus::intermediary_ggml_tensor> {
		using value_type				 = nihilus::intermediary_ggml_tensor;
		static constexpr auto parseValue = createValue<&value_type::dims, &value_type::name, &value_type::op, &value_type::type, &value_type::data>();
	};
}

namespace nihilus {

	NIHILUS_FORCE_INLINE std::string convert_op_to_string(op_types type, uint64_t current_block) {
		std::string block{ std::to_string(current_block) };
		switch (type) {
			case op_types::inp_embd: {
				return "inp_embd";
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
			default: {
				return {};
			}
		}
	}

	std::vector<std::unordered_map<std::string, intermediary_tensor>> get_tensors_multi_iteration(std::string_view base_path, std::string_view base_name) {
		std::vector<std::unordered_map<std::string, intermediary_tensor>> return_values{};

		// Try to load files with incrementing indices until we can't find any more
		for (int32_t iteration = 0;; ++iteration) {
			// Construct the filename: "Node_Data_0.json", "Node_Data_1.json", etc.
			std::string filename = std::string(base_path) + "/" + std::string(base_name) + "_" + std::to_string(iteration) + ".json";

			// Check if file exists
			std::ifstream test_file(filename);
			if (!test_file.good()) {
				break;// No more files found, stop iteration
			}
			test_file.close();

			std::unordered_map<std::string, intermediary_ggml_tensor> iteration_ggml{};
			std::unordered_map<std::string, intermediary_tensor> iteration_tensors{};

			try {
				file_loader<false> file_loader{ filename };
				jsonifier::jsonifier_core parser{};
				parser.parseJson<jsonifier::parse_options{ .minified = true }>(iteration_ggml, file_loader.operator const std::string&());

				for (auto& [key, value]: iteration_ggml) {
					iteration_tensors[key] = value;
					//std::cout << "Iteration " << iteration << " - " << key << std::endl;
					//std::cout << value << std::endl;
				}

				for (auto& value: parser.getErrors()) {
					//std::cout << "Iteration " << iteration << " - Error: " << value << std::endl;
				}

				return_values.push_back(std::move(iteration_tensors));

			} catch (const std::exception& e) {
				std::cout << "Failed to parse iteration " << iteration << ": " << e.what() << std::endl;
				break;
			}
		}

		return return_values;
	}

	struct tensor_debugger {
		inline static std::vector<std::unordered_map<std::string, intermediary_tensor>> leaf_iterations{ get_tensors_multi_iteration("C:/users/chris/source/repos/ft-tl", "Leaf_Data") };

		inline static std::vector<std::unordered_map<std::string, intermediary_tensor>> node_iterations{ get_tensors_multi_iteration("C:/users/chris/source/repos/ft-tl", "Node_Data") };

		// Helper methods to access specific iterations
		static const std::unordered_map<std::string, intermediary_tensor>& get_leaf_iteration(size_t iteration) {
			static const std::unordered_map<std::string, intermediary_tensor> empty{};
			return (iteration < leaf_iterations.size()) ? leaf_iterations[iteration] : empty;
		}

		static const std::unordered_map<std::string, intermediary_tensor>& get_node_iteration(size_t iteration) {
			static const std::unordered_map<std::string, intermediary_tensor> empty{};
			return (iteration < node_iterations.size()) ? node_iterations[iteration] : empty;
		}

		// Get total number of iterations available
		static size_t get_iteration_count() {
			return std::max(leaf_iterations.size(), node_iterations.size());
		}

		template<typename tensor_type> static bool compare_tensor_data(tensor_type& tensor, uint64_t current_block, uint64_t iteration) {
			std::string tensor_name{ convert_op_to_string(tensor.type, current_block) };
			// Get the appropriate iteration data
			const auto& current_leafs = get_leaf_iteration(iteration);
			const auto& current_nodes = get_node_iteration(iteration);
			intermediary_tensor tensor_new{ tensor, tensor_name, current_block };

			if (current_leafs.contains(tensor_name)) {
				bool return_value{ tensor_new == current_leafs.at(tensor_name) };
				if (!return_value) {
					std::cout << "Found an op of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << " (iteration " << iteration << ")" << std::endl;
				}
				return return_value;
			} else if (current_nodes.contains(tensor_name)) {
				bool return_value{ tensor_new == current_nodes.at(tensor_name) };
				if (!return_value) {
					std::cout << "Found an op of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << " (iteration " << iteration << ")" << std::endl;
				}
				return return_value;
			} else {
				std::cout << "Not Found: Tensor of name: " << tensor_name << ", OF TYPE: " << ( int32_t )tensor.type << " (iteration " << iteration << ")" << std::endl;
				return false;
			}
		}
	};

}
#endif