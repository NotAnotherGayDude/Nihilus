#include <nihilus>

using namespace nihilus;

uint64_t get_data_size(uint64_t element_count, data_types type) {
	switch (type) {
		case data_types::f16: {
			return element_count * 2;
		}
		case data_types::f32: {
			return element_count * 4;
		}
		case data_types::f64: {
			return element_count * 8;
		}
		case data_types::i32: {
			return element_count * 4;
		}
		case data_types::i64: {
			return element_count * 8;
		}
		case data_types::i8: {
			return element_count * 1;
		}
		case data_types::q8_0: {
			return (element_count + 31) / 32 * 34;
		}
	}
}

struct data_stream {
	uint64_t element_count{};
	data_types data_type{};
};

struct tensor_op {
	std::string_view name{};
	std::vector<data_stream> inputs{};
	data_stream output{};
};

struct read_write {
	uint64_t written_bytes{};
	uint64_t read_bytes{};
};


read_write get_read_writes(std::vector<tensor_op> inputs) {
	read_write return_values{};
	for (auto& value: inputs) {
		for (auto& value_new: value.inputs) {
			return_values.read_bytes += get_data_size(value_new.element_count, value_new.data_type);
		}
		return_values.written_bytes += get_data_size(value.output.element_count, value.output.data_type);
	}
	return return_values;
}

constexpr uint32_t embedding_length		   = 4096;
constexpr uint32_t vocab_size			   = 128256;
constexpr uint32_t feed_forward_length	   = 14336;
constexpr uint32_t attention_head_count	   = 32;
constexpr uint32_t block_count			   = 32;
constexpr uint32_t attention_head_count_kv = 8;
constexpr uint32_t rope_dimension_count	   = 128;
constexpr uint64_t n_embd_kv_gqa		   = rope_dimension_count * attention_head_count_kv;
constexpr uint32_t seq_length			   = 1024;
constexpr uint64_t full_cache_size_k	   = seq_length * n_embd_kv_gqa;
constexpr uint64_t full_cache_size_v	   = seq_length * n_embd_kv_gqa;

constexpr uint64_t token_embd_elements		= embedding_length * vocab_size;
constexpr uint64_t blk_attn_q_elements		= embedding_length * embedding_length;
constexpr uint64_t blk_attn_k_elements		= embedding_length * n_embd_kv_gqa;
constexpr uint64_t blk_attn_v_elements		= embedding_length * n_embd_kv_gqa;
constexpr uint64_t blk_attn_output_elements = embedding_length * embedding_length;
constexpr uint64_t blk_attn_norm_elements	= embedding_length;
constexpr uint64_t blk_ffn_gate_elements	= embedding_length * feed_forward_length;
constexpr uint64_t blk_ffn_up_elements		= embedding_length * feed_forward_length;
constexpr uint64_t blk_ffn_down_elements	= feed_forward_length * embedding_length;
constexpr uint64_t blk_ffn_norm_elements	= embedding_length;
constexpr uint64_t cache_k_l0_elements		= full_cache_size_k;
constexpr uint64_t cache_v_l0_elements		= full_cache_size_v;
constexpr uint64_t inp_tokens_elements		= 2ULL;
constexpr uint64_t inp_pos_elements			= 2ULL;
constexpr uint64_t rope_freqs_elements		= rope_dimension_count / 2;
constexpr uint64_t kq_mask_elements			= attention_head_count * attention_head_count;
constexpr uint64_t output_norm_elements		= embedding_length;
constexpr uint64_t inp_out_ids_elements		= 1ULL;

std::vector<tensor_op> create_layer_tensor_ops_with_seqlen() {
	std::vector<tensor_op> ops;

	ops.emplace_back(tensor_op{ .name = "inp_embd",
		.inputs = { { .element_count = embedding_length * vocab_size, .data_type = data_types::q8_0 }, { .element_count = seq_length, .data_type = data_types::i32 } },
		.output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

	for (uint64_t x = 0; x < block_count; ++x) {
		ops.emplace_back(tensor_op{ .name = "norm-0",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "attn_norm-0",
			.inputs = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 }, { .element_count = embedding_length, .data_type = data_types::f32 } },
			.output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "Qcur-0",
			.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });
		/*
		ops.emplace_back(tensor_op{ .name = "Qcur-0 (reshaped)",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32 } });
			*/
		ops.emplace_back(tensor_op{ .name = "Qcur-0",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32 },
									  { .element_count = seq_length, .data_type = data_types::i32 }, { .element_count = rope_dimension_count / 2, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "Kcur-0",
			.inputs						  = { { .element_count = embedding_length * n_embd_kv_gqa, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = n_embd_kv_gqa * seq_length, .data_type = data_types::f32 } });
		/*
		ops.emplace_back(tensor_op{ .name = "Kcur-0 (reshaped)",
			.inputs						  = { { .element_count = n_embd_kv_gqa * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = data_types::f32 } });
			*/
		ops.emplace_back(tensor_op{ .name = "Kcur-0",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = data_types::f32 },
									  { .element_count = seq_length, .data_type = data_types::i32 }, { .element_count = rope_dimension_count / 2, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "Vcur-0",
			.inputs						  = { { .element_count = embedding_length * n_embd_kv_gqa, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = n_embd_kv_gqa * seq_length, .data_type = data_types::f32 } });
		/*
		ops.emplace_back(tensor_op{ .name = "k_cache_view-0",
			.inputs						  = { { .element_count = full_cache_size_k, .data_type = data_types::f16 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });
			*/
		ops.emplace_back(tensor_op{ .name = "k_cache_view-0 (copy of Kcur-0)",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = data_types::f32 },
									  { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });
		/*
		ops.emplace_back(tensor_op{ .name = "Vcur-0 (transposed)",
			.inputs						  = { { .element_count = n_embd_kv_gqa * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "v_cache_view-0",
			.inputs						  = { { .element_count = full_cache_size_v, .data_type = data_types::f16 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });
			*/
		ops.emplace_back(tensor_op{ .name = "v_cache_view-0 (copy of Vcur-0 (transposed))",
			.inputs						  = { { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f32 },
									  { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });
		/*
		ops.emplace_back(tensor_op{ .name = "v-0",
			.inputs						  = { { .element_count = full_cache_size_v, .data_type = data_types::f16 } },
			.output						  = { .element_count = attention_head_count * rope_dimension_count * attention_head_count_kv, .data_type = data_types::f16 } });
			
		ops.emplace_back(tensor_op{ .name = "k-0",
			.inputs						  = { { .element_count = full_cache_size_k, .data_type = data_types::f16 } },
			.output						  = { .element_count = rope_dimension_count * attention_head_count * attention_head_count_kv, .data_type = data_types::f16 } });
			
		ops.emplace_back(tensor_op{ .name = "q-0",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = data_types::f32 } });
			*/
		ops.emplace_back(tensor_op{ .name = "kq-0",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * attention_head_count_kv, .data_type = data_types::f16 },
									  { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = data_types::f32 } },
			.output						  = { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "kq_soft_max_ext-0",
			.inputs						  = { { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = data_types::f32 },
									  { .element_count = attention_head_count * attention_head_count, .data_type = data_types::f32 } },
			.output						  = { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "kqv-0",
			.inputs						  = { { .element_count = attention_head_count * rope_dimension_count * attention_head_count_kv, .data_type = data_types::f16 },
									  { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = data_types::f32 } });
		/*
		ops.emplace_back(tensor_op{ .name = "kqv_merged-0",
			.inputs						  = { { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32 } });
			*/
		ops.emplace_back(tensor_op{ .name = "kqv_merged_cont-0",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "kqv_out-0",
			.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_inp-0",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "norm-0",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_norm-0",
			.inputs = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 }, { .element_count = embedding_length, .data_type = data_types::f32 } },
			.output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_gate-0",
			.inputs						  = { { .element_count = embedding_length * feed_forward_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_silu-0",
			.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_up-0",
			.inputs						  = { { .element_count = embedding_length * feed_forward_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_gate_par-0",
			.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 },
									  { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_out-0",
			.inputs						  = { { .element_count = feed_forward_length * embedding_length, .data_type = data_types::q8_0 },
									  { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "l_out-0",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });
	}

	ops.emplace_back(tensor_op{ .name = "norm",
		.inputs						  = { { .element_count = embedding_length, .data_type = data_types::f32 } },
		.output						  = { .element_count = embedding_length, .data_type = data_types::f32 } });

	ops.emplace_back(tensor_op{ .name = "result_norm",
		.inputs = { { .element_count = embedding_length, .data_type = data_types::f32 }, { .element_count = embedding_length, .data_type = data_types::f32 } },
		.output = { .element_count = embedding_length, .data_type = data_types::f32 } });

	ops.emplace_back(tensor_op{ .name = "result_output",
		.inputs = { { .element_count = embedding_length * vocab_size, .data_type = data_types::q8_0 }, { .element_count = embedding_length, .data_type = data_types::f32 } },
		.output = { .element_count = vocab_size, .data_type = data_types::f32 } });

	return ops;
}

std::vector<tensor_op> create_optimized_layer_tensor_ops() {
	std::vector<tensor_op> ops;

	ops.emplace_back(tensor_op{ .name = "GET_ROWS",
		.inputs = { { .element_count = embedding_length * vocab_size, .data_type = data_types::q8_0 }, { .element_count = seq_length, .data_type = data_types::i32 } },
		.output = { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 } });

	for (uint64_t x = 0; x < block_count; ++x) {
		ops.emplace_back(tensor_op{ .name = "RMS_NORM_MUL_MUL_MAT_RESHAPE_Q",
			.inputs = { { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 }, { .element_count = embedding_length, .data_type = data_types::f32 },
				{ .element_count = embedding_length * embedding_length, .data_type = data_types::q8_0 } },
			.output = { .element_count = rope_dimension_count * block_count * seq_length, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "RMS_NORM_MUL_MUL_MAT_RESHAPE_K",
			.inputs = { { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 }, { .element_count = embedding_length, .data_type = data_types::f32 },
				{ .element_count = embedding_length * n_embd_kv_gqa, .data_type = data_types::q8_0 } },
			.output = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "RMS_NORM_MUL_MUL_MAT_TRANSPOSE_COPY_V",
			.inputs = { { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 }, { .element_count = embedding_length, .data_type = data_types::f32 },
				{ .element_count = embedding_length * n_embd_kv_gqa, .data_type = data_types::q8_0 }, { .element_count = full_cache_size_v, .data_type = data_types::f16 } },
			.output = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });

		ops.emplace_back(tensor_op{ .name = "ROPE_PERMUTE",
			.inputs						  = { { .element_count = rope_dimension_count * block_count * seq_length, .data_type = data_types::q8_0 },
									  { .element_count = seq_length, .data_type = data_types::i32 }, { .element_count = rope_dimension_count / 2, .data_type = data_types::f32 } },
			.output						  = { .element_count = rope_dimension_count * 1 * block_count, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "ROPE_COPY",
			.inputs						  = { { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = data_types::q8_0 },
									  { .element_count = seq_length, .data_type = data_types::i32 }, { .element_count = rope_dimension_count / 2, .data_type = data_types::f32 },
									  { .element_count = full_cache_size_k, .data_type = data_types::f16 } },
			.output						  = { .element_count = n_embd_kv_gqa * seq_length, .data_type = data_types::f16 } });

		ops.emplace_back(tensor_op{ .name = "MUL_MAT_ATTN_SCORES",
			.inputs						  = { { .element_count = rope_dimension_count * block_count * attention_head_count_kv, .data_type = data_types::f16 },
									  { .element_count = rope_dimension_count * 1 * block_count, .data_type = data_types::q8_0 } },
			.output						  = { .element_count = block_count * 1 * block_count, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "SOFT_MAX_MUL_MAT_PERMUTE_CONT",
			.inputs						  = { { .element_count = block_count * 1 * block_count, .data_type = data_types::f32 },
									  { .element_count = block_count * block_count, .data_type = data_types::f32 },
									  { .element_count = block_count * rope_dimension_count * attention_head_count_kv, .data_type = data_types::f16 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "MUL_MAT_ATTN_OUTPUT",
			.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "ADD_RMS_NORM_MUL_MAT_SILU",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 }, { .element_count = embedding_length, .data_type = data_types::f32 },
									  { .element_count = embedding_length * feed_forward_length, .data_type = data_types::q8_0 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "ADD_RMS_NORM_MUL_MAT",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 }, { .element_count = embedding_length, .data_type = data_types::f32 },
									  { .element_count = embedding_length * feed_forward_length, .data_type = data_types::q8_0 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::q8_0 } });

		ops.emplace_back(tensor_op{ .name = "MUL_MUL_MAT_ADD",
			.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = data_types::q8_0 },
									  { .element_count = feed_forward_length * seq_length, .data_type = data_types::q8_0 },
									  { .element_count = feed_forward_length * embedding_length, .data_type = data_types::q8_0 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::q8_0 } });
	}

	ops.emplace_back(tensor_op{ .name = "result_output_composite",
		.inputs						  = { { .element_count = embedding_length, .data_type = data_types::f32 }, { .element_count = embedding_length, .data_type = data_types::f32 },
								  { .element_count = embedding_length * vocab_size, .data_type = data_types::f32 } },
		.output						  = { .element_count = seq_length, .data_type = data_types::i32 } });

	return ops;
}

int32_t main(int32_t argc, char** argv) {
	auto result = get_read_writes(create_layer_tensor_ops_with_seqlen());
	std::cout << "READ BYTES: " << result.read_bytes << std::endl;
	std::cout << "WRITTEN BYTES: " << result.written_bytes << std::endl;
	auto result02 = get_read_writes(create_optimized_layer_tensor_ops());
	std::cout << "READ BYTES: " << result02.read_bytes << std::endl;
	std::cout << "WRITTEN BYTES: " << result02.written_bytes << std::endl;
	constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama);
	nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new->process_input(cli_args.prompt)) {
	}
	return 0;
};