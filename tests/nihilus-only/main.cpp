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
	const std::string_view name{};
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

template<uint64_t seq_length> std::vector<tensor_op> create_original_llama_cpp_layer_tensor_ops_with_seqlen() {
	constexpr uint32_t embedding_length			= 4096;
	constexpr uint32_t vocab_size				= 128256;
	constexpr uint32_t feed_forward_length		= 14336;
	constexpr uint32_t attention_head_count		= 32;
	constexpr uint32_t block_count				= 32;
	constexpr uint32_t attention_head_count_kv	= 8;
	constexpr uint32_t rope_dimension_count		= 128;
	constexpr uint64_t n_embd_kv_gqa			= rope_dimension_count * attention_head_count_kv;
	constexpr uint64_t total_cache_size_k		= seq_length * n_embd_kv_gqa;
	constexpr uint64_t total_cache_size_v		= seq_length * n_embd_kv_gqa;
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
	constexpr uint64_t cache_k_l0_elements		= total_cache_size_k;
	constexpr uint64_t cache_v_l0_elements		= total_cache_size_v;
	constexpr uint64_t inp_tokens_elements		= 2ULL;
	constexpr uint64_t inp_pos_elements			= 2ULL;
	constexpr uint64_t rope_freqs_elements		= rope_dimension_count / 2;
	constexpr uint64_t kq_mask_elements			= attention_head_count * attention_head_count;
	constexpr uint64_t output_norm_elements		= embedding_length;
	constexpr uint64_t inp_out_ids_elements		= 1ULL;
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
			.inputs						  = { { .element_count = total_cache_size_k, .data_type = data_types::f16 } },
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
			.inputs						  = { { .element_count = total_cache_size_v, .data_type = data_types::f16 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });
			*/
		ops.emplace_back(tensor_op{ .name = "v_cache_view-0 (copy of Vcur-0 (transposed))",
			.inputs						  = { { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f32 },
									  { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } },
			.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = data_types::f16 } });
		/*
		ops.emplace_back(tensor_op{ .name = "v-0",
			.inputs						  = { { .element_count = total_cache_size_v, .data_type = data_types::f16 } },
			.output						  = { .element_count = attention_head_count * rope_dimension_count * attention_head_count_kv, .data_type = data_types::f16 } });
			
		ops.emplace_back(tensor_op{ .name = "k-0",
			.inputs						  = { { .element_count = total_cache_size_k, .data_type = data_types::f16 } },
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

template<uint64_t seq_length> std::vector<tensor_op> create_mega_pipeline_layer_tensor_ops_with_seqlen() {
	constexpr uint32_t embedding_length			= 4096;
	constexpr uint32_t vocab_size				= 128256;
	constexpr uint32_t feed_forward_length		= 14336;
	constexpr uint32_t attention_head_count		= 32;
	constexpr uint32_t block_count				= 32;
	constexpr uint32_t attention_head_count_kv	= 8;
	constexpr uint32_t rope_dimension_count		= 128;
	constexpr uint64_t n_embd_kv_gqa			= rope_dimension_count * attention_head_count_kv;
	constexpr uint64_t total_cache_size_k		= seq_length * n_embd_kv_gqa;
	constexpr uint64_t total_cache_size_v		= seq_length * n_embd_kv_gqa;
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
	constexpr uint64_t cache_k_l0_elements		= total_cache_size_k;
	constexpr uint64_t cache_v_l0_elements		= total_cache_size_v;
	constexpr uint64_t inp_tokens_elements		= 2ULL;
	constexpr uint64_t inp_pos_elements			= 2ULL;
	constexpr uint64_t rope_freqs_elements		= rope_dimension_count / 2;
	constexpr uint64_t kq_mask_elements			= attention_head_count * attention_head_count;
	constexpr uint64_t output_norm_elements		= embedding_length;
	constexpr uint64_t inp_out_ids_elements		= 1ULL;
	std::vector<tensor_op> ops;

	// --- Token embeddings ----------------------------------------------------
	ops.emplace_back(tensor_op{
        .name   = "token_embeddings/GET_ROWS",
        .inputs = {
            { .element_count = embedding_length * vocab_size, .data_type = data_types::q8_0 }, // token_embd.weight
            { .element_count = seq_length,                   .data_type = data_types::i32  }  // inp_tokens
        },
        .output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } // inp_embd
    });

	// --- Per block / layer ---------------------------------------------------
	for (uint64_t x = 0; x < block_count; ++x) {
		// MEGA_QKV_PREP: {RMS_NORM + scale(attn_norm.weight)} → {Q,K,V} GEMMs → ROPE(Q,K) →
		//                direct writes to K/V caches (fp16 pack on store) → Q_rope_out (f32)
		ops.emplace_back(tensor_op{
            .name   = "blk." + std::to_string(x) + "/MEGA_QKV_PREP_AND_CACHE",
            .inputs = {
                { .element_count = embedding_length * seq_length,          .data_type = data_types::f32  }, // inp_embd
                { .element_count = embedding_length,                       .data_type = data_types::f32  }, // attn_norm.weight
                { .element_count = embedding_length * embedding_length,    .data_type = data_types::q8_0 }, // attn_q.weight
                { .element_count = n_embd_kv_gqa * embedding_length,       .data_type = data_types::q8_0 }, // attn_k.weight
                { .element_count = n_embd_kv_gqa * embedding_length,       .data_type = data_types::q8_0 }, // attn_v.weight
                { .element_count = seq_length,                              .data_type = data_types::i32  }, // inp_pos
                { .element_count = (rope_dimension_count / 2),             .data_type = data_types::f32  }, // rope_freqs.weight
                { .element_count = total_cache_size_k,                      .data_type = data_types::f16  }, // cache_k_l0 (dest view)
                { .element_count = total_cache_size_v,                      .data_type = data_types::f16  }  // cache_v_l0 (dest view)
            },
            // Q_rope_out [rope_dim * n_heads * seq_length] (f32)
            .output = { .element_count = rope_dimension_count * attention_head_count * seq_length,
                        .data_type     = data_types::f32 }
        });

		// BARRIERs: publish K then V (real graph fences for shared state)
		ops.emplace_back(tensor_op{
			.name	= "blk." + std::to_string(x) + "/BARRIER_publish_K_cache",
			.inputs = {},
			.output = { .element_count = 0, .data_type = data_types::f32 }// sentinel
		});
		ops.emplace_back(tensor_op{
			.name	= "blk." + std::to_string(x) + "/BARRIER_publish_V_cache",
			.inputs = {},
			.output = { .element_count = 0, .data_type = data_types::f32 }// sentinel
		});

		// MEGA_ATTENTION_APPLY: Flash-style KQ + masked softmax (streaming) + V*softmax
		//                       + head merge (indexing) + attn_output GEMM + residual add
		ops.emplace_back(tensor_op{
            .name   = "blk." + std::to_string(x) + "/MEGA_ATTENTION_APPLY",
            .inputs = {
                // Q_rope_out
                { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = data_types::f32  },
                // K & V cache views
                { .element_count = total_cache_size_k,                                       .data_type = data_types::f16  },
                { .element_count = total_cache_size_v,                                       .data_type = data_types::f16  },
                // KQ mask (context_length × seq_length) — keep as you model it
                { .element_count = block_count * block_count,                                .data_type = data_types::f32  },
                // attn_output.weight
                { .element_count = embedding_length * embedding_length,                      .data_type = data_types::q8_0 },
                // residual (inp_embd)
                { .element_count = embedding_length * seq_length,                            .data_type = data_types::f32  }
            },
            // ffn_inp [embedding_length * seq_length] (f32)
            .output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 }
        });

		// MEGA_FFN: {ADD (residual is already inside prev) + RMS_NORM + scale(ffn_norm.weight)}
		//           → {gate, up} GEMMs → SiLU → pointwise mul → down GEMM → residual add
		ops.emplace_back(tensor_op{
            .name   = "blk." + std::to_string(x) + "/MEGA_FFN",
            .inputs = {
                { .element_count = embedding_length * seq_length,          .data_type = data_types::f32  }, // ffn_inp
                { .element_count = embedding_length,                       .data_type = data_types::f32  }, // ffn_norm.weight
                { .element_count = feed_forward_length * embedding_length, .data_type = data_types::q8_0 }, // ffn_gate.weight
                { .element_count = feed_forward_length * embedding_length, .data_type = data_types::q8_0 }, // ffn_up.weight
                { .element_count = embedding_length * feed_forward_length, .data_type = data_types::q8_0 }  // ffn_down.weight
            },
            // l_out [embedding_length * seq_length] (f32)
            .output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 }
        });
	}

	// --- Final norm + IN-KERNEL SAMPLING (last-token) ------------------------
	ops.emplace_back(tensor_op{
        .name   = "final/MEGA_FINAL_NORM_SAMPLE_TOPK_LAST",
        .inputs = {
            // last position of l_out is selected internally by index math
            { .element_count = embedding_length,               .data_type = data_types::f32  }, // l_out (selected pos)
            { .element_count = embedding_length,               .data_type = data_types::f32  }, // output_norm.weight
            { .element_count = vocab_size * embedding_length,  .data_type = data_types::q8_0 }, // output.weight

            // Sampler params
            { .element_count = 1, .data_type = data_types::f32  }, // temperature
            { .element_count = 1, .data_type = data_types::i32  }, // top_k
            { .element_count = 1, .data_type = data_types::f32  }, // top_p
            { .element_count = 1, .data_type = data_types::f32  }, // repetition_penalty
            { .element_count = 1, .data_type = data_types::f32  }, // presence_penalty
            { .element_count = 1, .data_type = data_types::f32  }, // frequency_penalty
            { .element_count = 1, .data_type = data_types::i32  }, // rep_window

            // Token history view (for penalties)
            { .element_count = seq_length, .data_type = data_types::i32  }, // token_history

            // Optional: bias/mask
            { .element_count = vocab_size, .data_type = data_types::f32  }, // logits_bias (optional; can be zero-length if unused)
            { .element_count = vocab_size, .data_type = data_types::i8   }, // allowed_vocab_mask (optional)

            // RNG state (e.g., xoroshiro128**: 2x u64)
            { .element_count = 2, .data_type = data_types::i64 }            // rng_state
        },
        // chosen token id for the last position
        .output = { .element_count = 1, .data_type = data_types::i32 } // result_token_id
    });

	// Optional: publish token if another thread consumes it concurrently
	// ops.emplace_back(tensor_op{ .name="final/BARRIER_publish_token", .inputs={}, .output={0, data_types::f32} });

	return ops;
}

int32_t main(int32_t argc, char** argv) {
	auto result = get_read_writes(create_original_llama_cpp_layer_tensor_ops_with_seqlen<32>());
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(32) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	auto result02 = get_read_writes(create_mega_pipeline_layer_tensor_ops_with_seqlen<32>());
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes(create_original_llama_cpp_layer_tensor_ops_with_seqlen<1024>());
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(1024) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes(create_mega_pipeline_layer_tensor_ops_with_seqlen<1024>());
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes(create_original_llama_cpp_layer_tensor_ops_with_seqlen<2048>());
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(2048) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes(create_mega_pipeline_layer_tensor_ops_with_seqlen<2048>());
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes(create_original_llama_cpp_layer_tensor_ops_with_seqlen<131072>());
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(131072) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes(create_mega_pipeline_layer_tensor_ops_with_seqlen<131072>());
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	constexpr auto model_config_00 =
		generate_model_config(model_generations::v3_1, model_sizes::llm_8B, kernel_type_profiles::q8_gqa, model_arches::llama, nihilus::device_types::cpu, true, false, 131072);
	constexpr auto model_config_01 = update_model_config_benchmark(model_config_00, true);
	cli_params cli_args			   = harbinger<model_config_01>::parse_cli_arguments(argc, argv);
	auto model_new_01{ harbinger<model_config_01>::parse_model_graph_data(cli_args) };
	while (model_new_01->process_input(cli_args.prompt)) {
	}
	return 0;
};