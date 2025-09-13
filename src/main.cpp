#include <nihilus>

using namespace nihilus;

static uint64_t get_data_size(uint64_t element_count, data_types type) {
	switch (static_cast<uint64_t>(type)) {
		case static_cast<uint64_t>(data_types::f16): {
			return element_count * 2;
		}
		case static_cast<uint64_t>(data_types::f32): {
			return element_count * 4;
		}
		case static_cast<uint64_t>(data_types::f64): {
			return element_count * 8;
		}
		case static_cast<uint64_t>(data_types::i32): {
			return element_count * 4;
		}
		case static_cast<uint64_t>(data_types::i64): {
			return element_count * 8;
		}
		case static_cast<uint64_t>(data_types::i8): {
			return element_count * 1;
		}
		case static_cast<uint64_t>(data_types::q8_0): {
			return (element_count + 31) / 32 * 34;
		}
		case static_cast<uint64_t>(data_types::bf16): {
			return (element_count + 31) / 32 * 34;
		}
		case static_cast<uint64_t>(data_types::i16): {
			return (element_count + 31) / 32 * 34;
		}
		case static_cast<uint64_t>(data_types::count): {
			return (element_count + 31) / 32 * 34;
		}
		default: {
			return 1;
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


static read_write get_read_writes(std::vector<tensor_op> inputs) {
	read_write return_values{};
	for (auto& value: inputs) {
		for (auto& value_new: value.inputs) {
			return_values.read_bytes += get_data_size(value_new.element_count, value_new.data_type);
		}
		return_values.written_bytes += get_data_size(value.output.element_count, value.output.data_type);
	}
	return return_values;
}

template<uint64_t seq_length> static std::vector<tensor_op> create_original_llama_cpp_layer_tensor_ops_with_seqlen() {
	constexpr uint32_t embedding_length		   = 16384;
	constexpr uint32_t vocab_size			   = 128256;
	constexpr uint32_t feed_forward_length	   = 53248;
	constexpr uint32_t attention_head_count	   = 128;
	constexpr uint32_t block_count			   = 126;
	constexpr uint32_t attention_head_count_kv = 8;
	constexpr uint32_t rope_dimension_count	   = embedding_length / attention_head_count;
	constexpr uint64_t n_embd_kv_gqa		   = rope_dimension_count * attention_head_count_kv;
	std::vector<tensor_op> ops;

	ops.emplace_back(tensor_op{ .name = "inp_embd",
		.inputs = { { .element_count = embedding_length * vocab_size, .data_type = data_types::i16 }, { .element_count = seq_length, .data_type = data_types::i32 } },
		.output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

	for (uint64_t x = 0; x < block_count; ++x) {
		ops.emplace_back(tensor_op{ .name = "norm-0",
			.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "attn_norm-0",
			.inputs = { { .element_count = embedding_length * seq_length, .data_type = data_types::f32 }, { .element_count = embedding_length, .data_type = data_types::f32 } },
			.output = { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "Qcur-0",
			.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = data_types::i16 },
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
			.inputs						  = { { .element_count = embedding_length * n_embd_kv_gqa, .data_type = data_types::i16 },
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
			.inputs						  = { { .element_count = embedding_length * n_embd_kv_gqa, .data_type = data_types::i16 },
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
			.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = data_types::i16 },
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
			.inputs						  = { { .element_count = embedding_length * feed_forward_length, .data_type = data_types::i16 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_silu-0",
			.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_up-0",
			.inputs						  = { { .element_count = embedding_length * feed_forward_length, .data_type = data_types::i16 },
									  { .element_count = embedding_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_gate_par-0",
			.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 },
									  { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } },
			.output						  = { .element_count = feed_forward_length * seq_length, .data_type = data_types::f32 } });

		ops.emplace_back(tensor_op{ .name = "ffn_out-0",
			.inputs						  = { { .element_count = feed_forward_length * embedding_length, .data_type = data_types::i16 },
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
		.inputs = { { .element_count = embedding_length * vocab_size, .data_type = data_types::i16 }, { .element_count = embedding_length, .data_type = data_types::f32 } },
		.output = { .element_count = vocab_size, .data_type = data_types::f32 } });

	return ops;
}

template<uint64_t seq_length> static std::vector<tensor_op> create_mega_pipeline_layer_tensor_ops_with_seqlen() {
	constexpr uint32_t embedding_length		   = 16384;
	constexpr uint32_t vocab_size			   = 128256;
	constexpr uint32_t feed_forward_length	   = 53248;
	constexpr uint32_t attention_head_count	   = 128;
	constexpr uint32_t block_count			   = 126;
	constexpr uint32_t attention_head_count_kv = 8;
	constexpr uint32_t rope_dimension_count	   = embedding_length / attention_head_count;
	constexpr uint64_t n_embd_kv_gqa		   = rope_dimension_count * attention_head_count_kv;
	constexpr uint64_t total_cache_size_k	   = seq_length * n_embd_kv_gqa;
	constexpr uint64_t total_cache_size_v	   = seq_length * n_embd_kv_gqa;

	std::vector<tensor_op> ops;

	// --- Token embeddings ----------------------------------------------------
	ops.emplace_back(tensor_op{
        .name   = "token_embeddings/GET_ROWS",
        .inputs = {
            { .element_count = embedding_length * vocab_size, .data_type = data_types::i16 }, // token_embd.weight
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
                { .element_count = embedding_length * embedding_length,    .data_type = data_types::i16 }, // attn_q.weight
                { .element_count = n_embd_kv_gqa * embedding_length,       .data_type = data_types::i16 }, // attn_k.weight
                { .element_count = n_embd_kv_gqa * embedding_length,       .data_type = data_types::i16 }, // attn_v.weight
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
                { .element_count = embedding_length * embedding_length,                      .data_type = data_types::i16 },
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
                { .element_count = feed_forward_length * embedding_length, .data_type = data_types::i16 }, // ffn_gate.weight
                { .element_count = feed_forward_length * embedding_length, .data_type = data_types::i16 }, // ffn_up.weight
                { .element_count = embedding_length * feed_forward_length, .data_type = data_types::i16 }  // ffn_down.weight
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
            { .element_count = vocab_size * embedding_length,  .data_type = data_types::i16 }, // output.weight

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

template<typename value_type> constexpr uint64_t popcount(value_type value) noexcept {
	uint64_t count = 0;
	while (value != 0) {
		value &= (value - 1);
		count++;
	}
	return count;
}

enum class dim_trait_static_assert_errors : uint8_t {
	reshape_total_element_count_mismatch,
	view_total_element_count_mismatch,
	transpose_total_element_count_mismatch,
	transpose_dimension_0_mismatch,
	transpose_dimension_1_mismatch,
	permute_total_element_count_mismatch,
};

template<uint64_t... runtime_mask> constexpr uint64_t get_runtime_mask() {
	static_assert(((runtime_mask < 4) && ...), "Sorry, but you can only define one of the first 4 dimensions as runtime mutable!");
	return ((1ULL << runtime_mask) | ...);
}

template<kernel_types kernel_type_new> struct kernel_types_type {
	static constexpr kernel_types kernel_type{ kernel_type_new };
};

template<typename value_type>
concept preserved_dimensions_kernel_types = detail::remove_cvref_t<value_type>::kernel_type == kernel_types::add ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::mul || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::sub ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::rms_norm || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::silu ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::softmax || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::rope ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::copy || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::top_k_filter ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::top_p_filter || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::repetition_penalty ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::presence_penalty || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::frequency_penalty ||
	detail::remove_cvref_t<value_type>::kernel_type == kernel_types::temperature_scale || detail::remove_cvref_t<value_type>::kernel_type == kernel_types::vocab_mask;

template<typename value_type>
concept kernel_types_types = requires() { detail::remove_cvref_t<value_type>::kernel_type; };

template<typename value_type>
concept weights_kernel_types = detail::remove_cvref_t<value_type>::kernel_type == kernel_types::weights && kernel_types_types<value_type>;

template<integral_or_enum_types auto index> struct tag_new : public std::integral_constant<uint64_t, static_cast<uint64_t>(index)> {};

template<uint64_t runtime_mask_new, uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> struct kernel_dims {
	static constexpr array<uint64_t, 4> dims{ dim_00, dim_01, dim_02, dim_03 };
	static constexpr uint64_t runtime_mask{ runtime_mask_new };
	static constexpr uint64_t runtime_dimension_count{ popcount(runtime_mask) };
	mutable uint64_t rt_dims[4]{ dim_00, dim_01, dim_02, dim_03 };

	template<typename IndexTag> NIHILUS_HOST constexpr uint64_t& operator[](IndexTag index_tag) {
		constexpr uint64_t index = IndexTag::value;
		static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
		static_assert(runtime_mask & (1ULL << index), "Error: Index is not enabled by the runtime_mask and cannot be modified at runtime!");
		return rt_dims[index_tag.value];
	}

	template<typename IndexTag> NIHILUS_HOST constexpr uint64_t operator[](IndexTag index_tag) const {
		constexpr uint64_t index = IndexTag::value;
		static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
		return rt_dims[index_tag.value];
	}
};

template<typename value_type>
concept kernel_dims_types = requires() {
	detail::remove_cvref_t<value_type>::runtime_mask;
	detail::remove_cvref_t<value_type>::rt_dims;
	detail::remove_cvref_t<value_type>::dims;
};

template<typename... value_types> struct get_first_type;

template<typename value_type, typename... value_types> struct get_first_type<value_type, value_types...> {
	using type = value_type;
};

template<typename value_type> struct get_first_type<value_type> {
	using type = value_type;
};

template<typename... value_types> using get_first_type_t = get_first_type<value_types...>::type;

template<bool batched, typename kernel_type, typename... dims_types> struct dim_traits;

template<bool batched, kernel_dims_types input_dims_01> struct dim_traits<batched, kernel_types_type<kernel_types::weights>, input_dims_01> {
	using dims_type = kernel_dims<input_dims_01::runtime_mask, input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>;
};

template<bool batched, preserved_dimensions_kernel_types kernel_type, typename... dims_types> struct dim_traits<batched, kernel_type, dims_types...> {
	using first_type = get_first_type_t<dims_types...>;
	using dims_type	 = kernel_dims<first_type::runtime_mask, first_type::dims[0], first_type::dims[1], first_type::dims[2], first_type::dims[3]>;
};

template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims> struct dim_traits<batched, kernel_types_type<kernel_types::reshape>, output_dims, input_dims> {
	static constexpr auto dims01			= input_dims::dims;
	static constexpr auto dims02			= output_dims::dims;
	static constexpr size_t input_elements	= compute_elements(dims01, input_dims::runtime_mask);
	static constexpr size_t output_elements = compute_elements(dims02, output_dims::runtime_mask);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::reshape_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims> struct dim_traits<batched, kernel_types_type<kernel_types::view>, output_dims, input_dims> {
	static constexpr auto dims01			= input_dims::dims;
	static constexpr auto dims02			= output_dims::dims;
	static constexpr size_t input_elements	= compute_elements(dims01, input_dims::runtime_mask);
	static constexpr size_t output_elements = compute_elements(dims02, output_dims::runtime_mask);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::view_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims>
struct dim_traits<batched, kernel_types_type<kernel_types::transpose>, output_dims, input_dims> {
	static constexpr auto dims01			= input_dims::dims;
	static constexpr auto dims02			= output_dims::dims;
	static constexpr size_t input_elements	= compute_elements(dims01, input_dims::runtime_mask);
	static constexpr size_t output_elements = compute_elements(dims02, output_dims::runtime_mask);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::transpose_total_element_count_mismatch, input_elements, output_elements>::impl);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || dims01[0] == dims02[1]),
		dim_trait_static_assert_errors::transpose_dimension_0_mismatch, dims01[0], dims02[1]>::impl);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || dims01[1] == dims02[0]),
		dim_trait_static_assert_errors::transpose_dimension_1_mismatch, dims01[1], dims02[0]>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<bool batched, kernel_dims_types output_dims, kernel_dims_types input_dims> struct dim_traits<batched, kernel_types_type<kernel_types::permute>, output_dims, input_dims> {
	static constexpr auto dims01			= input_dims::dims;
	static constexpr auto dims02			= output_dims::dims;
	static constexpr size_t input_elements	= compute_elements(dims01, input_dims::runtime_mask);
	static constexpr size_t output_elements = compute_elements(dims02, output_dims::runtime_mask);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::permute_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<bool batched, kernel_dims_types input_dims_01, kernel_dims_types input_dims_02>
struct dim_traits<batched, kernel_types_type<kernel_types::mul_mat>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dims;
	static constexpr auto dims02 = input_dims_02::dims;
	using dims_type				 = kernel_dims<input_dims_01::runtime_mask, dims02[0], dims01[1], (batched ? dims01[2] : 1), (batched ? dims01[3] : 1)>;
};

template<bool batched, kernel_dims_types input_dims_01, kernel_dims_types input_dims_02>
struct dim_traits<batched, kernel_types_type<kernel_types::get_rows>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dims;
	static constexpr auto dims02 = input_dims_02::dims;
	using dims_type				 = kernel_dims<input_dims_01::runtime_mask, dims01[0], dims02[1], (batched ? dims01[2] : 1), (batched ? dims01[3] : 1)>;
};

template<bool batched, kernel_types_types kernel_type_new, typename... input_types_new> struct kernel_traits;

template<bool batched, weights_kernel_types kernel_type_new, typename output_type_new, kernel_dims_types dims_type>
struct kernel_traits<batched, kernel_type_new, output_type_new, dims_type> : public dim_traits<batched, kernel_type_new, dims_type>::dims_type {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	using output_type = output_type_new;
};

template<bool batched, enum_types auto enum_value_new, weights_kernel_types kernel_type_new, typename output_type_new, typename... input_types_new> struct op_traits;

template<bool batched, enum_types auto enum_value_new, weights_kernel_types kernel_type_new, typename output_type_new, kernel_dims_types dims_type>
struct op_traits<batched, enum_value_new, kernel_type_new, output_type_new, dims_type> : dims_type {
	static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	using output_type = output_type_new;
	static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::dims) };
};

template<typename config_type_new, core_types> struct core_traits;

template<typename config_type_new> struct core_traits<config_type_new, core_types::weights>
	: public core_elem_base<core_types::weights, core_traits<config_type_new, core_types::weights>> {
	static constexpr core_types core_type{ core_types::weights };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
	using config_type		  = config_type_new;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using weight_type		  = typename kernel_profile_type::weight_type;
	using norm_type			  = typename kernel_profile_type::norm_type;

	using attn_q_weight_traits = op_traits<config_type::batched_processing, weight_types::attn_q, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::embedding_length, 1, 1>>;

	using attn_k_weight_traits = op_traits<config_type::batched_processing, weight_types::attn_k, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::n_embd_kv_gqa, 1, 1>>;

	using attn_v_weight_traits = op_traits<config_type::batched_processing, weight_types::attn_v, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::n_embd_kv_gqa, 1, 1>>;

	using attn_output_weight_traits = op_traits<config_type::batched_processing, weight_types::attn_output, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::embedding_length, 1, 1>>;

	using attn_norm_weight_traits = op_traits<config_type::batched_processing, weight_types::attn_norm, kernel_types_type<kernel_types::weights>, norm_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, 1, 1, 1>>;

	using ffn_gate_weight_traits = op_traits<config_type::batched_processing, weight_types::ffn_gate, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::feed_forward_length, 1, 1>>;

	using ffn_up_weight_traits = op_traits<config_type::batched_processing, weight_types::ffn_up, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::feed_forward_length, 1, 1>>;

	using ffn_down_weight_traits = op_traits<config_type::batched_processing, weight_types::ffn_down, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::feed_forward_length, model_traits_type<config_type>::embedding_length, 1, 1>>;

	using ffn_norm_weight_traits = op_traits<config_type::batched_processing, weight_types::ffn_norm, kernel_types_type<kernel_types::weights>, norm_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, 1, 1, 1>>;

	using token_embd_weight_traits = op_traits<config_type::batched_processing, weight_types::token_embd, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::vocab_size, 1, 1>>;

	using rope_freqs_weight_traits = op_traits<config_type::batched_processing, weight_types::rope_freqs, kernel_types_type<kernel_types::weights>, norm_type,
		kernel_dims<0, model_traits_type<config_type>::rope_dimension_count / 2, 1, 1, 1>>;

	using output_norm_weight_traits = op_traits<config_type::batched_processing, weight_types::output_norm, kernel_types_type<kernel_types::weights>, norm_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, 1, 1, 1>>;

	using output_weight_traits = op_traits<config_type::batched_processing, weight_types::output, kernel_types_type<kernel_types::weights>, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::vocab_size, 1, 1>>;

	using composite_ops = get_core_base_t<config_type, attn_q_weight_traits, attn_k_weight_traits, attn_v_weight_traits, attn_output_weight_traits, attn_norm_weight_traits,
		ffn_gate_weight_traits, ffn_up_weight_traits, ffn_down_weight_traits, ffn_norm_weight_traits, token_embd_weight_traits, rope_freqs_weight_traits, output_norm_weight_traits,
		output_weight_traits>;
	composite_ops values{};

	static constexpr uint64_t total_required_bytes{ attn_q_weight_traits::total_required_bytes + attn_k_weight_traits::total_required_bytes +
		attn_v_weight_traits::total_required_bytes + attn_output_weight_traits::total_required_bytes + attn_norm_weight_traits::total_required_bytes +
		ffn_gate_weight_traits::total_required_bytes + ffn_up_weight_traits::total_required_bytes + ffn_down_weight_traits::total_required_bytes +
		ffn_norm_weight_traits::total_required_bytes + token_embd_weight_traits::total_required_bytes + rope_freqs_weight_traits::total_required_bytes +
		output_norm_weight_traits::total_required_bytes + output_weight_traits::total_required_bytes };

	static constexpr bool has_total_required_bytes{ config_type::device_type == device_types::gpu };
};

template<typename config_type, core_types core_type> struct core_traits {};

int32_t main(int32_t argc, char** argv) {
	try {
		[[maybe_unused]] op_traits<true, weight_types::attn_k, kernel_types_type<kernel_types::weights>, kernel_type_profile_traits<kernel_type_profiles::q8_gqa>::weight_type,
			kernel_dims<0, 64, 64, 1, 1>> attn_k{};
		static constexpr model_config config{};
		[[maybe_unused]] core_traits<model_config_type<config>, core_types::weights> core_traits{};

		[[maybe_unused]] auto dims = op_traits<true, weight_types::attn_q, kernel_types_type<kernel_types::weights>,
			kernel_type_profile_traits<kernel_type_profiles::fp16_mha>::weight_type, kernel_dims<0, 64, 64, 1, 1>>::dims;
		//std::cout << "TOTAL REQUIREDS BYTES: " << attn_q.total_required_bytes_new << std::endl;

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
		static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::batched_processing_type::enabled, nihilus::model_generations::v3_1,
			nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, nihilus::device_types::cpu, nihilus::exception_type::enabled,
			nihilus::default_max_sequence_length_type{ 1024 }, nihilus::benchmark_type::enabled);
		cli_params cli_args					  = harbinger::parse_cli_arguments(argc, argv);
		nihilus::model_collection_type<model_config_00> collection{ cli_args };
		collection.process_input(cli_args.prompt);
	} catch (const std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}
