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
		case static_cast<uint64_t>(data_types::i8): {
			return element_count * 1;
		}
		case static_cast<uint64_t>(data_types::i16): {
			return element_count * 2;
		}
		case static_cast<uint64_t>(data_types::i32): {
			return element_count * 4;
		}
		case static_cast<uint64_t>(data_types::i64): {
			return element_count * 8;
		}
		case static_cast<uint64_t>(data_types::q8_0): {
			return (element_count + 31) / 32 * 34;
		}
		case static_cast<uint64_t>(data_types::bf16): {
			return element_count * 2;
		}
		case static_cast<uint64_t>(data_types::count): {
			return 0;
		}
		default: {
			return 0;
		}
	}
}

enum class libraries {
	nihilus,
	llama,
};

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
	std::string model_name{};
	uint64_t written_bytes{};
	uint64_t read_bytes{};
};

template<libraries library, kernel_type_profiles kernel_type_profile, model_arches model_arch, model_sizes model_size, model_generations model_generation, uint64_t seq_length>
struct create_tensor_ops;

template<model_arches model_arch, kernel_type_profiles kernel_type_profile, model_sizes model_size, model_generations model_generation, uint64_t seq_length>
struct create_tensor_ops<libraries::llama, kernel_type_profile, model_arch, model_size, model_generation, seq_length> {
	static std::vector<tensor_op> impl() {
		constexpr uint32_t embedding_length		   = model_traits<model_arch, model_size, model_generation>::embedding_length;
		constexpr uint32_t vocab_size			   = model_traits<model_arch, model_size, model_generation>::vocab_size;
		constexpr uint32_t feed_forward_length	   = model_traits<model_arch, model_size, model_generation>::feed_forward_length;
		constexpr uint32_t attention_head_count	   = model_traits<model_arch, model_size, model_generation>::attention_head_count;
		constexpr uint32_t block_count			   = model_traits<model_arch, model_size, model_generation>::block_count;
		constexpr uint32_t attention_head_count_kv = model_traits<model_arch, model_size, model_generation>::attention_head_count_kv;
		constexpr uint32_t rope_dimension_count	   = model_traits<model_arch, model_size, model_generation>::rope_dimension_count;
		constexpr uint64_t n_embd_kv_gqa		   = model_traits<model_arch, model_size, model_generation>::n_embd_kv_gqa;
		std::vector<tensor_op> ops;
		static constexpr data_types weight_type	  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::weight_type>::data_type;
		static constexpr data_types index_type	  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::index_type>::data_type;
		static constexpr data_types compute_type  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::compute_type>::data_type;
		static constexpr data_types kv_cache_type = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::kv_cache_type>::data_type;

		ops.emplace_back(tensor_op{ .name = "inp_embd",
			.inputs = { { .element_count = embedding_length * vocab_size, .data_type = weight_type }, { .element_count = seq_length, .data_type = index_type } },
			.output = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

		for (uint64_t x = 0; x < block_count; ++x) {
			ops.emplace_back(tensor_op{ .name = "norm-0",
				.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "attn_norm-0",
				.inputs = { { .element_count = embedding_length * seq_length, .data_type = compute_type }, { .element_count = embedding_length, .data_type = compute_type } },
				.output = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "Qcur-0",
				.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = weight_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });
			/*
ops.emplace_back(tensor_op{ .name = "Qcur-0 (reshaped)",
	.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = compute_type } },
	.output						  = { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type } });
	*/
			ops.emplace_back(tensor_op{ .name = "Qcur-0",
				.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type },
										  { .element_count = seq_length, .data_type = index_type }, { .element_count = rope_dimension_count / 2, .data_type = compute_type } },
				.output						  = { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "Kcur-0",
				.inputs						  = { { .element_count = embedding_length * n_embd_kv_gqa, .data_type = weight_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = n_embd_kv_gqa * seq_length, .data_type = compute_type } });
			/*
ops.emplace_back(tensor_op{ .name = "Kcur-0 (reshaped)",
	.inputs						  = { { .element_count = n_embd_kv_gqa * seq_length, .data_type = compute_type } },
	.output						  = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = compute_type } });
	*/
			ops.emplace_back(tensor_op{ .name = "Kcur-0",
				.inputs						  = { { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = compute_type },
										  { .element_count = seq_length, .data_type = index_type }, { .element_count = rope_dimension_count / 2, .data_type = compute_type } },
				.output						  = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "Vcur-0",
				.inputs						  = { { .element_count = embedding_length * n_embd_kv_gqa, .data_type = weight_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = n_embd_kv_gqa * seq_length, .data_type = compute_type } });
			/*
ops.emplace_back(tensor_op{ .name = "k_cache_view-0",
	.inputs						  = { { .element_count = total_cache_size_k, .data_type = kv_cache_type } },
	.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = kv_cache_type } });
	*/
			ops.emplace_back(tensor_op{ .name = "k_cache_view-0 (copy of Kcur-0)",
				.inputs						  = { { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = compute_type },
										  { .element_count = seq_length * n_embd_kv_gqa, .data_type = kv_cache_type } },
				.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = kv_cache_type } });
			/*
ops.emplace_back(tensor_op{ .name = "Vcur-0 (transposed)",
	.inputs						  = { { .element_count = n_embd_kv_gqa * seq_length, .data_type = compute_type } },
	.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = compute_type } });

ops.emplace_back(tensor_op{ .name = "v_cache_view-0",
	.inputs						  = { { .element_count = total_cache_size_v, .data_type = kv_cache_type } },
	.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = kv_cache_type } });
	*/
			ops.emplace_back(tensor_op{ .name = "v_cache_view-0 (copy of Vcur-0 (transposed))",
				.inputs						  = { { .element_count = seq_length * n_embd_kv_gqa, .data_type = compute_type },
										  { .element_count = seq_length * n_embd_kv_gqa, .data_type = kv_cache_type } },
				.output						  = { .element_count = seq_length * n_embd_kv_gqa, .data_type = kv_cache_type } });
			/*
ops.emplace_back(tensor_op{ .name = "v-0",
	.inputs						  = { { .element_count = total_cache_size_v, .data_type = kv_cache_type } },
	.output						  = { .element_count = attention_head_count * rope_dimension_count * attention_head_count_kv, .data_type = kv_cache_type } });
	
ops.emplace_back(tensor_op{ .name = "k-0",
	.inputs						  = { { .element_count = total_cache_size_k, .data_type = kv_cache_type } },
	.output						  = { .element_count = rope_dimension_count * attention_head_count * attention_head_count_kv, .data_type = kv_cache_type } });
	
ops.emplace_back(tensor_op{ .name = "q-0",
	.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type } },
	.output						  = { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = compute_type } });
	*/
			ops.emplace_back(tensor_op{ .name = "kq-0",
				.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * attention_head_count_kv, .data_type = kv_cache_type },
										  { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = compute_type } },
				.output						  = { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "kq_soft_max_ext-0",
				.inputs						  = { { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = compute_type },
										  { .element_count = attention_head_count * attention_head_count, .data_type = compute_type } },
				.output						  = { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "kqv-0",
				.inputs						  = { { .element_count = attention_head_count * rope_dimension_count * attention_head_count_kv, .data_type = kv_cache_type },
										  { .element_count = attention_head_count * seq_length * attention_head_count, .data_type = compute_type } },
				.output						  = { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = compute_type } });
			/*
ops.emplace_back(tensor_op{ .name = "kqv_merged-0",
	.inputs						  = { { .element_count = rope_dimension_count * seq_length * attention_head_count, .data_type = compute_type } },
	.output						  = { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type } });
	*/
			ops.emplace_back(tensor_op{ .name = "kqv_merged_cont-0",
				.inputs						  = { { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "kqv_out-0",
				.inputs						  = { { .element_count = embedding_length * embedding_length, .data_type = weight_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_inp-0",
				.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = compute_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "norm-0",
				.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_norm-0",
				.inputs = { { .element_count = embedding_length * seq_length, .data_type = compute_type }, { .element_count = embedding_length, .data_type = compute_type } },
				.output = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_gate-0",
				.inputs						  = { { .element_count = embedding_length * feed_forward_length, .data_type = weight_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = feed_forward_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_silu-0",
				.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = feed_forward_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_up-0",
				.inputs						  = { { .element_count = embedding_length * feed_forward_length, .data_type = weight_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = feed_forward_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_gate_par-0",
				.inputs						  = { { .element_count = feed_forward_length * seq_length, .data_type = compute_type },
										  { .element_count = feed_forward_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = feed_forward_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "ffn_out-0",
				.inputs						  = { { .element_count = feed_forward_length * embedding_length, .data_type = weight_type },
										  { .element_count = feed_forward_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });

			ops.emplace_back(tensor_op{ .name = "l_out-0",
				.inputs						  = { { .element_count = embedding_length * seq_length, .data_type = compute_type },
										  { .element_count = embedding_length * seq_length, .data_type = compute_type } },
				.output						  = { .element_count = embedding_length * seq_length, .data_type = compute_type } });
		}

		ops.emplace_back(tensor_op{ .name = "norm",
			.inputs						  = { { .element_count = embedding_length, .data_type = compute_type } },
			.output						  = { .element_count = embedding_length, .data_type = compute_type } });

		ops.emplace_back(tensor_op{ .name = "result_norm",
			.inputs						  = { { .element_count = embedding_length, .data_type = compute_type }, { .element_count = embedding_length, .data_type = compute_type } },
			.output						  = { .element_count = embedding_length, .data_type = compute_type } });

		ops.emplace_back(tensor_op{ .name = "result_output",
			.inputs = { { .element_count = embedding_length * vocab_size, .data_type = weight_type }, { .element_count = embedding_length, .data_type = compute_type } },
			.output = { .element_count = vocab_size, .data_type = compute_type } });

		return ops;
	}
};

template<model_arches model_arch, kernel_type_profiles kernel_type_profile, model_sizes model_size, model_generations model_generation, uint64_t seq_length>
struct create_tensor_ops<libraries::nihilus, kernel_type_profile, model_arch, model_size, model_generation, seq_length> {
	static std::vector<tensor_op> impl() {
		constexpr uint32_t embedding_length		   = model_traits<model_arch, model_size, model_generation>::embedding_length;
		constexpr uint32_t vocab_size			   = model_traits<model_arch, model_size, model_generation>::vocab_size;
		constexpr uint32_t feed_forward_length	   = model_traits<model_arch, model_size, model_generation>::feed_forward_length;
		constexpr uint32_t attention_head_count	   = model_traits<model_arch, model_size, model_generation>::attention_head_count;
		constexpr uint32_t block_count			   = model_traits<model_arch, model_size, model_generation>::block_count;
		constexpr uint32_t attention_head_count_kv = model_traits<model_arch, model_size, model_generation>::attention_head_count_kv;
		constexpr uint32_t rope_dimension_count	   = model_traits<model_arch, model_size, model_generation>::rope_dimension_count;
		constexpr uint64_t n_embd_kv_gqa		   = model_traits<model_arch, model_size, model_generation>::n_embd_kv_gqa;
		static constexpr data_types weight_type	   = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::weight_type>::data_type;
		static constexpr data_types index_type	   = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::index_type>::data_type;
		static constexpr data_types compute_type   = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::compute_type>::data_type;
		static constexpr data_types kv_cache_type  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::kv_cache_type>::data_type;
		constexpr uint64_t total_cache_size_k	   = seq_length * n_embd_kv_gqa;
		constexpr uint64_t total_cache_size_v	   = seq_length * n_embd_kv_gqa;
		std::vector<tensor_op> ops;

		// --- Token embeddings ----------------------------------------------------
		ops.emplace_back(tensor_op{
        .name   = "token_embeddings/GET_ROWS",
        .inputs = {
            { .element_count = embedding_length * vocab_size, .data_type = weight_type}, // token_embd.weight
            { .element_count = seq_length,                   .data_type = index_type}  // inp_tokens
        },
        .output = { .element_count = embedding_length * seq_length, .data_type = compute_type} // inp_embd
    });

		// --- Per block / layer ---------------------------------------------------
		for (uint64_t x = 0; x < block_count; ++x) {
			// MEGA_QKV_PREP: {RMS_NORM + scale(attn_norm.weight)} → {Q,K,V} GEMMs → ROPE(Q,K) →
			//                direct writes to K/V caches (fp16 pack on store) → Q_rope_out (f32)
			ops.emplace_back(tensor_op{
            .name   = "blk." + std::to_string(x) + "/MEGA_QKV_PREP_AND_CACHE",
            .inputs = {
                { .element_count = embedding_length * seq_length,          .data_type = compute_type}, // inp_embd
                { .element_count = embedding_length,                       .data_type = compute_type}, // attn_norm.weight
                { .element_count = embedding_length * embedding_length,    .data_type = weight_type }, // attn_q.weight
                { .element_count = n_embd_kv_gqa * embedding_length,       .data_type = weight_type}, // attn_k.weight
                { .element_count = n_embd_kv_gqa * embedding_length,       .data_type = weight_type}, // attn_v.weight
                { .element_count = seq_length,                              .data_type = index_type }, // inp_pos
                { .element_count = (rope_dimension_count / 2),             .data_type = compute_type }, // rope_freqs.weight
                { .element_count = total_cache_size_k,                      .data_type = kv_cache_type}, // cache_k_l0 (dest view)
                { .element_count = total_cache_size_v,                      .data_type = kv_cache_type}  // cache_v_l0 (dest view)
            },
            // Q_rope_out [rope_dim * n_heads * seq_length] (f32)
            .output = { .element_count = rope_dimension_count * attention_head_count * seq_length,
                        .data_type     = compute_type }
        });

			// BARRIERs: publish K then V (real graph fences for shared state)
			ops.emplace_back(tensor_op{
				.name	= "blk." + std::to_string(x) + "/BARRIER_publish_K_cache",
				.inputs = {},
				.output = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = kv_cache_type }// sentinel
			});
			ops.emplace_back(tensor_op{
				.name	= "blk." + std::to_string(x) + "/BARRIER_publish_V_cache",
				.inputs = {},
				.output = { .element_count = rope_dimension_count * attention_head_count_kv * seq_length, .data_type = kv_cache_type }// sentinel
			});

			// MEGA_ATTENTION_APPLY: Flash-style KQ + masked softmax (streaming) + V*softmax
			//                       + head merge (indexing) + attn_output GEMM + residual add
			ops.emplace_back(tensor_op{
            .name   = "blk." + std::to_string(x) + "/MEGA_ATTENTION_APPLY",
            .inputs = {
                // Q_rope_out
                { .element_count = rope_dimension_count * attention_head_count * seq_length, .data_type = compute_type  },
                // K & V cache views
                { .element_count = total_cache_size_k,                                       .data_type = kv_cache_type  },
                { .element_count = total_cache_size_v,                                       .data_type = kv_cache_type  },
                // KQ mask (context_length × seq_length) — keep as you model it
                { .element_count = block_count * block_count,                                .data_type = compute_type  },
                // attn_output.weight
                { .element_count = embedding_length * embedding_length,                      .data_type = weight_type },
                // residual (inp_embd)
                { .element_count = embedding_length * seq_length,                            .data_type = compute_type  }
            },
            // ffn_inp [embedding_length * seq_length] (f32)
            .output = { .element_count = embedding_length * seq_length, .data_type = compute_type }
        });

			// MEGA_FFN: {ADD (residual is already inside prev) + RMS_NORM + scale(ffn_norm.weight)}
			//           → {gate, up} GEMMs → SiLU → pointwise mul → down GEMM → residual add
			ops.emplace_back(tensor_op{
            .name   = "blk." + std::to_string(x) + "/MEGA_FFN",
            .inputs = {
                { .element_count = embedding_length * seq_length,          .data_type = compute_type  }, // ffn_inp
                { .element_count = embedding_length,                       .data_type = compute_type  }, // ffn_norm.weight
                { .element_count = feed_forward_length * embedding_length, .data_type = weight_type }, // ffn_gate.weight
                { .element_count = feed_forward_length * embedding_length, .data_type = weight_type }, // ffn_up.weight
                { .element_count = embedding_length * feed_forward_length, .data_type = weight_type }  // ffn_down.weight
            },
            // l_out [embedding_length * seq_length] (f32)
            .output = { .element_count = embedding_length * seq_length, .data_type = compute_type }
        });
		}

		// --- Final norm + IN-KERNEL SAMPLING (last-token) ------------------------
		ops.emplace_back(tensor_op{
        .name   = "final/MEGA_FINAL_NORM_SAMPLE_TOPK_LAST",
        .inputs = {
            // last position of l_out is selected internally by index math
            { .element_count = embedding_length,               .data_type = compute_type  }, // l_out (selected pos)
            { .element_count = embedding_length,               .data_type = compute_type  }, // output_norm.weight
            { .element_count = vocab_size * embedding_length,  .data_type = weight_type }, // output.weight

            // Sampler params
            { .element_count = 1, .data_type = compute_type  }, // temperature
            { .element_count = 1, .data_type = index_type  }, // top_k
            { .element_count = 1, .data_type = compute_type  }, // top_p
            { .element_count = 1, .data_type = compute_type  }, // repetition_penalty
            { .element_count = 1, .data_type = compute_type  }, // presence_penalty
            { .element_count = 1, .data_type = compute_type  }, // frequency_penalty
            { .element_count = 1, .data_type = index_type  }, // rep_window

            // Token history view (for penalties)
            { .element_count = seq_length, .data_type = index_type  }, // token_history

            // RNG state (e.g., xoroshiro128**: 2x u64)
            { .element_count = 2, .data_type = data_types::i64 }            // rng_state
        },
        // chosen token id for the last position
        .output = { .element_count = 1, .data_type = index_type } // result_token_id
    });

		// Optional: publish token if another thread consumes it concurrently
		// ops.emplace_back(tensor_op{ .name="final/BARRIER_publish_token", .inputs={}, .output={0, compute_type} });

		return ops;
	}
};

template<libraries library, kernel_type_profiles kernel_type_profile, model_arches model_arch, model_sizes model_size, model_generations model_generation, uint64_t seq_length>
static read_write get_read_writes() {
	auto inputs{ create_tensor_ops<library, kernel_type_profile, model_arch, model_size, model_generation, seq_length>::impl() };
	read_write return_values{};
	return_values.model_name = std::string{ model_traits<model_arch, model_size, model_generation>::name } + "-" + kernel_type_profile_traits<kernel_type_profile>::name;
	for (auto& value: inputs) {
		for (auto& value_new: value.inputs) {
			return_values.read_bytes += get_data_size(value_new.element_count, value_new.data_type);
		}
		return_values.written_bytes += get_data_size(value.output.element_count, value.output.data_type);
	}
	return return_values;
}

template<model_arches model_arch, model_sizes model_size, model_generations model_generation, kernel_type_profiles kernel_type_profile> void print_memory_bandwidth() {
	auto result = get_read_writes<libraries::llama, kernel_type_profile, model_arch, model_size, model_generation, 32>();
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(32) << ", on Model: " << result.model_name << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	auto result02 = get_read_writes<libraries::nihilus, kernel_type_profile, model_arch, model_size, model_generation, 32>();
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes<libraries::llama, kernel_type_profile, model_arch, model_size, model_generation, 1024>();
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(1024) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes<libraries::nihilus, kernel_type_profile, model_arch, model_size, model_generation, 1024>();
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes<libraries::llama, kernel_type_profile, model_arch, model_size, model_generation, 2048>();
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(2048) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes<libraries::nihilus, kernel_type_profile, model_arch, model_size, model_generation, 2048>();
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes<libraries::llama, kernel_type_profile, model_arch, model_size, model_generation, 16384>();
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(16384) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes<libraries::nihilus, kernel_type_profile, model_arch, model_size, model_generation, 16384>();
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
	result = get_read_writes<libraries::llama, kernel_type_profile, model_arch, model_size, model_generation, 131072>();
	std::cout << "Bandwidth used per Inference Run - For Length: " << std::to_string(131072) << std::endl;
	std::cout << "---------------------------------" << std::endl;
	std::cout << "Read bytes (llama.cpp): " << result.read_bytes << std::endl;
	std::cout << "Written bytes (llama.cpp): " << result.written_bytes << std::endl;
	result02 = get_read_writes<libraries::nihilus, kernel_type_profile, model_arch, model_size, model_generation, 131072>();
	std::cout << "Read bytes (Nihilus): " << result02.read_bytes << std::endl;
	std::cout << "Written bytes (Nihilus): " << result02.written_bytes << std::endl;
}

void nihilus::test_function() {
	{
		std::cout << "Meta exists: " << std::filesystem::exists("C:/users/chris/downloads/test.void_meta") << std::endl;
		std::cout << "Data exists: " << std::filesystem::exists("C:/users/chris/downloads/test.void") << std::endl;

		if (std::filesystem::exists("C:/users/chris/downloads/test.void_meta")) {
			std::cout << "Meta size: " << std::filesystem::file_size("C:/users/chris/downloads/test.void_meta") << " bytes" << std::endl;
		}
		if (std::filesystem::exists("C:/users/chris/downloads/test.void")) {
			std::cout << "Data size: " << std::filesystem::file_size("C:/users/chris/downloads/test.void") << " bytes" << std::endl;
		}
		nihilus::nihilus_db_file<int32_t> file{ nihilus::nihilus_db_file<int32_t>::open("C:/users/chris/downloads/test") };
		file.add_record(22432);
		file.get_record(22432) = 95959595;
		//std::cout << "CURRENT VALUE: " << file[92245].byte_offset << std::endl;
		//file.delete_record(256);
		file.flush();
		//std::cout << "CURRENT VALUE: " << file.get_record(snowflake{ 256 }) << std::endl;
	}
	{
		std::cout << "Meta exists: " << std::filesystem::exists("C:/users/chris/downloads/test.void_meta") << std::endl;
		std::cout << "Data exists: " << std::filesystem::exists("C:/users/chris/downloads/test.void") << std::endl;

		if (std::filesystem::exists("C:/users/chris/downloads/test.void_meta")) {
			std::cout << "Meta size: " << std::filesystem::file_size("C:/users/chris/downloads/test.void_meta") << " bytes" << std::endl;
		}
		if (std::filesystem::exists("C:/users/chris/downloads/test.void")) {
			std::cout << "Data size: " << std::filesystem::file_size("C:/users/chris/downloads/test.void") << " bytes" << std::endl;
		}
		nihilus::nihilus_db_file<int32_t> file{ nihilus::nihilus_db_file<int32_t>::open("C:/users/chris/downloads/test") };
		//file.add_record(snowflake{ 245 }, 2323);
		std::cout << "Metadata count: " << file.metadata_buffer.size() << std::endl;
		std::cout << "Index map size: " << file.snowflake_to_index.size() << std::endl;

		//file[92245] = {};
		//file[256]	= {};
		//file.flush();
		std::cout << "CURRENT VALUE: " << file[256] << std::endl;
		std::cout << "CURRENT VALUE: " << file[22432] << std::endl;
		std::cout << "CURRENT VALUE: " << file[92245] << std::endl;
		/*
		if (file.get_record(snowflake{ 92245 })) {
			std::cout << "CURRENT VALUE: " << value_new << std::endl;
		}
		if (file.get_record(snowflake{ 22492245 })) {
			std::cout << "CURRENT VALUE: " << value_new << std::endl;
		}*/
	}
}

template<const auto& model_config_00, const auto& model_config_01, typename enum_value> void print_comparisons() {
	constexpr uint64_t global_input_types_batched{ get_total_required_bytes_new<model_config_type<model_config_00>, enum_value>() };
	std::cout << "END OF NON-BATCHED!" << std::endl;
	constexpr uint64_t weight_types_non_batched{ get_total_required_bytes_new<model_config_type<model_config_01>, enum_value>() };
	std::cout << "NON-BATCHED TOTAL BYTES: " << global_input_types_batched << std::endl;
	std::cout << "BATCHED TOTAL BYTES: " << weight_types_non_batched << std::endl;
}

template<uint64_t size_val> struct rt_dims {

	constexpr rt_dims(const uint64_t (&str)[size_val]) {};
};

template<uint64_t size> rt_dims(uint64_t (&)[size]) -> rt_dims<size>;

template<rt_dims dims> struct test_struct {};

template<typename caberiht_type, uint64_t index = 0> uint64_t collect_total_byte_size(caberiht_type& caberiht, uint64_t value_new = 0) {
	if constexpr (index < caberiht_type::size) {
		std::cout << "CURRENT INDEX: " << index << std::endl;
		value_new += caberiht.template get_core<index>().total_required_bytes;
		std::cout << "CURRENT REQUIRED BYTES: " << caberiht.template get_core<index>().total_required_bytes << std::endl;
		return collect_total_byte_size<caberiht_type, index + 1>(caberiht, value_new);
	}
	return value_new;
}

template<dim_03_runtime_mutable value_type> void test_function_new(value_type){};

int32_t main([[maybe_unused]] int32_t argc, [[maybe_unused]] char** argv) {
	try {
		static constexpr uint64_t values[2]{};
		test_struct<values> test{};
		test_function_new(rt_dimensions<0b1000, 1, 1, 1, 1>{});
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		[[maybe_unused]] static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::batched_processing_type::disabled, nihilus::model_generations::v3_1,
			nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, nihilus::device_types::cpu, nihilus::exception_type::enabled,
			nihilus::default_max_sequence_length_type{ 1024 }, nihilus::benchmark_type::enabled);
		[[maybe_unused]] static constexpr auto model_config_01 = nihilus::generate_model_config(nihilus::batched_processing_type::enabled, nihilus::model_generations::v3_1,
			nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, nihilus::device_types::cpu, nihilus::exception_type::enabled,
			nihilus::default_max_sequence_length_type{ 1024 }, batch_size_type{ 16 }, nihilus::benchmark_type::enabled);
		[[maybe_unused]] indexed_data_holder<tensor_types::attn_q, model_config_type<model_config_00>> attn_q{};
		//std::cout << "TOTAL REQUIRED BYTES: " << attn_q.byte_count << std::endl;
		//std::cout << "DIMS: " << attn_q.dims[0] << ", " << attn_q.dims[1] << ", " << attn_q.dims[2] << ", " << attn_q.dims[3] << std::endl;
		get_nihilus_cathedral_array_t<model_config_type<model_config_00>, tensor_types, data_holder_aggregator, indexed_data_holder> caberiht_01{};
		std::cout << "TOTAL REQUIRED BYTES: " << collect_total_byte_size(caberiht_01) << std::endl;
		get_nihilus_cathedral_array_t<model_config_type<model_config_01>, tensor_types, data_holder_aggregator, indexed_data_holder> caberiht{};
		std::cout << "TOTAL REQUIRED BYTES: " << collect_total_byte_size(caberiht) << std::endl;
		caberiht.operator[](tag<tensor_types::attn_q>{});
		//print_comparisons<model_config_00, model_config_01, attn_prep_and_score_sub_kernel_types>();
		//print_comparisons<model_config_00, model_config_01, attn_out_and_ffn_sub_kernel_types>();
		//print_comparisons<model_config_00, model_config_01, global_output_and_sampling_sub_kernel_types>();
		//uint64_t weight_types_batched{ get_total_required_bytes_new<model_config_type<model_config_00>, weight_types>() };

		//uint64_t global_input_types_batched{ get_total_required_bytes_new<model_config_type<model_config_00>, global_input_types>() };

		//uint64_t weight_types_non_batched{ get_total_required_bytes_new<model_config_type<model_config_01>, global_input_types>() };

		//uint64_t global_input_types_non_batched{ get_total_required_bytes_new<model_config_type<model_config_01>, global_input_types>() };
		//uint64_t attn_prep_and_score_sub_kernel_types_batched{ get_total_required_bytes_new<model_config_type<model_config_00>, attn_prep_and_score_sub_kernel_types>() };
		//uint64_t  attn_prep_and_score_sub_kernel_types_non_batched{ get_total_required_bytes_new<model_config_type<model_config_01>,  attn_prep_and_score_sub_kernel_types>() };
		//uint64_t attn_out_and_ffn_sub_kernel_types_batched{ get_total_required_bytes_new<model_config_type<model_config_00>, attn_out_and_ffn_sub_kernel_types>() };
		//uint64_t attn_out_and_ffn_sub_kernel_types_non_batched{ get_total_required_bytes_new<model_config_type<model_config_01>, attn_out_and_ffn_sub_kernel_types>() };
		//uint64_t global_output_and_sampling_sub_kernel_types_batched{ get_total_required_bytes_new<model_config_type<model_config_00>, global_output_and_sampling_sub_kernel_types>() };
		//uint64_t global_output_and_sampling_sub_kernel_types_non_batched{ get_total_required_bytes_new<model_config_type<model_config_01>, global_output_and_sampling_sub_kernel_types>() };
		//std::cout << "WEIGHTS (BATCHED): " << weight_types_batched << std::endl;
		//std::cout << "WEIGHTS (NON-BATCHED): " << weight_types_non_batched << std::endl;
		//std::cout << "GLOBAL_INPUTS (BATCHED): " << global_input_types_batched << std::endl;
		//std::cout << "GLOBAL_INPUTS (NON-BATCHED): " << global_input_types_non_batched << std::endl;
		//std::cout << "GLOBAL_INPUTS (BATCHED): " << attn_prep_and_score_sub_kernel_types_batched << std::endl;
		//std::cout << "GLOBAL_INPUTS (NON-BATCHED): " << attn_prep_and_score_sub_kernel_types_non_batched << std::endl;
		//std::cout << "GLOBAL_INPUTS (BATCHED): " << attn_out_and_ffn_sub_kernel_types_batched << std::endl;
		//std::cout << "GLOBAL_INPUTS (NON-BATCHED): " << attn_out_and_ffn_sub_kernel_types_non_batched << std::endl;
		/*
		{
			[[maybe_unused]] core_traits<model_config_type<model_config_00>, core_types::weights> new_core_traits_weights{};
			std::cout << "WEIGHTS (NON-BATCHED): " << new_core_traits_weights.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_00>, core_types::global_inputs> new_core_traits_global_inputs{};
			std::cout << "WEIGHTS (NON-BATCHED): " << new_core_traits_global_inputs.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_00>, core_types::token_embeddings> new_core_traits_token_embeddings{};
			std::cout << "WEIGHTS (NON-BATCHED): " << new_core_traits_token_embeddings.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_00>, core_types::attn_prep_and_score> new_core_traits_attn_prep_and_score{};
			std::cout << "WEIGHTS (NON-BATCHED): " << new_core_traits_attn_prep_and_score.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_00>, core_types::attn_out_and_ffn> new_core_traits_attn_out_and_ffn{};
			std::cout << "WEIGHTS (NON-BATCHED): " << new_core_traits_attn_out_and_ffn.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_00>, core_types::global_output_and_sampling> new_core_traits_global_output_and_sampling{};
			std::cout << "WEIGHTS (NON-BATCHED): " << new_core_traits_global_output_and_sampling.total_required_bytes << std::endl;
		}
		{
			[[maybe_unused]] core_traits<model_config_type<model_config_01>, core_types::weights> new_core_traits_weights{};
			std::cout << "WEIGHTS (BATCHED): " << new_core_traits_weights.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_01>, core_types::global_inputs> new_core_traits_global_inputs{};
			std::cout << "WEIGHTS (BATCHED): " << new_core_traits_global_inputs.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_01>, core_types::token_embeddings> new_core_traits_token_embeddings{};
			std::cout << "WEIGHTS (BATCHED): " << new_core_traits_token_embeddings.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_01>, core_types::attn_prep_and_score> new_core_traits_attn_prep_and_score{};
			std::cout << "WEIGHTS (BATCHED): " << new_core_traits_attn_prep_and_score.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_01>, core_types::attn_out_and_ffn> new_core_traits_attn_out_and_ffn{};
			std::cout << "WEIGHTS (BATCHED): " << new_core_traits_attn_out_and_ffn.total_required_bytes << std::endl;
			[[maybe_unused]] core_traits<model_config_type<model_config_01>, core_types::global_output_and_sampling> new_core_traits_global_output_and_sampling{};
			std::cout << "WEIGHTS (BATCHED): " << new_core_traits_global_output_and_sampling.total_required_bytes << std::endl;
		}
		*/
		cli_params cli_args = harbinger::parse_cli_arguments(argc, argv);
		nihilus::model_collection_type<model_config_00> collection{ cli_args };
		collection.process_input(cli_args.prompt);
	} catch (const std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}
