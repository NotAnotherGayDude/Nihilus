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
		constexpr uint32_t embedding_length		  = model_traits<model_arch, model_size, model_generation>::embedding_length;
		constexpr uint32_t vocab_size			  = model_traits<model_arch, model_size, model_generation>::vocab_size;
		constexpr uint32_t feed_forward_length	  = model_traits<model_arch, model_size, model_generation>::feed_forward_length;
		constexpr uint32_t attention_head_count	  = model_traits<model_arch, model_size, model_generation>::attention_head_count;
		constexpr uint32_t block_count			  = model_traits<model_arch, model_size, model_generation>::block_count;
		constexpr uint32_t rope_dimension_count	  = model_traits<model_arch, model_size, model_generation>::rope_dimension_count;
		constexpr uint64_t n_embd_kv_gqa		  = model_traits<model_arch, model_size, model_generation>::n_embd_kv_gqa;
		constexpr uint64_t total_cache_size_k	  = seq_length * n_embd_kv_gqa;
		constexpr uint64_t total_cache_size_v	  = seq_length * n_embd_kv_gqa;
		static constexpr data_types weight_type	  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::weight_type>::data_type;
		static constexpr data_types index_type	  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::index_type>::data_type;
		static constexpr data_types compute_type  = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::compute_type>::data_type;
		static constexpr data_types kv_cache_type = type_traits<typename kernel_type_profile_traits<kernel_type_profile>::kv_cache_type>::data_type;
		std::vector<tensor_op> ops;

		// --- Token embeddings ----------------------------------------------------
		ops.emplace_back(tensor_op{
        .name   = "token_embeddings",
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
            .name   = "blk." + std::to_string(x) + "/attn_prep_and_score",
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
				.output = { .element_count = 0, .data_type = kv_cache_type }// sentinel
			});
			ops.emplace_back(tensor_op{
				.name	= "blk." + std::to_string(x) + "/BARRIER_publish_V_cache",
				.inputs = {},
				.output = { .element_count = 0, .data_type = kv_cache_type }// sentinel
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

template<uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> struct dimensions {
	static constexpr array<uint64_t, 4> dims{ dim_00, dim_01, dim_02, dim_03 };
};

template<uint64_t... runtime_mask> consteval uint64_t get_runtime_mask_new() {
	static_assert(((runtime_mask < 4) && ...), "Sorry, but you can only define a dimension within the first 4 dimensions as runtime mutable!");
	constexpr uint64_t result = ((1ULL << runtime_mask) | ...);
	static_assert(has_at_most_two_bits_set(result), "Sorry, but you can only define one or two of the first 4 dimensions as runtime mutable!");
	return result & 0xF;
}

template<uint64_t runtime_mask_new, uint64_t dim_00, uint64_t dim_01, uint64_t dim_02, uint64_t dim_03> struct rt_dimensions : public dimensions<dim_00, dim_01, dim_02, dim_03> {
	using base_type = dimensions<dim_00, dim_01, dim_02, dim_03>;
	static_assert(has_at_most_two_bits_set(runtime_mask_new), "Sorry, but you can only define one or two of the first 4 dimensions as runtime mutable!");
	static constexpr uint64_t runtime_mask{ runtime_mask_new & 0xF };

	mutable array<uint64_t, 4> rt_dims{ base_type::dims[0], base_type::dims[1], base_type::dims[2], base_type::dims[3] };

	static constexpr array<uint64_t, 4> get_array() {
		return array<uint64_t, 4>{ base_type::dims[0], base_type::dims[1], base_type::dims[2], base_type::dims[3] };
	}

	NIHILUS_HOST const array<uint64_t, 4>& get_array_rt() const {
		return rt_dims;
	}

	template<typename index_tag> NIHILUS_HOST uint64_t& get_dims(index_tag index) {
		static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
		static_assert(static_assert_printer_val<((runtime_mask & (1ULL << index)) != 0), incorrect_runtime_dims::incorrect_runtime_dim, index>::impl);
		return rt_dims[index];
	}

	template<typename index_tag> NIHILUS_HOST uint64_t get_dims(index_tag index) const {
		static_assert(index < 4, "Error: Index is out of bounds [0-3] for the fixed dimension storage!");
		static_assert(static_assert_printer_val<((runtime_mask & (1ULL << index)) != 0), incorrect_runtime_dims::incorrect_runtime_dim, index>::impl);
		return rt_dims[index];
	}
};

template<typename config_type_new, core_types_new core_type> struct sync_base_new {};

template<typename config_type_new, core_types_new core_type>
	requires(config_type_new::device_type == device_types::cpu && core_type != core_types_new::token_embeddings && core_type != core_types_new::global_output_and_sampling)
struct sync_base_new<config_type_new, core_type> {
	array<aligned_atomic<int64_t>, model_traits_type<config_type_new>::block_count> current_chunk_prompt_eval{};
	array<aligned_atomic<int64_t>, model_traits_type<config_type_new>::block_count> current_chunk_eval{};
	array<aligned_atomic<int64_t>, model_traits_type<config_type_new>::block_count> latch_prompt_eval{};
	array<aligned_atomic<int64_t>, model_traits_type<config_type_new>::block_count> latch_eval{};
};

template<typename config_type_new, core_types_new core_type>
	requires(config_type_new::device_type == device_types::cpu && (core_type == core_types_new::token_embeddings || core_type == core_types_new::global_output_and_sampling))
struct sync_base_new<config_type_new, core_type> {
	aligned_atomic<int64_t> current_chunk_prompt_eval{};
	aligned_atomic<int64_t> current_chunk_eval{};
	aligned_atomic<int64_t> latch_prompt_eval{};
	aligned_atomic<int64_t> latch_eval{};
};

template<typename value_type>
concept kernel_dims_types_new = requires() { detail::remove_cvref_t<value_type>::dims; };

template<typename value_type>
concept dim_00_runtime_mutable = (value_type::runtime_mask & 0b0001) != 0;

template<typename value_type>
concept dim_01_runtime_mutable = (value_type::runtime_mask & 0b0010) != 0;

template<typename value_type>
concept dim_02_runtime_mutable = (value_type::runtime_mask & 0b0100) != 0;

template<typename value_type>
concept dim_03_runtime_mutable = (value_type::runtime_mask & 0b1000) != 0;

template<typename value_type>
concept runtime_dims_types = dim_00_runtime_mutable<value_type> || dim_01_runtime_mutable<value_type> || dim_02_runtime_mutable<value_type> || dim_03_runtime_mutable<value_type>;

template<typename kernel_type, typename... dims_types> struct dim_traits_new;

template<kernel_dims_types_new input_dims_01> struct dim_traits_new<kernel_types_type<kernel_types::weights>, input_dims_01> {
	using dims_type = kernel_dims<input_dims_01::runtime_mask, input_dims_01::dims[0], input_dims_01::dims[1], input_dims_01::dims[2], input_dims_01::dims[3]>;
};

template<preserved_dimensions_kernel_types kernel_type, kernel_dims_types_new... dims_types> struct dim_traits_new<kernel_type, dims_types...> {
	using first_type = get_first_type_t<dims_types...>;
	using dims_type	 = kernel_dims<first_type::runtime_mask, first_type::dims[0], first_type::dims[1], first_type::dims[2], first_type::dims[3]>;
};

template<kernel_dims_types_new output_dims, kernel_dims_types_new input_dims> struct dim_traits_new<kernel_types_type<kernel_types::reshape>, output_dims, input_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::reshape_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<kernel_dims_types_new output_dims, kernel_dims_types_new input_dims> struct dim_traits_new<kernel_types_type<kernel_types::view>, output_dims, input_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::view_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<kernel_dims_types_new output_dims, kernel_dims_types_new input_dims> struct dim_traits_new<kernel_types_type<kernel_types::transpose>, output_dims, input_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::transpose_total_element_count_mismatch, input_elements, output_elements>::impl);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || dims01[0] == dims02[1]),
		dim_trait_static_assert_errors::transpose_dimension_0_mismatch, dims01[0], dims02[1]>::impl);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || dims01[1] == dims02[0]),
		dim_trait_static_assert_errors::transpose_dimension_1_mismatch, dims01[1], dims02[0]>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<kernel_dims_types_new output_dims, kernel_dims_types_new input_dims> struct dim_traits_new<kernel_types_type<kernel_types::permute>, output_dims, input_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::permute_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<kernel_dims_types_new input_dims_01, kernel_dims_types_new input_dims_02> struct dim_traits_new<kernel_types_type<kernel_types::mul_mat>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dims;
	static constexpr auto dims02 = input_dims_02::dims;
	using dims_type				 = kernel_dims<input_dims_02::runtime_mask, dims02[0], dims01[2], dims02[2], dims02[3]>;
};

template<kernel_dims_types_new input_dims_01, kernel_dims_types_new input_dims_02> struct dim_traits_new<kernel_types_type<kernel_types::get_rows>, input_dims_01, input_dims_02> {
	static constexpr auto dims01 = input_dims_01::dims;
	static constexpr auto dims02 = input_dims_02::dims;
	using dims_type				 = kernel_dims<5, dims02[0], dims01[1], dims02[1], dims01[3]>;
};

template<kernel_dims_types_new output_dims, kernel_dims_types_new input_dims> struct dim_traits_new<kernel_types_type<kernel_types::cont>, output_dims, input_dims> {
	static constexpr auto dims01			  = input_dims::dims;
	static constexpr auto dims02			  = output_dims::dims;
	static constexpr uint64_t input_elements  = compute_elements(dims01);
	static constexpr uint64_t output_elements = compute_elements(dims02);
	static_assert(static_assert_printer_val<(input_dims::runtime_mask != 0 || output_dims::runtime_mask != 0 || input_elements == output_elements),
		dim_trait_static_assert_errors::cont_total_element_count_mismatch, input_elements, output_elements>::impl);
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<kernel_dims_types_new output_dims, kernel_dims_types_new input_dims_01, kernel_dims_types_new input_dims_02>
struct dim_traits_new<kernel_types_type<kernel_types::sample_logits>, output_dims, input_dims_01, input_dims_02> {
	using dims_type = kernel_dims<output_dims::runtime_mask, output_dims::dims[0], output_dims::dims[1], output_dims::dims[2], output_dims::dims[3]>;
};

template<llama_arch_config_types config_type_new, enum_types auto enum_value_new, typename enum_type = decltype(enum_value_new)> struct data_traits;

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::attn_q>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::embedding_length, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::attn_q };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::attn_k>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::n_embd_kv_gqa, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::attn_k };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::attn_v>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::n_embd_kv_gqa, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::attn_v };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::attn_output>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::embedding_length, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::attn_output };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::attn_norm>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::attn_norm };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_gate>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::feed_forward_length, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_gate };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_up>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::feed_forward_length, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_up };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_down>
	: public dimensions<1, model_dimensions<config_type_new>::feed_forward_length, model_dimensions<config_type_new>::embedding_length, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_down };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_norm>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_norm };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::token_embd>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::vocab_size, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::token_embd };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::rope_freqs>
	: public dimensions<1, model_dimensions<config_type_new>::rope_dimension_count / 2, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::rope_freqs };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::output_norm>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::output_norm };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::output>
	: public dimensions<1, model_dimensions<config_type_new>::embedding_length, model_dimensions<config_type_new>::vocab_size, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::output };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::weight_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::inp_tokens>
	: public rt_dimensions<get_runtime_mask_new<0, 1>(), config_type_new::batch_size, config_type_new::max_sequence_length, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::inp_tokens };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::inp_pos>
	: public rt_dimensions<get_runtime_mask_new<0, 1>(), config_type_new::batch_size, config_type_new::max_sequence_length, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::inp_pos };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::cache_k>
	: public rt_dimensions<get_runtime_mask_new<0, 1>(), config_type_new::batch_size, config_type_new ::max_sequence_length, model_dimensions<config_type_new>::n_embd_kv_gqa,
		  1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::cache_k };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::cache_v>
	: public rt_dimensions<get_runtime_mask_new<0, 1>(), config_type_new::batch_size, config_type_new ::max_sequence_length, model_dimensions<config_type_new>::n_embd_kv_gqa,
		  1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::cache_v };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::per_block };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kq_mask> : public dimensions<32ull, 32ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::kq_mask };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::mask_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::inp_out_ids>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::inp_out_ids };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::index_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::temperature>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::temperature };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::top_k>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::top_k };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::top_p>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::top_p };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::repetition_penalty>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::repetition_penalty };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::presence_penalty>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::presence_penalty };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::frequency_penalty>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::frequency_penalty };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::rep_window>
	: public rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::rep_window };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::token_history>
	: public rt_dimensions<get_runtime_mask_new<0, 1>(), config_type_new::batch_size, config_type_new::max_sequence_length, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::token_history };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::rng_state> : public dimensions<256ull, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::rng_state };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::logits_bias>
	: public dimensions<model_dimensions<config_type_new>::vocab_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::logits_bias };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::allowed_vocab_mask>
	: public dimensions<model_dimensions<config_type_new>::vocab_size, 1ull, 1ull, 1ull> {
	static constexpr auto sub_kernel_type{ tensor_types::allowed_vocab_mask };
	using output_type = kernel_type_profile_traits<config_type_new::kernel_type_profile>::activation_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::inp_embd_get_rows>
	: public dim_traits_new<kernel_types_type<kernel_types::get_rows>, data_traits<config_type_new, tensor_types::token_embd>,
		  data_traits<config_type_new, tensor_types::inp_tokens>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::inp_embd_get_rows };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::norm_pre_attn_rms_norm>
	: public dim_traits_new<kernel_types_type<kernel_types::rms_norm>, data_traits<config_type_new, tensor_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::norm_pre_attn_rms_norm };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::attn_norm_mul>
	: public dim_traits_new<kernel_types_type<kernel_types::mul>, data_traits<config_type_new, tensor_types::norm_pre_attn_rms_norm>,
		  data_traits<config_type_new, tensor_types::attn_norm>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::attn_norm_mul };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::qcur_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::attn_q>,
		  data_traits<config_type_new, tensor_types::attn_norm_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::qcur_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::qcur_reshape>
	: public dim_traits_new<kernel_types_type<kernel_types::reshape>,
		  rt_dimensions<get_runtime_mask_new<0, 3>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::rope_dimension_count,
			  model_dimensions<config_type_new>::attention_head_count, config_type_new::max_sequence_length>,
		  data_traits<config_type_new, tensor_types::qcur_mul_mat>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::qcur_reshape };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::qcur_rope>
	: public dim_traits_new<kernel_types_type<kernel_types::rope>, data_traits<config_type_new, tensor_types::qcur_reshape>, data_traits<config_type_new, tensor_types::inp_pos>,
		  data_traits<config_type_new, tensor_types::rope_freqs>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::qcur_rope };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kcur_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::attn_k>,
		  data_traits<config_type_new, tensor_types::attn_norm_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kcur_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kcur_reshape>
	: public dim_traits_new<kernel_types_type<kernel_types::reshape>,
		  rt_dimensions<get_runtime_mask_new<0, 3>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::rope_dimension_count,
			  model_dimensions<config_type_new>::attention_head_count_kv, config_type_new::max_sequence_length>,
		  data_traits<config_type_new, tensor_types::kcur_mul_mat>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kcur_reshape };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kcur_rope>
	: public dim_traits_new<kernel_types_type<kernel_types::rope>, data_traits<config_type_new, tensor_types::kcur_reshape>, data_traits<config_type_new, tensor_types::inp_pos>,
		  data_traits<config_type_new, tensor_types::rope_freqs>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kcur_rope };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::vcur_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::attn_v>,
		  data_traits<config_type_new, tensor_types::attn_norm_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::vcur_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::k_cache_view>
	: public dim_traits_new<kernel_types_type<kernel_types::view>,
		  rt_dimensions<get_runtime_mask_new<1>(), model_dimensions<config_type_new>::batch_size,
			  model_dimensions<config_type_new>::rope_dimension_count * model_dimensions<config_type_new>::attention_head_count_kv * config_type_new::max_sequence_length, 1, 1>,
		  data_traits<config_type_new, tensor_types::cache_k>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::k_cache_view };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::k_cache_cpy>
	: public dim_traits_new<kernel_types_type<kernel_types::copy>,
		  rt_dimensions<get_runtime_mask_new<1>(), model_dimensions<config_type_new>::batch_size,
			  model_dimensions<config_type_new>::rope_dimension_count * model_dimensions<config_type_new>::attention_head_count_kv * config_type_new::max_sequence_length, 1, 1>,
		  data_traits<config_type_new, tensor_types::cache_k>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::k_cache_cpy };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::vcur_transpose>
	: public dim_traits_new<kernel_types_type<kernel_types::transpose>,
		  rt_dimensions<get_runtime_mask_new<0, 1>(), model_dimensions<config_type_new>::batch_size, config_type_new::max_sequence_length,
			  model_dimensions<config_type_new>::n_embd_kv_gqa, 1>,
		  data_traits<config_type_new, tensor_types::vcur_mul_mat>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::vcur_transpose };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::v_cache_view>
	: public dim_traits_new<kernel_types_type<kernel_types::view>,
		  rt_dimensions<get_runtime_mask_new<0, 1>(), model_dimensions<config_type_new>::batch_size, config_type_new::max_sequence_length,
			  model_dimensions<config_type_new>::n_embd_kv_gqa, 1>,
		  data_traits<config_type_new, tensor_types::cache_v>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::v_cache_view };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::v_cache_cpy>
	: public dim_traits_new<kernel_types_type<kernel_types::copy>, data_traits<config_type_new, tensor_types::vcur_transpose>,
		  data_traits<config_type_new, tensor_types::v_cache_view>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::v_cache_cpy };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::v_from_cache_view>
	: public dim_traits_new<kernel_types_type<kernel_types::view>,
		  rt_dimensions<get_runtime_mask_new<0>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::attention_head_count,
			  model_dimensions<config_type_new>::rope_dimension_count, model_dimensions<config_type_new>::attention_head_count_kv>,
		  data_traits<config_type_new, tensor_types::cache_v>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::v_from_cache_view };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::k_from_cache_view>
	: public dim_traits_new<kernel_types_type<kernel_types::view>,
		  rt_dimensions<get_runtime_mask_new<0>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::rope_dimension_count,
			  model_dimensions<config_type_new>::attention_head_count, model_dimensions<config_type_new>::attention_head_count_kv>,
		  data_traits<config_type_new, tensor_types::cache_v>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::k_from_cache_view };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::kv_cache_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::q_permute>
	: public dim_traits_new<kernel_types_type<kernel_types::permute>,
		  rt_dimensions<get_runtime_mask_new<0, 2>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::rope_dimension_count,
			  config_type_new::max_sequence_length, model_dimensions<config_type_new>::attention_head_count>,
		  data_traits<config_type_new, tensor_types::qcur_rope>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::q_permute };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kq_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::k_from_cache_view>,
		  data_traits<config_type_new, tensor_types::q_permute>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kq_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kq_soft_max>
	: public dim_traits_new<kernel_types_type<kernel_types::softmax>, data_traits<config_type_new, tensor_types::kq_mul_mat>,
		  data_traits<config_type_new, tensor_types::kq_mask>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kq_soft_max };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kqv_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::v_from_cache_view>,
		  data_traits<config_type_new, tensor_types::kq_soft_max>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kqv_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kqv_merged_permute>
	: public dim_traits_new<kernel_types_type<kernel_types::permute>,
		  rt_dimensions<get_runtime_mask_new<0, 2>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::rope_dimension_count,
			  model_dimensions<config_type_new>::attention_head_count, config_type_new::max_sequence_length>,
		  data_traits<config_type_new, tensor_types::kqv_mul_mat>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kqv_merged_permute };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kqv_merged_cont>
	: public dim_traits_new<kernel_types_type<kernel_types::cont>,
		  rt_dimensions<get_runtime_mask_new<1>(), model_dimensions<config_type_new>::batch_size, model_dimensions<config_type_new>::embedding_length,
			  config_type_new::max_sequence_length, 1>,
		  data_traits<config_type_new, tensor_types::kqv_merged_permute>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kqv_merged_cont };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::kqv_out_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::attn_output>,
		  data_traits<config_type_new, tensor_types::kqv_merged_cont>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::kqv_out_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_inp_add>
	: public dim_traits_new<kernel_types_type<kernel_types::add>, data_traits<config_type_new, tensor_types::kqv_out_mul_mat>,
		  data_traits<config_type_new, tensor_types::inp_embd_get_rows>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_inp_add };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::norm_pre_ffn_rms_norm>
	: public dim_traits_new<kernel_types_type<kernel_types::rms_norm>, data_traits<config_type_new, tensor_types::ffn_inp_add>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::norm_pre_ffn_rms_norm };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_norm_mul>
	: public dim_traits_new<kernel_types_type<kernel_types::mul>, data_traits<config_type_new, tensor_types::norm_pre_ffn_rms_norm>,
		  data_traits<config_type_new, tensor_types::ffn_norm>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_norm_mul };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_gate_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::ffn_gate>,
		  data_traits<config_type_new, tensor_types::ffn_norm_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_gate_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_silu_unary>
	: public dim_traits_new<kernel_types_type<kernel_types::silu>, data_traits<config_type_new, tensor_types::ffn_gate_mul_mat>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_silu_unary };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_up_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::ffn_up>,
		  data_traits<config_type_new, tensor_types::ffn_norm_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_up_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_gate_par_mul>
	: public dim_traits_new<kernel_types_type<kernel_types::mul>, data_traits<config_type_new, tensor_types::ffn_silu_unary>,
		  data_traits<config_type_new, tensor_types::ffn_up_mul_mat>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_gate_par_mul };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::ffn_out_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::ffn_down>,
		  data_traits<config_type_new, tensor_types::ffn_gate_par_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::ffn_out_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::layer_out_add>
	: public dim_traits_new<kernel_types_type<kernel_types::add>, data_traits<config_type_new, tensor_types::ffn_out_mul_mat>,
		  data_traits<config_type_new, tensor_types::ffn_inp_add>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::layer_out_add };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::node_1016_get_rows>
	: public dim_traits_new<kernel_types_type<kernel_types::get_rows>, data_traits<config_type_new, tensor_types::kqv_out_mul_mat>,
		  data_traits<config_type_new, tensor_types::inp_out_ids>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::node_1016_get_rows };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::node_1017_get_rows>
	: public dim_traits_new<kernel_types_type<kernel_types::get_rows>, data_traits<config_type_new, tensor_types::layer_out_add>,
		  data_traits<config_type_new, tensor_types::inp_out_ids>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::node_1017_get_rows };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::final_ffn_inp_add>
	: public dim_traits_new<kernel_types_type<kernel_types::add>, data_traits<config_type_new, tensor_types::node_1016_get_rows>,
		  data_traits<config_type_new, tensor_types::node_1017_get_rows>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::final_ffn_inp_add };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::final_norm_pre_rms_norm>
	: public dim_traits_new<kernel_types_type<kernel_types::rms_norm>, data_traits<config_type_new, tensor_types::layer_out_add>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::final_norm_pre_rms_norm };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::final_norm_mul>
	: public dim_traits_new<kernel_types_type<kernel_types::mul>, data_traits<config_type_new, tensor_types::final_norm_pre_rms_norm>,
		  data_traits<config_type_new, tensor_types::output_norm>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::final_norm_mul };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::result_output_mul_mat>
	: public dim_traits_new<kernel_types_type<kernel_types::mul_mat>, data_traits<config_type_new, tensor_types::output>,
		  data_traits<config_type_new, tensor_types::final_norm_mul>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::result_output_mul_mat };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_repetition_penalty>
	: public dim_traits_new<kernel_types_type<kernel_types::repetition_penalty>, data_traits<config_type_new, tensor_types::result_output_mul_mat>,
		  data_traits<config_type_new, tensor_types::token_history>, data_traits<config_type_new, tensor_types::repetition_penalty>,
		  data_traits<config_type_new, tensor_types::rep_window>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_repetition_penalty };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_presence_penalty>
	: public dim_traits_new<kernel_types_type<kernel_types::presence_penalty>, data_traits<config_type_new, tensor_types::apply_repetition_penalty>,
		  data_traits<config_type_new, tensor_types::token_history>, data_traits<config_type_new, tensor_types::presence_penalty>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_presence_penalty };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_frequency_penalty>
	: public dim_traits_new<kernel_types_type<kernel_types::frequency_penalty>, data_traits<config_type_new, tensor_types::apply_presence_penalty>,
		  data_traits<config_type_new, tensor_types::token_history>, data_traits<config_type_new, tensor_types::frequency_penalty>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_frequency_penalty };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_logits_bias>
	: public dim_traits_new<kernel_types_type<kernel_types::add>, data_traits<config_type_new, tensor_types::apply_frequency_penalty>,
		  data_traits<config_type_new, tensor_types::logits_bias>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_logits_bias };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_vocab_mask>
	: public dim_traits_new<kernel_types_type<kernel_types::mul>, data_traits<config_type_new, tensor_types::apply_logits_bias>,
		  data_traits<config_type_new, tensor_types::allowed_vocab_mask>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_vocab_mask };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_temperature>
	: public dim_traits_new<kernel_types_type<kernel_types::div>, data_traits<config_type_new, tensor_types::apply_vocab_mask>,
		  data_traits<config_type_new, tensor_types::temperature>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_temperature };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::compute_softmax>
	: public dim_traits_new<kernel_types_type<kernel_types::softmax>, data_traits<config_type_new, tensor_types::apply_temperature>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::compute_softmax };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_top_k_filter>
	: public dim_traits_new<kernel_types_type<kernel_types::top_k_filter>, data_traits<config_type_new, tensor_types::compute_softmax>,
		  data_traits<config_type_new, tensor_types::top_k>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_top_k_filter };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::apply_top_p_filter>
	: public dim_traits_new<kernel_types_type<kernel_types::top_p_filter>, data_traits<config_type_new, tensor_types::apply_top_k_filter>,
		  data_traits<config_type_new, tensor_types::top_p>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::apply_top_p_filter };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::compute_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<llama_arch_config_types config_type_new> struct data_traits<config_type_new, tensor_types::sample_token>
	: public dim_traits_new<kernel_types_type<kernel_types::sample_logits>, rt_dimensions<get_runtime_mask_new<0>(), config_type_new::batch_size, 1, 1, 1>,
		  data_traits<config_type_new, tensor_types::apply_top_p_filter>, data_traits<config_type_new, tensor_types::rng_state>>::dims_type {
	static constexpr auto sub_kernel_type{ tensor_types::sample_token };
	using output_type = typename kernel_type_profile_traits<config_type_new::kernel_type_profile>::token_type;
	static constexpr data_strategy_types data_strategy_type{ data_strategy_types::global };
};

template<typename config_type_new, data_strategy_types data_strategy_type, typename derived_type> struct data_mixin_new;

template<typename config_type_new, core_types_new> struct core_traits;

template<typename config_type_new, typename derived_type> struct data_mixin_new<config_type_new, data_strategy_types::global, derived_type> : public derived_type {
	using output_type = typename derived_type::output_type;
	static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(derived_type::dims) };

	template<typename output_type_newer = output_type> NIHILUS_HOST output_type_newer* get_data() {
		return static_cast<output_type_newer*>(data);
	}

	NIHILUS_HOST void** get_data_ptr() {
		return &data;
	}

	NIHILUS_HOST void set_data(void* data_new) {
		data = data_new;
	}

  protected:
	void* data{};
};

template<typename config_type_new, typename derived_type> struct data_mixin_new<config_type_new, data_strategy_types::per_block, derived_type> : public derived_type {
	using output_type = typename derived_type::output_type;
	static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(derived_type::dims) * model_dimensions<config_type_new>::block_count };

	template<typename output_type_newer = output_type> NIHILUS_HOST output_type_newer* get_data(uint64_t index) {
		return static_cast<output_type_newer*>(data[index]);
	}

	NIHILUS_HOST void** get_data_ptr(uint64_t index) {
		return &data[index];
	}

	NIHILUS_HOST void set_data(void* data_new, uint64_t index) {
		data[index] = data_new;
	}

  protected:
	array<void*, model_traits_type<config_type_new>::block_count> data{};
};

template<typename config_type_new, integral_or_enum_types auto enum_value_new> struct data_input_aggregator;
template<typename config_type_new, integral_or_enum_types auto enum_value_new> struct data_output_aggregator;
template<typename config_type_new, integral_or_enum_types auto enum_value_new> struct kernel_aggregator;

template<integral_or_enum_types auto enum_value_new, typename config_type_new> struct data_interface
	: public data_mixin_new<config_type_new, data_traits<config_type_new, enum_value_new>::data_strategy_type, data_traits<config_type_new, enum_value_new>>,
	  public core_elem_base<enum_value_new, data_interface<enum_value_new, config_type_new>> {};

template<llama_arch_config_types config_type_new> struct data_output_aggregator<config_type_new, core_types_new::weights> {
	static constexpr array values{ tensor_types::attn_q, tensor_types::attn_k, tensor_types::attn_v, tensor_types::attn_output, tensor_types::attn_norm, tensor_types::ffn_gate,
		tensor_types::ffn_up, tensor_types::ffn_down, tensor_types::ffn_norm, tensor_types::token_embd, tensor_types::rope_freqs, tensor_types::output_norm, tensor_types::output };
};

template<llama_arch_config_types config_type_new> struct data_output_aggregator<config_type_new, core_types_new::global_inputs> {
	static constexpr array values{ tensor_types::inp_tokens, tensor_types::inp_pos, tensor_types::cache_k, tensor_types::cache_v, tensor_types::kq_mask, tensor_types::inp_out_ids,
		tensor_types::temperature, tensor_types::top_k, tensor_types::top_p, tensor_types::repetition_penalty, tensor_types::presence_penalty, tensor_types::frequency_penalty,
		tensor_types::rep_window, tensor_types::token_history, tensor_types::rng_state, tensor_types::logits_bias, tensor_types::allowed_vocab_mask };
};

template<llama_arch_config_types config_type_new> struct data_input_aggregator<config_type_new, core_types_new::token_embeddings> {
	static constexpr array values{ tensor_types::inp_tokens, tensor_types::token_embd };
};

template<llama_arch_config_types config_type_new> struct data_output_aggregator<config_type_new, core_types_new::token_embeddings> {
	static constexpr array values{ tensor_types::inp_embd_get_rows };
};

template<llama_arch_config_types config_type_new> struct kernel_aggregator<config_type_new, core_types_new::token_embeddings> {
	static constexpr array values{ tensor_types::inp_embd_get_rows };
};

template<llama_arch_config_types config_type_new> struct data_input_aggregator<config_type_new, core_types_new::attn_prep_and_score> {
	static constexpr array values{ tensor_types::inp_embd_get_rows, tensor_types::attn_norm, tensor_types::attn_q, tensor_types::attn_k, tensor_types::attn_v,
		tensor_types::attn_output, tensor_types::inp_pos, tensor_types::rope_freqs, tensor_types::cache_k, tensor_types::cache_v, tensor_types::kq_mask };
};

template<llama_arch_config_types config_type_new> struct data_output_aggregator<config_type_new, core_types_new::attn_prep_and_score> {
	static constexpr array values{ tensor_types::kq_mul_mat };
};

template<llama_arch_config_types config_type_new> struct kernel_aggregator<config_type_new, core_types_new::attn_prep_and_score> {
	static constexpr array values{ tensor_types::norm_pre_attn_rms_norm, tensor_types::attn_norm_mul, tensor_types::qcur_mul_mat, tensor_types::qcur_reshape,
		tensor_types::qcur_rope, tensor_types::kcur_mul_mat, tensor_types::kcur_reshape, tensor_types::kcur_rope, tensor_types::vcur_mul_mat, tensor_types::k_cache_view,
		tensor_types::k_cache_cpy, tensor_types::vcur_transpose, tensor_types::v_cache_view, tensor_types::v_cache_cpy, tensor_types::v_from_cache_view,
		tensor_types::k_from_cache_view, tensor_types::q_permute, tensor_types::kq_mul_mat };
};

template<llama_arch_config_types config_type_new> struct data_input_aggregator<config_type_new, core_types_new::attn_out_and_ffn> {
	static constexpr array values{ tensor_types::kq_mul_mat, tensor_types::cache_v, tensor_types::kq_mask, tensor_types::attn_output, tensor_types::inp_embd_get_rows,
		tensor_types::ffn_norm, tensor_types::ffn_gate, tensor_types::ffn_up, tensor_types::ffn_down };
};

template<llama_arch_config_types config_type_new> struct data_output_aggregator<config_type_new, core_types_new::attn_out_and_ffn> {
	static constexpr array values{ tensor_types::layer_out_add };
};

template<llama_arch_config_types config_type_new> struct kernel_aggregator<config_type_new, core_types_new::attn_out_and_ffn> {
	static constexpr array values{ tensor_types::kq_soft_max, tensor_types::kqv_mul_mat, tensor_types::kqv_merged_permute, tensor_types::kqv_merged_cont,
		tensor_types::kqv_out_mul_mat, tensor_types::ffn_inp_add, tensor_types::norm_pre_ffn_rms_norm, tensor_types::ffn_norm_mul, tensor_types::ffn_gate_mul_mat,
		tensor_types::ffn_silu_unary, tensor_types::ffn_up_mul_mat, tensor_types::ffn_gate_par_mul, tensor_types::ffn_out_mul_mat, tensor_types::layer_out_add };
};

template<llama_arch_config_types config_type_new> struct data_input_aggregator<config_type_new, core_types_new::global_output_and_sampling> {
	static constexpr array values{ tensor_types::layer_out_add, tensor_types::kqv_out_mul_mat, tensor_types::inp_out_ids, tensor_types::ffn_norm, tensor_types::ffn_gate,
		tensor_types::ffn_up, tensor_types::ffn_down, tensor_types::output_norm, tensor_types::output, tensor_types::token_history, tensor_types::repetition_penalty,
		tensor_types::presence_penalty, tensor_types::frequency_penalty, tensor_types::rep_window, tensor_types::logits_bias, tensor_types::allowed_vocab_mask,
		tensor_types::temperature, tensor_types::top_k, tensor_types::top_p, tensor_types::rng_state };
};

template<llama_arch_config_types config_type_new> struct data_output_aggregator<config_type_new, core_types_new::global_output_and_sampling> {
	static constexpr array values{ tensor_types::sample_token };
};

template<llama_arch_config_types config_type_new> struct kernel_aggregator<config_type_new, core_types_new::global_output_and_sampling> {
	static constexpr array values{ tensor_types::node_1016_get_rows, tensor_types::node_1017_get_rows, tensor_types::final_ffn_inp_add, tensor_types::final_norm_pre_rms_norm,
		tensor_types::final_norm_mul, tensor_types::result_norm_mul, tensor_types::result_output_mul_mat, tensor_types::apply_repetition_penalty,
		tensor_types::apply_presence_penalty, tensor_types::apply_frequency_penalty, tensor_types::apply_logits_bias, tensor_types::apply_vocab_mask,
		tensor_types::apply_temperature, tensor_types::compute_softmax, tensor_types::apply_top_k_filter, tensor_types::apply_top_p_filter, tensor_types::sample_token };
};

template<typename config_type_new, core_types_new core_type_new> struct core_traits_new;

template<typename config_type_new> struct core_traits_new<config_type_new, core_types_new::weights>
	: public core_elem_base<core_types_new::weights, core_traits_new<config_type_new, core_types_new::weights>>, public sync_base_new<config_type_new, core_types_new::weights> {
	static constexpr core_types_new core_type{ core_types_new::weights };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
	using config_type				 = config_type_new;
	using cathedral_output_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_output_aggregator, data_interface>;
	cathedral_output_data_type output_data_vals{};
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types_new::global_inputs>
	: public core_elem_base<core_types_new::global_inputs, core_traits_new<config_type_new, core_types_new::global_inputs>>,
	  public sync_base_new<config_type_new, core_types_new::global_inputs> {
	static constexpr core_types_new core_type{ core_types_new::global_inputs };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
	using config_type				 = config_type_new;
	using cathedral_output_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_output_aggregator, data_interface>;
	cathedral_output_data_type output_data_vals{};
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types_new::token_embeddings>
	: public core_elem_base<core_types_new::token_embeddings, core_traits_new<config_type_new, core_types_new::token_embeddings>>,
	  public sync_base_new<config_type_new, core_types_new::token_embeddings> {
	static constexpr core_types_new core_type{ core_types_new::token_embeddings };
	static constexpr uint64_t depth{ 0 };
	using config_type				= config_type_new;
	using cathedral_input_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_input_aggregator, data_interface>;
	cathedral_input_data_type input_data_vals{};
	using cathedral_output_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_output_aggregator, data_interface>;
	cathedral_output_data_type output_data_vals{};
	using cathedral_kernel_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, kernel_aggregator, data_interface>;
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types_new::attn_prep_and_score>
	: public core_elem_base<core_types_new::attn_prep_and_score, core_traits_new<config_type_new, core_types_new::attn_prep_and_score>>,
	  public sync_base_new<config_type_new, core_types_new::attn_prep_and_score> {
	static constexpr core_types_new core_type{ core_types_new::attn_prep_and_score };
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types_new>(static_cast<uint64_t>(core_types_new::attn_prep_and_score) - 1)>::depth + 1 };
	using config_type				= config_type_new;
	using cathedral_input_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_input_aggregator, data_interface>;
	cathedral_input_data_type input_data_vals{};
	using cathedral_output_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_output_aggregator, data_interface>;
	cathedral_output_data_type output_data_vals{};
	using cathedral_kernel_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, kernel_aggregator, data_interface>;
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types_new::attn_out_and_ffn>
	: public core_elem_base<core_types_new::attn_out_and_ffn, core_traits_new<config_type_new, core_types_new::attn_out_and_ffn>>,
	  public sync_base_new<config_type_new, core_types_new::attn_out_and_ffn> {
	static constexpr core_types_new core_type{ core_types_new::attn_out_and_ffn };
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types_new>(static_cast<uint64_t>(core_types_new::attn_out_and_ffn) - 1)>::depth + 1 };
	using config_type				= config_type_new;
	using cathedral_input_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_input_aggregator, data_interface>;
	cathedral_input_data_type input_data_vals{};
	using cathedral_output_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_output_aggregator, data_interface>;
	cathedral_output_data_type output_data_vals{};
	using cathedral_kernel_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, kernel_aggregator, data_interface>;
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types_new::global_output_and_sampling>
	: public core_elem_base<core_types_new::global_output_and_sampling, core_traits_new<config_type_new, core_types_new::global_output_and_sampling>>,
	  public sync_base_new<config_type_new, core_types_new::global_output_and_sampling> {
	static constexpr core_types_new core_type{ core_types_new::global_output_and_sampling };
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types_new>(static_cast<uint64_t>(core_types_new::global_output_and_sampling) - 1)>::depth +
		1 };
	using config_type				= config_type_new;
	using cathedral_input_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_input_aggregator, data_interface>;
	cathedral_input_data_type input_data_vals{};
	using cathedral_output_data_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, data_output_aggregator, data_interface>;
	cathedral_output_data_type output_data_vals{};
	using cathedral_kernel_type = get_nihilus_cathedral_array_t<config_type_new, tensor_types, core_type, kernel_aggregator, data_interface>;
};

template<typename config_type, typename base_type_new> struct memory_planner_impl {
	NIHILUS_HOST memory_planner_impl() noexcept {
	}
	NIHILUS_HOST memory_planner_impl& operator=(const memory_planner_impl&) noexcept = delete;
	NIHILUS_HOST memory_planner_impl(const memory_planner_impl&) noexcept			 = delete;
	NIHILUS_HOST memory_planner_impl& operator=(memory_planner_impl&&) noexcept		 = delete;
	NIHILUS_HOST memory_planner_impl(memory_planner_impl&&) noexcept				 = delete;
	using base_type																	 = base_type_new;
	using base_derived_type															 = typename base_type::derived_type;

	NIHILUS_HOST static constexpr bool filter() {
		return true;
	}

	NIHILUS_HOST constexpr static void impl(const base_type&, uint64_t& memory_amount) {
		memory_amount += base_type::total_required_bytes;
	}
};

template<typename config_type, typename base_type_new> struct memory_planner {
	NIHILUS_HOST memory_planner() noexcept {
	}
	NIHILUS_HOST memory_planner& operator=(const memory_planner&) noexcept = delete;
	NIHILUS_HOST memory_planner(const memory_planner&) noexcept			   = delete;
	NIHILUS_HOST memory_planner& operator=(memory_planner&&) noexcept	   = delete;
	NIHILUS_HOST memory_planner(memory_planner&&) noexcept				   = delete;
	using base_type														   = base_type_new;
	NIHILUS_HOST static constexpr bool filter() {
		return base_type::core_type != core_types_new::weights || config_type::device_type == device_types::gpu;
	}

	NIHILUS_HOST constexpr static void impl(const base_type& parse_core, uint64_t& current_index, const memory_plan& values) {
		uint64_t internal_offset{};
		parse_core.output_data_vals.template impl<memory_planner_impl>(internal_offset);
		values.footprints[current_index].offset				  = values.currently_allocated_bytes;
		values.footprints[current_index].core_type			  = static_cast<core_types_new>(current_index);
		values.footprints[current_index].depth				  = base_type::depth;
		values.footprints[current_index].is_active			  = true;
		values.footprints[current_index].total_required_bytes = internal_offset;
		values.currently_allocated_bytes += internal_offset;
		if (values.currently_allocated_bytes > values.peak_allocated_bytes) {
			values.peak_allocated_bytes = values.currently_allocated_bytes;
		}
		constexpr uint64_t cur_depth = base_type::depth;
		if constexpr (cur_depth >= 2) {
			for (int64_t x = 0; x < static_cast<int64_t>(current_index); ++x) {
				const auto footprint = values.footprints[static_cast<uint64_t>(x)];
				uint64_t threshold	 = base_type::depth - 2;
				if (footprint.is_active && footprint.depth <= threshold && footprint.depth != std::numeric_limits<uint64_t>::max()) {
					values.footprints[static_cast<uint64_t>(x)].is_active = false;
					values.currently_allocated_bytes -= values.footprints[static_cast<uint64_t>(x)].total_required_bytes;
				}
			}
		}
		++current_index;
	}
};

template<typename config_type> static constexpr memory_plan nihilus_cathedral_memory_plan_new{ []() {
	get_nihilus_cathedral_enum_t<config_type, core_types_new, core_traits_new> cathedral{};
	uint64_t current_index{};
	memory_plan values{};
	cathedral.impl<memory_planner>(current_index, values);
	return values;
}() };

template<typename config_type_new, typename aggregator_type, core_types_new core_type, uint64_t current_index = 0>
uint64_t print_contained_kernel_dimensions_et_al(uint64_t current_byte_size = 0) {
	if constexpr (current_index < aggregator_type::values.size() - 1) {
		std::cout << "CURRENT TYPE: " << aggregator_type::values[current_index] << std::endl;
		auto dims = typename core_traits_new<config_type_new, core_type>::cathedral_output_data_type{}.template get_core<aggregator_type::values[current_index]>().dims;
		current_byte_size +=
			typename core_traits_new<config_type_new, core_type>::cathedral_output_data_type{}.template get_core<aggregator_type::values[current_index]>().total_required_bytes;
		std::cout << "DIMS: [" << dims[0] << "," << dims[1] << "," << dims[2] << "," << dims[3] << "]" << std::endl;
		return print_contained_kernel_dimensions_et_al<config_type_new, aggregator_type, core_type, current_index + 1>(current_byte_size);
	}
	return current_byte_size;
}

int32_t main(int32_t argc, char** argv) {
	try {
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		static constexpr auto model_config_00 =
			nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama,
				nihilus::device_types::cpu, nihilus::exception_type::enabled, nihilus::default_max_sequence_length_type{ 1024 }, nihilus::benchmark_type::enabled);
		cli_params cli_args = harbinger::parse_cli_arguments(argc, argv);
		//print_contained_kernel_dimensions_et_al<model_config_type<model_config_00>, kernel_aggregator<model_config_type<model_config_00>, core_types_new::token_embeddings>,
		//core_types_new::token_embeddings>();
		//print_contained_kernel_dimensions_et_al<model_config_type<model_config_00>, kernel_aggregator<model_config_type<model_config_00>, core_types_new::attn_prep_and_score>,
		//			core_types_new::attn_prep_and_score>();
		//print_contained_kernel_dimensions_et_al<model_config_type<model_config_00>, kernel_aggregator<model_config_type<model_config_00>, core_types_new::attn_out_and_ffn>,
		//			core_types_new::attn_out_and_ffn>();
		[[maybe_unused]] data_interface<tensor_types::attn_q, model_config_type<model_config_00>> test{};
		{
			[[maybe_unused]] core_traits_new<model_config_type<model_config_00>, core_types_new::weights> data_cathedral{};
			data_cathedral.output_data_vals.get_core<tensor_types::attn_q>();
		}
		{
			[[maybe_unused]] core_traits_new<model_config_type<model_config_00>, core_types_new::global_inputs> data_cathedral{};
			data_cathedral.output_data_vals.get_core<tensor_types::inp_pos>();
			//auto bytes = print_contained_kernel_dimensions_et_al<model_config_type<model_config_00>,
			//	data_output_aggregator<model_config_type<model_config_00>, core_types_new::global_inputs>, core_types_new::global_inputs>();
			//std::cout << "TOTAL BYTES: " << bytes << std::endl;
		}
		{
			[[maybe_unused]] core_traits_new<model_config_type<model_config_00>, core_types_new::token_embeddings> data_cathedral{};
			std::cout << "REQUIRED BYTES: " << data_cathedral.output_data_vals.get_core<tensor_types::inp_embd_get_rows>().total_required_bytes << std::endl;
		}
		{
			[[maybe_unused]] core_traits_new<model_config_type<model_config_00>, core_types_new::attn_prep_and_score> data_cathedral{};
			std::cout << "REQUIRED BYTES: " << data_cathedral.output_data_vals.get_core<tensor_types::kq_mul_mat>().total_required_bytes << std::endl;
		}
		{
			[[maybe_unused]] core_traits_new<model_config_type<model_config_00>, core_types_new::attn_out_and_ffn> data_cathedral{};
			std::cout << "REQUIRED BYTES: " << data_cathedral.output_data_vals.get_core<tensor_types::layer_out_add>().total_required_bytes << std::endl;
		}
		{
			[[maybe_unused]] core_traits_new<model_config_type<model_config_00>, core_types_new::global_output_and_sampling> data_cathedral{};
			std::cout << "REQUIRED BYTES: " << data_cathedral.output_data_vals.get_core<tensor_types::sample_token>().total_required_bytes << std::endl;
		}
		std::cout << "TOTAL REQUIRED BYTES: " << nihilus_cathedral_memory_plan_new<model_config_type<model_config_00>>.peak_allocated_bytes << std::endl;
		//nihilus::model_collection_type<model_config_00> collection{ cli_args };
		//collection.process_input(cli_args.prompt);
	} catch (const std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl;
	}
	return 0;
}
