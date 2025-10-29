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

template<typename value_type>
concept has_output_types = requires() { typename detail::remove_cvref_t<value_type>::output_type; };

template<typename value_type>
concept raw_kernel_traits_types = requires() { detail::remove_cvref_t<value_type>::raw_kernel_type; };

template<typename value_type>
concept kernel_traits_types_new = requires() {
	detail::remove_cvref_t<value_type>::kernel_type;
	typename detail::remove_cvref_t<value_type>::output_type;
};

template<typename value_type>
concept only_dims_types = kernel_dims_types<value_type> && !raw_kernel_traits_types<value_type> && !kernel_traits_types_new<value_type>;

template<typename config_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type, typename output_type_new> struct raw_data_type
	: data_mixin<config_type_new, data_strategy_type, output_type_new> {
	using output_type = typename output_type_new::output_type;
	using dims_type	  = output_type_new;
	static constexpr uint64_t total_required_bytes{ get_total_required_bytes<round_up_to_multiple<64>(type_traits<output_type>::total_byte_size(dims_type::get_array())),
		model_traits_type<config_type_new>::block_count, data_strategy_type> };
};

template<typename config_type_new, kernel_types kernel_type_new, typename output_type_new, kernel_dims_types dims_type_new> struct raw_kernel_traits : dims_type_new {
	using output_type = output_type_new;
	using dims_type	  = dims_type_new;
	static constexpr bool raw_kernel_type{ true };
	static constexpr kernel_types kernel_type{ kernel_type_new };
};

template<bool batched, kernel_types_types kernel_type_new, typename... input_types_new> struct kernel_traits_new;

template<bool batched, kernel_types_types kernel_type_new, has_output_types input_type_01> struct kernel_traits_new<batched, kernel_type_new, input_type_01>
	: public dim_traits<batched, kernel_type_new, input_type_01>::dims_type {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	using output_type = typename input_type_01::output_type;
};

template<bool batched, kernel_types_types kernel_type_new, only_dims_types input_dims_01, has_output_types input_type_01>
struct kernel_traits_new<batched, kernel_type_new, input_dims_01, input_type_01> : input_dims_01 {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	using output_type = typename input_type_01::output_type;
	static constexpr bool raw_kernel_type{ true };
};

template<bool batched, kernel_types_types kernel_type_new, only_dims_types dims_type_new, typename output_type_new, has_output_types input_type_01, has_output_types input_type_02>
struct kernel_traits_new<batched, kernel_type_new, dims_type_new, output_type_new, input_type_01, input_type_02> : dims_type_new {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	using output_type = output_type_new;
};

template<bool batched, kernel_types_types kernel_type_new, typename output_type_new, has_output_types input_type_01, has_output_types input_type_02>
struct kernel_traits_new<batched, kernel_type_new, output_type_new, input_type_01, input_type_02>
	: public dim_traits<batched, kernel_type_new, input_type_01, input_type_02>::dims_type {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	static constexpr bool quantized{ !std::is_same_v<output_type_new, typename input_type_01::output_type> };
	using output_type = output_type_new;
};

template<bool batched, kernel_types_types kernel_type_new, has_output_types input_type_01, has_output_types input_type_02>
struct kernel_traits_new<batched, kernel_type_new, input_type_01, input_type_02> : public dim_traits<batched, kernel_type_new, input_type_01, input_type_02>::dims_type {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	static constexpr bool quantized{ !std::is_same_v<typename input_type_01::output_type, typename input_type_02::output_type> };
	using output_type = typename input_type_01::output_type;
};

template<bool batched, kernel_types_types kernel_type_new, has_output_types input_type_01, has_output_types input_type_02, has_output_types input_type_03>
struct kernel_traits_new<batched, kernel_type_new, input_type_01, input_type_02, input_type_03>
	: public dim_traits<batched, kernel_type_new, input_type_01, input_type_02, input_type_03>::dims_type {
	static constexpr kernel_types kernel_type{ kernel_type_new::kernel_type };
	using output_type = typename input_type_01::output_type;
};

template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
	enum_types auto enum_value_new, kernel_traits_types_new... sub_kernel_types>
struct op_traits_new;

template<typename config_type_new, core_types core_type_new, allocation_strategy_types allocation_strategy_type, data_strategy_types data_strategy_type,
	enum_types auto enum_value_new, kernel_traits_types_new... sub_kernel_types_new>
struct op_traits_new
	: core_elem_base<enum_value_new, op_traits_new<config_type_new, core_type_new, allocation_strategy_type, data_strategy_type, enum_value_new, sub_kernel_types_new...>>,
	  raw_data_type<config_type_new, allocation_strategy_type, data_strategy_type, get_last_tuple_type<tuple<sub_kernel_types_new...>>>,
	  get_last_tuple_type<tuple<sub_kernel_types_new...>> {
	static constexpr decltype(enum_value_new) enum_value{ enum_value_new };
	static constexpr core_types core_type{ core_type_new };
	using enum_type		   = decltype(enum_value_new);
	using sub_kernel_types = tuple<sub_kernel_types_new...>;
	using output_type	   = typename get_last_tuple_type<sub_kernel_types>::output_type;
};

template<typename config_type_new, core_types> struct core_traits_new;

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::weights>
	: public core_elem_base<core_types::weights, core_traits_new<config_type_new, core_types::weights>> {
	static constexpr core_types core_type{ core_types::weights };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
	using config_type		  = config_type_new;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using weight_type		  = typename kernel_profile_type::weight_type;
	using norm_type			  = typename kernel_profile_type::norm_type;

	using attn_q_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::embedding_length, 1, 1>>;

	using attn_k_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::n_embd_kv_gqa, 1, 1>>;

	using attn_v_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::n_embd_kv_gqa, 1, 1>>;

	using attn_output_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::embedding_length, 1, 1>>;

	using attn_norm_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, norm_type, kernel_dims<0, model_traits_type<config_type>::embedding_length, 1, 1, 1>>;

	using ffn_gate_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::feed_forward_length, 1, 1>>;

	using ffn_up_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::feed_forward_length, 1, 1>>;

	using ffn_down_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::feed_forward_length, model_traits_type<config_type>::embedding_length, 1, 1>>;

	using ffn_norm_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, norm_type, kernel_dims<0, model_traits_type<config_type>::embedding_length, 1, 1, 1>>;

	using token_embd_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::vocab_size, 1, 1>>;

	using rope_freqs_weight_kernel =
		raw_kernel_traits<config_type, kernel_types::weights, norm_type, kernel_dims<0, model_traits_type<config_type>::rope_dimension_count / 2, 1, 1, 1>>;

	using output_norm_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, norm_type, kernel_dims<0, model_traits_type<config_type>::embedding_length, 1, 1, 1>>;

	using output_weight_kernel = raw_kernel_traits<config_type, kernel_types::weights, weight_type,
		kernel_dims<0, model_traits_type<config_type>::embedding_length, model_traits_type<config_type>::vocab_size, 1, 1>>;

	using attn_q_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::attn_q, attn_q_weight_kernel>;

	using attn_k_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::attn_k, attn_k_weight_kernel>;

	using attn_v_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::attn_v, attn_v_weight_kernel>;

	using attn_output_weight_type = op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block,
		weight_types::attn_output, attn_output_weight_kernel>;

	using attn_norm_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::attn_norm, attn_norm_weight_kernel>;

	using ffn_gate_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::ffn_gate, ffn_gate_weight_kernel>;

	using ffn_up_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::ffn_up, ffn_up_weight_kernel>;

	using ffn_down_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::ffn_down, ffn_down_weight_kernel>;

	using ffn_norm_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::per_block, weight_types::ffn_norm, ffn_norm_weight_kernel>;

	using token_embd_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::global, weight_types::token_embd, token_embd_weight_kernel>;

	using rope_freqs_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::global, weight_types::rope_freqs, rope_freqs_weight_kernel>;

	using output_norm_weight_type = op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::global,
		weight_types::output_norm, output_norm_weight_kernel>;

	using output_weight_type =
		op_traits_new<config_type, core_type, allocation_strategy_type<config_type::device_type>, data_strategy_types::global, weight_types::output, output_weight_kernel>;

	using composite_ops =
		get_nihilus_cathedral_t<config_type, attn_q_weight_type, attn_k_weight_type, attn_v_weight_type, attn_output_weight_type, attn_norm_weight_type, ffn_gate_weight_type,
			ffn_up_weight_type, ffn_down_weight_type, ffn_norm_weight_type, token_embd_weight_type, rope_freqs_weight_type, output_norm_weight_type, output_weight_type>;
	composite_ops values{};

	static constexpr uint64_t total_required_bytes{ attn_q_weight_type::total_required_bytes + attn_k_weight_type::total_required_bytes + attn_v_weight_type::total_required_bytes +
		attn_output_weight_type::total_required_bytes + attn_norm_weight_type::total_required_bytes + ffn_gate_weight_type::total_required_bytes +
		ffn_up_weight_type::total_required_bytes + ffn_down_weight_type::total_required_bytes + ffn_norm_weight_type::total_required_bytes +
		token_embd_weight_type::total_required_bytes + rope_freqs_weight_type::total_required_bytes + output_norm_weight_type::total_required_bytes +
		output_weight_type::total_required_bytes };

	static constexpr bool has_total_required_bytes{ config_type::device_type == device_types::gpu };
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::global_inputs>
	: public core_elem_base<core_types::global_inputs, core_traits_new<config_type_new, core_types::global_inputs>> {
	static constexpr core_types core_type{ core_types::global_inputs };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
	using config_type		  = config_type_new;
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;

	using inp_tokens_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::token_type,
		kernel_dims<get_runtime_mask<0>(), config_type::max_sequence_length, 1, 1, 1>>;

	using inp_pos_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::token_type,
		kernel_dims<get_runtime_mask<0>(), config_type::max_sequence_length, 1, 1, 1>>;

	using cache_k_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::kv_cache_type,
		kernel_dims<get_runtime_mask<1>(), mtt::block_count, config_type::max_sequence_length, mtt::attention_head_count_kv, mtt::rope_dimension_count>>;

	using cache_v_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::kv_cache_type,
		kernel_dims<get_runtime_mask<1>(), mtt::block_count, config_type::max_sequence_length, mtt::attention_head_count_kv, mtt::rope_dimension_count>>;

	using kq_mask_kernel =
		raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::mask_type, kernel_dims<0, mtt::block_count, mtt::block_count, 1, 1>>;

	using inp_out_ids_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::token_type,
		kernel_dims<get_runtime_mask<0>(), config_type::max_sequence_length, 1, 1, 1>>;

	using temperature_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<0, 1, 1, 1, 1>>;

	using top_k_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, int32_t, kernel_dims<0, 1, 1, 1, 1>>;

	using top_p_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<0, 1, 1, 1, 1>>;

	using repetition_penalty_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<0, 1, 1, 1, 1>>;

	using presence_penalty_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<0, 1, 1, 1, 1>>;

	using frequency_penalty_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<0, 1, 1, 1, 1>>;

	using rep_window_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, int32_t, kernel_dims<0, 1, 1, 1, 1>>;

	using token_history_kernel =
		raw_kernel_traits<config_type, kernel_types::global_inputs, int32_t, kernel_dims<get_runtime_mask<0>(), config_type::max_sequence_length, 1, 1, 1>>;

	using rng_state_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, uint64_t, kernel_dims<0, 2, 1, 1, 1>>;

	using logits_bias_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<0, mtt::vocab_size, 1, 1, 1>>;

	using allowed_vocab_mask_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, uint8_t, kernel_dims<0, mtt::vocab_size, 1, 1, 1>>;

	using inp_tokens_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::inp_tokens, inp_tokens_kernel>;

	using inp_pos_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::inp_pos, inp_pos_kernel>;

	using cache_k_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::cache_k, cache_k_kernel>;

	using cache_v_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::cache_v, cache_v_kernel>;

	using kq_mask_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::kq_mask, kq_mask_kernel>;

	using inp_out_ids_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::inp_out_ids, inp_out_ids_kernel>;

	using temperature_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::temperature, temperature_kernel>;

	using top_k_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::top_k, top_k_kernel>;

	using top_p_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::top_p, top_p_kernel>;

	using repetition_penalty_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::repetition_penalty, repetition_penalty_kernel>;

	using presence_penalty_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::presence_penalty, presence_penalty_kernel>;

	using frequency_penalty_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::frequency_penalty, frequency_penalty_kernel>;

	using rep_window_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::rep_window, rep_window_kernel>;

	using token_history_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::token_history, token_history_kernel>;

	using rng_state_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::rng_state, rng_state_kernel>;

	using logits_bias_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::logits_bias, logits_bias_kernel>;

	using allowed_vocab_mask_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::allowed_vocab_mask, allowed_vocab_mask_kernel>;

	using composite_ops =
		get_nihilus_cathedral_t<config_type, inp_tokens_type, inp_pos_type, cache_k_type, cache_v_type, kq_mask_type, inp_out_ids_type, temperature_type, top_k_type, top_p_type,
			repetition_penalty_type, presence_penalty_type, frequency_penalty_type, rep_window_type, token_history_type, rng_state_type, logits_bias_type, allowed_vocab_mask_type>;

	composite_ops values{};

	static constexpr uint64_t total_required_bytes{ inp_tokens_type::total_required_bytes + inp_pos_type::total_required_bytes + cache_k_type::total_required_bytes +
		cache_v_type::total_required_bytes + kq_mask_type::total_required_bytes + inp_out_ids_type::total_required_bytes + temperature_type::total_required_bytes +
		top_k_type::total_required_bytes + top_p_type::total_required_bytes + repetition_penalty_type::total_required_bytes + presence_penalty_type::total_required_bytes +
		frequency_penalty_type::total_required_bytes + rep_window_type::total_required_bytes + token_history_type::total_required_bytes + rng_state_type::total_required_bytes +
		logits_bias_type::total_required_bytes + allowed_vocab_mask_type::total_required_bytes };

	static constexpr bool has_total_required_bytes{ true };
};

template<batched_processing_config_types config_type_new> struct core_traits_new<config_type_new, core_types::global_inputs>
	: public core_elem_base<core_types::global_inputs, core_traits_new<config_type_new, core_types::global_inputs>> {
	static constexpr core_types core_type{ core_types::global_inputs };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };
	using config_type		  = config_type_new;
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;

	using inp_tokens_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::token_type,
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, 1, 1>>;

	using inp_pos_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::token_type,
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, 1, 1>>;

	using cache_k_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::kv_cache_type,
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size * mtt::block_count, config_type::max_sequence_length, mtt::attention_head_count_kv,
			mtt::rope_dimension_count>>;

	using cache_v_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::kv_cache_type,
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size * mtt::block_count, config_type::max_sequence_length, mtt::attention_head_count_kv,
			mtt::rope_dimension_count>>;

	using kq_mask_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::mask_type,
		kernel_dims<get_runtime_mask<0>(), config_type::batch_size, mtt::block_count, mtt::block_count, 1>>;

	using inp_out_ids_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, typename kernel_profile_type::token_type,
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, 1, 1>>;

	using temperature_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using top_k_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, int32_t, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using top_p_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using repetition_penalty_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using presence_penalty_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using frequency_penalty_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using rep_window_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, int32_t, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 1, 1, 1>>;

	using token_history_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, int32_t,
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, 1, 1>>;

	using rng_state_kernel = raw_kernel_traits<config_type, kernel_types::global_inputs, uint64_t, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, 2, 1, 1>>;

	using logits_bias_kernel =
		raw_kernel_traits<config_type, kernel_types::global_inputs, float, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, mtt::vocab_size, 1, 1>>;

	using allowed_vocab_mask_kernel =
		raw_kernel_traits<config_type, kernel_types::global_inputs, uint8_t, kernel_dims<get_runtime_mask<0>(), config_type::batch_size, mtt::vocab_size, 1, 1>>;

	using inp_tokens_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::inp_tokens, inp_tokens_kernel>;

	using inp_pos_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::inp_pos, inp_pos_kernel>;

	using cache_k_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::cache_k, cache_k_kernel>;

	using cache_v_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::cache_v, cache_v_kernel>;

	using kq_mask_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::kq_mask, kq_mask_kernel>;

	using inp_out_ids_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::inp_out_ids, inp_out_ids_kernel>;

	using temperature_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::temperature, temperature_kernel>;

	using top_k_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::top_k, top_k_kernel>;

	using top_p_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::top_p, top_p_kernel>;

	using repetition_penalty_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::repetition_penalty, repetition_penalty_kernel>;

	using presence_penalty_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::presence_penalty, presence_penalty_kernel>;

	using frequency_penalty_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::frequency_penalty, frequency_penalty_kernel>;

	using rep_window_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::rep_window, rep_window_kernel>;

	using token_history_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::token_history, token_history_kernel>;

	using rng_state_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::rng_state, rng_state_kernel>;

	using logits_bias_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::logits_bias, logits_bias_kernel>;

	using allowed_vocab_mask_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, global_input_types::allowed_vocab_mask, allowed_vocab_mask_kernel>;

	using composite_ops =
		get_nihilus_cathedral_t<config_type, inp_tokens_type, inp_pos_type, cache_k_type, cache_v_type, kq_mask_type, inp_out_ids_type, temperature_type, top_k_type, top_p_type,
			repetition_penalty_type, presence_penalty_type, frequency_penalty_type, rep_window_type, token_history_type, rng_state_type, logits_bias_type, allowed_vocab_mask_type>;

	composite_ops values{};

	static constexpr uint64_t total_required_bytes{ inp_tokens_type::total_required_bytes + inp_pos_type::total_required_bytes + cache_k_type::total_required_bytes +
		cache_v_type::total_required_bytes + kq_mask_type::total_required_bytes + inp_out_ids_type::total_required_bytes + temperature_type::total_required_bytes +
		top_k_type::total_required_bytes + top_p_type::total_required_bytes + repetition_penalty_type::total_required_bytes + presence_penalty_type::total_required_bytes +
		frequency_penalty_type::total_required_bytes + rep_window_type::total_required_bytes + token_history_type::total_required_bytes + rng_state_type::total_required_bytes +
		logits_bias_type::total_required_bytes + allowed_vocab_mask_type::total_required_bytes };

	static constexpr bool has_total_required_bytes{ true };
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::token_embeddings>
	: public core_elem_base<core_types::token_embeddings, core_traits_new<config_type_new, core_types::token_embeddings>>,
	  public sync_base<config_type_new, core_types::token_embeddings> {
	static constexpr core_types core_type{ core_types::token_embeddings };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ 0 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using token_embd_type	  = typename core_traits_new<config_type_new, core_types::weights>::token_embd_weight_type;
	using inp_tokens_type	  = typename core_traits_new<config_type_new, core_types::global_inputs>::inp_tokens_type;

	using inp_embeddings_kernel_traits =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::get_rows>, compute_type, token_embd_type, inp_tokens_type>;

	using inp_embeddings_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::per_block, token_embeddings_types::get_rows, inp_embeddings_kernel_traits>;

	using composite_ops = get_nihilus_cathedral_t<config_type, inp_embeddings_type>;
	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_type>;
	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ inp_embeddings_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::mega_qkv_prep_and_cache_publish>
	: public core_elem_base<core_types::mega_qkv_prep_and_cache_publish, core_traits_new<config_type_new, core_types::mega_qkv_prep_and_cache_publish>>,
	  public sync_base<config_type_new, core_types::mega_qkv_prep_and_cache_publish> {
	static constexpr core_types core_type{ core_types::mega_qkv_prep_and_cache_publish };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_qkv_prep_and_cache_publish) - 1)>::depth + 1 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using weight_type		  = typename kernel_profile_type::weight_type;
	using norm_type			  = typename kernel_profile_type::norm_type;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using kv_store_type		  = typename kernel_profile_type::kv_cache_type;
	using inp_embd_type		  = typename core_traits_new<config_type_new, core_types::token_embeddings>::inp_embeddings_type;
	using attn_norm_w_type	  = typename core_traits_new<config_type_new, core_types::weights>::attn_norm_weight_type;
	using attn_q_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::attn_q_weight_type;
	using attn_k_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::attn_k_weight_type;
	using attn_v_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::attn_v_weight_type;
	using inp_pos_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::inp_pos_type;
	using rope_freqs_type	  = typename core_traits_new<config_type_new, core_types::weights>::rope_freqs_weight_type;
	using cache_k_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::cache_k_type;
	using cache_v_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::cache_v_type;

	using rms_norm_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rms_norm>, inp_embd_type>;

	using mul_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul>, rms_norm_trait, attn_norm_w_type>;

	using q_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_q_w_type, mul_trait>;

	using k_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_k_w_type, mul_trait>;

	using v_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_v_w_type, mul_trait>;

	using q_reshape_dims  = kernel_dims<get_runtime_mask<2>(), mtt::rope_dimension_count, mtt::attention_head_count, config_type::max_sequence_length, 1>;
	using q_reshape_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::reshape>, q_reshape_dims, q_mul_mat_trait>;

	using k_reshape_dims  = kernel_dims<get_runtime_mask<2>(), mtt::rope_dimension_count, mtt::attention_head_count_kv, config_type::max_sequence_length, 1>;
	using k_reshape_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::reshape>, k_reshape_dims, k_mul_mat_trait>;

	using q_rope_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rope>, q_reshape_trait, inp_pos_type, rope_freqs_type>;

	using k_rope_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rope>, k_reshape_trait, inp_pos_type, rope_freqs_type>;

	using v_transpose_dims	= kernel_dims<get_runtime_mask<0>(), config_type::max_sequence_length, mtt::n_embd_kv_gqa, 1, 1>;
	using v_transpose_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::transpose>, v_transpose_dims, v_mul_mat_trait>;

	using k_cache_window_dims		= kernel_dims<get_runtime_mask<1>(), mtt::rope_dimension_count, config_type::max_sequence_length, mtt::attention_head_count_kv, 1>;
	using k_cache_window_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, k_cache_window_dims, cache_k_type>;

	using v_cache_window_dims		= kernel_dims<get_runtime_mask<0>(), config_type::max_sequence_length, mtt::rope_dimension_count, mtt::attention_head_count_kv, 1>;
	using v_cache_window_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, v_cache_window_dims, cache_v_type>;

	using k_cache_store_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::copy>, k_rope_trait, k_cache_window_view_trait>;

	using v_cache_store_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::copy>, v_transpose_trait, v_cache_window_view_trait>;

	using q_out_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, mega_qkv_prep_and_cache_publish_types::q_out,
		inp_embd_type, rms_norm_trait, mul_trait, k_mul_mat_trait, k_reshape_trait, k_rope_trait, k_cache_window_view_trait, k_cache_store_trait, v_mul_mat_trait,
		v_transpose_trait, v_cache_window_view_trait, v_cache_store_trait, q_mul_mat_trait, q_reshape_trait, q_rope_trait>;

	using composite_ops = get_nihilus_cathedral_t<config_type, q_out_type>;

	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_types::mega_qkv_prep_and_cache_publish>;

	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ q_out_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

template<batched_processing_config_types config_type_new> struct core_traits_new<config_type_new, core_types::mega_qkv_prep_and_cache_publish>
	: public core_elem_base<core_types::mega_qkv_prep_and_cache_publish, core_traits_new<config_type_new, core_types::mega_qkv_prep_and_cache_publish>>,
	  public sync_base<config_type_new, core_types::mega_qkv_prep_and_cache_publish> {
	static constexpr core_types core_type{ core_types::mega_qkv_prep_and_cache_publish };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_qkv_prep_and_cache_publish) - 1)>::depth + 1 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using weight_type		  = typename kernel_profile_type::weight_type;
	using norm_type			  = typename kernel_profile_type::norm_type;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using kv_store_type		  = typename kernel_profile_type::kv_cache_type;
	using inp_embd_type		  = typename core_traits_new<config_type_new, core_types::token_embeddings>::inp_embeddings_type;
	using attn_norm_w_type	  = typename core_traits_new<config_type_new, core_types::weights>::attn_norm_weight_type;
	using attn_q_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::attn_q_weight_type;
	using attn_k_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::attn_k_weight_type;
	using attn_v_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::attn_v_weight_type;
	using inp_pos_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::inp_pos_type;
	using rope_freqs_type	  = typename core_traits_new<config_type_new, core_types::weights>::rope_freqs_weight_type;
	using cache_k_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::cache_k_type;
	using cache_v_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::cache_v_type;

	using rms_norm_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rms_norm>, inp_embd_type>;

	using mul_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul>, rms_norm_trait, attn_norm_w_type>;

	using q_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_q_w_type, mul_trait>;

	using k_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_k_w_type, mul_trait>;

	using v_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_v_w_type, mul_trait>;

	using q_reshape_dims =
		kernel_dims<get_runtime_mask<0, 3>(), config_type_new::batch_size, mtt::rope_dimension_count, mtt::attention_head_count, config_type::max_sequence_length>;
	using q_reshape_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::reshape>, q_reshape_dims, q_mul_mat_trait>;

	using k_reshape_dims =
		kernel_dims<get_runtime_mask<0, 3>(), config_type::batch_size, mtt::rope_dimension_count, mtt::attention_head_count_kv, config_type::max_sequence_length>;
	using k_reshape_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::reshape>, k_reshape_dims, k_mul_mat_trait>;

	using q_rope_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rope>, q_reshape_trait, inp_pos_type, rope_freqs_type>;

	using k_rope_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rope>, k_reshape_trait, inp_pos_type, rope_freqs_type>;

	using v_transpose_dims	= kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, mtt::n_embd_kv_gqa, 1>;
	using v_transpose_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::transpose>, v_transpose_dims, v_mul_mat_trait>;

	using k_cache_window_dims =
		kernel_dims<get_runtime_mask<0, 2>(), config_type::batch_size, mtt::rope_dimension_count, config_type::max_sequence_length, mtt::attention_head_count_kv>;
	using k_cache_window_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, k_cache_window_dims, cache_k_type>;

	using v_cache_window_dims =
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, mtt::rope_dimension_count, mtt::attention_head_count_kv>;
	using v_cache_window_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, v_cache_window_dims, cache_v_type>;

	using k_cache_store_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::copy>, k_rope_trait, k_cache_window_view_trait>;

	using v_cache_store_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::copy>, v_transpose_trait, v_cache_window_view_trait>;

	using q_out_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, mega_qkv_prep_and_cache_publish_types::q_out,
		inp_embd_type, rms_norm_trait, mul_trait, k_mul_mat_trait, k_reshape_trait, k_rope_trait, k_cache_window_view_trait, k_cache_store_trait, v_mul_mat_trait,
		v_transpose_trait, v_cache_window_view_trait, v_cache_store_trait, q_mul_mat_trait, q_reshape_trait, q_rope_trait>;

	using composite_ops = get_nihilus_cathedral_t<config_type, q_out_type>;

	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_types::mega_qkv_prep_and_cache_publish>;

	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ q_out_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::mega_attention_apply>
	: public core_elem_base<core_types::mega_attention_apply, core_traits_new<config_type_new, core_types::mega_attention_apply>>,
	  public sync_base<config_type_new, core_types::mega_attention_apply> {
	static constexpr core_types core_type{ core_types::mega_attention_apply };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_attention_apply) - 1)>::depth + 1 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using weight_type		  = typename kernel_profile_type::weight_type;
	using norm_type			  = typename kernel_profile_type::norm_type;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using kv_store_type		  = typename kernel_profile_type::kv_cache_type;
	using cache_k_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::cache_k_type;
	using cache_v_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::cache_v_type;
	using kq_mask_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::kq_mask_type;
	using attn_output_w_type  = typename core_traits_new<config_type_new, core_types::weights>::attn_output_weight_type;
	using inp_embd_type		  = typename core_traits_new<config_type_new, core_types::token_embeddings>::inp_embeddings_type;
	using q_rope_type		  = typename core_traits_new<config_type_new, core_types::mega_qkv_prep_and_cache_publish>::q_out_type;

	using k_cache_read_dims =
		kernel_dims<get_runtime_mask<0, 2>(), config_type::batch_size, mtt::rope_dimension_count, config_type::max_sequence_length, mtt::attention_head_count_kv>;
	using k_cache_read_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, k_cache_read_dims, cache_k_type>;

	using v_cache_read_dims =
		kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, config_type::max_sequence_length, mtt::rope_dimension_count, mtt::attention_head_count_kv>;
	using v_cache_read_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, v_cache_read_dims, cache_v_type>;

	using kq_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, k_cache_read_view_trait, q_rope_type>;

	using softmax_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::softmax>, kq_mul_mat_trait, kq_mask_type>;

	using kqv_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, v_cache_read_view_trait, softmax_trait>;

	using merge_permute_dims =
		kernel_dims<get_runtime_mask<0, 3>(), config_type::batch_size, mtt::rope_dimension_count, mtt::attention_head_count, config_type::max_sequence_length>;
	using merge_permute_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::permute>, merge_permute_dims, kqv_mul_mat_trait>;

	using cont_dims	 = kernel_dims<get_runtime_mask<0, 2>(), config_type::batch_size, mtt::embedding_length, config_type::max_sequence_length, 1>;
	using cont_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::cont>, cont_dims, merge_permute_trait>;

	using attn_out_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, attn_output_w_type, cont_trait>;

	using residual_add_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::add>, attn_out_mul_mat_trait, inp_embd_type>;

	using ffn_inp_type =
		op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, mega_attention_apply_types::ffn_inp, k_cache_read_view_trait,
			v_cache_read_view_trait, kq_mul_mat_trait, softmax_trait, kqv_mul_mat_trait, merge_permute_trait, cont_trait, attn_out_mul_mat_trait, residual_add_trait>;

	using composite_ops = get_nihilus_cathedral_t<config_type, ffn_inp_type>;
	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_types::mega_attention_apply>;

	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ ffn_inp_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::mega_ffn>
	: public core_elem_base<core_types::mega_ffn, core_traits_new<config_type_new, core_types::mega_ffn>>, public sync_base<config_type_new, core_types::mega_ffn> {
	static constexpr core_types core_type{ core_types::mega_ffn };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types>(static_cast<uint64_t>(core_types::mega_ffn) - 1)>::depth + 1 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using ffn_norm_w_type	  = typename core_traits_new<config_type_new, core_types::weights>::ffn_norm_weight_type;
	using ffn_gate_w_type	  = typename core_traits_new<config_type_new, core_types::weights>::ffn_gate_weight_type;
	using ffn_up_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::ffn_up_weight_type;
	using ffn_down_w_type	  = typename core_traits_new<config_type_new, core_types::weights>::ffn_down_weight_type;

	using ffn_inp_type = typename core_traits_new<config_type_new, core_types::mega_attention_apply>::ffn_inp_type;

	using ffn_rms_norm_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::rms_norm>, compute_type, ffn_inp_type>;

	using ffn_norm_mul_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::mul>, compute_type, ffn_rms_norm_trait, ffn_norm_w_type>;

	using ffn_gate_mul_mat_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, compute_type, ffn_gate_w_type, ffn_norm_mul_trait>;

	using ffn_up_mul_mat_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, compute_type, ffn_up_w_type, ffn_norm_mul_trait>;

	using ffn_silu_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::silu>, compute_type, ffn_gate_mul_mat_trait>;

	using ffn_gate_par_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::mul>, compute_type, ffn_silu_trait, ffn_up_mul_mat_trait>;

	using ffn_down_mul_mat_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, compute_type, ffn_down_w_type, ffn_gate_par_trait>;

	using ffn_residual_add_trait = kernel_traits<config_type::batched_processing, kernel_types_type<kernel_types::add>, compute_type, ffn_down_mul_mat_trait, ffn_inp_type>;

	using l_out_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global, mega_ffn_types::l_out, ffn_rms_norm_trait,
		ffn_norm_mul_trait, ffn_gate_mul_mat_trait, ffn_up_mul_mat_trait, ffn_silu_trait, ffn_gate_par_trait, ffn_down_mul_mat_trait, ffn_residual_add_trait>;

	using composite_ops = get_nihilus_cathedral_t<config_type, l_out_type>;
	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_types::mega_ffn>;

	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ l_out_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

template<typename config_type_new> struct core_traits_new<config_type_new, core_types::final_norm_and_sampling>
	: public core_elem_base<core_types::final_norm_and_sampling, core_traits_new<config_type_new, core_types::final_norm_and_sampling>>,
	  public sync_base<config_type_new, core_types::final_norm_and_sampling> {
	static constexpr core_types core_type{ core_types::final_norm_and_sampling };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types>(static_cast<uint64_t>(core_types::final_norm_and_sampling) - 1)>::depth + 1 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using output_norm_w_type  = typename core_traits_new<config_type_new, core_types::weights>::output_norm_weight_type;
	using output_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::output_weight_type;

	using temperature_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::temperature_type;
	using top_k_type			  = typename core_traits_new<config_type_new, core_types::global_inputs>::top_k_type;
	using top_p_type			  = typename core_traits_new<config_type_new, core_types::global_inputs>::top_p_type;
	using repetition_penalty_type = typename core_traits_new<config_type_new, core_types::global_inputs>::repetition_penalty_type;
	using presence_penalty_type	  = typename core_traits_new<config_type_new, core_types::global_inputs>::presence_penalty_type;
	using frequency_penalty_type  = typename core_traits_new<config_type_new, core_types::global_inputs>::frequency_penalty_type;
	using rep_window_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::rep_window_type;
	using token_history_type	  = typename core_traits_new<config_type_new, core_types::global_inputs>::token_history_type;
	using rng_state_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::rng_state_type;
	using logits_bias_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::logits_bias_type;
	using allowed_vocab_mask_type = typename core_traits_new<config_type_new, core_types::global_inputs>::allowed_vocab_mask_type;

	using l_out_type_from_ffn = typename core_traits_new<config_type_new, core_types::mega_ffn>::l_out_type;

	using last_token_view_dims	= kernel_dims<get_runtime_mask<0>(), mtt::embedding_length, 1, 1, 1>;
	using last_token_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, last_token_view_dims, l_out_type_from_ffn>;

	using final_rms_norm_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rms_norm>, last_token_view_trait>;

	using final_norm_mul_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul>, final_rms_norm_trait, output_norm_w_type>;

	using logits_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, output_w_type, final_norm_mul_trait>;

	using logits_bias_add_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::add>, logits_mul_mat_trait, logits_bias_type>;

	using rep_penalty_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::repetition_penalty>, logits_bias_add_trait, token_history_type, repetition_penalty_type>;

	using presence_penalty_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::presence_penalty>, rep_penalty_trait, token_history_type, presence_penalty_type>;

	using frequency_penalty_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::frequency_penalty>, presence_penalty_trait, token_history_type, frequency_penalty_type>;

	using vocab_mask_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::vocab_mask>, frequency_penalty_trait, allowed_vocab_mask_type>;

	using temperature_scale_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::temperature_scale>, vocab_mask_trait, temperature_type>;

	using top_k_filter_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::top_k_filter>, temperature_scale_trait, top_k_type>;

	using top_p_filter_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::top_p_filter>, top_k_filter_trait, top_p_type>;

	using sample_dims = kernel_dims<1, 1, 1, 1, 1>;
	using sample_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::sample_logits>, sample_dims, int32_t, top_p_filter_trait, rng_state_type>;

	using result_token_id_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global,
		final_norm_and_sampling_types::result_token_id, last_token_view_trait, final_rms_norm_trait, final_norm_mul_trait, logits_mul_mat_trait, logits_bias_add_trait,
		rep_penalty_trait, presence_penalty_trait, frequency_penalty_trait, vocab_mask_trait, temperature_scale_trait, top_k_filter_trait, top_p_filter_trait, sample_trait>;

	using composite_ops = get_nihilus_cathedral_t<config_type, result_token_id_type>;
	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_types::final_norm_and_sampling>;

	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ result_token_id_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

template<batched_processing_config_types config_type_new> struct core_traits_new<config_type_new, core_types::final_norm_and_sampling>
	: public core_elem_base<core_types::final_norm_and_sampling, core_traits_new<config_type_new, core_types::final_norm_and_sampling>>,
	  public sync_base<config_type_new, core_types::final_norm_and_sampling> {
	static constexpr core_types core_type{ core_types::final_norm_and_sampling };
	using config_type = config_type_new;
	static constexpr uint64_t depth{ core_traits_new<config_type_new, static_cast<core_types>(static_cast<uint64_t>(core_types::final_norm_and_sampling) - 1)>::depth + 1 };
	using mtt				  = model_traits_type<config_type>;
	using kernel_profile_type = kernel_type_profile_traits<config_type::kernel_type_profile>;
	using compute_type		  = typename kernel_profile_type::compute_type;
	using output_norm_w_type  = typename core_traits_new<config_type_new, core_types::weights>::output_norm_weight_type;
	using output_w_type		  = typename core_traits_new<config_type_new, core_types::weights>::output_weight_type;

	using temperature_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::temperature_type;
	using top_k_type			  = typename core_traits_new<config_type_new, core_types::global_inputs>::top_k_type;
	using top_p_type			  = typename core_traits_new<config_type_new, core_types::global_inputs>::top_p_type;
	using repetition_penalty_type = typename core_traits_new<config_type_new, core_types::global_inputs>::repetition_penalty_type;
	using presence_penalty_type	  = typename core_traits_new<config_type_new, core_types::global_inputs>::presence_penalty_type;
	using frequency_penalty_type  = typename core_traits_new<config_type_new, core_types::global_inputs>::frequency_penalty_type;
	using rep_window_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::rep_window_type;
	using token_history_type	  = typename core_traits_new<config_type_new, core_types::global_inputs>::token_history_type;
	using rng_state_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::rng_state_type;
	using logits_bias_type		  = typename core_traits_new<config_type_new, core_types::global_inputs>::logits_bias_type;
	using allowed_vocab_mask_type = typename core_traits_new<config_type_new, core_types::global_inputs>::allowed_vocab_mask_type;

	using l_out_type_from_ffn = typename core_traits_new<config_type_new, core_types::mega_ffn>::l_out_type;

	using last_token_view_dims	= kernel_dims<get_runtime_mask<0, 1>(), config_type::batch_size, mtt::embedding_length, 1, 1>;
	using last_token_view_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::view>, last_token_view_dims, l_out_type_from_ffn>;

	using final_rms_norm_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::rms_norm>, last_token_view_trait>;

	using final_norm_mul_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul>, final_rms_norm_trait, output_norm_w_type>;

	using logits_mul_mat_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::mul_mat>, output_w_type, final_norm_mul_trait>;

	using logits_bias_add_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::add>, logits_mul_mat_trait, logits_bias_type>;

	using rep_penalty_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::repetition_penalty>, logits_bias_add_trait, token_history_type, repetition_penalty_type>;

	using presence_penalty_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::presence_penalty>, rep_penalty_trait, token_history_type, presence_penalty_type>;

	using frequency_penalty_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::frequency_penalty>, presence_penalty_trait, token_history_type, frequency_penalty_type>;

	using vocab_mask_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::vocab_mask>, frequency_penalty_trait, allowed_vocab_mask_type>;

	using temperature_scale_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::temperature_scale>, vocab_mask_trait, temperature_type>;

	using top_k_filter_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::top_k_filter>, temperature_scale_trait, top_k_type>;

	using top_p_filter_trait = kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::top_p_filter>, top_k_filter_trait, top_p_type>;

	using sample_dims = kernel_dims<1, 1, 1, 1, 1>;
	using sample_trait =
		kernel_traits_new<config_type::batched_processing, kernel_types_type<kernel_types::sample_logits>, sample_dims, int32_t, top_p_filter_trait, rng_state_type>;

	using result_token_id_type = op_traits_new<config_type, core_type, allocation_strategy_types::alloc, data_strategy_types::global,
		final_norm_and_sampling_types::result_token_id, last_token_view_trait, final_rms_norm_trait, final_norm_mul_trait, logits_mul_mat_trait, logits_bias_add_trait,
		rep_penalty_trait, presence_penalty_trait, frequency_penalty_trait, vocab_mask_trait, temperature_scale_trait, top_k_filter_trait, top_p_filter_trait, sample_trait>;

	using composite_ops = get_nihilus_cathedral_t<config_type, result_token_id_type>;
	composite_ops values{};

	using kernel_data_ptrs_type = kernel_data_ptrs<core_types::final_norm_and_sampling>;

	kernel_data_ptrs_type data_ptrs{};

	static constexpr uint64_t total_required_bytes{ result_token_id_type::total_required_bytes };
	static constexpr bool has_total_required_bytes{ true };
};

#include <bitset>

int32_t main(int32_t argc, char** argv) {
	try {
		std::cout << "BITS SET: " << std::bitset<4>{ get_runtime_mask<0, 2>() } << std::endl;
		//test_function();
		static constexpr model_config config{};
		static constexpr model_config config_02{ generate_model_config(batched_processing_type::enabled, batch_size_type{ 244 }) };
		std::cout << "ENABLED?: " << config.batched_processing << std::endl;
		std::cout << "ENABLED?: " << config_02.batch_size << std::endl;
		{
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::weights> core_traits00{};
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::global_inputs> core_traits01{};
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::token_embeddings> core_traits02{};
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::mega_qkv_prep_and_cache_publish> core_traits03{};
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::mega_attention_apply> core_traits04{};
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::mega_ffn> core_traits05{};
			[[maybe_unused]] core_traits_new<model_config_type<config>, core_types::final_norm_and_sampling> core_traits06{};
		}
		{
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::weights> core_traits00{};
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::global_inputs> core_traits01{};
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::token_embeddings> core_traits02{};
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::mega_qkv_prep_and_cache_publish> core_traits03{};
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::mega_attention_apply> core_traits04{};
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::mega_ffn> core_traits05{};
			[[maybe_unused]] core_traits_new<model_config_type<config_02>, core_types::final_norm_and_sampling> core_traits06{};
		}

		print_memory_bandwidth<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_405B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_70B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1, kernel_type_profiles::q8_gqa>();
		print_memory_bandwidth<model_arches::llama, model_sizes::llm_8B, model_generations::v3_1, kernel_type_profiles::fp16_mha>();
		//std::cout << "TOTAL REQUIREDS BYTES: " << attn_q.total_required_bytes_new << std::endl;
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
