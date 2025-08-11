#include <nihilus>

using namespace nihilus;

enum class compute_soverignties {
	// inputs: token_embd.weight (q8_0), inp_tokens (i32)
	// get_rows (inp_embd): [embedding_length × 1] from [embedding_length × vocab_size], output: f32
	barrier_0_initial_embedding,

	// inputs: inp_embd (f32), blk.n.attn_norm.weight (f32)
	// rms_norm_mul (attn_norm-n): [embedding_length × 1], output: q8_0
	barrier_1_layer_norm,

	// inputs: blk.n.attn_q.weight (q8_0), attn_norm-n (q8_0)
	// mul_mat_reshape (qcur-n): [embedding_length × 1] → [rope_dimension_count × attention_head_count × 1], output: f32
	// inputs: blk.n.attn_k.weight (q8_0), attn_norm-n (q8_0)
	// mul_mat_reshape (kcur-n): [n_embd_kv_gqa × 1] → [rope_dimension_count × attention_head_count_kv × 1], output: f32
	// inputs: blk.n.attn_v.weight (q8_0), attn_norm-n (q8_0) v_cache_view-n (f16)
	// mul_mat_transpose_copy (vcur-n): [1 × n_embd_kv_gqa], output: f16
	barrier_2_qkv_projections,

	// inputs: qcur-n (f32), inp_pos (i32), rope_freqs.weight (f32)
	// rope_permute (qcur-n): [rope_dimension_count × 1 × attention_head_count × 1], output: f32
	// inputs: kcur-n (f32), inp_pos (i32), rope_freqs.weight (f32) cache_k_ln (f16)
	// rope_copy (kcur-n): [n_embd_kv_gqa × 1], output: f16
	barrier_3_reshape_rope,

	// inputs: k-n (f16), q-n (f32)
	// mul_mat (kq-n): [rope_dimension_count × seq_len × attention_head_count_kv] × [rope_dimension_count × 1 × attention_head_count] = [seq_len × 1 × attention_head_count × 1], output: f32
	barrier_4_attention_scores,

	// inputs: kq-n (f32), kq_mask (f32)
	// soft_max (kq_soft_max_ext-n): [seq_len × 1 × attention_head_count × 1], output: f32
	barrier_5_attention_softmax,

	// inputs: v-n (f16), kq_soft_max_ext-n (f32)
	// mul_mat_permute_cont (kqv-n): [rope_dimension_count × attention_head_count × 1 × 1] → [embedding_length × 1], output: q8_0
	barrier_6_attention_output,

	// inputs: blk.n.attn_output.weight (q8_0), kqv-n (q8_0)
	// mul_mat (kqv_out-n): [embedding_length × embedding_length] × [embedding_length × 1] = [embedding_length × 1], output: f32
	barrier_7_output_projection,

	// inputs: kqv_out-n (f32), inp_embd (f32), blk.n.ffn_norm.weight (f32)
	// add_rms_norm_mul (ffn_norm-n): [embedding_length × 1], output: q8_0
	barrier_8_ffn_norm,

	// parallel execution:
	// inputs: blk.n.ffn_gate.weight (q8_0), ffn_norm-n (q8_0)
	// mul_mat (ffn_gate_raw-n): [feed_forward_length × 1], output: f32
	// inputs: blk.n.ffn_up.weight (q8_0), ffn_norm-n (q8_0)
	// mul_mat (ffn_up-n): [feed_forward_length × 1], output: f32
	barrier_9_ffn_parallel_projections,

	// inputs: ffn_gate_raw-n (f32), ffn_up-n (f32), blk.n.ffn_down.weight (q8_0), ffn_inp-n (f32)
	// silu_mul_mat_add (l_out-n): silu(gate) × up → down_proj → add residual, output: f32
	barrier_10_ffn_output,
	count
};

template<model_config config> struct core_traits_new<config, op_types::inp_embd> : public config_holder<config> {
	using model_traits_type		= model_traits<config.arch, config.model_size, config.model_generation>;
	using input_01_type			= core_traits<config, op_types::token_embd_weight>;
	using input_02_type			= core_traits<config, op_types::inp_tokens>;
	using output_type			= typename kernel_type_profile_traits<config.kernel_profile>::compute_type;
	using core_traits_dims_type = core_trait_dims<model_traits_type::embedding_length, config.default_max_sequence_length, 1, 1, 1>;
	static constexpr uint64_t global_input_count{ 0 };
	static constexpr uint64_t total_required_bytes{ round_up_to_multiple<cpu_alignment_holder::cpu_alignment>(
		type_traits<output_type>::total_byte_size(core_traits_dims_type::get_array())) };
	static constexpr kernel_types kernel_type{ kernel_types::get_rows };
	static constexpr op_types op_type{ op_types::inp_embd };
};

template<typename value_type, typename... value_types> constexpr uint64_t get_total_required_bytes(uint64_t total_size = 0) {
	if constexpr (sizeof...(value_types) > 0) {
		total_size += value_type::total_required_bytes;
		return get_total_required_bytes<value_types...>(total_size);

	} else {
		return total_size + value_type::total_required_bytes;
	}
}

template<model_config config, compute_soverignties sov> struct compute_soveriegnty {};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_0_initial_embedding> {
	using output_type			= typename kernel_type_profile_traits<config.kernel_profile>::compute_type;
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::global_input };
	static constexpr input_types input_type{ input_types::one };
	array<linked_latch<false>, model_traits_type<config>::block_count> latch{};
	output_type* data{};
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_1_layer_norm> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::one };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_2_qkv_projections> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::three };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_3_reshape_rope> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::two };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_4_attention_scores> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::one };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_5_attention_softmax> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::one };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_6_attention_output> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::one };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_7_output_projection> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::one };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_8_ffn_norm> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::two };
};

template<model_config config> struct compute_soveriegnty<config, compute_soverignties::barrier_9_ffn_parallel_projections> {
	static constexpr thread_strategy_types thread_strategy_type{ thread_strategy_types::per_block };
	static constexpr input_types input_type{ input_types::one };
};

// For prompt processing (seq_len = 8 for example):
// - Attention scores: [32 × 8 × 32 × 8] = 8,192 elements
// For generation (seq_len = 1):
// - Attention scores: [32 × 1 × 32 × 1] = 1,024 elements
// Maximum context: seq_len ≤ context_length = 131,072

int32_t main(int32_t argc, char** argv) {
	static constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama);
	static constexpr uint64_t total_bytes01 = core_traits_new<model_config, op_types::inp_embd>::total_required_bytes;
	printf("FIRST: %d\n", total_bytes01);
	static constexpr uint64_t total_bytes = get_total_required_bytes<core_traits_new<model_config, op_types::inp_embd>, core_traits_new<model_config, op_types::inp_embd>>();
	printf("FIRST: %d\n", total_bytes);
	nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new->process_input(cli_args.prompt)) {
	}
	return 0;
}
