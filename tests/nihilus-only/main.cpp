#include <nihilus>

using namespace nihilus;

template<model_config config, typename output_type, data_strategy_types data_strategy_type> struct data_mixin {};

template<model_config config, typename output_type> struct data_mixin<config, output_type, data_strategy_types::global> {
	output_type* data{};
};

template<model_config config, typename output_type> struct data_mixin<config, output_type, data_strategy_types::per_block> {
	array<output_type*, model_traits_type<config>::block_count> data{};
};

template<model_config config, composite_kernel_types kernel_type, data_strategy_types data_strategy_type, allocation_strategy_types allocation_strategy_type,
	typename composite_kernel_type_new, typename... input_composite_kernel_types_new>
struct op_traits_new : public data_mixin<config, typename composite_kernel_type_new::output_type, data_strategy_type>, public composite_kernel_type_new {
	using output_type = composite_kernel_type_new::output_type;
	using dims_type	  = composite_kernel_type_new::dims_type;
	static constexpr uint64_t total_required_bytes{ type_traits<output_type>::total_byte_size(dims_type::get_array()) };
};

template<model_config config> struct core_traits<config, core_types::weights> {
	static constexpr core_types core_type{ core_types::weights };

	using attn_q_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::embedding_length, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using attn_k_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using attn_v_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using attn_output_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::embedding_length, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using attn_norm_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

	using ffn_gate_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::feed_forward_length, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using ffn_up_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::feed_forward_length, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using ffn_down_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::feed_forward_length, model_traits_type<config>::embedding_length, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using ffn_norm_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

	using token_embd_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::vocab_size, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using rope_freqs_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::rope_dimension_count / 2, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

	using output_norm_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::norm_type>;

	using output_weight_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, model_traits_type<config>::vocab_size, 1, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::weight_type>;

	using attn_q_weight_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_q_weight_kernel_traits>;

	using attn_k_weight_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_k_weight_kernel_traits>;

	using attn_v_weight_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_v_weight_kernel_traits>;

	using attn_output_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_output_weight_kernel_traits>;

	using attn_norm_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, attn_norm_weight_kernel_traits>;

	using ffn_gate_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_gate_weight_kernel_traits>;

	using ffn_up_weight_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_up_weight_kernel_traits>;

	using ffn_down_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_down_weight_kernel_traits>;

	using ffn_norm_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::mmap, ffn_norm_weight_kernel_traits>;

	using token_embd_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap, token_embd_weight_kernel_traits>;

	using rope_freqs_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap, rope_freqs_weight_kernel_traits>;

	using output_norm_weight_type =
		op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap, output_norm_weight_kernel_traits>;

	using output_weight_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::mmap, output_weight_kernel_traits>;

	attn_q_weight_type attn_q_weight{};
	attn_k_weight_type attn_k_weight{};
	attn_v_weight_type attn_v_weight{};
	attn_output_weight_type attn_output_weight{};
	attn_norm_weight_type attn_norm_weight{};
	ffn_gate_weight_type ffn_gate_weight{};
	ffn_up_weight_type ffn_up_weight{};
	ffn_down_weight_type ffn_down_weight{};
	ffn_norm_weight_type ffn_norm_weight{};
	token_embd_weight_type token_embd_weight{};
	rope_freqs_weight_type rope_freqs_weight{};
	output_norm_weight_type output_norm_weight{};
	output_weight_type output_weight{};
};

template<model_config config, weight_types weight_type, typename op_traits_type>
void pack_weight_pointers_impl(op_traits_type& op, array<array<void*, model_traits_type<config>::block_count>, weight_types::count>& data) {
	if constexpr (array_types<decltype(op.data)>) {
		for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
			data[weight_type][x] == reinterpret_cast<void*>(&op.data[x]);
		}
	} else {
		for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
			data[weight_type][x] == reinterpret_cast<void*>(&op.data);
		}
	}
}

template<model_config config, typename core_traits_type>
NIHILUS_INLINE void pack_weight_pointers(core_traits_type& core_traits, array<array<void*, model_traits_type<config>::block_count>, weight_types::count>& data) {
	pack_weight_pointers_impl<config, weight_types::attn_q>(core_traits.attn_q_weight, data);
	pack_weight_pointers_impl<config, weight_types::attn_k>(core_traits.attn_k_weight, data);
	pack_weight_pointers_impl<config, weight_types::attn_v>(core_traits.attn_v_weight, data);
	pack_weight_pointers_impl<config, weight_types::attn_output>(core_traits.attn_output_weight, data);
	pack_weight_pointers_impl<config, weight_types::attn_norm>(core_traits.attn_norm_weight, data);
	pack_weight_pointers_impl<config, weight_types::ffn_gate>(core_traits.ffn_gate_weight, data);
	pack_weight_pointers_impl<config, weight_types::ffn_up>(core_traits.ffn_up_weight, data);
	pack_weight_pointers_impl<config, weight_types::ffn_down>(core_traits.ffn_down_weight, data);
	pack_weight_pointers_impl<config, weight_types::ffn_norm>(core_traits.ffn_norm_weight, data);
	pack_weight_pointers_impl<config, weight_types::token_embd>(core_traits.token_embd_weight, data);
	pack_weight_pointers_impl<config, weight_types::rope_freqs>(core_traits.rope_freqs_weight, data);
	pack_weight_pointers_impl<config, weight_types::output_norm>(core_traits.output_norm_weight, data);
	pack_weight_pointers_impl<config, weight_types::output>(core_traits.output_weight, data);
};

template<model_config config> struct core_traits<config, core_types::global_inputs> {
	static constexpr core_types core_type{ core_types::global_inputs };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };

	using inp_tokens_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::input_token_type>;

	using inp_pos_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::position_type>;

	using cache_k_kernel_traits = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::block_count, model_traits_type<config>::attention_head_count_kv,
			config.default_max_sequence_length, 1>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type>;

	using cache_v_kernel_traits = kernel_traits<config,
		core_trait_dims<config.default_max_sequence_length, model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count_kv,
			model_traits_type<config>::block_count, 0>,
		kernel_types::none, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type>;

	using kq_mask_kernel_traits = kernel_traits<config, core_trait_dims<model_traits_type<config>::block_count, model_traits_type<config>::block_count, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::mask_type>;

	using inp_out_ids_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::none,
		typename kernel_type_profile_traits<config.kernel_profile>::output_token_type>;

	using inp_tokens_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc, inp_tokens_kernel_traits>;

	using inp_pos_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc, inp_pos_kernel_traits>;

	using cache_k_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::alloc, cache_k_kernel_traits>;

	using cache_v_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::per_block, allocation_strategy_types::alloc, cache_v_kernel_traits>;

	using kq_mask_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc, kq_mask_kernel_traits>;

	using inp_out_ids_type = op_traits_new<config, composite_kernel_types::none, data_strategy_types::global, allocation_strategy_types::alloc, inp_out_ids_kernel_traits>;

	inp_tokens_type inp_tokens{};
	inp_pos_type inp_pos{};
	cache_k_type cache_k{};
	cache_v_type cache_v{};
	kq_mask_type kq_mask{};
	inp_out_ids_type inp_out_ids{};

	static constexpr uint64_t total_required_bytes{ inp_tokens_type::total_required_bytes + inp_pos_type::total_required_bytes +
		(model_traits_type<config>::block_count * cache_k_type::total_required_bytes) + (model_traits_type<config>::block_count * cache_v_type::total_required_bytes) +
		kq_mask_type::total_required_bytes + inp_out_ids_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::global_outputs> {
	static constexpr core_types core_type{ core_types::global_outputs };
	static constexpr uint64_t depth{ std::numeric_limits<uint64_t>::max() };

	using rms_norm_kernel_traits = kernel_traits<config,
		get_new_dims_new_1_t<kernel_types::rms_norm, typename core_traits<config, core_types::token_embeddings>::input_embedding_kernel_traits>, kernel_types::rms_norm,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, typename core_traits<config, core_types::token_embeddings>::input_embedding_kernel_traits>;

	using mul_kernel_traits =
		kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, rms_norm_kernel_traits, typename core_traits<config, core_types::weights>::output_norm_weight_kernel_traits>,
			kernel_types::mul, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_traits,
			typename core_traits<config, core_types::weights>::output_norm_weight_kernel_traits>;

	using mul_mat_kernel_traits =
		kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, typename core_traits<config, core_types::weights>::output_weight_kernel_traits, mul_kernel_traits>,
			kernel_types::mul_mat, typename kernel_type_profile_traits<config.kernel_profile>::compute_type,
			typename core_traits<config, core_types::weights>::output_weight_kernel_traits, mul_kernel_traits>;

	using logit_sample_kernel_traits = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, 1, 1, 1>, kernel_types::logit_sample,
		typename kernel_type_profile_traits<config.kernel_profile>::output_token_type, mul_mat_kernel_traits>;

	using result_output_composite_kernel_traits =
		composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_logit_sample, typename kernel_type_profile_traits<config.kernel_profile>::output_token_type,
			rms_norm_kernel_traits, mul_kernel_traits, mul_mat_kernel_traits, logit_sample_kernel_traits>;

	using result_output_type = op_traits_new<config, composite_kernel_types::rms_norm_mul_mul_mat_logit_sample, data_strategy_types::global, allocation_strategy_types::alloc,
		result_output_composite_kernel_traits>;

	result_output_type result_output{};

	static constexpr uint64_t total_required_bytes{ result_output_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::token_embeddings> {
	static constexpr core_types core_type{ core_types::token_embeddings };
	static constexpr uint64_t depth{ 0 };

	using input_01_type = typename core_traits<config, core_types::weights>::token_embd_weight_type;
	using input_02_type = typename core_traits<config, core_types::global_inputs>::inp_tokens_type;

	using input_embedding_kernel_traits = kernel_traits<config, get_new_dims_new_2_t<kernel_types::get_rows, input_01_type, input_02_type>, kernel_types::get_rows,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

	using token_embeddings_type =
		op_traits_new<config, composite_kernel_types::get_rows, data_strategy_types::global, allocation_strategy_types::alloc, input_embedding_kernel_traits>;

	token_embeddings_type norm{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ token_embeddings_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::qkv_projection_layer> {
	static constexpr core_types core_type{ core_types::qkv_projection_layer };
	static constexpr uint64_t depth{ 1 };

	using input_01_type = typename core_traits<config, core_types::token_embeddings>::token_embeddings_type;
	using input_02_type = typename core_traits<config, core_types::weights>::attn_norm_weight_type;
	using input_03_type = typename core_traits<config, core_types::weights>::attn_q_weight_type;
	using input_04_type = typename core_traits<config, core_types::weights>::attn_k_weight_type;
	using input_05_type = typename core_traits<config, core_types::weights>::attn_v_weight_type;
	using input_06_type = typename core_traits<config, core_types::global_inputs>::cache_k_type;
	using input_07_type = typename core_traits<config, core_types::global_inputs>::cache_v_type;

	using rms_norm_kernel_trait = kernel_traits<config, get_new_dims_new_1_t<kernel_types::rms_norm, input_01_type>, kernel_types::rms_norm,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type>;

	using mul_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, rms_norm_kernel_trait, input_02_type>, kernel_types::mul,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, input_02_type>;

	using q_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_03_type, mul_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_03_type, mul_kernel_trait>;

	using q_reshape_kernel_trait = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count, config.default_max_sequence_length, 1, 2>,
		kernel_types::reshape, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, q_mul_mat_kernel_trait>;

	using k_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_04_type, mul_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_04_type, mul_kernel_trait>;

	using k_reshape_kernel_trait = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count_kv, config.default_max_sequence_length, 1, 2>,
		kernel_types::reshape, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, k_mul_mat_kernel_trait>;

	using v_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_05_type, mul_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_05_type, mul_kernel_trait>;

	using v_transpose_kernel_trait = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1, 0>,
		kernel_types::transpose, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, v_mul_mat_kernel_trait>;

	using v_cache_view_kernel_trait = kernel_traits<config, core_trait_dims<config.default_max_sequence_length, model_traits_type<config>::n_embd_kv_gqa, 1, 1, 0>,
		kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_07_type>;

	using v_cache_copy_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::copy, v_transpose_kernel_trait, v_cache_view_kernel_trait>, kernel_types::copy,
		typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, v_transpose_kernel_trait, v_cache_view_kernel_trait>;

	using q_cur_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_reshape,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, mul_kernel_trait, q_mul_mat_kernel_trait, q_reshape_kernel_trait>;

	using k_cur_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_reshape,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, mul_kernel_trait, k_mul_mat_kernel_trait, k_reshape_kernel_trait>;

	using v_cur_composite_kernel_traits =
		composite_kernel_traits<config, composite_kernel_types::rms_norm_mul_mul_mat_transpose_copy, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type,
			rms_norm_kernel_trait, mul_kernel_trait, v_mul_mat_kernel_trait, v_transpose_kernel_trait, v_cache_copy_kernel_trait>;

	using q_cur_type = op_traits_new<config, composite_kernel_types::rms_norm_mul_mul_mat_reshape, data_strategy_types::per_block, allocation_strategy_types::alloc,
		q_cur_composite_kernel_traits>;

	using k_cur_type = op_traits_new<config, composite_kernel_types::rms_norm_mul_mul_mat_reshape, data_strategy_types::per_block, allocation_strategy_types::alloc,
		k_cur_composite_kernel_traits>;

	using v_cur_type = op_traits_new<config, composite_kernel_types::rms_norm_mul_mul_mat_transpose_copy, data_strategy_types::per_block, allocation_strategy_types::alloc,
		v_cur_composite_kernel_traits>;

	q_cur_type q_cur{};
	k_cur_type k_cur{};
	v_cur_type v_cur{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ q_cur_type::total_required_bytes + k_cur_type::total_required_bytes + v_cur_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::rope_and_cache_operations> {
	static constexpr core_types core_type{ core_types::rope_and_cache_operations };
	static constexpr uint64_t depth{ 2 };

	using input_01_type = typename core_traits<config, core_types::qkv_projection_layer>::q_cur_type;
	using input_02_type = typename core_traits<config, core_types::qkv_projection_layer>::k_cur_type;
	using input_03_type = typename core_traits<config, core_types::qkv_projection_layer>::v_cur_type;
	using input_04_type = typename core_traits<config, core_types::global_inputs>::inp_pos_type;
	using input_05_type = typename core_traits<config, core_types::weights>::rope_freqs_weight_type;
	using input_06_type = typename core_traits<config, core_types::global_inputs>::cache_k_type;
	using input_07_type = typename core_traits<config, core_types::global_inputs>::cache_v_type;

	using rope_q_kernel_trait = kernel_traits<config, get_new_dims_new_3_t<kernel_types::rope, input_01_type, input_04_type, input_05_type>, kernel_types::rope,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_04_type, input_05_type>;

	using rope_k_kernel_trait = kernel_traits<config, get_new_dims_new_3_t<kernel_types::rope, input_02_type, input_04_type, input_05_type>, kernel_types::rope,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_02_type, input_04_type, input_05_type>;

	using k_cache_view_kernel_trait = kernel_traits<config, core_trait_dims<model_traits_type<config>::n_embd_kv_gqa, config.default_max_sequence_length, 1, 1, 1>,
		kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_06_type>;

	using k_cache_copy_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::copy, rope_k_kernel_trait, k_cache_view_kernel_trait>, kernel_types::copy,
		typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, rope_k_kernel_trait, k_cache_view_kernel_trait>;

	using k_rope_view_kernel_trait = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count, model_traits_type<config>::attention_head_count_kv, 1>,
		kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_06_type>;

	using v_rope_view_kernel_trait = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::attention_head_count, model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count_kv, 1>,
		kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, input_07_type>;

	using rope_q_permute_kernel_trait = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::rope_dimension_count, config.default_max_sequence_length, model_traits_type<config>::attention_head_count, 1>,
		kernel_types::permute, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rope_q_kernel_trait>;

	using rope_q_permute_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rope_permute,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rope_q_kernel_trait, rope_q_permute_kernel_trait>;

	using rope_k_copy_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::rope_copy,
		typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, rope_k_kernel_trait, k_cache_view_kernel_trait, k_cache_copy_kernel_trait>;

	using k_rope_view_composite_trait_type =
		composite_kernel_traits<config, composite_kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, k_rope_view_kernel_trait>;

	using v_rope_view_composite_trait_type =
		composite_kernel_traits<config, composite_kernel_types::view, typename kernel_type_profile_traits<config.kernel_profile>::kv_cache_type, v_rope_view_kernel_trait>;

	using rope_q_permute_type =
		op_traits_new<config, composite_kernel_types::rope_permute, data_strategy_types::per_block, allocation_strategy_types::alloc, rope_q_permute_composite_kernel_traits>;

	using rope_k_copy_type =
		op_traits_new<config, composite_kernel_types::rope_copy, data_strategy_types::per_block, allocation_strategy_types::alloc, rope_k_copy_composite_kernel_traits>;

	using k_rope_view_type =
		op_traits_new<config, composite_kernel_types::view, data_strategy_types::per_block, allocation_strategy_types::alloc, k_rope_view_composite_trait_type>;

	using v_rope_view_type =
		op_traits_new<config, composite_kernel_types::view, data_strategy_types::per_block, allocation_strategy_types::alloc, v_rope_view_composite_trait_type>;

	rope_q_permute_type rope_q_permute{};
	rope_k_copy_type rope_k_copy{};
	k_rope_view_type k_rope_view{};
	v_rope_view_type v_rope_view{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ rope_q_permute_type::total_required_bytes + rope_k_copy_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::attention_scores_computation> {
	static constexpr core_types core_type{ core_types::attention_scores_computation };
	static constexpr uint64_t depth{ 3 };

	using input_01_type = typename core_traits<config, core_types::rope_and_cache_operations>::rope_q_permute_type;
	using input_02_type = typename core_traits<config, core_types::rope_and_cache_operations>::k_rope_view_type;

	using kq_scores_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_02_type, input_01_type>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_02_type, input_01_type>;

	using kq_scores_composite_kernel_traits =
		composite_kernel_traits<config, composite_kernel_types::mul_mat, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, kq_scores_kernel_trait>;

	using kq_scores_type =
		op_traits_new<config, composite_kernel_types::mul_mat, data_strategy_types::per_block, allocation_strategy_types::alloc, kq_scores_composite_kernel_traits>;

	kq_scores_type kq_scores{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ kq_scores_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::attention_weighted_values> {
	static constexpr core_types core_type{ core_types::attention_weighted_values };
	static constexpr uint64_t depth{ 4 };

	using input_01_type = typename core_traits<config, core_types::attention_scores_computation>::kq_scores_type;
	using input_02_type = typename core_traits<config, core_types::global_inputs>::kq_mask_type;
	using input_03_type = typename core_traits<config, core_types::rope_and_cache_operations>::v_rope_view_type;

	using softmax_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::softmax, input_01_type, input_02_type>, kernel_types::softmax,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

	using attention_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_03_type, softmax_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_03_type, softmax_kernel_trait>;

	using attention_permute_kernel_trait = kernel_traits<config,
		core_trait_dims<model_traits_type<config>::rope_dimension_count, model_traits_type<config>::attention_head_count, config.default_max_sequence_length, 1, 2>,
		kernel_types::permute, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, attention_mul_mat_kernel_trait>;

	using attention_cont_kernel_trait = kernel_traits<config, core_trait_dims<model_traits_type<config>::embedding_length, config.default_max_sequence_length, 1, 1, 1>,
		kernel_types::cont, typename kernel_type_profile_traits<config.kernel_profile>::compute_type, attention_permute_kernel_trait>;

	using attention_output_composite_kernel_traits =
		composite_kernel_traits<config, composite_kernel_types::softmax_mul_mat_permute_cont, typename kernel_type_profile_traits<config.kernel_profile>::compute_type,
			softmax_kernel_trait, attention_mul_mat_kernel_trait, attention_permute_kernel_trait, attention_cont_kernel_trait>;

	using attention_output_type = op_traits_new<config, composite_kernel_types::softmax_mul_mat_permute_cont, data_strategy_types::per_block, allocation_strategy_types::alloc,
		attention_output_composite_kernel_traits>;

	attention_output_type attention_output{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ attention_output_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::attention_output_projection> {
	static constexpr core_types core_type{ core_types::attention_output_projection };
	static constexpr uint64_t depth{ 5 };

	using input_01_type = typename core_traits<config, core_types::attention_weighted_values>::attention_output_type;
	using input_02_type = typename core_traits<config, core_types::weights>::attn_output_weight_type;

	using attn_output_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_02_type, input_01_type>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_02_type, input_01_type>;

	using attn_output_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, attn_output_mul_mat_kernel_trait>;

	using attn_output_type =
		op_traits_new<config, composite_kernel_types::mul_mat, data_strategy_types::per_block, allocation_strategy_types::alloc, attn_output_composite_kernel_traits>;

	attn_output_type attn_output{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ attn_output_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::ffn_parallel_projections> {
	static constexpr core_types core_type{ core_types::ffn_parallel_projections };
	static constexpr uint64_t depth{ 6 };

	using input_01_type = typename core_traits<config, core_types::attention_output_projection>::attn_output_type;
	using input_02_type = typename core_traits<config, core_types::token_embeddings>::token_embeddings_type;
	using input_03_type = typename core_traits<config, core_types::weights>::ffn_norm_weight_type;
	using input_04_type = typename core_traits<config, core_types::weights>::ffn_gate_weight_type;
	using input_05_type = typename core_traits<config, core_types::weights>::ffn_up_weight_type;

	using add_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::add, input_01_type, input_02_type>, kernel_types::add,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

	using rms_norm_kernel_trait = kernel_traits<config, get_new_dims_new_1_t<kernel_types::rms_norm, add_kernel_trait>, kernel_types::rms_norm,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, add_kernel_trait>;

	using mul_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, rms_norm_kernel_trait, input_03_type>, kernel_types::mul,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, rms_norm_kernel_trait, input_03_type>;

	using gate_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_04_type, mul_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_04_type, mul_kernel_trait>;

	using gate_silu_kernel_trait = kernel_traits<config, get_new_dims_new_1_t<kernel_types::silu, gate_mul_mat_kernel_trait>, kernel_types::silu,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, gate_mul_mat_kernel_trait>;

	using up_mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_05_type, mul_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_05_type, mul_kernel_trait>;

	using ffn_gate_composite_kernel_traits =
		composite_kernel_traits<config, composite_kernel_types::add_rms_norm_mul_mat_silu, typename kernel_type_profile_traits<config.kernel_profile>::compute_type,
			add_kernel_trait, rms_norm_kernel_trait, mul_kernel_trait, gate_mul_mat_kernel_trait, gate_silu_kernel_trait>;

	using ffn_up_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::add_rms_norm_mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, add_kernel_trait, rms_norm_kernel_trait, mul_kernel_trait, up_mul_mat_kernel_trait>;

	using ffn_gate_type = op_traits_new<config, composite_kernel_types::add_rms_norm_mul_mat_silu, data_strategy_types::per_block, allocation_strategy_types::alloc,
		ffn_gate_composite_kernel_traits>;

	using ffn_up_type =
		op_traits_new<config, composite_kernel_types::add_rms_norm_mul_mat, data_strategy_types::per_block, allocation_strategy_types::alloc, ffn_up_composite_kernel_traits>;

	ffn_gate_type ffn_gate{};
	ffn_up_type ffn_up{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ ffn_gate_type::total_required_bytes + ffn_up_type::total_required_bytes };
};

template<model_config config> struct core_traits<config, core_types::ffn_down_projection> {
	static constexpr core_types core_type{ core_types::ffn_down_projection };
	static constexpr uint64_t depth{ 7 };

	using input_01_type = typename core_traits<config, core_types::ffn_parallel_projections>::ffn_gate_type;
	using input_02_type = typename core_traits<config, core_types::ffn_parallel_projections>::ffn_up_type;
	using input_03_type = typename core_traits<config, core_types::weights>::ffn_down_weight_type;
	using input_04_type = typename core_traits<config, core_types::attention_output_projection>::attn_output_type;

	using mul_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul, input_01_type, input_02_type>, kernel_types::mul,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_01_type, input_02_type>;

	using mul_mat_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::mul_mat, input_03_type, mul_kernel_trait>, kernel_types::mul_mat,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, input_03_type, mul_kernel_trait>;

	using add_kernel_trait = kernel_traits<config, get_new_dims_new_2_t<kernel_types::add, mul_mat_kernel_trait, input_04_type>, kernel_types::add,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, mul_mat_kernel_trait, input_04_type>;

	using ffn_down_composite_kernel_traits = composite_kernel_traits<config, composite_kernel_types::mul_mul_mat_add,
		typename kernel_type_profile_traits<config.kernel_profile>::compute_type, mul_kernel_trait, mul_mat_kernel_trait, add_kernel_trait>;

	using ffn_down_type =
		op_traits_new<config, composite_kernel_types::mul_mul_mat_add, data_strategy_types::per_block, allocation_strategy_types::alloc, ffn_down_composite_kernel_traits>;

	ffn_down_type ffn_down{};
	op_latch latch{};

	static constexpr uint64_t total_required_bytes{ ffn_down_type::total_required_bytes };
};

template<model_config config, typename... bases> struct core_bases_new : public bases... {
	NIHILUS_INLINE core_bases_new()					 = default;
	core_bases_new& operator=(core_bases_new&&)		 = delete;
	core_bases_new(core_bases_new&&)				 = delete;
	core_bases_new& operator=(const core_bases_new&) = delete;
	core_bases_new(const core_bases_new&)			 = delete;
	template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) const {
		(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
	}

	template<template<model_config, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) {
		(impl_internal_filtered<mixin_type, bases>(args...), ...);
	}

	template<template<model_config, typename, processing_phase> typename mixin_type, processing_phase phase, typename... arg_types>
	NIHILUS_INLINE constexpr void impl_thread(arg_types&&... args) {
		(impl_internal_filtered_thread<mixin_type, phase, bases>(args...), ...);
	}

  protected:
	template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
	NIHILUS_INLINE constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) const {
		if constexpr (mixin_type<config, base_type>::filter()) {
			mixin_type<config, base_type>::impl(*static_cast<const base_type*>(this), detail::forward<arg_types>(args)...);
		}
	}

	template<template<model_config, typename> typename mixin_type, typename base_type, typename... arg_types>
	NIHILUS_INLINE constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) {
		if constexpr (mixin_type<config, base_type>::filter()) {
			mixin_type<config, base_type>::impl(*static_cast<base_type*>(this), detail::forward<arg_types>(args)...);
		}
	}

	template<template<model_config, typename, processing_phase> typename mixin_type, processing_phase phase, typename base_type, typename... arg_types>
	NIHILUS_INLINE constexpr void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) {
		if constexpr (mixin_type<config, base_type, phase>::filter()) {
			mixin_type<config, base_type, phase>::impl(*static_cast<base_type*>(this), detail::forward<arg_types>(args)...);
		}
	}
};

template<model_config config, typename index_sequence> struct get_core_bases_new;

template<model_config config, size_t... index> struct get_core_bases_new<config, std::index_sequence<index...>> {
	using type = core_bases_new<config, core_traits<config, static_cast<core_types>(index)>...>;
};

template<model_config config> using get_core_bases_new_t = typename get_core_bases_new<config, std::make_index_sequence<static_cast<uint64_t>(core_types::count)>>::type;

struct memory_plan_new {
	uint64_t currently_allocated_bytes{};
	array<uint64_t, 8> allocations{};
	array<uint64_t, 8> offsets{};
};

template<model_config config, typename base_type_new> struct total_bytes_collector {
	NIHILUS_INLINE total_bytes_collector() noexcept										   = default;
	NIHILUS_INLINE total_bytes_collector& operator=(const total_bytes_collector&) noexcept = delete;
	NIHILUS_INLINE total_bytes_collector(const total_bytes_collector&) noexcept			   = delete;
	NIHILUS_INLINE total_bytes_collector& operator=(total_bytes_collector&&) noexcept	   = delete;
	NIHILUS_INLINE total_bytes_collector(total_bytes_collector&&) noexcept				   = delete;
	using base_type																		   = base_type_new;
	NIHILUS_INLINE static constexpr bool filter() {
		return base_type::core_type != core_types::weights;
	}
	NIHILUS_INLINE static constexpr void impl(const base_type&, memory_plan_new& mem_plan) {
		if constexpr (base_type::depth == std::numeric_limits<uint64_t>::max()) {
			mem_plan.offsets[0]		= mem_plan.currently_allocated_bytes;
			mem_plan.allocations[0] = base_type::total_required_bytes;
		} else {
			mem_plan.offsets[base_type::depth]	   = mem_plan.currently_allocated_bytes;
			mem_plan.allocations[base_type::depth] = base_type::total_required_bytes;
		}

		// ADD current memory FIRST
		mem_plan.currently_allocated_bytes += base_type::total_required_bytes;

		// THEN free old memory (keep sliding window of 2 depths)
		if constexpr (base_type::depth != std::numeric_limits<uint64_t>::max() && base_type::depth > 1) {
			mem_plan.currently_allocated_bytes -= mem_plan.allocations[base_type::depth - 2];
			mem_plan.allocations[base_type::depth - 2] = 0;
		}
	}
};

template<model_config config, typename base_type_new> struct memory_mapper_new {
	NIHILUS_INLINE memory_mapper_new() noexcept								   = default;
	NIHILUS_INLINE memory_mapper_new& operator=(const memory_mapper_new&) noexcept = delete;
	NIHILUS_INLINE memory_mapper_new(const memory_mapper_new&) noexcept			   = delete;
	NIHILUS_INLINE memory_mapper_new& operator=(memory_mapper_new&&) noexcept	   = delete;
	NIHILUS_INLINE memory_mapper_new(memory_mapper_new&&) noexcept				   = delete;
	using base_type														   = base_type_new;
	NIHILUS_INLINE static constexpr bool filter() {
		return base_type::data_strategy_type != data_strategy_types::global;
	}
	NIHILUS_INLINE static void impl(base_type& parse_core, const memory_plan& plan, memory_buffer<config>& memory_buffer) {
		using output_type = typename base_type::output_type;
		if constexpr (base_type::data_strategy_type == data_strategy_types::global) {
			using input_01_type	  = typename base_type::input_01_type;
			using other_data_type = decltype(input_01_type::data);
			if constexpr (array_types<decltype(parse_core.data)> && array_types<other_data_type>) {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					parse_core.data[x] = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data[x];
				}
			} else if constexpr (array_types<decltype(parse_core.data)> && !array_types<other_data_type>) {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					parse_core.data[x] = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data;
				}
			} else if constexpr (!array_types<decltype(parse_core.data)> && array_types<other_data_type>) {
				parse_core.data = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data[0];
			} else if constexpr (!array_types<decltype(parse_core.data)> && !array_types<other_data_type>) {
				parse_core.data = static_cast<input_01_type*>(static_cast<thread_pool<config>*>(&parse_core))->data;
			} else {
				std::cout << "Sorry, but failed to map op of type: " << base_type::op_type << ", " << std::endl;
			}
		} else {
			auto* new_ptr = memory_buffer.claim_memory(static_cast<uint64_t>(plan.offsets[base_type::op_type].offset));
			if constexpr (array_types<decltype(parse_core.data)>) {
				for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
					parse_core.data[x] = static_cast<output_type*>(new_ptr);
				}
			} else {
				parse_core.data = static_cast<output_type*>(new_ptr);
			}
		}
	}
};

template<model_config config, typename base_type_new> struct weight_mapper_new {
	NIHILUS_INLINE weight_mapper_new() noexcept								   = default;
	NIHILUS_INLINE weight_mapper_new& operator=(const weight_mapper_new&) noexcept = delete;
	NIHILUS_INLINE weight_mapper_new(const weight_mapper_new&) noexcept			   = delete;
	NIHILUS_INLINE weight_mapper_new& operator=(weight_mapper_new&&) noexcept	   = delete;
	NIHILUS_INLINE weight_mapper_new(weight_mapper_new&&) noexcept				   = delete;
	using base_type														   = base_type_new;
	NIHILUS_INLINE static constexpr bool filter() {
		return static_cast<uint64_t>(base_type::op_type) <= 12;
	}
	NIHILUS_INLINE static void impl(base_type& parse_core, array<array<void*, model_traits_type<config>::block_count>, op_types::count>& data) {
		if constexpr (array_types<decltype(parse_core.data)>) {
			for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
				data[base_type::op_type][x] = reinterpret_cast<void*>(&parse_core.data[x]);
			}
		} else {
			for (uint64_t x = 0; x < model_traits_type<config>::block_count; ++x) {
				data[base_type::op_type][x] = reinterpret_cast<void*>(&parse_core.data);
			}
		}
	}
};

int32_t main(int32_t argc, char** argv) {
	static constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama);
	get_core_bases_new_t<model_config> core_bases{};
	array<array<void*, model_traits_type<model_config>::block_count>, weight_types::count> data{};
	pack_weight_pointers<model_config>(*static_cast<core_traits<model_config, core_types::weights>*>(&core_bases), data);
	//core_bases.template impl<weight_mapper>(core_bases, data);
	memory_plan_new total_required_bytes{ [&]() {
		memory_plan_new return_values{};
		core_bases.template impl<total_bytes_collector>(return_values);
		return return_values;
	}() };
	printf("TOTAL REQUIRED BYTES: %d\n", total_required_bytes.currently_allocated_bytes);
	//core_bases.v_cur.data[0];
	
	nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new->process_input(cli_args.prompt)) {
	}
	return 0;
}
