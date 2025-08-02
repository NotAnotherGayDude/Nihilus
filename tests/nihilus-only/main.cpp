#include <nihilus>
using namespace nihilus;

struct core_base {
	kernel_types kernel_type{};
	array<int64_t, 4> dims{};
	data_types data_type{};
	op_types op_type{};
	uint64_t block{};
};

template<uint64_t input_count> struct core : public core_base {
	array<core_base*, input_count> inputs{};
};

template<uint64_t current_zero_input_size_new = 0, uint64_t current_one_input_size_new = 0, uint64_t current_two_input_size_new = 0, uint64_t current_three_input_size_new = 0>
struct core_base_graph {
	constexpr core_base_graph() = default;
	static constexpr uint64_t current_zero_input_size{ current_zero_input_size_new };
	static constexpr uint64_t current_one_input_size{ current_one_input_size_new };
	static constexpr uint64_t current_two_input_size{ current_two_input_size_new };
	static constexpr uint64_t current_three_input_size{ current_three_input_size_new };

	template<uint64_t input_count> constexpr auto operator+(const core<input_count>& value_new) const {
		if constexpr (input_count == 0) {
			core_base_graph<current_zero_input_size + 1> return_value{};
			//return_value.zero_input_values[current_zero_input_size] = value_new;
			return return_value;
		} else if constexpr (input_count == 1) {
			core_base_graph<current_zero_input_size, current_one_input_size + 1> return_value{};
			//return_value.one_input_values[current_zero_input_size] = value_new;
			return return_value;
		} else if constexpr (input_count == 2) {
			core_base_graph<current_zero_input_size, current_one_input_size, current_two_input_size + 1> return_value{};
			//return_value.two_input_values[current_zero_input_size] = value_new;
			return return_value;
		} else if constexpr (input_count == 3) {
			core_base_graph<current_zero_input_size, current_one_input_size, current_two_input_size, current_three_input_size + 1> return_value{};
			//return_value.three_input_values[current_zero_input_size] = value_new;
			return return_value;
		} else {
			return nullptr;
		}
	}
	template<uint64_t input_count = 0>
	constexpr core_base* get_core_base(op_types op_type, uint64_t block = 0) {
		if constexpr (input_count == 0) {
			for (uint64_t x = 0; x < zero_input_values.size();++x) {
				if (zero_input_values[x].op_type == op_type && zero_input_values[x].block == block) {
					return &zero_input_values[x];
				}
			}
			return nullptr;
		} else if constexpr (input_count == 1) {
			for (uint64_t x = 0; x < one_input_values.size(); ++x) {
				if (one_input_values[x].op_type == op_type && one_input_values[x].block == block) {
					return &one_input_values[x];
				}
			}
			return nullptr;
		} else if constexpr (input_count == 2) {
			for (uint64_t x = 0; x < two_input_values.size(); ++x) {
				if (two_input_values[x].op_type == op_type && two_input_values[x].block == block) {
					return &two_input_values[x];
				}
			}
			return nullptr;
		} else if constexpr (input_count == 3) {
			for (uint64_t x = 0; x < three_input_values.size(); ++x) {
				if (three_input_values[x].op_type == op_type && three_input_values[x].block == block) {
					return &three_input_values[x];
				}
			}
			return nullptr;
		} else {
			return nullptr;
		}
	}
	array<core<0>, current_zero_input_size> zero_input_values{};
	array<core<1>, current_one_input_size> one_input_values{};
	array<core<2>, current_two_input_size> two_input_values{};
	array<core<3>, current_three_input_size> three_input_values{};
};

template<uint64_t input_count = 0> static constexpr core<input_count> new_tensor_impl(op_types op_type, data_types data_type, kernel_types kernel_type, uint64_t current_block,
	const std::initializer_list<int64_t>& dims) {
	core<input_count> result{};
	result.data_type   = data_type;
	result.kernel_type = kernel_type;
	result.block	   = current_block;
	result.op_type	   = op_type;

	return result;
}

static constexpr core<2> get_rows(core_base* a, core_base* b, uint64_t current_block) {
	data_types type = data_types::f32;
	if (a->data_type == data_types::f32) {
		type = a->data_type;
	}
	core<2> result	 = new_tensor_impl<2>(op_types::inp_embd, type, kernel_types::get_rows, current_block, { a->dims[0], b->dims[0], b->dims[1], b->dims[2] });
	result.inputs[0] = a;
	result.inputs[1] = b;

	return result;
}

template<model_config config>
	requires(config.arch == model_arches::llama && config.model_size == model_sizes::llm_8B && config.model_generation == model_generations::v3)
static constexpr auto return_block(const auto& core_graph, uint64_t x) {
	using model_traits_type = model_traits<model_arches::llama, model_sizes::llm_8B, model_generations::v3>;
	auto kq_mask = new_tensor_impl<0>(op_types::kq_mask, data_types::f32, kernel_types::none, x, { model_traits_type::block_count, model_traits_type::block_count, 1, 1 });
	auto graph5	 = core_graph + kq_mask;

	auto blk0_attn_q_weight =
		new_tensor_impl<0>(op_types::attn_q_weight, data_types::q8_0, kernel_types::none, x, { model_traits_type::embedding_length, model_traits_type::embedding_length, 1, 1 });
	auto graph6 = graph5 + blk0_attn_q_weight;

	auto blk0_attn_k_weight = new_tensor_impl<0>(op_types::attn_k_weight, data_types::q8_0, kernel_types::none, x,
		{ model_traits_type::embedding_length, model_traits_type::rope_dimension_count * model_traits_type::attention_head_count_kv, 1, 1 });
	auto graph7				= graph6 + blk0_attn_k_weight;

	auto blk0_attn_v_weight = new_tensor_impl<0>(op_types::attn_v_weight, data_types::q8_0, kernel_types::none, x,
		{ model_traits_type::embedding_length, model_traits_type::rope_dimension_count * model_traits_type::attention_head_count_kv, 1, 1 });
	auto graph8				= graph7 + blk0_attn_v_weight;

	auto blk0_attn_output_weight = new_tensor_impl<0>(op_types::attn_output_weight, data_types::q8_0, kernel_types::none, x,
		{ model_traits_type::embedding_length, model_traits_type::embedding_length, 1, 1 });
	auto graph9					 = graph8 + blk0_attn_output_weight;

	auto blk0_attn_norm_weight = new_tensor_impl<0>(op_types::attn_norm_weight, data_types::f32, kernel_types::none, x, { model_traits_type::embedding_length, 1, 1, 1 });
	auto graph10			   = graph9 + blk0_attn_norm_weight;

	auto blk0_ffn_gate_weight = new_tensor_impl<0>(op_types::ffn_gate_weight, data_types::q8_0, kernel_types::none, x,
		{ model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1 });
	auto graph11			  = graph10 + blk0_ffn_gate_weight;

	auto blk0_ffn_up_weight =
		new_tensor_impl<0>(op_types::ffn_up_weight, data_types::q8_0, kernel_types::none, x, { model_traits_type::embedding_length, model_traits_type::feed_forward_length, 1, 1 });
	auto graph12 = graph11 + blk0_ffn_up_weight;

	auto blk0_ffn_down_weight = new_tensor_impl<0>(op_types::ffn_down_weight, data_types::q8_0, kernel_types::none, x,
		{ model_traits_type::feed_forward_length, model_traits_type::embedding_length, 1, 1 });
	auto graph13			  = graph12 + blk0_ffn_down_weight;

	auto blk0_ffn_norm_weight = new_tensor_impl<0>(op_types::ffn_norm_weight, data_types::f32, kernel_types::none, x, { model_traits_type::embedding_length, 1, 1, 1 });
	auto graph14			  = graph13 + blk0_ffn_norm_weight;

	auto cache_k_l0 =
		new_tensor_impl<0>(op_types::cache_k, data_types::f16, kernel_types::none, x, { model_traits_type::n_embd_kv_gqa * model_traits_type::embedding_length, 1, 1, 1 });
	auto graph15 = graph14 + cache_k_l0;

	auto cache_v_l0 =
		new_tensor_impl<0>(op_types::cache_v, data_types::f16, kernel_types::none, x, { model_traits_type::n_embd_kv_gqa * model_traits_type::embedding_length, 1, 1, 1 });
	return graph15 + cache_v_l0;
}

template<model_config config>
	requires(config.arch == model_arches::llama && config.model_size == model_sizes::llm_8B && config.model_generation == model_generations::v3)
static constexpr auto return_weights(const auto& core_graph) {
	using model_traits_type = model_traits<model_arches::llama, model_sizes::llm_8B, model_generations::v3>;
	auto token_embd_weight	= new_tensor_impl<0>(op_types::token_embd_weight, data_types::q8_0, kernel_types::none, 0,
		 { model_traits_type::embedding_length, model_traits_type::vocab_size, 1, 1 });
	auto graph1				= core_graph + token_embd_weight;

	auto rope_freqs_weight =
		new_tensor_impl<0>(op_types::rope_freqs_weight, data_types::f32, kernel_types::none, 0, { model_traits_type::rope_dimension_count / 2, 1, 1, 1 });
	auto graph2 = graph1 + rope_freqs_weight;

	auto inp_tokens = new_tensor_impl<0>(op_types::inp_tokens, data_types::i32, kernel_types::none, 0, { 1, 1, 1, 1 });
	auto graph3		= graph2 + inp_tokens;

	auto inp_pos = new_tensor_impl<0>(op_types::inp_pos, data_types::i32, kernel_types::none, 0, { 1, 1, 1, 1 });
	auto graph4	 = graph3 + inp_pos;
	auto graph_new_01 = return_block<config>(graph4, 0);
	return graph_new_01;
	
}

template<model_config config, uint64_t current_block = 0> static constexpr auto return_graph() {
	auto base_graph{ return_weights<config>(core_base_graph<0, 0, 0, 0>{}) };
	auto inp_tokens			 = new_tensor_impl<0>(op_types::inp_tokens, data_types::i32, kernel_types::none, current_block, { 1024, 1, 1, 1 });
	auto base_graph_inp_embd = get_rows(base_graph.get_core_base<0>(op_types::token_embd_weight), &inp_tokens, current_block);
	return base_graph;
}


int32_t main(int32_t argc, char** argv) {
	static constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama);
	static constexpr auto values = return_graph<model_config>();
	std::cout << "SIZE: " << values.current_zero_input_size << std::endl;
//	std::cout << "SIZE: " << ( int32_t )values.two_input_values[0].data_type << std::endl;
	nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	//auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	//while (model_new->process_input(cli_args.prompt)) {
	//}
	return 0;
}
