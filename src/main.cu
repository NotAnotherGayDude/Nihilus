#include <nihilus>

static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::default_max_sequence_length_types{ 8192 },
	nihilus::benchmark_types::enabled, nihilus::model_sizes::llm_405B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, nihilus::device_types::gpu);
static constexpr auto model_config_01 = nihilus::generate_model_config(nihilus::model_sizes::llm_405B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama,
	nihilus::device_types::gpu, nihilus::exception_types::disabled, nihilus::default_max_sequence_length_types{ 8192 });
static constexpr auto model_config_02 = nihilus::generate_model_config();

int32_t main(int64_t argc, char** argv) {
	const nihilus::cli_params cli_args_00 = nihilus::harbinger<model_config_00>::parse_cli_arguments(argc, argv);
	const nihilus::cli_params cli_args_01 = nihilus::harbinger<model_config_01>::parse_cli_arguments(argc, argv);
	nihilus::aligned_vector<std::unique_ptr<nihilus::model_base>> models{};
	models.emplace_back(nihilus::harbinger<model_config_00>::parse_model_graph_data(cli_args_00));
	models.emplace_back(nihilus::harbinger<model_config_01>::parse_model_graph_data(cli_args_01));
	while (true) {
		for (auto& value: models) {
			value->process_input();
		}
	}
	return 0;
}