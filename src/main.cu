#include <nihilus>
static constexpr auto model_config_01 = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::device_types::gpu, true, NIHILUS_DEV, 1024);
static constexpr auto model_config_02 = nihilus::generate_model_config(nihilus::model_generations::v3_2, nihilus::model_sizes::llm_3B, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::device_types::gpu, true, NIHILUS_DEV, 1024);
int main(int32_t argc, char** argv) {
	const nihilus::cli_params cli_args_01 = nihilus::harbinger<model_config_01>::parse_cli_arguments(argc, argv);
	const nihilus::cli_params cli_args_02 = nihilus::harbinger<model_config_02>::parse_cli_arguments(argc, argv);
	nihilus::aligned_vector<std::unique_ptr<nihilus::model_base>> models{};
	models.emplace_back(nihilus::harbinger<model_config_01>::parse_model_graph_data(cli_args_01));
	models.emplace_back(nihilus::harbinger<model_config_02>::parse_model_graph_data(cli_args_02));
	while (true) {
		for (auto& value: models) {
			value->process_input();
		}
	}
	return 0;
}