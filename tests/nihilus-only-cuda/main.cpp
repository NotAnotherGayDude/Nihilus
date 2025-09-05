#include <nihilus>

using namespace nihilus;

int32_t main(int32_t argc, char** argv) {
	print_cuda_arch();
	constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
	constexpr auto model_config_00 = update_model_device_type(model_config, device_types::gpu);
	constexpr auto model_config_01 = update_model_config_benchmark(model_config_00, true);
	//constexpr auto model_config_01	= update_model_config_dev(model_config_001, true);
	const cli_params cli_args = harbinger<model_config_01>::parse_cli_arguments(argc, argv);
	auto model_new_01{ harbinger<model_config_01>::parse_model_graph_data(cli_args) };
	while (model_new_01->process_input(cli_args.prompt)) {
	}
	return 0;
};