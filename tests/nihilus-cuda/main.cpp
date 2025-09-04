#include <nihilus>

using namespace nihilus;

int32_t main(int32_t argc, char** argv) {
	std::cout << "CURRENT ENUM VALUE: " << std::thread::hardware_concurrency() << std::endl;
	print_cuda_arch();
	constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
	const cli_params cli_args = harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new_01{ harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new_01->process_input(cli_args.prompt)) {
	}
	return 0;
};