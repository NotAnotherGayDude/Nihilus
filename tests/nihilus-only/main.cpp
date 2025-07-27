#include <nihilus>

int32_t main(int32_t argc, char** argv) {
	static constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);

	nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new->process_input(cli_args.prompt)) {
	}
	return 0;
}
