#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nihilus/index.hpp>

#if !defined(LLAMA_MODEL_SIZE)
static constexpr nihilus::model_sizes model_size{ nihilus::model_sizes::llm_8B };
#else
static constexpr nihilus::model_sizes model_size{ LLAMA_MODEL_SIZE };
#endif

int32_t main(int32_t argc, char** argv) {
	try {
		static constexpr auto model_config =
			nihilus::generate_model_config(nihilus::model_generations::v3, model_size, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		nihilus::cli_params cli_args_final{ nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv) };
		auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args_final) };
		while (model_new->process_input(cli_args_final.prompt)) {
		}
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}