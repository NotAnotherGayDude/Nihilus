#include <nihilus>

using namespace nihilus;

struct core_base_base {
  protected:
	~core_base_base() noexcept = default;
};

template<core_types core_type>
struct core_base : public core_base_base {
};

struct model : public core_base<core_types::count>, core_base<core_types::final_norm_and_sampling>, core_base<core_types::mega_attention_apply> {
	model() {
		for (size_t x = 0; x < static_cast<size_t>(core_types::count); ++x) {
			bases.emplace_back();
		}
	}
	aligned_vector<core_base_base*> bases{};
};

int32_t main(int32_t argc, char** argv) {
	std::cout << "CURRENT ENUM VALUE: " << nihilus::model_sizes::llm_109M << std::endl;
	print_cuda_arch();
	constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3_1, model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
	constexpr auto model_config_01 = nihilus::update_model_config_benchmark(model_config, true);
	constexpr auto model_config_02 = nihilus::update_model_device_type(model_config_01, device_types::gpu);
	const cli_params cli_args	   = harbinger<model_config_02>::parse_cli_arguments(argc, argv);
	auto model_new_01{ harbinger<model_config_02>::parse_model_graph_data(cli_args) };
	while (model_new_01->process_input(cli_args.prompt)) {
	}
	return 0;
};