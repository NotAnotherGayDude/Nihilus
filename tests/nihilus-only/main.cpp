#include <nihilus>
#include <iostream>
using namespace nihilus;
op_latch latch{};

enum test_enum { one, two, three, four, five };

template<model_config config> struct core_traits_new {};

void test_function(size_t index) {
	std::string new_string{ "THREAD: " + std::to_string(index) + " WAITING!" };
	nihilus::log<nihilus::log_levels::status>(new_string);
	if (index == 0) {
		std::this_thread::sleep_for(std::chrono::seconds{ 4 });
	}
	latch.arrive_and_wait();
	std::string new_string02{ "THREAD: " + std::to_string(index) + " FINSIHEED!" };
	nihilus::log<nihilus::log_levels::status>(new_string02);
}

int32_t main(int32_t argc, char** argv) {
	latch.init(2);
	std::thread thread01{ [] {
		test_function(0);
	} };
	std::thread thread02{ [] {
		test_function(1);
	} };
	/*
	std::jthread thread03{ [] {
		test_function(2);
	} };
	std::jthread thread04{ [] {
		test_function(3);
	} };
	std::jthread thread05{ [] {
		test_function(4);
	} };
	std::jthread thread06{ [] {
		test_function(5);
	} };
	std::jthread thread07{ [] {
		test_function(6);
	} }; */
	thread01.join();
	thread02.join();
	static constexpr auto model_config =
		nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);

	nihilus::cli_params cli_args = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
	auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args) };
	while (model_new->process_input(cli_args.prompt)) {
	}
	return 0;
}
