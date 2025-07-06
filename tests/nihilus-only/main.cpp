#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

//#include <BnchSwt/BenchmarkSuite.hpp>
#include <nihilus/index.hpp>

using namespace nihilus;
/*
template<typename EnumType> NIHILUS_FORCE_INLINE static consteval auto get_enum_count() {
	if constexpr (std::is_same_v<EnumType, model_sizes>) {
		return static_cast<uint64_t>(model_sizes::count);
	} else if constexpr (std::is_same_v<EnumType, model_arches>) {
		return static_cast<uint64_t>(model_arches::count);
	}
}

template<uint64_t N> using make_index_sequence_t = std::make_index_sequence<N>;

using model_sizes_indices  = make_index_sequence_t<static_cast<uint64_t>(model_sizes::count)>;
using model_arches_indices = make_index_sequence_t<static_cast<uint64_t>(model_arches::count)>;

template<uint64_t size_idx, uint64_t arch_idx> NIHILUS_FORCE_INLINE static consteval auto make_single_config() {
	return generate_model_config(model_generations::v1_v2,
		static_cast<model_sizes>(size_idx),
		kernel_type_profiles::fp16_mha,
		static_cast<model_arches>(arch_idx),
		false,
		kv_cache_strategies::paged,
		false,
		rope_scaling_types::linear,
		tokenizer_pre_types::llama3,
		16,
		true,
		true,
		norm_types::rms_standard,
		vocab_types::bpe,
		model_format::gguf,
		1e-6f,
		false
	);
}

template<uint64_t size_idx, size_t... arch_indices> NIHILUS_FORCE_INLINE static consteval auto make_configs_for_size(std::index_sequence<arch_indices...>) {
	return nihilus::make_tuple(make_single_config<size_idx, arch_indices>()...);
}

template<size_t... size_indices> NIHILUS_FORCE_INLINE static consteval auto generate_all_combinations_fold(std::index_sequence<size_indices...>) {
	return nihilus::tuple_cat(make_configs_for_size<size_indices>(model_arches_indices{})...);
}

NIHILUS_FORCE_INLINE static consteval auto get_all_model_configurations() {
	return generate_all_combinations_fold(model_sizes_indices{});
}

template<size_t... Indices> NIHILUS_FORCE_INLINE static consteval auto generate_cartesian_product(std::index_sequence<Indices...>) {
	constexpr auto arch_count = get_enum_count<model_arches>();

	return nihilus::make_tuple(make_single_config<Indices / arch_count,
		Indices % arch_count
		>()...);
}

NIHILUS_FORCE_INLINE static consteval auto get_all_model_configurations_single_fold() {
	constexpr auto size_count		  = get_enum_count<model_sizes>();
	constexpr auto arch_count		  = get_enum_count<model_arches>();
	constexpr auto total_combinations = size_count * arch_count;

	return generate_cartesian_product(std::make_index_sequence<total_combinations>{});
}

inline static constexpr auto all_model_configurations = get_all_model_configurations_single_fold();

using all_model_configs_tuple_type = decltype(all_model_configurations);

inline static constexpr auto total_model_combinations = nihilus::tuple_size_v<all_model_configs_tuple_type>;

template<uint64_t index> NIHILUS_FORCE_INLINE static consteval auto get_model_config() {
	static_assert(index < total_model_combinations, "Index out of range for model configurations");
	return nihilus::get<index>(all_model_configurations);
}

template<model_sizes size, model_arches arch> NIHILUS_FORCE_INLINE static consteval auto find_model_config() {
	constexpr auto arch_count		 = get_enum_count<model_arches>();
	constexpr auto size_index		 = static_cast<uint64_t>(size);
	constexpr auto arch_index		 = static_cast<uint64_t>(arch);
	constexpr auto combination_index = size_index * arch_count + arch_index;

	return get_model_config<combination_index>();
}


struct alignas(64) op_latch_branchless {
	NIHILUS_FORCE_INLINE op_latch_branchless() noexcept = default;
	NIHILUS_FORCE_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0, std::memory_order_release);
	}
	NIHILUS_FORCE_INLINE void reset() {
		flag.store(0, std::memory_order_release);
	}
	NIHILUS_FORCE_INLINE void arrive_and_wait() {
		auto new_value			  = flag.fetch_add(1, std::memory_order_acq_rel);
		constexpr int64_t garbage = -1;
		int64_t expected		  = new_value ^ ((new_value ^ garbage) & -(new_value == get_thread_count() - 1));
		flag.wait(expected + 1, std::memory_order_acquire);
		flag.notify_all();
	}
	NIHILUS_FORCE_INLINE int64_t get_thread_count() const noexcept {
		return thread_count;
	}
	alignas(64) std::atomic<int64_t> flag{};
	alignas(64) int64_t thread_count{};
};

struct alignas(64) op_latch_branchless_real {
	NIHILUS_FORCE_INLINE op_latch_branchless_real() noexcept = default;

	NIHILUS_FORCE_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0, std::memory_order_release);
	}

	NIHILUS_FORCE_INLINE void arrive_and_wait() {
		auto new_value					 = flag.fetch_add(1, std::memory_order_acq_rel);
		static constexpr int64_t garbage = -1;
		int64_t expected				 = new_value ^ ((new_value ^ garbage) & -(new_value == get_thread_count() - 1));
		(new_value == thread_count - 1) ? (flag.notify_all(), flag.store(0, std::memory_order_release)) : flag.wait(new_value + 1, std::memory_order_acquire);
	}

	NIHILUS_FORCE_INLINE int64_t get_thread_count() const noexcept {
		return thread_count;
	}

  protected:
	alignas(64) std::atomic<int64_t> flag{};
	alignas(64) int64_t thread_count{};
};

struct alignas(64) op_latch {
	NIHILUS_FORCE_INLINE op_latch() noexcept = default;
	NIHILUS_FORCE_INLINE void init(uint64_t thread_count_new) {
		thread_count = thread_count_new;
		flag.store(0, std::memory_order_release);
	}

	NIHILUS_FORCE_INLINE void reset() {
		flag.store(0, std::memory_order_release);
	}

	NIHILUS_FORCE_INLINE void arrive_and_wait() {
		auto new_value				 = flag.fetch_add(1, std::memory_order_acq_rel);
		constexpr int64_t garbage	 = -1;
		static constexpr auto lambda = []<uint64_t I>(std::atomic_signed_lock_free& flag_new, int64_t expected_new) {
			if constexpr (I == 0) {
				flag_new.notify_all();
			} else {
				flag_new.wait(expected_new, std::memory_order_acquire);
			}
		};
		static constexpr auto jump_table = []<size_t... I>(std::index_sequence<I...>) {
			return nihilus::array{ +[](std::atomic_signed_lock_free& flag_new, int64_t expected_new, decltype(lambda)& l) {
				l.template operator()<I>(flag_new, expected_new);
			}... };
		}(std::make_index_sequence<2>{});
		uint64_t jump_index = -(new_value == thread_count - 1) & 0;
		jump_index |= (~-(new_value == thread_count - 1)) & 1;
		jump_table[jump_index](flag, new_value + 1, lambda);
	}

	NIHILUS_FORCE_INLINE int64_t get_thread_count() const noexcept {
		return thread_count;
	}

	alignas(64) std::atomic_signed_lock_free flag{};
	alignas(64) int64_t thread_count{};
};

::op_latch latch{};
*/
std::thread spawn_thread(::op_latch& latch, uint64_t thread_index) {
	return std::thread{ [&, thread_index] {
		std::cout << "THREAD INDEX: " << thread_index << std::endl;
		latch.arrive_and_wait();
	} };
}

template<uint64_t index> struct base_test_struct {
	NIHILUS_FORCE_INLINE void test_function() {
		std::cout << "CURRENT INDEX: " << index << std::endl;
	}
};

struct test_struct : public base_test_struct<0>, public base_test_struct<1> {
	template<uint64_t index> NIHILUS_FORCE_INLINE static void impl(test_struct* value);
	using function_type = decltype(&impl<0>);
	static constexpr nihilus::array<function_type, 2> values{ [] {
		nihilus::array<function_type, 2> return_values{};
		return_values[0] = impl<0>;
		return_values[0] = impl<1>;
		return return_values;
	}() };
	test_struct() {
	}
};

template<uint64_t index> NIHILUS_FORCE_INLINE void test_struct::impl(test_struct* value) {
	static_cast<base_test_struct<index>*>(value)->test_function();
}
int32_t main(int32_t argc, char** argv) {
	try {
		nihilus::op_latch latch{};
		latch.init(4);
		auto thread01					   = spawn_thread(latch, 0);
		auto thread02					   = spawn_thread(latch, 1);
		auto thread03					   = spawn_thread(latch, 2);
		auto thread04					   = spawn_thread(latch, 3); 
		if (thread01.joinable()) {
			thread01.join();
		}
		if (thread02.joinable()) {
			thread02.join();
		}
		if (thread03.joinable()) {
			thread03.join();
		}
		if (thread04.joinable()) {
			thread04.join();
		}
		latch.init(4);
		thread01 = spawn_thread(latch, 0);
		thread02 = spawn_thread(latch, 1);
		thread03 = spawn_thread(latch, 2);
		thread04 = spawn_thread(latch, 3);
		if (thread01.joinable()) {
			thread01.join();
		}
		if (thread02.joinable()) {
			thread02.join();
		}
		if (thread03.joinable()) {
			thread03.join();
		}
		if (thread04.joinable()) {
			thread04.join();
		}
		/*
		static constexpr auto model_config = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
			nihilus::model_arches::llama, false);
		nihilus::cli_params cli_args_final;
		cli_args_final = nihilus::harbinger<model_config>::parse_cli_arguments(argc, argv);
		using model_type = nihilus::harbinger<model_config>::model_type;
		auto model_new{ nihilus::harbinger<model_config>::parse_model_graph_data(cli_args_final) };*/
		return 0;
		//cli_args_final.n_tokens;
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}