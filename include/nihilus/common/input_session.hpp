/*
Copyright (c) 2025 RealTimeChris (Chris M.)

This file is part of software offered under a restricted-use license to a designated Licensee,
whose identity is confirmed in writing by the Author.

License Terms (Summary):
- Exclusive, non-transferable license for internal use only.
- Redistribution, sublicensing, or public disclosure is prohibited without written consent.
- Full ownership remains with the Author.
- License may terminate if unused for [X months], if materially breached, or by mutual agreement.
- No warranty is provided, express or implied.

Full license terms are provided in the LICENSE file distributed with this software.

Signed,
RealTimeChris (Chris M.)
2025
*/

#pragma once

#include <nihilus/common/tokenizer.hpp>
#include <nihilus/cpu/thread_pool.hpp>
#include <nihilus/common/config.hpp>
#include <iterator>

namespace nihilus {

	struct input_session_config {
		NIHILUS_FORCE_INLINE input_session_config& operator=(const input_session_config&) = delete;
		NIHILUS_FORCE_INLINE input_session_config(const input_session_config&)			  = delete;
		NIHILUS_FORCE_INLINE input_session_config(std::istream& stream_new, uint64_t max_tokens_new) : stream{ stream_new }, max_tokens{ max_tokens_new } {};
		std::istream& stream;
		uint64_t max_tokens{};
	};

	template<model_config config, typename model_type> struct input_session
		: public tokenizer<config, model_type, model_traits<config.arch, config.model_size, config.model_generation>::arch, config.vocab_type> {
		using tokenizer_type	= tokenizer<config, model_type, model_traits<config.arch, config.model_size, config.model_generation>::arch, config.vocab_type>;
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

		NIHILUS_FORCE_INLINE input_session() noexcept = default;

		NIHILUS_FORCE_INLINE input_session(const input_session_config& config_new) : stream{ &config_new.stream } {
			exec_params.token_count = config_new.max_tokens;
			//exec_params.thread_count = model.get_thread_count();// Assuming thread_pool has this method
		};

		NIHILUS_FORCE_INLINE bool process_input(const std::string& input) {
			// Tokenize the input using the model's tokenizer
			exec_params.token_count = tokenizer_type::tokenize(input, static_cast<model_type*>(this)->template get_core<model_type::op_type_type::inp_tokens>().data);
			for (size_t x = 0; x < 8; ++x) {
				std::cout << "CURRENT TOkEN: " << static_cast<model_type*>(this)->template get_core<model_type::op_type_type::inp_tokens>().data[x] << std::endl;
			}

			// Execute the model
			static_cast<model_type*>(this)->execute_model(exec_params);
			std::cout << "FOR " << exec_params.thread_count << " THREADS, WITH " << 500 << " NANOSECONDS OF SPINLOCK PER KERNEL, "
					  << "NIHILUS AVERAGE COMPUTE TIME, OVER: " << std::setw(50 - std::size("NIHILUS AVERAGE COMPUTE TIME, OVER: ")) << nihilus::stop_watch_val_nihilus.get_count()
					  << " TOKENS: " << nihilus::stop_watch_val_nihilus.get_average() << std::endl;
			return false;
		}

		execution_parameters exec_params{};

	  private:
		std::istream* stream{};

		NIHILUS_FORCE_INLINE std::string get_input_text() {
			// For demo purposes, return the expected input
			// In reality, you'd read from stdin, file, or network
			return "\nWhat is the meaning of life?";
		}
	};

}
