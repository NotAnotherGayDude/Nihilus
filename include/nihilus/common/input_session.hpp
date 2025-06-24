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

	struct input_session_base {
		virtual bool process_input() = 0;
		virtual operator bool()		 = 0;

		execution_parameters exec_params{};
		virtual ~input_session_base() noexcept = default;
	};

	template<typename model_type> struct input_session : public input_session_base, public tokenizer<model_type::model_traits_type::arch> {
		using base_type			= input_session_base;
		using tokenizer_type	= tokenizer<model_type::model_traits_type::arch>;
		using model_traits_type = typename model_type::model_traits_type;

		NIHILUS_FORCE_INLINE input_session() noexcept = default;

		NIHILUS_FORCE_INLINE input_session(const input_session_config& config, model_type& model) : model_ptr{ &model } {
			exec_params.token_count	 = config.max_tokens;
		};

		NIHILUS_FORCE_INLINE bool process_input() {
			//this->tokenize(input, model_ptr->template get_core<model_type::op_type_type::inp_tokens>().data);
			model_ptr->execute_model(exec_params);
			std::cout << "FOR " << exec_params.thread_count << " THREADS, WITH " << 500 << " NANOSECONDS OF SPINLOCK PER KERNEL, "
					  << "NIHILUS AVERAGE COMPUTE TIME, OVER: " << std::setw(50 - std::size("NIHILUS AVERAGE COMPUTE TIME, OVER: ")) << stop_watch_val_nihilus.get_count()
					  << " TOKENS: " << stop_watch_val_nihilus.get_average() << std::endl;
			return false;
		}

		/*
		NIHILUS_FORCE_INLINE bool process_input() {
			// Get input text (you'd implement input reading here)
			std::string input_text = get_input_text();

			if (input_text.empty()) {
				return false;
			}

			// Tokenize input text into the model's input tensor
			auto& token_tensor = model_ptr->template get_core<model_traits_type::op_type_type::inp_tokens>();
			size_t token_count = this->tokenize(input_text, token_tensor.data, token_tensor.dim00);

			// Update execution parameters
			exec_params.token_count	 = token_count;
			exec_params.input_tokens = token_tensor.data;

			// Execute the model
			model_ptr->execute_model(exec_params);

			// Output performance metrics
			std::cout << "FOR " << exec_params.thread_count << " THREADS, WITH " << spinlock_time << " NANOSECONDS OF SPINLOCK PER KERNEL, "
					  << "NIHILUS AVERAGE COMPUTE TIME, OVER: " << std::setw(50 - std::size("NIHILUS AVERAGE COMPUTE TIME, OVER: ")) << stop_watch_val_nihilus.get_count()
					  << " TOKENS: " << stop_watch_val_nihilus.get_average() << std::endl;

			return token_count > 0;
		}*/

		NIHILUS_FORCE_INLINE operator bool() {
			return model_ptr != nullptr;
		}

	  private:
		model_type* model_ptr{};

		// Get input text - you'd implement this based on your input method
		NIHILUS_FORCE_INLINE std::string get_input_text() {
			// Placeholder - implement based on whether you're reading from:
			// - Command line arguments
			// - Standard input
			// - File
			// - Interactive prompt
			return "What is the meaning of life?";// Example
		}
	};

}
