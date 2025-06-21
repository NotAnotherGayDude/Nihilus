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
		NIHILUS_FORCE_INLINE input_session_config(const input_session_config&)			= delete;
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
		using base_type = input_session_base;

		NIHILUS_FORCE_INLINE input_session() noexcept = default;
		NIHILUS_FORCE_INLINE input_session(const input_session_config&, model_type& model) : model_ptr{ &model } {
			exec_params.thread_count = model.thread_count;
		};

		NIHILUS_FORCE_INLINE bool process_input() {
			this->tokenize(input, model_ptr->template get_core<model_type::op_type_type::inp_tokens>().data);
			model_ptr->execute_model(exec_params);
			std::cout << "FOR " << exec_params.thread_count << " THREADS, WITH " << spinlock_time << " NANOSECONDS OF SPINLOCK PER KERNEL, "
					  << "NIHILUS AVERAGE COMPUTE TIME, OVER: " << std::setw(50 - std::size("NIHILUS AVERAGE COMPUTE TIME, OVER: ")) << stop_watch_val_nihilus.get_count()
					  << " TOKENS: " << stop_watch_val_nihilus.get_average() << std::endl;
			return false;
		}

		NIHILUS_FORCE_INLINE operator bool() {
			return false;
		}

	  protected:
		model_type* model_ptr{};
		std::string input{};
	};

}
