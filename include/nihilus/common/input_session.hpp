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

	template<model_config config, typename model_type> struct input_session
		: public tokenizer<config, model_type, model_traits<config.arch, config.model_size, config.model_generation>::arch, config.vocab_type> {
		using tokenizer_type	= tokenizer<config, model_type, model_traits<config.arch, config.model_size, config.model_generation>::arch, config.vocab_type>;
		using model_traits_type = model_traits<config.arch, config.model_size, config.model_generation>;

		NIHILUS_FORCE_INLINE input_session() noexcept = default;

		NIHILUS_FORCE_INLINE input_session(const cli_params& config_new) {
			exec_params.token_count = config_new.n_tokens;
		};

		NIHILUS_FORCE_INLINE bool process_input_impl(const std::string& input) {
			exec_params.sequence_length = tokenizer_type::tokenize(input, static_cast<model_type*>(this)->template get_core<model_type::op_type_type::inp_tokens>().data);
			static_cast<model_type*>(this)->execute_model(exec_params);
			return false;
		}

		execution_parameters exec_params{};
	};

}
