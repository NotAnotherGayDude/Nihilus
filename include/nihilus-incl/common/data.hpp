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

#include <nihilus-incl/common/config.hpp>

namespace nihilus {

	template<typename data_type_new> struct data {
		using data_type = data_type_new;

		NIHILUS_HOST data_type* get_data() {
			return data_val;
		}

		NIHILUS_HOST void set_data(data_type* data_val_new) {
			data_val = data_val_new;
		}

	  protected:
		data_type* data_val{};
	};

	template<typename data_type_new, data_strategy_types data_strategy_type, uint64_t byte_count_new, uint64_t block_count_new = 0> struct data_holder;

	template<typename data_type_new, uint64_t byte_count_new, uint64_t block_count_new>
	struct data_holder<data_type_new, data_strategy_types::global, byte_count_new, block_count_new> {
		using data_type = data_type_new;
		static constexpr uint64_t byte_count{ byte_count_new };

		NIHILUS_HOST data_type* get_data() {
			return data_val.get_data();
		}

		NIHILUS_HOST void set_data(data_type* data_val_new) {
			data_val.set_data(data_val_new);
		}

	  protected:
		data<data_type_new> data_val{};
	};

	template<typename data_type_new, uint64_t byte_count_new, uint64_t block_count_new>
	struct data_holder<data_type_new, data_strategy_types::per_block, byte_count_new, block_count_new> {
		using data_type = data_type_new;
		static constexpr uint64_t block_count{ block_count_new };
		static constexpr uint64_t byte_count{ byte_count_new };

		NIHILUS_HOST data_type* get_data(uint64_t index) {
			return data_val[index].get_data();
		}

		NIHILUS_HOST void set_data(uint64_t index, data_type* data_val_new) {
			data_val[index].set_data(data_val_new);
		}

	  protected:
		data<data_type_new> data_val[block_count]{};
	};

}
