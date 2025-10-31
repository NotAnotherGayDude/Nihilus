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

	enum class kernel_traits_errors {
		incorrect_rms_norm_dimensions,
		incorrect_add_dimensions,
		incorrect_sub_dimensions,
	};

	template<bool batched, kernel_types kernel_type, typename... input_dims_types> struct kernel_traits;

	template<bool batched, typename input_type_01_new, typename input_type_02_new> struct kernel_traits<batched, kernel_types::rms_norm, input_type_01_new, input_type_02_new> {
		static constexpr auto input_01_dims = input_type_01_new::dims;
		static constexpr auto input_02_dims = input_type_02_new::dims;
		static constexpr bool equals{ (input_01_dims[0] == input_02_dims[0]) && (input_01_dims[1] == input_02_dims[1]) && (input_01_dims[2] == input_02_dims[2]) &&
			(input_01_dims[3] == input_02_dims[3]) };
		static_assert(static_assert_printer_val<equals, kernel_traits_errors::incorrect_rms_norm_dimensions, input_01_dims[0], input_01_dims[1], input_01_dims[2], input_01_dims[3],
			input_02_dims[0], input_02_dims[1], input_02_dims[2], input_02_dims[3]>::impl);
		//static constexpr 
	};

	template<bool batched, typename input_type_01_new, typename input_type_02_new> struct kernel_traits<batched, kernel_types::add, input_type_01_new, input_type_02_new> {
		using input_type_01 = input_type_01_new;
		using input_type_02 = input_type_02_new;
		//static_assert
	};

}
