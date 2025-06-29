#pragma once
#include <nihilus/common/kernel_traits.hpp>

#if defined(NIHILUS_AVX522)

namespace nihilus {

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::add_rms_norm_mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add_rms_norm_mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_types::copy, core_type, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::cont, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::silu, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_types::rms_norm, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_types::get_rows, core_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_types::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_types::mul_mat, core_type, float, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::softmax, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_types::add, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_types::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_types::rope, core_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02, const typename core_type::input_type03& input03) {
		}
	};

};

#endif
