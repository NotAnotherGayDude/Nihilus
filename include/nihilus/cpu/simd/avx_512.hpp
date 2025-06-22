#pragma once
#include <nihilus/common/kernel_traits.hpp>

#if defined(NIHILUS_AVX522)

namespace nihilus {

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::copy, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_type::copy, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::copy, transform_type, core_type, half, float>
		: public kernel_base<core_type::type, kernel_type::copy, core_type, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::cont, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_type::cont, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::silu, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_type::silu, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::rms_norm, transform_type, core_type, float, float>
		: public kernel_base<core_type::type, kernel_type::rms_norm, core_type, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::get_rows, transform_type, core_type, float, block_q8_0<half>, int32_t>
		: public kernel_base<core_type::type, kernel_type::get_rows, core_type, float, block_q8_0<half>, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::get_rows, transform_type, core_type, float, float, int32_t>
		: public kernel_base<core_type::type, kernel_type::get_rows, core_type, float, float, int32_t> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_type::mul, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul, transform_type, core_type, float, float, block_q8_0<half>>
		: public kernel_base<core_type::type, kernel_type::mul, core_type, float, float, block_q8_0<half>> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, core_type, float, block_q8_0<half>, float>
		: public kernel_base<core_type::type, kernel_type::mul_mat, core_type, float, block_q8_0<half>, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_type::mul_mat, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::mul_mat, transform_type, core_type, float, half, float>
		: public kernel_base<core_type::type, kernel_type::mul_mat, core_type, float, half, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::softmax, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_type::softmax, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::add, transform_type, core_type, float, float, float>
		: public kernel_base<core_type::type, kernel_type::add, core_type, float, float, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02) {
		}
	};

	template<typename transform_type, typename core_type> struct kernel_dispatcher_impl<2, kernel_type::rope, transform_type, core_type, float, float, int32_t, float>
		: public kernel_base<core_type::type, kernel_type::rope, core_type, float, float, int32_t, float> {
		NIHILUS_FORCE_INLINE static void impl(uint64_t thread_index, uint64_t thread_count, core_type& output, const typename core_type::input_type01& input01,
			const typename core_type::input_type02& input02, const typename core_type::input_type03& input03) {
		}
	};

};

#endif
