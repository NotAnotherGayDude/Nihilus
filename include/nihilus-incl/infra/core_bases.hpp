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

#include <nihilus-incl/infra/behavioral_axes.hpp>

namespace nihilus {

	template<const model_config& config, typename... bases> struct core_bases : public bases... {
		using bases::operator[]...;
		NIHILUS_INLINE core_bases()				 = default;
		core_bases& operator=(core_bases&&)		 = delete;
		core_bases(core_bases&&)				 = delete;
		core_bases& operator=(const core_bases&) = delete;
		core_bases(const core_bases&)			 = delete;

		template<template<const model_config&, typename> typename mixin_type, typename... arg_types> NIHILUS_INLINE constexpr void impl(arg_types&&... args) noexcept {
			(impl_internal_filtered<mixin_type, bases>(detail::forward<arg_types>(args)...), ...);
		}

		template<template<const model_config&, typename, auto...> typename mixin_type, auto... values, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_thread(arg_types&&... args) noexcept {
			(impl_internal_filtered_thread<mixin_type, bases, values...>(detail::forward<arg_types>(args)...), ...);
		}

		template<enum_types enum_type, enum_type enum_value> NIHILUS_INLINE decltype(auto) get_core() noexcept {
			return (*this)[tag<enum_value>()];
		}

		template<enum_types enum_type, enum_type enum_value> NIHILUS_INLINE static decltype(auto) get_core_static() noexcept {
			return core_bases{}.template get_core<enum_type, enum_value>();
		}

		template<enum_types enum_type, enum_type enum_value> using core_base_type = decltype(get_core_static<enum_type, enum_value>());

		core_bases* data_ptr{};

	  protected:
		template<template<const model_config&, typename> typename mixin_type, typename base_type, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered([[maybe_unused]] arg_types&&... args) noexcept {
			if constexpr (mixin_type<config, base_type>::filter()) {
				mixin_type<config, base_type>::impl(*static_cast<typename base_type::derived_type*>(this), detail::forward<arg_types>(args)...);
			}
		}

		template<template<const model_config&, typename, auto...> typename mixin_type, typename base_type, auto... values, typename... arg_types>
		NIHILUS_INLINE constexpr void impl_internal_filtered_thread([[maybe_unused]] arg_types&&... args) noexcept {
			if constexpr (mixin_type<config, base_type, values...>::filter()) {
				mixin_type<config, base_type, values...>::impl(*static_cast<typename base_type::derived_type*>(this), detail::forward<arg_types>(args)...);
			}
		}
	};

	template<const model_config& config, typename... value_type> struct get_core_bases {
		using type = core_bases<config, value_type...>;
	};

	template<const model_config& config, typename enum_type, size_t... index> struct get_core_bases<config, enum_type, std::index_sequence<index...>> {
		using type = core_bases<config, core_traits<config, static_cast<enum_type>(index)>...>;
	};

	template<const model_config& config, typename... value_type> using get_core_base_t = typename get_core_bases<config, value_type...>::type;

	template<const model_config& config, typename enum_type> using get_core_bases_t =
		typename get_core_bases<config, enum_type, std::make_index_sequence<static_cast<uint64_t>(enum_type::count)>>::type;

	template<const model_config& config, typename base_type>
	NIHILUS_INLINE void memory_mapper<config, base_type>::impl(base_type& parse_core, const memory_plan& plan, memory_buffer<config>& memory_buffer) {
		uint64_t internal_offset{};
		parse_core.values.template impl<memory_mapper_impl>(plan, memory_buffer, internal_offset);
		if constexpr (static_cast<uint64_t>(base_type::core_type) == static_cast<uint64_t>(core_types::count) - 1 && config.device_type == device_types::gpu) {
			static_cast<get_core_bases_t<config, core_types>*>(&parse_core)->data_ptr =
				static_cast<get_core_bases_t<config, core_types>*>(memory_buffer.claim_memory(plan.metadata_offset));
			memory_transfer<config>::host_to_device(*static_cast<get_core_bases_t<config, core_types>*>(&parse_core),
				static_cast<get_core_bases_t<config, core_types>*>(&parse_core)->data_ptr);
		}
		if constexpr (config.dev) {
#if NIHILUS_CUDA_ENABLED
			if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
				static constexpr auto location = std::source_location::current();
				nihilus_exception<config, "Cuda Error: ", location>::impl(cudaGetErrorString(err));
			}
#endif
		}
	}

	template<const model_config& config> struct core_bases_traits {
		static constexpr memory_plan total_required_bytes{ []() {
			auto return_values{ get_memory_plan<config>() };
			if constexpr (config.device_type == device_types::gpu) {
				return_values.metadata_offset = return_values.currently_allocated_bytes;
				return_values.currently_allocated_bytes += sizeof(get_core_bases_t<config, core_types>);
			}
			return return_values;
		}() };
	};
}
