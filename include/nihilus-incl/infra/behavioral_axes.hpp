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

#include <nihilus-incl/infra/monolithic_dispatcher.hpp>
#include <nihilus-incl/common/type_traits.hpp>
#include <nihilus-incl/cpu/memory_buffer.hpp>
#include <nihilus-incl/common/common.hpp>
#include <nihilus-incl/common/tuple.hpp>
#include <atomic>
#include <thread>
#include <latch>

namespace nihilus {

	struct core_traits_memory_footprint {
		uint64_t total_required_bytes{};
		core_types core_type{};
		uint64_t offset{};
		uint64_t depth{};
		bool is_active{};
	};

	struct memory_plan {
		array<core_traits_memory_footprint, core_types::count> footprints{};
		uint64_t currently_allocated_bytes{};
		uint64_t peak_allocated_bytes{};
	};

	template<device_types device_type> constexpr uint64_t base_index{ [] {
		if constexpr (device_type == device_types::cpu) {
			return 1;
		} else {
			return 0;
		}
	}() };

	consteval bool is_valid_free_type(core_traits_memory_footprint footprint, uint64_t current_depth) {
		uint64_t threshold = current_depth - 2;
		return footprint.is_active && footprint.depth <= threshold && footprint.depth != std::numeric_limits<uint64_t>::max();
	}

	template<typename config_type, uint64_t current_index = base_index<config_type::device_type>> consteval memory_plan get_memory_plan(memory_plan values = {}) {
		constexpr uint64_t max_index{ static_cast<uint64_t>(core_types::count) };
		if constexpr (current_index == base_index<config_type::device_type>) {
			values.currently_allocated_bytes = 0;
			values.peak_allocated_bytes		 = 0;
		}
		if constexpr (current_index < max_index) {
			values.footprints[current_index].offset				  = values.currently_allocated_bytes;
			values.footprints[current_index].core_type			  = static_cast<core_types>(current_index);
			values.footprints[current_index].depth				  = core_traits<config_type, static_cast<core_types>(current_index)>::depth;
			values.footprints[current_index].is_active			  = true;
			values.footprints[current_index].total_required_bytes = core_traits<config_type, static_cast<core_types>(current_index)>::total_required_bytes;
			values.currently_allocated_bytes += core_traits<config_type, static_cast<core_types>(current_index)>::total_required_bytes;
			if (values.currently_allocated_bytes > values.peak_allocated_bytes) {
				values.peak_allocated_bytes = values.currently_allocated_bytes;
			}
			constexpr uint64_t cur_depth = core_traits<config_type, static_cast<core_types>(current_index)>::depth;
			if constexpr (cur_depth >= 2) {
				for (int64_t x = 0; x < static_cast<int64_t>(current_index); ++x) {
					if (is_valid_free_type(values.footprints[static_cast<uint64_t>(x)], cur_depth)) {
						values.footprints[static_cast<uint64_t>(x)].is_active = false;
						values.currently_allocated_bytes -= values.footprints[static_cast<uint64_t>(x)].total_required_bytes;
					}
				}
			}
			return get_memory_plan<config_type, current_index + 1>(values);
		} else {
			return values;
		}
	}

	template<typename config_type> struct thread_pool;

	template<typename config_type, typename base_type_new> struct memory_mapper_impl {
		NIHILUS_HOST memory_mapper_impl() noexcept {
		}
		NIHILUS_HOST memory_mapper_impl& operator=(const memory_mapper_impl&) noexcept = delete;
		NIHILUS_HOST memory_mapper_impl(const memory_mapper_impl&) noexcept			   = delete;
		NIHILUS_HOST memory_mapper_impl& operator=(memory_mapper_impl&&) noexcept	   = delete;
		NIHILUS_HOST memory_mapper_impl(memory_mapper_impl&&) noexcept				   = delete;
		using base_type																   = base_type_new;
		using base_derived_type														   = typename base_type::derived_type;
		NIHILUS_HOST static constexpr bool filter() {
			return has_total_required_bytes_types<base_derived_type>;
		}
		NIHILUS_HOST static void impl(base_derived_type& core_traits, const memory_plan& plan, memory_buffer<config_type>& memory_buffer, uint64_t& internal_offset) {
			using data_type = typename base_derived_type::output_type;
			if constexpr (base_type::data_strategy_type == data_strategy_types::per_block) {
				for (uint64_t x = 0; x < model_traits_type<config_type>::block_count; ++x) {
					data_type* ptr	= static_cast<data_type*>(memory_buffer.claim_memory(plan.footprints[base_type::derived_type::core_type].offset + internal_offset));
					internal_offset = core_traits.total_required_bytes;
					core_traits.set_data(ptr, x);
				}
			} else {
				data_type* ptr	= static_cast<data_type*>(memory_buffer.claim_memory(plan.footprints[base_type::derived_type::core_type].offset + internal_offset));
				internal_offset = core_traits.total_required_bytes;
				core_traits.set_data(ptr);
			}
		}
	};

	template<typename config_type, typename base_type_new> struct memory_mapper {
		NIHILUS_HOST memory_mapper() noexcept {
		}
		NIHILUS_HOST memory_mapper& operator=(const memory_mapper&) noexcept = delete;
		NIHILUS_HOST memory_mapper(const memory_mapper&) noexcept			 = delete;
		NIHILUS_HOST memory_mapper& operator=(memory_mapper&&) noexcept		 = delete;
		NIHILUS_HOST memory_mapper(memory_mapper&&) noexcept				 = delete;
		using base_type														 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::has_total_required_bytes;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, const memory_plan& plan, memory_buffer<config_type>& memory_buffer) {
			uint64_t internal_offset{};
			parse_core.values.template impl<memory_mapper_impl>(plan, memory_buffer, internal_offset);
		}
	};

	template<typename config_type, typename base_type_new> struct tensor_debugger {
		NIHILUS_HOST tensor_debugger() noexcept {
		}
		NIHILUS_HOST tensor_debugger& operator=(const tensor_debugger&) noexcept = delete;
		NIHILUS_HOST tensor_debugger(const tensor_debugger&) noexcept			 = delete;
		NIHILUS_HOST tensor_debugger& operator=(tensor_debugger&&) noexcept		 = delete;
		NIHILUS_HOST tensor_debugger(tensor_debugger&&) noexcept				 = delete;
		using base_type															 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return true;
		}

		NIHILUS_HOST static void impl(base_type&, uint64_t, core_types) {
		}
	};

	template<typename config_type, typename base_type_new> struct sync_resetter {
		NIHILUS_HOST sync_resetter() noexcept {
		}
		NIHILUS_HOST sync_resetter& operator=(const sync_resetter&) noexcept = delete;
		NIHILUS_HOST sync_resetter(const sync_resetter&) noexcept			 = delete;
		NIHILUS_HOST sync_resetter& operator=(sync_resetter&&) noexcept		 = delete;
		NIHILUS_HOST sync_resetter(sync_resetter&&) noexcept				 = delete;
		using base_type														 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return has_chunk_types<base_type>;
		}
		NIHILUS_HOST static void impl(base_type& parse_core, int64_t thread_count) {
			if constexpr (array_types<decltype(parse_core.current_chunk_eval)>) {
				for (uint64_t x = 0; x < model_traits_type<config_type>::block_count; ++x) {
					parse_core.current_chunk_eval[x].store(0);
					parse_core.current_chunk_prompt_eval[x].store(0);
					parse_core.latch_eval[x].store(thread_count);
					parse_core.latch_prompt_eval[x].store(thread_count);
				}
			} else {
				parse_core.current_chunk_eval.store(0);
				parse_core.current_chunk_prompt_eval.store(0);
				parse_core.latch_eval.store(thread_count);
				parse_core.latch_prompt_eval.store(thread_count);
			}
		}
	};

	template<typename config_type, typename base_type_new> struct dim_updater_impl {
		NIHILUS_HOST dim_updater_impl() noexcept {
		}
		NIHILUS_HOST dim_updater_impl& operator=(const dim_updater_impl&) noexcept = delete;
		NIHILUS_HOST dim_updater_impl(const dim_updater_impl&) noexcept			   = delete;
		NIHILUS_HOST dim_updater_impl& operator=(dim_updater_impl&&) noexcept	   = delete;
		NIHILUS_HOST dim_updater_impl(dim_updater_impl&&) noexcept				   = delete;
		using base_type															   = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return runtime_dims_t<base_type>;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, uint64_t runtime_dimension, uint64_t& total_required_bytes) {
			parse_core.get_mutable_dim()	   = runtime_dimension;
			parse_core.total_required_bytes_rt = type_traits<typename base_type::output_type>::total_byte_size(parse_core.get_array_rt());
			total_required_bytes += parse_core.total_required_bytes_rt;
		}
	};

	template<typename config_type, typename base_type_new> struct dim_updater {
		NIHILUS_HOST dim_updater() noexcept {
		}
		NIHILUS_HOST dim_updater& operator=(const dim_updater&) noexcept = delete;
		NIHILUS_HOST dim_updater(const dim_updater&) noexcept			 = delete;
		NIHILUS_HOST dim_updater& operator=(dim_updater&&) noexcept		 = delete;
		NIHILUS_HOST dim_updater(dim_updater&&) noexcept				 = delete;
		using base_type													 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return has_total_required_bytes_types<base_type>;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, uint64_t runtime_dimension) {
			uint64_t total_required_bytes{};
			parse_core.values.template impl<dim_updater_impl>(runtime_dimension, total_required_bytes);
			parse_core.total_required_bytes_rt = total_required_bytes;
		}
	};

	template<typename config_type, typename base_type_new> struct weight_mapper_impl {
		NIHILUS_HOST weight_mapper_impl() noexcept {
		}
		NIHILUS_HOST weight_mapper_impl& operator=(const weight_mapper_impl&) noexcept = delete;
		NIHILUS_HOST weight_mapper_impl(const weight_mapper_impl&) noexcept			   = delete;
		NIHILUS_HOST weight_mapper_impl& operator=(weight_mapper_impl&&) noexcept	   = delete;
		NIHILUS_HOST weight_mapper_impl(weight_mapper_impl&&) noexcept				   = delete;
		using base_type																   = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return std::is_same_v<typename base_type::enum_type, weight_types>;
		}

		NIHILUS_HOST static void impl(base_type& core_traits, array<array<void*, model_traits_type<config_type>::block_count>, weight_types::count>& data) {
			if constexpr (base_type::data_strategy_type == data_strategy_types::per_block) {
				for (uint64_t x = 0; x < model_traits_type<config_type>::block_count; ++x) {
					data[base_type::enum_value][x] = static_cast<void*>(core_traits.get_data_ptr(x));
				}
			} else {
				data[base_type::enum_value][0] = static_cast<void*>(core_traits.get_data_ptr());
			}
		}
	};

	template<typename config_type, typename core_traits_type> struct weight_mapper {
		NIHILUS_HOST static void impl(core_traits_type& core_traits, array<array<void*, model_traits_type<config_type>::block_count>, weight_types::count>& data) {
			core_traits.values.template impl<weight_mapper_impl>(data);
		}
	};

	template<typename config_type, typename base_type_new, processing_phases processing_phase> struct global_input_thread_function {
		NIHILUS_HOST global_input_thread_function() noexcept {
		}
		NIHILUS_HOST global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_HOST global_input_thread_function(const global_input_thread_function&) noexcept			   = delete;
		NIHILUS_HOST global_input_thread_function& operator=(global_input_thread_function&&) noexcept	   = delete;
		NIHILUS_HOST global_input_thread_function(global_input_thread_function&&) noexcept				   = delete;
		using base_type																					   = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::core_type == core_types::token_embeddings;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, int64_t thread_count) {
			if constexpr (config_type::dev && config_type::device_type != device_types::gpu) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << " [STARTING] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
			kernel_dispatcher<config_type, processing_phase, base_type>::impl(parse_core, thread_count);

			if constexpr (config_type::dev && config_type::device_type != device_types::gpu) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << " [FINISHED] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
		}
	};

	template<typename config_type, typename base_type_new, processing_phases processing_phase> struct per_block_thread_function {
		NIHILUS_HOST per_block_thread_function() noexcept {
		}
		NIHILUS_HOST per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_HOST per_block_thread_function(const per_block_thread_function&) noexcept			 = delete;
		NIHILUS_HOST per_block_thread_function& operator=(per_block_thread_function&&) noexcept		 = delete;
		NIHILUS_HOST per_block_thread_function(per_block_thread_function&&) noexcept				 = delete;
		using base_type																				 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::core_type != core_types::weights && base_type::core_type != core_types::global_inputs &&
				base_type::core_type != core_types::final_norm_and_sampling && base_type::core_type != core_types::token_embeddings;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, int64_t current_block, int64_t thread_count) {
			if constexpr (config_type::dev && config_type::device_type != device_types::gpu) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << " [STARTING] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << ", for [BLOCK]: " << current_block << std::endl;
				log<log_levels::status>(stream.str());
			}
			kernel_dispatcher<config_type, processing_phase, base_type>::impl(parse_core, thread_count, current_block);
			if constexpr (config_type::dev && config_type::device_type != device_types::gpu) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << " [FINISHED] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << ", for [BLOCK]: " << current_block << std::endl;
				log<log_levels::status>(stream.str());
			}
		}
	};

	template<typename config_type, typename base_type_new, processing_phases processing_phase> struct global_output_thread_function {
		NIHILUS_HOST global_output_thread_function() noexcept {
		}
		NIHILUS_HOST global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_HOST global_output_thread_function(const global_output_thread_function&) noexcept			 = delete;
		NIHILUS_HOST global_output_thread_function& operator=(global_output_thread_function&&) noexcept		 = delete;
		NIHILUS_HOST global_output_thread_function(global_output_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::core_type == core_types::final_norm_and_sampling;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, int64_t thread_count) {
			if constexpr (config_type::dev && config_type::device_type != device_types::gpu) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << " [STARTING] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
			kernel_dispatcher<config_type, processing_phase, base_type>::impl(parse_core, thread_count);
			if constexpr (config_type::dev && config_type::device_type != device_types::gpu) {
				std::stringstream stream{};
				stream << "[DEBUG] Thread (ID: " << std::this_thread::get_id() << ") " << " [FINISHED] a barrier with " << thread_count
					   << " expected threads, for Op: " << base_type::core_type << std::endl;
				log<log_levels::status>(stream.str());
			}
		}
	};

	template<gpu_device_types config_type, typename base_type_new, processing_phases processing_phase>
	struct global_input_thread_function<config_type, base_type_new, processing_phase> {
		NIHILUS_HOST global_input_thread_function() noexcept {
		}
		NIHILUS_HOST global_input_thread_function& operator=(const global_input_thread_function&) noexcept = delete;
		NIHILUS_HOST global_input_thread_function(const global_input_thread_function&) noexcept			   = delete;
		NIHILUS_HOST global_input_thread_function& operator=(global_input_thread_function&&) noexcept	   = delete;
		NIHILUS_HOST global_input_thread_function(global_input_thread_function&&) noexcept				   = delete;
		using base_type																					   = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::core_type == core_types::token_embeddings;
		}

		NIHILUS_HOST static void impl(base_type& parse_core) {
			kernel_dispatcher<config_type, processing_phase, base_type>::impl(parse_core);
		}
	};

	template<gpu_device_types config_type, typename base_type_new, processing_phases processing_phase>
	struct per_block_thread_function<config_type, base_type_new, processing_phase> {
		NIHILUS_HOST per_block_thread_function() noexcept {
		}
		NIHILUS_HOST per_block_thread_function& operator=(const per_block_thread_function&) noexcept = delete;
		NIHILUS_HOST per_block_thread_function(const per_block_thread_function&) noexcept			 = delete;
		NIHILUS_HOST per_block_thread_function& operator=(per_block_thread_function&&) noexcept		 = delete;
		NIHILUS_HOST per_block_thread_function(per_block_thread_function&&) noexcept				 = delete;
		using base_type																				 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::core_type != core_types::weights && base_type::core_type != core_types::global_inputs &&
				base_type::core_type != core_types::final_norm_and_sampling && base_type::core_type != core_types::token_embeddings;
		}

		NIHILUS_HOST static void impl(base_type& parse_core, int64_t current_block) {
			kernel_dispatcher<config_type, processing_phase, base_type>::impl(parse_core, current_block);
		}
	};

	template<gpu_device_types config_type, typename base_type_new, processing_phases processing_phase>
	struct global_output_thread_function<config_type, base_type_new, processing_phase> {
		NIHILUS_HOST global_output_thread_function() noexcept {
		}
		NIHILUS_HOST global_output_thread_function& operator=(const global_output_thread_function&) noexcept = delete;
		NIHILUS_HOST global_output_thread_function(const global_output_thread_function&) noexcept			 = delete;
		NIHILUS_HOST global_output_thread_function& operator=(global_output_thread_function&&) noexcept		 = delete;
		NIHILUS_HOST global_output_thread_function(global_output_thread_function&&) noexcept				 = delete;
		using base_type																						 = base_type_new;
		NIHILUS_HOST static constexpr bool filter() {
			return base_type::core_type == core_types::final_norm_and_sampling;
		}

		NIHILUS_HOST static void impl(base_type& parse_core) {
			kernel_dispatcher<config_type, processing_phase, base_type>::impl(parse_core);
		}
	};

}
