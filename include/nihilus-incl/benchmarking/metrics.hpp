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

#include <nihilus-incl/common/string_literal.hpp>
#include <nihilus-incl/benchmarking/event_count.hpp>
#include <nihilus-incl/common/aligned_vector.hpp>
#include <optional>
#include <iomanip>
#include <cstdint>
#include <span>

namespace nihilus::benchmarking {

	struct performance_metrics {
		double throughput_percentage_deviation{ std::numeric_limits<double>::max() };
		std::optional<double> cache_references_per_execution{};
		std::optional<double> instructions_per_execution{};
		std::optional<double> branch_misses_per_execution{};
		std::optional<double> cache_misses_per_execution{};
		std::optional<uint64_t> totalIterationCount{};
		std::optional<double> instructions_per_cycle{};
		std::optional<double> branchesPerExecution{};
		std::optional<double> instructions_per_byte{};
		std::optional<double> cyclesPerExecution{};
		std::optional<double> cycles_per_byte{};
		std::optional<double> frequencyGHz{};
		double throughput_mb_per_sec{};
		uint64_t byes_processed{};
		double timeInNs{};

		NIHILUS_HOST bool operator>(const performance_metrics& other) const {
			return throughput_mb_per_sec > other.throughput_mb_per_sec;
		}
	};

	NIHILUS_HOST static double calculate_throughput_mbps(double nanoseconds, double byes_processed) {
		constexpr double bytes_per_mb	= 1024.0 * 1024.0;
		constexpr double nanosPerSecond = 1e9;
		double megabytes				= byes_processed / bytes_per_mb;
		double seconds					= nanoseconds / nanosPerSecond;
		if (seconds == 0.0) {
			return 0.0;
		}
		return megabytes / seconds;
	}

	NIHILUS_HOST static double calculate_unitsps(double nanoseconds, double byes_processed) {
		return (byes_processed * 1000000000.0) / nanoseconds;
	}

	NIHILUS_HOST static performance_metrics collect_metrics(std::span<event_count>&& eventsNewer, uint64_t totalIterationCount) {
		performance_metrics metrics{};
		metrics.totalIterationCount.emplace(totalIterationCount);
		double throughput{};
		double throughput_total{};
		double through_put_avg{};
		double throughPutMin{ std::numeric_limits<double>::max() };
		uint64_t byes_processed{};
		uint64_t byes_processedTotal{};
		uint64_t byes_processedAvg{};
		double ns{};
		double nsTotal{};
		double nsAvg{};
		double cycles{};
		double cyclesTotal{};
		double cycles_avg{};
		double instructions{};
		double instructionsTotal{};
		double instructions_avg{};
		double branches{};
		double branchesTotal{};
		double branchesAvg{};
		double branch_misses{};
		double branch_missesTotal{};
		double branch_missesAvg{};
		double cache_references{};
		double cache_referencesTotal{};
		double cache_references_avg{};
		double cache_misses{};
		double cache_missesTotal{};
		double cache_misses_avg{};
		for (const event_count& e: eventsNewer) {
			ns = e.elapsedNs();
			nsTotal += ns;

			if (e.byes_processed(byes_processed)) {
				byes_processedTotal += byes_processed;
				throughput = calculate_throughput_mbps(ns, static_cast<double>(byes_processed));
				throughput_total += throughput;
				throughPutMin = throughput < throughPutMin ? throughput : throughPutMin;
			}

			if (e.cycles(cycles)) {
				cyclesTotal += cycles;
			}

			if (e.instructions(instructions)) {
				instructionsTotal += instructions;
			}

			if (e.branches(branches)) {
				branchesTotal += branches;
			}

			if (e.branch_misses(branch_misses)) {
				branch_missesTotal += branch_misses;
			}

			if (e.cache_references(cache_references)) {
				cache_referencesTotal += cache_references;
			}

			if (e.cache_misses(cache_misses)) {
				cache_missesTotal += cache_misses;
			}
		}
		if (eventsNewer.size() > 0) {
			byes_processedAvg	 = byes_processedTotal / eventsNewer.size();
			nsAvg				 = nsTotal / static_cast<double>(eventsNewer.size());
			through_put_avg		 = throughput_total / static_cast<double>(eventsNewer.size());
			cycles_avg			 = cyclesTotal / static_cast<double>(eventsNewer.size());
			instructions_avg	 = instructionsTotal / static_cast<double>(eventsNewer.size());
			branchesAvg			 = branchesTotal / static_cast<double>(eventsNewer.size());
			branch_missesAvg	 = branch_missesTotal / static_cast<double>(eventsNewer.size());
			cache_references_avg = cache_referencesTotal / static_cast<double>(eventsNewer.size());
			cache_misses_avg	 = cache_missesTotal / static_cast<double>(eventsNewer.size());
			metrics.timeInNs	 = nsAvg;
		} else {
			return {};
		}

		constexpr double epsilon = 1e-6;
		if (std::abs(nsAvg) > epsilon) {
			metrics.byes_processed					= byes_processedAvg;
			metrics.throughput_mb_per_sec			= through_put_avg;
			metrics.throughput_percentage_deviation = ((through_put_avg - throughPutMin) * 100.0) / through_put_avg;
		}
		if (std::abs(cycles_avg) > epsilon) {
			if (metrics.byes_processed > 0) {
				metrics.cycles_per_byte.emplace(cycles_avg / static_cast<double>(metrics.byes_processed));
			}
			metrics.cyclesPerExecution.emplace(cyclesTotal / static_cast<double>(eventsNewer.size()));
			metrics.frequencyGHz.emplace(cycles_avg / nsAvg);
		}
		if (std::abs(instructions_avg) > epsilon) {
			if (metrics.byes_processed > 0) {
				metrics.instructions_per_byte.emplace(instructions_avg / static_cast<double>(metrics.byes_processed));
			}
			if (std::abs(cycles_avg) > epsilon) {
				metrics.instructions_per_cycle.emplace(instructions_avg / cycles_avg);
			}
			metrics.instructions_per_execution.emplace(instructionsTotal / static_cast<double>(eventsNewer.size()));
		}
		if (std::abs(branchesAvg) > epsilon) {
			metrics.branch_misses_per_execution.emplace(branch_missesAvg / static_cast<double>(eventsNewer.size()));
			metrics.branchesPerExecution.emplace(branchesAvg / static_cast<double>(eventsNewer.size()));
		}
		if (std::abs(cache_misses_avg) > epsilon) {
			metrics.cache_misses_per_execution.emplace(cache_misses_avg / static_cast<double>(eventsNewer.size()));
		}
		if (std::abs(cache_references_avg) > epsilon) {
			metrics.cache_references_per_execution.emplace(cache_references_avg / static_cast<double>(eventsNewer.size()));
		}
		return metrics;
	}

	template<typename metrics_type>
	concept optional_t = requires(detail::remove_cvref_t<metrics_type> opt) {
		{ opt.has_value() } -> detail::same_as<bool>;
		{ opt.value() } -> detail::same_as<typename detail::remove_cvref_t<metrics_type>::value_type&>;
		{ *opt } -> detail::same_as<typename detail::remove_cvref_t<metrics_type>::value_type&>;
		{ opt.reset() } -> detail::same_as<void>;
		{ opt.emplace(typename detail::remove_cvref_t<metrics_type>::value_type{}) } -> detail::same_as<typename detail::remove_cvref_t<metrics_type>::value_type&>;
	};

	NIHILUS_HOST static std::string printResults(const performance_metrics& metrics, uint64_t thread, const std::string_view kernel_name) {
		std::stringstream stream{};
		stream << "Performance Metrics: " << std::endl;
		stream << "Metrics for: " << kernel_name << ", For Thread #" << std::to_string(thread) << std::endl;
		stream << std::fixed << std::setprecision(2);

		static constexpr auto printMetric = []<typename metrics_type>(const std::string_view label, const metrics_type& metrics_new, std::stringstream& stream_new) {
			if constexpr (optional_t<metrics_type>) {
				if (metrics_new.has_value()) {
					stream_new << std::left << std::setw(60ull) << label << ": " << metrics_new.value() << std::endl;
				}
			} else {
				stream_new << std::left << std::setw(60ull) << label << ": " << metrics_new << std::endl;
			}
		};
		std::string instructionCount{};
		std::string throughPutString{};
		std::string cycleCount{};
		std::string metricName{};
		throughPutString = "Throughput (MB/s)";
		metricName		 = "Bytes Processed";
		cycleCount		 = "Cycles per Byte";
		instructionCount = "Instructions per Byte";
		printMetric("Iterations", metrics.totalIterationCount, stream);
		printMetric(metricName, metrics.byes_processed, stream);
		printMetric("Nanoseconds per Execution", metrics.timeInNs, stream);
		printMetric("Frequency (GHz)", metrics.frequencyGHz, stream);
		printMetric(throughPutString, metrics.throughput_mb_per_sec, stream);
		printMetric("Cycles per Execution", metrics.cyclesPerExecution, stream);
		printMetric(cycleCount, metrics.cycles_per_byte, stream);
		printMetric("Instructions per Execution", metrics.instructions_per_execution, stream);
		printMetric("Instructions per Cycle", metrics.instructions_per_cycle, stream);
		printMetric(instructionCount, metrics.instructions_per_byte, stream);
		printMetric("Branches per Execution", metrics.branchesPerExecution, stream);
		printMetric("Branch Misses per Execution", metrics.branch_misses_per_execution, stream);
		printMetric("Cache References per Execution", metrics.cache_references_per_execution, stream);
		printMetric("Cache Misses per Execution", metrics.cache_misses_per_execution, stream);
		stream << "----------------------------------------" << std::endl;
		return stream.str();
	}

}
