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
#include <optional>
#include <iomanip>
#include <cstdint>
#include <span>

namespace nihilus::benchmarking {

	struct performance_metrics {
		double throughputPercentageDeviation{ std::numeric_limits<double>::max() };
		std::optional<double> cacheReferencesPerExecution{};
		std::optional<double> instructionsPerExecution{};
		std::optional<double> branchMissesPerExecution{};
		std::optional<double> cacheMissesPerExecution{};
		std::optional<uint64_t> totalIterationCount{};
		std::optional<double> instructionsPerCycle{};
		std::optional<double> branchesPerExecution{};
		std::optional<double> instructionsPerByte{};
		std::optional<double> cyclesPerExecution{};
		std::optional<double> cyclesPerByte{};
		std::optional<double> frequencyGHz{};
		double throughputMbPerSec{};
		uint64_t bytesProcessed{};
		double timeInNs{};

		NIHILUS_INLINE bool operator>(const performance_metrics& other) const {
			return throughputMbPerSec > other.throughputMbPerSec;
		}
	};
}

namespace nihilus::benchmarking::internal {

	NIHILUS_INLINE double calculateThroughputMBps(double nanoseconds, double bytesProcessed) {
		constexpr double bytesPerMB		= 1024.0 * 1024.0;
		constexpr double nanosPerSecond = 1e9;
		double megabytes = bytesProcessed / bytesPerMB;
		double seconds	 = nanoseconds / nanosPerSecond;
		if (seconds == 0.0) {
			return 0.0;
		}
		return megabytes / seconds;
	}

	NIHILUS_INLINE double calculateUnitsPs(double nanoseconds, double bytesProcessed) {
		return (bytesProcessed * 1000000000.0) / nanoseconds;
	}

	NIHILUS_INLINE static performance_metrics collectMetrics(std::span<event_count>&& eventsNewer, size_t totalIterationCount) {
		performance_metrics metrics{};
		metrics.totalIterationCount.emplace(totalIterationCount);
		double throughPut{};
		double throughPutTotal{};
		double throughPutAvg{};
		double throughPutMin{ std::numeric_limits<double>::max() };
		uint64_t bytesProcessed{};
		uint64_t bytesProcessedTotal{};
		uint64_t bytesProcessedAvg{};
		double ns{};
		double nsTotal{};
		double nsAvg{};
		double cycles{};
		double cyclesTotal{};
		double cyclesAvg{};
		double instructions{};
		double instructionsTotal{};
		double instructionsAvg{};
		double branches{};
		double branchesTotal{};
		double branchesAvg{};
		double branchMisses{};
		double branchMissesTotal{};
		double branchMissesAvg{};
		double cacheReferences{};
		double cacheReferencesTotal{};
		double cacheReferencesAvg{};
		double cacheMisses{};
		double cacheMissesTotal{};
		double cacheMissesAvg{};
		for (const event_count& e: eventsNewer) {
			ns = e.elapsedNs();
			nsTotal += ns;

			if (e.bytesProcessed(bytesProcessed)) {
				bytesProcessedTotal += bytesProcessed;
				throughPut = calculateThroughputMBps(ns, static_cast<double>(bytesProcessed));
				throughPutTotal += throughPut;
				throughPutMin = throughPut < throughPutMin ? throughPut : throughPutMin;
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

			if (e.branchMisses(branchMisses)) {
				branchMissesTotal += branchMisses;
			}

			if (e.cacheReferences(cacheReferences)) {
				cacheReferencesTotal += cacheReferences;
			}

			if (e.cacheMisses(cacheMisses)) {
				cacheMissesTotal += cacheMisses;
			}
		}
		if (eventsNewer.size() > 0) {
			bytesProcessedAvg  = bytesProcessedTotal / eventsNewer.size();
			nsAvg			   = nsTotal / static_cast<double>(eventsNewer.size());
			throughPutAvg	   = throughPutTotal / static_cast<double>(eventsNewer.size());
			cyclesAvg		   = cyclesTotal / static_cast<double>(eventsNewer.size());
			instructionsAvg	   = instructionsTotal / static_cast<double>(eventsNewer.size());
			branchesAvg		   = branchesTotal / static_cast<double>(eventsNewer.size());
			branchMissesAvg	   = branchMissesTotal / static_cast<double>(eventsNewer.size());
			cacheReferencesAvg = cacheReferencesTotal / static_cast<double>(eventsNewer.size());
			cacheMissesAvg	   = cacheMissesTotal / static_cast<double>(eventsNewer.size());
			metrics.timeInNs   = nsAvg;
		} else {
			return {};
		}

		constexpr double epsilon = 1e-6;
		if (std::abs(nsAvg) > epsilon) {
			metrics.bytesProcessed				  = bytesProcessedAvg;
			metrics.throughputMbPerSec			  = throughPutAvg;
			metrics.throughputPercentageDeviation = ((throughPutAvg - throughPutMin) * 100.0) / throughPutAvg;
		}
		if (std::abs(cyclesAvg) > epsilon) {
			if (metrics.bytesProcessed > 0) {
				metrics.cyclesPerByte.emplace(cyclesAvg / static_cast<double>(metrics.bytesProcessed));
			}
			metrics.cyclesPerExecution.emplace(cyclesTotal / static_cast<double>(eventsNewer.size()));
			metrics.frequencyGHz.emplace(cyclesAvg / nsAvg);
		}
		if (std::abs(instructionsAvg) > epsilon) {
			if (metrics.bytesProcessed > 0) {
				metrics.instructionsPerByte.emplace(instructionsAvg / static_cast<double>(metrics.bytesProcessed));
			}
			if (std::abs(cyclesAvg) > epsilon) {
				metrics.instructionsPerCycle.emplace(instructionsAvg / cyclesAvg);
			}
			metrics.instructionsPerExecution.emplace(instructionsTotal / static_cast<double>(eventsNewer.size()));
		}
		if (std::abs(branchesAvg) > epsilon) {
			metrics.branchMissesPerExecution.emplace(branchMissesAvg / static_cast<double>(eventsNewer.size()));
			metrics.branchesPerExecution.emplace(branchesAvg / static_cast<double>(eventsNewer.size()));
		}
		if (std::abs(cacheMissesAvg) > epsilon) {
			metrics.cacheMissesPerExecution.emplace(cacheMissesAvg / static_cast<double>(eventsNewer.size()));
		}
		if (std::abs(cacheReferencesAvg) > epsilon) {
			metrics.cacheReferencesPerExecution.emplace(cacheReferencesAvg / static_cast<double>(eventsNewer.size()));
		}
		return metrics;
	}

	template<typename metrics_type>
	concept optional_t = requires(std::remove_cvref_t<metrics_type> opt) {
		{ opt.has_value() } -> std::same_as<bool>;
		{ opt.value() } -> std::same_as<typename std::remove_cvref_t<metrics_type>::value_type&>;
		{ *opt } -> std::same_as<typename std::remove_cvref_t<metrics_type>::value_type&>;
		{ opt.reset() } -> std::same_as<void>;
		{ opt.emplace(typename std::remove_cvref_t<metrics_type>::value_type{}) } -> std::same_as<typename std::remove_cvref_t<metrics_type>::value_type&>;
	};

	NIHILUS_INLINE static std::string printResults(const performance_metrics& metrics, uint64_t thread, std::string_view kernel_name, bool showComparison = true,
		bool showMetrics = true) {
		std::stringstream stream{};
		stream << "Performance Metrics: " << std::endl;
		if (showMetrics) {
			stream << "Metrics for: " << kernel_name << ", For Thread #" << std::to_string(thread) << std::endl;
			stream << std::fixed << std::setprecision(2);

			static constexpr auto printMetric = []<typename metrics_type>(const std::string_view& label, const metrics_type& metricsNew, std::stringstream& stream) {
				if constexpr (optional_t<metrics_type>) {
					if (metricsNew.has_value()) {
						stream << std::left << std::setw(60ull) << label << ": " << metricsNew.value() << std::endl;
					}
				} else {
					stream << std::left << std::setw(60ull) << label << ": " << metricsNew << std::endl;
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
			printMetric(metricName, metrics.bytesProcessed, stream);
			printMetric("Nanoseconds per Execution", metrics.timeInNs, stream);
			printMetric("Frequency (GHz)", metrics.frequencyGHz, stream);
			printMetric(throughPutString, metrics.throughputMbPerSec, stream);
			printMetric("Cycles per Execution", metrics.cyclesPerExecution, stream);
			printMetric(cycleCount, metrics.cyclesPerByte, stream);
			printMetric("Instructions per Execution", metrics.instructionsPerExecution, stream);
			printMetric("Instructions per Cycle", metrics.instructionsPerCycle, stream);
			printMetric(instructionCount, metrics.instructionsPerByte, stream);
			printMetric("Branches per Execution", metrics.branchesPerExecution, stream);
			printMetric("Branch Misses per Execution", metrics.branchMissesPerExecution, stream);
			printMetric("Cache References per Execution", metrics.cacheReferencesPerExecution,stream);
			printMetric("Cache Misses per Execution", metrics.cacheMissesPerExecution, stream);
			stream << "----------------------------------------" << std::endl;
			return stream.str();
		}
	}

}
