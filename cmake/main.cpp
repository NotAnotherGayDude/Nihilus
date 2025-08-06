// Sampled mostly from Simdjson: https://github.com/simdjson/simdjson
#if defined(NIHILUS_DETECT_ARCH)
	#include <iostream>
	#include <cstring>
	#include <cstdint>
	#include <cstdlib>
	#if defined(_MSC_VER)
		#include <intrin.h>
	#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
		#include <cpuid.h>
	#endif

enum class instruction_set {
	FALLBACK = 0x0,
	AVX2	 = 0x1,
	AVX512f	 = 0x2,
	NEON	 = 0x4,
	SVE2	 = 0x8,
};

namespace {
	static constexpr uint32_t cpuid_avx2_bit	 = 1ul << 5;
	static constexpr uint32_t cpuid_avx512_bit	 = 1ul << 16;
	static constexpr uint64_t cpuid_avx256_saved = 1ull << 2;
	static constexpr uint64_t cpuid_avx512_saved = 7ull << 5;
	static constexpr uint32_t cpuid_osx_save	 = (1ul << 26) | (1ul << 27);
}

	#if defined(__x86_64__) || defined(_M_AMD64)
inline static void cpuid(uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx);
inline static uint64_t xgetbv();
	#endif

	#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
		#if defined(__linux__)
			#include <sys/auxv.h>
			#include <asm/hwcap.h>
		#elif defined(__APPLE__)
			#include <sys/sysctl.h>
		#endif

inline static uint32_t detect_supported_architectures() {
	uint32_t host_isa = static_cast<uint32_t>(instruction_set::NEON);

		#if defined(__linux__)
	unsigned long hwcap = getauxval(AT_HWCAP);
	if (hwcap & HWCAP_SVE) {
		host_isa |= static_cast<uint32_t>(instruction_set::SVE2);
		std::cout << "ARM SVE detected\n";
	}
		#elif defined(__APPLE__)
	std::cout << "Apple ARM64 - NEON baseline\n";
		#endif

	return host_isa;
}

	#elif defined(__x86_64__) || defined(_M_AMD64)
inline static void cpuid(uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
		#if defined(_MSC_VER)
	int32_t cpu_info[4];
	__cpuidex(cpu_info, *eax, *ecx);
	*eax = cpu_info[0];
	*ebx = cpu_info[1];
	*ecx = cpu_info[2];
	*edx = cpu_info[3];
		#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	uint32_t level = *eax;
	__get_cpuid(level, eax, ebx, ecx, edx);
		#else
	uint32_t a = *eax, b, c = *ecx, d;
	asm volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(a), "c"(c));
	*eax = a;
	*ebx = b;
	*ecx = c;
	*edx = d;
		#endif
}

inline static uint64_t xgetbv() {
		#if defined(_MSC_VER)
	return _xgetbv(0);
		#else
	uint32_t eax, edx;
	asm volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
	return (( uint64_t )edx << 32) | eax;
		#endif
}

inline static uint32_t detect_supported_architectures() {
	std::uint32_t eax	   = 0;
	std::uint32_t ebx	   = 0;
	std::uint32_t ecx	   = 0;
	std::uint32_t edx	   = 0;
	std::uint32_t host_isa = static_cast<uint32_t>(instruction_set::FALLBACK);

	eax = 0x1;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if ((ecx & cpuid_osx_save) != cpuid_osx_save) {
		return host_isa;
	}

	uint64_t xcr0 = xgetbv();
	if ((xcr0 & cpuid_avx256_saved) == 0) {
		return host_isa;
	}

	eax = 0x7;
	ecx = 0x0;
	cpuid(&eax, &ebx, &ecx, &edx);

	if (ebx & cpuid_avx2_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX2);
	}

	if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
		return host_isa;
	}

	if (ebx & cpuid_avx512_bit) {
		host_isa |= static_cast<uint32_t>(instruction_set::AVX512f);
	}

	return host_isa;
}

	#else
inline static uint32_t detect_supported_architectures() {
	return static_cast<uint32_t>(instruction_set::FALLBACK);
}
	#endif

int32_t main() {
	const auto supported_isa = detect_supported_architectures();
	return supported_isa;
}
#else
	#include <thread>
int32_t main() {
	return std::thread::hardware_concurrency();
}
#endif