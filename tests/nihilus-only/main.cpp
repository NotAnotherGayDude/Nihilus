#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <BnchSwt/BenchmarkSuite.hpp>
#include <nihilus/index.hpp>


template<typename value_type> extern const value_type external;

/**
	 * @brief Struct to remove member pointers.
	 *
	 * Define a struct to remove member pointers from the given type.
	 *
	 * @tparam value_type The type from which to remove member pointers.
	 */
template<typename member_type> struct remove_member_pointer {
	using type = member_type;
};

template<typename class_type, typename member_type> struct remove_member_pointer<member_type class_type::*> {
	using type = class_type;
};

template<typename value_type> using remove_member_pointer_t = typename remove_member_pointer<value_type>::type;

template<typename member_type> struct remove_class_pointer {
	using type = member_type;
};

template<typename class_type, typename member_type> struct remove_class_pointer<member_type class_type::*> {
	using type = member_type;
};

template<typename value_type> using remove_class_pointer_t = typename remove_class_pointer<value_type>::type;

#if defined(NIHILUS_COMPILER_CLANG)
constexpr auto pretty_function_tail = "]";
#elif defined(NIHILUS_COMPILER_GNUCXX)
constexpr auto pretty_function_tail = ";";
#endif

// Sampled from Stephen Berry and his library, Glaze library: https://github.com/stephenberry/glaze
/**
	 * @brief Get the name of a member pointer.
	 *
	 * Function to extract the name of a member pointer.
	 *
	 * @tparam p The member pointer.
	 * @return The name of the member pointer.
	 */
#if defined(NIHILUS_COMPILER_MSVC) && !defined(NIHILUS_COMPILER_CLANG)
template<typename value_type, auto p> static consteval std::string_view getNameImpl() noexcept {
	std::string_view str = std::source_location::current().function_name();
	str					 = str.substr(str.find("->") + 2);
	return str.substr(0, str.find(">"));
}
#else
template<auto p> consteval std::string_view getNameImpl() noexcept {
	std::string_view str = std::source_location::current().function_name();
	str					 = str.substr(str.find("&") + 1);
	str					 = str.substr(0, str.find(pretty_function_tail));
	return str.substr(str.rfind("::") + 2);
}
#endif

template<auto p>
	requires(std::is_member_pointer_v<decltype(p)>)
inline static constexpr auto getName() noexcept {
#if defined(NIHILUS_COMPILER_MSVC) && !defined(NIHILUS_COMPILER_CLANG)
	using value_type		 = remove_member_pointer_t<decltype(p)>;
	constexpr auto pNew		 = p;
	constexpr auto newString = getNameImpl<value_type, &(external<value_type>.*pNew)>();
#else
	constexpr auto newString = getNameImpl<p>();
#endif
	return newString;
}

template<typename value_type> struct parse_core {};

template<typename value_type> struct base_parse_entity {
	using class_type = value_type;
	inline static constexpr size_t index{ 0 };
};

template<auto memberPtrNew, nihilus::string_literal nameNew> struct parse_entity_temp {
	using member_type = remove_class_pointer_t<decltype(memberPtrNew)>;
	using class_type  = remove_member_pointer_t<decltype(memberPtrNew)>;
	inline static constexpr member_type class_type::* memberPtr{ memberPtrNew };
	inline static constexpr nihilus::string_literal name{ nameNew };
};

template<auto memberPtrNew, nihilus::string_literal nameNew, size_t indexNew, size_t maxIndex> struct parse_entity {
	using member_type = remove_class_pointer_t<decltype(memberPtrNew)>;
	using class_type  = remove_member_pointer_t<decltype(memberPtrNew)>;
	inline static constexpr member_type class_type::* memberPtr{ memberPtrNew };
	inline static constexpr bool isItLast{ indexNew == maxIndex - 1 };
	inline static constexpr nihilus::string_literal name{ nameNew };
	inline static constexpr size_t index{ indexNew };
};

template<typename value_type>
concept is_base_parse_entity = requires { typename value_type::member_type; } && !std::is_member_pointer_v<value_type>;

template<typename value_type>
concept is_parse_entity_temp = requires {
	typename value_type::class_type;
	value_type::memberPtr;
} && is_base_parse_entity<value_type>;

template<size_t maxIndex, size_t index, auto value> inline static constexpr auto makeJsonEntityAuto() noexcept {
	if constexpr (is_parse_entity_temp<decltype(value)>) {
		constexpr parse_entity<value.memberPtr, value.name, index, maxIndex> parseEntity{};
		return parseEntity;
	} else {
		constexpr auto nameNew = getName<value>();
		constexpr parse_entity<value, nihilus::stringLiteralFromView<nameNew.size()>(nameNew), index, maxIndex> parseEntity{};
		return parseEntity;
	}
}

template<typename value_type>
concept convertible_to_parse_entity = is_parse_entity_temp<value_type> || std::is_member_pointer_v<value_type>;

template<auto... values, size_t... indices> inline static constexpr auto createValueImpl(std::index_sequence<indices...>) {
	static_assert((convertible_to_parse_entity<decltype(values)> && ...), "All arguments passed to createValue must be convertible to a parse_entity.");
	return nihilus::makeTuple(makeJsonEntityAuto<sizeof...(values), indices, values>()...);
}
template<auto memberPtr, nihilus::string_literal nameNew> inline static constexpr auto makeJsonEntity() {
	return parse_entity_temp<memberPtr, nameNew>{};
}

template<auto memberPtr> inline static constexpr auto makeJsonEntity() {
	return parse_entity_temp<memberPtr, getName<memberPtr>()>{};
}

template<auto... values> inline static constexpr auto createValue() noexcept {
	return createValueImpl<values...>(std::make_index_sequence<sizeof...(values)>{});
}

struct tuple_reference {
	uint8_t oldIndex{};
	std::string_view key{};
};

struct tuple_references {
	const tuple_reference* rootPtr{};
	size_t count{};
};

template<typename value_type>
concept has_name = requires(std::remove_cvref_t<value_type> value) { value.name; };

template<size_t maxIndex, size_t currentIndex = 0, typename tuple_type>
static constexpr auto collectTupleRefsImpl(const tuple_type& tuple, nihilus::array<tuple_reference, maxIndex> tupleRefsRaw) {
	if constexpr (currentIndex < maxIndex) {
		auto potentialKey = nihilus::get<currentIndex>(tuple);
		if constexpr (has_name<decltype(potentialKey)>) {
			tupleRefsRaw[currentIndex].key = potentialKey.name.operator std::string_view();
		}
		tupleRefsRaw[currentIndex].oldIndex = currentIndex;
		return collectTupleRefsImpl<maxIndex, currentIndex + 1>(tuple, tupleRefsRaw);
	}
	return tupleRefsRaw;
}

template<typename tuple_type> static constexpr auto collectTupleRefs(const tuple_type& tuple) {
	constexpr auto tupleSize = nihilus::tuple_size_v<tuple_type>;
	nihilus::array<tuple_reference, tupleSize> tupleRefsRaw{};
	return collectTupleRefsImpl<tupleSize>(tuple, tupleRefsRaw);
}

template<size_t size> static constexpr auto consolidateTupleRefs(const nihilus::array<tuple_reference, size>& tupleRefsRaw) {
	tuple_references returnValues{};
	if constexpr (size > 0) {
		returnValues.rootPtr = &tupleRefsRaw[0];
		returnValues.count	 = size;
	}
	return returnValues;
}

template<typename value_type> struct core;

template<typename value_type> inline constexpr auto tupleRefs{ collectTupleRefs(core<value_type>::parse_value) };
template<typename value_type> inline constexpr auto tupleReferences{ consolidateTupleRefs(tupleRefs<value_type>) };

struct key_stats_t {
	size_t minLength{ (std::numeric_limits<size_t>::max)() };
	size_t uniqueIndex{};
	size_t maxLength{};
};

inline static constexpr size_t findUniqueColumnIndex(const tuple_references& tupleRefsRaw, size_t maxIndex, size_t startingIndex = 0) noexcept {
	constexpr size_t alphabetSize = 256;
	std::string_view key{};
	for (size_t index = startingIndex; index < maxIndex; ++index) {
		nihilus::array<bool, alphabetSize> seen{};
		bool allDifferent = true;

		for (size_t x = 0; x < tupleRefsRaw.count; ++x) {
			key				  = tupleRefsRaw.rootPtr[x].key;
			const char c	  = key[index];
			uint8_t charIndex = static_cast<uint8_t>(c);

			if (seen[charIndex]) {
				allDifferent = false;
				break;
			}
			seen[charIndex] = true;
		}

		if (allDifferent) {
			return index;
		}
	}

	return std::numeric_limits<size_t>::max();
}

inline static constexpr auto keyStatsImpl(const tuple_references& tupleRefsRaw) noexcept {
	key_stats_t stats{};
	for (size_t x = 0; x < tupleRefsRaw.count; ++x) {
		const std::string_view& key{ tupleRefsRaw.rootPtr[x].key };
		auto num{ key.size() };
		if (num > stats.maxLength) {
			stats.maxLength = num;
		}
		if (num < stats.minLength) {
			stats.minLength = num;
		}
	}
	stats.uniqueIndex = findUniqueColumnIndex(tupleRefsRaw, stats.minLength);
	return stats;
}

struct test_struct {
	int32_t test_val{};
};

template<typename value_type> inline static constexpr auto keyStatsVal = keyStatsImpl(tupleReferences<value_type>);

template<> struct core<test_struct> {
	using value_type				  = test_struct;
	static constexpr auto parse_value = createValue<&test_struct::test_val>();
};

inline constexpr nihilus::array<uint64_t, 135> prns{ { 1033321092324544984ull, 2666561049963377653ull, 3901177690447069239ull, 4218182233242110882ull, 5911765535454950103ull,
	6788651254494793497ull, 7100864855074445223ull, 8121427956336305945ull, 9038010914689427860ull, 14840306302415334885ull, 2861875790078914964ull, 3162274379479658823ull,
	4716213344225307449ull, 540950270129450019ull, 6138393194460717092ull, 7344427311844191385ull, 8475133706542525636ull, 9707373313909664576ull, 13125261184447140558ull,
	2935828130652229499ull, 3352961464321085856ull, 4654333323360932970ull, 5071886467123008198ull, 6337413869067417456ull, 7068363609472928302ull, 8706829452892616150ull,
	9383326841165471636ull, 16102866716245881820ull, 2811014628691939071ull, 3268225168635854144ull, 4143407405368768949ull, 5597712091605167573ull, 6100647393685909969ull,
	7810560643861675820ull, 8193567265249468576ull, 9274898615585908930ull, 1186958974127274710ull, 246203706832441443ull, 3668316000003001120ull, 4918933721064431133ull,
	5627034507966762943ull, 6439181573813589114ull, 7007274452014357082ull, 8727797399712164036ull, 9543719554692367837ull, 17847026727775507706ull, 2455135551339952688ull,
	3249111793759010315ull, 4777639692643446085ull, 5509261895474102266ull, 6044529605818700585ull, 7171416927005376731ull, 8039758273674712696ull, 9590025961231307183ull,
	11664492738409977550ull, 2284380607188886295ull, 3813446608469001272ull, 4331825983983119949ull, 5837226587704917004ull, 6635783511790253542ull, 705765947185415012ull,
	8161069307156878177ull, 9569010482089894937ull, 10396430003774290631ull, 2470420036324932164ull, 3824517089721070566ull, 4514057289782578484ull, 5632633334704453746ull,
	6925174598726443993ull, 7516137935779736019ull, 824571755531910559ull, 9361703638678697870ull, 13796235445108347584ull, 2481146946454909733ull, 3823008181066037442ull,
	4754601782272553608ull, 588747081408207180ull, 6322894155156329605ull, 7737051281621502357ull, 8728044800884920985ull, 9923282424466541678ull, 18161647536849824815ull,
	2607456623799892498ull, 3651449820230355939ull, 4624058760756378704ull, 564341449426358799ull, 6732980169420780216ull, 7082954320121844082ull, 8326246156222992233ull,
	9417642353078551282ull, 17539099562248686315ull, 226802774388233589ull, 3258991441457498839ull, 4386515027804287469ull, 5492870100834679754ull, 6249105792560430415ull,
	7289628920991893817ull, 8241072433031030544ull, 9727644451441173921ull, 13586305903807621608ull, 200516020872926547ull, 3730616597292952024ull, 4256645544584917949ull,
	5613544969337462956ull, 6264647669092059269ull, 7960331042009250727ull, 8582958636583556090ull, 9171663339272942914ull, 1619903645133495717ull, 2840616349619109328ull,
	3166096472286566799ull, 4494229804275778550ull, 5497884137148100871ull, 6572487097017879223ull, 738706937289335047ull, 8122825727823277447ull, 9131968469543030694ull,
	14054393997887833558ull, 2874030832643593377ull, 3673904271267944876ull, 4542812785880908260ull, 5621946313585701459ull, 6176632143181793702ull, 7512972502278041818ull,
	8724494295438506783ull, 9277533619161797917ull, 13495127262014153477ull, 2883303557104387784ull, 3039599040070277986ull, 4196273005435491662ull, 5417879022829474871ull,
	6476778602757520149ull, 7959620869796075525ull, 8518936512742009562ull, 9635246566869230345ull } };

template<typename value_type> static constexpr value_type readBitsCt(const char* ptr) noexcept {
	value_type chunk{};
	for (uint64_t x = 0; x < sizeof(value_type); ++x) {
		chunk |= static_cast<value_type>(static_cast<uint8_t>(ptr[x])) << (x * 8);
	}
	return chunk;
}

struct ct_key_hasher {
	uint64_t seed{};///< seed value for the hashing algorithm.
	uint64_t index{};
	/**
		 * @brief Default constructor that initializes the seed using a random_num value.
		 */
	constexpr ct_key_hasher() noexcept {
		updateSeed();
	}

	/**
		 * @brief Sets the seed value at compile-time.
		 *
		 * @param seedNew The new seed value.
		 */
	constexpr void updateSeed() noexcept {
		seed = prns[index];
		++index;
	}

	/**
		 * @brief Hashes a key at compile-time.
		 *
		 * @param value The value to be hashed.
		 * @param length The length of the value.
		 * @return The hashed value.
		 */
	constexpr uint64_t hashKeyCt(const char* value, uint64_t length) const noexcept {
		uint64_t seed64{ seed };
		while (length >= 8) {
			seed64 ^= readBitsCt<uint64_t>(value);
			value += 8;
			length -= 8;
		}

		if (length >= 4) {
			seed64 ^= readBitsCt<uint32_t>(value);
			value += 4;
			length -= 4;
		}

		if (length >= 2) {
			seed64 ^= readBitsCt<uint16_t>(value);
			value += 2;
			length -= 2;
		}

		if (length == 1) {
			seed64 ^= *value;
		}
		return seed64;
	}
};

template<uint64_t seedNew> struct rt_key_hasher {
	inline static constexpr auto seed{ seedNew };
	/**
		 * @brief Hashes a key at runtime.
		 *
		 * @param value The value to be hashed.
		 * @param length The length of the value.
		 * @return The hashed value.
		 */
	NIHILUS_FORCE_INLINE uint64_t hashKeyRt(const char* value, uint64_t length) const noexcept {
		uint64_t seed64{ seed };
		while (length >= 8) {
			std::memcpy(&chunk64, value, 8);
			seed64 ^= chunk64;
			value += 8;
			length -= 8;
		}

		if (length >= 4) {
			std::memcpy(&chunk32, value, 4);
			seed64 ^= chunk32;
			value += 4;
			length -= 4;
		}

		if (length >= 2) {
			std::memcpy(&chunk16, value, 2);
			seed64 ^= chunk16;
			value += 2;
			length -= 2;
		}

		if (length == 1) {
			seed64 ^= *value;
		}
		return seed64;
	}

  protected:
	mutable uint64_t chunk64{};
	mutable uint32_t chunk32{};
	mutable uint16_t chunk16{};
};

template<typename value_type01, typename value_type02> inline static constexpr bool contains(const value_type01* hashData, value_type02 byteToCheckFor, size_t size) noexcept {
	for (size_t x = 0; x < size; ++x) {
		if (hashData[x] == byteToCheckFor) {
			return true;
		}
	}
	return false;
}

enum class hash_map_type {
	unset						= 0,
	empty						= 1,
	single_element				= 2,
	double_element				= 3,
	triple_element				= 4,
	single_byte					= 5,
	first_byte_and_unique_index = 6,
	unique_byte_and_length		= 7,
	unique_per_length			= 8,
	simd_full_length			= 9,
};

enum class avx_type { m128 = 0, m256 = 1, m512 = 2 };

template<avx_type type> struct avx_type_wrapper;

template<> struct avx_type_wrapper<avx_type::m128> {
	using type = nihilus::nihilus_simd_int_128;
};

template<> struct avx_type_wrapper<avx_type::m256> {
	using type = nihilus::nihilus_simd_int_256;
};

template<> struct avx_type_wrapper<avx_type::m512> {
	using type = nihilus::nihilus_simd_int_512;
};

template<size_t length> struct map_simd {
	using type = std::conditional_t<length >= 64 && cpu_alignment >= 64, avx_type_wrapper<avx_type::m512>,
		std::conditional_t<length >= 32 && cpu_alignment >= 32, avx_type_wrapper<avx_type::m256>, avx_type_wrapper<avx_type::m128>>>;
};

template<size_t length> using map_simd_t = typename map_simd<length>::type::type;

inline static constexpr size_t setSimdWidth(size_t length) noexcept {
	return length >= 64ull && cpu_alignment >= 64ull ? 64ull : length >= 32ull && cpu_alignment >= 32ull ? 32ull : 16ull;
}

struct hash_map_construction_data {
	using simd_type = map_simd_t<2048>;
	nihilus::array<uint16_t, 2048 / setSimdWidth(2048)> bucketSizes{};
	NIHILUS_ALIGN(cpu_alignment) nihilus::array<uint8_t, 2049> controlBytes {};
	nihilus::array<uint8_t, 256> uniqueIndices{};
	nihilus::array<uint16_t, 2049> indices{};
	size_t bucketSize{ setSimdWidth(2048) };
	size_t numGroups{ 2048 / bucketSize };
	ct_key_hasher hasher{};
	hash_map_type type{};
	size_t uniqueIndex{};
	char firstChar{};

	constexpr hash_map_construction_data() noexcept = default;
};
struct single_byte_data {
	inline static constexpr size_t storageSize{ 256 };
	constexpr single_byte_data(const hash_map_construction_data& newData) noexcept
		: uniqueIndices{ newData.uniqueIndices }, uniqueIndex{ newData.uniqueIndex }, type{ newData.type } {};
	nihilus::array<uint8_t, 256> uniqueIndices{};
	size_t uniqueIndex{};
	hash_map_type type{};
};

struct simd_full_length_data {
	inline static constexpr size_t storageSize{ 2048 };
	constexpr simd_full_length_data(const hash_map_construction_data& newData) noexcept
		: bucketSize{ newData.bucketSize }, numGroups{ newData.numGroups }, uniqueIndex{ newData.uniqueIndex }, seed{ newData.hasher.seed } {
		std::copy(newData.controlBytes.data(), newData.controlBytes.data() + newData.controlBytes.size(), controlBytes.data());
		std::copy(newData.indices.data(), newData.indices.data() + newData.indices.size(), indices.data());
	};
	NIHILUS_ALIGN(cpu_alignment) nihilus::array<uint8_t, storageSize + 1> controlBytes {};
	char padding01[cpu_alignment - ((storageSize + 1) % 8)]{};
	size_t bucketSize{ setSimdWidth(storageSize) };
	size_t numGroups{ storageSize / bucketSize };
	nihilus::array<uint16_t, storageSize + 1> indices{};
	size_t uniqueIndex{};
	size_t seed{};
};

template<typename value_type> inline static constexpr auto collectSimdFullLengthHashMapData(const tuple_references& pairsNew) noexcept {
	hash_map_construction_data returnValues{};
	bool collided{};
	for (size_t w = keyStatsVal<value_type>.minLength; w < keyStatsVal<value_type>.maxLength; ++w) {
		returnValues.uniqueIndex = w;
		for (size_t x = 0; x < 2; ++x) {
			returnValues.controlBytes.fill(std::numeric_limits<uint8_t>::max());
			returnValues.indices.fill(static_cast<uint16_t>(returnValues.indices.size() - 1));
			returnValues.hasher.updateSeed();
			collided = false;
			for (size_t y = 0; y < pairsNew.count; ++y) {
				const auto keyLength	 = returnValues.uniqueIndex > pairsNew.rootPtr[y].key.size() ? pairsNew.rootPtr[y].key.size() : returnValues.uniqueIndex;
				const auto hash			 = returnValues.hasher.hashKeyCt(pairsNew.rootPtr[y].key.data(), keyLength);
				const auto groupPos		 = (hash >> 8) % returnValues.numGroups;
				const auto ctrlByte		 = static_cast<uint8_t>(hash);
				const auto bucketSizeNew = returnValues.bucketSizes[groupPos]++;
				const auto slot			 = ((groupPos * returnValues.bucketSize) + bucketSizeNew);

				if (bucketSizeNew >= returnValues.bucketSize || returnValues.indices[slot] != returnValues.indices.size() - 1 ||
					contains(returnValues.controlBytes.data() + groupPos * returnValues.bucketSize, ctrlByte, returnValues.bucketSize)) {
					returnValues.bucketSizes.fill(0);
					collided = true;
					break;
				}
				returnValues.controlBytes[slot] = ctrlByte;
				returnValues.indices[slot]		= pairsNew.rootPtr[y].oldIndex;
			}
			if (!collided) {
				break;
			}
		}
		if (!collided) {
			break;
		}
	}
	if (collided) {
		returnValues.uniqueIndex = std::numeric_limits<size_t>::max();
		return returnValues;
	} else {
		return returnValues;
	}
}

template<typename value_type> inline static constexpr auto collectSingleByteHashMapData(const tuple_references& pairsNew) noexcept {
	hash_map_construction_data returnValues{};
	returnValues.uniqueIndex = keyStatsVal<value_type>.uniqueIndex;
	if (returnValues.uniqueIndex != std::numeric_limits<size_t>::max()) {
		returnValues.uniqueIndices.fill(static_cast<uint8_t>(returnValues.uniqueIndices.size() - 1));
		for (size_t x = 0; x < pairsNew.count; ++x) {
			auto& newRef					 = pairsNew.rootPtr[pairsNew.rootPtr[x].oldIndex];
			const auto slot					 = static_cast<uint8_t>(newRef.key.data()[returnValues.uniqueIndex]);
			returnValues.uniqueIndices[slot] = static_cast<uint8_t>(newRef.oldIndex);
		}
		returnValues.type = hash_map_type::single_byte;
		return returnValues;
	} else {
		return collectSimdFullLengthHashMapData<value_type>(pairsNew);
	}
}

	template<typename value_type> inline static constexpr auto collectMapConstructionDataImpl() noexcept {
	return collectSingleByteHashMapData<value_type>(tupleReferences<value_type>);
}

template<typename value_type> inline static constexpr auto collectMapConstructionData() noexcept {
	constexpr auto constructionData = collectMapConstructionDataImpl<value_type>();
	if constexpr (constructionData.type == hash_map_type::single_byte) {
		return single_byte_data{ constructionData };
	} else if constexpr (constructionData.type == hash_map_type::simd_full_length) {
		return simd_full_length_data{ constructionData };
	} else {
		static_assert(constructionData.type != hash_map_type::unset, "Failed to construct that hashmap!");
	}
}

template<nihilus::model_config config, typename derived_type> struct core<nihilus::vocab<config, config.vocab_type, derived_type>> {
	using value_type				  = nihilus::vocab<config, config.vocab_type, derived_type>;
	static constexpr auto parse_value = createValue<&test_struct::test_val>();
};

template<typename value_type> inline static constexpr auto hashData = collectMapConstructionData<value_type>();
//static constexpr auto parse_value = createValue<&test_struct::test_val>();

int main(int argc, char** argv) {
	try {
		static constexpr auto config_v1_v2_1b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_fp16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_fp16_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_fp16_moe	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_fp16_moe = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_bf16_mha	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_bf16_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_bf16_gqa	 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_bf16_gqa = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q4_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q4_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q4_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q4_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q4_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q4_gqa = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q4_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q4_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q4_moe = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q8_mha   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q8_mha  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q8_mha = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q8_gqa   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q8_gqa  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q8_gqa = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_q8_moe   = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_q8_moe  = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_q8_moe = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_mixed_fp16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_mixed_fp16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);

		static constexpr auto config_v1_v2_1b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_3b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_7b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_8b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_11b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_13b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_70b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_90b_mixed_bf16_fp32	= nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v1_v2_405b_mixed_bf16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v1_v2, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_fp16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_fp16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_fp16_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_fp16_moe	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_fp16_moe  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_fp16_moe = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::fp16_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_bf16_mha	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_bf16_mha  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_bf16_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_bf16_gqa	  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_bf16_gqa  = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_bf16_gqa = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::bf16_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q4_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q4_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q4_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q4_gqa = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q4_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q4_moe = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q4_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q8_mha	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q8_mha = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_mha, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q8_gqa	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q8_gqa = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_gqa, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_q8_moe	= nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_q8_moe = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::q8_moe, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_mixed_fp16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_mixed_fp16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_fp16_fp32, nihilus::model_arches::llama, false);

		static constexpr auto config_v3_1b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_1B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_3b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_3B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_7b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_7B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_8b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B,
			  nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_11b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_11B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_13b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_13B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_70b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_70B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_90b_mixed_bf16_fp32	 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_90B,
			 nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);
		static constexpr auto config_v3_405b_mixed_bf16_fp32 = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_405B,
			nihilus::kernel_type_profiles::mixed_bf16_fp32, nihilus::model_arches::llama, false);

		nihilus::stop_watch stop_watch_val{ 0 };
		static constexpr auto model_config = nihilus::generate_model_config(nihilus::model_generations::v3, nihilus::model_sizes::llama_8B, nihilus::kernel_type_profiles::q8_gqa,
			nihilus::model_arches::llama, false);
		static constexpr auto model_config_new = nihilus::update_model_config_vocab_pre_type(model_config, nihilus::vocab_pre_types::llama3);
		nihilus::cli_params cli_args_final;
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::runBenchmark<"nihilus">([&] {
			cli_args_final = { nihilus::harbinger<model_config_new>::parse_cli_arguments(argc, argv) };
			auto model_new{ nihilus::harbinger<model_config_new>::parse_model_graph_data(cli_args_final) };
			hashData<nihilus::vocab<model_config_new, nihilus::vocab_types::bpe, decltype(model_new)>>;
			while (model_new->process_input(cli_args_final.prompt)) {
			}
			return cli_args_final.n_tokens;
		});
		bnch_swt::benchmark_stage<"nihilus-vs_llama.cpp", 4, 2, true, "Token">::printResults();
	} catch (const std::exception& error) {
		std::cout << "Error: " << error.what() << std::endl;
	}
	return 0;
}