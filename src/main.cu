#include <nihilus>

static constexpr auto model_config_00 = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_8B, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::device_types::gpu, false, 8192);
static constexpr auto model_config_01 = model_config_00.update_benchmark(true);
static constexpr auto model_config_02 = nihilus::generate_model_config(nihilus::model_generations::v3_1, nihilus::model_sizes::llm_405B, nihilus::kernel_type_profiles::q8_gqa,
	nihilus::model_arches::llama, nihilus::device_types::gpu, false, 8192);

template<typename value_type_01, typename value_type_02>
concept same_or_convertible_to = std::is_same_v<std::remove_cvref_t<value_type_01>, std::remove_cvref_t<value_type_02>> ||
	std::convertible_to<std::remove_cvref_t<value_type_02>, std::remove_cvref_t<value_type_01>>;

template<typename T, same_or_convertible_to<T> U> constexpr auto nv_std_max(const T& a, const U& b) {
	return (b > a) ? b : a;
}

#if defined(__CUDA_ARCH__) && !defined(__INTELLISENSE__)
	#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
		#define NIHILUS_PRAGMA_UNROLL _Pragma("unroll")
		#define NIHILUS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
	#else
		#define NIHILUS_PRAGMA_UNROLL #pragma unroll
		#define NIHILUS_PRAGMA_NO_UNROLL #pragma unroll 1
	#endif
	#define NIHILUS_GEMM_LOOP NIHILUS_PRAGMA_NO_UNROLL

#else

	#define NIHILUS_PRAGMA_UNROLL
	#define NIHILUS_PRAGMA_NO_UNROLL
	#define NIHILUS_GEMM_LOOP

#endif

template<auto index> using tag = std::integral_constant<uint64_t, static_cast<uint64_t>(index)>;

struct OpMultiplyAdd {};

struct OpMultiplyAddSaturate {};

struct OpMultiplyAddFastBF16 {};

struct OpMultiplyAddFastF16 {};

struct OpMultiplyAddMixedInputUpcast {};

struct OpMultiplyAddFastF32 {};

struct OpMultiplyAddComplexFastF32 {};

struct OpMultiplyAddFastAccum;

struct OpMultiplyAddComplex {};

struct OpMultiplyAddGaussianComplex {};

struct OpXorPopc {};

struct OpAndPopc {};

struct OpClassSimt {};

struct OpClassTensorOp {};

struct OpClassWmmaTensorOp {};

struct OpClassSparseTensorOp {};

struct OpClassBlockScaledTensorOp {};

struct OpClassBlockScaledSparseTensorOp {};

template<uint64_t Rank_,///< Logical rank of coordinate
	typename Index_		= uint64_t,///< Index type used for each dimension
	typename LongIndex_ = uint64_t///< Long index type used for linear offsets
	>
struct Coord {
  public:
	//
	// Type and constant definitions
	//

	/// Number of elements in Coord
	static uint64_t const kRank = Rank_;

	/// Index type used to store elements
	using Index = Index_;

	/// Type used to represent linear offsets
	using LongIndex = LongIndex_;

  private:
	//
	// Data members
	//

	/// Indices
	Index idx[kRank];

  public:
	//
	// Methods
	//
	NIHILUS_HOST_DEVICE
	Coord() = default;
	/// Default ctor initializes uniformly
	NIHILUS_HOST_DEVICE
	Coord(Index value) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = value;
		}
	}

	NIHILUS_HOST_DEVICE
	explicit Coord(Index value_01, Index value_02) {
		idx[0] = value_01;
		idx[1] = value_02;
	}

	/// Constructs from an array of integers
	NIHILUS_HOST_DEVICE
	Coord(Index const (&_idx)[kRank]) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = _idx[i];
		}
	}

	/// Constructs from some other Coord
	template<uint64_t R, typename I, typename L> NIHILUS_HOST_DEVICE Coord(Coord<R, I, L> other) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = other[i];
		}
	}

	/// Returns a slice of the Coord which may be larger or smaller in rank
	/// than this.
	template<uint64_t Slice> NIHILUS_HOST_DEVICE Coord<Slice, Index, LongIndex> slice(uint64_t start = 0, Index identity = 0) const {
		Coord<Slice, Index, LongIndex> result;
		for (uint64_t i = 0; i < Slice; ++i) {
			if (i + start < kRank) {
				result[i] = idx[i + start];
			} else {
				result[i] = identity;
			}
		}
		return result;
	}

	/// Returns the index of the dimension with least value
	NIHILUS_HOST_DEVICE
	uint64_t min_dim_index() const {
		uint64_t i = 0;
		for (uint64_t j = 1; j < kRank; ++j) {
			if (idx[j] < idx[i]) {
				i = j;
			}
		}
		return i;
	}

	/// Returns the index of the dimension with greatest value
	NIHILUS_HOST_DEVICE
	uint64_t max_dim_index() const {
		uint64_t i = 0;
		for (uint64_t j = 1; j < kRank; ++j) {
			if (idx[j] > idx[i]) {
				i = j;
			}
		}
		return i;
	}

	/// Returns true if Coord is non-zero.
	NIHILUS_HOST_DEVICE
	explicit operator bool() const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (idx[i]) {
				return true;
			}
		}
		return false;
	}

	/// Returns true if Coord is uniformly zero.
	NIHILUS_HOST_DEVICE
	bool operator!() const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (idx[i]) {
				return false;
			}
		}
		return true;
	}

	/// Element-wise addition
	NIHILUS_HOST_DEVICE
	Coord operator+(Coord const& b) const {
		Coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] + b.idx[i];
		}
		return c;
	}

	/// Element-wise subtraction
	NIHILUS_HOST_DEVICE
	Coord operator-(Coord const& b) const {
		Coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] - b.idx[i];
		}
		return c;
	}

	/// Element-wise multiplication
	NIHILUS_HOST_DEVICE
	Coord operator*(Coord const& b) const {
		Coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] * b.idx[i];
		}
		return c;
	}

	/// Element-wise division
	NIHILUS_HOST_DEVICE
	Coord operator/(Coord const& b) const {
		Coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] / b.idx[i];
		}
		return c;
	}

	/// In-place addition
	NIHILUS_HOST_DEVICE
	Coord& operator+=(Coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] += b.idx[i];
		}
		return *this;
	}

	/// In-place subtraction
	NIHILUS_HOST_DEVICE
	Coord& operator-=(Coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] -= b.idx[i];
		}
		return *this;
	}

	/// In-place multiplication
	NIHILUS_HOST_DEVICE
	Coord& operator*=(Coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] *= b.idx[i];
		}
		return *this;
	}

	/// In-place division
	NIHILUS_HOST_DEVICE
	Coord& operator/=(Coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] /= b.idx[i];
		}
		return *this;
	}

	/// Member access operator
	NIHILUS_HOST_DEVICE Index& operator[](uint64_t dim) {
		return idx[dim];
	}

	/// Member access operator
	NIHILUS_HOST_DEVICE Index const& operator[](uint64_t dim) const {
		return idx[dim];
	}

	/// Computes the dot product with anotherCoord object
	NIHILUS_HOST_DEVICE
	LongIndex dot(Coord const& b, LongIndex sum = LongIndex(0)) const {
		for (uint64_t i = 0; i < kRank; ++i) {
			sum += idx[i] * b.idx[i];
		}
		return sum;
	}

	/// Gets the index of a given Coord element
	template<uint64_t Dim> NIHILUS_HOST_DEVICE Index& at() {
		return idx[Dim];
	}

	/// Access via index; may limit unrolling potential
	NIHILUS_HOST_DEVICE
	Index& at(uint64_t dim) {
		return idx[dim];
	}

	/// Gets the index of a given Coord element
	template<uint64_t Dim> NIHILUS_HOST_DEVICE Index const& at() const {
		return idx[Dim];
	}

	/// Access via index; may limit unrolling potential
	NIHILUS_HOST_DEVICE
	Index const& at(uint64_t dim) const {
		return idx[dim];
	}

	/// Determines if two Coord<> objects are equal
	NIHILUS_HOST_DEVICE
	bool operator==(Coord const& b) const {
		bool equal = true;
		for (uint64_t i = 0; equal && i < kRank; ++i) {
			equal = (idx[i] == b.idx[i]);
		}
		return equal;
	}

	/// Not equal
	NIHILUS_HOST_DEVICE
	bool operator!=(Coord const& b) const {
		return !(*this == b);
	}

	/// Clamps a coordinate to a range specified by maximum and minimum values
	NIHILUS_HOST_DEVICE
	Coord& clamp(Coord const& max, Coord const& min = Coord()) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = __NV_STD_MAX(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
		}
		return *this;
	}

	/// Returns the sum of all elements
	NIHILUS_HOST_DEVICE
	Index sum() const {
		Index sum_(idx[0]);
		for (uint64_t i = 1; i < kRank; ++i) {
			sum_ += idx[i];
		}
		return sum_;
	}

	/// Returns the product of all elements
	NIHILUS_HOST_DEVICE
	LongIndex product() const {
		LongIndex product_(idx[0]);
		for (uint64_t i = 1; i < kRank; ++i) {
			product_ *= idx[i];
		}
		return product_;
	}

	/// Less than operator
	NIHILUS_HOST_DEVICE
	bool operator<(Coord const& b) const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (!(idx[i] < b[i])) {
				return false;
			}
		}
		return true;
	}

	/// Less than or equals operator
	NIHILUS_HOST_DEVICE
	bool operator<=(Coord const& b) const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (!(idx[i] <= b[i])) {
				return false;
			}
		}
		return true;
	}

	/// Greater than operator
	NIHILUS_HOST_DEVICE
	bool operator>(Coord const& b) const {
		return !(*this <= b);
	}

	/// Greater than or equals operator
	NIHILUS_HOST_DEVICE
	bool operator>=(Coord const& b) const {
		return !(*this < b);
	}
};

template<int64_t Rank_, uint64_t...> struct constexpresh_coord;

template<int64_t Rank_, uint64_t... indices> struct constexpresh_coord {
  public:
	static constexpr uint64_t kRank = Rank_;

	using Index = uint64_t;

	using LongIndex = uint64_t;

	Index idx[kRank];

  public:
	NIHILUS_HOST_DEVICE constexpr constexpresh_coord(){};

	NIHILUS_HOST_DEVICE explicit constexpr constexpresh_coord(Index value_01) {
		for (uint64_t x = 0; x < kRank; ++x) {
			idx[x] = value_01;
		}
	}

	NIHILUS_HOST_DEVICE explicit constexpr constexpresh_coord(Index value_01, Index value_02, Index value_03) {
		idx[0] = value_01;
		idx[1] = value_02;
		idx[2] = value_03;
	}

	NIHILUS_HOST_DEVICE explicit constexpr constexpresh_coord(Index value_01, Index value_02, Index value_03, Index value_04) {
		idx[0] = value_01;
		idx[1] = value_02;
		idx[2] = value_03;
		idx[3] = value_04;
	}

	NIHILUS_HOST_DEVICE explicit constexpr constexpresh_coord(Index value_01, Index value_02, Index value_03, Index value_04, Index value_05) {
		idx[0] = value_01;
		idx[1] = value_02;
		idx[2] = value_03;
		idx[3] = value_04;
		idx[4] = value_05;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord(Index const (&_idx)[kRank]) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = _idx[i];
		}
	}

	template<uint64_t R> NIHILUS_HOST_DEVICE constexpresh_coord(constexpresh_coord<R> other) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = other[i];
		}
	}

	template<uint64_t Slice> NIHILUS_HOST_DEVICE constexpresh_coord<Slice> slice(uint64_t start = 0, Index identity = 0) const {
		constexpresh_coord<Slice> result;
		for (uint64_t i = 0; i < Slice; ++i) {
			if (i + start < kRank) {
				result[i] = idx[i + start];
			} else {
				result[i] = identity;
			}
		}
		return result;
	}

	NIHILUS_HOST_DEVICE uint64_t min_dim_index() const {
		uint64_t i = 0;
		for (uint64_t j = 1; j < kRank; ++j) {
			if (idx[j] < idx[i]) {
				i = j;
			}
		}
		return i;
	}

	NIHILUS_HOST_DEVICE uint64_t max_dim_index() const {
		uint64_t i = 0;
		for (uint64_t j = 1; j < kRank; ++j) {
			if (idx[j] > idx[i]) {
				i = j;
			}
		}
		return i;
	}

	NIHILUS_HOST_DEVICE explicit operator bool() const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (idx[i]) {
				return true;
			}
		}
		return false;
	}

	NIHILUS_HOST_DEVICE bool operator!() const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (idx[i]) {
				return false;
			}
		}
		return true;
	}

	NIHILUS_HOST_DEVICE constexpr Index m() const {
		return idx[0];
	}

	NIHILUS_HOST_DEVICE constexpr Index k() const {
		return idx[2];
	}

	NIHILUS_HOST_DEVICE Index const& n() const {
		return idx[1];
	}

	NIHILUS_HOST_DEVICE constexpresh_coord operator+(constexpresh_coord const& b) const {
		constexpresh_coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] + b.idx[i];
		}
		return c;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord operator-(constexpresh_coord const& b) const {
		constexpresh_coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] - b.idx[i];
		}
		return c;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord operator*(constexpresh_coord const& b) const {
		constexpresh_coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] * b.idx[i];
		}
		return c;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord operator/(constexpresh_coord const& b) const {
		constexpresh_coord c;
		for (uint64_t i = 0; i < kRank; ++i) {
			c.idx[i] = idx[i] / b.idx[i];
		}
		return c;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator+=(constexpresh_coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] += b.idx[i];
		}
		return *this;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator-=(constexpresh_coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] -= b.idx[i];
		}
		return *this;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator*=(constexpresh_coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] *= b.idx[i];
		}
		return *this;
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator/=(constexpresh_coord const& b) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] /= b.idx[i];
		}
		return *this;
	}

	NIHILUS_HOST_DEVICE Index& operator[](uint64_t dim) {
		return idx[dim];
	}

	NIHILUS_HOST_DEVICE Index const& operator[](uint64_t dim) const {
		return idx[dim];
	}

	NIHILUS_HOST_DEVICE LongIndex dot(constexpresh_coord const& b, LongIndex sum = LongIndex(0)) const {
		for (uint64_t i = 0; i < kRank; ++i) {
			sum += idx[i] * b.idx[i];
		}
		return sum;
	}

	template<uint64_t Dim> NIHILUS_HOST_DEVICE Index& at() {
		return idx[Dim];
	}

	NIHILUS_HOST_DEVICE Index& at(uint64_t dim) {
		return idx[dim];
	}

	template<uint64_t Dim> NIHILUS_HOST_DEVICE Index const& at() const {
		return idx[Dim];
	}

	NIHILUS_HOST_DEVICE Index const& at(uint64_t dim) const {
		return idx[dim];
	}

	NIHILUS_HOST_DEVICE bool operator==(constexpresh_coord const& b) const {
		bool equal = true;
		for (uint64_t i = 0; equal && i < kRank; ++i) {
			equal = (idx[i] == b.idx[i]);
		}
		return equal;
	}

	NIHILUS_HOST_DEVICE bool operator!=(constexpresh_coord const& b) const {
		return !(*this == b);
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& clamp(constexpresh_coord const& max, constexpresh_coord const& min = constexpresh_coord()) {
		for (uint64_t i = 0; i < kRank; ++i) {
			idx[i] = nv_std_max(__NV_STD_MIN(idx[i], max.idx[i]), min.idx[i]);
		}
		return *this;
	}

	NIHILUS_HOST_DEVICE Index sum() const {
		Index sum_(idx[0]);
		for (uint64_t i = 1; i < kRank; ++i) {
			sum_ += idx[i];
		}
		return sum_;
	}

	NIHILUS_HOST_DEVICE LongIndex product() const {
		LongIndex product_(idx[0]);
		for (uint64_t i = 1; i < kRank; ++i) {
			product_ *= idx[i];
		}
		return product_;
	}

	NIHILUS_HOST_DEVICE bool operator<(constexpresh_coord const& b) const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (!(idx[i] < b[i])) {
				return false;
			}
		}
		return true;
	}

	NIHILUS_HOST_DEVICE bool operator<=(constexpresh_coord const& b) const {
		for (uint64_t i = 0; i < kRank; ++i) {
			if (!(idx[i] <= b[i])) {
				return false;
			}
		}
		return true;
	}

	NIHILUS_HOST_DEVICE bool operator>(constexpresh_coord const& b) const {
		return !(*this <= b);
	}

	NIHILUS_HOST_DEVICE bool operator>=(constexpresh_coord const& b) const {
		return !(*this < b);
	}
};

template<uint64_t M_new, uint64_t K_new> struct constexpresh_coord<3, M_new, K_new> {
  public:
	static constexpr int64_t kRank = 3;

	static constexpr int64_t M = M_new;
	static constexpr int64_t K = K_new;

	using Index = uint64_t;

	mutable Index N;

  public:
	NIHILUS_HOST_DEVICE constexpr explicit constexpresh_coord(Index value = Index(0)) {
		N = value;
	}

	NIHILUS_HOST_DEVICE constexpr constexpresh_coord(const constexpresh_coord& other) {
		N = other.N;
	}

	NIHILUS_HOST_DEVICE int64_t min_dim_index() const {
		int64_t i = 0;
		uint64_t lowest{ M };
		if (K < lowest) {
			lowest = K;
			i	   = 1;
		}
		if (N < lowest) {
			lowest = N;
			i	   = 2;
		}
		return i;
	}

	NIHILUS_HOST_DEVICE int64_t max_dim_index() const {
		int64_t i = 0;
		uint64_t highest{ M };
		if (K > highest) {
			highest = K;
			i		= 1;
		}
		if (N > highest) {
			highest = N;
			i		= 2;
		}
		return i;
	}

	NIHILUS_HOST_DEVICE explicit operator bool() const {
		if (M) {
			return true;
		}
		if (K) {
			return true;
		}
		if (N) {
			return true;
		}
		return false;
	}

	NIHILUS_HOST_DEVICE bool operator!() const {
		if (M) {
			return false;
		}
		if (K) {
			return false;
		}
		if (N) {
			return false;
		}
		return true;
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr decltype(auto) operator+(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		constexpr uint64_t M_final = M + M_newer;
		constexpr uint64_t K_final = K + K_newer;
		return constexpresh_coord<3, M_final, K_final>{ N + b.N };
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr decltype(auto) operator-(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		constexpr uint64_t M_final = M - M_newer;
		constexpr uint64_t K_final = K - K_newer;
		return constexpresh_coord<3, M_final, K_final>{ N - b.N };
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr decltype(auto) operator*(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		constexpr uint64_t M_final = M * M_newer;
		constexpr uint64_t K_final = K * K_newer;
		return constexpresh_coord<3, M_final, K_final>{ N * b.N };
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr decltype(auto) operator/(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		constexpr uint64_t M_final = M / M_newer;
		constexpr uint64_t K_final = K / K_newer;
		return constexpresh_coord<3, M_final, K_final>{ N / b.N };
	}

	template<uint64_t index> NIHILUS_HOST_DEVICE Index operator[](tag<index> dim) const {
		if constexpr (index == 0) {
			return static_cast<Index>(M);
		}
		if constexpr (index == 1) {
			return N;
		}
		return static_cast<Index>(K);
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr Index dot(constexpresh_coord<3, M_newer, K_newer> const& b, Index sum = Index(0)) const {
		constexpr Index m_component = static_cast<Index>(M * M_newer);
		constexpr Index k_component = static_cast<Index>(K * K_newer);
		return sum + m_component + (N * b.N) + k_component;
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE bool operator==(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		if constexpr (M != M_newer) {
			return false;
		}
		if constexpr (K != K_newer) {
			return false;
		}
		return N == b.N;
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr bool operator!=(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		return !(*this == b);
	}

	template<uint64_t M_max, uint64_t K_max, uint64_t M_min = 0, uint64_t K_min = 0>
	NIHILUS_HOST_DEVICE constexpr decltype(auto) clamp(constexpresh_coord<3, M_max, K_max> const& max, constexpresh_coord<3, M_min, K_min> const& min = constexpresh_coord<3, M_min, K_min>{}) const {
		constexpr uint64_t M_clamped = (M < M_min) ? M_min : ((M > M_max) ? M_max : M);
		constexpr uint64_t K_clamped = (K < K_min) ? K_min : ((K > K_max) ? K_max : K);
		Index N_clamped				 = nv_std_max(__NV_STD_MIN(N, max.N), min.N);
		return constexpresh_coord<3, M_clamped, K_clamped>{ N_clamped };
	}

	NIHILUS_HOST_DEVICE constexpr Index sum() const {
		constexpr uint64_t MK{ M + K };
		return MK + N;
	}

	NIHILUS_HOST_DEVICE constexpr Index product() const {
		constexpr uint64_t MK{ M * K };
		return MK * N;
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr bool operator<(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		if constexpr (M >= M_newer)
			return false;
		if constexpr (K >= K_newer)
			return false;
		return N < b.N;
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr bool operator<=(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		if constexpr (M > M_newer)
			return false;
		if constexpr (K > K_newer)
			return false;
		return N <= b.N;
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr bool operator>(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		return !(*this <= b);
	}

	template<uint64_t M_newer, uint64_t K_newer> NIHILUS_HOST_DEVICE constexpr bool operator>=(constexpresh_coord<3, M_newer, K_newer> const& b) const {
		return !(*this < b);
	}
};

template<> struct constexpresh_coord<2> {
  public:
	static constexpr int64_t kRank = 2;

	using Index = uint64_t;

	mutable Index M;
	mutable Index N;

  public:
	NIHILUS_HOST_DEVICE explicit constexpresh_coord(Index M_new = Index(0), Index N_new = Index(0)) : M{ M_new }, N{ N_new } {
	}

	NIHILUS_HOST_DEVICE constexpresh_coord(const constexpresh_coord& other) : M{ other.M }, N{ other.N } {};

	NIHILUS_HOST_DEVICE int64_t min_dim_index() const {
		int64_t i = 0;
		uint64_t lowest{ M };
		if (N < lowest) {
			lowest = N;
			i	   = 1;
		}
		return i;
	}

	NIHILUS_HOST_DEVICE int64_t max_dim_index() const {
		int64_t i = 0;
		uint64_t highest{ M };
		if (N > highest) {
			highest = N;
			i		= 1;
		}
		return i;
	}

	NIHILUS_HOST_DEVICE Index const& at(uint64_t dim) const {
		if (dim == 0) {
			return M;
		} else {
			return N;
		}
	}

	NIHILUS_HOST_DEVICE explicit operator bool() const {
		if (M) {
			return true;
		}
		if (N) {
			return true;
		}
		return false;
	}

	NIHILUS_HOST_DEVICE bool operator!() const {
		if (M) {
			return false;
		}
		if (N) {
			return false;
		}
		return true;
	}

	NIHILUS_HOST_DEVICE Index& operator[](uint64_t dim) {
		if (dim == 0) {
			return M;
		} else {
			return N;
		}
	}

	NIHILUS_HOST_DEVICE Index const& operator[](uint64_t dim) const {
		if (dim == 0) {
			return M;
		} else {
			return N;
		}
	}

	NIHILUS_HOST_DEVICE decltype(auto) operator+(const constexpresh_coord<2>& b) const {
		uint64_t M_final = M + b.M;
		uint64_t N_final = N + b.N;
		return constexpresh_coord<2>{ M_final, N_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator+=(constexpresh_coord const& b) {
		M += b.M;
		N += b.N;
		return *this;
	}

	NIHILUS_HOST_DEVICE decltype(auto) operator-(const constexpresh_coord<2>& b) const {
		uint64_t M_final = M - b.M;
		uint64_t N_final = N - b.N;
		return constexpresh_coord<2>{ M_final, N_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator-=(constexpresh_coord const& b) {
		M -= b.M;
		N -= b.N;
		return *this;
	}

	NIHILUS_HOST_DEVICE decltype(auto) operator*(const constexpresh_coord<2>& b) const {
		uint64_t M_final = M * b.M;
		uint64_t N_final = N * b.N;
		return constexpresh_coord<2>{ M_final, N_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator*=(constexpresh_coord const& b) {
		M *= b.M;
		N *= b.N;
		return *this;
	}

	NIHILUS_HOST_DEVICE decltype(auto) operator/(const constexpresh_coord<2>& b) const {
		uint64_t M_final = M / b.M;
		uint64_t N_final = N / b.N;
		return constexpresh_coord<2>{ M_final, N_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator/=(constexpresh_coord const& b) {
		M /= b.M;
		N /= b.N;
		return *this;
	}
};

template<> struct constexpresh_coord<1> {
  public:
	static constexpr int64_t kRank = 1;

	using Index = uint64_t;

  public:
	mutable Index M;

  public:
	NIHILUS_HOST_DEVICE constexpr explicit constexpresh_coord(Index M_new = Index(0), Index N_new = Index(0)) : M{ M_new } {
	}

	NIHILUS_HOST_DEVICE constexpr constexpresh_coord(const constexpresh_coord& other) : M{ other.M } {};

	NIHILUS_HOST_DEVICE int64_t min_dim_index() const {
		int64_t i = 0;
		uint64_t lowest{ M };
		return i;
	}

	NIHILUS_HOST_DEVICE Index& operator[](uint64_t dim) {
		return M;
	}

	NIHILUS_HOST_DEVICE Index const& operator[](uint64_t dim) const {
		return M;
	}

	NIHILUS_HOST_DEVICE int64_t max_dim_index() const {
		int64_t i = 0;
		uint64_t highest{ M };
		return i;
	}

	NIHILUS_HOST_DEVICE Index const& at(uint64_t dim) const {
		return M;
	}

	NIHILUS_HOST_DEVICE explicit operator bool() const {
		if (M) {
			return true;
		}
		return false;
	}

	NIHILUS_HOST_DEVICE bool operator!() const {
		if (M) {
			return false;
		}
		return true;
	}

	NIHILUS_HOST_DEVICE constexpr decltype(auto) operator+(const constexpresh_coord<1>& b) const {
		uint64_t M_final = M + b.M;
		return constexpresh_coord<1>{ M_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator+=(constexpresh_coord const& b) {
		M += b.M;
		return *this;
	}

	NIHILUS_HOST_DEVICE constexpr decltype(auto) operator-(const constexpresh_coord<1>& b) const {
		uint64_t M_final = M - b.M;
		return constexpresh_coord<1>{ M_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator-=(constexpresh_coord const& b) {
		M -= b.M;
		return *this;
	}

	NIHILUS_HOST_DEVICE constexpr decltype(auto) operator*(const constexpresh_coord<1>& b) const {
		uint64_t M_final = M * b.M;
		return constexpresh_coord<1>{ M_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator*=(constexpresh_coord const& b) {
		M *= b.M;
		return *this;
	}

	NIHILUS_HOST_DEVICE constexpr decltype(auto) operator/(const constexpresh_coord<1>& b) const {
		uint64_t M_final = M / b.M;
		return constexpresh_coord<1>{ M_final };
	}

	NIHILUS_HOST_DEVICE constexpresh_coord& operator/=(constexpresh_coord const& b) {
		M /= b.M;
		return *this;
	}
};

template<uint64_t M_new, uint64_t K_new> NIHILUS_HOST_DEVICE constexpresh_coord<3, M_new, K_new> make_Coord(uint64_t N) {
	return constexpresh_coord<3, M_new, K_new>{ N };
}

template<uint64_t M = 1, uint64_t N = 1, uint64_t K = 1> struct GemmShape {
	static constexpr uint64_t kM = M;
	static constexpr uint64_t kN = N;
	static constexpr uint64_t kK = K;

	static constexpr uint64_t kMN  = M * N;
	static constexpr uint64_t kMK  = M * K;
	static constexpr uint64_t kKN  = N * K;
	static constexpr uint64_t kMNK = M * N * K;

	static constexpr uint64_t kCount = kMNK;

	NIHILUS_HOST_DEVICE static constexpresh_coord<3, M, K> toCoord() {
		return make_Coord<kM, kK>(kN);
	}
};

template<typename Shape> using GemmShapeTranspose = GemmShape<Shape::kN, Shape::kM, Shape::kK>;

template<typename value_type>
concept coord_type = requires() {
	std::remove_cvref_t<value_type>::M;
	std::remove_cvref_t<value_type>::K;
	std::remove_cvref_t<value_type>::N;
};

template<uint64_t M_new, uint64_t K_new> struct GemmCoord : public constexpresh_coord<3, M_new, K_new> {
	using Index = uint64_t;

	using Base = constexpresh_coord<3, M_new, K_new>;

	static constexpr uint64_t kM = 0;

	static constexpr uint64_t kN = 1;

	static constexpr uint64_t kK = 2;

	NIHILUS_HOST_DEVICE constexpr GemmCoord() {
	}

	NIHILUS_HOST_DEVICE constexpr GemmCoord(constexpresh_coord<3, M_new, K_new> const& coord) : Base(make_Coord<M_new, K_new>(coord.N)) {
	}

	NIHILUS_HOST_DEVICE constexpr GemmCoord(Index N) : Base(N) {
	}

	NIHILUS_HOST_DEVICE constexpr Index m() const {
		return Base::M;
	}

	NIHILUS_HOST_DEVICE constexpr Index k() const {
		return Base::K;
	}

	NIHILUS_HOST_DEVICE Index const& n() const {
		return Base::N;
	}

	template<coord_type value_type> NIHILUS_HOST_DEVICE decltype(auto) operator+(const value_type& b) const {
		return Base::operator+(b);
	}

	template<coord_type value_type> NIHILUS_HOST_DEVICE decltype(auto) operator-(const value_type& b) const {
		return Base::operator-(b);
	}

	template<coord_type value_type> NIHILUS_HOST_DEVICE decltype(auto) operator*(const value_type& b) const {
		return Base::operator*(b);
	}

	template<coord_type value_type> NIHILUS_HOST_DEVICE decltype(auto) operator/(const value_type& b) const {
		return Base::operator/(b);
	}
};

enum class FloatRoundStyle {
	round_indeterminate,
	round_toward_zero,
	round_to_nearest,
	round_to_nearest_satfinite,
	round_toward_infinity,
	round_toward_neg_infinity,
	round_half_ulp_truncate,
	round_half_ulp_trunc_dntz
};

template<typename T, typename S, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
	requires(std::is_same_v<T, S>)
struct NumericConverter {
	using result_type							 = T;
	using source_type							 = S;
	static constexpr FloatRoundStyle round_style = Round;

	NIHILUS_HOST_DEVICE static result_type impl(source_type const& s) {
		return s;
	}
};

template<typename T, typename S, FloatRoundStyle Round>
	requires(!std::is_same_v<T, S>)
struct NumericConverter<T, S, Round> {
	using result_type							 = T;
	using source_type							 = S;
	static constexpr FloatRoundStyle round_style = Round;

	NIHILUS_HOST_DEVICE static result_type impl(source_type const& s) {
		return static_cast<result_type>(s);
	}
};

namespace UnaryTransform {
	struct Identity;
	struct Conjugate;
}

template<typename T, typename S, uint64_t N, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest, typename Transform = UnaryTransform::Identity>
struct NumericArrayConverter {
	using result_type							 = nihilus::array<T, N>;
	using source_type							 = nihilus::array<S, N>;
	static constexpr FloatRoundStyle round_style = Round;

	NIHILUS_HOST_DEVICE static result_type impl(source_type const& s) {
		result_type result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			if constexpr (std::is_same<Transform, UnaryTransform::Identity>::value) {
				result[i] = NumericConverter<T, S, Round>::impl(s[i]);
			} else {
				result[i] = NumericConverter<T, S, Round>::impl(NumericConverter<T, S, Round>::impl(s[i]));
			}
		}

		return result;
	}
};

struct ScaleType {
	enum Kind { Default, NoBetaScaling, OnlyAlphaScaling, PerChannelScaling, OnlyAlphaPerChannelScaling, Nothing };
};

template<typename T> struct multiplies {
	NIHILUS_HOST_DEVICE static T impl(T lhs, T const& rhs) {
		lhs *= rhs;
		return lhs;
	}
};

template<typename T, uint64_t N> struct multiplies<nihilus::array<T, N>> {
	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& lhs, nihilus::array<T, N> const& rhs) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiplies<T>::impl(lhs[i], rhs[i]);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& lhs, T const& scalar) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiplies<T>::impl(lhs[i], scalar);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(T const& scalar, nihilus::array<T, N> const& rhs) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiplies<T>::impl(scalar, rhs[i]);
		}

		return result;
	}
};

template<typename A, typename B = A, typename C = A> struct multiply_add {
	NIHILUS_HOST_DEVICE static C impl(A const& a, B const& b, C const& c) {
		return C(a) * C(b) + c;
	}
};

template<typename T, uint64_t N> struct multiply_add<nihilus::array<T, N>, nihilus::array<T, N>, nihilus::array<T, N>> {
	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& a, nihilus::array<T, N> const& b, nihilus::array<T, N> const& c) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiply_add<T>::impl(a[i], b[i], c[i]);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& a, T const& scalar, nihilus::array<T, N> const& c) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiply_add<T>::impl(a[i], scalar, c[i]);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(T const& scalar, nihilus::array<T, N> const& b, nihilus::array<T, N> const& c) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiply_add<T>::impl(scalar, b[i], c[i]);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& a, nihilus::array<T, N> const& b, T const& scalar) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiply_add<T>::impl(a[i], b[i], scalar);
		}

		return result;
	}


	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& a, T const& scalar_b, T const& scalar_c) {
		nihilus::array<T, N> result;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = multiply_add<T>::impl(a[i], scalar_b, scalar_c);
		}

		return result;
	}
};

template<typename ElementOutput_, uint64_t Count, typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_,
	ScaleType::Kind Scale = ScaleType::Default, FloatRoundStyle Round = FloatRoundStyle::round_to_nearest, typename ElementSource_ = ElementOutput_, ElementCompute_ alpha = 1.0f,
	ElementCompute_ beta = 0.0f>
class LinearCombination {
  public:
	using ElementOutput		 = ElementOutput_;
	using ElementSource		 = ElementSource_;
	using ElementAccumulator = ElementAccumulator_;
	using ElementCompute	 = ElementCompute_;
	using ElementScalar		 = ElementCompute;
	using ElementC			 = ElementSource_;
	using ElementD			 = ElementOutput_;

	static constexpr uint64_t kCount		= Count;
	static constexpr ScaleType::Kind kScale = Scale;
	using FragmentOutput					= nihilus::array<ElementOutput, kCount>;
	using FragmentSource					= nihilus::array<ElementSource, kCount>;
	using FragmentAccumulator				= nihilus::array<ElementAccumulator, kCount>;
	using FragmentCompute					= nihilus::array<ElementCompute, kCount>;

	static constexpr FloatRoundStyle kRound = Round;

	struct Params {
		ElementCompute const* alpha_ptr;
		ElementCompute const* beta_ptr;
		ElementCompute const* const* alpha_ptr_array;
		ElementCompute const* const* beta_ptr_array;

		NIHILUS_HOST_DEVICE Params() : alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		NIHILUS_HOST_DEVICE Params(ElementCompute alpha, ElementCompute beta) : alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		NIHILUS_HOST_DEVICE Params(ElementCompute alpha) : alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		NIHILUS_HOST_DEVICE Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
			: alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		NIHILUS_HOST_DEVICE Params(ElementCompute const* alpha_ptr) : alpha_ptr(alpha_ptr), beta_ptr(nullptr), alpha_ptr_array(nullptr), beta_ptr_array(nullptr) {
		}

		NIHILUS_HOST_DEVICE Params(ElementCompute const* const* alpha_ptr_array, ElementCompute const* const* beta_ptr_array)
			: alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(alpha_ptr_array), beta_ptr_array(beta_ptr_array) {
		}

		NIHILUS_HOST_DEVICE Params(ElementCompute const* const* alpha_ptr_array)
			: alpha_ptr(nullptr), beta_ptr(nullptr), alpha_ptr_array(alpha_ptr_array), beta_ptr_array(nullptr) {
		}
	};

  public:
	NIHILUS_HOST_DEVICE explicit LinearCombination() = default;

	NIHILUS_HOST_DEVICE explicit LinearCombination(const Params& params) : LinearCombination(params, 0) {
	}

	NIHILUS_HOST_DEVICE constexpr bool is_source_needed() const {
		if constexpr (Scale == ScaleType::NoBetaScaling)
			return true;

		if constexpr (Scale == ScaleType::OnlyAlphaScaling)
			return false;

		if constexpr (Scale == ScaleType::Nothing)
			return false;

		if constexpr (beta == ElementCompute(0)) {
			return false;
		} else {
			return true;
		}
	}

	NIHILUS_HOST_DEVICE void set_k_partition(uint64_t k_partition, uint64_t k_partition_count) {
		if (k_partition) {
			beta = ElementCompute(1);
		}
	}

	NIHILUS_HOST_DEVICE FragmentOutput operator()(const FragmentAccumulator& accumulator, const FragmentSource& source) const {
		FragmentCompute converted_source	  = NumericArrayConverter<ElementCompute, ElementSource, kCount, Round>::impl(source);
		FragmentCompute converted_accumulator = NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>::impl(accumulator);

		if constexpr (Scale == ScaleType::Nothing)
			return NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>::impl(converted_accumulator);

		FragmentCompute intermediate{};

		if constexpr (Scale == ScaleType::NoBetaScaling) {
			intermediate = converted_source;
		} else {
			intermediate = multiplies<FragmentCompute>::impl(beta, converted_source);
		}

		intermediate = multiply_add<FragmentCompute>::impl(alpha, converted_accumulator, intermediate);

		return NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>::impl(intermediate);
	}

	NIHILUS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
		FragmentCompute converted_accumulator = NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>::impl(accumulator);

		if constexpr (Scale == ScaleType::Nothing) {
			return NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>::impl(converted_accumulator);
		}

		FragmentCompute intermediate{ multiplies<FragmentCompute>::impl(alpha, converted_accumulator) };

		return NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>::impl(intermediate);
	}
};

template<uint64_t Contiguous, uint64_t Strided> struct PitchLinearShape {
	static constexpr uint64_t kContiguous = Contiguous;
	static constexpr uint64_t kStrided	  = Strided;
	static constexpr uint64_t kCount	  = Contiguous * Strided;
};

struct PitchLinearCoord : public Coord<2> {
  public:
	using Index = uint64_t;

	using Base = Coord<2>;

	using LongIndex = uint64_t;

  public:
	static constexpr uint64_t kContiguous = 0;

	static constexpr uint64_t kStrided = 1;

  public:
	NIHILUS_HOST_DEVICE PitchLinearCoord() {
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord(Coord<2> const& coord) : Base(coord) {
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord(Index contiguous_, Index strided_) : Base(contiguous_, strided_) {
	}

	NIHILUS_HOST_DEVICE Index const& contiguous() const {
		return this->at(kContiguous);
	}

	NIHILUS_HOST_DEVICE Index & contiguous() {
		return this->at(kContiguous);
	}

	NIHILUS_HOST_DEVICE Index const& strided() const {
		return this->at(kStrided);
	}

	NIHILUS_HOST_DEVICE Index & strided() {
		return this->at(kStrided);
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord operator+(Base const& b) const {
		return PitchLinearCoord(Base::operator+(b));
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord operator-(Base const& b) const {
		return PitchLinearCoord(Base::operator-(b));
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord operator-() const {
		return PitchLinearCoord(-at(0), -at(1));
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord operator*(Base const& b) const {
		return PitchLinearCoord(Base::operator*(b));
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord operator/(Base const& b) const {
		return PitchLinearCoord(Base::operator/(b));
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE PitchLinearCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};

class PermuteBase {
  public:
	using Index = int64_t;

	using LongIndex = int64_t;
};

class NoPermute : public PermuteBase {
  public:
	NIHILUS_HOST_DEVICE NoPermute(PitchLinearCoord extent, Index stride) {};

	NIHILUS_HOST_DEVICE LongIndex operator()(PitchLinearCoord coord) const {
		return 0;
	}
};

template<typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator> struct DefaultGemmConfiguration;

template<typename ArchTag, typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<OpClassSimt, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator> {
	static constexpr uint64_t kAlignmentA = 1;
	static constexpr uint64_t kAlignmentB = 1;
	using ThreadblockShape				  = GemmShape<128, 128, 8>;
	using WarpShape						  = GemmShape<32, 64, 8>;
	using InstructionShape				  = GemmShape<1, 1, 1>;
	static constexpr uint64_t kStages	  = 2;

	using EpilogueOutputOp = LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>;

	using Operator = OpMultiplyAdd;
};

struct Sm120 {
	static constexpr uint64_t kMinComputeCapability = 120;
};

struct MatrixCoord : public constexpresh_coord<2> {
  public:
	using Index = uint64_t;

	using Base = constexpresh_coord<2>;

	using LongIndex = typename Base::Index;

  public:
	static constexpr uint64_t kRow = 0;

	static constexpr uint64_t kColumn = 1;

  public:
	NIHILUS_HOST_DEVICE MatrixCoord() {
	}

	NIHILUS_HOST_DEVICE MatrixCoord(constexpresh_coord<2> const& coord) : Base(coord) {
	}

	NIHILUS_HOST_DEVICE MatrixCoord(Index row, Index column) : Base(row, column) {
	}

	NIHILUS_HOST_DEVICE Index const row() const {
		return this->at(kRow);
	}

	NIHILUS_HOST_DEVICE Index row() {
		return this->at(kRow);
	}

	NIHILUS_HOST_DEVICE Index& operator[](uint64_t dim) {
		return Base::operator[](dim);
	}

	NIHILUS_HOST_DEVICE Index const& operator[](uint64_t dim) const {
		return Base::operator[](dim);
	}

	NIHILUS_HOST_DEVICE Index const column() const {
		return this->at(kColumn);
	}

	NIHILUS_HOST_DEVICE Index column() {
		return this->at(kColumn);
	}

	NIHILUS_HOST_DEVICE MatrixCoord operator+(Base const& b) const {
		return MatrixCoord(Base::operator+(b));
	}

	NIHILUS_HOST_DEVICE MatrixCoord operator-(Base const& b) const {
		return MatrixCoord(Base::operator-(b));
	}

	NIHILUS_HOST_DEVICE MatrixCoord operator*(Base const& b) const {
		return MatrixCoord(Base::operator*(b));
	}

	NIHILUS_HOST_DEVICE MatrixCoord operator/(Base const& b) const {
		return MatrixCoord(Base::operator/(b));
	}

	NIHILUS_HOST_DEVICE MatrixCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE MatrixCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE MatrixCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE MatrixCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};

class ColumnMajor {
  public:
	static constexpr uint64_t kRank = 2;

	static constexpr uint64_t kStrideRank = 1;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = MatrixCoord;

	using Stride = constexpresh_coord<kStrideRank>;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE ColumnMajor(LongIndex ldm = 0) : stride_(ldm) {
	}

	NIHILUS_HOST_DEVICE ColumnMajor(Stride stride) : stride_(stride) {
	}

	NIHILUS_HOST_DEVICE static ColumnMajor packed(MatrixCoord const& extent) {
		return ColumnMajor(extent.row());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(MatrixCoord const& coord) const {
		return LongIndex(coord.column()) * LongIndex(stride_.at(0)) + coord.row();
	}

	NIHILUS_HOST_DEVICE MatrixCoord inverse(LongIndex offset) const {
		return MatrixCoord(Index(offset % stride_.at(0)), Index(offset / stride_.at(0)));
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE typename Stride::Index stride(uint64_t idx) const {
		return stride_.at(idx);
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(MatrixCoord const& extent) const {
		return LongIndex(extent.column()) * LongIndex(stride_.at(0));
	}
};


class RowMajor {
  public:
	static constexpr uint64_t kRank = 2;

	static constexpr uint64_t kStrideRank = 1;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = MatrixCoord;

	using Stride = constexpresh_coord<kStrideRank>;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE RowMajor(LongIndex ldm = 0) : stride_(ldm) {
	}

	NIHILUS_HOST_DEVICE RowMajor(Stride stride) : stride_(stride) {
	}

	NIHILUS_HOST_DEVICE static RowMajor packed(MatrixCoord const& extent) {
		return RowMajor(extent.column());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(MatrixCoord const& coord) const {
		return LongIndex(coord.row()) * LongIndex(stride_.at(0)) + coord.column();
	}

	NIHILUS_HOST_DEVICE MatrixCoord inverse(LongIndex offset) const {
		return MatrixCoord(Index(offset / stride_.at(0)), Index(offset % stride_.at(0)));
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE Stride& stride() {
		return stride_;
	}

	NIHILUS_HOST_DEVICE typename Stride::Index stride(uint64_t idx) const {
		return stride_.at(idx);
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(MatrixCoord const& extent) const {
		return LongIndex(extent.row()) * LongIndex(stride_.at(0));
	}
};

NIHILUS_DEVICE
uint64_t RematerializeThreadIdxX() {
	return threadIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeThreadIdxY() {
	return threadIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeThreadIdxZ() {
	return threadIdx.z;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeBlockIdxX() {
	return blockIdx.x;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeBlockIdxY() {
	return blockIdx.y;
}

/// Helper to rematerialize block Idx. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeBlockIdxZ() {
	return blockIdx.z;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeBlockDimX() {
	return blockDim.x;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeBlockDimY() {
	return blockDim.y;
}

/// Helper to rematerialize block Dim. Reduces register liveness.
NIHILUS_DEVICE
uint64_t RematerializeBlockDimZ() {
	return blockDim.z;
}

template<uint64_t M, uint64_t K, uint64_t N = 1> struct GemmIdentityThreadblockSwizzle {
	template<uint64_t split_k_slices, uint64_t M_new, uint64_t K_new> NIHILUS_HOST_DEVICE static constexpr decltype(auto) get_tiled_shape(GemmCoord<M_new, K_new> tile_size) {
		constexpr uint64_t M_newer = (M + M_new - 1) / M_new;
		constexpr uint64_t K_newer = split_k_slices;
		return GemmCoord<M_newer, K_newer>{};
	}	

	template<uint64_t M_new, uint64_t K_new>
	NIHILUS_HOST_DEVICE static constexpr uint64_t get_log_tile(GemmCoord<M_new, K_new> tiled_shape) {
		auto n = tiled_shape.n();

		if (N >= 8 && n >= 6)
			return 3;
		else if (N >= 4 && n >= 3)
			return 2;
		else if (N >= 2 && n >= 2)
			return 1;
		else
			return 0;
	}

	template<typename GemmCoordType> NIHILUS_HOST_DEVICE static dim3 get_grid_shape(GemmCoordType tiled_shape) {
		uint64_t tile = 1 << get_log_tile(tiled_shape);
		return dim3(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile, tiled_shape.k());
	}

	NIHILUS_DEVICE static decltype(auto) get_tile_offset(uint64_t log_tile) {
		uint64_t block_idx_x = RematerializeBlockIdxX();
		uint64_t block_idx_y = RematerializeBlockIdxY();
		uint64_t block_idx_z = RematerializeBlockIdxZ();

		return constexpresh_coord<3>{ (block_idx_x >> log_tile), (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)), block_idx_z };
	}

	/*
	NIHILUS_HOST_DEVICE	static GemmCoord get_tiled_shape(conv::Operator conv_operator, conv::Conv2dProblemSize const& problem_size, GemmCoord tile_size, uint64_t split_k_slices) {
		gemm::GemmCoord implicit_gemm_problem_size = conv::implicit_gemm_problem_size(conv_operator, problem_size);

		return get_tiled_shape(implicit_gemm_problem_size, tile_size, split_k_slices);
	}

	NIHILUS_HOST_DEVICE	static GemmCoord get_tiled_shape(conv::Operator conv_operator, conv::Conv3dProblemSize const& problem_size, GemmCoord tile_size, uint64_t split_k_slices) {
		gemm::GemmCoord implicit_gemm_problem_size = conv::implicit_gemm_problem_size(conv_operator, problem_size);

		return get_tiled_shape(implicit_gemm_problem_size, tile_size, split_k_slices);
	}

	NIHILUS_DEVICE static GemmCoord get_tile_offset(GemmCoord tiled_shape) {
		uint64_t const kTile = N;
		uint64_t block_idx_x = RematerializeBlockIdxX();
		uint64_t block_idx_y = RematerializeBlockIdxY();

		if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
			return GemmCoord{ block_idx_x, block_idx_y, RematerializeBlockIdxZ() };

		return GemmCoord{ (block_idx_x / kTile), (block_idx_y * kTile) + (block_idx_x % kTile), RematerializeBlockIdxZ() };
	}*/
};

template<typename T> struct sizeof_bits {
	static constexpr uint64_t value = uint64_t(sizeof(T) * 8);
};

template<typename T> struct sizeof_bits<T const> : sizeof_bits<T> {};

template<typename T> struct sizeof_bits<T volatile> : sizeof_bits<T> {};

template<typename T> struct sizeof_bits<T const volatile> : sizeof_bits<T> {};

template<> struct sizeof_bits<void> {
	static constexpr uint64_t value = 0;
};

template<typename Element_, typename Storage_ = uint8_t, class = void> class ConstSubbyteReference {
  public:
	using Element		 = Element_;
	using Storage		 = Storage_;
	using StoragePointer = Storage const*;

	static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value, "Size of Element must not be greater than Storage.");

	static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value), "Storage must be divisible by Element");

  public:
	uint64_t const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

	Storage const kMask = ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1) : ~Storage(0));

  public:
	StoragePointer ptr_;

	uint64_t offset_;

  public:
	NIHILUS_HOST_DEVICE ConstSubbyteReference() : ptr_(nullptr), offset_(0) {
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference(Element const* ptr, int64_t offset) : ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
		int64_t offset_in_vectors  = offset / kElementsPerVector;
		int64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = uint64_t(offset_in_elements);
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference(Element* ptr = nullptr) : ConstSubbyteReference(ptr, 0) {
	}

	NIHILUS_HOST_DEVICE StoragePointer storage_pointer() const {
		return ptr_;
	}

	NIHILUS_HOST_DEVICE uint64_t element_offset() const {
		return offset_;
	}

	NIHILUS_HOST_DEVICE Element get() const {
		Storage item = Storage((*ptr_ >> (offset_ * sizeof_bits<Element>::value)) & kMask);
		return reinterpret_cast<Element const&>(item);
	}

	NIHILUS_HOST_DEVICE operator Element() const {
		return get();
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference& operator+=(uint64_t offset) {
		offset += offset_;

		uint64_t offset_in_vectors	= offset / kElementsPerVector;
		uint64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference& operator+=(long long offset) {
		offset += offset_;

		long long offset_in_vectors = offset / kElementsPerVector;
		uint64_t offset_in_elements = uint64_t(offset % kElementsPerVector);

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference& operator-=(uint64_t offset) {
		uint64_t offset_in_vectors	= offset / kElementsPerVector;
		uint64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference& operator-=(long long offset) {
		long long offset_in_vectors = offset / kElementsPerVector;
		uint64_t offset_in_elements = uint64_t(offset % kElementsPerVector);

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference operator+(uint64_t offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference operator+(long long offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference operator-(uint64_t offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE ConstSubbyteReference operator-=(long long offset) const {
		ConstSubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE ptrdiff_t operator-(ConstSubbyteReference ref) const {
		return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
	}

	NIHILUS_HOST_DEVICE explicit operator uint64_t() const {
		return uint64_t(get());
	}

	NIHILUS_HOST_DEVICE explicit operator int64_t() const {
		return int64_t(get());
	}

	NIHILUS_HOST_DEVICE explicit operator float() const {
		return float(get());
	}

	NIHILUS_HOST_DEVICE explicit operator double() const {
		return double(get());
	}
};

template<typename Element_,
	typename Storage_ =
#if defined(__CUDA_ARCH__)
	#if (__CUDA_ARCH__ >= 700)
		uint16_t
	#else
		uint32_t
	#endif
#else
		uint8_t
#endif
	,
	class = void>
class SubbyteReference {
  public:
	using Element		 = Element_;
	using Storage		 = Storage_;
	using StoragePointer = Storage*;

	static_assert(sizeof_bits<Element>::value <= sizeof_bits<Storage>::value, "Size of Element must not be greater than Storage.");

	static_assert(!(sizeof_bits<Storage>::value % sizeof_bits<Element>::value), "Storage must be divisible by Element");

  public:
	uint64_t const kElementsPerVector = sizeof_bits<Storage>::value / sizeof_bits<Element>::value;

	Storage const kMask = ((sizeof_bits<Element>::value < sizeof_bits<Storage>::value) ? (Storage(1) << sizeof_bits<Element>::value) - Storage(1) : ~Storage(0));

  public:
	StoragePointer ptr_;

	uint64_t offset_;

  public:
	NIHILUS_HOST_DEVICE SubbyteReference() : ptr_(nullptr), offset_(0) {
	}

	NIHILUS_HOST_DEVICE SubbyteReference(Element* ptr, int64_t offset) : ptr_(reinterpret_cast<StoragePointer>(ptr)), offset_(0) {
		int64_t offset_in_vectors  = offset / kElementsPerVector;
		int64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = uint64_t(offset_in_elements);
	}

	NIHILUS_HOST_DEVICE SubbyteReference(Element* ptr = nullptr) : SubbyteReference(ptr, 0) {
	}

	NIHILUS_HOST_DEVICE StoragePointer storage_pointer() const {
		return ptr_;
	}

	NIHILUS_HOST_DEVICE Element* operator&() const {
		return reinterpret_cast<Element*>(ptr_);
	}

	NIHILUS_HOST_DEVICE uint64_t element_offset() const {
		return offset_;
	}

	NIHILUS_HOST_DEVICE Element get() const {
		uint8_t const* byte_ptr				 = reinterpret_cast<uint8_t const*>(ptr_);
		constexpr uint64_t elements_per_byte = sizeof_bits<uint8_t>::value / sizeof_bits<Element>::value;
		byte_ptr += offset_ / elements_per_byte;
		uint64_t byte_offset = offset_ % elements_per_byte;
		uint8_t item		 = uint8_t((*byte_ptr >> (byte_offset * sizeof_bits<Element>::value)) & kMask);
		return reinterpret_cast<Element const&>(item);
	}

	NIHILUS_HOST_DEVICE SubbyteReference& set(Element const& x) {
		Storage item		= (reinterpret_cast<Storage const&>(x) & kMask);
		Storage kUpdateMask = Storage(~(kMask << (offset_ * sizeof_bits<Element>::value)));
		Storage new_bits	= Storage(item << (offset_ * sizeof_bits<Element>::value));

#if defined(__CUDA_ARCH__)
		Storage original;
		Storage updated;

		do {
			original = (*ptr_);

			updated = Storage((original & kUpdateMask) | new_bits);

			original = atomicCAS(ptr_, original, updated);

		} while (updated != original);

#else

		Storage original = (*ptr_);
		Storage updated	 = Storage((original & kUpdateMask) | new_bits);
		*ptr_			 = updated;

#endif

		return *this;
	}

	NIHILUS_HOST_DEVICE operator Element() const {
		return get();
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator=(Element const& x) {
		return set(x);
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator=(SubbyteReference const& x) {
		return set(x.get());
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator=(ConstSubbyteReference<Element, Storage> const& x) {
		return set(x.get());
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator+=(uint64_t offset) {
		offset += offset_;

		uint64_t offset_in_vectors	= offset / kElementsPerVector;
		uint64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator+=(long long offset) {
		offset += offset_;

		long long offset_in_vectors = offset / kElementsPerVector;
		uint64_t offset_in_elements = uint64_t(offset % kElementsPerVector);

		ptr_ += offset_in_vectors;
		offset_ = offset_in_elements;

		return *this;
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator-=(uint64_t offset) {
		uint64_t offset_in_vectors	= offset / kElementsPerVector;
		uint64_t offset_in_elements = offset % kElementsPerVector;

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE SubbyteReference& operator-=(long long offset) {
		long long offset_in_vectors = offset / kElementsPerVector;
		uint64_t offset_in_elements = uint64_t(offset % kElementsPerVector);

		ptr_ -= offset_in_vectors;
		offset_ -= offset_in_elements;

		if (offset_ < 0) {
			offset_ += kElementsPerVector;
			--ptr_;
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE SubbyteReference operator+(uint64_t offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE SubbyteReference operator+(long long offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref += offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE SubbyteReference operator-(uint64_t offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE SubbyteReference operator-=(long long offset) const {
		SubbyteReference ref(ptr_, offset_);
		ref -= offset;

		return ref;
	}

	NIHILUS_HOST_DEVICE ptrdiff_t operator-(SubbyteReference ref) const {
		return (ptr_ - ref.ptr_) * kElementsPerVector + (offset_ - ref.offset_);
	}

	NIHILUS_HOST_DEVICE explicit operator uint64_t() const {
		return uint64_t(get());
	}

	NIHILUS_HOST_DEVICE explicit operator int64_t() const {
		return int64_t(get());
	}

	NIHILUS_HOST_DEVICE explicit operator float() const {
		return float(get());
	}

	NIHILUS_HOST_DEVICE explicit operator double() const {
		return double(get());
	}
};

template<typename Element, bool subbyte = (sizeof_bits<Element>::value < 8)> struct ReferenceFactory;

template<typename Element> struct ReferenceFactory<Element, false> {
	static constexpr uint64_t kElementsPerVector = 1;

	NIHILUS_HOST_DEVICE static Element& get(Element* ptr, int64_t offset) {
		return ptr[offset];
	}

	NIHILUS_HOST_DEVICE static Element const& get(Element const* ptr, int64_t offset) {
		return ptr[offset];
	}

	NIHILUS_HOST_DEVICE static Element* add_pointer_offset(Element* ptr, int64_t offset) {
		return ptr + offset;
	}

	NIHILUS_HOST_DEVICE static Element const* add_pointer_offset(Element const* ptr, int64_t offset) {
		return ptr + offset;
	}
};

template<typename Element_, typename Layout_> class TensorRef {
  public:
	using Element = Element_;

	using Layout = Layout_;

	using Reference = typename std::conditional<sizeof_bits<Element>::value >= 8, Element&, SubbyteReference<Element>>::type;

	static constexpr uint64_t kRank = Layout::kRank;

	using Index = typename Layout::Index;

	using LongIndex = typename Layout::LongIndex;

	using TensorCoord = typename Layout::TensorCoord;

	using Stride = typename Layout::Stride;

	using ConstTensorRef = TensorRef<typename std::remove_const<Element>::type const, Layout>;

	using NonConstTensorRef = TensorRef<typename std::remove_const<Element>::type, Layout>;

	static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

  public:
	Element* ptr_;

	Layout layout_;

  public:
	NIHILUS_HOST_DEVICE TensorRef() : ptr_(nullptr) {
	}

	NIHILUS_HOST_DEVICE TensorRef(Element* ptr, Layout const& layout) : ptr_(ptr), layout_(layout) {
	}

	template<typename _Magic = uint64_t> NIHILUS_HOST_DEVICE TensorRef(NonConstTensorRef const& ref,
		_Magic magic = ( typename std::enable_if<!std::is_same<NonConstTensorRef, TensorRef<Element_, Layout_>>::value, _Magic>::type )0)
		: ptr_(ref.data()), layout_(ref.layout()) {
	}

	NIHILUS_HOST_DEVICE ConstTensorRef const_ref() const {
		return ConstTensorRef(ptr_, layout_);
	}

	NIHILUS_HOST_DEVICE NonConstTensorRef non_const_ref() const {
		return NonConstTensorRef(const_cast<typename std::remove_const<Element>::type*>(ptr_), layout_);
	}

	NIHILUS_HOST_DEVICE void reset(Element* ptr = nullptr) {
		ptr_ = ptr;
	}

	NIHILUS_HOST_DEVICE void reset(Element* ptr, Layout const& layout) {
		ptr_	= ptr;
		layout_ = layout;
	}

	NIHILUS_HOST_DEVICE bool good() const {
		return ptr_ != nullptr;
	}

	NIHILUS_HOST_DEVICE Element* data() const {
		return ptr_;
	}

	NIHILUS_HOST_DEVICE Reference data(LongIndex idx) const {
		return ReferenceFactory<typename std::remove_const<Element>::type, (sizeof_bits<Element>::value < 8)>::get(ptr_, idx);
	}

	NIHILUS_HOST_DEVICE Layout& layout() {
		return layout_;
	}

	NIHILUS_HOST_DEVICE Layout layout() const {
		return layout_;
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return layout_.stride();
	}

	NIHILUS_HOST_DEVICE Stride& stride() {
		return layout_.stride();
	}

	NIHILUS_HOST_DEVICE typename Layout::Stride::Index stride(uint64_t dim) const {
		return layout_.stride().at(dim);
	}

	NIHILUS_HOST_DEVICE LongIndex offset(TensorCoord const& coord) const {
		return layout_(coord);
	}

	NIHILUS_HOST_DEVICE Reference at(TensorCoord const& coord) const {
		return data(offset(coord));
	}

	NIHILUS_HOST_DEVICE Reference operator[](TensorCoord const& coord) const {
		return data(offset(coord));
	}

	NIHILUS_HOST_DEVICE TensorRef& add_pointer_offset(LongIndex offset_) {
		ptr_ = ReferenceFactory<typename std::remove_const<Element>::type, (sizeof_bits<Element>::value < 8)>::add_pointer_offset(ptr_, offset_);
		return *this;
	}

	NIHILUS_HOST_DEVICE TensorRef& add_coord_offset(TensorCoord const& coord) {
		add_pointer_offset(offset(coord));
		return *this;
	}

	NIHILUS_HOST_DEVICE TensorRef operator+(TensorCoord const& b) const {
		TensorRef result(*this);
		result.add_coord_offset(b);
		return result;
	}

	NIHILUS_HOST_DEVICE TensorRef& operator+=(TensorCoord const& b) {
		add_coord_offset(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE TensorRef operator-(TensorCoord const& b) const {
		TensorRef result(*this);
		result.add_pointer_offset(-offset(b));
		return result;
	}

	NIHILUS_HOST_DEVICE TensorRef& operator-=(TensorCoord const& b) {
		add_pointer_offset(-offset(b));
		return *this;
	}
};

enum class SharedMemoryClearOption {
	kNone,
	kZfill,
	kClearLastStage,
};

template<typename OperatorClass> struct WarpSize {
	static constexpr uint64_t value = 32;
};

template<typename Shape_, uint64_t Threads, uint64_t ElementsPerAccess = 1> struct PitchLinearStripminedThreadMap {
	using TensorCoord							 = PitchLinearCoord;
	using Shape									 = Shape_;
	static constexpr uint64_t kThreads			 = Threads;
	static constexpr uint64_t kElementsPerAccess = ElementsPerAccess;

	using ThreadAccessShape = PitchLinearShape<kElementsPerAccess, 1>;

	struct Detail {
		static_assert(!(Shape::kContiguous % kElementsPerAccess), "");

		using ShapeVec = PitchLinearShape<Shape::kContiguous / kElementsPerAccess, Shape::kStrided>;

		static_assert((Threads < ShapeVec::kContiguous && !(ShapeVec::kContiguous % kThreads)) || (!(kThreads % ShapeVec::kContiguous)),
			"Shape must be divisible by number of iterations of each thread.");
	};
	using Iterations = typename std::conditional<Threads >= Detail::ShapeVec::kContiguous,
		PitchLinearShape<1,
			(Threads >= Detail::ShapeVec::kContiguous ? (Detail::ShapeVec::kStrided + (kThreads / Detail::ShapeVec::kContiguous - 1)) / (kThreads / Detail::ShapeVec::kContiguous)
													  : 0)>,
		PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads, Detail::ShapeVec::kStrided>>::type;
	using Delta		 = typename std::conditional<Threads >= Detail::ShapeVec::kContiguous, PitchLinearShape<1, kThreads / Detail::ShapeVec::kContiguous>,
			 PitchLinearShape<kThreads * kElementsPerAccess, 1>>::type;

	using StorageShape = typename std::conditional<Threads >= Detail::ShapeVec::kContiguous,
		PitchLinearShape<Shape::kContiguous, Iterations::kStrided*(kThreads / Detail::ShapeVec::kContiguous)>, PitchLinearShape<Shape::kContiguous, Shape::kStrided>>::type;

	NIHILUS_HOST_DEVICE static TensorCoord initial_offset(uint64_t thread_id) {
		return TensorCoord((thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess, thread_id / Detail::ShapeVec::kContiguous);
	}
};

template<typename ThreadMap_> struct TransposePitchLinearThreadMapSimt {
	using ThreadMap								 = ThreadMap_;
	using TensorCoord							 = typename ThreadMap::TensorCoord;
	using Shape									 = typename ThreadMap::Shape;
	static constexpr uint64_t kThreads			 = ThreadMap::kThreads;
	static constexpr uint64_t kElementsPerAccess = ThreadMap::kElementsPerAccess;

	static_assert(kElementsPerAccess == 1, "Simt transpose requires elements per access to be 1");
	using Iterations = PitchLinearShape<ThreadMap::Iterations::kStrided, ThreadMap::Iterations::kContiguous>;

	static_assert(Iterations::kCount, "Number of iterations must be non-zero");

	static_assert(Iterations::kStrided == 1, "Strided iteration has to be one to reuse the same shared store function with those that don't need transpose");

	using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

	using Delta = PitchLinearShape<ThreadMap::Delta::kStrided, ThreadMap::Delta::kContiguous>;

	NIHILUS_HOST_DEVICE static TensorCoord initial_offset(uint64_t thread_id) {
		TensorCoord coord = ThreadMap::initial_offset(thread_id);

		return TensorCoord(coord.strided(), coord.contiguous());
	}
};

class PitchLinear {
  public:
	static constexpr uint64_t kRank = 2;

	static constexpr uint64_t kStrideRank = 1;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = PitchLinearCoord;

	using Stride = constexpresh_coord<kStrideRank>;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE PitchLinear(LongIndex ldm = 0) : stride_(ldm) {
	}

	NIHILUS_HOST_DEVICE PitchLinear(Stride _stride) : stride_(_stride) {
	}

	NIHILUS_HOST_DEVICE static PitchLinear packed(TensorCoord const& extent) {
		return PitchLinear(extent.contiguous());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(TensorCoord const& coord) const {
		return LongIndex(coord.contiguous()) + LongIndex(coord.strided()) * LongIndex(stride_.at(0));
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE LongIndex stride(uint64_t rank) const {
		return stride_.at(rank);
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(TensorCoord const& extent) const {
		return extent.strided() * stride_.at(0);
	}
};

template<typename Shape, typename Element, typename Layout, uint64_t AdvanceRank, typename ThreadMap,
	uint64_t Alignment = sizeof_bits<Element>::value * ThreadMap::kElementsPerAccess / 8>
class RegularTileIterator;

template<typename Shape_, typename Element_, uint64_t AdvanceRank, typename ThreadMap_, uint64_t Alignment>
class RegularTileIterator<Shape_, Element_, PitchLinear, AdvanceRank, ThreadMap_, Alignment> {
  public:
	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = PitchLinear;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;
	static constexpr uint64_t kAlignment   = Alignment;

	using Index		  = typename Layout::Index;
	using LongIndex	  = typename Layout::LongIndex;
	using StrideIndex = typename Layout::Stride::Index;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorCoord = typename Layout::TensorCoord;

	using Fragment = nihilus::array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

	using AccessType = nihilus::array<Element, ThreadMap::kElementsPerAccess>;

	static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, "Advance rank may only be along the contiguous or strided dimensions.");

  public:
	uint8_t* pointer_;

	StrideIndex stride_;

	Index increment_strided_;

	Index increment_advance_;

  public:
	NIHILUS_DEVICE RegularTileIterator() : pointer_(nullptr), increment_strided_(0), increment_advance_(0) {
	}

	NIHILUS_DEVICE RegularTileIterator(TensorRef const& ref, uint64_t thread_idx)
		: pointer_(reinterpret_cast<uint8_t*>(ref.data()) + (ref.offset(ThreadMap::initial_offset(thread_idx)) * sizeof_bits<Element>::value / 8)) {
		stride_			   = ref.stride()[0ll];
		increment_strided_ = (ref.stride()[0ll] * sizeof_bits<Element>::value) * ThreadMap::Delta::kStrided / 8;

		increment_advance_ = (kAdvanceRank == 0 ? Shape::kContiguous * sizeof_bits<Element>::value / 8 : Shape::kStrided * (ref.stride()[0] * sizeof_bits<Element>::value / 8));
	}

	NIHILUS_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
		AccessType* frag_ptr		= reinterpret_cast<AccessType*>(&frag);
		uint8_t const* byte_pointer = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

#pragma unroll
		for (uint64_t s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
			AccessType const* access_ptr = reinterpret_cast<AccessType const*>(byte_pointer);

#pragma unroll
			for (uint64_t c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
				uint64_t idx  = c + s * ThreadMap::Iterations::kContiguous;
				frag_ptr[idx] = access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess];
			}

			if (s + 1 < ThreadMap::Iterations::kStrided) {
				byte_pointer += increment_strided_;
			}
		}
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag, TensorCoord const& tile_offset) {
		load_with_pointer_offset(frag, tile_offset.contiguous() * Shape::kContiguous / ThreadMap::kElementsPerAccess + tile_offset.strided() * Shape::kStrided * stride_);
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag) {
		load_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
		AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
		uint8_t* byte_pointer	   = pointer_ + pointer_offset * sizeof_bits<Element>::value / 8;

#pragma unroll
		for (uint64_t s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
			AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_pointer);

#pragma unroll
			for (uint64_t c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
				uint64_t idx																  = c + s * ThreadMap::Iterations::kContiguous;
				access_ptr[c * ThreadMap::Delta::kContiguous / ThreadMap::kElementsPerAccess] = frag_ptr[idx];
			}

			if (s + 1 < ThreadMap::Iterations::kStrided) {
				byte_pointer += increment_strided_;
			}
		}
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag, TensorCoord const& tile_offset) {
		store_with_pointer_offset(frag, tile_offset.contiguous() * Shape::kContiguous + tile_offset.strided() * Shape::kStrided * stride_);
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag) {
		store_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE RegularTileIterator& operator++() {
		pointer_ += increment_advance_;
		return *this;
	}

	NIHILUS_HOST_DEVICE RegularTileIterator& operator--() {
		pointer_ -= increment_advance_;
		return *this;
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		pointer_ += pointer_offset;
	}

	NIHILUS_DEVICE void add_tile_offset(TensorCoord const& coord) {
		uint64_t offset = sizeof_bits<Element>::value * (coord.contiguous() * Shape::kContiguous + coord.strided() * Shape::kStrided * stride_) / 8;
		add_pointer_offset(offset);
	}

	NIHILUS_HOST_DEVICE void set_iteration_index(uint64_t index) {
	}

	NIHILUS_HOST_DEVICE AccessType* get() const {
		return reinterpret_cast<AccessType*>(pointer_);
	}
};

template<typename Shape_, typename Element_, uint64_t AdvanceRank, typename ThreadMap_, uint64_t Alignment>
class RegularTileIterator<Shape_, Element_, RowMajor, AdvanceRank, ThreadMap_, Alignment> {
  public:
	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = RowMajor;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;
	static constexpr uint64_t kAlignment   = Alignment;

	using Index		= typename Layout::Index;
	using LongIndex = typename Layout::LongIndex;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorCoord = typename Layout::TensorCoord;

	using Fragment = nihilus::array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

	using Underlying = RegularTileIterator<PitchLinearShape<Shape::kColumn, Shape::kRow>, Element, PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap, kAlignment>;

	using AccessType = typename Underlying::AccessType;

	static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, "Advance rank may only be along the row or column dimensions.");

  public:
	Underlying iterator_;

  public:
	NIHILUS_DEVICE RegularTileIterator() {
	}

	NIHILUS_DEVICE RegularTileIterator(TensorRef const& ref, uint64_t thread_idx) : iterator_({ ref.data(), ref.stride() }, thread_idx) {
	}

	NIHILUS_HOST_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
		iterator_.load_with_pointer_offset(frag, pointer_offset);
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag, TensorCoord const& tile_offset) {
		iterator_.load_with_pointer_offset(frag, { tile_offset.column(), tile_offset.row() });
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag) {
		iterator_.load_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
		iterator_.store_with_pointer_offset(frag, pointer_offset);
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag, TensorCoord const& tile_offset) {
		iterator_.store_with_pointer_offset(frag, { tile_offset.column(), tile_offset.row() });
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag) {
		iterator_.store_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE RegularTileIterator& operator++() {
		++iterator_;
		return *this;
	}

	NIHILUS_HOST_DEVICE RegularTileIterator& operator--() {
		--iterator_;
		return *this;
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		iterator_.add_pointer_offset(pointer_offset);
	}

	NIHILUS_DEVICE void add_tile_offset(TensorCoord const& coord) {
		iterator_.add_tile_offset({ coord.column(), coord.row() });
	}

	NIHILUS_HOST_DEVICE void set_iteration_index(uint64_t index) {
	}

	NIHILUS_HOST_DEVICE AccessType* get() const {
		return iterator_.get();
	}
};

template<typename Shape_, typename Element_, uint64_t AdvanceRank, typename ThreadMap_, uint64_t Alignment>
class RegularTileIterator<Shape_, Element_, ColumnMajor, AdvanceRank, ThreadMap_, Alignment> {
  public:
	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = ColumnMajor;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;
	static constexpr uint64_t kAlignment   = Alignment;

	using Index		= typename Layout::Index;
	using LongIndex = typename Layout::LongIndex;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorCoord = typename Layout::TensorCoord;

	using Fragment = nihilus::array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

	using Underlying = RegularTileIterator<PitchLinearShape<Shape::kRow, Shape::kColumn>, Element, PitchLinear, (kAdvanceRank == 0 ? 0 : 1), ThreadMap>;

	using AccessType = typename Underlying::AccessType;

	static_assert(kAdvanceRank == 0 || kAdvanceRank == 1, "Advance rank may only be along the row or column dimensions.");

  public:
	Underlying iterator_;

  public:
	NIHILUS_DEVICE RegularTileIterator() {
	}

	NIHILUS_DEVICE RegularTileIterator(TensorRef const& ref, uint64_t thread_idx) : iterator_({ ref.data(), ref.stride() }, thread_idx) {
	}

	NIHILUS_HOST_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
		iterator_.load_with_pointer_offset(frag, pointer_offset);
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag, TensorCoord const& tile_offset) {
		iterator_.load_with_pointer_offset(frag, { tile_offset.row(), tile_offset.column() });
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag) {
		iterator_.load_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
		iterator_.store_with_pointer_offset(frag, pointer_offset);
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag, TensorCoord const& tile_offset) {
		iterator_.store_with_pointer_offset(frag, { tile_offset.row(), tile_offset.column() });
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag) {
		iterator_.store_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE RegularTileIterator& operator++() {
		++iterator_;
		return *this;
	}

	NIHILUS_HOST_DEVICE RegularTileIterator& operator--() {
		--iterator_;
		return *this;
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		iterator_.add_pointer_offset(pointer_offset);
	}

	NIHILUS_DEVICE void add_tile_offset(TensorCoord const& coord) {
		iterator_.add_tile_offset({ coord.row(), coord.column() });
	}

	NIHILUS_HOST_DEVICE void set_iteration_index(uint64_t index) {
	}

	NIHILUS_HOST_DEVICE AccessType* get() const {
		return iterator_.get();
	}
};

template<uint64_t Row_, uint64_t Column_> struct MatrixShape {
	static constexpr uint64_t kRow	  = Row_;
	static constexpr uint64_t kColumn = Column_;
	static constexpr uint64_t kCount  = Row_ * Column_;

	NIHILUS_HOST_DEVICE static constexpresh_coord<2> toCoord() {
		return constexpresh_coord<2>{ kRow, kColumn };
	}
};

template<typename WarpShape> constexpr uint64_t simt_get_warp_threads_m() {
	return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}

constexpr uint64_t simt_transpose_padding(uint64_t threads, uint64_t crosswise, uint64_t size_in_bits) {
	return (size_in_bits >= 32 ? threads / crosswise / (size_in_bits / 32) : threads / crosswise * (32 / size_in_bits));
}

NIHILUS_HOST_DEVICE constexpr uint64_t const_min(uint64_t a, uint64_t b) {
	return (b < a ? b : a);
}

NIHILUS_HOST_DEVICE constexpr uint64_t const_max(uint64_t a, uint64_t b) {
	return (b > a ? b : a);
}

template<typename WarpShape_, typename LaneLayout_, typename LaneMmaShape_> struct MmaSimtPolicy {
	using WarpShape	   = WarpShape_;
	using LaneLayout   = LaneLayout_;
	using LaneMmaShape = LaneMmaShape_;
	using MmaShape	   = LaneMmaShape;

	NIHILUS_HOST_DEVICE static LaneLayout get_lane_layout() {
		return LaneLayout::packed({ WarpShape::kRow, WarpShape::kColumn });
	}
};

enum class ComplexTransform {
	kNone,
	kConjugate,
};

template<typename Shape_, uint64_t kThreads_, typename ElementA, typename LayoutA, typename ElementB, typename LayoutB, typename ElementC, typename LayoutC, typename Operator>
struct arch_mma;

template<typename ElementA, typename LayoutA, typename ElementB, typename LayoutB, typename ElementC_, typename LayoutC, typename Operator_>
struct arch_mma<GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB, ElementC_, LayoutC, Operator_> {
	using Shape	   = GemmShape<1, 1, 1>;
	using Operator = Operator_;
	using ElementC = ElementC_;

	NIHILUS_HOST_DEVICE void operator()(nihilus::array<ElementC, 1>& d, nihilus::array<ElementA, 1> const& a, nihilus::array<ElementB, 1> const& b,
		nihilus::array<ElementC, 1> const& c) {
		multiply_add<ElementA, ElementB, ElementC> op;

		d[0] = op(a[0], b[0], c[0]);
	}
};

template<typename Shape_, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_, typename Operator_>
struct MmaGeneric {
	using Shape = Shape_;

	using ElementA = ElementA_;

	using LayoutA = LayoutA_;

	using ElementB = ElementB_;

	using LayoutB = LayoutB_;

	using ElementC = ElementC_;

	using LayoutC = LayoutC_;

	using Operator = Operator_;

	using FragmentA = nihilus::array<ElementA, Shape::kMK>;

	using FragmentB = nihilus::array<ElementB, Shape::kKN>;

	using FragmentC = nihilus::array<ElementC, Shape::kMN>;

	using MmaOp = arch_mma<GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator>;

	static constexpr bool kMultipleOf2 = ((Shape::kM % 2 == 0) && (Shape::kN % 2 == 0));

	static constexpr bool kAllFp32 = std::is_same<ElementA, float>::value && std::is_same<ElementB, float>::value && std::is_same<ElementC, float>::value;

	NIHILUS_HOST_DEVICE void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B, FragmentC const& C) {
		TensorRef<ElementA const, LayoutA> a_ref(reinterpret_cast<ElementA const*>(&A), LayoutA::packed({ Shape::kM, Shape::kK }));

		TensorRef<ElementB const, LayoutB> b_ref(reinterpret_cast<ElementB const*>(&B), LayoutB::packed({ Shape::kK, Shape::kN }));

		TensorRef<ElementC, LayoutC> d_ref(reinterpret_cast<ElementC*>(&D), LayoutC::packed(make_Coord(Shape::kM, Shape::kN)));

		MmaOp mma_op;
		D = C;
#pragma unroll
		for (uint64_t k = 0; k < Shape::kK; ++k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 860)
			if constexpr (kMultipleOf2 && kAllFp32) {
	#pragma unroll
				for (uint64_t n = 0; n < Shape::kN; n += 2) {
	#pragma unroll
					for (uint64_t m = 0; m < Shape::kM; m += 2) {
						uint64_t m_serpentine = (n % 4) ? (Shape::kM - 2 - m) : m;

						{
							MatrixCoord mn(m_serpentine, n);
							MatrixCoord mk(m_serpentine, k);
							MatrixCoord kn(k, n);
							nihilus::array<ElementC, 1> d;
							nihilus::array<ElementA, 1> a;
							nihilus::array<ElementB, 1> b;
							d[0] = d_ref.at(mn);
							a[0] = a_ref.at(mk);
							b[0] = b_ref.at(kn);
							mma_op(d, a, b, d);
							d_ref.at(mn) = d[0];
						}


						{
							MatrixCoord mn(m_serpentine + 1, n);
							MatrixCoord mk(m_serpentine + 1, k);
							MatrixCoord kn(k, n);
							nihilus::array<ElementC, 1> d;
							nihilus::array<ElementA, 1> a;
							nihilus::array<ElementB, 1> b;
							d[0] = d_ref.at(mn);
							a[0] = a_ref.at(mk);
							b[0] = b_ref.at(kn);
							mma_op(d, a, b, d);
							d_ref.at(mn) = d[0];
						}

						{
							MatrixCoord mn(m_serpentine + 1, n + 1);
							MatrixCoord mk(m_serpentine + 1, k);
							MatrixCoord kn(k, n + 1);
							nihilus::array<ElementC, 1> d;
							nihilus::array<ElementA, 1> a;
							nihilus::array<ElementB, 1> b;
							d[0] = d_ref.at(mn);
							a[0] = a_ref.at(mk);
							b[0] = b_ref.at(kn);
							mma_op(d, a, b, d);
							d_ref.at(mn) = d[0];
						}

						{
							MatrixCoord mn(m_serpentine, n + 1);
							MatrixCoord mk(m_serpentine, k);
							MatrixCoord kn(k, n + 1);
							nihilus::array<ElementC, 1> d;
							nihilus::array<ElementA, 1> a;
							nihilus::array<ElementB, 1> b;
							d[0] = d_ref.at(mn);
							a[0] = a_ref.at(mk);
							b[0] = b_ref.at(kn);
							mma_op(d, a, b, d);
							d_ref.at(mn) = d[0];
						}
					}
				}
			} else
#endif
			{
#pragma unroll
				for (uint64_t n = 0; n < Shape::kN; ++n) {
#pragma unroll
					for (uint64_t m = 0; m < Shape::kM; ++m) {
						uint64_t m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;

						MatrixCoord mn(m_serpentine, n);
						MatrixCoord mk(m_serpentine, k);
						MatrixCoord kn(k, n);

						nihilus::array<ElementC, 1> d;
						nihilus::array<ElementA, 1> a;
						nihilus::array<ElementB, 1> b;

						d[0] = d_ref.at(mn);
						a[0] = a_ref.at(mk);
						b[0] = b_ref.at(kn);

						mma_op(d, a, b, d);

						d_ref.at(mn) = d[0];
					}
				}
			}
		}
	}
};

template<typename Shape, typename ElementA, typename LayoutA, typename ElementB, typename LayoutB, typename ElementC, typename LayoutC, typename Operator = OpMultiplyAdd,
	typename Enable = bool>
struct Mma;

template<typename Shape_, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_>
struct Mma<Shape_, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, OpMultiplyAdd, bool> {
	using Shape			  = Shape_;
	using ElementA		  = ElementA_;
	using LayoutA		  = LayoutA_;
	using ElementB		  = ElementB_;
	using LayoutB		  = LayoutB_;
	using ElementC		  = ElementC_;
	using LayoutC		  = LayoutC_;
	using Operator		  = OpMultiplyAdd;
	using FragmentA		  = nihilus::array<ElementA, Shape::kMK>;
	using FragmentB		  = nihilus::array<ElementB, Shape::kKN>;
	using FragmentC		  = nihilus::array<ElementC, Shape::kMN>;
	using ArchMmaOperator = typename MmaGeneric<Shape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator>::MmaOp;

	NIHILUS_HOST_DEVICE void operator()(FragmentC& D, FragmentA const& A, FragmentB const& B, FragmentC const& C) {
		MmaGeneric<Shape, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, Operator> mma;

		mma(D, A, B, C);
	}
};

template<uint64_t Interleave> struct RowMajorInterleaved {
	static constexpr uint64_t kRank = 2;

	static constexpr uint64_t kStrideRank = 1;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = MatrixCoord;

	using Stride = constexpresh_coord<kStrideRank>;

	static constexpr uint64_t kInterleave = Interleave;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE RowMajorInterleaved(LongIndex ldm = 0) : stride_(ldm) {
	}

	NIHILUS_HOST_DEVICE RowMajorInterleaved(Stride stride) : stride_(stride) {
	}

	NIHILUS_HOST_DEVICE static RowMajorInterleaved packed(MatrixCoord const& extent) {
		return RowMajorInterleaved(extent.column() * kInterleave);
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(MatrixCoord const& coord) const {
		Index row_major = coord.row() / kInterleave;
		Index row_minor = coord.row() % kInterleave;
		return LongIndex(row_major) * LongIndex(stride_.at(0)) + LongIndex(coord.column()) * kInterleave + row_minor;
	}

	NIHILUS_HOST_DEVICE MatrixCoord inverse(LongIndex offset) const {
		Index row_major = Index(offset / stride_.at(0));
		Index residual	= Index(offset % stride_.at(0));

		Index column	= residual / kInterleave;
		Index row_minor = residual % kInterleave;

		return MatrixCoord(row_major * kInterleave + row_minor, column);
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE Stride& stride() {
		return stride_;
	}

	NIHILUS_HOST_DEVICE typename Stride::Index stride(uint64_t idx) const {
		return stride_.at(idx);
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(MatrixCoord const& extent) const {
		return (extent.row() + kInterleave - 1) / kInterleave * stride_.at(0);
	}
};

template<uint64_t Interleave> struct ColumnMajorInterleaved {
	static constexpr uint64_t kRank = 2;

	static constexpr uint64_t kStrideRank = 1;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = MatrixCoord;

	using Stride = constexpresh_coord<kStrideRank>;

	static constexpr uint64_t kInterleave = Interleave;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE ColumnMajorInterleaved(LongIndex ldm = 0) : stride_(ldm) {
	}

	NIHILUS_HOST_DEVICE ColumnMajorInterleaved(Stride stride) : stride_(stride) {
	}


	NIHILUS_HOST_DEVICE static ColumnMajorInterleaved packed(MatrixCoord const& extent) {
		return ColumnMajorInterleaved(extent.row() * kInterleave);
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(MatrixCoord const& coord) const {
		Index column_major = coord.column() / kInterleave;
		Index column_minor = coord.column() % kInterleave;
		return LongIndex(column_major) * LongIndex(stride_.at(0)) + LongIndex(coord.row()) * kInterleave + column_minor;
	}

	NIHILUS_HOST_DEVICE MatrixCoord inverse(LongIndex offset) const {
		Index column_major = Index(offset / stride_.at(0));
		Index residual	   = Index(offset % stride_.at(0));

		Index row		   = residual / kInterleave;
		Index column_minor = residual % kInterleave;

		return MatrixCoord(row, column_major * kInterleave + column_minor);
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE typename Stride::Index stride(uint64_t idx) const {
		return stride_.at(idx);
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(MatrixCoord const& extent) const {
		return (extent.column() + kInterleave - 1) / kInterleave * stride_.at(0);
	}
};

enum class Operand {
	kA,
	kB,
	kC,
	kD,
};

template<uint64_t Bytes> NIHILUS_DEVICE void shared_load(void* dst, uint32_t ptr);

template<> NIHILUS_DEVICE void shared_load<2>(void* dst, uint32_t ptr) {
	asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(*reinterpret_cast<uint16_t*>(dst)) : "r"(ptr));
}

template<> NIHILUS_DEVICE void shared_load<4>(void* dst, uint32_t ptr) {
	asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(*reinterpret_cast<uint32_t*>(dst)) : "r"(ptr));
}

template<> NIHILUS_DEVICE void shared_load<8>(void* dst, uint32_t ptr) {
	uint2* dst_u64 = reinterpret_cast<uint2*>(dst);
	asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];\n" : "=r"(dst_u64->x), "=r"(dst_u64->y) : "r"(ptr));
}

template<> NIHILUS_DEVICE void shared_load<16>(void* dst, uint32_t ptr) {
	uint4* dst_u128 = reinterpret_cast<uint4*>(dst);
	asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(dst_u128->x), "=r"(dst_u128->y), "=r"(dst_u128->z), "=r"(dst_u128->w) : "r"(ptr));
}

template<typename Shape_, Operand Operand, typename Element_, typename Layout_, typename Policy_, uint64_t PartitionsK = 1, uint64_t PartitionGroupSize = 1>
class MmaSimtTileIterator;

template<typename Shape_, typename Element_, typename Policy_, uint64_t PartitionsK, uint64_t PartitionGroupSize>
class MmaSimtTileIterator<Shape_, Operand::kA, Element_, ColumnMajor, Policy_, PartitionsK, PartitionGroupSize> {
  public:
	using Shape = Shape_;

	static constexpr Operand kOperand = Operand::kA;

	using Element = Element_;

	using Layout = ColumnMajor;

	using Policy = Policy_;

	using TensorRef = TensorRef<Element, Layout>;

	using Index = typename TensorRef::Index;

	using LongIndex = typename TensorRef::LongIndex;

	using TensorCoord = typename TensorRef::TensorCoord;

	static_assert(!(Shape::kRow % Policy::WarpShape::kRow), "The warp-level GEMM M size must be divisible by the number of threads arranged along the M dimension.");

	static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
	static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
	static_assert(Policy::WarpShape::kRow > 0, "Policy::WarpShape::kRow must be greater than zero.");
	static_assert(Shape::kRow / Policy::WarpShape::kRow > 0, "Shape::kRow / Policy::WarpShape::kRow must be greater than zero.");

	using ThreadShape = MatrixShape<Shape::kRow / Policy::WarpShape::kRow, Shape::kColumn>;

	static_assert(!(ThreadShape::kRow % Policy::LaneMmaShape::kM), "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

	using Iterations = MatrixShape<ThreadShape::kRow / Policy::LaneMmaShape::kM, ThreadShape::kColumn>;

	using Fragment = nihilus::array<Element, ThreadShape::kCount>;

  public:
	TensorRef ref_;

  public:
	NIHILUS_HOST_DEVICE MmaSimtTileIterator() {
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator(TensorRef ref, uint64_t lane_id) {
		typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

		MatrixCoord lane_offset = lane_layout.inverse(lane_id) * MatrixCoord(Policy::LaneMmaShape::kM, 0);

		ref.add_coord_offset(lane_offset);

		ref_.reset(reinterpret_cast<typename nihilus::array<Element, Policy::LaneMmaShape::kM>::value_type*>(ref.data()), { ref.stride(0) / Policy::LaneMmaShape::kM });
	}


	NIHILUS_HOST_DEVICE MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
		ref_.add_pointer_offset(offset);
		return *this;
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
		ref_.add_coord_offset({ coord.row() * Shape::kRow / Policy::LaneMmaShape::kM, coord.column() * Shape::kColumn });

		return *this;
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& operator++() {
		ref_.add_coord_offset({ 0, Shape::kColumn });

		return *this;
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& operator--() {
		ref_.add_coord_offset({ 0, -Shape::kColumn });

		return *this;
	}

	NIHILUS_HOST_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
		nihilus::array<Element, Policy::LaneMmaShape::kM>* dst_ptr = reinterpret_cast<nihilus::array<Element, Policy::LaneMmaShape::kM>*>(&frag);

#pragma unroll
		for (uint64_t k = 0; k < Iterations::kColumn; ++k) {
#pragma unroll
			for (uint64_t m = 0; m < Iterations::kRow; ++m) {
#if 0
        dst_ptr[m + k * Iterations::kRow] = 
          *(ref_.data() + ref_.offset({m * Policy::WarpShape::kRow, k}) + pointer_offset / Policy::LaneMmaShape::kM);
#endif

				auto ptr = ref_.data() + ref_.offset({ m * Policy::WarpShape::kRow, k }) + pointer_offset / Policy::LaneMmaShape::kM;
				shared_load(dst_ptr[m + k * Iterations::kRow], ptr);
			}
		}
	}
	NIHILUS_HOST_DEVICE void load(Fragment& frag) const {
		load_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) const {
		nihilus::array<Element, Policy::LaneMmaShape::kM> const* src_ptr = reinterpret_cast<nihilus::array<Element, Policy::LaneMmaShape::kM>*>(&frag);

#pragma unroll
		for (uint64_t k = 0; k < Iterations::kN; ++k) {
#pragma unroll
			for (uint64_t m = 0; m < Iterations::kM; ++m) {
				*(ref_.data() + ref_.offset(m * Policy::WarpShape::kM, k) + pointer_offset / Policy::LaneMmaShape::kM) = src_ptr[m + k * Iterations::kM];
			}
		}
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag) const {
		store_with_pointer_offset(frag, 0);
	}

	NIHILUS_DEVICE void set_kgroup_index(uint64_t k_group) {
	}
};

template<typename Shape_, typename Element_, typename Policy_, uint64_t PartitionsK, uint64_t PartitionGroupSize>
class MmaSimtTileIterator<Shape_, Operand::kB, Element_, RowMajor, Policy_, PartitionsK, PartitionGroupSize> {
  public:
	using Shape = Shape_;

	static constexpr Operand kOperand = Operand::kB;

	using Element = Element_;

	using Layout = RowMajor;

	using Policy = Policy_;

	using TensorRef = TensorRef<Element, Layout>;

	using Index = typename TensorRef::Index;

	using LongIndex = typename TensorRef::LongIndex;

	using TensorCoord = typename TensorRef::TensorCoord;


	static_assert(!(Shape::kColumn % Policy::WarpShape::kColumn), "The warp-level GEMM N size must be divisible by the number of threads arranged along the N dimension.");

	static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
	static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
	static_assert(Policy::WarpShape::kColumn > 0, "Policy::WarpShape::kColumn must be greater than zero.");
	static_assert(Shape::kColumn / Policy::WarpShape::kColumn > 0, "Shape::kColumn / Policy::WarpShape::kColumn must be greater than zero.");

	using ThreadShape = MatrixShape<Shape::kRow, Shape::kColumn / Policy::WarpShape::kColumn>;

	static_assert(!(ThreadShape::kColumn % Policy::LaneMmaShape::kN), "Thread-level GEMM must be divisible by Policy::LaneMmaShape.");

	using Iterations = MatrixShape<ThreadShape::kRow, ThreadShape::kColumn / Policy::LaneMmaShape::kN>;

	using Fragment = nihilus::array<Element, ThreadShape::kCount>;

  protected:
	::TensorRef<nihilus::array<Element, Policy::LaneMmaShape::kN>, RowMajor> ref_;

  public:
	NIHILUS_HOST_DEVICE MmaSimtTileIterator() {
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator(TensorRef ref, uint64_t lane_id) {
		typename Policy::LaneLayout lane_layout = Policy::get_lane_layout();

		MatrixCoord lane_offset = lane_layout.inverse(lane_id) * MatrixCoord(0, Policy::LaneMmaShape::kN);

		ref.add_coord_offset(lane_offset);

		ref_.reset(reinterpret_cast<nihilus::array<Element, Policy::LaneMmaShape::kN>*>(ref.data()), ref.stride(0) / Policy::LaneMmaShape::kN);
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& add_pointer_offset(LongIndex offset) {
		ref_.add_pointer_offset(offset);
		return *this;
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& add_tile_offset(TensorCoord const& coord) {
		ref_.add_coord_offset({ coord.row() * Shape::kRow, coord.column() * Shape::kColumn / Policy::LaneMmaShape::kN });

		return *this;
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& operator++() {
		ref_.add_coord_offset({ Shape::kRow, 0 });

		return *this;
	}

	NIHILUS_HOST_DEVICE MmaSimtTileIterator& operator--() {
		ref_.add_coord_offset({ -Shape::kRow, 0 });

		return *this;
	}

	NIHILUS_HOST_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
		nihilus::array<Element, Policy::LaneMmaShape::kN>* dst_ptr = reinterpret_cast<nihilus::array<Element, Policy::LaneMmaShape::kN>*>(&frag);

#pragma unroll
		for (uint64_t k = 0; k < Iterations::kRow; ++k) {
#pragma unroll
			for (uint64_t n = 0; n < Iterations::kColumn; ++n) {
#if 0
        dst_ptr[n + k * Iterations::kColumn] = 
          *(ref_.data() + ref_.offset({k, n * Policy::WarpShape::kColumn}) + pointer_offset / Policy::LaneMmaShape::kN);
#endif

				void const* ptr = ref_.data() + ref_.offset({ k, n * Policy::WarpShape::kColumn }) + pointer_offset / Policy::LaneMmaShape::kN;
				shared_load(dst_ptr[n + k * Iterations::kColumn], ptr);
			}
		}
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag) const {
		load_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) const {
		nihilus::array<Element, Policy::LaneMmaShape::kN> const* src_ptr = reinterpret_cast<nihilus::array<Element, Policy::LaneMmaShape::kN>*>(&frag);

#pragma unroll
		for (uint64_t k = 0; k < Iterations::kM; ++k) {
#pragma unroll
			for (uint64_t n = 0; n < Iterations::kN; ++n) {
				*(ref_.data() + ref_.offset({ k, n * Policy::WarpShape::kN }) + pointer_offset / Policy::LaneMmaShape::kN) = src_ptr[n + k * Iterations::kN];
			}
		}
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag, Index pointer_offset) const {
		store_with_pointer_offset(frag, 0);
	}

	NIHILUS_DEVICE void set_kgroup_index(uint64_t k_group) {
	}
};

template<typename T> class complex {
  public:
	using value_type = T;

  public:
	T _real;
	T _imag;

  public:
};

template<class T> NIHILUS_HOST_DEVICE complex<T> conj(complex<T> const& z) {
	return { z.real(), -z.imag() };
}

template<typename T, typename Enable = void> struct has_cutlass_conj : std::false_type {};

template<typename T> struct has_cutlass_conj<T, decltype(conj(std::declval<T>()), void())> : std::true_type {};

template<typename T> constexpr bool has_cutlass_conj_v = has_cutlass_conj<T>::value;

template<typename T, typename Enable = void> struct has_unqualified_conj : std::false_type {};

template<typename T> constexpr bool has_unqualified_conj_v = has_unqualified_conj<T>::value;

template<typename T> struct conjugate {
	NIHILUS_HOST_DEVICE static T impl(T const& z) {
		if constexpr (std::is_arithmetic_v<T>) {
			return z;
		} else if constexpr (has_unqualified_conj_v<T> || has_cutlass_conj_v<T>) {
			return conj(z);
		} else {
			return z;
		}
	}
};

template<typename T, uint64_t N> struct conjugate<nihilus::array<T, N>> {
	NIHILUS_HOST_DEVICE static nihilus::array<T, N> impl(nihilus::array<T, N> const& a) {
		nihilus::array<T, N> ca;
#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			ca[i] = conjugate<T>::impl(a[i]);
		}
		return ca;
	}
};

template<typename Shape_, typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_, typename ElementC_, typename LayoutC_, typename Policy_,
	uint64_t PartitionsK = 1, ComplexTransform TransformA = ComplexTransform::kNone, ComplexTransform TransformB = ComplexTransform::kNone, typename Enable = bool>
class MmaSimt {
  public:
	using Shape = Shape_;

	using ElementA = ElementA_;

	using LayoutA = LayoutA_;

	using ElementB = ElementB_;

	using LayoutB = LayoutB_;

	using ElementC = ElementC_;

	using LayoutC = LayoutC_;

	using Policy = Policy_;

	using OperatorClass = OpClassSimt;

	using ArchTag = Sm120;

	static ComplexTransform const kTransformA = TransformA;

	static ComplexTransform const kTransformB = TransformB;

	using ThreadLayoutA = typename std::conditional<std::is_same<ColumnMajorInterleaved<4>, LayoutA>::value, ColumnMajor,
		typename std::conditional<std::is_same<RowMajorInterleaved<4>, LayoutA>::value, RowMajor, LayoutA>::type>::type;

	using ThreadLayoutB = typename std::conditional<std::is_same<ColumnMajorInterleaved<4>, LayoutB>::value, ColumnMajor,
		typename std::conditional<std::is_same<RowMajorInterleaved<4>, LayoutB>::value, RowMajor, LayoutB>::type>::type;

	static constexpr bool use_dp4a = (std::is_same<ColumnMajorInterleaved<4>, LayoutA>::value || std::is_same<RowMajorInterleaved<4>, LayoutA>::value) &&
		std::is_same<ElementA, int8_t>::value && std::is_same<ElementB, int8_t>::value;

	using dp4a_type = typename std::conditional<use_dp4a, int8_t, bool>::type;

	using ThreadMma = Mma<GemmShape<Shape::kM / Policy::WarpShape::kRow, Shape::kN / Policy::WarpShape::kColumn, Policy::LaneMmaShape::kK>, ElementA, ThreadLayoutA, ElementB,
		ThreadLayoutB, ElementC, LayoutC, OpMultiplyAdd, dp4a_type>;

	using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

	using MathOperator = typename ArchMmaOperator::Operator;

	using InstructionShape = GemmShape<1, 1, use_dp4a ? 4 : 1>;

  public:
	using IteratorA = MmaSimtTileIterator<MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA, ElementA, LayoutA, Policy, PartitionsK, Shape::kK>;

	using FragmentA = typename IteratorA::Fragment;

	using TransformedFragmentA = FragmentA;

	using IteratorB = MmaSimtTileIterator<MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>, Operand::kB, ElementB, LayoutB, Policy, PartitionsK, Shape::kK>;

	using FragmentB = typename IteratorB::Fragment;

	using TransformedFragmentB = FragmentB;

	using IteratorC = MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC, ElementC, LayoutC, Policy>;

	using FragmentC = typename ThreadMma::FragmentC;

  public:
	NIHILUS_DEVICE MmaSimt() {
	}

	NIHILUS_DEVICE void operator()(FragmentC& d, FragmentA a, FragmentB b, FragmentC const& c, uint64_t group_idx = 0) const {
		ThreadMma mma;

		if (kTransformA == ComplexTransform::kConjugate) {
			conjugate<FragmentA> conj_a;
			a = conj_a(a);
		}

		if (kTransformB == ComplexTransform::kConjugate) {
			conjugate<FragmentB> conj_b;
			b = conj_b(b);
		}

		mma(d, a, b, c);
	}

	NIHILUS_DEVICE void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B, FragmentA const& A, FragmentB const& B) const {
		dst_A = A;
		dst_B = B;
	}
};

template<typename Operator_, typename SmemPaddingA_, typename SmemPaddingB_, uint64_t PartitionsK = 1> struct MmaPolicy {
	using Operator	   = Operator_;
	using SmemPaddingA = SmemPaddingA_;

	using SmemPaddingB = SmemPaddingB_;

	static constexpr uint64_t kPartitionsK = PartitionsK;
};

enum class StrideSupport {
	kStrided,
	kUnity,
	kFixed,
};

template<typename Shape, uint64_t WarpsRemaining, uint64_t ElementsPerAccess, uint64_t ElementSize, bool Is2dTile> struct RowArrangement;

template<typename Shape, uint64_t WarpsRemaining, uint64_t ElementsPerAccess, uint64_t ElementSize>
struct RowArrangement<Shape, WarpsRemaining, ElementsPerAccess, ElementSize, false> {
	static constexpr uint64_t kWarpSize			 = 32;
	static constexpr uint64_t kElementsPerAccess = ElementsPerAccess;
	static constexpr uint64_t kElementSize		 = ElementSize;

	static constexpr uint64_t kIterationsRow	= 1;
	static constexpr uint64_t kDeltaRow			= 1;
	static constexpr uint64_t kIterationsColumn = Shape::kColumn / kElementsPerAccess / kWarpSize;
	static constexpr uint64_t kDeltaColumn		= kWarpSize * kElementsPerAccess;

	static constexpr uint64_t kAccessWidth			= kWarpSize;
	static constexpr uint64_t kAccessRows			= 1;
	static constexpr uint64_t kWarpPartitionsRow	= 1;
	static constexpr uint64_t kWarpPartitionsColumn = WarpsRemaining;
};

template<uint64_t Column, uint64_t Row, uint64_t Group, uint64_t Cluster, uint64_t Tile> struct OutputTileShape {
	static constexpr uint64_t kColumn  = Column;
	static constexpr uint64_t kRow	   = Row;
	static constexpr uint64_t kGroup   = Group;
	static constexpr uint64_t kCluster = Cluster;
	static constexpr uint64_t kTile	   = Tile;

	static constexpr uint64_t kCount = kColumn * kRow * kGroup * kCluster * kTile;
};

template<typename Iterations, typename Delta> struct OutputTileThreadMapHelpers {
	NIHILUS_HOST_DEVICE static void iteration_index(uint64_t& column_idx, uint64_t& row_idx, uint64_t& group_idx, uint64_t& cluster_idx, uint64_t& tile_idx, uint64_t iter_idx) {
		column_idx		  = iter_idx % Iterations::kColumn;
		uint64_t residual = iter_idx / Iterations::kColumn;

		row_idx	 = residual % Iterations::kRow;
		residual = residual / Iterations::kRow;

		group_idx = residual % Iterations::kGroup;
		residual  = residual / Iterations::kGroup;

		cluster_idx = residual % Iterations::kCluster;
		tile_idx	= residual / Iterations::kCluster;
	}

	NIHILUS_HOST_DEVICE static MatrixCoord iteration_offset(uint64_t iter_idx) {
		uint64_t column_idx;
		uint64_t row_idx;
		uint64_t group_idx;
		uint64_t cluster_idx;
		uint64_t tile_idx;

		iteration_index(column_idx, row_idx, group_idx, cluster_idx, tile_idx, iter_idx);

		return MatrixCoord(row_idx * Delta::kRow + group_idx * Delta::kGroup + cluster_idx * Delta::kCluster + tile_idx * Delta::kTile,

			column_idx * Delta::kColumn);
	}
};

template<typename Shape_, typename Count_, uint64_t Threads, uint64_t ElementsPerAccess, uint64_t ElementSize> struct OutputTileOptimalThreadMap {
	using Shape = Shape_;
	using Count = Count_;

	static constexpr uint64_t kWarpSize	 = 32;
	static constexpr uint64_t kThreads	 = Threads;
	static constexpr uint64_t kWarpCount = kThreads / kWarpSize;

	static constexpr uint64_t kElementsPerAccess = ElementsPerAccess;
	static constexpr uint64_t kElementSize		 = ElementSize;

	struct Detail {
		static constexpr uint64_t kIterationsCluster = ((Shape::kCluster > kWarpCount) ? Shape::kCluster / kWarpCount : 1);

		static constexpr uint64_t kDeltaCluster =
			((Shape::kCluster > kWarpCount) ? Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup * Shape::kCluster / kIterationsCluster : 1);

		static constexpr uint64_t kCompactedDeltaCluster = ((Shape::kCluster > kWarpCount) ? Shape::kRow * Shape::kGroup * Shape::kCluster / kIterationsCluster : 1);

		static constexpr uint64_t kWarpPartitionsCluster = ((Shape::kCluster > kWarpCount) ? kWarpCount : kWarpCount / Shape::kCluster);

		static constexpr uint64_t kWarpsRemainingForGroups = ((Shape::kCluster > kWarpCount) ? 1 : kWarpCount / Shape::kCluster);

		static constexpr uint64_t kIterationsGroup = ((Shape::kGroup > kWarpsRemainingForGroups) ? Shape::kGroup / kWarpsRemainingForGroups : 1);

		static constexpr uint64_t kDeltaGroup = ((Shape::kGroup > kWarpsRemainingForGroups) ? Shape::kRow * Count::kRow * Shape::kGroup / kIterationsGroup : 1);

		static constexpr uint64_t kCompactedDeltaGroup = ((Shape::kGroup > kWarpsRemainingForGroups) ? Shape::kRow * Shape::kGroup / kIterationsGroup : 1);

		static constexpr uint64_t kWarpPartitionsGroup = ((Shape::kGroup > kWarpsRemainingForGroups) ? 1 : kWarpsRemainingForGroups / Shape::kGroup);

		static constexpr uint64_t kWarpsRemainingForRows = ((Shape::kGroup > kWarpsRemainingForGroups) ? 1 : kWarpsRemainingForGroups / Shape::kGroup);

		using RowArrangement = RowArrangement<Shape, kWarpsRemainingForRows, kElementsPerAccess, kElementSize, (Shape::kRow > kWarpsRemainingForRows)>;

		using WarpPartitions = OutputTileShape<RowArrangement::kWarpPartitionsColumn, RowArrangement::kWarpPartitionsRow, kWarpPartitionsGroup, kWarpPartitionsCluster, 1>;

		static constexpr uint64_t kAccessWidth = RowArrangement::kAccessWidth;
		static constexpr uint64_t kAccessRows  = RowArrangement::kAccessRows;
	};

	using Iterations = OutputTileShape<Detail::RowArrangement::kIterationsColumn, Detail::RowArrangement::kIterationsRow, Detail::kIterationsGroup, Detail::kIterationsCluster, 1>;

	using Delta = OutputTileShape<Detail::RowArrangement::kDeltaColumn, Detail::RowArrangement::kDeltaRow, Detail::kDeltaGroup, Detail::kDeltaCluster, 1>;

	NIHILUS_HOST_DEVICE static MatrixCoord initial_offset(uint64_t thread_idx) {
		uint64_t warp_idx = thread_idx / kWarpSize;
		uint64_t lane_idx = thread_idx % kWarpSize;

		uint64_t cluster_idx	  = warp_idx / Detail::WarpPartitions::kCluster;
		uint64_t residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

		uint64_t group_idx		= residual_cluster / Detail::WarpPartitions::kGroup;
		uint64_t residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

		uint64_t row_idx = residual_group / Detail::WarpPartitions::kRow;
		uint64_t col_idx = residual_group % Detail::WarpPartitions::kRow;

		uint64_t lane_row_offset = lane_idx / Detail::kAccessWidth;
		uint64_t lane_col_offset = lane_idx % Detail::kAccessWidth;

		uint64_t cluster_offset = cluster_idx * Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup;
		uint64_t group_offset	= group_idx * Shape::kRow * Count::kRow;
		uint64_t row_offset		= row_idx * Iterations::kRow * Detail::kAccessRows;
		uint64_t column_offset	= col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

		return MatrixCoord(cluster_offset + group_offset + row_offset + lane_row_offset, column_offset + lane_col_offset * kElementsPerAccess);
	}

	NIHILUS_HOST_DEVICE static MatrixCoord iteration_offset(uint64_t iter_idx) {
		return OutputTileThreadMapHelpers<Iterations, Delta>::iteration_offset(iter_idx);
	}

	struct CompactedThreadMap {
		using Shape = Shape_;

		using TileShape = MatrixShape<Shape::kTile * Shape::kCluster * Shape::kGroup * Shape::kRow, Shape::kColumn>;

		using Iterations =
			OutputTileShape<Detail::RowArrangement::kIterationsColumn, Detail::RowArrangement::kIterationsRow, Detail::kIterationsGroup, Detail::kIterationsCluster, 1>;

		using Delta = OutputTileShape<Detail::RowArrangement::kDeltaColumn, Detail::RowArrangement::kDeltaRow, Detail::kCompactedDeltaGroup, Detail::kCompactedDeltaCluster, 1>;

		static constexpr uint64_t kElementsPerAccess = ElementsPerAccess;

		static constexpr uint64_t kThreads = Threads;

		NIHILUS_HOST_DEVICE static MatrixCoord initial_offset(uint64_t thread_idx) {
			uint64_t warp_idx = thread_idx / kWarpSize;
			uint64_t lane_idx = thread_idx % kWarpSize;

			uint64_t cluster_idx	  = warp_idx / Detail::WarpPartitions::kCluster;
			uint64_t residual_cluster = warp_idx % Detail::WarpPartitions::kCluster;

			uint64_t group_idx		= residual_cluster / Detail::WarpPartitions::kGroup;
			uint64_t residual_group = residual_cluster % Detail::WarpPartitions::kGroup;

			uint64_t row_idx = residual_group / Detail::WarpPartitions::kRow;
			uint64_t col_idx = residual_group % Detail::WarpPartitions::kRow;

			uint64_t lane_row_offset = lane_idx / Detail::kAccessWidth;
			uint64_t lane_col_offset = lane_idx % Detail::kAccessWidth;

			uint64_t cluster_offset = cluster_idx * Shape::kRow * Shape::kGroup;
			uint64_t group_offset	= group_idx * Shape::kRow;
			uint64_t row_offset		= row_idx * Iterations::kRow * Detail::kAccessRows;
			uint64_t column_offset	= col_idx * Iterations::kColumn * Detail::kAccessWidth * kElementsPerAccess;

			MatrixCoord coord(cluster_offset + group_offset + row_offset + lane_row_offset, column_offset + lane_col_offset * kElementsPerAccess);

			return coord;
		}
	};
};

template<typename ThreadblockShape_, typename WarpShape_, typename MmaSimtPolicy_, uint64_t PartitionsK, typename Element_, uint64_t ElementsPerAccess>
struct DefaultThreadMapSimt {
	using ThreadblockShape						 = ThreadblockShape_;
	using WarpShape								 = WarpShape_;
	using MmaSimtPolicy							 = MmaSimtPolicy_;
	static constexpr uint64_t kPartitionsK		 = PartitionsK;
	using Element								 = Element_;
	static constexpr uint64_t kElementsPerAccess = ElementsPerAccess;

	struct Detail {
		static constexpr uint64_t kWarpSize = 32;

		static_assert(!(ThreadblockShape::kM % WarpShape::kM) && !(ThreadblockShape::kN % WarpShape::kN), "Divisibility");

		using WarpCount						  = GemmShape<ThreadblockShape::kM / WarpShape::kM, ThreadblockShape::kN / WarpShape::kN, kPartitionsK>;
		static constexpr uint64_t kGroupCount = WarpShape::kM / (MmaSimtPolicy::WarpShape::kRow * MmaSimtPolicy::LaneMmaShape::kM);

		static constexpr uint64_t kThreads	  = WarpCount::kCount * kWarpSize;
		static constexpr uint64_t kIterations = MmaSimtPolicy::LaneMmaShape::kM * kGroupCount;
	};

	using Type = OutputTileOptimalThreadMap<OutputTileShape<ThreadblockShape::kN, 1, MmaSimtPolicy::WarpShape::kRow, Detail::WarpCount::kM, 1>,
		OutputTileShape<1, MmaSimtPolicy::LaneMmaShape::kM, Detail::kGroupCount, 1, Detail::kIterations>, Detail::kThreads, kElementsPerAccess, sizeof_bits<Element>::value>;
};

enum class Status {
	kSuccess,
	kErrorMisalignedOperand,
	kErrorInvalidDataType,
	kErrorInvalidLayout,
	kErrorInvalidProblem,
	kErrorNotSupported,
	kErrorWorkspaceNull,
	kErrorInternal,
	kErrorArchMismatch,
	kErrorInsufficientDriver,
	kErrorMemoryAllocation,
	kInvalid
};


struct OutputTileShapeDesc {
	uint64_t column;
	uint64_t row;
	uint64_t group;
	uint64_t cluster;
	uint64_t tile;

	NIHILUS_HOST_DEVICE OutputTileShapeDesc() : column(0), row(0), group(0), cluster(0), tile(0) {
	}

	NIHILUS_HOST_DEVICE OutputTileShapeDesc(uint64_t column_, uint64_t row_, uint64_t group_, uint64_t cluster_, uint64_t tile_)
		: column(column_), row(row_), group(group_), cluster(cluster_), tile(tile_) {
	}

	NIHILUS_HOST_DEVICE uint64_t count() const {
		return column * row * group * cluster * tile;
	}

#if 0
  NIHILUS_HOST_DEVICE   void print() const {
    printf("{%d, %d, %d, %d, %d}", column, row, group, cluster, tile);
  }
#endif
};

struct OutputTileThreadMapDesc {
	uint64_t threads;
	uint64_t elements_per_access;
	OutputTileShapeDesc shape;
	OutputTileShapeDesc iterations;
	OutputTileShapeDesc delta;
	OutputTileShapeDesc count;

	NIHILUS_HOST_DEVICE OutputTileThreadMapDesc() {
	}

	NIHILUS_HOST_DEVICE OutputTileThreadMapDesc(uint64_t threads_, uint64_t elements_per_access_, OutputTileShapeDesc shape_, OutputTileShapeDesc iterations_,
		OutputTileShapeDesc delta_, OutputTileShapeDesc count_)
		: threads(threads_), elements_per_access(elements_per_access_), shape(shape_), iterations(iterations_), delta(delta_), count(count_) {
	}
};

struct PredicatedTileIteratorParams {
	using Index		= int64_t;
	using LongIndex = int64_t;

	LongIndex stride;

	LongIndex increment_row;
	LongIndex increment_group;
	LongIndex increment_cluster;
	LongIndex advance_row;
	LongIndex advance_group;
	LongIndex advance_cluster;
	LongIndex advance_tile;

	NIHILUS_HOST_DEVICE Status initialize(LongIndex stride_, OutputTileThreadMapDesc thread_map) {
		stride = stride_;

		increment_row = stride * thread_map.delta.row;

		increment_group = stride * thread_map.delta.group - stride * thread_map.delta.row * (thread_map.iterations.row - 1);

		increment_cluster = stride * thread_map.delta.cluster - stride * thread_map.delta.group * (thread_map.iterations.group - 1) -
			stride * thread_map.delta.row * (thread_map.iterations.row - 1);

		advance_row = stride * thread_map.shape.row;

		advance_group = stride * (thread_map.shape.group - 1) * thread_map.shape.row * thread_map.count.row;

		advance_cluster = stride * thread_map.count.group * thread_map.shape.group * thread_map.count.row * thread_map.shape.row;

		advance_tile = stride * thread_map.shape.group * thread_map.shape.row * thread_map.shape.cluster * thread_map.shape.tile;

		return Status::kSuccess;
	}

	NIHILUS_HOST_DEVICE PredicatedTileIteratorParams() {
		initialize(LongIndex(0), OutputTileThreadMapDesc());
	}

	NIHILUS_HOST_DEVICE PredicatedTileIteratorParams(Index stride, OutputTileThreadMapDesc thread_map) {
		initialize(stride, thread_map);
	}
};

struct Tensor4DCoord : public constexpresh_coord<4> {
	using Base					 = constexpresh_coord<4>;
	using Index					 = typename Base::Index;
	using LongIndex				 = typename Base::LongIndex;
	static constexpr uint64_t kN = 0;
	static constexpr uint64_t kH = 1;
	static constexpr uint64_t kW = 2;
	static constexpr uint64_t kC = 3;
	NIHILUS_HOST_DEVICE Tensor4DCoord() {
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord(constexpresh_coord<4> const& coord) : Base(coord) {
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord(Index n, Index h, Index w, Index c) : Base(n, h, w, c) {
	}

	NIHILUS_HOST_DEVICE Index const& n() const {
		return this->at(kN);
	}

	NIHILUS_HOST_DEVICE Index& n() {
		return this->at(kN);
	}

	NIHILUS_HOST_DEVICE Index const& h() const {
		return this->at(kH);
	}

	NIHILUS_HOST_DEVICE Index& h() {
		return this->at(kH);
	}

	NIHILUS_HOST_DEVICE Index const& w() const {
		return this->at(kW);
	}

	NIHILUS_HOST_DEVICE Index& w() {
		return this->at(kW);
	}

	NIHILUS_HOST_DEVICE Index const& c() const {
		return this->at(kC);
	}

	NIHILUS_HOST_DEVICE Index& c() {
		return this->at(kC);
	}


	NIHILUS_HOST_DEVICE Tensor4DCoord operator+(Base const& b) const {
		return Tensor4DCoord(Base::operator+(b));
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord operator-(Base const& b) const {
		return Tensor4DCoord(Base::operator-(b));
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord operator*(Base const& b) const {
		return Tensor4DCoord(Base::operator*(b));
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord operator/(Base const& b) const {
		return Tensor4DCoord(Base::operator/(b));
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE Tensor4DCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};


struct Tensor5DCoord : public constexpresh_coord<5> {
	using Base = constexpresh_coord<5>;

	using Index = typename Base::Index;

	using LongIndex = typename Base::LongIndex;

	static constexpr uint64_t kN = 0;

	static constexpr uint64_t kD = 1;

	static constexpr uint64_t kH = 2;

	static constexpr uint64_t kW = 3;

	static constexpr uint64_t kC = 4;


	NIHILUS_HOST_DEVICE Tensor5DCoord() {
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord(constexpresh_coord<5> const& coord) : Base(coord) {
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord(Index n, Index d, Index h, Index w, Index c) : Base(n, d, h, w, c) {
	}

	NIHILUS_HOST_DEVICE Index const& n() const {
		return this->at(kN);
	}

	NIHILUS_HOST_DEVICE Index& n() {
		return this->at(kN);
	}

	NIHILUS_HOST_DEVICE Index const& d() const {
		return this->at(kD);
	}

	NIHILUS_HOST_DEVICE Index& d() {
		return this->at(kD);
	}

	NIHILUS_HOST_DEVICE Index const& h() const {
		return this->at(kH);
	}

	NIHILUS_HOST_DEVICE Index& h() {
		return this->at(kH);
	}

	NIHILUS_HOST_DEVICE Index const& w() const {
		return this->at(kW);
	}

	NIHILUS_HOST_DEVICE Index& w() {
		return this->at(kW);
	}

	NIHILUS_HOST_DEVICE Index const& c() const {
		return this->at(kC);
	}

	NIHILUS_HOST_DEVICE Index& c() {
		return this->at(kC);
	}


	NIHILUS_HOST_DEVICE Tensor5DCoord operator+(Base const& b) const {
		return Tensor5DCoord(Base::operator+(b));
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord operator-(Base const& b) const {
		return Tensor5DCoord(Base::operator-(b));
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord operator*(Base const& b) const {
		return Tensor5DCoord(Base::operator*(b));
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord operator/(Base const& b) const {
		return Tensor5DCoord(Base::operator/(b));
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord& operator+=(Base const& b) {
		Base::operator+=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord& operator-=(Base const& b) {
		Base::operator-=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord& operator*=(Base const& b) {
		Base::operator*=(b);
		return *this;
	}

	NIHILUS_HOST_DEVICE Tensor5DCoord& operator/=(Base const& b) {
		Base::operator/=(b);
		return *this;
	}
};

struct CacheOperation {
	enum Kind { Always, Global, Streaming, LastUse, Volatile, WriteBack, WriteThrough };
};

template<
	/// Fragment type to store loaded data
	typename AccessType,
	/// The bytes of loading
	int64_t LoadBytes,
	/// Cache operation
	CacheOperation::Kind cache_op = CacheOperation::Always>
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
	#define NIHILUS_ENABLE_L2_PREFETCH 1
#else
	#define NIHILUS_ENABLE_L2_PREFETCH 0
#endif

// The redundant mov PTX instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template<typename AccessType> struct global_load<AccessType, 64, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint4* data = reinterpret_cast<uint4*>(&D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %9, 0;\n"
					 "  mov.b32 %0, %10;\n"
					 "  mov.b32 %1, %11;\n"
					 "  mov.b32 %2, %12;\n"
					 "  mov.b32 %3, %13;\n"
					 "  mov.b32 %4, %14;\n"
					 "  mov.b32 %5, %15;\n"
					 "  mov.b32 %6, %16;\n"
					 "  mov.b32 %7, %17;\n"
#if NIHILUS_ENABLE_L2_PREFETCH
					 "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%8];\n"
					 "  @p ld.global.L2::128B.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#else
					 "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
					 "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#endif
					 "}\n"
			: "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w), "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w),
			"l"((( uint8_t* )ptr) + 16));
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// The redundant mov PTX instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template<typename AccessType> struct global_load<AccessType, 32, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint4* data = reinterpret_cast<uint4*>(&D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %9, 0;\n"
					 "  mov.b32 %0, %10;\n"
					 "  mov.b32 %1, %11;\n"
					 "  mov.b32 %2, %12;\n"
					 "  mov.b32 %3, %13;\n"
					 "  mov.b32 %4, %14;\n"
					 "  mov.b32 %5, %15;\n"
					 "  mov.b32 %6, %16;\n"
					 "  mov.b32 %7, %17;\n"
#if NIHILUS_ENABLE_L2_PREFETCH
					 "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%8];\n"
					 "  @p ld.global.L2::128B.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#else
					 "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
					 "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%18];\n"
#endif
					 "}\n"
			: "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w), "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w),
			"l"((( uint8_t* )ptr) + 16));
	}
};

template<typename AccessType> struct global_load<AccessType, 32, CacheOperation::LastUse> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint4* data = reinterpret_cast<uint4*>(&D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %9, 0;\n"
					 "  mov.b32 %0, %10;\n"
					 "  mov.b32 %1, %11;\n"
					 "  mov.b32 %2, %12;\n"
					 "  mov.b32 %3, %13;\n"
					 "  mov.b32 %4, %14;\n"
					 "  mov.b32 %5, %15;\n"
					 "  mov.b32 %6, %16;\n"
					 "  mov.b32 %7, %17;\n"
					 "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%8];\n"
					 "  @p ld.global.lu.v4.u32 {%4, %5, %6, %7}, [%18];\n"
					 "}\n"
			: "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w), "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w), "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w),
			"l"((( uint8_t* )ptr) + 16));
	}
};

template<typename AccessType> struct global_load<AccessType, 16, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint4& data = reinterpret_cast<uint4&>(D);
		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %5, 0;\n"
					 "  mov.b32 %0, %6;\n"
					 "  mov.b32 %1, %7;\n"
					 "  mov.b32 %2, %8;\n"
					 "  mov.b32 %3, %9;\n"
#if NIHILUS_ENABLE_L2_PREFETCH
					 "  @p ld.global.L2::128B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#else
					 "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
#endif
					 "}\n"
			: "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
	}
};

template<typename AccessType> struct global_load<AccessType, 16, CacheOperation::LastUse> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint4& data = reinterpret_cast<uint4&>(D);
		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %5, 0;\n"
					 "  mov.b32 %0, %6;\n"
					 "  mov.b32 %1, %7;\n"
					 "  mov.b32 %2, %8;\n"
					 "  mov.b32 %3, %9;\n"
					 "  @p ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];\n"
					 "}\n"
			: "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
	}
};

template<typename AccessType> struct global_load<AccessType, 8, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint2& data = reinterpret_cast<uint2&>(D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %3, 0;\n"
					 "  mov.b32 %0, %4;\n"
					 "  mov.b32 %1, %5;\n"
#if NIHILUS_ENABLE_L2_PREFETCH
					 "  @p ld.global.L2::128B.v2.u32 {%0, %1}, [%2];\n"
#else
					 "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
#endif
					 "}\n"
			: "=r"(data.x), "=r"(data.y)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data.x), "r"(data.y));
	}
};

template<typename AccessType> struct global_load<AccessType, 8, CacheOperation::LastUse> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint2& data = reinterpret_cast<uint2&>(D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %3, 0;\n"
					 "  mov.b32 %0, %4;\n"
					 "  mov.b32 %1, %5;\n"
					 "  @p ld.global.lu.v2.u32 {%0, %1}, [%2];\n"
					 "}\n"
			: "=r"(data.x), "=r"(data.y)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data.x), "r"(data.y));
	}
};

template<typename AccessType> struct global_load<AccessType, 4, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		unsigned& data = reinterpret_cast<unsigned&>(D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %2, 0;\n"
					 "  mov.b32 %0, %3;\n"
#if NIHILUS_ENABLE_L2_PREFETCH
					 "  @p ld.global.L2::128B.u32 %0, [%1];\n"
#else
					 "  @p ld.global.u32 %0, [%1];\n"
#endif
					 "}\n"
			: "=r"(data)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data));
	}
};

template<typename AccessType> struct global_load<AccessType, 4, CacheOperation::LastUse> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		unsigned& data = reinterpret_cast<unsigned&>(D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %2, 0;\n"
					 "  mov.b32 %0, %3;\n"
					 "  @p ld.global.lu.u32 %0, [%1];\n"
					 "}\n"
			: "=r"(data)
			: "l"(ptr), "r"(( int )pred_guard), "r"(data));
	}
};

template<typename AccessType> struct global_load<AccessType, 2, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint16_t& data = reinterpret_cast<uint16_t&>(D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %2, 0;\n"
					 "  mov.b16 %0, %3;\n"
#if NIHILUS_ENABLE_L2_PREFETCH
					 "  @p ld.global.L2::128B.u16 %0, [%1];\n"
#else
					 "  @p ld.global.u16 %0, [%1];\n"
#endif
					 "}\n"
			: "=h"(data)
			: "l"(ptr), "r"(( int )pred_guard), "h"(data));
	}
};

template<typename AccessType> struct global_load<AccessType, 2, CacheOperation::LastUse> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		uint16_t& data = reinterpret_cast<uint16_t&>(D);

		asm volatile("{\n"
					 "  .reg .pred p;\n"
					 "  setp.ne.b32 p, %2, 0;\n"
					 "  mov.b16 %0, %3;\n"
					 "  @p ld.global.lu.u16 %0, [%1];\n"
					 "}\n"
			: "=h"(data)
			: "l"(ptr), "r"(( int )pred_guard), "h"(data));
	}
};

template<typename AccessType> struct global_load<AccessType, 1, CacheOperation::Always> {
	NIHILUS_DEVICE
	global_load(AccessType& D, void const* ptr, bool pred_guard) {
		if (pred_guard)
			D = *(reinterpret_cast<AccessType const*>(ptr));
	}
};


template<typename Element_, typename Layout_> class TensorView : public TensorRef<Element_, Layout_> {
  public:
	using Base = TensorRef<Element_, Layout_>;

	using Layout = Layout_;

	using ConstTensorRef = typename Base::ConstTensorRef;

	using TensorRef = Base;

	using Element = Element_;

	using Reference = Element&;

	static uint64_t const kRank = Layout::kRank;

	using Index = typename Layout::Index;

	using LongIndex = typename Layout::LongIndex;

	using TensorCoord = typename Layout::TensorCoord;

	using Stride = typename Layout::Stride;

	using ConstTensorView = TensorView<typename std::remove_const<Element>::type const, Layout>;

	using NonConstTensorView = TensorView<typename std::remove_const<Element>::type, Layout>;

	static_assert(kRank > 0, "Cannot define a zero-rank TensorRef");

  public:
	TensorCoord extent_;

  public:
	NIHILUS_HOST_DEVICE TensorView() {
	}

	NIHILUS_HOST_DEVICE TensorView(Element* ptr, Layout const& layout, TensorCoord const& extent) : Base(ptr, layout), extent_(extent) {
	}

	NIHILUS_HOST_DEVICE TensorView(TensorRef const& ref, TensorCoord const& extent) : Base(ref), extent_(extent) {
	}

	NIHILUS_HOST_DEVICE TensorView(NonConstTensorView const& view) : Base(view), extent_(view.extent_) {
	}

	NIHILUS_HOST_DEVICE void reset(Element* ptr, Layout const& layout, TensorCoord const& extent) {
		Base::reset(ptr, layout);
		this->resize(extent);
	}

	NIHILUS_HOST_DEVICE void reset(Element* ptr) {
		Base::reset(ptr);
	}

	NIHILUS_HOST_DEVICE void resize(TensorCoord const& extent) {
		this->extent_ = extent;
	}

	NIHILUS_HOST_DEVICE TensorCoord const& extent() const {
		return extent_;
	}

	NIHILUS_HOST_DEVICE Index extent(uint64_t dim) const {
		return extent_.at(dim);
	}

	NIHILUS_HOST_DEVICE LongIndex size() const {
		return extent_.product();
	}

	NIHILUS_HOST_DEVICE bool contains(TensorCoord const& coord) const {
#pragma unroll
		for (uint64_t dim = 0; dim < kRank; ++dim) {
			if (!(coord[dim] >= 0 && coord[dim] < extent(dim))) {
				return false;
			}
		}
		return true;
	}

	NIHILUS_HOST_DEVICE TensorRef ref() const {
		return TensorRef(this->data(), this->layout());
	}

	NIHILUS_HOST_DEVICE ConstTensorRef const_ref() const {
		return ConstTensorRef(this->data(), this->layout());
	}

	NIHILUS_HOST_DEVICE ConstTensorView const_view() const {
		return ConstTensorView(const_ref(), extent_);
	}

	NIHILUS_HOST_DEVICE TensorView subview(TensorCoord extent, TensorCoord const& location = TensorCoord()) const {
		TensorView result(this->ref(), extent.clamp(extent_ - location));
		result.add_coord_offset(location);
		return result;
	}

	NIHILUS_HOST_DEVICE size_t capacity() const {
		return Base::layout().capacity(extent_);
	}

	NIHILUS_HOST_DEVICE TensorView operator+(TensorCoord const& b) const {
		TensorView result(*this);
		result.add_pointer_offset(this->offset(b));
		return result;
	}

	NIHILUS_HOST_DEVICE TensorView& operator+=(TensorCoord const& b) {
		this->add_pointer_offset(this->offset(b));
		return *this;
	}

	NIHILUS_HOST_DEVICE TensorView operator-(TensorCoord const& b) const {
		TensorRef result(*this);
		result.add_pointer_offset(-this->offset(b));
		return result;
	}

	NIHILUS_HOST_DEVICE TensorView& operator-=(TensorCoord const& b) {
		this->add_pointer_offset(-this->offset(b));
		return *this;
	}
};

template<typename Shape_, typename Element_, typename Layout_, uint64_t AdvanceRank, typename ThreadMap_, typename AccessType_> class PredicatedTileAccessIteratorPredicates {
  public:
	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = Layout_;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;
	using AccessType					   = AccessType_;

	using Index		= typename Layout::Index;
	using LongIndex = typename Layout::LongIndex;

	using TensorCoord = typename Layout::TensorCoord;

	static constexpr uint64_t kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::size_val;

	static_assert(!(ThreadMap::kElementsPerAccess % AccessType::size_val), "Vectors implied by the thread map must be divisible by the access type.");

	static constexpr uint64_t kPredicatesPerByte = 4;
	static constexpr uint64_t kPredicatesPerWord = 4 * kPredicatesPerByte;

	static constexpr uint64_t kPredicateCount = ThreadMap::Iterations::kCount * kAccessesPerVector;

	static constexpr uint64_t kPredicateByteCount = (kPredicateCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
	static constexpr uint64_t kPredicateWordCount = (kPredicateByteCount + 3) / 4;

	static constexpr unsigned kPredicateMask = (1u << kPredicatesPerByte) - 1u;

	static_assert(kPredicateWordCount <= 4, "Too many predicates.");

	using Mask = nihilus::array<uint32_t, kPredicateWordCount>;

	uint32_t predicates_[kPredicateWordCount];

	TensorCoord extent_;

	TensorCoord thread_offset_;

	TensorCoord residue_offset_;

	uint64_t iteration_vector_;

	uint64_t iteration_contiguous_;

	uint64_t iteration_strided_;

  public:
	NIHILUS_DEVICE void compute_predicates_(TensorCoord extent, bool is_steady_state = false) {
		NIHILUS_PRAGMA_UNROLL
		for (uint64_t i = 0; i < kPredicateWordCount; ++i) {
			predicates_[i] = 0u;
		}

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t access_idx = 0; access_idx < ThreadMap::Iterations::kCount * kAccessesPerVector; ++access_idx) {
			uint64_t s = access_idx / (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

			uint64_t access_residual = access_idx % (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

			uint64_t c = access_residual / kAccessesPerVector;
			uint64_t v = access_residual % kAccessesPerVector;

			TensorCoord iteration_coord(c * ThreadMap::Delta::kContiguous + v * AccessType::size_val, s * ThreadMap::Delta::kStrided);

			TensorCoord coord = thread_offset_ + iteration_coord;

			bool guard;

			if (is_steady_state) {
				if (kAdvanceRank == 0) {
					guard = (coord.strided() < extent.strided());
				} else {
					guard = (coord.contiguous() < extent.contiguous());
				}
			} else {
				guard = (coord.strided() < extent.strided() && coord.contiguous() < extent.contiguous());
			}

			uint64_t pred_idx = v + kAccessesPerVector * (c + ThreadMap::Iterations::kContiguous * s);

			uint64_t word_idx = pred_idx / kPredicatesPerWord;
			uint64_t residual = pred_idx % kPredicatesPerWord;
			uint64_t byte_idx = residual / kPredicatesPerByte;
			uint64_t bit_idx  = residual % kPredicatesPerByte;

			predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));
		}
	}

	NIHILUS_HOST_DEVICE void set_predicates(uint64_t thread_id, TensorCoord const& threadblock_offset) {
		TensorCoord residue_extent;
		if (kAdvanceRank) {
			typename TensorCoord::Index residue_size = (extent_[tag<kAdvanceRank>{}] - threadblock_offset.strided()) % Shape::kStrided;
			if (!residue_size) {
				residue_size = Shape::kStrided;
			}

			residue_offset_ = { 0, residue_size };
			residue_extent	= { extent_.contiguous(), min(threadblock_offset.strided() + residue_size, extent_.strided()) };
		} else {
			typename TensorCoord::Index residue_size = (extent_[kAdvanceRank] - threadblock_offset.contiguous()) % Shape::kContiguous;
			if (!residue_size) {
				residue_size = Shape::kContiguous;
			}

			residue_offset_ = { residue_size, 0 };

			residue_extent = { min(extent_.contiguous(), threadblock_offset.contiguous() + residue_size), extent_.strided() };
		}

		thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);

		compute_predicates_(residue_extent, false);

		set_iteration_index(0);
	}

	PredicatedTileAccessIteratorPredicates() = default;

	NIHILUS_HOST_DEVICE PredicatedTileAccessIteratorPredicates(TensorCoord extent) : extent_(extent) {
	}

	NIHILUS_HOST_DEVICE void set_iteration_index(uint64_t index) {
		iteration_vector_		 = index % kAccessesPerVector;
		uint64_t residual_access = index / kAccessesPerVector;

		iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
		iteration_strided_	  = residual_access / ThreadMap::Iterations::kContiguous;
	}

	NIHILUS_HOST_DEVICE PredicatedTileAccessIteratorPredicates& operator++() {
		return *this;
	}

	NIHILUS_HOST_DEVICE void clear_mask(bool enable = true) {
		NIHILUS_PRAGMA_UNROLL
		for (uint64_t i = 0; i < kPredicateWordCount; ++i) {
			predicates_[i] = enable ? 0u : predicates_[i];
		}
	}

	NIHILUS_HOST_DEVICE void enable_mask() {
		NIHILUS_PRAGMA_UNROLL
		for (uint64_t i = 0; i < kPredicateWordCount; ++i) {
			predicates_[i] = 0xffffffff;
		}
	}

	NIHILUS_HOST_DEVICE void set_mask(Mask const& mask) {
		NIHILUS_PRAGMA_UNROLL
		for (uint64_t i = 0; i < kPredicateWordCount; ++i) {
			predicates_[i] = mask[i];
		}
	}

	NIHILUS_HOST_DEVICE void get_mask(Mask& mask) {
		NIHILUS_PRAGMA_UNROLL
		for (uint64_t i = 0; i < kPredicateWordCount; ++i) {
			mask[i] = predicates_[i];
		}
	}

	NIHILUS_HOST_DEVICE bool valid() const {
		uint64_t pred_idx = iteration_vector_ + kAccessesPerVector * (iteration_contiguous_ + iteration_strided_ * ThreadMap::Iterations::kContiguous);

		uint64_t word_idx = pred_idx / kPredicatesPerWord;
		uint64_t residual = pred_idx % kPredicatesPerWord;
		uint64_t byte_idx = residual / kPredicatesPerByte;
		uint64_t bit_idx  = residual % kPredicatesPerByte;

		bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
		return pred;
	}
};

template<typename Permute> struct InversePermute {
	static_assert(!std::is_same<Permute, Permute>::value,
		"To apply permutation to a GEMM input operand (A or B), an inverse permutation for the desired "
		"permute class must be defined and enabled by specializing cutlass::InversePermute trait.");
};

template<> struct InversePermute<NoPermute> {
	using type = NoPermute;
};

struct PredicatedTileAccessIteratorDesc {
	uint64_t element_size_bits = -1;
	uint64_t advance_rank	   = -1;
	PitchLinearCoord threadblock_shape;
	PitchLinearCoord threadmap_iterations;
	PitchLinearCoord threadmap_delta;


	PredicatedTileAccessIteratorDesc() = default;

	NIHILUS_HOST_DEVICE PredicatedTileAccessIteratorDesc(uint64_t element_size_bits_, uint64_t advance_rank_, PitchLinearCoord threadblock_shape_,
		PitchLinearCoord threadmap_iterations_, PitchLinearCoord threadmap_delta_)
		: element_size_bits(element_size_bits_), advance_rank(advance_rank_), threadblock_shape(threadblock_shape_), threadmap_iterations(threadmap_iterations_),
		  threadmap_delta(threadmap_delta_) {
#if 0
    printf("PredicatedTileAccessIteratorDesc(%d, %d, {%d, %d}, {%d, %d}, {%d, %d}})\n",
      element_size_bits,
      advance_rank,
      threadblock_shape.contiguous(), threadblock_shape.strided(),
      threadmap_iterations.contiguous(), threadmap_iterations.strided(),
      threadmap_delta.contiguous(), threadmap_delta.strided());
#endif
	}
};

struct PredicatedTileAccessIteratorParams {
	using Index		= int64_t;
	using LongIndex = int64_t;

	LongIndex stride_	   = 0;
	LongIndex inc_strided_ = 0;
	LongIndex inc_next_	   = 0;
	LongIndex inc_advance_ = 0;


	NIHILUS_HOST_DEVICE Status initialize(LongIndex stride, PredicatedTileAccessIteratorDesc desc) {
		stride_ = stride;

		inc_strided_ = (LongIndex(stride_) * desc.threadmap_delta.strided()) * desc.element_size_bits / 8;

		if (desc.advance_rank) {
			inc_advance_ = desc.threadblock_shape.strided() * LongIndex(stride_) * desc.element_size_bits / 8;
		} else {
			inc_advance_ = desc.threadblock_shape.contiguous() * desc.element_size_bits / 8;
		}

		inc_next_ = inc_advance_ - LongIndex(desc.threadmap_iterations.strided() - 1) * desc.threadmap_delta.strided() * LongIndex(stride_) * desc.element_size_bits / 8;

		return Status::kSuccess;
	}

	PredicatedTileAccessIteratorParams() = default;

	NIHILUS_HOST_DEVICE PredicatedTileAccessIteratorParams(Index stride, PredicatedTileAccessIteratorDesc desc) {
		initialize(stride, desc);
	}
};

template<typename Shape> NIHILUS_HOST_DEVICE OutputTileShapeDesc make_OutputTileShapeDesc() {
	return OutputTileShapeDesc(Shape::kColumn, Shape::kRow, Shape::kGroup, Shape::kCluster, Shape::kTile);
}

template<typename ThreadMap> NIHILUS_HOST_DEVICE OutputTileThreadMapDesc make_OutputTileThreadMapDesc() {
	return OutputTileThreadMapDesc(ThreadMap::kThreads, ThreadMap::kElementsPerAccess, make_OutputTileShapeDesc<typename ThreadMap::Shape>(),
		make_OutputTileShapeDesc<typename ThreadMap::Iterations>(), make_OutputTileShapeDesc<typename ThreadMap::Delta>(), make_OutputTileShapeDesc<typename ThreadMap::Count>());
}

template<typename Permute> inline bool constexpr is_trivial_permute = std::is_same<Permute, NoPermute>::value;

template<typename Shape, typename Element, typename Layout, uint64_t AdvanceRank, typename ThreadMap> struct MakePredicatedTileAccessIteratorDesc;

template<typename Shape, typename Element, uint64_t AdvanceRank, typename ThreadMap>
struct MakePredicatedTileAccessIteratorDesc<Shape, Element, PitchLinear, AdvanceRank, ThreadMap> {
	NIHILUS_HOST_DEVICE
	PredicatedTileAccessIteratorDesc operator()() {
		return PredicatedTileAccessIteratorDesc(sizeof_bits<Element>::value, AdvanceRank, { Shape::kContiguous, Shape::kStrided },
			{ ThreadMap::Iterations::kContiguous, ThreadMap::Iterations::kStrided }, { ThreadMap::Delta::kContiguous, ThreadMap::Delta::kStrided });
	}
};

template<typename Shape, typename Element, uint64_t AdvanceRank, typename ThreadMap>
struct MakePredicatedTileAccessIteratorDesc<Shape, Element, RowMajor, AdvanceRank, ThreadMap> {
	static constexpr uint64_t kAdvanceRank = AdvanceRank;

	using UnderlyingMakeOperator =
		MakePredicatedTileAccessIteratorDesc<PitchLinearShape<Shape::kColumn, Shape::kRow>, Element, PitchLinear, (kAdvanceRank == 0 ? 1 : 0), ThreadMap>;

	NIHILUS_HOST_DEVICE
	PredicatedTileAccessIteratorDesc operator()() {
		return UnderlyingMakeOperator()();
	}
};

template<typename Element> NIHILUS_HOST_DEVICE int64_t OffsetBytes(int64_t index) {
	static_assert((sizeof_bits<Element>::value >= 8 && !(sizeof_bits<Element>::value % 8)) || (sizeof_bits<Element>::value < 8 && !(8 % sizeof_bits<Element>::value)),
		"Size of numeric type in bits must either be divisible by 8 bits, or 8 bits must be divisible by the size.");

	if (sizeof_bits<Element>::value >= 8) {
		return index * (sizeof_bits<Element>::value / 8);
	} else {
		int const kElementsPerByte = ((8 / sizeof_bits<Element>::value) + ((sizeof_bits<Element>::value >= 8) ? 1 : 0));
		return index / kElementsPerByte;
	}
}

NIHILUS_HOST_DEVICE int64_t OffsetBytes(int64_t index, int64_t element_sizeof_bits) {
	if (element_sizeof_bits >= 8) {
		return index * (element_sizeof_bits / 8);
	} else {
		int64_t const kElementsPerByte = ((8 / element_sizeof_bits) + ((element_sizeof_bits >= 8) ? 1 : 0));
		return index / kElementsPerByte;
	}
}

template<typename Shape, typename Element, typename Layout, uint64_t AdvanceRank, typename ThreadMap, typename AccessType, bool Gather = false, typename PermuteLayout = NoPermute>
class PredicatedTileAccessIterator;


template<typename Shape_, typename Element_, uint64_t AdvanceRank, typename ThreadMap_, typename AccessType_, bool Gather, typename PermuteLayout>
class PredicatedTileAccessIterator<Shape_, Element_, PitchLinear, AdvanceRank, ThreadMap_, AccessType_, Gather, PermuteLayout> {
  public:
	static_assert(AdvanceRank == 0 || AdvanceRank == 1,
		"Specialization for pitch-linear iterator may along advance along the "
		"contiguous(rank=0) or strided(rank=1) dimension.");

	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = PitchLinear;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;
	using AccessType					   = AccessType_;

	using Index		= typename Layout::Index;
	using LongIndex = typename Layout::LongIndex;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorView  = TensorView<Element, Layout>;
	using TensorCoord = typename Layout::TensorCoord;

	using Pointer		  = Element*;
	using NonConstPointer = typename std::remove_const<Element>::type*;

	using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<Shape, Element, Layout, AdvanceRank, ThreadMap, AccessType>;

	static constexpr uint64_t kAccessesPerVector = ThreadMap::kElementsPerAccess / AccessType::size_val;

	static_assert(!(ThreadMap::kElementsPerAccess % AccessType::size_val), "Vectors implied by the thread map must be divisible by the access type.");

	static bool constexpr Permute = !std::is_same<PermuteLayout, NoPermute>::value && !std::is_same<PermuteLayout, InversePermute<NoPermute>>::value;

	using Mask = typename UnderlyingPredicates::Mask;

	struct Params : PredicatedTileAccessIteratorParams {
		using Base = PredicatedTileAccessIteratorParams;

		Params() = default;

		NIHILUS_HOST_DEVICE Params(Layout const& layout) : Base(layout.stride(0), MakePredicatedTileAccessIteratorDesc<Shape, Element, Layout, kAdvanceRank, ThreadMap>()()) {
		}

		NIHILUS_HOST_DEVICE Params(Base const& base) : Base(base) {
		}
	};

  public:
	using BytePointer = char*;

  public:
	UnderlyingPredicates the_predicates;

	Params params_;

	BytePointer pointer_;

	bool is_residue_tile_;


	uint64_t const* indices_;

	PermuteLayout permute_layout_;

	TensorCoord coord_offset_;

  public:
	NIHILUS_DEVICE void compute_predicates_(TensorCoord extent, bool is_steady_state = false) {
		the_predicates.compute_predicates_(extent, is_steady_state);
	}

  public:
	PredicatedTileAccessIterator() = default;

	NIHILUS_HOST_DEVICE PredicatedTileAccessIterator(Params const& params, Pointer pointer, TensorCoord extent, uint64_t thread_id, TensorCoord const& threadblock_offset,
		uint64_t const* indices = nullptr)
		: params_(params), pointer_(reinterpret_cast<BytePointer>(const_cast<NonConstPointer>(pointer))), the_predicates(extent), is_residue_tile_(true), indices_(indices),
		  permute_layout_(TensorCoord(extent.contiguous(), extent.strided()), params.stride_) {
		the_predicates.set_predicates(thread_id, threadblock_offset);

		Layout layout(params_.stride_);

		if (!Gather && !Permute) {
			add_pointer_offset(layout({ the_predicates.thread_offset_ }));
		} else {
			coord_offset_ = the_predicates.thread_offset_;
			if (!Permute) {
				add_pointer_offset(layout({ coord_offset_.contiguous(), 0 }));
			}
		}
	}

	NIHILUS_HOST_DEVICE PredicatedTileAccessIterator(Params const& params, Pointer pointer, TensorCoord extent, uint64_t thread_id)
		: PredicatedTileAccessIterator(params, pointer, extent, thread_id, { 0, 0 }) {
	}

	NIHILUS_HOST_DEVICE void set_iteration_index(uint64_t index) {
		the_predicates.set_iteration_index(index);
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
	}

	NIHILUS_DEVICE void add_tile_offset(TensorCoord const& tile_offset) {
		if (is_residue_tile_) {
			the_predicates.thread_offset_ += the_predicates.residue_offset_;

			the_predicates.compute_predicates_(the_predicates.extent_, true);

			Layout layout(params_.stride_);

			if (!Gather && !Permute) {
				add_pointer_offset(layout(the_predicates.residue_offset_));

				if (kAdvanceRank) {
					pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided() - 1);
					pointer_ += Shape::kContiguous * tile_offset.contiguous() * sizeof_bits<Element>::value / 8;
				} else {
					pointer_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous() - 1);
					pointer_ += Shape::kStrided * tile_offset.strided() * sizeof_bits<Element>::value / 8;
				}
			} else {
				coord_offset_.strided() = the_predicates.thread_offset_.strided() + Shape::kStrided * (tile_offset.strided() - kAdvanceRank);
				if (!Permute) {
					add_pointer_offset(layout(PitchLinearCoord{ Coord<2>{ (the_predicates.residue_offset_.contiguous(), 0) } }));
					add_pointer_offset(Shape::kContiguous * (tile_offset.contiguous() - (1 - kAdvanceRank)));
				} else {
					coord_offset_.contiguous() = the_predicates.thread_offset_.contiguous() + Shape::kContiguous * (tile_offset.contiguous() - (1 - kAdvanceRank));
				}
			}
		} else {
			if (!Gather && !Permute) {
				if (kAdvanceRank) {
					pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
					pointer_ += Shape::kContiguous * tile_offset.contiguous();
				} else {
					pointer_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous());
					pointer_ += Shape::kStrided * tile_offset.strided();
				}
			} else {
				coord_offset_.strided() += Shape::kStrided * tile_offset.strided();
				if (!Permute) {
					add_pointer_offset(Shape::kContiguous * tile_offset.contiguous());
				} else {
					coord_offset_.contiguous() += Shape::kContiguous * tile_offset.contiguous();
				}
			}
		}

		is_residue_tile_ = false;
	}

	NIHILUS_HOST_DEVICE AccessType* get() const {
		if (Gather || Permute) {
			if (!valid()) {
				return nullptr;
			}

			Index coord_contig = (Permute ? coord_offset_.contiguous() : 0) + the_predicates.iteration_contiguous_ * ThreadMap::Delta::kContiguous +
				the_predicates.iteration_vector_ * AccessType::size_val;
			Index coord_strided = coord_offset_.strided() + the_predicates.iteration_strided_ * ThreadMap::Delta::kStrided;
			if (Gather) {
				coord_strided = indices_[coord_strided];
			}

			LongIndex offset = Permute ? permute_layout_(TensorCoord(coord_contig, coord_strided)) : (coord_strided * LongIndex(params_.stride_) + coord_contig);
			return reinterpret_cast<AccessType*>(pointer_ + OffsetBytes<Element>(offset));
		}

		return reinterpret_cast<AccessType*>(pointer_ + the_predicates.iteration_contiguous_ * (ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value) / 8) +
			the_predicates.iteration_vector_;
	}

	NIHILUS_HOST_DEVICE PredicatedTileAccessIterator& operator++() {
		the_predicates.operator++();

		++the_predicates.iteration_vector_;
		if (the_predicates.iteration_vector_ < kAccessesPerVector) {
			return *this;
		}

		the_predicates.iteration_vector_ = 0;
		++the_predicates.iteration_contiguous_;

		if (the_predicates.iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
			return *this;
		}

		the_predicates.iteration_contiguous_ = 0;
		++the_predicates.iteration_strided_;

		if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
			if (!Gather && !Permute) {
				pointer_ += params_.inc_strided_;
			}

			return *this;
		}

		the_predicates.iteration_strided_ = 0;

		if (!Gather && !Permute) {
			pointer_ += params_.inc_next_;

			pointer_ -= params_.inc_advance_;
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE PredicatedTileAccessIterator operator++(int32_t) {
		PredicatedTileAccessIterator self(*this);
		operator++();
		return self;
	}

	NIHILUS_HOST_DEVICE void clear_mask(bool enable = true) {
		the_predicates.clear_mask(enable);
	}

	NIHILUS_HOST_DEVICE void enable_mask() {
		the_predicates.enable_mask();
	}

	NIHILUS_HOST_DEVICE void set_mask(Mask const& mask) {
		the_predicates.set_mask(mask);
	}

	NIHILUS_HOST_DEVICE void get_mask(Mask& mask) {
		the_predicates.get_mask(mask);
	}

	NIHILUS_HOST_DEVICE bool valid() const {
		return the_predicates.valid();
	}
};

class TensorNDHWC {
  public:
	static constexpr uint64_t kRank = 5;

	static constexpr uint64_t kStrideRank = 4;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = Tensor5DCoord;

	using Stride = constexpresh_coord<kStrideRank>;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE TensorNDHWC(Stride const& stride = Stride(0)) : stride_(stride) {
	}

	NIHILUS_HOST_DEVICE TensorNDHWC(typename Stride::Index c, typename Stride::Index wc, typename Stride::Index hwc, typename Stride::Index dhwc) : stride_(c, wc, hwc, dhwc) {
	}

	NIHILUS_HOST_DEVICE static TensorNDHWC packed(TensorCoord const& extent) {
		return TensorNDHWC(extent.c(), extent.w() * extent.c(), extent.h() * extent.w() * extent.c(), extent.d() * extent.h() * extent.w() * extent.c());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(TensorCoord const& coord) const {
		return coord.c() + LongIndex(stride_[0] * coord.w()) + LongIndex(stride_[1] * coord.h()) + LongIndex(stride_[2] * coord.d()) + LongIndex(stride_[3] * coord.n());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(PitchLinearCoord coord) const {
		return coord.contiguous() + LongIndex(coord.strided() * stride_[3]);
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE Stride& stride() {
		return stride_;
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(TensorCoord const& extent) const {
		return extent.n() * stride_[3];
	}
};

template<typename value_t> NIHILUS_HOST_DEVICE constexpr value_t clz(value_t x) {
	for (uint64_t i = 31; i >= 0; --i) {
		if ((1 << i) & x)
			return value_t(31 - i);
	}
	return value_t(32);
}

template<typename value_t> NIHILUS_HOST_DEVICE constexpr value_t find_log2(value_t x) {
	uint64_t a = uint64_t(31 - clz(x));
	a += (x & (x - 1)) != 0;
	return a;
}

NIHILUS_HOST_DEVICE constexpr void find_divisor(uint64_t& mul, uint64_t& shr, uint64_t denom) {
	if (denom == 1) {
		mul = 0;
		shr = 0;
	} else {
		uint64_t p = 31 + find_log2(denom);
		unsigned m = unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

		mul = m;
		shr = p - 32;
	}
}

NIHILUS_HOST_DEVICE constexpr void fast_divmod(uint64_t& quo, uint64_t& rem, uint64_t src, uint64_t div, uint64_t mul, uint64_t shr) {
#if defined(__CUDA_ARCH__)
	quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
	quo = uint64_t((div != 1) ? uint64_t((( int64_t )src * mul) >> 32) >> shr : src);
#endif

	rem = src - (quo * div);
}

class TensorNHWC {
  public:
	static constexpr uint64_t kRank = 4;

	static constexpr uint64_t kStrideRank = 3;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = Tensor4DCoord;

	using Stride = constexpresh_coord<kStrideRank>;

  public:
	Stride stride_;

  public:
	NIHILUS_HOST_DEVICE TensorNHWC(Stride const& stride = Stride(0)) : stride_(stride) {
	}

	NIHILUS_HOST_DEVICE TensorNHWC(typename Stride::Index stride_w, typename Stride::Index stride_h, typename Stride::Index stride_n) : stride_(stride_w, stride_h, stride_n) {
	}

	NIHILUS_HOST_DEVICE static TensorNHWC packed(TensorCoord const& extent) {
		return TensorNHWC(extent.c(), extent.w() * extent.c(), extent.h() * extent.w() * extent.c());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(TensorCoord const& coord) const {
		return coord.c() + LongIndex(stride_[0] * coord.w()) + LongIndex(stride_[1] * coord.h()) + LongIndex(stride_[2] * coord.n());
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(PitchLinearCoord coord) const {
		return coord.contiguous() + LongIndex(coord.strided() * stride_[2]);
	}

	NIHILUS_HOST_DEVICE TensorCoord inverse(LongIndex index) const {
		uint64_t n = 0, h = 0, w = 0, c = 0;

#if defined(__CUDA_ARCH__)
		uint64_t tmp = 0;
		c			 = uint64_t(index % static_cast<uint64_t>(stride_[0]));

		uint64_t hw_mul, hw_shr, w_mul, w_shr, c_mul, c_shr;

		find_divisor(hw_mul, hw_shr, stride_[2]);
		find_divisor(w_mul, w_shr, stride_[1]);
		find_divisor(c_mul, c_shr, stride_[0]);

		fast_divmod(n, tmp, index, uint64_t(stride_[2]), hw_mul, hw_shr);
		fast_divmod(h, w, tmp, uint64_t(stride_[1]), w_mul, w_shr);
		fast_divmod(w, tmp, w, uint64_t(stride_[0]), c_mul, c_shr);
#else

		n				   = uint64_t(index / stride_[2]);
		LongIndex residual = index % stride_[2];

		h		 = uint64_t(residual / stride_[1]);
		residual = (residual % stride_[1]);

		w = uint64_t(residual / stride_[0]);
		c = uint64_t(residual % stride_[0]);

#endif
		return TensorCoord(n, h, w, c);
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return stride_;
	}

	NIHILUS_HOST_DEVICE Stride& stride() {
		return stride_;
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(TensorCoord const& extent) const {
		return extent.n() * stride_[2];
	}
};

struct FastDivmod {
	using value_div_type = uint64_t;
	using value_mod_type = int64_t;
	int64_t divisor		 = 1;
	uint32_t multiplier	 = 0u;
	uint32_t shift_right = 0u;

	NIHILUS_HOST_DEVICE void fast_divmod(uint64_t& quotient, uint64_t& remainder, uint64_t dividend) const {
#if defined(__CUDA_ARCH__)
		quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
		quotient = uint64_t((divisor != 1) ? uint64_t((( int64_t )dividend * multiplier) >> 32) >> shift_right : dividend);
#endif

		remainder = dividend - (quotient * divisor);
	}

	NIHILUS_HOST_DEVICE void fast_divmod(uint64_t& quotient, int64_t& remainder, int64_t dividend) const {
#if defined(__CUDA_ARCH__)
		quotient = (divisor != 1) ? __umulhi(dividend, multiplier) >> shift_right : dividend;
#else
		quotient = uint64_t((divisor != 1) ? ((dividend * multiplier) >> 32) >> shift_right : dividend);
#endif
		remainder = dividend - (quotient * divisor);
	}



	constexpr FastDivmod() = default;

	NIHILUS_HOST_DEVICE FastDivmod(uint64_t divisor_) : divisor(divisor_) {
		if (divisor != 1) {
			uint64_t p = 31 + find_log2(divisor);
			unsigned m = unsigned(((1ull << p) + unsigned(divisor) - 1) / unsigned(divisor));

			multiplier	= m;
			shift_right = p - 32;
		}
	}

	NIHILUS_HOST_DEVICE void operator()(uint64_t& quotient, uint64_t& remainder, uint64_t dividend) const {
		fast_divmod(quotient, remainder, dividend);
	}

	NIHILUS_HOST_DEVICE uint64_t div(uint64_t dividend) const {
		uint64_t quotient, remainder;
		fast_divmod(quotient, remainder, dividend);
		return quotient;
	}

	NIHILUS_HOST_DEVICE uint64_t divide(uint64_t dividend) const {
		return div(dividend);
	}

	NIHILUS_HOST_DEVICE uint64_t rem(uint64_t dividend) const {
		uint64_t quotient, remainder;
		fast_divmod(quotient, remainder, dividend);
		return remainder;
	}

	NIHILUS_HOST_DEVICE uint64_t remainder(uint64_t dividend) const {
		return rem(dividend);
	}

	NIHILUS_HOST_DEVICE uint64_t divmod(uint64_t& remainder, uint64_t dividend) const {
		uint64_t quotient;
		fast_divmod(quotient, remainder, dividend);
		return quotient;
	}

	NIHILUS_HOST_DEVICE void operator()(uint64_t& quotient, int64_t& remainder, int64_t dividend) const {
		fast_divmod(quotient, remainder, dividend);
	}

	NIHILUS_HOST_DEVICE uint64_t divmod(int64_t& remainder, int64_t dividend) const {
		uint64_t quotient;
		fast_divmod(quotient, remainder, dividend);
		return quotient;
	}

	NIHILUS_HOST_DEVICE operator uint64_t() const {
		return divisor;
	}
};

template<typename ThreadMap_, typename Element_, bool ScatterD = false, typename PermuteDLayout = NoPermute, bool UseCUDAStore = false, uint64_t Rank = 4>
class PredicatedTileIteratorConv {
  public:
	using ThreadMap = ThreadMap_;
	using Shape		= typename ThreadMap::Shape;

	using Element = Element_;

	static constexpr uint64_t kRank = Rank;
	using Layout					= typename std::conditional<kRank == 4, TensorNHWC, TensorNDHWC>::type;

	using Stride						  = typename Layout::Stride;
	static constexpr uint64_t kStrideRank = Layout::kStrideRank;

	using TensorRef		 = TensorRef<Element, Layout>;
	using ConstTensorRef = typename TensorRef::ConstTensorRef;

	using MappedLayout = RowMajor;
	using Index		   = typename MappedLayout::Index;
	using LongIndex	   = typename MappedLayout::LongIndex;
	using TensorCoord  = typename MappedLayout::TensorCoord;

	static constexpr uint64_t kElementsPerAccess = ThreadMap::kElementsPerAccess;
	static constexpr uint64_t kThreads			 = ThreadMap::kThreads;
	static constexpr uint64_t kIterations		 = ThreadMap::Count::kTile;

	static bool constexpr PermuteD = !is_trivial_permute<PermuteDLayout>;

	static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
	static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
	static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
	static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

	using Fragment = nihilus::array<Element,
		ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

	using AccessType = nihilus::array<Element, ThreadMap::kElementsPerAccess>;


	struct Params : PredicatedTileIteratorParams {
		using Base = PredicatedTileIteratorParams;

		FastDivmod divmod[kStrideRank - 1];
		Stride tensor_stride;

		NIHILUS_HOST_DEVICE Params() {
		}

		NIHILUS_HOST_DEVICE Params(Layout const& layout, Tensor4DCoord const& tensor_extent)
			: PredicatedTileIteratorParams(layout.stride()[0] * uint64_t(sizeof(AccessType)) / kElementsPerAccess, make_OutputTileThreadMapDesc<ThreadMap>()) {
			divmod[0] = FastDivmod(tensor_extent[2] /* Q for Fprop & W for Deconv*/);
			divmod[1] = FastDivmod(tensor_extent[1] /* P for Fprop & H for Deconv*/);

#pragma unroll
			for (uint64_t i = 0; i < kStrideRank; ++i) {
				tensor_stride[i] = layout.stride()[i];
			}
		}

		NIHILUS_HOST_DEVICE Params(Layout const& layout, Tensor5DCoord const& tensor_extent)
			: PredicatedTileIteratorParams(layout.stride()[0] * uint64_t(sizeof(AccessType)) / kElementsPerAccess, make_OutputTileThreadMapDesc<ThreadMap>()) {
			divmod[0] = FastDivmod(tensor_extent[3] /* Q for Fprop & W for Deconv*/);
			divmod[1] = FastDivmod(tensor_extent[2] /* P for Fprop & H for Deconv*/);
			divmod[2] = FastDivmod(tensor_extent[1] /* Z for Fprop & D for Deconv*/);

#pragma unroll
			for (uint64_t i = 0; i < kStrideRank; ++i) {
				tensor_stride[i] = layout.stride()[i];
			}
		}

		NIHILUS_HOST_DEVICE Params(Base const& base) : Base(base) {
		}
	};

	struct Mask {
		static constexpr uint64_t kCount = ThreadMap::Iterations::kColumn;

		bool predicates[kCount];

		NIHILUS_HOST_DEVICE Mask() {
			enable();
		}

		NIHILUS_HOST_DEVICE void clear() {
#pragma unroll
			for (uint64_t i = 0; i < kCount; ++i) {
				predicates[i] = false;
			}
		}

		NIHILUS_DEVICE void enable() {
#pragma unroll
			for (uint64_t i = 0; i < kCount; ++i) {
				predicates[i] = true;
			}
		}
	};

  public:
	Params params_;

	uint8_t* byte_pointer_;

	Mask mask_;

	Index extent_row_;

	Index extent_column_;

	Index thread_start_row_;

	Index thread_start_column_;

	uint64_t state_[3];


	static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
	static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
	static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

  public:
  public:
	NIHILUS_DEVICE PredicatedTileIteratorConv(Params const& params, Element* pointer, TensorCoord extent, uint64_t thread_idx, TensorCoord threadblock_offset = TensorCoord())
		: params_(params) {
		TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

		extent_row_	   = extent.row();
		extent_column_ = extent.column();

		thread_start_row_	 = thread_offset.row();
		thread_start_column_ = thread_offset.column();

#pragma unroll
		for (uint64_t c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
			mask_.predicates[c] = ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
		}

		if (!pointer) {
			mask_.clear();
		}

		byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) + LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

		state_[0] = state_[1] = state_[2] = 0;
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
	}

	NIHILUS_DEVICE void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const {
		uint8_t* byte_pointer = byte_pointer_;
		AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
#pragma unroll
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
#pragma unroll
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					uint64_t row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

					bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

					AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

					Stride tensor_coord = CoordinateDecompositionLittleEndian<kStrideRank>(row_offset + thread_start_row_, params_.divmod);

					LongIndex tensor_offset = dot(tensor_coord, params_.tensor_stride);

#pragma unroll
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						bool guard = row_guard && mask_.predicates[column];

						global_load<AccessType, sizeof(AccessType)>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
							( void* )&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess + tensor_offset / kElementsPerAccess], guard);
					}
				}
			}
		}
	}

	NIHILUS_DEVICE void load(Fragment& frag) const {
		load_with_byte_offset(frag, 0);
	}

	NIHILUS_DEVICE void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const {
		uint8_t* byte_pointer	   = byte_pointer_;
		AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

#pragma unroll
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
#pragma unroll
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
#pragma unroll
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					uint64_t row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

					bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

					Stride tensor_coord = CoordinateDecompositionLittleEndian<kStrideRank>((row_offset + thread_start_row_), params_.divmod);

					LongIndex tensor_offset = dot(tensor_coord, params_.tensor_stride);

					AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

#pragma unroll
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						bool guard = row_guard && mask_.predicates[column];

						if (UseCUDAStore) {
							if (guard) {
								memory_pointer[tensor_offset / kElementsPerAccess] = frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
							}
						} else {
							global_store<AccessType, sizeof(AccessType)>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
								( void* )&memory_pointer[tensor_offset / kElementsPerAccess], guard);
						}

						memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
					}
				}
			}
		}
	}

	NIHILUS_DEVICE void store(Fragment const& frag) const {
		store_with_byte_offset(frag, 0);
	}

	NIHILUS_DEVICE MatrixCoord thread_start() const {
		return MatrixCoord(thread_start_row_, thread_start_column_);
	}

	NIHILUS_DEVICE int64_t thread_start_row() const {
		return thread_start_row_;
	}

	NIHILUS_DEVICE int64_t thread_start_column() const {
		return thread_start_column_;
	}

	NIHILUS_DEVICE Index extent_row() const {
		return extent_row_;
	}

	NIHILUS_DEVICE Index extent_column() const {
		return extent_column_;
	}

	NIHILUS_HOST_DEVICE PredicatedTileIteratorConv& operator++() {
		++state_[0];

		thread_start_row_ += ThreadMap::Shape::kRow;

		if (state_[0] == ThreadMap::Count::kRow) {
			state_[0] = 0;
			++state_[1];

			thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

			if (state_[1] == ThreadMap::Count::kGroup) {
				state_[1] = 0;
				++state_[2];

				thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

				if (state_[2] == ThreadMap::Count::kCluster) {
					state_[2] = 0;

					thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
				}
			}
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE PredicatedTileIteratorConv& operator+=(uint64_t increment) {
		state_[0] += increment;
		uint64_t increment_row = state_[0] / ThreadMap::Count::kRow;
		state_[0]			   = state_[0] % ThreadMap::Count::kRow;

		thread_start_row_ += (ThreadMap::Shape::kRow * increment);

		state_[1] += increment_row;
		uint64_t increment_group = state_[1] / ThreadMap::Count::kGroup;
		state_[1]				 = state_[1] % ThreadMap::Count::kGroup;

		thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow * increment_row;

		state_[2] += increment_group;
		uint64_t increment_cluster = state_[2] / ThreadMap::Count::kCluster;
		state_[2]				   = state_[2] % ThreadMap::Count::kCluster;

		thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow * increment_group;

		thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile * increment_cluster;

		return *this;
	}

	NIHILUS_DEVICE void clear_mask() {
		mask_.clear();
	}

	NIHILUS_DEVICE void enable_mask() {
		mask_.enable();
	}

	NIHILUS_DEVICE void get_mask(Mask& mask) const {
		mask = mask_;
	}

	NIHILUS_DEVICE void set_mask(Mask const& mask) {
		mask_ = mask;
	}
};

template<typename WarpShape, typename Operator, typename Layout, typename MmaSimtPolicy> struct SimtPolicy;


template<typename WarpShape_, typename Operator_, typename MmaSimtPolicy_> struct SimtPolicy<WarpShape_, Operator_, RowMajor, MmaSimtPolicy_> {
	using WarpShape		= WarpShape_;
	using Operator		= Operator_;
	using MmaSimtPolicy = MmaSimtPolicy_;

	static_assert(!(WarpShape::kM % MmaSimtPolicy::WarpShape::kRow), "Divisibility");
	static_assert(!(WarpShape::kN % MmaSimtPolicy::WarpShape::kColumn), "Divisibility");

	static constexpr uint64_t kIterations = WarpShape::kM / MmaSimtPolicy::WarpShape::kRow;

	static constexpr uint64_t kElementsPerIteration = (WarpShape::kN / MmaSimtPolicy::WarpShape::kColumn);

	static constexpr uint64_t kAccumulatorElementCount = kElementsPerIteration * kIterations;

	static constexpr uint64_t kElementsPerAccess = MmaSimtPolicy::LaneMmaShape::kN;

	static constexpr uint64_t kRowsPerIteration = MmaSimtPolicy::WarpShape::kRow;

	static constexpr uint64_t kAccessesPerIteration = kElementsPerIteration / kElementsPerAccess;

	using Delta = MatrixShape<MmaSimtPolicy::WarpShape::kRow * MmaSimtPolicy::LaneMmaShape::kM, MmaSimtPolicy::WarpShape::kColumn * MmaSimtPolicy::LaneMmaShape::kN>;
};

template<typename WarpShape, typename Operator, typename Layout, typename MmaSimtPolicy> class FragmentIteratorSimt;


template<typename WarpShape_, typename Operator_, typename MmaSimtPolicy_> class FragmentIteratorSimt<WarpShape_, Operator_, RowMajor, MmaSimtPolicy_> {
  public:
	using WarpShape = WarpShape_;
	using Operator	= Operator_;
	using Layout	= RowMajor;

	using Policy = SimtPolicy<WarpShape, Operator, Layout, MmaSimtPolicy_>;

	using Fragment = nihilus::array<typename Operator::ElementC, Policy::kElementsPerIteration>;

	using AccumulatorTile = nihilus::array<typename Operator::ElementC, Policy::kAccumulatorElementCount>;

	using OutputAccumulatorTile = AccumulatorTile;

	static constexpr uint64_t kIterations = Policy::kIterations;

  public:
	using AccessType = nihilus::array<typename Operator::ElementC, Policy::kElementsPerAccess>;

  public:
	AccessType const* accumulators_;

	uint64_t index_;

  public:
	NIHILUS_HOST_DEVICE FragmentIteratorSimt(AccumulatorTile const& accum) : accumulators_(reinterpret_cast<AccessType const*>(&accum)), index_(0) {
	}

	NIHILUS_HOST_DEVICE FragmentIteratorSimt& operator++() {
		++index_;
		return *this;
	}

	NIHILUS_HOST_DEVICE FragmentIteratorSimt& operator--() {
		--index_;
		return *this;
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag, uint64_t index_offset = 0) const {
		AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
		for (uint64_t n = 0; n < Policy::kAccessesPerIteration; ++n) {
			uint64_t accumulator_access_offset = index_ * Policy::kAccessesPerIteration + n;

			frag_ptr[n] = accumulators_[accumulator_access_offset];
		}
	}
};

template<typename WarpShape, typename Operator, typename Element, typename Layout, typename MmaSimtPolicy> class TileIteratorSimt;


template<typename WarpShape_, typename Operator_, typename Element_, typename MmaSimtPolicy_> class TileIteratorSimt<WarpShape_, Operator_, Element_, RowMajor, MmaSimtPolicy_> {
  public:
	using WarpShape = WarpShape_;
	using Operator	= Operator_;
	using Element	= Element_;
	using Layout	= RowMajor;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorCoord = MatrixCoord;
	using Index		  = typename TensorRef::Index;
	using LongIndex	  = typename TensorRef::LongIndex;

	using Policy = SimtPolicy<WarpShape, Operator, Layout, MmaSimtPolicy_>;

	using Shape = MatrixShape<Policy::kRowsPerIteration, WarpShape::kN>;

	using Fragment = nihilus::array<typename Operator::ElementC, Policy::kElementsPerIteration>;

	using AccumulatorTile = nihilus::array<typename Operator::ElementC, Policy::kAccumulatorElementCount>;

	static constexpr uint64_t kIterations = Policy::kIterations;

	using Padding = MatrixShape<0,
		4 * Policy::kElementsPerAccess
#if NIHILUS_SIMT_EPILOGUE_USE_SCALAR_STORES
			+ 1
#endif
		>;

  public:
#if NIHILUS_SIMT_EPILOGUE_USE_SCALAR_STORES
	using AccessType = nihilus::array<Element, 1>;

#else
	using AccessType = nihilus::array<Element, Policy::kElementsPerAccess>;
#endif


	AccessType* pointer_;

	Layout layout_;

  public:
	NIHILUS_HOST_DEVICE TileIteratorSimt() : pointer_(nullptr) {
	}

	NIHILUS_HOST_DEVICE TileIteratorSimt(TensorRef const& ref, unsigned lane_id)
		: pointer_(reinterpret_cast<AccessType*>(ref.data())), layout_(ref.stride()[0] / AccessType::size_val) {
		auto lane_layout		= Policy::MmaSimtPolicy::get_lane_layout();
		MatrixCoord lane_offset = lane_layout.inverse(lane_id);

		pointer_ += layout_({ lane_offset.row(), lane_offset.column() * Policy::kElementsPerAccess / uint64_t(AccessType::size_val) });
	}

	NIHILUS_HOST_DEVICE TileIteratorSimt& add_pointer_offset(Index pointer_offset) {
		pointer_ += pointer_offset / AccessType::size_val;
		return *this;
	}

	NIHILUS_HOST_DEVICE TileIteratorSimt& add_tile_offset(TensorCoord const& tile_offset) {
		pointer_ += layout_({ tile_offset.row() * Shape::kRow, (tile_offset.column() * Shape::kColumn / uint64_t(AccessType::size_val)) });

		return *this;
	}

	NIHILUS_HOST_DEVICE TileIteratorSimt& operator+=(TensorCoord const& tile_offset) {
		add_tile_offset(tile_offset);

		return *this;
	}

	NIHILUS_HOST_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
#if NIHILUS_SIMT_EPILOGUE_USE_SCALAR_STORES
		using ScalarAccessType				  = nihilus::array<Element, 1>;
		ScalarAccessType const* scalarFragPtr = reinterpret_cast<ScalarAccessType const*>(&frag);
		ScalarAccessType* scalarPointer		  = reinterpret_cast<ScalarAccessType*>(pointer_) + pointer_offset;

	#pragma unroll
		for (uint64_t n = 0; n < Policy::kAccessesPerIteration; ++n) {
	#pragma unroll
			for (uint64_t s = 0; s < Policy::kElementsPerAccess; s++) {
				scalarPointer[n * Policy::MmaSimtPolicy::WarpShape::kColumn * Policy::kElementsPerAccess + s] = scalarFragPtr[n * Policy::kElementsPerAccess + s];
			}
		}
#else
		AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);
	#pragma unroll
		for (uint64_t n = 0; n < Policy::kAccessesPerIteration; ++n) {
			pointer_[n * Policy::MmaSimtPolicy::WarpShape::kColumn + pointer_offset / uint64_t(AccessType::size_val)] = frag_ptr[n];
		}
#endif
	}

	NIHILUS_HOST_DEVICE void store(Fragment const& frag) {
		store_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
		AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

#pragma unroll
		for (uint64_t n = 0; n < Policy::kAccessesPerIteration; ++n) {
			frag_ptr[n] = pointer_[n * Policy::MmaSimtPolicy::WarpShape::kColumn + pointer_offset / uint64_t(AccessType::size_val)];
		}
	}

	NIHILUS_HOST_DEVICE void load(Fragment& frag) const {
		load_with_pointer_offset(frag, 0);
	}

	NIHILUS_HOST_DEVICE void set_smem_base_address(Index address) {
	}
};

template<typename ThreadMap_, typename Element_, uint64_t MaxAlignment = ThreadMap_::kElementsPerAccess * sizeof_bits<Element_>::value / 8> class SharedLoadIterator {
  public:
	using ThreadMap = ThreadMap_;
	using Shape		= typename ThreadMap::TileShape;

	using Element = Element_;

	using Layout		 = RowMajor;
	using TensorRef		 = TensorRef<Element, Layout>;
	using ConstTensorRef = typename TensorRef::ConstTensorRef;

	using Index		  = typename Layout::Index;
	using LongIndex	  = typename Layout::LongIndex;
	using TensorCoord = MatrixCoord;

	static constexpr uint64_t kElementsPerAccess = ThreadMap::kElementsPerAccess;

	static constexpr uint64_t kMinAlignment = ThreadMap_::kElementsPerAccess * sizeof_bits<Element_>::value / 8;

	static constexpr uint64_t kAlignment = (MaxAlignment < kMinAlignment ? MaxAlignment : kMinAlignment);

	static constexpr uint64_t kThreads = ThreadMap::kThreads;

	using Fragment = nihilus::array<Element,
		ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

	using AccessType = nihilus::array<Element, ThreadMap::kElementsPerAccess>;

	using LoadType = nihilus::array<Element, const_min(128 / sizeof_bits<Element>::value, ThreadMap::kElementsPerAccess)>;

	static constexpr uint64_t kLoadsPerAccess = AccessType::size_val / LoadType::size_val;

  public:
	uint8_t* byte_pointer_;

	uint64_t stride_;

  public:
	NIHILUS_DEVICE SharedLoadIterator(TensorRef ref, uint64_t thread_idx)
		: byte_pointer_(reinterpret_cast<uint8_t*>(ref.data())), stride_((ref.stride(0) * sizeof_bits<Element>::value) / 8) {
		TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

		byte_pointer_ += thread_offset.row() * stride_ + thread_offset.column() * sizeof(AccessType) / kElementsPerAccess;
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
	}

	NIHILUS_DEVICE void add_tile_offset(TensorCoord const& offset) {
		byte_pointer_ += offset.row() * Shape::kRow * stride_ + offset.column() * Shape::kColumn * sizeof_bits<Element>::value / 8;
	}

	NIHILUS_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const {
#pragma unroll
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
#pragma unroll
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
#pragma unroll
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint8_t const* byte_pointer = byte_pointer_ + row * ThreadMap::Delta::kRow * stride_ + group * ThreadMap::Delta::kGroup * stride_ +
						cluster * ThreadMap::Delta::kCluster * stride_ + pointer_offset * sizeof_bits<Element>::value / 8;

					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					LoadType* frag_ptr			   = reinterpret_cast<LoadType*>(&frag);
					LoadType const* memory_pointer = reinterpret_cast<LoadType const*>(byte_pointer);

#pragma unroll
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						uint64_t frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;

#pragma unroll
						for (uint64_t v = 0; v < kLoadsPerAccess; ++v) {
							frag_ptr[frag_idx * kLoadsPerAccess + v] = memory_pointer[(column * ThreadMap::Delta::kColumn / kElementsPerAccess) * kLoadsPerAccess + v];
						}
					}
				}
			}
		}
	}

	NIHILUS_DEVICE void set_smem_base_address(Index address) {
	}

	NIHILUS_DEVICE void load(Fragment& frag) const {
		load_with_pointer_offset(frag, 0);
	}
};

template<class> struct TypeSink {
	typedef void type;
};

template<class T> using TypeSinkT = typename TypeSink<T>::type;

template<class T, class = void> struct IsEpilogueFunctorHeavy {
	static constexpr bool value = false;
};

template<class T> struct IsEpilogueFunctorHeavy<T, TypeSinkT<decltype(T::kIsHeavy)>> {
	static constexpr bool value = T::kIsHeavy;
};

template<typename Shape_, typename WarpShape_, uint64_t PartitionsK, typename AccumulatorFragmentIterator_, typename WarpTileIterator_, typename Padding_,
	uint64_t FragmentsPerIteration = 1>
class EpilogueBase {
  public:
	using Shape							   = Shape_;
	using WarpShape						   = WarpShape_;
	static constexpr uint64_t kPartitionsK = PartitionsK;
	using AccumulatorFragmentIterator	   = AccumulatorFragmentIterator_;
	using WarpTileIterator				   = WarpTileIterator_;
	using Padding						   = Padding_;

	using Layout = RowMajor;

	using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

	using ElementAccumulator = typename AccumulatorTile::value_type;

	using WarpCount = GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN, kPartitionsK>;

	static constexpr uint64_t kFragmentsPerIteration = FragmentsPerIteration;

  public:
	struct SharedStorage {
		using Element = typename WarpTileIterator::Element;

		using TensorRef = typename WarpTileIterator::TensorRef;

		using Layout = typename WarpTileIterator::Layout;

		using Shape = MatrixShape<WarpCount::kM * WarpTileIterator::Shape::kRow * WarpCount::kK, WarpCount::kN * WarpTileIterator::Shape::kColumn>;

		using StorageShape = MatrixShape<(Shape::kRow + Padding::kRow) * kFragmentsPerIteration, Shape::kColumn + Padding::kColumn>;


		nihilus::array<Element, StorageShape::kCount> storage;


		NIHILUS_DEVICE Element* data() {
			return storage.data();
		}

		NIHILUS_DEVICE TensorRef reference() {
			return TensorRef(storage.data(), Layout::packed({ StorageShape::kRow, StorageShape::kColumn }));
		}
	};

  protected:
	SharedStorage& shared_storage_;

	WarpTileIterator warp_tile_iterator_;

  public:
	NIHILUS_DEVICE EpilogueBase(SharedStorage& shared_storage, uint64_t thread_idx, uint64_t warp_idx, uint64_t lane_idx)
		: shared_storage_(shared_storage), warp_tile_iterator_(shared_storage.reference(), lane_idx) {
		uint64_t warp_k	 = warp_idx / (WarpCount::kM * WarpCount::kN);
		uint64_t warp_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
		uint64_t warp_m	 = warp_mn % WarpCount::kM;
		uint64_t warp_n	 = warp_mn / WarpCount::kM;

		MatrixCoord warp_offset{ warp_k * WarpCount::kM + warp_m, warp_n };

		warp_tile_iterator_.add_tile_offset(warp_offset);
	}
};

template<typename T, uint64_t Limit> struct AccessWidth {
	template<uint64_t ObjectBytes, uint64_t AlignBytes, bool IsAligned = ((AlignBytes <= Limit) && (ObjectBytes % AlignBytes == 0))> struct Detail {
		static constexpr uint64_t value = Detail<ObjectBytes, AlignBytes * 2>::value;
	};

	template<uint64_t ObjectBytes, uint64_t AlignBytes> struct Detail<ObjectBytes, AlignBytes, false> {
		static constexpr uint64_t value = AlignBytes / 2;
	};

	static constexpr uint64_t value = Detail<( uint64_t )sizeof(T), 1>::value;
};

template<typename T, uint64_t TransferBytes = AccessWidth<T, 16>::value> struct alignas(TransferBytes) StripedAccessType : public T {};

template<typename T, same_or_convertible_to<T> U> constexpr auto nv_std_max(T&& a, U&& b) -> std::remove_cvref_t<T> {
	using common_type = std::remove_cvref_t<T>;
	if constexpr (std::is_same_v<std::remove_cvref_t<T>, std::remove_cvref_t<U>>) {
		return (b > a) ? std::forward<U>(b) : std::forward<T>(a);
	} else {
		common_type b_converted = static_cast<common_type>(b);
		return (b_converted > a) ? b_converted : static_cast<common_type>(a);
	}
}

template<typename T> struct plus {
	NIHILUS_HOST_DEVICE T operator()(T lhs, T const& rhs) const {
		lhs += rhs;
		return lhs;
	}
};

template<typename T, uint64_t N> struct plus<nihilus::array<T, N>> {
	NIHILUS_HOST_DEVICE nihilus::array<T, N> operator()(nihilus::array<T, N> const& lhs, nihilus::array<T, N> const& rhs) const {
		nihilus::array<T, N> result;
		plus<T> scalar_op;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = scalar_op(lhs[i], rhs[i]);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE nihilus::array<T, N> operator()(nihilus::array<T, N> const& lhs, T const& scalar) const {
		nihilus::array<T, N> result;
		plus<T> scalar_op;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = scalar_op(lhs[i], scalar);
		}

		return result;
	}

	NIHILUS_HOST_DEVICE nihilus::array<T, N> operator()(T const& scalar, nihilus::array<T, N> const& rhs) const {
		nihilus::array<T, N> result;
		plus<T> scalar_op;

#pragma unroll
		for (uint64_t i = 0; i < N; ++i) {
			result[i] = scalar_op(scalar, rhs[i]);
		}

		return result;
	}
};


template<typename T, uint64_t N, uint64_t TransferBytes> struct StripedAccessType<nihilus::array<T, N>, TransferBytes>
	: public nihilus::array<T, nv_std_max(1, TransferBytes / ( uint64_t )sizeof(T))> {};

template<uint64_t BlockThreads, typename ArrayT, typename AccessT = StripedAccessType<ArrayT>> struct BlockStriped {
	static constexpr uint64_t kStripes = uint64_t(sizeof(ArrayT) / sizeof(AccessT));
	static_assert(kStripes > 0, "AccessT type must be smaller than or equal to ArrayT type");

	NIHILUS_DEVICE static void load(ArrayT& data, ArrayT* ptr, uint64_t thread_idx) {
		AccessT* access_input = reinterpret_cast<AccessT*>(ptr);
		AccessT* access_data  = reinterpret_cast<AccessT*>(&data);

#pragma unroll
		for (uint64_t i = 0; i < kStripes; ++i) {
			access_data[i] = access_input[(BlockThreads * i) + thread_idx];
		}
	}

	NIHILUS_DEVICE static void load_add(ArrayT& data, ArrayT* ptr, uint64_t thread_idx) {
		AccessT* access_input = reinterpret_cast<AccessT*>(ptr);
		AccessT* access_data  = reinterpret_cast<AccessT*>(&data);

		plus<AccessT> add;

#pragma unroll
		for (uint64_t i = 0; i < kStripes; ++i) {
			access_data[i] = add(access_data[i], access_input[(BlockThreads * i) + thread_idx]);
		}
	}

	NIHILUS_DEVICE static void store(ArrayT* ptr, const ArrayT& data, uint64_t thread_idx) {
		AccessT* access_output	   = reinterpret_cast<AccessT*>(ptr);
		const AccessT* access_data = reinterpret_cast<const AccessT*>(&data);

#pragma unroll
		for (uint64_t i = 0; i < kStripes; ++i) {
			access_output[(BlockThreads * i) + thread_idx] = access_data[i];
		}
	}
};

template<typename Shape, uint64_t PartitionsK, typename WarpMmaOperator, typename AccumulatorFragmentIterator> class EpilogueBaseStreamK {
  protected:
	using AccumulatorTile = typename AccumulatorFragmentIterator::AccumulatorTile;

	using WarpCount = GemmShape<Shape::kM / WarpMmaOperator::Shape::kM, Shape::kN / WarpMmaOperator::Shape::kN, PartitionsK>;

	static constexpr uint64_t kBlockThreads = 32 * WarpCount::kCount;

	using ElementAccumulator = typename WarpMmaOperator::ElementC;

	using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  public:
	static constexpr uint64_t kAccumulatorFragments = AccumulatorFragmentIterator::Policy::kIterations;

  protected:
	static constexpr uint64_t kOutputTileFragments = kBlockThreads * kAccumulatorFragments;

	using BlockStripedT = BlockStriped<kBlockThreads, AccumulatorFragment>;

	static constexpr uint64_t kPeerFragmentStride = kOutputTileFragments * 2;

  public:
	static size_t const kWorkspaceBytesPerBlock = sizeof(AccumulatorFragment) * kPeerFragmentStride;

  public:
	uint64_t thread_idx;

  public:
	NIHILUS_DEVICE EpilogueBaseStreamK(uint64_t thread_idx) : thread_idx(thread_idx) {
	}


	NIHILUS_DEVICE void reduce(AccumulatorFragment& accum_fragment, uint64_t peer_idx_begin, uint64_t peer_idx_end, uint64_t reduce_fragment_idx, void* workspace_ptr) {
		plus<AccumulatorFragment> add_fragments;

		AccumulatorFragment* fragment_workspace = reinterpret_cast<AccumulatorFragment*>(workspace_ptr);

		uint64_t fragment_offset = (peer_idx_begin * kPeerFragmentStride) + (reduce_fragment_idx * kBlockThreads);

		BlockStripedT::load(accum_fragment, fragment_workspace + fragment_offset, this->thread_idx);

		fragment_offset += kPeerFragmentStride;
		fragment_offset += kOutputTileFragments;
#pragma unroll 2
		for (; fragment_offset < peer_idx_end * kPeerFragmentStride; fragment_offset += kPeerFragmentStride) {
			AccumulatorFragment addend_fragment;
			BlockStripedT::load(addend_fragment, fragment_workspace + fragment_offset, this->thread_idx);

			accum_fragment = add_fragments(accum_fragment, addend_fragment);
		}
	}


	NIHILUS_DEVICE void share(uint64_t peer_idx, void* workspace_ptr, AccumulatorTile const& accumulators, bool started_tile) {
		AccumulatorFragment* fragment_workspace = reinterpret_cast<AccumulatorFragment*>(workspace_ptr);

		uint64_t fragment_offset = peer_idx * kPeerFragmentStride;

		if (!started_tile) {
			fragment_offset += kOutputTileFragments;
		}

		AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

#pragma unroll
		for (uint64_t iter = 0; iter < kAccumulatorFragments; ++iter) {
			AccumulatorFragment accum_fragment;
			accum_fragment_iterator.load(accum_fragment);
			++accum_fragment_iterator;

			BlockStripedT::store(fragment_workspace + fragment_offset, accum_fragment, this->thread_idx);

			fragment_offset += kBlockThreads;
		}
	}
};

class PackedVectorLayout {
  public:
	static constexpr uint64_t kRank = 1;

	static constexpr uint64_t kStrideRank = 1;

	using Index = int64_t;

	using LongIndex = int64_t;

	using TensorCoord = constexpresh_coord<kRank>;

	using Stride = constexpresh_coord<kStrideRank>;

  public:
  public:
	NIHILUS_HOST_DEVICE PackedVectorLayout() {
	}

	NIHILUS_HOST_DEVICE static PackedVectorLayout packed(TensorCoord const& size) {
		return PackedVectorLayout();
	}

	NIHILUS_HOST_DEVICE LongIndex operator()(TensorCoord const& coord) const {
		return coord.at(0);
	}

	NIHILUS_HOST_DEVICE Stride stride() const {
		return Stride{ 1 };
	}

	NIHILUS_HOST_DEVICE LongIndex capacity(TensorCoord const& size) const {
		return size.at(0);
	}
};

template<typename Shape_, typename WarpMmaOperator_, uint64_t PartitionsK, typename OutputTileIterator_, typename AccumulatorFragmentIterator_, typename WarpTileIterator_,
	typename SharedLoadIterator_, typename OutputOp_, typename Padding_, uint64_t FragmentsPerPartition = 1,
	uint64_t IterationsUnroll = (!IsEpilogueFunctorHeavy<OutputOp_>::value)>
class Epilogue : public EpilogueBase<Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_, Padding_, FragmentsPerPartition>,
				 public EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_> {
  public:
	using Base = EpilogueBase<Shape_, typename WarpMmaOperator_::Shape, PartitionsK, AccumulatorFragmentIterator_, WarpTileIterator_, Padding_, FragmentsPerPartition>;

	using BaseStreamK = EpilogueBaseStreamK<Shape_, PartitionsK, WarpMmaOperator_, AccumulatorFragmentIterator_>;

	using Shape							   = Shape_;
	using WarpMmaOperator				   = WarpMmaOperator_;
	static constexpr uint64_t kPartitionsK = PartitionsK;
	using OutputTileIterator			   = OutputTileIterator_;
	using AccumulatorFragmentIterator	   = AccumulatorFragmentIterator_;
	using WarpTileIterator				   = WarpTileIterator_;
	using SharedLoadIterator			   = SharedLoadIterator_;
	using OutputOp						   = OutputOp_;
	using Padding						   = Padding_;
	using Layout						   = RowMajor;
	using LongIndex						   = typename Layout::LongIndex;

	using WarpCount = typename Base::WarpCount;

	static constexpr uint64_t kBlockThreads = 32 * WarpCount::kCount;

	using AccumulatorTile = typename Base::AccumulatorTile;

	using ElementAccumulator = typename WarpMmaOperator::ElementC;

	using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

	using ElementOutput = typename OutputTileIterator::Element;

	static constexpr uint64_t kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

	using TensorRef = typename OutputTileIterator::TensorRef;

	using SyncTensorRef = ::TensorRef<uint64_t, PackedVectorLayout>;

	using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

	using OutputAccessType = nihilus::array<typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

	using AccumulatorAccessType = nihilus::array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

	static uint64_t constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

	static uint64_t constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;


  public:
	static_assert(SharedLoadIterator::Fragment::size_val == OutputTileIterator::Fragment::size_val, "Mismatch between shared load iterator and output tile iterator.");

	static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

	static_assert(!(OutputTileIterator::Fragment::size_val % OutputTileIterator::kElementsPerAccess), "Divisibility");

	static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");


  public:
	struct SourceAspectNotNeeded {
		NIHILUS_DEVICE SourceAspectNotNeeded() {
		}

		NIHILUS_DEVICE void load() {
		}

		NIHILUS_DEVICE void apply_output_operator(typename OutputTileIterator::Fragment& output_fragment, OutputOp const& output_op,
			typename SharedLoadIterator::Fragment const& aligned_accum_fragment) {
			OutputAccessType* output_frag_ptr = reinterpret_cast<OutputAccessType*>(&output_fragment);

			AccumulatorAccessType const* compute_frag_ptr = reinterpret_cast<AccumulatorAccessType const*>(&aligned_accum_fragment);

			uint64_t const kOutputOpIterations = OutputTileIterator::Fragment::size_val / OutputTileIterator::kElementsPerAccess;

#pragma unroll
			for (uint64_t i = 0; i < kOutputOpIterations; ++i) {
				output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
			}
		}
	};
};

template<typename ThreadMap_, typename Element_, bool ScatterD = false, typename PermuteDLayout = NoPermute, bool UseCUDAStore = false> class PredicatedTileIterator {
  public:
	using ThreadMap = ThreadMap_;
	using Shape		= typename ThreadMap::Shape;

	using Element = Element_;

	using Layout		 = RowMajor;
	using TensorRef		 = TensorRef<Element, Layout>;
	using ConstTensorRef = typename TensorRef::ConstTensorRef;

	using Index		  = int32_t;
	using LongIndex	  = typename Layout::LongIndex;
	using TensorCoord = MatrixCoord;

	static constexpr uint64_t kElementsPerAccess = ThreadMap::kElementsPerAccess;
	static constexpr uint64_t kThreads			 = ThreadMap::kThreads;
	static constexpr uint64_t kIterations		 = ThreadMap::Count::kTile;

	static bool constexpr PermuteD = !is_trivial_permute<PermuteDLayout>;

	static_assert(ThreadMap::Iterations::kRow > 0, "ThreadMap::Iterations::kRow must be > 0");
	static_assert(ThreadMap::Iterations::kGroup > 0, "ThreadMap::Iterations::kGroup must be > 0");
	static_assert(ThreadMap::Iterations::kCluster > 0, "ThreadMap::Iterations::kCluster must be > 0");
	static_assert(ThreadMap::Iterations::kColumn > 0, "ThreadMap::Iterations::kColumn must be > 0");

	using Fragment = nihilus::array<Element,
		ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

	using AccessType = nihilus::array<Element, ThreadMap::kElementsPerAccess>;


	struct Params : PredicatedTileIteratorParams {
		using Base = PredicatedTileIteratorParams;

		NIHILUS_HOST_DEVICE Params() {
		}

		NIHILUS_HOST_DEVICE Params(Layout const& layout)
			: PredicatedTileIteratorParams(layout.stride(0) * uint64_t(sizeof(AccessType)) / kElementsPerAccess, make_OutputTileThreadMapDesc<ThreadMap>()) {
		}

		NIHILUS_HOST_DEVICE Params(Layout const& layout, Tensor4DCoord const& tensor_extent) : Params(layout) {
		}

		NIHILUS_HOST_DEVICE Params(Layout const& layout, Tensor5DCoord const& tensor_extent) : Params(layout) {
		}

		NIHILUS_HOST_DEVICE Params(Base const& base) : Base(base) {
		}
	};

	struct Mask {
		static constexpr uint64_t kCount = ThreadMap::Iterations::kColumn;

		bool predicates[kCount];

		NIHILUS_HOST_DEVICE Mask() {
			enable();
		}

		NIHILUS_HOST_DEVICE void clear() {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t i = 0; i < kCount; ++i) {
				predicates[i] = false;
			}
		}

		NIHILUS_DEVICE void enable() {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t i = 0; i < kCount; ++i) {
				predicates[i] = true;
			}
		}
	};

  public:
	PredicatedTileIteratorParams params_;

	uint8_t* byte_pointer_;

	uint8_t* store_byte_pointer_;

	Mask mask_;

	Index extent_row_;

	Index extent_column_;

	Index thread_start_row_;

	Index thread_start_column_;

	uint64_t state_[3];

	uint64_t const* indices_;

	PermuteDLayout permute_layout_;


	static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
	static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
	static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

  public:
  public:
	NIHILUS_DEVICE PredicatedTileIterator(PredicatedTileIteratorParams const& params, Element* pointer, TensorCoord extent, uint64_t thread_idx,
		TensorCoord threadblock_offset = TensorCoord(), uint64_t const* indices = nullptr)
		: params_(params), indices_(indices), permute_layout_(PitchLinearCoord(extent.column(), extent.row()), params_.stride * kElementsPerAccess / sizeof(AccessType)) {
		TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;

		extent_row_	   = extent.row();
		extent_column_ = extent.column();

		thread_start_row_	 = thread_offset.row();
		thread_start_column_ = thread_offset.column();

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
			mask_.predicates[c] = ((thread_offset.column() + ThreadMap::Delta::kColumn * c) < extent.column());
		}

		if (!pointer) {
			mask_.clear();
		}

		if (ScatterD && !indices) {
			mask_.clear();
		}

		byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) + LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
			LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;

		if (ScatterD) {
			byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) + LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
		}

		store_byte_pointer_ = PermuteD ? reinterpret_cast<uint8_t*>(pointer) : byte_pointer_;

		state_[0] = state_[1] = state_[2] = 0;
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
		byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
	}

	NIHILUS_DEVICE void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const {
		uint8_t* byte_pointer = byte_pointer_;
		AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
				NIHILUS_PRAGMA_UNROLL
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					uint64_t row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

					bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

					AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

					if (ScatterD && row_guard) {
						assert(indices_);

						memory_pointer =
							reinterpret_cast<AccessType*>(byte_pointer + byte_offset + LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
					}

					NIHILUS_PRAGMA_UNROLL
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						bool guard = row_guard && mask_.predicates[column];

						global_load<AccessType, sizeof(AccessType)>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
							( void* )&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess], guard);
					}

					if (row + 1 < ThreadMap::Iterations::kRow) {
						if (!ScatterD) {
							byte_pointer += params_.increment_row;
						}
					}
				}

				if (group + 1 < ThreadMap::Iterations::kGroup) {
					byte_pointer += params_.increment_group;
				}
			}

			if (cluster + 1 < ThreadMap::Iterations::kCluster) {
				byte_pointer += params_.increment_cluster;
			}
		}
	}

	NIHILUS_DEVICE void load(Fragment& frag) const {
		load_with_byte_offset(frag, 0);
	}

	NIHILUS_DEVICE void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const {
		uint8_t* byte_pointer	   = store_byte_pointer_;
		AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
				NIHILUS_PRAGMA_UNROLL
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					uint64_t row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

					bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

					AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

					if (ScatterD && row_guard) {
						assert(indices_);

						memory_pointer =
							reinterpret_cast<AccessType*>(byte_pointer + byte_offset + LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
					}

					NIHILUS_PRAGMA_UNROLL
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						bool guard = row_guard && mask_.predicates[column];

						if (PermuteD) {
							uint64_t col_offset = column * ThreadMap::Delta::kColumn;

							uint64_t col = col_offset + thread_start_column_;
							uint64_t row = row_offset + thread_start_row_;

							memory_pointer =
								reinterpret_cast<AccessType*>(byte_pointer + byte_offset + permute_layout_(PitchLinearCoord(col, row)) * sizeof(AccessType) / kElementsPerAccess);
						}

						if (UseCUDAStore) {
							if (guard) {
								memory_pointer[0] = frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
							}
						} else {
							global_store<AccessType, sizeof(AccessType)>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column], ( void* )&memory_pointer[0], guard);
						}

						if (!PermuteD) {
							memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
						}
					}

					if (row + 1 < ThreadMap::Iterations::kRow) {
						if (!ScatterD && !PermuteD) {
							byte_pointer += params_.increment_row;
						}
					}
				}

				if (group + 1 < ThreadMap::Iterations::kGroup) {
					if (!ScatterD && !PermuteD) {
						byte_pointer += params_.increment_group;
					}
				}
			}

			if (cluster + 1 < ThreadMap::Iterations::kCluster) {
				if (!ScatterD && !PermuteD) {
					byte_pointer += params_.increment_cluster;
				}
			}
		}
	}

	NIHILUS_DEVICE void store(Fragment const& frag) const {
		store_with_byte_offset(frag, 0);
	}

	NIHILUS_DEVICE void downsample_load_with_byte_offset(Fragment& frag, int64_t byte_offset, uint64_t convolution_P, uint64_t convolution_Q, uint64_t add_P, uint64_t add_Q,
		uint64_t problem_N) const {
		uint8_t* byte_pointer = byte_pointer_;
		AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
				NIHILUS_PRAGMA_UNROLL
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					uint64_t row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

					bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

					uint64_t output_row = row_offset + thread_start_row_;
					uint64_t output_N	= output_row / (convolution_P * convolution_Q);
					uint64_t output_PQ	= output_row % (convolution_P * convolution_Q);
					uint64_t output_P	= output_PQ / convolution_Q;
					uint64_t output_Q	= output_PQ % convolution_Q;

					uint64_t input_row = output_N * 2 * convolution_P * 2 * convolution_Q + (2 * output_P + add_P) * 2 * convolution_Q + 2 * output_Q + add_Q;

					int64_t byte_offset = (input_row - output_row) * problem_N * sizeof(float);

					AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

					NIHILUS_PRAGMA_UNROLL
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						bool guard = row_guard && mask_.predicates[column];

						global_load<AccessType, sizeof(AccessType)>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
							( void* )&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess], guard);
					}

					if (row + 1 < ThreadMap::Iterations::kRow) {
						byte_pointer += params_.increment_row;
					}
				}

				if (group + 1 < ThreadMap::Iterations::kGroup) {
					byte_pointer += params_.increment_group;
				}
			}

			if (cluster + 1 < ThreadMap::Iterations::kCluster) {
				byte_pointer += params_.increment_cluster;
			}
		}
	}

	NIHILUS_DEVICE void upsample_load_with_byte_offset(Fragment& frag, int64_t byte_offset, uint64_t convolution_P, uint64_t convolution_Q, uint64_t add_P, uint64_t add_Q,
		uint64_t problem_N) const {
		uint8_t* byte_pointer = byte_pointer_;
		AccessType* frag_ptr  = reinterpret_cast<AccessType*>(&frag);

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
				NIHILUS_PRAGMA_UNROLL
				for (uint64_t row = 0; row < ThreadMap::Iterations::kRow; ++row) {
					uint64_t frag_row_idx = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

					uint64_t row_offset = row * ThreadMap::Delta::kRow + group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;

					bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

					uint64_t output_row = row_offset + thread_start_row_;
					uint64_t output_N	= output_row / (convolution_P * convolution_Q);
					uint64_t output_PQ	= output_row % (convolution_P * convolution_Q);
					uint64_t output_P	= output_PQ / convolution_Q;
					uint64_t output_Q	= output_PQ % convolution_Q;
					uint64_t row_add_P	= add_P;
					uint64_t row_add_Q	= add_Q;
					if (output_P > convolution_P - 2)
						row_add_P = 0;
					if (output_Q > convolution_Q - 2)
						row_add_Q = 0;

					uint64_t input_row = output_N * (convolution_P / 2) * (convolution_Q / 2) + ((output_P + row_add_P) / 2) * (convolution_Q / 2) + (output_Q + row_add_Q) / 2;

					int64_t byte_offset = (input_row - output_row) * problem_N * sizeof(float);

					AccessType* memory_pointer = reinterpret_cast<AccessType*>(byte_pointer + byte_offset);

					NIHILUS_PRAGMA_UNROLL
					for (uint64_t column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
						bool guard = row_guard && mask_.predicates[column];

						global_load<AccessType, sizeof(AccessType)>(frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
							( void* )&memory_pointer[column * ThreadMap::Delta::kColumn / kElementsPerAccess], guard);
					}

					if (row + 1 < ThreadMap::Iterations::kRow) {
						byte_pointer += params_.increment_row;
					}
				}

				if (group + 1 < ThreadMap::Iterations::kGroup) {
					byte_pointer += params_.increment_group;
				}
			}

			if (cluster + 1 < ThreadMap::Iterations::kCluster) {
				byte_pointer += params_.increment_cluster;
			}
		}
	}

	NIHILUS_DEVICE MatrixCoord thread_start() const {
		return MatrixCoord(thread_start_row_, thread_start_column_);
	}

	NIHILUS_DEVICE int64_t thread_start_row() const {
		return thread_start_row_;
	}

	NIHILUS_DEVICE int64_t thread_start_column() const {
		return thread_start_column_;
	}

	NIHILUS_DEVICE Index extent_row() const {
		return extent_row_;
	}

	NIHILUS_DEVICE Index extent_column() const {
		return extent_column_;
	}

	NIHILUS_HOST_DEVICE PredicatedTileIterator& operator++() {
		++state_[0];

		if (!ScatterD) {
			byte_pointer_ += params_.advance_row;
		}

		if (!ScatterD && !PermuteD) {
			store_byte_pointer_ += params_.advance_row;
		}

		thread_start_row_ += ThreadMap::Shape::kRow;

		if (state_[0] == ThreadMap::Count::kRow) {
			state_[0] = 0;
			++state_[1];

			if (!ScatterD) {
				byte_pointer_ += params_.advance_group;
			}

			if (!ScatterD && !PermuteD) {
				store_byte_pointer_ += params_.advance_group;
			}

			thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

			if (state_[1] == ThreadMap::Count::kGroup) {
				state_[1] = 0;
				++state_[2];

				if (!ScatterD) {
					byte_pointer_ += params_.advance_cluster;
				}

				if (!ScatterD && !PermuteD) {
					store_byte_pointer_ += params_.advance_cluster;
				}

				thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

				if (state_[2] == ThreadMap::Count::kCluster) {
					state_[2] = 0;

					if (!ScatterD) {
						byte_pointer_ += params_.advance_tile;
					}

					if (!ScatterD && !PermuteD) {
						store_byte_pointer_ += params_.advance_tile;
					}

					thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
				}
			}
		}

		return *this;
	}

	NIHILUS_HOST_DEVICE PredicatedTileIterator& operator+=(uint64_t increment) {
		state_[0] += increment;
		uint64_t increment_row = state_[0] / ThreadMap::Count::kRow;
		state_[0]			   = state_[0] % ThreadMap::Count::kRow;

		byte_pointer_ += (params_.advance_row * increment);
		store_byte_pointer_ += (params_.advance_row * increment);
		thread_start_row_ += (ThreadMap::Shape::kRow * increment);

		state_[1] += increment_row;
		uint64_t increment_group = state_[1] / ThreadMap::Count::kGroup;
		state_[1]				 = state_[1] % ThreadMap::Count::kGroup;

		byte_pointer_ += (params_.advance_group * increment_row);
		store_byte_pointer_ += (params_.advance_group * increment_row);
		thread_start_row_ += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow * increment_row;


		state_[2] += increment_group;
		uint64_t increment_cluster = state_[2] / ThreadMap::Count::kCluster;
		state_[2]				   = state_[2] % ThreadMap::Count::kCluster;

		byte_pointer_ += (params_.advance_cluster * increment_group);
		store_byte_pointer_ += (params_.advance_cluster * increment_group);
		thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow * increment_group;

		byte_pointer_ += (params_.advance_tile * increment_cluster);
		store_byte_pointer_ += (params_.advance_tile * increment_cluster);
		thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile * increment_cluster;

		return *this;
	}

	NIHILUS_DEVICE void clear_mask() {
		mask_.clear();
	}

	NIHILUS_DEVICE void enable_mask() {
		mask_.enable();
	}

	NIHILUS_DEVICE void get_mask(Mask& mask) const {
		mask = mask_;
	}

	NIHILUS_DEVICE void set_mask(Mask const& mask) {
		mask_ = mask;
	}
};

template<typename Shape_, typename WarpMmaSimt_, typename OutputOp_, uint64_t ElementsPerAccess, bool ScatterD = false, typename PermuteDLayout = NoPermute,
	StrideSupport StrideSupport_val = StrideSupport::kUnity, uint64_t Rank = 4>
struct DefaultEpilogueSimt {
	using Shape									 = Shape_;
	using WarpMmaSimt							 = WarpMmaSimt_;
	using OutputOp								 = OutputOp_;
	static constexpr uint64_t kElementsPerAccess = ElementsPerAccess;
	static constexpr uint64_t kPartitionsK		 = Shape::kK / WarpMmaSimt::Shape::kK;

	using ElementOutput						  = typename OutputOp::ElementOutput;
	using LayoutC							  = typename WarpMmaSimt::LayoutC;
	using ElementAccumulator				  = typename WarpMmaSimt::ElementC;
	static StrideSupport const kStrideSupport = StrideSupport_val;
	static constexpr uint64_t kRank			  = Rank;

	using OutputTileThreadMap =
		typename DefaultThreadMapSimt<Shape, typename WarpMmaSimt::Shape, typename WarpMmaSimt::Policy, kPartitionsK, ElementOutput, kElementsPerAccess>::Type;

	static constexpr bool UseCUDAStore = std::is_same<ElementOutput, double>::value;

	using PackedOutputTileIterator = PredicatedTileIterator<OutputTileThreadMap, ElementOutput, ScatterD, PermuteDLayout, UseCUDAStore>;

	using StridedOutputTileIterator = PredicatedTileIteratorConv<OutputTileThreadMap, ElementOutput, ScatterD, PermuteDLayout, UseCUDAStore, kRank>;

	using OutputTileIterator = typename std::conditional<StrideSupport_val == StrideSupport::kUnity, PackedOutputTileIterator, StridedOutputTileIterator>::type;

	using AccumulatorFragmentIterator = FragmentIteratorSimt<typename WarpMmaSimt::Shape, typename WarpMmaSimt::ThreadMma, RowMajor, typename WarpMmaSimt::Policy>;

	using WarpTileIterator = TileIteratorSimt<typename WarpMmaSimt::Shape, typename WarpMmaSimt::ThreadMma, ElementAccumulator, RowMajor, typename WarpMmaSimt::Policy>;

	using SharedLoadIterator = SharedLoadIterator<typename OutputTileThreadMap::CompactedThreadMap, ElementAccumulator>;

	using Padding = typename WarpTileIterator::Padding;

	using Epilogue = Epilogue<Shape, WarpMmaSimt, kPartitionsK, OutputTileIterator, AccumulatorFragmentIterator, WarpTileIterator, SharedLoadIterator, OutputOp, Padding>;
};

static const uint64_t NumThreadsPerWarp		 = 32;
static const uint64_t NumThreadsPerWarpGroup = 128;
static const uint64_t NumWarpsPerWarpGroup	 = NumThreadsPerWarpGroup / NumThreadsPerWarp;
static const uint64_t NumThreadsPerHalfWarp	 = NumThreadsPerWarp / 2;
static const uint64_t NumThreadsPerQuad		 = 4;
static const uint64_t NumThreadsPerQuadPair	 = NumThreadsPerQuad * 2;

NIHILUS_HOST_DEVICE bool thread0() {
#if defined(__CUDA_ARCH__)
	return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
#else
	return false;
#endif
}

NIHILUS_DEVICE uint64_t canonical_lane_idx() {
#if defined(__CUDA_ARCH__)
	return threadIdx.x % NumThreadsPerWarp;
#else
	return 0;
#endif
}

NIHILUS_DEVICE uint64_t canonical_warp_idx_sync() {
#if defined(__CUDA_ARCH__)
	return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
#else
	return 0;
#endif
}

NIHILUS_DEVICE uint64_t canonical_warp_idx() {
#if defined(__CUDA_ARCH__)
	return threadIdx.x / NumThreadsPerWarp;
#else
	return 0;
#endif
}

NIHILUS_DEVICE uint64_t canonical_warp_group_idx() {
#if defined(__CUDA_ARCH__)
	return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarpGroup, 0);
#else
	return 0;
#endif
}

template<uint64_t M_new, uint64_t K_new, typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool SplitKSerial> struct Gemm {
	using Mma							= Mma_;
	using Epilogue						= Epilogue_;
	using OutputOp						= typename Epilogue::OutputOp;
	using ThreadblockSwizzle			= ThreadblockSwizzle_;
	static constexpr bool kSplitKSerial = SplitKSerial;

	using WarpCount						   = typename Mma::WarpCount;
	static constexpr uint64_t kThreadCount = 32 * WarpCount::kCount;
	template<uint64_t M_newer, uint64_t K_newer> struct Params {
		GemmCoord<M_new, K_new> problem_size;
		GemmCoord<M_newer, K_newer> grid_tiled_shape;
		uint64_t swizzle_log_tile;
		typename Mma::IteratorA::Params params_A;
		typename Mma::IteratorA::TensorRef ref_A;
		typename Mma::IteratorB::Params params_B;
		typename Mma::IteratorB::TensorRef ref_B;
		typename Epilogue::OutputTileIterator::Params params_C;
		typename Epilogue::OutputTileIterator::TensorRef ref_C;
		typename Epilogue::OutputTileIterator::Params params_D;
		typename Epilogue::OutputTileIterator::TensorRef ref_D;
		typename OutputOp::Params output_op;
		uint64_t* semaphore;
		uint64_t gemm_k_size;
		uint64_t const* gather_A_indices;
		uint64_t const* gather_B_indices;
		uint64_t const* scatter_D_indices;


		NIHILUS_HOST_DEVICE Params() : swizzle_log_tile(0), semaphore(0), gemm_k_size(0) {
		}

		NIHILUS_HOST_DEVICE Params(GemmCoord<M_new, K_new> const& problem_size, GemmCoord<M_newer, K_newer> const& grid_tiled_shape, typename Mma::IteratorA::TensorRef ref_A,
			typename Mma::IteratorB::TensorRef ref_B, typename Epilogue::OutputTileIterator::TensorRef ref_C, typename Epilogue::OutputTileIterator::TensorRef ref_D,
			typename OutputOp::Params output_op = typename OutputOp::Params(), uint64_t* workspace = nullptr, uint64_t const* gather_A_indices = nullptr,
			uint64_t const* gather_B_indices = nullptr, uint64_t const* scatter_D_indices = nullptr)
			: problem_size(problem_size), grid_tiled_shape(grid_tiled_shape), swizzle_log_tile(ThreadblockSwizzle::get_log_tile(grid_tiled_shape)), params_A(ref_A.layout()),
			  ref_A(ref_A), params_B(ref_B.layout()), ref_B(ref_B), params_C(ref_C.layout()), ref_C(ref_C), params_D(ref_D.layout()), ref_D(ref_D), output_op(output_op),
			  gather_A_indices(gather_A_indices), gather_B_indices(gather_B_indices), scatter_D_indices(scatter_D_indices) {
			uint64_t total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
			uint64_t gemm_k_iterations		 = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();

			gemm_k_size = gemm_k_iterations * Mma::Shape::kK;

			semaphore = workspace;
		}
	};

	union SharedStorage {
		typename Mma::SharedStorage main_loop;
		typename Epilogue::SharedStorage epilogue;
	};


	NIHILUS_HOST_DEVICE Gemm() {
	}

	NIHILUS_HOST_DEVICE static Status can_implement(GemmCoord<M_new, K_new> const& problem_size, typename Mma::IteratorA::TensorRef ref_A, typename Mma::IteratorB::TensorRef ref_B,
		typename Epilogue::OutputTileIterator::TensorRef ref_C, typename Epilogue::OutputTileIterator::TensorRef ref_D) {
		static constexpr uint64_t kAlignmentA = (std::is_same<typename Mma::IteratorA::Layout, ColumnMajorInterleaved<32>>::value) ? 32
			: (std::is_same<typename Mma::IteratorA::Layout, ColumnMajorInterleaved<64>>::value)								   ? 64
																																   : Mma::IteratorA::AccessType::size_val;
		static constexpr uint64_t kAlignmentB = (std::is_same<typename Mma::IteratorB::Layout, RowMajorInterleaved<32>>::value) ? 32
			: (std::is_same<typename Mma::IteratorB::Layout, RowMajorInterleaved<64>>::value)									? 64
																																: Mma::IteratorB::AccessType::size_val;
		static constexpr uint64_t kAlignmentC = (std::is_same<typename Epilogue::OutputTileIterator::Layout, ColumnMajorInterleaved<32>>::value) ? 32
			: (std::is_same<typename Epilogue::OutputTileIterator::Layout, ColumnMajorInterleaved<64>>::value)									 ? 64
																											   : Epilogue::OutputTileIterator::kElementsPerAccess;

		if (!TensorRef_aligned(ref_A, kAlignmentA)) {
			return Status::kErrorMisalignedOperand;
		}

		if (!TensorRef_aligned(ref_B, kAlignmentB)) {
			return Status::kErrorMisalignedOperand;
		}

		if (!TensorRef_aligned(ref_C, kAlignmentC)) {
			return Status::kErrorMisalignedOperand;
		}

		if (!TensorRef_aligned(ref_D, kAlignmentC)) {
			return Status::kErrorMisalignedOperand;
		}

		return Status::kSuccess;
	}

	__device__ static void impl(auto const& params, SharedStorage& shared_storage) {
		ThreadblockSwizzle threadblock_swizzle;

		constexpresh_coord<3> threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

		MatrixCoord tb_offset_A{
			threadblock_tile_offset.m() * Mma::Shape::kM,
			threadblock_tile_offset.k() * params.gemm_k_size,
		};

		MatrixCoord tb_offset_B{ threadblock_tile_offset.k() * params.gemm_k_size, threadblock_tile_offset.n() * Mma::Shape::kN };

		uint64_t problem_size_k = min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

		uint64_t gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

#ifdef __CUDA_ARCH__
		uint64_t thread_idx = threadIdx.x;
#else
		uint64_t thread_idx = 0;
#endif

		typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(), { params.problem_size.m(), problem_size_k }, thread_idx, tb_offset_A, params.gather_A_indices);

		typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(), { problem_size_k, params.problem_size.n() }, thread_idx, tb_offset_B, params.gather_B_indices);

		uint64_t warp_idx = canonical_warp_idx_sync();
		uint64_t lane_idx = threadIdx.x % 32;


		Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

		typename Mma::FragmentC accumulators;

		if (!kSplitKSerial || gemm_k_iterations > 0) {
			mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
		}


		OutputOp output_op(params.output_op);


		threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

		MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM, threadblock_tile_offset.n() * Mma::Shape::kN);

		uint64_t block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

		if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
			output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
		}

		typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(), params.problem_size.mn(), thread_idx, threadblock_offset, params.scatter_D_indices);

		typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(), params.problem_size.mn(), thread_idx, threadblock_offset, params.scatter_D_indices);

		if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
			if (threadblock_tile_offset.k()) {
				iterator_C = iterator_D;
			}
		}

		Epilogue::impl(output_op, iterator_D, accumulators, iterator_C);


		if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
			uint64_t lock = 0;
			if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {
				lock = 0;
			} else {
				lock = threadblock_tile_offset.k() + 1;
			}
		}
	}
};

template<typename Shape, typename WarpShape, typename InstructionShape, typename ElementA, typename LayoutA, typename ElementB, typename LayoutB, typename ElementC,
	typename LayoutC, typename OperatorClass, uint64_t Stages = 2, typename Operator = OpMultiplyAdd, bool AccumulatorsInRowMajor = false,
	CacheOperation::Kind CacheOpA = CacheOperation::Global, CacheOperation::Kind CacheOpB = CacheOperation::Global, ComplexTransform TransformA = ComplexTransform::kNone,
	ComplexTransform TransformB = ComplexTransform::kNone, bool IsComplex = false>
struct DefaultMmaCore;

template<uint64_t M_, uint64_t K_, typename ElementA, typename LayoutA, uint64_t kAlignmentA, typename ElementB, typename LayoutB, uint64_t kAlignmentB,
	typename ElementAccumulator, typename LayoutC, typename OperatorClass, typename ArchTag, typename ThreadblockShape, typename WarpShape, typename InstructionShape,
	uint64_t Stages, typename Operator, bool AccumulatorsInRowMajor = false, SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone, bool GatherA = false,
	bool GatherB = false, typename PermuteALayout = NoPermute, typename PermuteBLayout = NoPermute>
struct DefaultMma {};

template<typename Shape_, typename WarpShape_, typename ElementA_, typename ElementB_, typename ElementC_, typename LayoutC_, typename Operator_>
struct DefaultMmaCore<Shape_, WarpShape_, GemmShape<1, 1, 1>, ElementA_, RowMajor, ElementB_, RowMajor, ElementC_, LayoutC_, OpClassSimt, 2, Operator_> {
	using Shape							  = Shape_;
	using WarpShape						  = WarpShape_;
	using InstructionShape				  = GemmShape<1, 1, 1>;
	using ElementA						  = ElementA_;
	using LayoutA						  = RowMajor;
	using ElementB						  = ElementB_;
	using LayoutB						  = RowMajor;
	using ElementC						  = ElementC_;
	using LayoutC						  = LayoutC_;
	using OperatorClass					  = OpClassSimt;
	static constexpr uint64_t PartitionsK = Shape::kK / WarpShape::kK;

	using Operator = Operator_;

	using WarpCount = GemmShape<Shape::kM / WarpShape::kM, Shape::kN / WarpShape::kN, PartitionsK>;

	static_assert(!(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN), "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

	static constexpr uint64_t kWarpSize = WarpSize<OpClassSimt>::value;

	static constexpr uint64_t kThreads = WarpCount::kCount * kWarpSize;

	static constexpr uint64_t kElementsPerAccess = 1;

	using SmemLayoutA = ColumnMajor;
	using SmemLayoutB = RowMajor;

	using IteratorThreadMapA = PitchLinearStripminedThreadMap<PitchLinearShape<Shape::kK, Shape::kM>, kThreads, kElementsPerAccess>;

	using SmemThreadMapA = TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

	using SmemIteratorA = RegularTileIterator<MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1, SmemThreadMapA>;

	using IteratorThreadMapB = PitchLinearStripminedThreadMap<PitchLinearShape<Shape::kN, Shape::kK>, kThreads, kElementsPerAccess>;

	using SmemIteratorB = RegularTileIterator<MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0, IteratorThreadMapB>;


	static constexpr uint64_t WarpNumThreadsM = simt_get_warp_threads_m<WarpShape>();
	static constexpr uint64_t WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
	static constexpr uint64_t ThreadTileM	  = WarpShape::kM / WarpNumThreadsM;
	static constexpr uint64_t ThreadTileN	  = WarpShape::kN / WarpNumThreadsN;
	static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN), "WarpShape must be divisible by ThreadTile shape.");
	static constexpr uint64_t LaneLayout   = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
	static constexpr uint64_t numElementsA = 128 / sizeof_bits<ElementA>::value;
	static constexpr uint64_t numElementsB = 128 / sizeof_bits<ElementB>::value;
	static constexpr uint64_t LaneM		   = const_min(numElementsA, ThreadTileM);
	static constexpr uint64_t LaneN		   = const_min(numElementsB, ThreadTileN);

	static constexpr uint64_t kPaddingM = simt_transpose_padding(kWarpSize, Shape::kK, sizeof_bits<ElementA>::value);

	static_assert(!(kPaddingM % LaneM), "Padding must be divisible by Lane");

	using LaneMmaShape = GemmShape<LaneM, LaneN, 1>;
	using Policy	   = MmaSimtPolicy<MatrixShape<WarpNumThreadsM, WarpNumThreadsN>, RowMajorInterleaved<LaneLayout>, LaneMmaShape>;

	using MmaWarpSimt = MmaSimt<WarpShape, ElementA, SmemLayoutA, ElementB, SmemLayoutB, ElementC, LayoutC, Policy>;

	using MmaPolicy = MmaPolicy<MmaWarpSimt, MatrixShape<kPaddingM, 0>, MatrixShape<0, 0>, WarpCount::kK>;
};

template<typename Shape_, typename Policy_, uint64_t Stages, typename Enable = bool> class MmaBase {
  public:
	using Shape = Shape_;

	using Policy = Policy_;


	using Operator = typename Policy::Operator;

	using WarpGemm = typename Policy::Operator::Shape;

	using WarpCount = GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;

	static constexpr uint64_t kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);

	static constexpr uint64_t kStages = Stages;

	using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

	using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

	static_assert(kWarpGemmIterations > 1,
		"The pipelined structure requires at least two warp-level "
		"GEMM operations.");

	static_assert((kWarpGemmIterations % 2) == 0, "Inner loop iteration must be an even number.");


	class SharedStorage {
	  public:
		using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow, Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;

		using ShapeB = MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow, Shape::kN + Policy::SmemPaddingB::kColumn>;

	  public:
		nihilus::array<typename Operator::ElementA, ShapeA::kCount> operand_A;

		nihilus::array<typename Operator::ElementB, ShapeB::kCount> operand_B;

	  public:
		NIHILUS_DEVICE static typename Operator::LayoutA LayoutA() {
			return Operator::LayoutA::packed({ ShapeA::kRow, ShapeA::kColumn });
		}

		NIHILUS_HOST_DEVICE static typename Operator::LayoutB LayoutB() {
			return Operator::LayoutB::packed({ ShapeB::kRow, ShapeB::kColumn });
		}

		NIHILUS_HOST_DEVICE TensorRefA operand_A_ref() {
			return TensorRefA{ operand_A.data(), LayoutA() };
		}

		NIHILUS_HOST_DEVICE TensorRefB operand_B_ref() {
			return TensorRefB{ operand_B.data(), LayoutB() };
		}
	};

  protected:
	typename Operator::IteratorA warp_tile_iterator_A_;

	typename Operator::IteratorB warp_tile_iterator_B_;

  public:
	NIHILUS_DEVICE MmaBase(SharedStorage& shared_storage, uint64_t thread_idx, uint64_t warp_idx, uint64_t lane_idx)
		: warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx), warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {
	}
};

template<typename Shape_, typename IteratorA_, typename SmemIteratorA_, typename IteratorB_, typename SmemIteratorB_, typename ElementC_, typename LayoutC_, typename Policy_,
	typename TransformA_ = NumericArrayConverter<typename SmemIteratorA_::Element, typename IteratorA_::Element, IteratorA_::Fragment::size_val>,
	typename TransformB_ = NumericArrayConverter<typename SmemIteratorB_::Element, typename IteratorB_::Element, IteratorB_::Fragment::size_val>, typename Enable = bool>
class MmaPipelined : public MmaBase<Shape_, Policy_, 2> {
  public:
	using Base = MmaBase<Shape_, Policy_, 2>;

	using Shape			= Shape_;
	using IteratorA		= IteratorA_;
	using IteratorB		= IteratorB_;
	using ElementC		= ElementC_;
	using LayoutC		= LayoutC_;
	using Policy		= Policy_;
	using SmemIteratorA = SmemIteratorA_;
	using SmemIteratorB = SmemIteratorB_;

	using TransformA = TransformA_;
	using TransformB = TransformB_;


	using FragmentA = typename IteratorA::Fragment;

	using FragmentB = typename IteratorB::Fragment;

	using FragmentC = typename Policy::Operator::FragmentC;

	using Operator = typename Policy::Operator;

	using ArchTag = typename Policy::Operator::ArchTag;

	static constexpr ComplexTransform kTransformA = Operator::kTransformA;

	static constexpr ComplexTransform kTransformB = Operator::kTransformB;

	static_assert((Base::kStages == 2), "MmaPipelined requires kStages set to value 2");

  protected:
	Operator warp_mma;

	SmemIteratorA smem_iterator_A_;

	SmemIteratorB smem_iterator_B_;

	TransformA transform_A_;

	TransformB transform_B_;

	uint64_t smem_write_stage_idx;

  public:
	NIHILUS_DEVICE MmaPipelined(typename Base::SharedStorage& shared_storage, uint64_t thread_idx, uint64_t warp_idx, uint64_t lane_idx, TransformA transform_A = TransformA(),
		TransformB transform_B = TransformB())
		: Base(shared_storage, thread_idx, warp_idx, lane_idx), smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
		  smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx), transform_A_(transform_A), transform_B_(transform_B), smem_write_stage_idx(0) {
		uint64_t warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
		uint64_t warp_idx_k	 = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

		uint64_t warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
		uint64_t warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

		this->warp_tile_iterator_A_.add_tile_offset({ warp_idx_m, Base::kWarpGemmIterations * warp_idx_k });
		this->warp_tile_iterator_B_.add_tile_offset({ Base::kWarpGemmIterations * warp_idx_k, warp_idx_n });
	}


	NIHILUS_DEVICE void advance_smem_write_stage() {
		++this->smem_iterator_A_;
		++this->smem_iterator_B_;

		if (smem_write_stage_idx == 1) {
			this->smem_iterator_A_.add_tile_offset({ 0, -Base::kStages });
			this->smem_iterator_B_.add_tile_offset({ -Base::kStages, 0 });
		}

		smem_write_stage_idx ^= 1;
	}

	NIHILUS_DEVICE void advance_smem_stages() {
		++this->smem_iterator_A_;
		++this->smem_iterator_B_;

		if (smem_write_stage_idx == 1) {
			this->smem_iterator_A_.add_tile_offset({ 0, -Base::kStages });
			this->smem_iterator_B_.add_tile_offset({ -Base::kStages, 0 });
		} else {
			this->warp_tile_iterator_A_.add_tile_offset({ 0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations });
			this->warp_tile_iterator_B_.add_tile_offset({ -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0 });
		}

		smem_write_stage_idx ^= 1;
	}


	NIHILUS_DEVICE void prologue(IteratorA& iterator_A, IteratorB& iterator_B, uint64_t& gemm_k_iterations) {
		FragmentA tb_frag_A;
		iterator_A.load(tb_frag_A);
		++iterator_A;

		FragmentB tb_frag_B;
		iterator_B.load(tb_frag_B);
		++iterator_B;

		this->smem_iterator_A_.store(TransformA::impl(tb_frag_A));
		this->smem_iterator_B_.store(TransformB::impl(tb_frag_B));

		advance_smem_write_stage();
	}

	NIHILUS_DEVICE void gmem_wait() {
		__syncthreads();
	}


	NIHILUS_DEVICE void gemm_iters(uint64_t gemm_k_iterations, FragmentC& accum, IteratorA& iterator_A, IteratorB& iterator_B) {
		using WarpFragmentA = typename Operator::FragmentA;
		using WarpFragmentB = typename Operator::FragmentB;

		WarpFragmentA warp_frag_A[2];
		WarpFragmentB warp_frag_B[2];

		this->warp_tile_iterator_A_.set_kgroup_index(0);
		this->warp_tile_iterator_A_.load(warp_frag_A[0]);
		++this->warp_tile_iterator_A_;

		this->warp_tile_iterator_B_.set_kgroup_index(0);
		this->warp_tile_iterator_B_.load(warp_frag_B[0]);
		++this->warp_tile_iterator_B_;

		FragmentA tb_frag_A;
		FragmentB tb_frag_B;

		iterator_A.clear_mask(gemm_k_iterations <= 1);
		iterator_B.clear_mask(gemm_k_iterations <= 1);


		NIHILUS_GEMM_LOOP
		for (; gemm_k_iterations > 0; --gemm_k_iterations) {
#pragma unroll
			for (uint64_t warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
				if (warp_mma_k == Base::kWarpGemmIterations - 1) {
					this->smem_iterator_A_.store(transform_A_(tb_frag_A));

					this->smem_iterator_B_.store(transform_B_(tb_frag_B));

					gmem_wait();

					advance_smem_stages();
				}

				this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
				this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

				this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
				this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

				++this->warp_tile_iterator_A_;
				++this->warp_tile_iterator_B_;

				if (warp_mma_k == 0) {
					tb_frag_A.clear();
					iterator_A.load(tb_frag_A);
					++iterator_A;

					tb_frag_B.clear();
					iterator_B.load(tb_frag_B);
					++iterator_B;

					iterator_A.clear_mask(gemm_k_iterations <= 2);
					iterator_B.clear_mask(gemm_k_iterations <= 2);
				}

				warp_mma(accum, warp_frag_A[warp_mma_k % 2], warp_frag_B[warp_mma_k % 2], accum);
			}
		}
	}


	NIHILUS_DEVICE void wind_down() {
#pragma unroll
		for (uint64_t warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
			this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
			this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

			++this->warp_tile_iterator_A_;
			++this->warp_tile_iterator_B_;
		}

		if (smem_write_stage_idx == 0) {
			this->warp_tile_iterator_A_.add_tile_offset({ 0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations });
			this->warp_tile_iterator_B_.add_tile_offset({ -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0 });
		}
	}

	NIHILUS_DEVICE void operator()(uint64_t gemm_k_iterations, FragmentC& accum, IteratorA iterator_A, IteratorB iterator_B, FragmentC const& src_accum) {
		prologue(iterator_A, iterator_B, gemm_k_iterations);

		gmem_wait();

		accum = src_accum;

		gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
	}
};

template<typename Shape, typename Element, typename Layout, uint64_t AdvanceRank, typename ThreadMap, uint64_t AccessSize = ThreadMap::kElementsPerAccess, bool Gather = false,
	typename PermuteLayout = NoPermute>
class transform_threadblock_predicated_tile_iterator;

template<typename Shape_, typename Element_, uint64_t AdvanceRank, typename ThreadMap_, uint64_t AccessSize, bool Gather, typename PermuteLayout>
class transform_threadblock_predicated_tile_iterator<Shape_, Element_, PitchLinear, AdvanceRank, ThreadMap_, AccessSize, Gather, PermuteLayout> {
  public:
	static_assert(AdvanceRank == 0 || AdvanceRank == 1,
		"Specialization for pitch-linear iterator may advance along the "
		"contiguous(rank=0) or strided(rank=1) dimension.");

	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = PitchLinear;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;

	using Index		= typename Layout::Index;
	using LongIndex = typename Layout::LongIndex;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorView  = TensorView<Element, Layout>;
	using TensorCoord = typename Layout::TensorCoord;

	using Pointer		  = Element*;
	using NonConstPointer = typename std::remove_const<Element>::type*;

	using AccessType = nihilus::array<Element, AccessSize>;

	using TileAccessIterator = PredicatedTileAccessIterator<Shape, Element, Layout, kAdvanceRank, ThreadMap, AccessType, Gather, PermuteLayout>;

	static constexpr uint64_t kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

	using Fragment = nihilus::array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

	using Mask = typename TileAccessIterator::Mask;

	class Params {
	  public:
		using Base = typename TileAccessIterator::Params::Base;

		friend transform_threadblock_predicated_tile_iterator;

	  public:
		typename TileAccessIterator::Params params_;

	  public:
		NIHILUS_HOST_DEVICE Params(Layout const& layout) : params_(layout) {
		}

		Params() = default;

		NIHILUS_HOST_DEVICE Params(Base const& base) : params_(base) {
		}
	};

  public:
	using BytePointer = char*;

  public:
	TileAccessIterator address_iterator_;

  public:
	transform_threadblock_predicated_tile_iterator() = default;

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator(Params const& params, Pointer pointer, TensorCoord extent, uint64_t thread_id,
		TensorCoord const& threadblock_offset, uint64_t const* indices = nullptr)
		: address_iterator_(params.params_, pointer, extent, thread_id, threadblock_offset, indices) {
	}

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator(Params const& params, Pointer pointer, TensorCoord extent, uint64_t thread_id)
		: transform_threadblock_predicated_tile_iterator(params, pointer, extent, thread_id, { 0, 0 }) {
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		address_iterator_.add_pointer_offset(pointer_offset);
	}

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator& operator++() {
		if (kAdvanceRank)
			address_iterator_.add_tile_offset({ 0, 1 });
		else
			address_iterator_.add_tile_offset({ 1, 0 });

		return *this;
	}

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator operator++(int32_t) {
		transform_threadblock_predicated_tile_iterator self(*this);
		operator++();
		return self;
	}

	NIHILUS_HOST_DEVICE void clear_mask(bool enable = true) {
		address_iterator_.clear_mask(enable);
	}

	NIHILUS_HOST_DEVICE void enable_mask() {
		address_iterator_.enable_mask();
	}

	NIHILUS_HOST_DEVICE void set_mask(Mask const& mask) {
		address_iterator_.set_mask(mask);
	}

	NIHILUS_HOST_DEVICE void get_mask(Mask& mask) {
		address_iterator_.get_mask(mask);
	}

	NIHILUS_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
		load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
	}

	NIHILUS_DEVICE void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
		AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
				NIHILUS_PRAGMA_UNROLL
				for (uint64_t v = 0; v < kAccessesPerVector; ++v) {
					uint64_t idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

					address_iterator_.set_iteration_index(idx);
					char const* byte_ptr = reinterpret_cast<char const*>(address_iterator_.get()) + byte_offset;

					AccessType const* access_ptr = reinterpret_cast<AccessType const*>(byte_ptr);

					global_load<typename AccessType, sizeof(AccessType)>(frag_ptr[idx], access_ptr, address_iterator_.valid());

					++address_iterator_;
				}
			}
		}
	}

	NIHILUS_DEVICE void load(Fragment& frag) {
		load_with_byte_offset(frag, 0);
	}

	NIHILUS_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
		store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
	}

	NIHILUS_DEVICE void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
		address_iterator_.set_iteration_index(0);
		AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

		NIHILUS_PRAGMA_UNROLL
		for (uint64_t s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
			NIHILUS_PRAGMA_UNROLL
			for (uint64_t c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
				NIHILUS_PRAGMA_UNROLL
				for (uint64_t v = 0; v < kAccessesPerVector; ++v) {
					uint64_t idx = v + kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

					char* byte_ptr		   = reinterpret_cast<char*>(address_iterator_.get()) + byte_offset;
					AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_ptr);

					if (address_iterator_.valid()) {
						*access_ptr = frag_ptr[idx];
					}
					++address_iterator_;
				}
			}
		}
	}

	NIHILUS_DEVICE void store(Fragment const& frag) {
		store_with_byte_offset(frag, 0);
	}
};


template<typename Shape_, typename Element_, uint64_t AdvanceRank, typename ThreadMap_, uint64_t AccessSize, bool Gather, typename PermuteLayout>
class transform_threadblock_predicated_tile_iterator<Shape_, Element_, RowMajor, AdvanceRank, ThreadMap_, AccessSize, Gather, PermuteLayout> {
  public:
	static_assert(AdvanceRank == 0 || AdvanceRank == 1,
		"Specialization for pitch-linear iterator may along advance along the "
		"contiguous(rank=0) or strided(rank=1) dimension.");

	using Shape							   = Shape_;
	using Element						   = Element_;
	using Layout						   = RowMajor;
	static constexpr uint64_t kAdvanceRank = AdvanceRank;
	using ThreadMap						   = ThreadMap_;

	using Index		= typename Layout::Index;
	using LongIndex = typename Layout::LongIndex;

	using TensorRef	  = TensorRef<Element, Layout>;
	using TensorView  = TensorView<Element, Layout>;
	using TensorCoord = typename Layout::TensorCoord;

	using Pointer		  = Element*;
	using NonConstPointer = typename std::remove_const<Element>::type*;

	using UnderlyingIterator = transform_threadblock_predicated_tile_iterator<PitchLinearShape<Shape::kColumn, Shape::kRow>, Element, PitchLinear, (kAdvanceRank == 0 ? 1 : 0),
		ThreadMap, AccessSize, Gather, PermuteLayout>;

	using AccessType = typename UnderlyingIterator::AccessType;

	using Fragment = nihilus::array<Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

	using Mask = typename UnderlyingIterator::Mask;

	class Params {
	  public:
		friend transform_threadblock_predicated_tile_iterator;

		typename UnderlyingIterator::Params params_;

	  public:
		Params() = default;

		NIHILUS_HOST_DEVICE Params(Layout const& layout) : params_(PitchLinear(layout.stride(0))) {
		}

		NIHILUS_HOST_DEVICE Params(typename UnderlyingIterator::Params::Base const& base) : params_(base) {
		}
	};

  public:
	UnderlyingIterator iterator_;

  public:
	transform_threadblock_predicated_tile_iterator() = default;

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator(Params const& params, Pointer pointer, TensorCoord extent, uint64_t thread_id,
		TensorCoord const& threadblock_offset, uint64_t const* indices = nullptr)
		: iterator_(params.params_, pointer, PitchLinearCoord(extent.column(), extent.row()), thread_id, PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row()),
			  indices) {
	}

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator(Params const& params, Pointer pointer, TensorCoord extent, uint64_t thread_id)
		: transform_threadblock_predicated_tile_iterator(params, pointer, extent, thread_id, { 0, 0 }) {
	}

	NIHILUS_HOST_DEVICE void add_pointer_offset(LongIndex pointer_offset) {
		iterator_.add_pointer_offset(pointer_offset);
	}

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator& operator++() {
		++iterator_;
		return *this;
	}

	NIHILUS_HOST_DEVICE transform_threadblock_predicated_tile_iterator operator++(int32_t) {
		transform_threadblock_predicated_tile_iterator self(*this);
		operator++();
		return self;
	}

	NIHILUS_HOST_DEVICE void clear_mask(bool enable = true) {
		iterator_.clear_mask(enable);
	}

	NIHILUS_HOST_DEVICE void enable_mask() {
		iterator_.enable_mask();
	}

	NIHILUS_HOST_DEVICE void set_mask(Mask const& mask) {
		iterator_.set_mask(mask);
	}

	NIHILUS_HOST_DEVICE void get_mask(Mask& mask) {
		iterator_.get_mask(mask);
	}

	NIHILUS_DEVICE void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
		iterator_.load_with_pointer_offset(frag, pointer_offset);
	}

	NIHILUS_DEVICE void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
		iterator_.load_with_byte_offset(frag, byte_offset);
	}

	NIHILUS_DEVICE void load(Fragment& frag) {
		load_with_pointer_offset(frag, 0);
	}

	NIHILUS_DEVICE void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
		iterator_.store_with_pointer_offset(frag, pointer_offset);
	}

	NIHILUS_DEVICE void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
		iterator_.store_with_byte_offset(frag, byte_offset);
	}

	NIHILUS_DEVICE void store(Fragment const& frag) {
		store_with_pointer_offset(frag, 0);
	}
};


template<uint64_t M_, uint64_t K_, typename ElementA, typename LayoutA, uint64_t kAlignmentA, typename ElementB, typename LayoutB, uint64_t kAlignmentB,
	typename ElementAccumulator, typename LayoutC, typename ArchTag, typename ThreadblockShape, typename WarpShape, typename InstructionShape, typename Operator, bool GatherA,
	bool GatherB, typename PermuteALayout, typename PermuteBLayout>
struct DefaultMma<M_, K_, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC, OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
	InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone, GatherA, GatherB, PermuteALayout, PermuteBLayout> {
	using MmaCore =
		typename DefaultMmaCore<ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator, LayoutC, OpClassSimt, 2, Operator>;

	using IteratorA = transform_threadblock_predicated_tile_iterator<MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA, LayoutA, 1,
		typename MmaCore::IteratorThreadMapA, kAlignmentA, GatherA, PermuteALayout>;

	using IteratorB = transform_threadblock_predicated_tile_iterator<MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB, LayoutB, 0,
		typename MmaCore::IteratorThreadMapB, kAlignmentB, GatherB, PermuteBLayout>;

	using ThreadblockMma = MmaPipelined<typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
		LayoutC, typename MmaCore::MmaPolicy>;
};

template<uint64_t M_, uint64_t K_, typename ElementA, typename LayoutA, uint64_t kAlignmentA, typename ElementB, typename LayoutB, uint64_t kAlignmentB, typename ElementC,
	typename LayoutC, typename ElementAccumulator, typename ArchTag, typename ThreadblockShape, typename WarpShape, typename EpilogueOutputOp, typename ThreadblockSwizzle,
	bool SplitKSerial, typename Operator, SharedMemoryClearOption SharedMemoryClear, bool GatherA, bool GatherB, bool ScatterD, typename PermuteDLayout, typename PermuteALayout,
	typename PermuteBLayout>
struct DefaultGemm {
	using Mma = typename DefaultMma<M_, K_, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC, OpClassSimt, Sm120, ThreadblockShape,
		WarpShape, GemmShape<1, 1, 1>, 2, Operator, false, SharedMemoryClear, GatherA, GatherB, PermuteALayout, PermuteBLayout>::ThreadblockMma;

	static constexpr uint64_t kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;

	using RegularEpilogue =
		typename DefaultEpilogueSimt<ThreadblockShape, typename Mma::Operator, EpilogueOutputOp, kEpilogueElementsPerAccess, ScatterD, PermuteDLayout>::Epilogue;

	using Epilogue = RegularEpilogue;

	using GemmKernel = Gemm<M_, K_, Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

template<typename Operator> NIHILUS_GLOBAL void Kernel(auto params) {
	extern __shared__ uint64_t SharedStorageBase[];
	typename Operator::SharedStorage* shared_storage = reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);

	Operator::impl(params, *shared_storage);
}

template<uint64_t M_, uint64_t K_, typename ElementA_, typename ElementB_, typename ElementC_, typename ElementAccumulator_ = ElementC_, typename OperatorClass_ = OpClassSimt,
	typename ArchTag_			 = Sm120,
	typename ThreadblockShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::ThreadblockShape,
	typename WarpShape_			 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::WarpShape,
	typename InstructionShape_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::InstructionShape,
	typename EpilogueOutputOp_	 = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::EpilogueOutputOp,
	typename ThreadblockSwizzle_ = typename GemmIdentityThreadblockSwizzle<M_, K_>,
	uint64_t Stages				 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kStages,
	uint64_t AlignmentA			 = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentA,
	uint64_t AlignmentB = DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentB, bool SplitKSerial = false,
	typename Operator_ = typename DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::Operator, bool GatherA = false,
	bool GatherB = false, bool ScatterD = false, typename PermuteDLayout = NoPermute>
class device_gemm {
  public:
	static constexpr uint64_t kM							   = M_;
	static constexpr uint64_t kK							   = K_;
	using ElementA											   = ElementA_;
	using LayoutA											   = RowMajor;
	using TensorRefA										   = TensorRef<ElementA const, LayoutA>;
	using ElementB											   = ElementB_;
	using LayoutB											   = RowMajor;
	using TensorRefB										   = TensorRef<ElementB const, LayoutB>;
	using ElementC											   = ElementC_;
	using LayoutC											   = RowMajor;
	using TensorRefC										   = TensorRef<ElementC const, LayoutC>;
	using TensorRefD										   = TensorRef<ElementC, LayoutC>;
	using ElementAccumulator								   = ElementAccumulator_;
	using OperatorClass										   = OperatorClass_;
	using ArchTag											   = ArchTag_;
	using ThreadblockShape									   = ThreadblockShape_;
	using WarpShape											   = WarpShape_;
	using InstructionShape									   = InstructionShape_;
	using EpilogueOutputOp									   = EpilogueOutputOp_;
	using ThreadblockSwizzle								   = ThreadblockSwizzle_;
	using Operator											   = Operator_;
	static constexpr uint64_t kStages						   = Stages;
	static constexpr uint64_t kAlignmentA					   = AlignmentA;
	static constexpr uint64_t kAlignmentB					   = AlignmentB;
	static constexpr bool kSplitKSerial						   = SplitKSerial;
	static constexpr SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone;
	static constexpr uint64_t kTiledM						   = (kM + ThreadblockShape::kM - 1) / ThreadblockShape::kM;
	static constexpr uint64_t kTiledK						   = (kK + ThreadblockShape::kK - 1) / ThreadblockShape::kK;

	using GemmKernel =
		typename DefaultGemm<M_, K_, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC, LayoutC, ElementAccumulator, ArchTag, ThreadblockShape, WarpShape,
			EpilogueOutputOp, ThreadblockSwizzle, SplitKSerial, Operator, SharedMemoryClear, GatherA, GatherB, ScatterD, PermuteDLayout, NoPermute, NoPermute>::GemmKernel;


	struct Arguments {
		uint64_t N;
		TensorRef<ElementA const, LayoutA> ref_A;
		TensorRef<ElementB const, LayoutB> ref_B;
		TensorRef<ElementC const, LayoutC> ref_C;
		TensorRef<ElementC, LayoutC> ref_D;
		typename EpilogueOutputOp::Params epilogue;
		static constexpr uint64_t split_k_slices{ 1 };


		NIHILUS_HOST_DEVICE Arguments() : N(0) {
		}

		NIHILUS_HOST_DEVICE Arguments(uint64_t N_, TensorRef<ElementA const, LayoutA> ref_A_, TensorRef<ElementB const, LayoutB> ref_B_, TensorRef<ElementC const, LayoutC> ref_C_,
			TensorRef<ElementC, LayoutC> ref_D_)
			: N(N_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_), epilogue(typename EpilogueOutputOp::Params()) {
		}

		NIHILUS_HOST_DEVICE GemmCoord<M_, K_> problem_size() const {
			return GemmCoord(N);
		}
	};

	static constexpr auto grid_shape{ [] {
		return GemmIdentityThreadblockSwizzle<M_, K_>::template get_tiled_shape<1>(GemmCoord<ThreadblockShape::kM, ThreadblockShape::kK>{ ThreadblockShape::kN });
	}() };

	using grid_shape_type = std::remove_cvref_t<decltype(grid_shape)>;

  public:
	using params_type = typename GemmKernel::template Params<grid_shape_type::M, grid_shape_type::K>;

  public:
	NIHILUS_HOST static params_type initialize(Arguments const& args) {
		uint64_t problem_size_n{ args.N };

		grid_shape.N = (problem_size_n + ThreadblockShape::kN - 1) / ThreadblockShape::kN;

		params_type params_ = params_type{ problem_size_n, grid_shape, args.ref_A.non_const_ref(), args.ref_B.non_const_ref(), args.ref_C.non_const_ref(), args.ref_D,
			args.epilogue, nullptr, nullptr, nullptr, nullptr };

		return params_;
	}		

	NIHILUS_HOST static Status impl(Arguments const& args) {
		params_type params = initialize(args);

		dim3 grid = ThreadblockSwizzle::get_grid_shape(params.grid_tiled_shape);
		dim3 block(GemmKernel::kThreadCount, 1, 1);
		cudaError_t result;

		uint64_t smem_size = uint64_t(sizeof(typename GemmKernel::SharedStorage));

		Kernel<GemmKernel><<<grid, block, smem_size>>>(params);

		result = cudaGetLastError();

		return result == cudaSuccess ? Status::kSuccess : Status::kErrorInternal;
	}
};

int32_t main(int64_t argc, char** argv) {
	using nihilus_gemm = device_gemm<4096, 4096, float, float, float>;
	float* ptr{};
	//nihilus_gemm::impl({ 4096ull, { ptr, 4096ull }, { ptr, 4096ull }, { ptr, 4096ull }, { ptr, 4096ull } });
	const nihilus::cli_params cli_args_01 = nihilus::harbinger<model_config_01>::parse_cli_arguments(argc, argv);
	const nihilus::cli_params cli_args_02 = nihilus::harbinger<model_config_02>::parse_cli_arguments(argc, argv);
	nihilus::aligned_vector<std::unique_ptr<nihilus::model_base>> models{};
	models.emplace_back(nihilus::harbinger<model_config_01>::parse_model_graph_data(cli_args_01));
	models.emplace_back(nihilus::harbinger<model_config_02>::parse_model_graph_data(cli_args_02));
	while (true) {
		for (auto& value: models) {
			value->process_input();
		}
	}
	return 0;
}