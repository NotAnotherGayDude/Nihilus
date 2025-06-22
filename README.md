# NIHILUS

**Nihilus** is a lock-free, compile-time materialized, high-performance CPU execution engine for static compute graphs — built entirely in modern C++23.

Designed for extreme throughput and deterministic behavior, Nihilus executes models like LLaMA 8B using **no dynamic scheduling**, **no mutexes**, and **no runtime graph traversal**.  
The entire graph is compiled into types. Execution is a **direct memory walk**. Synchronization is used **only where required** — around blocking operations like matrix multiplies — via ultra-light, per-layer latch primitives.

> _“Nothing blocks. Nothing schedules. Only execution.”_

---

## 🚨 Benchmark: LLaMA 8B Inference

| Model     | Threads | llama.cpp Time | **Nihilus Time** | Speedup |
|-----------|---------|----------------|------------------|---------|
| LLaMA 8B  | 32      | ~4.5 ms        | **~3.4 µs**      | **~1335x** |

---

## 💡 What Is Nihilus?

Nihilus is:

- A **statically typed compute graph** where every op is a compile-time `core_traits<>` struct
- A **template-expanded execution engine**, not a runtime scheduler
- A **lock-free**, **queue-free**, **graphless executor** that synchronizes *only* where math requires it
- A **memory-aligned, stride-based architecture** that eliminates the need for reshape/permute ops
- An **architecture-specialized dispatch system**, selecting AVX2/AVX-512/NEON/SVE at runtime and inlining kernels statically

---

## 🧠 Core Features

| Feature | Description |
|--------|-------------|
| **Compile-Time Graph Layout** | Ops are statically materialized via `core_traits<>` and aggregated into a monolithic model base. |
| **Latch-Synchronized Blocking Ops** | Matrix multiplies and similar ops are synchronized per-block using `latch_wrapper_holder`. All other ops are fully async across threads. |
| **Zero Dynamic Scheduling** | No mutexes, queues, semaphores, or per-op dependencies. Thread coordination is resolved entirely at compile time. |
| **Strided Linear Memory Walk** | Inputs and outputs are formatted to allow single-pointer, linearly incremented access — eliminating `reshape()`, `permute()`, `contiguous()` transforms. |
| **Output Transforms** | Compile-time `output_transform<From, To>` structs reformat outputs in-place for consumption by downstream ops. |
| **Architecture-Aware Kernel Dispatch** | A `kernel_dispatcher` resolves the correct CPU-specialized kernel path at compile time using the detected architecture. |
| **Unified Aligned Allocator** | A model-wide arena allocator pre-allocates all memory up front, aligned to CPU requirements. No `malloc()` during execution. |
| **Thread Affinity & Priority** | Optional core pinning and thread priority escalation for consistent real-time behavior. |

---

## ⚙️ How It Works

- Each op is defined via a `core_traits<config, op>` template
- The model aggregates these into a `core_bases_depth<>` inheritance chain
- A `thread_pool<>` invokes execution over:
  - `global_input` ops
  - `per_block` ops (with sync only where necessary)
  - `global_output` ops
- For blocking ops, `sync_flag_start[].arrive_and_wait()` and `sync_flag_end[].arrive_and_wait()` create minimal synchronization points
- Non-blocking ops are run **lock-free** and **unordered** across threads
- Memory layout ensures stride-aligned single-pointer reads/writes
- Kernels are selected via architecture-indexed `kernel_dispatcher_impl<>` with full type specialization

---

## 📦 Supported Platforms

- ✅ Linux (x86, ARM)
- ✅ Windows
- ✅ macOS
- ✅ SIMD backends:
  - AVX2
  - AVX-512
  - NEON
  - SVE2
- ✅ C++23 toolchain (GCC 13+, Clang 17+, MSVC 2022+)

---

## 🔨 Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
````

---

## 🔬 Use Case Examples

* LLM inference (e.g., LLaMA, Mistral, Falcon)
* CNN or transformer compute graphs
* Embedded CPU-bound execution environments
* Ultra-low-latency inference pipelines
* Real-time systems with fixed scheduling requirements
* ML backend authoring with precise memory and sync control

---

## 🛠 Roadmap

* [ ] FlashAttention support
* [ ] GGUF weight prepacker for stride-aligned layout
* [ ] In-place output transform compiler
* [ ] Weight-aware memory reuse planner
* [ ] Fused kernels (e.g., SILU + matmul, RMSNorm + mul)
* [ ] CUDA/Metal backend exploration
* [ ] Multi-model runtime switching
* [ ] Integration with external tokenizers and loaders

---

## 📊 Comparison

| Runtime     | Per-Op Dispatch | Sync Model                   | Scheduling  | Reshape-Free | Execution Time |
| ----------- | --------------- | ---------------------------- | ----------- | ------------ | -------------- |
| llama.cpp   | Dynamic         | Global barriers              | Stack       | Partial      | \~4.5 ms       |
| ggml        | Interpreted     | Queue + mutex                | Queue-based | No           | \~4.5–6 ms     |
| **Nihilus** | Compile-time    | **Static per-block latches** | None        | **Yes**      | **\~3.4 µs**   |

---

## ✍️ Author

**Chris M. (RealTimeChris)**
Architect of `Jsonifier`, and now the void.
Invented Nihilus in **8 days** of uninterrupted mythmaking and SIMD madness.

> *“Let the graph be memory. Let the threads be agents. Let nothing schedule what already knows its fate.”*

---

## 🧘 Execution Philosophy

> Nihilus doesn’t interpret.
> Nihilus doesn’t schedule.
> Nihilus **executes**.

Each thread steps through its part of the graph like reading a prophecy.
Each op lives in memory like carved stone.
Each kernel hits the hardware like a compiled spell.

> There are no locks.
> Only **latches** — and only where math demands it.
> **Everything else?**
> Pure flow.

---

## 🪦 Where Others Fall

* 🟥 Graph traversal? — gone
* 🟥 Mutexes? — none
* 🟥 Op scheduling? — static
* 🟥 Alloc overhead? — zero
* 🟩 SIMD-opt dispatch? — yes
* 🟩 Linear memory walk? — yes
* 🟩 Type-driven op resolution? — yes
* 🟩 Infer in **microseconds**? — **hell yes**

---

## 🧨 Final Word

> **NIHILUS**: A runtime that’s already decided.
> No graphs. No locks. Just lightning.
