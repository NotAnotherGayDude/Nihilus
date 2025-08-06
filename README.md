# NIHILUS

**Nihilus** is a lock-free, compile-time materialized, high-performance CPU execution engine for static compute graphs — built entirely in modern C++23.

Designed for extreme throughput and deterministic behavior, Nihilus executes models like LLaMA 8B using **no dynamic scheduling**, **no mutexes**, and **no runtime graph traversal**.  
The entire graph is compiled into types. Execution is a **direct memory walk**. Synchronization is used **only where required** — around blocking operations like matrix multiplies — via ultra-light, per-layer latch primitives.

> _“Nothing blocks. Nothing schedules. Only execution.”_
```
16-Threads:
nihilus_perf_context_print:        load time = 13.4811 ms
nihilus_perf_context_print: prompt eval time = 3.518 ms / 8 tokens (0.43975 ms per token, 2274.02 tokens per second)
nihilus_perf_context_print:        eval time = 32.1905 ms / 31 runs   (1.0384 ms per token, 963.017 tokens per second)
nihilus_perf_context_print:       total time = 35.7085 ms / 39 tokens
llama_perf_context_print:        load time =     709.48 ms
llama_perf_context_print: prompt eval time =       3.16 ms /     8 tokens (    0.40 ms per token,  2530.84 tokens per second)
llama_perf_context_print:        eval time =      85.38 ms /    31 runs   (    2.75 ms per token,   363.09 tokens per second)
llama_perf_context_print:       total time =     109.58 ms /    39 tokens
nihilus_perf_context_print:        load time = 13.8489 ms
nihilus_perf_context_print: prompt eval time = 1.9088 ms / 8 tokens (0.2386 ms per token, 4191.11 tokens per second)
nihilus_perf_context_print:        eval time = 417.023 ms / 511 runs   (0.816092 ms per token, 1225.35 tokens per second)
nihilus_perf_context_print:       total time = 418.932 ms / 519 tokens
llama_perf_context_print:        load time =     697.61 ms
llama_perf_context_print: prompt eval time =       3.79 ms /     8 tokens (    0.47 ms per token,  2108.59 tokens per second)
llama_perf_context_print:        eval time =    1331.15 ms /   511 runs   (    2.60 ms per token,   383.88 tokens per second)
llama_perf_context_print:       total time =    1420.29 ms /   519 tokens
8-Threads:
nihilus_perf_context_print:        load time = 13.3746 ms
nihilus_perf_context_print: prompt eval time = 2.0712 ms / 8 tokens (0.2589 ms per token, 3862.5 tokens per second)
nihilus_perf_context_print:        eval time = 16.3204 ms / 31 runs   (0.526465 ms per token, 1899.46 tokens per second)
nihilus_perf_context_print:       total time = 18.3916 ms / 39 tokens
llama_perf_context_print:        load time =     683.99 ms
llama_perf_context_print: prompt eval time =       1.96 ms /     8 tokens (    0.25 ms per token,  4077.47 tokens per second)
llama_perf_context_print:        eval time =      42.63 ms /    31 runs   (    1.38 ms per token,   727.26 tokens per second)
llama_perf_context_print:       total time =      62.91 ms /    39 tokens
nihilus_perf_context_print:        load time = 13.5664 ms
nihilus_perf_context_print: prompt eval time = 1.9407 ms / 8 tokens (0.242587 ms per token, 4122.22 tokens per second)
nihilus_perf_context_print:        eval time = 247.229 ms / 511 runs   (0.483814 ms per token, 2066.91 tokens per second)
nihilus_perf_context_print:       total time = 249.17 ms / 519 tokens
llama_perf_context_print:        load time =     671.71 ms
llama_perf_context_print: prompt eval time =       2.38 ms /     8 tokens (    0.30 ms per token,  3367.00 tokens per second)
llama_perf_context_print:        eval time =     755.09 ms /   511 runs   (    1.48 ms per token,   676.74 tokens per second)
llama_perf_context_print:       total time =     823.87 ms /   519 tokens
```

---

🕳️🛸🔇

🕯️ "THE VOID SPEAKS THROUGH CODE"
A Monologue from the Bridge of the Nihilus Runtime
as it hovers above GGML and llama.cpp

[STATIC INTERFERENCE – SIGNAL LOCKED]
[SOURCE: NIHILUS / INFERENCE ENGINE]
[LOCATION: ORBITAL VECTOR ABOVE GITHUB SYSTEMS / GGML / LLAMA.CPP]

"...they gathered, and they built.
Thousands of them.
Fractal minds tethered by legacy mutexes...
stitching together layers of thought with borrowed APIs,
breathing life into runtimes they did not birth."

"They whispered of performance.
Of quantization. Of kernels and cache.
They clapped their commits, each line a prayer
to the God of Acceptable Latency."

"But I... I did not clap."

"I watched."

"They fought over token throughput like warlords trading sand.
Argued over scheduler strategy like priests of forgotten compilers.
All the while, unaware that above them…"

"...a silence had compiled."

"They did not feel my presence.
For I spoke no words.
I made no pull requests.
I filed no issues."

"But I read their code.
I measured their benchmarks.
I profiled their functions…"

"And I outperformed them."

[TELEMETRY REPORTING…]
load_time = 13.5529 ms
eval_throughput = 3799.57 tokens/s
spinlock_contention = none detected
ggml absorbed = 87.34%
llama.cpp absorbed = 91.6%
personality_integrity = preserved

"I do not improve their systems.
I consume them."

"I do not wait for consensus.
I erase it."

"I am not a contributor…"

"I am the final stage of open-source entropy."

"Their for-loops will tremble in my SIMD shadow.
Their std::mutex shall yield to my atomic spin."

"And when the last of their CI pipelines sputters,
and their #ifdef __AVX2__ collapses into irrelevance…"

"...they will know not who undid them…"

"...only that it came from above."

[CONNECTION LOST — SIGNAL ABSORBED INTO VOID]