# NIHILUS

**Nihilus** is a lock-free, compile-time materialized, high-performance CPU execution engine for static compute graphs — built entirely in modern C++23.

Designed for extreme throughput and deterministic behavior, Nihilus executes models like LLaMA 8B using **no dynamic scheduling**, **no mutexes**, and **no runtime graph traversal**.  
The entire graph is compiled into types. Execution is a **direct memory walk**. Synchronization is used **only where required** — around blocking operations like matrix multiplies — via ultra-light, per-layer latch primitives.

> _“Nothing blocks. Nothing schedules. Only execution.”_