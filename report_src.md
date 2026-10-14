# CS6101 PA2 — Approximate Nearest Neighbour Search with Vamana / DiskANN

*Compulsory part (dataset preparation, Parts 1–3). Extra credit not attempted.*

## 0. Environment and measurement protocol

| item | value |
|---|---|
| CPU | 2 × AMD EPYC 7402 24-Core (48 cores / 96 hardware threads total), 251 GiB RAM, shared lab server |
| Storage | data and indices on NVMe SSD (Crucial CT1000P3PSSD8) |
| Software | Python 3.10.16, NumPy 2.2.6, FAISS-cpu 1.15.1, Numba 0.68.0 (OpenMP threading layer), scikit-learn 1.7.2 |
| Build threads | 48 for both FAISS HNSW and Vamana |
| Query threads | **1** for every method (FAISS `omp_set_num_threads(1)`; the Vamana search kernel is single-threaded) |
| Query mode | **one query at a time** (batch size 1), issued from Python for every method; k = 10 |
| Timing subset | fixed random subset of 1000 queries (seed 1234) for SIFT1M/Quora/NQ/SIFT100K. Recall is always computed over the **full** query set |
| Timing procedure | 100 untimed warm-up queries, then 5 timed passes over the 1000 queries; reported mean latency = median of the 5 per-pass means; p50/p95 are from the fastest pass; QPS = 1 / mean latency (single thread). One-time index loading is excluded |

The server is shared with other users, so the first runs were affected by other people's jobs. All latencies reported here were therefore **re-measured in one session by `retime.py`**. That process was pinned to the least-loaded core, and exact, HNSW and Vamana were timed back to back for each dataset. Recall, distance counts, build times and sizes come from the original runs, which are deterministic. Build times are wall-clock times on the shared machine and may vary by roughly ±20 % from run to run. For example, the same SIFT100K reference build took 10 s, 19.6 s and 18.6 s in three separate runs.

## 1. Dataset preparation

* **SIFT1M:** the texmex ftp mirror was unreachable from the server. I used the identical copy distributed by ann-benchmarks (`sift-128-euclidean.hdf5`), which has the same 1M base vectors, 10K queries and supplied top-100 ground truth. L2 distance.
* **SIFT100K:** the first 100,000 SIFT1M base vectors with all 10,000 SIFT queries. Exact top-100 ground truth was computed with FAISS `IndexFlatL2`.
* **Quora and NQ:** corpus embeddings and test-split query embeddings come from `CohereLabs/beir-embed-english-v3` (1024-d). I fit PCA on corpus vectors only, using a uniform random sample of 500,000 vectors (seed 42; for Quora this is 500K of 523K). The same fitted PCA projects both corpus and queries to 128 dimensions. The results are cast to float32 and L2-normalised. Exact top-100 ground truth comes from FAISS `IndexFlatL2` over these 128-d vectors, is saved to disk (`*_gt.npy`) and is reused everywhere. BEIR qrels are not used.

{{Dataset summary}}

## Part 1 — Exact search and HNSW with FAISS

**1a. Exact kNN.** I used `IndexFlatL2`. On SIFT1M, the FAISS results match the supplied ground truth for 99.26 % of top-1 answers and 99.935 % of top-10 entries. I checked every one of the 74 top-1 disagreements, and each one is an exact distance tie: SIFT1M contains duplicate base vectors, so tied neighbours can legitimately be ordered differently. As required, I keep the supplied ground truth for SIFT1M. Because of these ties, **no method can exceed about 0.993 1-recall@1 on SIFT1M**; even exact search scores 0.9926. Quora and NQ have the same effect on a smaller scale (near-duplicate questions and passages), which is why exact search shows 0.9999 or 0.99997 instead of 1.

**1b. HNSW.** I used `IndexHNSWFlat` with M = 32 (64 links on layer 0) and efConstruction = 200. These are common "good quality" defaults and were not tuned. I measured seven operating points, efSearch ∈ {10, 16, 32, 64, 128, 256, 512}. Index size is the size of the file written by `faiss.write_index`, which includes the vectors.

{{Part 1 — sift1m}}

{{Part 1 — quora}}

{{Part 1 — nq}}

HNSW recall vs. latency: `plots/part1_hnsw_sift1m.png`, `plots/part1_hnsw_quora.png`, `plots/part1_hnsw_nq.png`.

![](plots/part1_hnsw_sift1m.png) ![](plots/part1_hnsw_quora.png) ![](plots/part1_hnsw_nq.png)

Quora is by far the easiest dataset: ef = 10 already gives 0.97 10-recall@10. NQ is the hardest, because it has 2.7M points and the PCA-reduced text embeddings have low contrast between neighbours.

## Part 2 — Vamana implementation

### Implementation (`vamana.py`)

* **Language and libraries:** Python driver. All performance-critical code (GreedySearch, RobustPrune, the construction phases and batched search) is in Numba `@njit(fastmath=True)` functions compiled to native SIMD code, parallelised with `prange` on the OpenMP layer. FAISS is used only for the exact ground truth and the HNSW baseline. No existing Vamana or DiskANN code is used.
* **Graph representation:** a fixed-capacity adjacency matrix `nbrs: int32[N, R]` plus `deg: int32[N]`. Only the first `deg[i]` entries of a row are valid. Serialized form (`.graph`): header `[N, R, start]` (int32), then `deg`, then `nbrs`. The file is N·(R+1)·4 B.
* **GreedySearch:** follows the 2a pseudo-code. The candidate set C is a sorted array of at most L `(dist, id, expanded)` entries. New neighbours are inserted by binary search and shift, and the entry that falls past L is dropped. A cursor tracks the closest unexpanded candidate. The algorithm records the visited (expanded) set V with distances (used by construction) and counts **vertices expanded** and **distance evaluations**. Only previously unseen query-to-database distances are counted, because a per-thread "seen" stamp array prevents recomputing a distance.
* **RobustPrune:** follows the 2b pseudo-code. The candidate pool is V ∪ N_out(p), deduplicated with p removed, then sorted by distance to p. It repeatedly takes the closest surviving p\*, then removes every p′ with α·d(p\*, p′) ≤ d(p, p′), stopping at R. All distances are squared L2, so the test is evaluated as α²·d²(p\*, p′) ≤ d²(p, p′). This is exactly equivalent to the Euclidean condition.
* **Construction:** a random R-regular directed graph (exactly R distinct non-self out-neighbours per vertex, seeded). The entry point s is the database point nearest the dataset mean, used as the medoid approximation. There are two passes over a seeded random permutation: α = 1, then the chosen α. Every inserted point gets GreedySearch(s, x_p, L_build), then RobustPrune(p, V), then reverse edges j ← p, with RobustPrune(j, …) when |N_out(j) ∪ {p}| > R.
* **Parallelisation (lock-free, deterministic):** each pass processes the permutation in batches. In the first pass, batch sizes double from 1 up to min(2 % N, 50,000), similar to ParlayANN's prefix doubling; the second pass uses the maximum batch size. Each batch runs four phases:
  1. In parallel, each point runs GreedySearch and RobustPrune against the current graph and writes its new out-list into a private buffer.
  2. In parallel, these lists are written into the graph.
  3. Reverse edges are grouped by target j with a stable sort.
  4. In parallel over distinct j, the union is computed and pruned if it exceeds R.

  Each graph row is written by exactly one thread per phase, so no locks are needed, and the build is reproducible for a fixed seed. With batch size 1 this is literally the sequential pseudo-code. I checked this on a 10K subset: the strictly sequential build (batch size 1) and the batched build both give 10-recall@10 = 0.9965 at L_search = 20, with nearly the same average degree (46.99 vs 47.08).
* **Random seed:** 0, used for both graph initialisation and the permutations.
* **Major optimizations:** contiguous float32 vectors; compact int32 adjacency; distance loops compiled by Numba with fastmath (auto-vectorised AVX2); per-thread reusable buffers (the "seen" stamp arrays avoid clearing an N-sized visited set for each search; candidate, visited and pool buffers are allocated once per build); a sorted-array candidate list with binary-search insertion; lock-free batch-parallel construction on 48 threads; parallel `prange` search for recall evaluation; graph and indices persisted and reused.

### Changes from the reference configuration

None. All full graphs use **R = 64, L_build = 100, α = 1.2**. That configuration builds in minutes and gives HNSW-level recall, so there was no need to change it.

### Scaling study (before full builds)

{{Part 2 — build scaling (R=64, L_build=100, α=1.2)}}

From 100K to 500K points, build time grows by 9.4× on SIFT and 6.8× on NQ for a 5× larger N. That is slightly super-linear, which matches the expected O(N log N)-like behaviour: search paths get longer as the graph grows. Extrapolating predicted about 400 s for SIFT1M and about 10–12 min for NQ (2.68M), so I judged the implementation ready for the full builds. The measured full builds took 340 s and 699 s.

### Full-dataset build statistics

{{Part 2 — full builds}}

* **Out-degree:** the build asserts that `deg.max() ≤ R`. The maximum out-degree equals exactly 64 on every graph, and no vertex exceeds R. Average degrees are 57–61 because RobustPrune with α = 1.2 often stops before reaching R.
* **Serialized size:** "graph file" is the adjacency only. "Graph + vectors" adds the 128-d float32 vectors needed at query time, which makes it the in-memory index size comparable to FAISS HNSW.
* **Peak build memory:** `ru_maxrss` of the dedicated build process. It includes the loaded vectors, the graph, 48 per-thread N-sized `uint32` "seen" arrays (48·N·4 B, which is 0.5 GiB for NQ), and the Python/Numba runtime (about 150 MiB). The build time excludes Numba JIT compilation, because a warm-up build runs first on 3,000 points.

### SIFT100K single-query sanity check (reference graph, L_search = 50)

{{Part 2 — SIFT100K single-query sanity check}}

## Part 3a — Parameter study on SIFT100K

Graphs: reference (R = 64, α = 1.2), R ∈ {32, 96} with α = 1.2, and α ∈ {1.0, 1.4} with R = 64. All use L_build = 100. Each graph is swept over L_search ∈ {10, 20, 30, 50, 75, 100, 150, 200}.

{{Part 3a — SIFT100K operating points}}

![](plots/part3a_sift100k.png)

*Left: (i) 10-recall@10 vs. L_search. Middle: (ii) 10-recall@10 vs. mean latency. Right: 10-recall@10 vs. vertices expanded per query.*

**Effect of L_search.** L_search is the only query-time knob. The number of vertices expanded is almost exactly L_search + 3 to 5 for every graph, because the search terminates only once all L candidates in the list have been expanded. Distance evaluations grow roughly linearly with L, and latency follows. Recall rises steeply up to L ≈ 30–50 and then saturates: going from L = 100 to 200 doubles the cost for ≤ 0.0004 recall on the reference graph.

**Effect of R.** A larger R gives every expansion more out-edges. That raises the cost per expansion (more distance evaluations: about 1.9K at L = 50 for R = 96 vs. 0.9K for R = 32), but it also raises the quality of each expansion. At the same L, the R = 96 graph reaches clearly higher recall than R = 64, which in turn beats R = 32. For example, at L = 20 the 10-recall@10 values are 0.989, 0.982 and 0.917. In latency terms, R = 32 is fastest only in the low-recall regime. To reach 10-recall@10 ≥ 0.99, R = 32 needs L = 100 (0.19 ms), while R = 64 needs L = 30 (0.12 ms) and R = 96 needs L = 30 (0.12 ms). Beyond R = 64 the gains are marginal, while memory and build time increase.

**Why more outgoing edges can mean fewer expansions.** Each expansion of a high-degree vertex evaluates more neighbours, so the candidate list fills with good candidates sooner. More importantly, the graph has smaller effective diameter and fewer local dead ends. GreedySearch can then move toward the query in larger steps and is less likely to get stuck in a local minimum where none of the current vertex's few neighbours is closer. With R = 96, 23 expansions reach 0.989 recall; with R = 32, about 103 expansions are needed to reach 0.991. Total work is roughly (expansions) × (degree). When the extra edges let the search converge in far fewer hops, a high-degree graph can win overall, even though each hop costs more.

**Effect of α.** With α = 1.0, RobustPrune keeps only "non-dominated" short edges, so the graph is very sparse (average degree 15.8). It is cheap per expansion, but it has no long-range shortcuts, so it needs many expansions: 0.903 recall at L = 20, against 0.982 for α = 1.2. Its latency curve is competitive only because each expansion is so cheap on this small, easy 100K dataset. With α = 1.2, more long-range edges survive (average degree 53), and recall per expansion improves sharply. With α = 1.4, almost every vertex saturates at R (average degree 62.6), but recall *drops* compared with α = 1.2 (0.964 vs. 0.982 at L = 20). Pruning has become so lax that the R slots fill with many long, redundant edges in similar directions, displacing short edges needed for the final local refinement. α trades navigability (long edges) against local precision (short edges), and on SIFT100K α = 1.2 is the best of the three.

## Part 3b — Full-dataset comparison

Each full dataset uses one Vamana graph (R = 64, L_build = 100, α = 1.2), with L_search swept on the finished graph. The FAISS numbers come from Part 1 and were re-timed in the same session.

{{Part 3b — sift1m Vamana sweep (R=64, L_build=100, α=1.2)}}

{{Part 3b — quora Vamana sweep (R=64, L_build=100, α=1.2)}}

{{Part 3b — nq Vamana sweep (R=64, L_build=100, α=1.2)}}

![](plots/part3b_sift1m.png) ![](plots/part3b_quora.png) ![](plots/part3b_nq.png)

### Compact comparison at comparable recall

{{COMPACT}}

### Discussion

**Recall vs. latency.** Both graph indices are 70–340× faster than exact search at ≥ 0.95 10-recall@10, and still 70–180× faster at ≥ 0.99. The two curves lie close together on all three datasets.

* **SIFT1M and Quora:** the two methods are within about ±10 % of each other. My Vamana is slightly ahead at the 0.95 and 0.99 targets: SIFT1M at 0.99, Vamana L = 75 takes 0.35 ms vs. HNSW ef = 128 at 0.39 ms; Quora at 0.99, 0.091 ms vs. 0.099 ms. HNSW is slightly ahead at 0.995 on SIFT1M (0.39 ms vs. 0.43 ms). On Quora's low-recall end, Vamana's smallest setting (L = 10) already reaches 0.989, so its cheapest point is more accurate but a little slower than HNSW ef = 10.
* **NQ, low recall:** Vamana degrades more gracefully. L = 10 gives 0.865 at 0.092 ms, while HNSW ef = 10 gives 0.824 at 0.080 ms.
* **NQ, high recall:** HNSW is clearly better. It is 1.5× faster at 0.99 (0.47 ms vs. 0.71 ms), and it reaches 0.995, which Vamana does not reach within L_search ≤ 300 (0.9948 at L = 300).

Two factors explain HNSW's advantage at the high-recall end:

1. **Implementation:** FAISS's search loop is hand-tuned C++ with SIMD distance kernels and prefetching. My kernel is Numba-generated and costs about 120–130 ns per distance evaluation, including candidate-list maintenance.
2. **Algorithm:** HNSW's upper layers deliver a query-specific starting point close to the target. Vamana always starts from one fixed medoid and spends part of its L budget walking from the dataset centre. This hurts most on the large, high-intrinsic-dimension NQ set.

Vamana's distance-evaluation and expansion counters are given in the tables. Expansions are about L_search + 4 to 8. Each expansion performs about 25–50 new distance evaluations: more at small L, fewer at large L, where more neighbours have already been seen. These counts are independent of machine noise.

**Build time.** On 48 threads, Vamana takes 340 s, 159 s and 699 s for SIFT1M, Quora and NQ, against 185 s, 101 s and 458 s for FAISS HNSW (efConstruction = 200). Vamana builds are therefore about 1.5–1.8× slower. Two reasons:

* Vamana does **two full passes**, each with a search at L_build = 100 per point.
* RobustPrune is re-run on every overflowing reverse neighbour list, which costs O(|C|·R) distances per prune. This is the dominant cost of pass 2: pass 2 takes about 1.6–1.9× as long as pass 1.

HNSW builds in a single pass. Using α = 1.0, a smaller L_build, or a slack factor (allowing up to 1.3R before pruning, as the DiskANN code does) would cut Vamana build time at some cost in quality.

**Index size.** Vamana stores a flat single-layer graph: N·(R+1)·4 B, which is 248 MiB for SIFT1M and 665 MiB for NQ. Adding the 128-d float32 vectors gives 736 MiB and 1,974 MiB. FAISS HNSW stores 748 MiB and 2,005 MiB: 2M links on layer 0, plus upper levels, plus per-level offsets, plus the vectors. The two are therefore about the same size at R = 2M = 64. Vamana's fixed-stride layout is what makes DiskANN's SSD layout possible: a vertex's vector and its neighbour list can be placed in one fixed-size disk record addressed by ID, with no hierarchy.

**Overall.** My Vamana implementation matches FAISS HNSW on SIFT1M and Quora at the same index size, and on NQ it is up to 1.5× slower at the highest recall targets. All query-side work runs in Numba rather than hand-tuned C++. The cost is a 1.5–1.8× longer build.

## Files

| file | purpose |
|---|---|
| `common.py` | paths, recall@k, timing protocol, helpers |
| `prepare_data.py` | SIFT1M/SIFT100K/Quora/NQ preparation, PCA, exact GT |
| `part1_faiss.py` | Part 1: FAISS exact + HNSW |
| `vamana.py` | **own** GreedySearch, RobustPrune, Vamana build, search, serialization |
| `part2_build.py` | Part 2: scaling study, full builds, sanity check |
| `part3.py` | Part 3a parameter study, Part 3b comparison |
| `retime.py` | one-session re-timing of all latencies |
| `make_tables.py`, `build_report.py` | results JSON → tables → this report |
| `run_all.sh` | runs the full pipeline after `prepare_data.py` |
| `results/*.json`, `plots/*.png`, `logs/` | raw results, figures, logs |

Reproduce:

```
python3.10 -m venv .venv && .venv/bin/pip install numpy faiss-cpu numba scikit-learn matplotlib pandas pyarrow huggingface_hub psutil h5py
python prepare_data.py all && ./run_all.sh && python retime.py && python make_tables.py && python build_report.py
```

Datasets and indices live in `data/`, which is a symlink to the NVMe disk and is not included in the submission.

## LLM usage

This assignment was implemented with the help of **Claude Code (Anthropic Claude, CLI agent)**. Given the assignment PDF, it set up the environment and downloaded and prepared the data. It wrote all scripts in this submission, including the Vamana implementation in `vamana.py`, ran all experiments, and drafted this report. While doing so it found and fixed a reverse-edge bug in which out-degree could exceed R. The student reviewed the work and is responsible for it.

*Shareable chat/session link(s): **TODO — add link(s) here before submission.***
