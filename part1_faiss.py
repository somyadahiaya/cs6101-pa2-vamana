"""Part 1: exact FAISS (IndexFlatL2) and FAISS HNSW (IndexHNSWFlat) on SIFT1M / Quora / NQ.

Recall is computed over the full query set (batched, multi-threaded search; results do not depend on
threading).  Latency is measured single-threaded with individually-issued queries on a fixed random
subset of 1000 queries (see common.time_queries).

Usage: python part1_faiss.py [dataset ...]
"""
import os
import sys

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (BUILD_THREADS, INDEX, PLOTS, Timer, cpu_info, file_size_mb, load_dataset, recall_at,
                    save_json, time_queries, timing_subset)

HNSW_M = 32
HNSW_EFC = 200
EF_SEARCH = [10, 16, 32, 64, 128, 256, 512]
K = 10


def metrics(I, gt):
    return {"1-recall@1": recall_at(I, gt, 1), "10-recall@10": recall_at(I, gt, 10)}


def run(name):
    xb, xq, gt = load_dataset(name)
    sub = timing_subset(len(xq))
    xq_t = xq[sub]
    res = {"dataset": name, **cpu_info(), "build_threads": BUILD_THREADS, "query_threads": 1,
           "query_mode": "individual queries (batch size 1)", "k": K}

    # ---- exact
    faiss.omp_set_num_threads(BUILD_THREADS)
    with Timer() as tb:
        flat = faiss.IndexFlatL2(xb.shape[1])
        flat.add(xb)
    flat_path = os.path.join(INDEX, f"flat_{name}.faiss")
    faiss.write_index(flat, flat_path)
    _, I = flat.search(xq, K)
    faiss.omp_set_num_threads(1)
    tm = time_queries(lambda q: flat.search(q, K), xq_t)
    res["exact"] = {"build_time_s": tb.dt, "index_size_MiB": file_size_mb(flat_path), **metrics(I, gt), **tm}
    print(name, "exact", res["exact"], flush=True)

    # ---- HNSW
    hnsw_path = os.path.join(INDEX, f"hnsw_{name}_M{HNSW_M}_efc{HNSW_EFC}.faiss")
    faiss.omp_set_num_threads(BUILD_THREADS)
    with Timer() as tb:
        hnsw = faiss.IndexHNSWFlat(xb.shape[1], HNSW_M)
        hnsw.hnsw.efConstruction = HNSW_EFC
        hnsw.add(xb)
    faiss.write_index(hnsw, hnsw_path)
    res["hnsw_params"] = {"M": HNSW_M, "efConstruction": HNSW_EFC, "build_time_s": tb.dt,
                          "index_size_MiB": file_size_mb(hnsw_path)}
    print(name, "hnsw build", res["hnsw_params"], flush=True)
    res["hnsw"] = []
    for ef in EF_SEARCH:
        hnsw.hnsw.efSearch = ef
        faiss.omp_set_num_threads(BUILD_THREADS)
        _, I = hnsw.search(xq, K)
        faiss.omp_set_num_threads(1)
        tm = time_queries(lambda q: hnsw.search(q, K), xq_t)
        row = {"efSearch": ef, "build_time_s": tb.dt, "index_size_MiB": res["hnsw_params"]["index_size_MiB"],
               **metrics(I, gt), **tm}
        res["hnsw"].append(row)
        print(name, "hnsw", row, flush=True)
    save_json(res, f"part1_{name}.json")
    plot(res)
    return res


def plot(res):
    h = res["hnsw"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([r["mean_latency_ms"] for r in h], [r["10-recall@10"] for r in h], "o-", label=f"FAISS HNSW (M={HNSW_M}, efC={HNSW_EFC})")
    for r in h:
        ax.annotate(f"ef={r['efSearch']}", (r["mean_latency_ms"], r["10-recall@10"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -10))
    ax.set_xscale("log")
    ax.set_xlabel("mean query latency (ms, 1 thread, log scale)")
    ax.set_ylabel("10-recall@10")
    ax.set_title(f"{res['dataset']}: HNSW recall vs latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"part1_hnsw_{res['dataset']}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["sift1m", "quora", "nq"]):
        run(n)
