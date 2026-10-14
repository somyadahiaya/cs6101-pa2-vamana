"""Re-measure all query latencies in one session (the server is shared, so earlier runs overlapped with
other jobs).  The process is pinned to one core; for each dataset, exact, HNSW and Vamana operating
points are timed back-to-back with 5 repeats.  Recall / counters / build stats are left unchanged;
only the latency/QPS fields of the result JSONs are replaced, then plots are regenerated.

Usage: python retime.py [core_id]
"""
import os
import sys

import common
common.REPEATS = 5
os.sched_setaffinity(0, {int(sys.argv[1]) if len(sys.argv) > 1 else 7})
os.environ["NUMBA_NUM_THREADS"] = "1"

import faiss
import numpy as np

import part1_faiss
import part3
import vamana as V
from common import INDEX, load_dataset, load_json, save_json, time_queries, timing_subset
from part2_build import index_prefix

faiss.omp_set_num_threads(1)
LAT = ("mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "qps", "timing_queries", "timing_repeats")


def upd(row, tm):
    for k in LAT:
        row[k] = tm[k]


def vamana_rows(name, rows, xb, xq_t):
    cache = {}
    for r in rows:
        key = (r["R"], r["L_build"], r["alpha"])
        if key not in cache:
            nbrs, deg, start, _ = V.load_index(index_prefix(name, *key))
            cache = {key: V.Searcher(xb, nbrs, deg, start)}
        s, L = cache[key], r["L_search"]
        upd(r, time_queries(lambda q: s.search(q[0], 10, L), xq_t))
        print(name, "vamana", key, L, round(r["mean_latency_ms"], 4), flush=True)


def main():
    for name in ("sift1m", "quora", "nq"):
        xb, xq, gt = load_dataset(name)
        xq_t = xq[timing_subset(len(xq))]
        p1 = load_json(f"part1_{name}.json")
        flat = faiss.read_index(os.path.join(INDEX, f"flat_{name}.faiss"))
        upd(p1["exact"], time_queries(lambda q: flat.search(q, 10), xq_t))
        del flat
        print(name, "exact", round(p1["exact"]["mean_latency_ms"], 3), flush=True)
        hp = p1["hnsw_params"]
        hnsw = faiss.read_index(os.path.join(INDEX, f"hnsw_{name}_M{hp['M']}_efc{hp['efConstruction']}.faiss"))
        for r in p1["hnsw"]:
            hnsw.hnsw.efSearch = r["efSearch"]
            upd(r, time_queries(lambda q: hnsw.search(q, 10), xq_t))
            print(name, "hnsw", r["efSearch"], round(r["mean_latency_ms"], 4), flush=True)
        del hnsw
        p1["timing_note"] = "latencies re-measured by retime.py (1 pinned core, 5 repeats)"
        save_json(p1, f"part1_{name}.json")
        part1_faiss.plot(p1)
        d = load_json(f"part3b_{name}.json")
        vamana_rows(name, d["vamana"], xb, xq_t)
        d["hnsw"], d["exact"] = p1["hnsw"], p1["exact"]
        save_json(d, f"part3b_{name}.json")
        part3.plot_3b(name, d)
    xb, xq, gt = load_dataset("sift100k")
    d = load_json("part3a_sift100k.json")
    vamana_rows("sift100k", d["rows"], xb, xq[timing_subset(len(xq))])
    save_json(d, "part3a_sift100k.json")
    part3.plot_3a(d["rows"])


if __name__ == "__main__":
    main()
