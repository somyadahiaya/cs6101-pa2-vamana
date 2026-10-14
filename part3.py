"""Part 3: Vamana parameter study on SIFT100K (3a) and full-dataset comparison with FAISS (3b).

  python part3.py 3a
  python part3.py 3b [dataset ...]
"""
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "48")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vamana as V
from common import (PLOTS, cpu_info, load_dataset, load_json, recall_at, save_json, time_queries, timing_subset)
from part2_build import index_prefix, spawn_build

K = 10
L_SEARCH = [10, 20, 30, 50, 75, 100, 150, 200]
CONFIGS_3A = [  # (R, L_build, alpha) -- reference first
    (64, 100, 1.2), (32, 100, 1.2), (96, 100, 1.2), (64, 100, 1.0), (64, 100, 1.4)]


def eval_graph(name, R, Lb, alpha, Ls, xb, xq, gt):
    prefix = index_prefix(name, R, Lb, alpha)
    if not os.path.exists(prefix + ".graph"):
        spawn_build(name, R=R, L=Lb, alpha=alpha)
    nbrs, deg, start, meta = V.load_index(prefix)
    s = V.Searcher(xb, nbrs, deg, start)
    xq_t = xq[timing_subset(len(xq))]
    rows = []
    for L in Ls:
        I, nexp, ndist = V.search_batch(xb, nbrs, deg, start, xq, K, max(L, K))   # recall: all queries
        tm = time_queries(lambda q: s.search(q[0], K, L), xq_t)                  # latency: 1 thread
        row = {"R": R, "alpha": alpha, "L_build": Lb, "L_search": L,
               "1-recall@1": recall_at(I, gt, 1), "10-recall@10": recall_at(I, gt, K),
               **tm, "dist_evals_per_query": float(ndist.mean()), "vertices_expanded_per_query": float(nexp.mean()),
               "build_time_s": meta["build_time_s"], "avg_out_degree": meta["avg_out_degree"],
               "max_out_degree": meta["max_out_degree"], "graph_file_MiB": meta["graph_file_MiB"],
               "index_size_MiB": meta["graph_plus_vectors_MiB"], "peak_build_rss_MiB": meta["peak_rss_MiB"]}
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return rows


def part3a():
    xb, xq, gt = load_dataset("sift100k")
    rows = []
    for R, Lb, a in CONFIGS_3A:
        rows += eval_graph("sift100k", R, Lb, a, L_SEARCH, xb, xq, gt)
    save_json({"rows": rows, **cpu_info(), "query_threads": 1, "query_mode": "individual"}, "part3a_sift100k.json")
    plot_3a(rows)


def _cfg_label(R, a):
    return f"R={R}, α={a:g}"


def plot_3a(rows):
    cfgs = []
    for r in rows:
        if (r["R"], r["alpha"]) not in cfgs:
            cfgs.append((r["R"], r["alpha"]))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for R, a in cfgs:
        rr = [r for r in rows if r["R"] == R and r["alpha"] == a]
        ls = "-" if (R, a) == (64, 1.2) else "--"
        lab = _cfg_label(R, a) + (" (ref)" if (R, a) == (64, 1.2) else "")
        axs[0].plot([r["L_search"] for r in rr], [r["10-recall@10"] for r in rr], "o" + ls, ms=4, label=lab)
        axs[1].plot([r["mean_latency_ms"] for r in rr], [r["10-recall@10"] for r in rr], "o" + ls, ms=4, label=lab)
        axs[2].plot([r["vertices_expanded_per_query"] for r in rr], [r["10-recall@10"] for r in rr], "o" + ls, ms=4,
                    label=lab)
    axs[0].set_xlabel("L_search")
    axs[1].set_xlabel("mean query latency (ms, 1 thread)")
    axs[2].set_xlabel("vertices expanded / query")
    axs[1].set_xscale("log")
    for ax, t in zip(axs, ["(i) 10-recall@10 vs L_search", "(ii) 10-recall@10 vs mean latency",
                           "10-recall@10 vs graph expansions"]):
        ax.set_ylabel("10-recall@10")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.8, 1.002)
    axs[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("SIFT100K Vamana parameter study (L_build = 100)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "part3a_sift100k.png"), dpi=150)
    plt.close(fig)


def part3b(datasets):
    out = {}
    for name in datasets:
        xb, xq, gt = load_dataset(name)
        rows = eval_graph(name, 64, 100, 1.2, L_SEARCH + [300], xb, xq, gt)
        p1 = load_json(f"part1_{name}.json")
        out[name] = {"vamana": rows, "hnsw": p1["hnsw"], "hnsw_params": p1["hnsw_params"], "exact": p1["exact"]}
        save_json(out[name], f"part3b_{name}.json")
        plot_3b(name, out[name])
    return out


def plot_3b(name, d):
    fig, ax = plt.subplots(figsize=(6, 4.4))
    h, v, e = d["hnsw"], d["vamana"], d["exact"]
    ax.plot([r["mean_latency_ms"] for r in h], [r["10-recall@10"] for r in h], "o-",
            label=f"FAISS HNSW (M={d['hnsw_params']['M']}, efC={d['hnsw_params']['efConstruction']}; efSearch sweep)")
    ax.plot([r["mean_latency_ms"] for r in v], [r["10-recall@10"] for r in v], "s-",
            label="Vamana (R=64, L_build=100, α=1.2; L_search sweep)")
    ax.plot([e["mean_latency_ms"]], [e["10-recall@10"]], "k*", ms=12, label="FAISS exact (IndexFlatL2)")
    ax.set_xscale("log")
    ax.set_xlabel("mean query latency (ms, 1 thread, log scale)")
    ax.set_ylabel("10-recall@10")
    ax.set_title(f"{name}: recall vs latency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"part3b_{name}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "3a":
        part3a()
    else:
        part3b(sys.argv[2:] or ["sift1m", "quora", "nq"])
