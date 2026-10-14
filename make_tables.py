"""Render results/*.json into Markdown tables (results/tables.md) that the report embeds."""
import os

from common import RESULTS, load_json


def fmt(v):
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    md = []
    ds = load_json("dataset_summary.json")
    md.append("## Dataset summary\n")
    rows = []
    for k in ("sift1m", "sift100k", "quora", "nq"):
        if k in ds:
            rows.append({"dataset": k, **ds[k]})
    md.append(table(rows, ["dataset", "corpus_size", "num_queries", "original_dim", "search_dim", "raw_float32_MiB"],
                    ["dataset", "corpus size", "#queries", "orig. dim", "search dim", "128-d float32 (MiB)"]))
    rows = [{"dataset": k, **ds[k]} for k in ("quora", "nq") if k in ds]
    md.append("\n\n" + table(rows, ["dataset", "pca_implementation", "pca_fit_sample_size", "pca_random_seed",
                                    "pca_explained_variance_128"],
                             ["dataset", "PCA implementation", "fit sample", "seed", "variance explained (128 comps)"]))
    if "sift1m" in ds:
        v = ds["sift1m"]["gt_verification"]
        md.append(f"\n\nSIFT1M FAISS-exact vs supplied GT: top-1 agreement {v['agree_top1']:.4f}, top-10 overlap "
                  f"{v['agree_top10_overlap']:.5f}, top-100 overlap {v['agree_top100_overlap']:.5f}; "
                  f"{v['n_top1_mismatch']} top-1 mismatches, all exact distance ties: "
                  f"{v['all_top1_mismatches_are_distance_ties']}.")

    cols = ["method", "1-recall@1", "10-recall@10", "mean_latency_ms", "p95_latency_ms", "qps", "build_time_s",
            "index_size_MiB"]
    hdr = ["method", "1-R@1", "10-R@10", "mean lat (ms)", "p95 lat (ms)", "QPS", "build (s)", "index (MiB)"]
    for name in ("sift1m", "quora", "nq"):
        p = os.path.join(RESULTS, f"part1_{name}.json")
        if not os.path.exists(p):
            continue
        r = load_json(f"part1_{name}.json")
        rows = [{"method": "FAISS exact (IndexFlatL2)", **r["exact"]}]
        rows += [{"method": f"HNSW efSearch={h['efSearch']}", **h} for h in r["hnsw"]]
        md.append(f"\n\n## Part 1 — {name}\n\nHNSW M={r['hnsw_params']['M']}, efConstruction="
                  f"{r['hnsw_params']['efConstruction']}, build threads {r['build_threads']}.\n\n" + table(rows, cols, hdr))

    if os.path.exists(os.path.join(RESULTS, "part2_scaling.json")):
        md.append("\n\n## Part 2 — build scaling (R=64, L_build=100, α=1.2)\n\n" + table(
            load_json("part2_scaling.json"), ["dataset", "N", "build_time_s", "avg_out_degree", "max_out_degree",
                                               "peak_rss_MiB"],
            ["dataset", "N", "build (s)", "avg deg", "max deg", "peak RSS (MiB)"]))
    if os.path.exists(os.path.join(RESULTS, "part2_full_builds.json")):
        fb = load_json("part2_full_builds.json")
        rows = [{"dataset": k, **v} for k, v in fb.items()]
        md.append("\n\n## Part 2 — full builds\n\n" + table(
            rows, ["dataset", "N", "R", "L_build", "alpha", "build_time_s", "avg_out_degree", "max_out_degree",
                   "min_out_degree", "graph_file_MiB", "graph_plus_vectors_MiB", "peak_rss_MiB",
                   "verified_max_out_degree_le_R"],
            ["dataset", "N", "R", "L_build", "α", "build (s)", "avg deg", "max deg", "min deg", "graph file (MiB)",
             "graph+vectors (MiB)", "peak build RSS (MiB)", "max deg ≤ R"]))
    if os.path.exists(os.path.join(RESULTS, "part2_sanity_sift100k.json")):
        s = load_json("part2_sanity_sift100k.json")
        md.append("\n\n## Part 2 — SIFT100K single-query sanity check\n\n" + "\n".join(
            f"- **{k}**: {v}" for k, v in s.items()))

    if os.path.exists(os.path.join(RESULTS, "part3a_sift100k.json")):
        rows = load_json("part3a_sift100k.json")["rows"]
        md.append("\n\n## Part 3a — SIFT100K operating points\n\n" + table(
            rows, ["R", "alpha", "L_build", "L_search", "1-recall@1", "10-recall@10", "mean_latency_ms", "qps",
                   "dist_evals_per_query", "vertices_expanded_per_query", "avg_out_degree", "build_time_s"],
            ["R", "α", "L_build", "L_search", "1-R@1", "10-R@10", "mean lat (ms)", "QPS", "dist evals/q",
             "expanded/q", "avg deg", "build (s)"]))

    for name in ("sift1m", "quora", "nq"):
        p = os.path.join(RESULTS, f"part3b_{name}.json")
        if not os.path.exists(p):
            continue
        d = load_json(f"part3b_{name}.json")
        rows = [{"method": f"Vamana L_search={v['L_search']}", **v} for v in d["vamana"]]
        md.append(f"\n\n## Part 3b — {name} Vamana sweep (R=64, L_build=100, α=1.2)\n\n" + table(
            rows, ["method", "1-recall@1", "10-recall@10", "mean_latency_ms", "p95_latency_ms", "qps",
                   "dist_evals_per_query", "vertices_expanded_per_query", "build_time_s", "index_size_MiB"],
            ["method", "1-R@1", "10-R@10", "mean lat (ms)", "p95 lat (ms)", "QPS", "dist evals/q", "expanded/q",
             "build (s)", "graph+vectors (MiB)"]))
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
