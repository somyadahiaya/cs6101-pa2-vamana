"""Assemble REPORT.md from report_src.md + results/tables.md + the compact Part 3b comparison."""
import os
import re

from common import RESULTS, ROOT, load_json
from make_tables import fmt, table

TARGETS = [0.95, 0.99, 0.995]


def compact():
    out = []
    cols = ["dataset", "target", "method", "1-recall@1", "10-recall@10", "mean_latency_ms", "qps", "build_time_s",
            "index_size_MiB", "dist_evals_per_query", "vertices_expanded_per_query", "speedup_vs_exact"]
    hdr = ["dataset", "target 10-R@10", "method", "1-R@1", "10-R@10", "mean lat (ms)", "QPS", "build (s)",
           "index (MiB)", "dist evals/q", "expanded/q", "speed-up vs exact"]
    for name in ("sift1m", "quora", "nq"):
        d = load_json(f"part3b_{name}.json")
        ex = d["exact"]
        out.append({"dataset": name, "target": "—", "method": "FAISS exact", **ex, "speedup_vs_exact": 1.0})
        for t in TARGETS:
            for meth, rows, lab in (("HNSW", d["hnsw"], lambda r: f"HNSW ef={r['efSearch']}"),
                                    ("Vamana", d["vamana"], lambda r: f"Vamana L={r['L_search']}")):
                ok = [r for r in rows if r["10-recall@10"] >= t]
                if not ok:
                    out.append({"dataset": name, "target": t, "method": f"{meth}: not reached"})
                    continue
                r = min(ok, key=lambda r: r["mean_latency_ms"])
                out.append({"dataset": name, "target": t, "method": lab(r), **r,
                            "speedup_vs_exact": ex["mean_latency_ms"] / r["mean_latency_ms"]})
    note = ("For each target, the row shows the **cheapest** (lowest mean latency) operating point of each method "
            "whose 10-recall@10 is ≥ the target. HNSW index size = FAISS file (graph + vectors); Vamana index "
            "size = graph file + float32 vectors.\n\n")
    return note + table(out, cols, hdr)


def main():
    src = open(os.path.join(ROOT, "report_src.md")).read()
    tables = open(os.path.join(RESULTS, "tables.md")).read()
    sections = {}
    for chunk in re.split(r"^## ", tables, flags=re.M):
        if chunk.strip():
            head, _, body = chunk.partition("\n")
            sections[head.strip()] = body.strip()
    sections["COMPACT"] = compact()

    def sub(m):
        key = m.group(1)
        if key not in sections:
            raise KeyError(key)
        return sections[key]
    rep = re.sub(r"\{\{(.+?)\}\}", sub, src)
    with open(os.path.join(ROOT, "REPORT.md"), "w") as f:
        f.write(rep)
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
