"""Part 2: build Vamana graphs, scaling study, full-dataset build statistics and SIFT100K sanity check.

  python part2_build.py build <dataset> [--n N] [--R 64] [--L 100] [--alpha 1.2]   # one build (own process)
  python part2_build.py scaling                                                    # 100K/250K/500K subsets
  python part2_build.py full [dataset ...]                                         # main full builds
  python part2_build.py sanity                                                     # SIFT100K single-query check
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "48")

import numpy as np

from common import (BUILD_THREADS, INDEX, RESULTS, ROOT, Timer, cpu_info, file_size_mb, load_dataset, load_json,
                    peak_rss_mb, recall_at, save_json)

SEED = 0


def index_prefix(name, R, L, alpha, n=None):
    tag = f"_n{n}" if n else ""
    return os.path.join(INDEX, f"vamana_{name}{tag}_R{R}_L{L}_a{alpha:g}")


def cmd_build(a):
    import numba
    import vamana as V
    xb, _, _ = load_dataset(a.dataset, mmap=True)
    xb = np.ascontiguousarray(xb[:a.n] if a.n else xb, dtype=np.float32)
    rss_data = peak_rss_mb()
    V.build_vamana(xb[:3000].copy(), a.R, a.L, a.alpha, seed=SEED, verbose=False)   # JIT warm-up, not timed
    nbrs, deg, start, st = V.build_vamana(xb, a.R, a.L, a.alpha, seed=SEED)
    assert deg.max() <= a.R, "out-degree exceeds R"
    prefix = index_prefix(a.dataset, a.R, a.L, a.alpha, a.n)
    meta = {"dataset": a.dataset, "N": int(xb.shape[0]), "R": a.R, "L_build": a.L, "alpha": a.alpha, **st,
            "numba_threads": numba.get_num_threads(), "peak_rss_MiB": peak_rss_mb(),
            "rss_after_loading_vectors_MiB": rss_data,
            "vectors_float32_MiB": xb.nbytes / 2**20,
            "verified_max_out_degree_le_R": bool(deg.max() <= a.R),
            "num_vertices_degree_lt_R": int((deg < a.R).sum()), **cpu_info()}
    path = V.save_index(prefix, nbrs, deg, start, meta)
    meta["graph_file_MiB"] = file_size_mb(path)
    meta["graph_plus_vectors_MiB"] = meta["graph_file_MiB"] + meta["vectors_float32_MiB"]
    V.save_index(prefix, nbrs, deg, start, meta)       # rewrite json with the size included
    print(meta, flush=True)
    return meta


def spawn_build(dataset, n=None, R=64, L=100, alpha=1.2):
    args = [sys.executable, os.path.join(ROOT, "part2_build.py"), "build", dataset,
            "--R", str(R), "--L", str(L), "--alpha", str(alpha)]
    if n:
        args += ["--n", str(n)]
    subprocess.run(args, check=True)
    import json
    with open(index_prefix(dataset, R, L, alpha, n) + ".json") as f:
        return json.load(f)


def cmd_scaling(a):
    out = []
    for ds in a.datasets or ["sift1m", "nq"]:
        for n in (100_000, 250_000, 500_000):
            m = spawn_build(ds, n=n)
            out.append({k: m[k] for k in ("dataset", "N", "build_time_s", "pass_times_s", "avg_out_degree",
                                          "max_out_degree", "peak_rss_MiB")})
            print(out[-1], flush=True)
    save_json(out, "part2_scaling.json")


def cmd_full(a):
    res = {}
    for ds in a.datasets or ["sift1m", "quora", "nq"]:
        res[ds] = spawn_build(ds)
    try:
        old = load_json("part2_full_builds.json")
        old.update(res)
        res = old
    except FileNotFoundError:
        pass
    save_json(res, "part2_full_builds.json")


def cmd_sanity(a):
    import vamana as V
    xb, xq, gt = load_dataset("sift100k")
    prefix = index_prefix("sift100k", 64, 100, 1.2)
    if not os.path.exists(prefix + ".graph"):
        spawn_build("sift100k")
    nbrs, deg, start, meta = V.load_index(prefix)
    s = V.Searcher(xb, nbrs, deg, start)
    qid, L = 0, 50
    ids, nexp, ndist = s.search(xq[qid], 10, L)
    ids = ids.astype(np.int64)
    r = {"query_id": qid, "L_search": L, "exact_top10": gt[qid, :10].tolist(), "vamana_top10": ids.tolist(),
         "10-recall@10": recall_at(ids[None], gt[qid:qid + 1], 10), "vertices_expanded": int(nexp),
         "distance_evaluations": int(ndist), "entry_point": start}
    print(r)
    save_json(r, "part2_sanity_sift100k.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("dataset")
    b.add_argument("--n", type=int, default=None)
    b.add_argument("--R", type=int, default=64)
    b.add_argument("--L", type=int, default=100)
    b.add_argument("--alpha", type=float, default=1.2)
    for c in ("scaling", "full"):
        x = sp.add_parser(c)
        x.add_argument("datasets", nargs="*")
    sp.add_parser("sanity")
    a = p.parse_args()
    {"build": cmd_build, "scaling": cmd_scaling, "full": cmd_full, "sanity": cmd_sanity}[a.cmd](a)
