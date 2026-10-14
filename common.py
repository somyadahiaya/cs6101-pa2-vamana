"""Shared paths, I/O, recall and timing helpers for CS6101 PA2."""
import json
import os
import platform
import resource
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")          # symlink to the NVMe data directory
RAW = os.path.join(DATA, "raw")
PREP = os.path.join(DATA, "prepared")      # 128-d corpus/query/GT arrays
INDEX = os.path.join(DATA, "indices")      # persisted HNSW / Vamana indices
RESULTS = os.path.join(ROOT, "results")    # json / csv results (small, go in submission)
PLOTS = os.path.join(ROOT, "plots")

for _d in (PREP, INDEX, RESULTS, PLOTS):
    os.makedirs(_d, exist_ok=True)

DATASETS = ["sift100k", "sift1m", "quora", "nq"]
TIMING_QUERIES = 1000          # fixed random subset used for latency measurement
TIMING_SEED = 1234
GT_K = 100


def load_dataset(name, mmap=False):
    """Return (corpus float32 [N,128], queries float32 [Q,128], gt int64 [Q,GT_K])."""
    mode = "r" if mmap else None
    xb = np.load(os.path.join(PREP, f"{name}_base.npy"), mmap_mode=mode)
    xq = np.load(os.path.join(PREP, f"{name}_query.npy"))
    gt = np.load(os.path.join(PREP, f"{name}_gt.npy"))
    return xb, xq, gt


def timing_subset(nq):
    """Fixed random subset of query ids used for timing (all queries if nq <= TIMING_QUERIES)."""
    if nq <= TIMING_QUERIES:
        return np.arange(nq)
    rng = np.random.default_rng(TIMING_SEED)
    return np.sort(rng.choice(nq, TIMING_QUERIES, replace=False))


def recall_at(I, gt, k):
    """k-recall@k = mean_q |A_k(q) ∩ G_k(q)| / k."""
    I = np.asarray(I)[:, :k]
    G = np.asarray(gt)[:, :k]
    hits = 0
    for a, g in zip(I, G):
        hits += len(np.intersect1d(a, g, assume_unique=False))
    return hits / (k * len(G))


def cpu_info():
    model = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu_model": model, "logical_cpus": os.cpu_count()}


def peak_rss_mb():
    """Peak resident set size of this process (Linux ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def file_size_mb(*paths):
    return sum(os.path.getsize(p) for p in paths) / 2**20


def save_json(obj, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0


# --------------------------------------------------------------------------- query timing
# Policy used for every method: 1 CPU thread, queries issued individually (batch size 1) from Python,
# WARMUP untimed queries first, then REPEATS passes over the fixed timing subset; the reported mean
# latency is the median over passes of the per-pass mean.
WARMUP = 100
REPEATS = 3
BUILD_THREADS = 48


def time_queries(search_one, xq_sub):
    """search_one(q[1,d]) -> anything.  Returns dict(mean_ms, p50_ms, p95_ms, qps)."""
    for i in range(min(WARMUP, len(xq_sub))):
        search_one(xq_sub[i:i + 1])
    means, all_lat = [], None
    for _ in range(REPEATS):
        lat = np.empty(len(xq_sub))
        for i in range(len(xq_sub)):
            t0 = time.perf_counter()
            search_one(xq_sub[i:i + 1])
            lat[i] = time.perf_counter() - t0
        means.append(lat.mean())
        if all_lat is None or lat.mean() <= min(means):
            all_lat = lat
    m = float(np.median(means))
    return {"mean_latency_ms": m * 1e3, "p50_latency_ms": float(np.percentile(all_lat, 50) * 1e3),
            "p95_latency_ms": float(np.percentile(all_lat, 95) * 1e3), "qps": 1.0 / m,
            "timing_queries": int(len(xq_sub)), "timing_repeats": REPEATS}
