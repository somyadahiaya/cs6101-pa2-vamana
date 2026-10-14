"""Dataset preparation: SIFT1M / SIFT100K / Quora / NQ -> 128-d float32 + exact top-100 ground truth.

Usage: python prepare_data.py [sift|quora|nq|all]
"""
import glob
import os
import sys
import time

import faiss
import h5py
import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.decomposition import PCA

from common import GT_K, PREP, RAW, Timer, save_json, load_json, RESULTS

PCA_SAMPLE = 500_000
PCA_SEED = 42
DIM = 128


def exact_gt(xb, xq, k=GT_K):
    faiss.omp_set_num_threads(os.cpu_count())
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    D, I = index.search(xq, k)
    return D, I.astype(np.int64)


def save(name, xb, xq, D, I):
    np.save(os.path.join(PREP, f"{name}_base.npy"), np.ascontiguousarray(xb, dtype=np.float32))
    np.save(os.path.join(PREP, f"{name}_query.npy"), np.ascontiguousarray(xq, dtype=np.float32))
    np.save(os.path.join(PREP, f"{name}_gt.npy"), I)
    np.save(os.path.join(PREP, f"{name}_gt_dist.npy"), D)


def summary_entry(xb, xq, orig_dim, **extra):
    e = {"corpus_size": int(xb.shape[0]), "num_queries": int(xq.shape[0]),
         "original_dim": orig_dim, "search_dim": int(xb.shape[1]),
         "raw_float32_MiB": xb.shape[0] * xb.shape[1] * 4 / 2**20}
    e.update(extra)
    return e


def prep_sift(summary):
    with h5py.File(os.path.join(RAW, "sift", "sift-128-euclidean.hdf5"), "r") as f:
        xb = f["train"][:].astype(np.float32)
        xq = f["test"][:].astype(np.float32)
        gt_supplied = f["neighbors"][:].astype(np.int64)
    print("SIFT1M", xb.shape, xq.shape)

    # 1a: exact FAISS search and verification against the supplied ground truth
    with Timer() as t:
        D, I = exact_gt(xb, xq)
    top1 = float(np.mean(I[:, 0] == gt_supplied[:, 0]))
    top10 = float(np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(I[:, :10], gt_supplied[:, :10])]))
    top100 = float(np.mean([len(np.intersect1d(a, b)) / 100 for a, b in zip(I, gt_supplied)]))
    # Disagreements can only come from exact distance ties; check that.
    mism = np.where(I[:, 0] != gt_supplied[:, 0])[0]
    tie_ok = all(np.isclose(((xb[I[q, 0]] - xq[q]) ** 2).sum(), ((xb[gt_supplied[q, 0]] - xq[q]) ** 2).sum())
                 for q in mism)
    verify = {"agree_top1": top1, "agree_top10_overlap": top10, "agree_top100_overlap": top100,
              "n_top1_mismatch": int(len(mism)), "all_top1_mismatches_are_distance_ties": bool(tie_ok),
              "faiss_flat_gt_time_s": t.dt}
    print("SIFT1M verification:", verify)
    save("sift1m", xb, xq, D, gt_supplied)   # use the supplied GT for SIFT1M as required
    summary["sift1m"] = summary_entry(xb, xq, 128, source="ann-benchmarks sift-128-euclidean.hdf5 (= texmex SIFT1M)",
                                      gt_verification=verify)

    # SIFT100K development set: first 100K base vectors, all 10K queries, FAISS exact GT.
    xb100 = xb[:100_000].copy()
    D, I = exact_gt(xb100, xq)
    save("sift100k", xb100, xq, D, I)
    summary["sift100k"] = summary_entry(xb100, xq, 128, source="first 100,000 SIFT1M base vectors; GT by FAISS IndexFlatL2")


def read_emb(files):
    chunks = []
    for fn in files:
        pf = pq.ParquetFile(fn)
        for batch in pf.iter_batches(batch_size=65536, columns=["emb"]):
            col = batch.column(0)
            vals = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            chunks.append(vals.reshape(len(col), -1))
    return np.concatenate(chunks)


def prep_ir(name, summary):
    d = os.path.join(RAW, "beir", name)
    files = sorted(glob.glob(os.path.join(d, "corpus", "*.parquet")))
    with Timer() as t:
        xb_full = read_emb(files)
        xq_full = read_emb([os.path.join(d, "queries", "test.parquet")])
    print(name, "raw", xb_full.shape, xq_full.shape, f"{t.dt:.1f}s")
    orig_dim = xb_full.shape[1]

    # Fit PCA on a uniform random sample of the corpus only.
    rng = np.random.default_rng(PCA_SEED)
    n_fit = min(PCA_SAMPLE, xb_full.shape[0])
    fit_idx = np.sort(rng.choice(xb_full.shape[0], n_fit, replace=False))
    solver = "covariance_eigh" if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 5) else "full"
    pca = PCA(n_components=DIM, svd_solver=solver, random_state=PCA_SEED)
    with Timer() as t:
        pca.fit(xb_full[fit_idx])
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"{name} PCA fit {t.dt:.1f}s, explained variance {evr:.4f}")

    def project(x):
        out = np.empty((x.shape[0], DIM), dtype=np.float32)
        for s in range(0, x.shape[0], 200_000):
            out[s:s + 200_000] = pca.transform(x[s:s + 200_000]).astype(np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    xb = project(xb_full)
    xq = project(xq_full)
    del xb_full, xq_full
    with Timer() as tg:
        D, I = exact_gt(xb, xq)
    save(name, xb, xq, D, I)
    summary[name] = summary_entry(
        xb, xq, orig_dim,
        source="CohereLabs/beir-embed-english-v3 (corpus + test queries)",
        pca_implementation=f"sklearn {sklearn.__version__} PCA(n_components=128, svd_solver='{solver}'), mean-centred",
        pca_fit_sample_size=n_fit, pca_random_seed=PCA_SEED,
        pca_explained_variance_128=evr, post_processing="float32 cast, then L2-normalised",
        gt="FAISS IndexFlatL2 exact top-100 on 128-d vectors", gt_time_s=tg.dt)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    try:
        summary = load_json("dataset_summary.json")
    except FileNotFoundError:
        summary = {}
    if which in ("sift", "all"):
        prep_sift(summary)
    for n in ("quora", "nq"):
        if which in (n, "all"):
            prep_ir(n, summary)
    save_json(summary, "dataset_summary.json")
    print("done")
