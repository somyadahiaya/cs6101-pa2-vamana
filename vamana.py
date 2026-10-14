"""Own implementation of Vamana (DiskANN, Subramanya et al., NeurIPS 2019): GreedySearch, RobustPrune, build.

Performance-critical code is compiled with Numba (nopython, fastmath, parallel prange).

Graph representation: fixed-capacity adjacency matrix `nbrs` (int32, [N, R]) + out-degree vector
`deg` (int32, [N]).  Only the first deg[i] entries of row i are valid.

Distances: squared L2 everywhere.  RobustPrune's condition alpha * d(p*, p') <= d(p, p') on true
Euclidean distances is evaluated as alpha^2 * d2(p*, p') <= d2(p, p'), which is equivalent.

Parallel construction: the permutation is processed in batches (sizes doubling from 1 up to
`max_batch` in the first pass, as in ParlayANN's prefix-doubling).  Within a batch:
  A. (parallel) every point p runs GreedySearch on the current graph and RobustPrune on
     visited ∪ N_out(p), writing its new out-list into a private buffer;
  B. (parallel) the new out-lists are written into the graph;
  C. reverse edges (j <- p) are grouped by j (stable sort);
  D. (parallel over distinct j) N_out(j) ∪ {p's}: if > R, RobustPrune(j, ...), else assign.
Every row is written by exactly one thread per phase, so no locks are needed and the build is
deterministic for a fixed seed and batch schedule.  With batch size 1 this is exactly the
sequential BuildVamana pseudo-code.
"""
import json
import os
import time

import numba
import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------- distances
@njit(inline="always", fastmath=True, nogil=True, cache=True)
def _d2_vec(X, i, q):
    s = np.float32(0.0)
    for t in range(X.shape[1]):
        v = X[i, t] - q[t]
        s += v * v
    return s


@njit(inline="always", fastmath=True, nogil=True, cache=True)
def _d2_pair(X, i, j):
    s = np.float32(0.0)
    for t in range(X.shape[1]):
        v = X[i, t] - X[j, t]
        s += v * v
    return s


# --------------------------------------------------------------------------- GreedySearch
@njit(fastmath=True, nogil=True, cache=True)
def greedy_search(X, nbrs, deg, start, q, L, seen, stamp, cand_id, cand_d, cand_exp, vis_id, vis_d):
    """GreedySearch(G, s, q, L).

    Candidate set C is kept as a sorted array of at most L (dist, id, expanded) entries; `cur` is
    the position of the closest unexpanded candidate.  `seen[u] == stamp` marks vertices whose
    distance to q has already been evaluated (reusable visitation buffer; bump `stamp` per query).
    Expanded vertices (the visited set V) are appended to vis_id / vis_d.

    Returns (|C|, |V| recorded, #vertices expanded, #distance evaluations).
    """
    seen[start] = stamp
    cand_id[0] = start
    cand_d[0] = _d2_vec(X, start, q)
    cand_exp[0] = False
    n = 1
    ndist = 1
    nexp = 0
    nvis = 0
    vcap = vis_id.shape[0]
    cur = 0
    while cur < n:
        p = cand_id[cur]
        cand_exp[cur] = True
        nexp += 1
        if nvis < vcap:
            vis_id[nvis] = p
            vis_d[nvis] = cand_d[cur]
            nvis += 1
        best = n
        for e in range(deg[p]):
            u = nbrs[p, e]
            if seen[u] == stamp:
                continue
            seen[u] = stamp
            du = _d2_vec(X, u, q)
            ndist += 1
            if n == L and du >= cand_d[n - 1]:
                continue
            lo = 0
            hi = n
            while lo < hi:                      # first position with cand_d >= du
                mid = (lo + hi) >> 1
                if cand_d[mid] < du:
                    lo = mid + 1
                else:
                    hi = mid
            last = n if n < L else L - 1        # drop the worst entry if C is full
            for t in range(last, lo, -1):
                cand_id[t] = cand_id[t - 1]
                cand_d[t] = cand_d[t - 1]
                cand_exp[t] = cand_exp[t - 1]
            cand_id[lo] = u
            cand_d[lo] = du
            cand_exp[lo] = False
            if n < L:
                n += 1
            if lo < best:
                best = lo
        # all positions < cur were expanded; a new insert at position <= cur is the next one.
        cur = best if best <= cur else cur + 1
        while cur < n and cand_exp[cur]:
            cur += 1
    return n, nvis, nexp, ndist


# --------------------------------------------------------------------------- RobustPrune
@njit(fastmath=True, nogil=True, cache=True)
def robust_prune(X, p, pool_id, pool_d, m, alpha2, R, out, seen, stamp):
    """RobustPrune(G, p, V, alpha, R) on candidate pool (pool_id[:m], pool_d[:m] = d2(p, .)).

    The caller passes V ∪ N_out(p).  Duplicates and p itself are removed here.  Writes the new
    out-neighbourhood into out[:cnt] and returns cnt (<= R).
    """
    k = 0
    for i in range(m):                       # V <- (V ∪ N_out(p)) \ {p}, deduplicated
        u = pool_id[i]
        if u == p or seen[u] == stamp:
            continue
        seen[u] = stamp
        pool_id[k] = u
        pool_d[k] = pool_d[i]
        k += 1
    order = np.argsort(pool_d[:k])
    sid = pool_id[:k][order]
    sd = pool_d[:k][order]
    alive = np.ones(k, dtype=np.bool_)
    cnt = 0
    for i in range(k):
        if not alive[i]:
            continue
        ps = sid[i]                          # p* = closest remaining candidate
        out[cnt] = ps
        cnt += 1
        if cnt == R:
            break
        for j in range(i + 1, k):            # remove p' with alpha*d(p*,p') <= d(p,p')
            if alive[j] and alpha2 * _d2_pair(X, ps, sid[j]) <= sd[j]:
                alive[j] = False
    return cnt


# --------------------------------------------------------------------------- construction
@njit(nogil=True, cache=True)
def random_regular_graph(N, R, seed):
    """Random directed graph where every vertex has exactly min(R, N-1) distinct out-neighbours."""
    np.random.seed(seed)
    r = min(R, N - 1)
    nbrs = np.empty((N, R), dtype=np.int32)
    deg = np.full(N, r, dtype=np.int32)
    for i in range(N):
        c = 0
        while c < r:
            u = np.random.randint(0, N)
            if u == i:
                continue
            dup = False
            for t in range(c):
                if nbrs[i, t] == u:
                    dup = True
                    break
            if not dup:
                nbrs[i, c] = u
                c += 1
    return nbrs, deg


@njit(parallel=True, fastmath=True, cache=True)
def _phase_search_prune(X, nbrs, deg, start, batch, L, alpha2, R, seen, stamps,
                        cand_id, cand_d, cand_exp, vis_id, vis_d, pool_id, pool_d, new_nbrs, new_deg):
    for b in prange(batch.shape[0]):
        tid = numba.get_thread_id()
        p = batch[b]
        stamps[tid] += 1
        st = stamps[tid]
        _, nvis, _, _ = greedy_search(X, nbrs, deg, start, X[p], L, seen[tid], st,
                                      cand_id[tid], cand_d[tid], cand_exp[tid], vis_id[tid], vis_d[tid])
        m = 0
        for i in range(nvis):
            pool_id[tid, m] = vis_id[tid, i]
            pool_d[tid, m] = vis_d[tid, i]
            m += 1
        for e in range(deg[p]):
            u = nbrs[p, e]
            pool_id[tid, m] = u
            pool_d[tid, m] = _d2_pair(X, p, u)
            m += 1
        stamps[tid] += 1
        new_deg[b] = robust_prune(X, p, pool_id[tid], pool_d[tid], m, alpha2, R, new_nbrs[b],
                                  seen[tid], stamps[tid])


@njit(parallel=True, cache=True)
def _phase_write(nbrs, deg, batch, new_nbrs, new_deg):
    for b in prange(batch.shape[0]):
        p = batch[b]
        deg[p] = new_deg[b]
        for e in range(new_deg[b]):
            nbrs[p, e] = new_nbrs[b, e]


@njit(parallel=True, fastmath=True)
def _phase_reverse(X, nbrs, deg, batch, new_nbrs, new_deg, alpha2, R, seen, stamps):
    # C: collect reverse edges (j <- p) and group them by j
    tot = 0
    for b in range(batch.shape[0]):
        tot += new_deg[b]
    tgt = np.empty(tot, dtype=np.int32)
    src = np.empty(tot, dtype=np.int32)
    c = 0
    for b in range(batch.shape[0]):
        for e in range(new_deg[b]):
            tgt[c] = new_nbrs[b, e]
            src[c] = batch[b]
            c += 1
    order = np.argsort(tgt, kind="mergesort")
    tgt = tgt[order]
    src = src[order]
    starts = [0]
    for i in range(1, tot):
        if tgt[i] != tgt[i - 1]:
            starts.append(i)
    starts.append(tot)
    gs = np.array(starts, dtype=np.int64)
    # D: for each distinct j, N_out(j) ∪ {p...}; prune if it exceeds R
    out_buf = np.empty((numba.get_num_threads(), R), dtype=np.int32)
    for g in prange(gs.shape[0] - 1):
        tid = numba.get_thread_id()
        a = gs[g]
        z = gs[g + 1]
        j = tgt[a]
        st = stamps[tid] + 1                     # st: existing nbr, st+1: new src, st+2: appended
        stamps[tid] += 4                         # st+3 is used by robust_prune below
        sj = seen[tid]
        d = deg[j]
        for e in range(d):
            sj[nbrs[j, e]] = st
        extra = 0
        for i in range(a, z):
            u = src[i]
            if sj[u] != st and sj[u] != st + 1:
                sj[u] = st + 1
                extra += 1
        if d + extra <= R:                       # OutNeighbors(j) <- C
            for i in range(a, z):
                u = src[i]
                if sj[u] == st + 1:
                    sj[u] = st + 2               # consumed: duplicates in the group are skipped
                    nbrs[j, d] = u
                    d += 1
            deg[j] = d
        else:                                    # RobustPrune(G, j, C, alpha, R)
            m = d + (z - a)
            pid = np.empty(m, dtype=np.int32)
            pd = np.empty(m, dtype=np.float32)
            for e in range(d):
                pid[e] = nbrs[j, e]
                pd[e] = _d2_pair(X, j, pid[e])
            for i in range(a, z):
                pid[d + i - a] = src[i]
                pd[d + i - a] = _d2_pair(X, j, src[i])
            cnt = robust_prune(X, j, pid, pd, m, alpha2, R, out_buf[tid], sj, st + 3)
            for e in range(cnt):
                nbrs[j, e] = out_buf[tid, e]
            deg[j] = cnt


def medoid_approx(X):
    """Database point closest to the dataset mean (practical medoid approximation)."""
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    best, best_d = -1, np.inf
    for s in range(0, X.shape[0], 262144):
        d = ((X[s:s + 262144] - mu) ** 2).sum(1)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best, best_d = s + i, float(d[i])
    return best


def build_vamana(X, R=64, L=100, alpha=1.2, seed=0, max_batch_frac=0.02, max_batch_cap=50_000,
                 verbose=True):
    """BuildVamana(P, R, L_build, alpha) with two passes (alpha=1, then alpha).  Returns (nbrs, deg, start, stats)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    N = X.shape[0]
    T = numba.get_num_threads()
    t0 = time.perf_counter()
    nbrs, deg = random_regular_graph(N, R, seed)
    start = medoid_approx(X)
    vcap = 4 * L + 64
    seen = np.zeros((T, N), dtype=np.uint32)
    stamps = np.zeros(T, dtype=np.uint32)
    cand_id = np.empty((T, L), dtype=np.int32)
    cand_d = np.empty((T, L), dtype=np.float32)
    cand_exp = np.empty((T, L), dtype=np.bool_)
    vis_id = np.empty((T, vcap), dtype=np.int32)
    vis_d = np.empty((T, vcap), dtype=np.float32)
    pool_id = np.empty((T, vcap + R), dtype=np.int32)
    pool_d = np.empty((T, vcap + R), dtype=np.float32)
    max_batch = int(max(1, min(max_batch_cap, max_batch_frac * N)))
    new_nbrs = np.empty((max_batch, R), dtype=np.int32)
    new_deg = np.empty(max_batch, dtype=np.int32)
    rng = np.random.default_rng(seed)
    pass_times = []
    for pi, a in enumerate([1.0, alpha]):
        tp = time.perf_counter()
        order = rng.permutation(N).astype(np.int32)
        pos, bs = 0, (1 if pi == 0 else max_batch)
        while pos < N:
            batch = order[pos:pos + bs]
            nb = batch.shape[0]
            _phase_search_prune(X, nbrs, deg, start, batch, L, np.float32(a * a), R, seen, stamps,
                                cand_id, cand_d, cand_exp, vis_id, vis_d, pool_id, pool_d,
                                new_nbrs[:nb], new_deg[:nb])
            _phase_write(nbrs, deg, batch, new_nbrs[:nb], new_deg[:nb])
            _phase_reverse(X, nbrs, deg, batch, new_nbrs[:nb], new_deg[:nb], np.float32(a * a), R, seen, stamps)
            pos += nb
            bs = min(max_batch, bs * 2)
        pass_times.append(time.perf_counter() - tp)
        if verbose:
            print(f"  pass {pi + 1} (alpha={a}) {pass_times[-1]:.1f}s  avg deg {deg.mean():.2f}", flush=True)
    stats = {"build_time_s": time.perf_counter() - t0, "pass_times_s": pass_times, "threads": T,
             "max_batch": max_batch, "seed": seed, "entry_point": int(start),
             "avg_out_degree": float(deg.mean()), "max_out_degree": int(deg.max()),
             "min_out_degree": int(deg.min())}
    return nbrs, deg, start, stats


# --------------------------------------------------------------------------- query-time search
class Searcher:
    """Reusable single-thread search buffers (for per-query latency measurement)."""

    def __init__(self, X, nbrs, deg, start, Lmax=1024):
        self.X, self.nbrs, self.deg, self.start = X, nbrs, deg, start
        self.seen = np.zeros(X.shape[0], dtype=np.uint32)
        self.stamp = 0
        self.cand_id = np.empty(Lmax, dtype=np.int32)
        self.cand_d = np.empty(Lmax, dtype=np.float32)
        self.cand_exp = np.empty(Lmax, dtype=np.bool_)
        self.vis_id = np.empty(0, dtype=np.int32)    # visited list not needed at query time
        self.vis_d = np.empty(0, dtype=np.float32)

    def search(self, q, k, L):
        L = max(L, k)
        self.stamp += 1
        n, _, nexp, ndist = greedy_search(self.X, self.nbrs, self.deg, self.start, q, L, self.seen,
                                          np.uint32(self.stamp), self.cand_id, self.cand_d, self.cand_exp,
                                          self.vis_id, self.vis_d)
        return self.cand_id[:min(k, n)], nexp, ndist


@njit(parallel=True, fastmath=True)
def search_batch(X, nbrs, deg, start, Q, k, L):
    """Parallel (over queries) GreedySearch; used for recall over the full query set."""
    T = numba.get_num_threads()
    nq = Q.shape[0]
    out = np.full((nq, k), -1, dtype=np.int64)
    nexp = np.zeros(nq, dtype=np.int64)
    ndist = np.zeros(nq, dtype=np.int64)
    seen = np.zeros((T, X.shape[0]), dtype=np.uint32)
    stamps = np.zeros(T, dtype=np.uint32)
    cid = np.empty((T, L), dtype=np.int32)
    cd = np.empty((T, L), dtype=np.float32)
    ce = np.empty((T, L), dtype=np.bool_)
    vid = np.empty((T, 0), dtype=np.int32)
    vd = np.empty((T, 0), dtype=np.float32)
    for i in prange(nq):
        tid = numba.get_thread_id()
        stamps[tid] += 1
        n, _, e, d = greedy_search(X, nbrs, deg, start, Q[i], L, seen[tid], stamps[tid],
                                   cid[tid], cd[tid], ce[tid], vid[tid], vd[tid])
        for t in range(min(k, n)):
            out[i, t] = cid[tid, t]
        nexp[i] = e
        ndist[i] = d
    return out, nexp, ndist


# --------------------------------------------------------------------------- persistence
def save_index(prefix, nbrs, deg, start, meta):
    """Serialized graph: <prefix>.graph = [int32 N, int32 R, int32 start] + deg[N] + nbrs[N*R] (int32)."""
    N, R = nbrs.shape
    with open(prefix + ".graph", "wb") as f:
        np.array([N, R, start], dtype=np.int32).tofile(f)
        deg.astype(np.int32).tofile(f)
        nbrs.astype(np.int32).tofile(f)
    with open(prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return prefix + ".graph"


def load_index(prefix):
    with open(prefix + ".graph", "rb") as f:
        N, R, start = np.fromfile(f, dtype=np.int32, count=3)
        deg = np.fromfile(f, dtype=np.int32, count=N)
        nbrs = np.fromfile(f, dtype=np.int32, count=N * R).reshape(N, R)
    meta = {}
    if os.path.exists(prefix + ".json"):
        with open(prefix + ".json") as f:
            meta = json.load(f)
    return nbrs, deg, int(start), meta
