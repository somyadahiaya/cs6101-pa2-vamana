# CS6101 PA2 — ANN search with Vamana (DiskANN) vs FAISS

The write-up with all results is in **[REPORT.md](REPORT.md)**. The assignment handout is `CS6101_2026_PA2.pdf`.

## Setup

Use Python 3.10. The `.venv` folder is not committed; recreate it from `requirements.txt`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # after this, `python` means the venv's Python 3.10
pip install -r requirements.txt
```

Datasets and indices are **not** in the repo because they take about 20 GB. They live in `data/`, which can be a real folder or a symlink to a large disk:

```bash
mkdir -p /path/with/space/pa2_data && ln -s /path/with/space/pa2_data data
```

## Running

```bash
python prepare_data.py all    # downloads are expected under data/raw (see below); writes data/prepared
./run_all.sh                  # Part 1 -> Part 2 -> Part 3 (about 1.5 h on 48 threads)
python retime.py              # re-measure all latencies in one pinned-core session
python make_tables.py && python build_report.py   # regenerate results/tables.md and REPORT.md
```

Raw data expected by `prepare_data.py`:

- `data/raw/sift/sift-128-euclidean.hdf5` from http://ann-benchmarks.com/sift-128-euclidean.hdf5
- `data/raw/beir/{quora,nq}/corpus/*.parquet` and `.../queries/test.parquet` from the HF dataset `CohereLabs/beir-embed-english-v3`. Download them with `huggingface_hub.snapshot_download(..., allow_patterns=[...], local_dir="data/raw/beir")`.

## What is where

| path | contents |
|---|---|
| `REPORT.md` | final report: dataset summary, Parts 1, 2, 3a and 3b, discussion, LLM-usage note |
| `report_src.md` | hand-written report text with `{{placeholders}}`; `build_report.py` fills them to produce `REPORT.md` |
| `common.py` | paths, k-recall@k, timing protocol (1 thread, single queries, warm-up, repeats), helpers |
| `prepare_data.py` | SIFT1M and SIFT100K prep, PCA 1024→128 for Quora and NQ, exact FAISS ground truth |
| `part1_faiss.py` | Part 1: FAISS `IndexFlatL2` exact search plus `IndexHNSWFlat` efSearch sweep, with plots |
| `vamana.py` | **our own Vamana**: GreedySearch, RobustPrune, parallel two-pass build, search, save and load (Numba) |
| `part2_build.py` | Part 2: 100K/250K/500K scaling builds, full builds with stats, SIFT100K sanity check |
| `part3.py` | Part 3a: R/α/L_search study on SIFT100K. Part 3b: Vamana vs HNSW vs exact on the full datasets |
| `retime.py` | re-times every operating point in one session; the shared server made early timings noisy |
| `make_tables.py` | turns `results/*.json` into `results/tables.md` |
| `build_report.py` | puts together `REPORT.md` |
| `run_all.sh` | runs the whole pipeline |
| `results/` | raw JSON results for every experiment, plus `tables.md` |
| `plots/` | all figures used in the report |
| `logs/` | stdout logs of the runs |
| `requirements.txt` | exact package versions from the venv used for the results |
