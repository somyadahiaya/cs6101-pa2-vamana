#!/usr/bin/env bash
# Runs the whole compulsory pipeline after prepare_data.py.  Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export NUMBA_NUM_THREADS=48
$PY part1_faiss.py nq quora        > logs/part1_nq_quora.log 2>&1
$PY part2_build.py scaling         > logs/part2_scaling.log 2>&1
$PY part2_build.py full            > logs/part2_full.log 2>&1
$PY part2_build.py sanity          > logs/part2_sanity.log 2>&1
$PY part3.py 3a                    > logs/part3a.log 2>&1
$PY part3.py 3b                    > logs/part3b.log 2>&1
echo ALL_DONE
