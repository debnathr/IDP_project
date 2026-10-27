# CQF joint routing and offset scheduling

Code of the Bachelor's thesis *GAT-based deep reinforcement learning for joint optimization of routing and scheduling in Time-Sensitive Networking*.
This stage contains the scheduling kernel and baseline algorithms for Cyclic Queuing and Forwarding (CQF); the learned policy is added in later stages.
Given a topology, a set of periodic flows and a cycle length T, the scheduler decides for every flow a **route**
(index into a set of candidate shortest paths) and an **offset** (injection delay in cycles), with the goal of
maximizing the number of admitted flows. Random and greedy baselines are implemented; the algorithm interface is
designed so that a learned policy can be plugged in later.

The system model (input format, candidate paths, per-hop timing, delay and bandwidth constraints, output format)
follows the GASA Multi-CQF implementation, reduced here to plain CQF with two alternating queues:

* Code: [tum-esi/gasa-mcqf-ti](https://github.com/tum-esi/gasa-mcqf-ti), `main` branch at commit
  `295178b` (2025-08-12); the files under `MCQF/gasa/` were used as reference. That code is released under the
  MIT license, Copyright (c) 2025 Associate Professorship of Embedded Systems and Internet of Things (TUM).
* Paper: R. Debnath, M. Barzegaran, S. Steinhorst, "Towards an Optimized Multi-Cyclic Queuing and Forwarding in
  Time Sensitive Networking With Time Injection," *IEEE Internet of Things Journal*, vol. 12, no. 20,
  pp. 43034–43051, 2025, doi: [10.1109/JIOT.2025.3597560](https://doi.org/10.1109/JIOT.2025.3597560)
  (preprint: [arXiv:2506.22671](https://arxiv.org/abs/2506.22671)).

## Layout

```
.
├── run_cqf.py                  entry point: load test case -> schedule -> validate -> export
├── cqf/
│   ├── data_io.py              parse topology / flow files (GASA format) into Stream objects
│   ├── network.py              Vertex / Link / Graph, k shortest candidate paths
│   ├── model.py                CQF kernel: check / commit / release / feasible_actions / verify
│   ├── validate.py             independent cycle-by-cycle simulation used to validate every schedule
│   ├── export.py               result files in GASA format plus a per-instance queue file and a summary
│   └── algorithms/
│       ├── base.py             SchedulerBase (admit flows one by one), flow ordering
│       ├── random_scheduler.py
│       └── greedy_scheduler.py
└── tests/                      unit tests with hand-checkable toy cases
```

Comments in the source are written in Chinese.

## Requirements

Python 3.10 or newer. Only the standard library is used.

## Input data

The test cases (`TC<i>_topo.txt`, `TC<i>_flows.txt`) are not part of this repository. By default they are read
from `../TCs_for_TSN/SA_TCs` relative to this directory; use `--data-dir` to point elsewhere.

## Usage

```bash
python run_cqf.py --tc 1 2 3 --cycle 50 --algo greedy
python run_cqf.py --tc 2 --cycle 50 --algo random --seed 0 --order random
python -m unittest tests.test_model tests.test_run_cqf -v
```

| Option | Meaning | Default |
|---|---|---|
| `--tc` | test case numbers (one or more) | required |
| `--cycle` | cycle length T in microseconds; must divide every flow period | required |
| `--algo` | `random` or `greedy` | `greedy` |
| `--order` | flow processing order: `input`, `deadline`, `random` | `input` |
| `--seed` | seed for the random scheduler and random ordering | `0` |
| `--k-paths` | number of candidate shortest paths per flow | `5` |
| `--max-offset` | upper bound on the offset in cycles | period − 1 |
| `--bw-fraction` | fraction of the 1 Gbps link usable by CQF traffic | `0.9` |
| `--count-source-link` | also account bandwidth on the talker → first switch link | off |
| `--data-dir`, `--results-dir`, `--tag` | paths and a free-text tag for result file names | |

## Model in brief

* Time is divided into cycles of length T; all periods must be integer multiples of T. The hyperperiod is the
  least common multiple of the periods in cycles.
* Each flow is routed over one of its k shortest candidate paths and injected `offset` cycles after the start of its
  period. Every switch holds a frame for exactly one cycle; a frame received in cycle c leaves in cycle c+1 and uses
  queue `c mod 2` (even/odd queue). Queue assignment is therefore derived, not decided.
* End-to-end delay in cycles = 1 + number of switches + offset, and must not exceed floor(deadline / T).
* Capacity: for every (cycle, directed link) slot the bytes of all frames sent in that slot must not exceed
  `bw_fraction × 1e9 bit/s / 8 × T / 1e6` bytes (T in microseconds; 5625 bytes for T = 50 µs at 0.9).
  All frame instances in the hyperperiod are checked; slots wrap modulo the
  hyperperiod. The talker → first-switch link is not counted unless `--count-source-link` is given.
* Flows are admitted one at a time: all (route, offset) pairs are checked against the current resource state and the
  algorithm picks one feasible pair, or the flow is rejected. Rejection reasons: `deadline`, `capacity`, `no_path`.

## Validation and outputs

Every run is checked twice: a ledger re-computation (`model.verify`) and an independent cycle-by-cycle simulation
(`validate.simulate_and_validate`) that does not reuse the kernel formulas. If either check fails, no result files
are written except `FAILED_<run>.txt`, and the process exits with status 1.

On success the following files are written to `results/` (file names contain test case, time stamp and the main
parameters; the complete parameter set is listed in the summary file):

| File | Content |
|---|---|
| `export_log_<run>.txt` | GASA-style log: admission ratio, then one line per flow |
| `res_<run>_sorted.csv` | GASA-style schedule (admitted flows only); queue numbers refer to the first frame instance |
| `queues_<run>.csv` | queue sequence of every frame instance of every admitted flow |
| `summary_<run>.txt` | test case, flow counts, admission ratio, run time, parameters, validation status |
| `time<i>_time_<run>.txt` | start / stop / total run time |
