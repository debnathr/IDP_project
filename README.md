# CQF joint routing and offset scheduling

Code of the Bachelor's thesis *GAT-based deep reinforcement learning for joint optimization of routing and scheduling in Time-Sensitive Networking*.
It contains the scheduling kernel, the rule-based baselines and the reinforcement-learning part (MaskablePPO with a flat
MLP or GAT policy, or with a per-action scoring policy that can use GAT embeddings) for Cyclic Queuing and Forwarding (CQF).
Given a topology, a set of periodic flows and a cycle length T, the scheduler decides for every flow a **route**
(index into a set of candidate shortest paths) and an **offset** (injection delay in cycles), with the goal of
maximizing the number of admitted flows. The random and greedy baselines and the trained policy all run through the same
algorithm interface and the same validation and export path.

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
├── run_baselines.py            runs the random / greedy baselines with several seeds and tabulates them
├── run_comparison.py           rule baselines and trained policies at one cycle time and flow order, one table
├── cqf/
│   ├── data_io.py              parse topology / flow files (GASA format) into Stream objects
│   ├── network.py              Vertex / Link / Graph, k shortest candidate paths
│   ├── model.py                CQF kernel: check / commit / release / feasible_actions / verify
│   ├── validate.py             independent cycle-by-cycle simulation used to validate every schedule
│   ├── export.py               result files in GASA format plus a per-instance queue file and a summary
│   ├── algorithms/
│   │   ├── base.py             SchedulerBase (admit flows one by one), flow ordering
│   │   ├── random_scheduler.py
│   │   ├── greedy_scheduler.py  greedy (smallest latency first), greedy_load (least loaded slot first),
│   │   │                        hops_load (fewest switches first, then least loaded slot)
│   │   └── ppo_scheduler.py    runs a trained MaskablePPO policy through the same interface
│   └── rl/
│       ├── features.py         vectorised action mask and observation (numpy), mirrors the kernel ledger
│       ├── env.py              Gymnasium environment: one episode = one test case, flows in a given order
│       ├── graph_obs.py        graph observation (nodes, directed links with utilisation profiles, candidate paths)
│       ├── gat.py              GAT encoder used as the policy's feature extractor (plain torch, no PyG)
│       ├── score.py            per-action scoring policy: one shared network scores every (route, offset) action
│       └── train.py            MaskablePPO training script (sb3-contrib), --policy mlp | gat | score | gat_score
└── tests/                      unit tests with hand-checkable toy cases
```

Comments in the source are written in Chinese.

## Requirements

Python 3.10 or newer. The kernel, the baselines and their tests use only the standard library.
The reinforcement-learning part (`cqf/rl/`, `--algo ppo`) additionally needs

```bash
pip install numpy gymnasium torch stable-baselines3 sb3-contrib
```

## Input data

The test cases (`TC<i>_topo.txt`, `TC<i>_flows.txt`) are not part of this repository. By default they are read
from `../TCs_for_TSN/SA_TCs` relative to this directory; use `--data-dir` to point elsewhere.

## Usage

```bash
python run_cqf.py --tc 1 2 3 --cycle 50 --algo greedy_load
python run_cqf.py --tc 2 --cycle 50 --algo random --seed 0 --order random
python run_baselines.py --tc 1 2 3 5 9 11 13 18 25 26 --cycle 50 --seeds 5   # baseline table, mean ± std over seeds
python run_comparison.py --cycle 50 --order input --mlp results/mlp_T50_input --gat results/gat_T50_input   # full comparison table
python run_comparison.py --cycle 50 --order input --policy "GAT-score=results/gs_s100/model.zip" --policy "GAT-score=results/gs_s101/model.zip"   # one column, mean ± std over the two training seeds
python -m unittest tests.test_model tests.test_run_cqf tests.test_rl_env -v
```

Baselines: `random` picks a feasible (route, offset) pair uniformly; `greedy` takes the pair with the smallest
end-to-end latency and breaks ties by the load of the busiest slot it would use; `greedy_load` reverses the two
criteria (least loaded slot first, then smallest latency), which spreads flows over the hyperperiod instead of
piling them into the earliest cycles; `hops_load` first restricts the choice to the candidate paths with the fewest
switches (a longer path occupies one more link slot per frame instance) and then takes the least loaded slot on
them, so the load is only used to choose the offset. Every summary file reports the mask density: the mean over
flows of feasible / candidate (route, offset) pairs at the moment the flow was processed, plus the number of flows
that had no feasible pair at all.

`run_comparison.py` runs the four baselines and any number of trained policies on the same test cases. `--mlp` and
`--gat` add one policy each; `--policy LABEL=PATH` adds a policy to the column `LABEL` and may be repeated with the
same label for several training seeds. A path is either a model `.zip` file or a directory, in which case
`best_model.zip` is used if it exists and `model.zip` otherwise.

Reinforcement learning (MaskablePPO):

```bash
python -m cqf.rl.train --cycle 50 --train-tc 2 --val-tc 2 --timesteps 2000 --out results/ppo_smoke   # smoke test, a few seconds
python -m cqf.rl.train --cycle 50 --order input --timesteps 1000000 --n-envs 16 --gamma 0.999 --batch-size 256 --net 256 256 --ent-coef 0.001 --lr-linear --eval-freq 40000 --seed 0 --out results/mlp_T50_input                                           # MLP policy, train on TC5/9/13, validate on TC11
python -m cqf.rl.train --cycle 50 --policy gat --order deadline --timesteps 1000000 --n-envs 16 --gamma 0.999 --batch-size 256 --net 256 256 --ent-coef 0.001 --lr-linear --eval-freq 40000 --seed 0 --out results/gat_T50_deadline                                        # GAT policy
python -m cqf.rl.train --cycle 50 --policy gat_score --order deadline --timesteps 1000000 --n-envs 16 --gamma 0.999 --batch-size 256 --net 256 256 --ent-coef 0.001 --lr-linear --eval-freq 40000 --seed 100 --out results/gat_score_T50_deadline_s100                       # per-action scoring with GAT embeddings
python -m cqf.rl.train --cycle 50 --policy gat_score --score-prior 20 --order deadline --timesteps 1000000 --n-envs 16 --gamma 0.999 --batch-size 256 --net 256 256 --ent-coef 0.001 --lr-linear --eval-freq 40000 --seed 100 --out results/gat_score_prior_T50_deadline_s100   # the same with the rule prior
python run_cqf.py --tc 18 25 26 --cycle 50 --order deadline --algo ppo --model results/gat_T50_deadline
python run_cqf.py --tc 18 25 26 --cycle 50 --order deadline --algo ppo --model results/gat_score_T50_deadline_s100/model.zip
```

The second to fifth commands are the settings of the comparison runs made so far in this project (1 M steps,
16 environments, gamma 0.999, batch 256, 256x256 head, entropy coefficient 0.001, linearly decaying learning rate,
validation every 40 000 steps); which experiments will go into the thesis has not been decided. The PPO defaults of
`train.py` are the stable-baselines3 defaults (gamma 0.99, batch 64, 64x64 head, no entropy bonus, constant learning
rate) with 200 000 steps, 8 environments and validation every 20 000 steps. A single-factor check on the validation
case showed no measurable difference between the two settings. Every run writes its actual values to `config.json`.

With `--policy gat` the observation is the whole topology: every node (switch / end station, degree, source or
destination of the current flow, number of candidate paths through it) and every directed link (binned utilisation
profile over the hyperperiod, mean / max / near-full share, number of candidate paths using it), padded to fixed
sizes (`--max-nodes`, `--max-links`, `--max-path-links`) so that one network can be trained on small topologies and
evaluated on larger ones. A graph attention encoder with edge features (`cqf/rl/gat.py`) produces one embedding per
candidate path and one for the whole graph; these are appended to the vector observation of the MLP policy and fed
to the same MLP head, so the two policies differ only in the encoder.

The head of `mlp` and `gat` is flat: one linear layer outputs the logits of all `k_paths × n_offsets` actions, with
a separate weight row per action. `--policy score` replaces it by a shared per-action scorer (`cqf/rl/score.py`).
The vector observation is rearranged into one row of eight features per action (load of the busiest slot after
admission, number of switches, offset / period, latency / deadline and the four features of the current flow), and
one small MLP (`--score-net`, default 32x32) maps every row to the logit of that action, so the number of policy
parameters does not depend on the number of actions. `--policy gat_score` additionally gives every action the GAT
embedding of its candidate path and of the whole graph; the value network reads the same features, but its gradient
is not propagated into the encoder. Masking, sampling and the PPO loss are those of MaskablePPO in all four cases.

`--score-prior beta` (for `score` and `gat_score`) adds `-beta × (switches + load)` to every logit and initialises
the last layer of the scorer to zero. Before training the greedy action is then the one chosen by the rule "fewest
switches, then least loaded slot" (`hops_load` up to tie-breaking), and training learns a residual on top of this
rule; such a policy is a heuristic prior plus a learned correction, not a policy learned from the reward alone.
`--gat-ablation no-edge` (link features zeroed), `--gat-ablation no-path` (path embeddings zeroed, only the graph
embedding is left) and `--gat-layers` are switches for ablation runs of the encoder.

Training runs `--n-envs` environments in parallel (default 8, one process each; `--vec dummy` keeps them in one
process) and uses the GPU when `torch.cuda.is_available()` (`--device auto`). The CPU-only torch wheel works but is
several times slower for the GAT policy; on an NVIDIA GPU install the matching CUDA build from
https://download.pytorch.org/whl/ instead.

The policy decides, for each flow in turn, one of `k_paths × n_offsets` actions (route index, offset); infeasible
actions are masked out using the same rules as the kernel. The reward of a decision is +1 minus the number of
flows that become unschedulable before the next decision (`--reject-penalty`, 0 gives plain +1 per admitted flow),
so an episode return equals admitted minus rejected flows (flows that are already infeasible at the start of the
episode are not counted). `train.py` writes
`model.zip`, `best_model.zip` (best validation return) and `config.json`; `--algo ppo` checks that the model was
trained with the same cycle length, candidate path count, capacity, source-link setting, offset range and
observation size as the current run.

| Option | Meaning | Default |
|---|---|---|
| `--tc` | test case numbers (one or more) | required |
| `--cycle` | cycle length T in microseconds; must divide every flow period | required |
| `--algo` | `random`, `greedy`, `greedy_load`, `hops_load` or `ppo` | `greedy` |
| `--model` | for `--algo ppo`: a model `.zip` file, or a directory (`best_model.zip` if present, else `model.zip`); `config.json` must be next to it | |
| `--order` | flow processing order: `input`, `deadline`, `random` | `input` |
| `--seed` | seed for the random scheduler | `0` |
| `--order-seed` | seed for `--order random`; defaults to `--seed + 10000` so that ordering and scheduler draw from different streams | |
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
