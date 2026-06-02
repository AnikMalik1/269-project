# Differentiable 2D Rope Control in JAX

This is a small research repo for comparing planners on a differentiable bead-spring rope simulator:

- random shooting
- cross-entropy method (CEM)
- gradient-based planning with `jax.grad`
- CEM followed by gradient refinement

The simulator is intentionally compact. It models a 2D rope as particles with adjacent springs, damping, bending regularization, semi-implicit Euler integration, optional circular obstacle repulsion, and a controlled gripper. The gripper has two modes:

- `attached`: the final rope particle is set directly to the gripper position.
- `soft_contact`: the gripper applies forces to nearby particles with
  `w = sigmoid((r_contact - distance) / tau)`.

## Setup

Use a virtual environment with JAX installed:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU or TPU JAX builds, install the platform-specific JAX wheel first using the official JAX instructions, then run `pip install -e .`.

## Quick Smoke Run

```bash
python scripts/run_experiments.py \
  --preset quick \
  --out outputs/quick \
  --make-plots \
  --make-gifs
```

This uses 2 seeds and smaller planner budgets.

## Full Experiments

```bash
python scripts/run_experiments.py \
  --preset full \
  --out outputs/full \
  --make-plots \
  --make-gifs
```

The full preset runs 20 seeds, all three tasks, all four planners, and the contact sweeps:

- `contact_tau = [0.2, 0.1, 0.05, 0.02]`
- `initial_gripper_distance = [0.1, 0.2, 0.4, 0.6]`

Default full planner budgets:

- random shooting: 512 sampled rollouts
- CEM: 8 iterations x 64 samples
- gradient descent: 128 gradient steps
- hybrid: 6 CEM iterations x 64 samples, then 64 gradient steps

Random shooting and CEM therefore use comparable sampled rollout budgets. Gradient methods report one differentiable simulator evaluation per gradient step plus final evaluation.

## Outputs

Each run directory contains:

- `results.csv`: one row per task, planner, seed, and sweep condition
- `summary.csv`: grouped mean/std final loss, success rate, wall time, and rollout count
- `summary.md`: Markdown version of the summary table
- `success_vs_contact_tau.png`
- `success_vs_initial_distance.png`
- `final_loss_by_planner.png`
- `gifs/*.gif`: one representative trajectory per method per task when `--make-gifs` is set

You can rebuild plots and summaries from any results CSV:

```bash
python scripts/make_plots.py outputs/full/results.csv --out outputs/full
```

## Tasks

`attached_line`

The endpoint is attached to the gripper. The target is a shifted straight rope, so the planner must pull the rope into a translated line.

`contact_discovery`

The gripper starts away from the rope and must discover contact through the sigmoid soft-contact model. This is the task used for the `contact_tau` and initial-distance sweeps.

`obstacle_arc`

The endpoint is attached to the gripper. A circular obstacle repels rope particles, and the target is an arc around the obstacle. The loss includes an obstacle collision penalty.

## Useful Commands

Run only one task:

```bash
python scripts/run_experiments.py --preset quick --tasks contact_discovery --out outputs/contact_quick
```

Run only CEM and hybrid:

```bash
python scripts/run_experiments.py --preset quick --planners cem hybrid --out outputs/cem_hybrid_quick
```

Use custom planner budgets:

```bash
python scripts/run_experiments.py \
  --preset full \
  --random-samples 1024 \
  --cem-iters 16 \
  --cem-samples 64 \
  --gd-iters 256 \
  --out outputs/bigger_budget
```

