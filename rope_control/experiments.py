from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
from tqdm import tqdm

from rope_control.planners import run_planner
from rope_control.tasks import make_task
from rope_control.viz import plot_results, save_rollout_gif, summarize_results


FULL_TAUS = [0.2, 0.1, 0.05, 0.02]
FULL_DISTANCES = [0.1, 0.2, 0.4, 0.6]
TASKS = ["attached_line", "contact_discovery", "obstacle_arc"]
PLANNERS = ["random", "cem", "gradient", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/full"))
    parser.add_argument("--preset", choices=["quick", "full"], default="full")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--planners", nargs="+", default=PLANNERS, choices=PLANNERS)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--sweep-contact", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--make-gifs", action="store_true")
    parser.add_argument("--gif-seed", type=int, default=0)

    parser.add_argument("--random-samples", type=int, default=None)
    parser.add_argument("--cem-iters", type=int, default=None)
    parser.add_argument("--cem-samples", type=int, default=None)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--gd-iters", type=int, default=None)
    parser.add_argument("--gd-lr", type=float, default=0.08)
    parser.add_argument("--hybrid-cem-iters", type=int, default=None)
    parser.add_argument("--hybrid-cem-samples", type=int, default=None)
    parser.add_argument("--hybrid-gd-iters", type=int, default=None)
    parser.add_argument("--hybrid-gd-lr", type=float, default=0.06)
    return finalize_args(parser.parse_args())


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "quick":
        args.seeds = 2 if args.seeds is None else args.seeds
        args.random_samples = 64 if args.random_samples is None else args.random_samples
        args.cem_iters = 4 if args.cem_iters is None else args.cem_iters
        args.cem_samples = 16 if args.cem_samples is None else args.cem_samples
        args.gd_iters = 24 if args.gd_iters is None else args.gd_iters
        args.hybrid_cem_iters = 3 if args.hybrid_cem_iters is None else args.hybrid_cem_iters
        args.hybrid_cem_samples = 16 if args.hybrid_cem_samples is None else args.hybrid_cem_samples
        args.hybrid_gd_iters = 16 if args.hybrid_gd_iters is None else args.hybrid_gd_iters
        args.sweep_contact = args.sweep_contact
    else:
        args.seeds = 20 if args.seeds is None else args.seeds
        args.random_samples = 512 if args.random_samples is None else args.random_samples
        args.cem_iters = 8 if args.cem_iters is None else args.cem_iters
        args.cem_samples = 64 if args.cem_samples is None else args.cem_samples
        args.gd_iters = 128 if args.gd_iters is None else args.gd_iters
        args.hybrid_cem_iters = 6 if args.hybrid_cem_iters is None else args.hybrid_cem_iters
        args.hybrid_cem_samples = 64 if args.hybrid_cem_samples is None else args.hybrid_cem_samples
        args.hybrid_gd_iters = 64 if args.hybrid_gd_iters is None else args.hybrid_gd_iters
        args.sweep_contact = True if not args.sweep_contact else args.sweep_contact
    return args


def task_conditions(args: argparse.Namespace) -> list[dict]:
    conditions = []
    for task_name in args.tasks:
        if task_name == "contact_discovery" and args.sweep_contact:
            for tau in FULL_TAUS:
                for distance in FULL_DISTANCES:
                    conditions.append(
                        {
                            "task": task_name,
                            "contact_tau": tau,
                            "initial_gripper_distance": distance,
                        }
                    )
        else:
            conditions.append(
                {
                    "task": task_name,
                    "contact_tau": 0.05,
                    "initial_gripper_distance": 0.2,
                }
            )
    return conditions


def result_fields() -> list[str]:
    return [
        "task",
        "planner",
        "seed",
        "contact_tau",
        "initial_gripper_distance",
        "mode",
        "final_loss",
        "success",
        "wall_time",
        "rollouts",
        "horizon",
        "n_particles",
    ]


def run(args: argparse.Namespace) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.csv"
    gif_dir = args.out / "gifs"
    gif_seen = set()

    conditions = task_conditions(args)
    total = len(conditions) * args.seeds * len(args.planners)

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields())
        writer.writeheader()

        with tqdm(total=total, desc="experiments") as pbar:
            for condition in conditions:
                task = make_task(
                    condition["task"],
                    contact_tau=condition["contact_tau"],
                    initial_gripper_distance=condition["initial_gripper_distance"],
                    n_particles=args.n_particles,
                    horizon=args.horizon,
                )

                for seed in range(args.seeds):
                    base_key = jax.random.PRNGKey(seed)
                    for planner_id, planner_name in enumerate(args.planners):
                        key = jax.random.fold_in(base_key, planner_id)
                        result = run_planner(planner_name, key, task, args)
                        success = float(result.final_loss <= task.success_threshold)
                        writer.writerow(
                            {
                                "task": task.name,
                                "planner": result.name,
                                "seed": seed,
                                "contact_tau": task.contact_tau,
                                "initial_gripper_distance": task.initial_gripper_distance,
                                "mode": task.mode,
                                "final_loss": result.final_loss,
                                "success": success,
                                "wall_time": result.wall_time,
                                "rollouts": result.rollouts,
                                "horizon": args.horizon,
                                "n_particles": args.n_particles,
                            }
                        )
                        f.flush()

                        gif_key = (task.name, result.name)
                        if args.make_gifs and seed == args.gif_seed and gif_key not in gif_seen:
                            save_rollout_gif(task, result.actions, gif_dir / f"{task.name}_{result.name}.gif")
                            gif_seen.add(gif_key)

                        pbar.update(1)

    summarize_results(results_path, args.out)
    if args.make_plots:
        plot_results(results_path, args.out)
    return results_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

