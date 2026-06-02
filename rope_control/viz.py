from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from rope_control.sim import rollout


def summarize_results(csv_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    summary = (
        df.groupby(["task", "planner"], dropna=False)
        .agg(
            mean_final_loss=("final_loss", "mean"),
            std_final_loss=("final_loss", "std"),
            success_rate=("success", "mean"),
            mean_wall_time=("wall_time", "mean"),
            mean_rollouts=("rollouts", "mean"),
            n=("success", "size"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_markdown_table(summary, out_dir / "summary.md")
    return summary


def write_markdown_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.5g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_results(csv_path: str | Path, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    _plot_success_vs_tau(df, out_dir / "success_vs_contact_tau.png")
    _plot_success_vs_distance(df, out_dir / "success_vs_initial_distance.png")
    _plot_final_loss_by_planner(df, out_dir / "final_loss_by_planner.png")


def _line_plot(grouped: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    for planner, planner_df in grouped.groupby("planner"):
        planner_df = planner_df.sort_values(x)
        ax.plot(planner_df[x], planner_df[y], marker="o", linewidth=2, label=planner)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_success_vs_tau(df: pd.DataFrame, path: Path) -> None:
    contact = df[df["task"] == "contact_discovery"].copy()
    if contact.empty:
        return
    grouped = contact.groupby(["planner", "contact_tau"], as_index=False)["success"].mean()
    _line_plot(
        grouped,
        x="contact_tau",
        y="success",
        path=path,
        title="Success vs contact sharpness",
        xlabel="contact_tau (smaller is sharper)",
    )


def _plot_success_vs_distance(df: pd.DataFrame, path: Path) -> None:
    contact = df[df["task"] == "contact_discovery"].copy()
    if contact.empty:
        return
    grouped = contact.groupby(["planner", "initial_gripper_distance"], as_index=False)["success"].mean()
    _line_plot(
        grouped,
        x="initial_gripper_distance",
        y="success",
        path=path,
        title="Success vs initial gripper distance",
        xlabel="initial gripper distance",
    )


def _plot_final_loss_by_planner(df: pd.DataFrame, path: Path) -> None:
    grouped = df.groupby("planner")["final_loss"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    ax.bar(grouped["planner"], grouped["mean"], yerr=grouped["std"], capsize=4)
    ax.set_title("Final loss by planner")
    ax.set_xlabel("Planner")
    ax.set_ylabel("Final loss")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_rollout_gif(task, actions, path: str | Path, fps: int = 16) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, trace = rollout(task, actions)
    trace = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), trace)
    target = np.asarray(jax.device_get(task.target_positions))
    init = np.asarray(jax.device_get(task.init_positions))

    positions = trace["positions"]
    grippers = trace["gripper"]
    all_xy = np.concatenate(
        [
            positions.reshape(-1, 2),
            target.reshape(-1, 2),
            init.reshape(-1, 2),
            grippers.reshape(-1, 2),
        ],
        axis=0,
    )
    pad = 0.18
    xmin, ymin = all_xy.min(axis=0) - pad
    xmax, ymax = all_xy.max(axis=0) + pad

    frame_ids = np.linspace(0, len(positions) - 1, min(80, len(positions))).astype(int)
    frames = []
    for idx in frame_ids:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=100)
        ax.plot(target[:, 0], target[:, 1], "--", color="#555555", linewidth=1.5, label="target")
        ax.plot(init[:, 0], init[:, 1], ":", color="#999999", linewidth=1.2, label="start")
        ax.plot(positions[idx, :, 0], positions[idx, :, 1], "-o", color="#1f77b4", markersize=4)
        ax.scatter(grippers[idx, 0], grippers[idx, 1], s=70, color="#d62728", marker="x", linewidths=2)
        if task.has_obstacle:
            circle = Circle(
                tuple(np.asarray(task.obstacle_center)),
                task.obstacle_radius,
                facecolor="#cccccc",
                edgecolor="#444444",
                alpha=0.55,
            )
            ax.add_patch(circle)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{task.name} | t={idx:03d}")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(path, frames, fps=fps)

