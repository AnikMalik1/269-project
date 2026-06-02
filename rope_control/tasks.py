from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

from rope_control.sim import SimConfig


@dataclass(frozen=True)
class RopeTask:
    name: str
    mode: str
    sim: SimConfig
    horizon: int
    init_positions: jnp.ndarray
    init_velocities: jnp.ndarray
    init_gripper: jnp.ndarray
    target_positions: jnp.ndarray
    contact_radius: float
    contact_tau: float
    initial_gripper_distance: float
    obstacle_center: jnp.ndarray
    obstacle_radius: float
    has_obstacle: float
    action_smoothness_weight: float
    collision_penalty_weight: float
    success_threshold: float


def straight_rope(n_particles: int, length: float = 1.0, y: float = 0.0) -> jnp.ndarray:
    x = jnp.linspace(-0.5 * length, 0.5 * length, n_particles)
    return jnp.stack([x, jnp.full_like(x, y)], axis=-1)


def arc_target(n_particles: int, center=(0.0, 0.02), radius: float = 0.34) -> jnp.ndarray:
    theta = jnp.linspace(jnp.pi, 0.0, n_particles)
    return jnp.stack(
        [
            center[0] + radius * jnp.cos(theta),
            center[1] + radius * jnp.sin(theta),
        ],
        axis=-1,
    )


def _base_config(n_particles: int, horizon: int) -> SimConfig:
    return SimConfig(
        n_particles=n_particles,
        rest_length=1.0 / (n_particles - 1),
        action_limit=0.035,
    )


def make_task(
    name: str,
    *,
    contact_tau: float = 0.05,
    initial_gripper_distance: float = 0.2,
    n_particles: int = 16,
    horizon: int = 64,
) -> RopeTask:
    """Create one of the benchmark tasks."""

    if name not in {"attached_line", "contact_discovery", "obstacle_arc"}:
        raise ValueError(f"Unknown task: {name}")

    sim = _base_config(n_particles, horizon)
    init_positions = straight_rope(n_particles)
    init_velocities = jnp.zeros_like(init_positions)
    obstacle_center = jnp.array([0.0, 0.0])
    obstacle_radius = 0.0
    has_obstacle = 0.0
    collision_weight = 0.0
    smooth_weight = 1e-3
    contact_radius = 0.12

    if name == "attached_line":
        target_positions = init_positions + jnp.array([0.0, 0.28])
        init_gripper = init_positions[-1]
        mode = "attached"
        success_threshold = 0.025

    elif name == "contact_discovery":
        target_positions = init_positions + jnp.array([0.12, 0.28])
        init_gripper = init_positions[-1] + jnp.array([0.02, initial_gripper_distance])
        mode = "soft_contact"
        success_threshold = 0.035

    else:
        target_positions = arc_target(n_particles)
        init_gripper = init_positions[-1]
        mode = "attached"
        obstacle_center = jnp.array([0.0, 0.08])
        obstacle_radius = 0.17
        has_obstacle = 1.0
        collision_weight = 20.0
        smooth_weight = 2e-3
        success_threshold = 0.045

    return RopeTask(
        name=name,
        mode=mode,
        sim=sim,
        horizon=horizon,
        init_positions=init_positions,
        init_velocities=init_velocities,
        init_gripper=init_gripper,
        target_positions=target_positions,
        contact_radius=contact_radius,
        contact_tau=contact_tau,
        initial_gripper_distance=initial_gripper_distance,
        obstacle_center=obstacle_center,
        obstacle_radius=obstacle_radius,
        has_obstacle=has_obstacle,
        action_smoothness_weight=smooth_weight,
        collision_penalty_weight=collision_weight,
        success_threshold=success_threshold,
    )


def with_horizon(task: RopeTask, horizon: int) -> RopeTask:
    return replace(task, horizon=horizon)

