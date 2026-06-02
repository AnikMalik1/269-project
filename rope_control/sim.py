from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class SimConfig:
    """Physical parameters for a bead-spring rope."""

    n_particles: int = 16
    dt: float = 0.025
    mass: float = 1.0
    rest_length: float = 1.0 / 15.0
    spring_k: float = 70.0
    bend_k: float = 0.06
    velocity_damping: float = 1.1
    obstacle_k: float = 120.0
    obstacle_margin: float = 0.01
    action_limit: float = 0.035
    gripper_k: float = 90.0
    gripper_damping: float = 1.5
    eps: float = 1e-6


class RopeState(NamedTuple):
    positions: Array
    velocities: Array
    gripper: Array


def safe_norm(x: Array, axis: int = -1, keepdims: bool = False, eps: float = 1e-6) -> Array:
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims) + eps)


def spring_forces(positions: Array, cfg: SimConfig) -> Array:
    """Hookean forces between adjacent particles."""

    diffs = positions[1:] - positions[:-1]
    lengths = safe_norm(diffs, keepdims=True, eps=cfg.eps)
    dirs = diffs / lengths
    edge_forces = cfg.spring_k * (lengths - cfg.rest_length) * dirs

    forces = jnp.zeros_like(positions)
    forces = forces.at[:-1].add(edge_forces)
    forces = forces.at[1:].add(-edge_forces)
    return forces


def bending_forces(positions: Array, cfg: SimConfig) -> Array:
    """Forces from squared second-difference bending energy."""

    second_diff = positions[:-2] - 2.0 * positions[1:-1] + positions[2:]
    forces = jnp.zeros_like(positions)
    forces = forces.at[:-2].add(-cfg.bend_k * second_diff)
    forces = forces.at[1:-1].add(2.0 * cfg.bend_k * second_diff)
    forces = forces.at[2:].add(-cfg.bend_k * second_diff)
    return forces


def obstacle_forces(positions: Array, task) -> Array:
    """Soft circular obstacle repulsion. The weight can be zero for no obstacle."""

    cfg = task.sim
    delta = positions - task.obstacle_center
    dist = safe_norm(delta, keepdims=True, eps=cfg.eps)
    penetration = jnp.maximum(task.obstacle_radius + cfg.obstacle_margin - dist, 0.0)
    dirs = delta / dist
    return cfg.obstacle_k * penetration * dirs * task.has_obstacle


def gripper_contact_forces(positions: Array, velocities: Array, gripper: Array, task) -> Array:
    """Soft contact force using a sigmoid contact gate."""

    cfg = task.sim
    delta = gripper[None, :] - positions
    dist = safe_norm(delta, axis=-1, keepdims=True, eps=cfg.eps)
    gate = jax.nn.sigmoid((task.contact_radius - dist) / jnp.maximum(task.contact_tau, cfg.eps))
    spring = cfg.gripper_k * gate * delta
    damping = -cfg.gripper_damping * gate * velocities
    return spring + damping


def total_forces(state: RopeState, task, gripper_next: Array) -> Array:
    cfg = task.sim
    forces = spring_forces(state.positions, cfg)
    forces = forces + bending_forces(state.positions, cfg)
    forces = forces - cfg.velocity_damping * state.velocities
    forces = forces + obstacle_forces(state.positions, task)

    if task.mode == "soft_contact":
        forces = forces + gripper_contact_forces(
            state.positions,
            state.velocities,
            gripper_next,
            task,
        )

    if task.mode == "attached":
        forces = forces.at[-1].set(0.0)

    return forces


def step(task, state: RopeState, action: Array) -> RopeState:
    """One semi-implicit Euler control step."""

    cfg = task.sim
    delta = jnp.clip(action, -cfg.action_limit, cfg.action_limit)
    gripper_next = state.gripper + delta
    forces = total_forces(state, task, gripper_next)

    velocities = state.velocities + cfg.dt * forces / cfg.mass
    positions = state.positions + cfg.dt * velocities

    if task.mode == "attached":
        positions = positions.at[-1].set(gripper_next)
        velocities = velocities.at[-1].set(delta / cfg.dt)

    return RopeState(positions=positions, velocities=velocities, gripper=gripper_next)


def rollout(task, actions: Array) -> tuple[RopeState, dict[str, Array]]:
    """Roll out a task for an action sequence.

    Actions are 2D gripper position deltas, one per control step.
    """

    init_state = RopeState(
        positions=task.init_positions,
        velocities=task.init_velocities,
        gripper=task.init_gripper,
    )

    def body(state: RopeState, action: Array):
        next_state = step(task, state, action)
        trace_item = {
            "positions": next_state.positions,
            "velocities": next_state.velocities,
            "gripper": next_state.gripper,
        }
        return next_state, trace_item

    final_state, trace = jax.lax.scan(body, init_state, actions)
    trace = {
        "positions": jnp.concatenate([task.init_positions[None, :, :], trace["positions"]], axis=0),
        "velocities": jnp.concatenate([task.init_velocities[None, :, :], trace["velocities"]], axis=0),
        "gripper": jnp.concatenate([task.init_gripper[None, :], trace["gripper"]], axis=0),
    }
    return final_state, trace


def shape_loss(final_positions: Array, target_positions: Array) -> Array:
    return jnp.mean(jnp.sum((final_positions - target_positions) ** 2, axis=-1))


def action_smoothness(actions: Array) -> Array:
    diffs = actions[1:] - actions[:-1]
    return jnp.mean(jnp.sum(diffs * diffs, axis=-1))


def obstacle_collision_loss(position_trace: Array, task) -> Array:
    cfg = task.sim
    delta = position_trace - task.obstacle_center
    dist = safe_norm(delta, axis=-1, eps=cfg.eps)
    penetration = jnp.maximum(task.obstacle_radius + cfg.obstacle_margin - dist, 0.0)
    return jnp.mean(penetration * penetration) * task.has_obstacle


def rollout_loss(task, actions: Array) -> Array:
    final_state, trace = rollout(task, actions)
    loss = shape_loss(final_state.positions, task.target_positions)
    loss = loss + task.action_smoothness_weight * action_smoothness(actions)
    loss = loss + task.collision_penalty_weight * obstacle_collision_loss(trace["positions"], task)
    return loss


def batched_rollout_loss(task):
    """A simple jitted/vmapped loss evaluator for population planners."""

    return jax.jit(jax.vmap(lambda actions: rollout_loss(task, actions)))


def loss_and_grad(task):
    return jax.jit(jax.value_and_grad(lambda actions: rollout_loss(task, actions)))
