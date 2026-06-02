from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rope_control.sim import batched_rollout_loss, loss_and_grad, rollout_loss


@dataclass(frozen=True)
class PlannerResult:
    name: str
    actions: jax.Array
    final_loss: float
    wall_time: float
    rollouts: int


def _block(x):
    return jax.block_until_ready(x)


def random_shooting(key, task, num_samples: int = 512) -> PlannerResult:
    """Uniformly sample action sequences and keep the best one."""

    start = time.perf_counter()
    losses_fn = batched_rollout_loss(task)
    actions = jax.random.uniform(
        key,
        (num_samples, task.horizon, 2),
        minval=-task.sim.action_limit,
        maxval=task.sim.action_limit,
    )
    losses = _block(losses_fn(actions))
    best_idx = int(jnp.argmin(losses))
    best_actions = actions[best_idx]
    best_loss = float(losses[best_idx])
    return PlannerResult("random", best_actions, best_loss, time.perf_counter() - start, num_samples)


def cem(
    key,
    task,
    num_iters: int = 8,
    num_samples: int = 64,
    elite_frac: float = 0.1,
    init_mean=None,
    init_std=None,
) -> PlannerResult:
    """Cross-entropy method over action sequences."""

    start = time.perf_counter()
    losses_fn = batched_rollout_loss(task)
    elite_count = max(1, int(num_samples * elite_frac))
    mean = jnp.zeros((task.horizon, 2)) if init_mean is None else init_mean
    std = jnp.full((task.horizon, 2), task.sim.action_limit) if init_std is None else init_std
    best_actions = mean
    best_loss = jnp.asarray(jnp.inf)

    for i in range(num_iters):
        key, subkey = jax.random.split(key)
        samples = mean[None, :, :] + std[None, :, :] * jax.random.normal(
            subkey,
            (num_samples, task.horizon, 2),
        )
        samples = jnp.clip(samples, -task.sim.action_limit, task.sim.action_limit)
        losses = _block(losses_fn(samples))
        order = jnp.argsort(losses)
        elites = samples[order[:elite_count]]
        elite_losses = losses[order[:elite_count]]
        mean = jnp.mean(elites, axis=0)
        std = jnp.maximum(jnp.std(elites, axis=0), 0.05 * task.sim.action_limit)

        iter_best_loss = elite_losses[0]
        iter_best_actions = elites[0]
        take_iter = iter_best_loss < best_loss
        best_loss = jnp.where(take_iter, iter_best_loss, best_loss)
        best_actions = jnp.where(take_iter, iter_best_actions, best_actions)

    best_loss = _block(rollout_loss(task, best_actions))
    return PlannerResult(
        "cem",
        best_actions,
        float(best_loss),
        time.perf_counter() - start,
        num_iters * num_samples + 1,
    )


def gradient_descent(
    task,
    init_actions,
    num_iters: int = 128,
    lr: float = 0.08,
    name: str = "gradient",
) -> PlannerResult:
    """Gradient descent on the differentiable rollout objective."""

    start = time.perf_counter()
    loss_grad_fn = loss_and_grad(task)
    actions = jnp.asarray(init_actions)

    for _ in range(num_iters):
        loss, grad = loss_grad_fn(actions)
        _block(loss)
        actions = actions - lr * grad
        actions = jnp.clip(actions, -task.sim.action_limit, task.sim.action_limit)

    final_loss = _block(rollout_loss(task, actions))
    return PlannerResult(
        name,
        actions,
        float(final_loss),
        time.perf_counter() - start,
        num_iters + 1,
    )


def hybrid(
    key,
    task,
    cem_iters: int = 6,
    cem_samples: int = 64,
    elite_frac: float = 0.1,
    gd_iters: int = 64,
    gd_lr: float = 0.06,
) -> PlannerResult:
    """Run CEM, then refine its best action sequence with gradients."""

    cem_result = cem(
        key,
        task,
        num_iters=cem_iters,
        num_samples=cem_samples,
        elite_frac=elite_frac,
    )
    gd_result = gradient_descent(
        task,
        cem_result.actions,
        num_iters=gd_iters,
        lr=gd_lr,
        name="hybrid",
    )
    return PlannerResult(
        "hybrid",
        gd_result.actions,
        gd_result.final_loss,
        cem_result.wall_time + gd_result.wall_time,
        cem_result.rollouts + gd_result.rollouts,
    )


def run_planner(name: str, key, task, args) -> PlannerResult:
    if name == "random":
        return random_shooting(key, task, num_samples=args.random_samples)
    if name == "cem":
        return cem(
            key,
            task,
            num_iters=args.cem_iters,
            num_samples=args.cem_samples,
            elite_frac=args.elite_frac,
        )
    if name == "gradient":
        init_actions = jnp.zeros((task.horizon, 2))
        return gradient_descent(task, init_actions, num_iters=args.gd_iters, lr=args.gd_lr)
    if name == "hybrid":
        return hybrid(
            key,
            task,
            cem_iters=args.hybrid_cem_iters,
            cem_samples=args.hybrid_cem_samples,
            elite_frac=args.elite_frac,
            gd_iters=args.hybrid_gd_iters,
            gd_lr=args.hybrid_gd_lr,
        )
    raise ValueError(f"Unknown planner: {name}")

