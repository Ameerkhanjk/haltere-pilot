"""Running a pilot in the world: evaluation metrics and recording episodes for the viewer."""
import numpy as np
import torch

from .env import DroneEnv, DT


def _dev(pilot):
    return next(pilot.parameters()).device


@torch.no_grad()
def evaluate(pilot, T=1500, B=64, seed=1234, env_cfg=None):
    """Deterministic (mean-action) flight. Returns per-minute rates so numbers are comparable
    with flydrone.baselines."""
    env = DroneEnv(batch=B, seed=seed, **(env_cfg or {}))
    dev = _dev(pilot)
    obs = env.reset()
    h = pilot.init_state(B)
    tot = torch.zeros(B)
    n = {"reached": 0, "crashed": 0, "hit_tree": 0, "hit_ground": 0}
    tilt = []
    for _ in range(T):
        h, mean = pilot(h, obs.to(dev))
        obs, r, info = env.step(mean.cpu())
        h[info["done"].to(dev)] = 0
        tot += r
        for k in n:
            n[k] += info[k].sum().item()
        tilt.append(info["tilt"].mean().item())
    minutes = T * B * DT / 60
    return {"return": tot.mean().item(), "fruit_per_min": n["reached"] / minutes,
            "crashes_per_min": n["crashed"] / minutes, "tree_crashes": n["hit_tree"],
            "ground_crashes": n["hit_ground"], "mean_tilt_deg": float(np.degrees(np.mean(tilt)))}
