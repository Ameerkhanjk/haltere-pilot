"""Sanity check: is the drone task flyable, and does the gyro (haltere) signal matter?"""
import time
import torch
from .env import DroneEnv, autopilot, N_ACT


@torch.no_grad()
def run(ctrl_fn, T=1500, B=64, device="cpu", seed=3, **cfg):
    env = DroneEnv(batch=B, device=device, seed=seed, **cfg)
    obs = env.reset()
    ctrl = ctrl_fn(env)
    tot = torch.zeros(B, device=device)
    n = {"reached": 0, "crashed": 0, "hit_tree": 0, "hit_ground": 0}
    for _ in range(T):
        obs, r, info = env.step(ctrl(obs))
        tot += r
        for k in n:
            n[k] += info[k].sum().item()
    minutes = T * B * 0.02 / 60
    return {"return/ep_len": tot.mean().item(), "fruit/min": n["reached"] / minutes,
            "crashes/min": n["crashed"] / minutes, "tree": n["hit_tree"], "ground": n["hit_ground"]}


if __name__ == "__main__":
    pilots = {
        "random": lambda env: (lambda o: torch.randn(o.shape[0], N_ACT) * 0.5),
        "hover (do nothing)": lambda env: (lambda o: torch.zeros(o.shape[0], N_ACT)),
        "autopilot + gyro": lambda env: autopilot(env, use_gyro=True, k_att=400, k_rate=32),
        "autopilot, no gyro": lambda env: autopilot(env, use_gyro=False, k_att=50, k_rate=16),
    }
    for name, p in pilots.items():
        t0 = time.time()
        r = run(p)
        print(f"{name:30s} " + " | ".join(f"{k} {v:7.2f}" for k, v in r.items()) + f"  ({time.time()-t0:.1f}s)")
