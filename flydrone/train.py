"""Train the connectome pilot.

Stage 1 -- DAgger imitation. The fly brain flies; a gyro-equipped autopilot (which sees the
true simulator state) says what it would have done; the brain is trained to match. The
fraction of steps where the autopilot actually has the controls decays from 1 to 0, so the
brain ends up flying on its own and learning to recover from its own mistakes.

Usage:
    python -m flydrone.train --graph real
    python -m flydrone.train --graph rewired
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .connectome import build_flight_graph
from .env import DroneEnv, autopilot, N_OBS
from .policy import ConnectomePilot
from .rollout import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_drone")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
TEACHER = dict(use_gyro=True, k_att=400.0, k_rate=32.0)


def make_pilot(variant, seed=0):
    g = build_flight_graph(verbose=False)
    if variant == "rewired":
        g = g.rewired(seed=seed + 1)
    return g, ConnectomePilot(g, seed=seed).to(DEVICE)


def dagger(pilot, iters=200, B=64, T=32, beta_iters=60, buffer=60_000, mb=512,
           grad_steps=8, lr=1e-3, gain_lr_mult=20.0, eval_every=20, seed=0, label=""):
    env = DroneEnv(batch=B, seed=seed)
    teacher = autopilot(env, **TEACHER)
    N = pilot.N
    H = torch.zeros(buffer, N, dtype=torch.float16, device=DEVICE)
    O = torch.zeros(buffer, N_OBS, device=DEVICE)
    A = torch.zeros(buffer, 4, device=DEVICE)
    size = ptr = 0

    other = [p for n, p in pilot.named_parameters() if n not in ("w_gain", "log_std")]
    opt = torch.optim.Adam([{"params": other, "lr": lr},
                            {"params": [pilot.w_gain], "lr": lr * gain_lr_mult}])
    gen = torch.Generator().manual_seed(seed)

    obs = env.reset()
    h = pilot.init_state(B)
    hist, t0 = [], time.time()
    for it in range(iters):
        beta = max(0.0, 1.0 - it / beta_iters)
        for _ in range(T):
            with torch.no_grad():
                h_new, mean = pilot(h, obs.to(DEVICE))
            a_t = teacher(obs)
            idx = torch.arange(ptr, ptr + B) % buffer
            H[idx], O[idx], A[idx] = h.half(), obs.to(DEVICE), torch.tanh(a_t).to(DEVICE)
            ptr, size = (ptr + B) % buffer, min(size + B, buffer)
            use_t = (torch.rand(B, generator=gen) < beta)[:, None]
            act = torch.where(use_t, a_t, mean.cpu() + 0.1 * torch.randn(B, 4, generator=gen))
            obs, _, info = env.step(act)
            h = h_new
            h[info["done"].to(DEVICE)] = 0

        losses = []
        for _ in range(grad_steps):
            b = torch.randint(0, size, (mb,), device=DEVICE)
            _, m = pilot(H[b].float(), O[b])
            loss = ((torch.tanh(m) - A[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pilot.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        rec = {"iter": it, "beta": beta, "loss": float(np.mean(losses)),
               "samples": (it + 1) * T * B, "minutes": (time.time() - t0) / 60}
        if eval_every and (it % eval_every == 0 or it == iters - 1):
            rec.update(evaluate(pilot, T=750, B=32))
            print(f"[{label}] it {it:4d} | beta {beta:.2f} | loss {rec['loss']:.4f} | "
                  f"fruit/min {rec['fruit_per_min']:5.2f} | crashes/min {rec['crashes_per_min']:5.2f} | "
                  f"{rec['minutes']:5.1f} min", flush=True)
        hist.append(rec)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="real", choices=["real", "rewired"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    name = f"{args.graph}_s{args.seed}"
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    g, pilot = make_pilot(args.graph, args.seed)
    print(f"{name}: {g.summary()} | device {DEVICE}", flush=True)
    hist = dagger(pilot, iters=args.iters, seed=args.seed, label=name)
    torch.save(pilot.state_dict(), os.path.join(out, "pilot.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    final = evaluate(pilot, T=1500, B=64, seed=999)
    json.dump({"variant": args.graph, "seed": args.seed, "final_eval": final},
              open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("final:", final)


if __name__ == "__main__":
    main()
