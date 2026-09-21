"""Experiments on the trained pilots. Writes figures + analysis.json to results_drone/.

1. Leaderboard       real connectome vs rewired control vs hand-written autopilots
2. Sense ablations   silence each sensory port; compare with silencing as many random neurons
3. Gust impulse      a controlled torque kick while hovering: peak tilt, recovery, and the
                     how each neuron group's activity changes
4. Reflex-arc cut    delete only the direct haltere -> wing-motor-neuron synapses
5. Synapse drift     how far training moved each synapse from its biological strength

Usage:  python -m flydrone.analyze
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .baselines import run as run_baseline
from .connectome import PORTS, PORT_LABELS
from .env import DroneEnv, autopilot, DT
from .record import neuron_groups
from .rollout import evaluate
from .train import make_pilot, OUT, DEVICE

FIG = os.path.join(OUT, "figures")
INK, MUTED, GRID = "#1d2a2e", "#5d6f74", "#e3e9ea"
C_REAL, C_RW, C_ABL = "#0f8b8d", "#d97706", "#c2410c"
GROUP_COLORS = {"haltere": "#0f8b8d", "jo": "#65a30d", "lptc": "#2563eb", "ocelli": "#ca8a04",
                "lc10": "#db2777", "loom": "#ea580c", "descending": "#9a3412", "vnc": "#475569",
                "wing_mn": "#e11d48"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d3d5")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


def load(variant):
    g, p = make_pilot(variant)
    p.load_state_dict(torch.load(os.path.join(OUT, f"{variant}_s0", "pilot.pt"), map_location=DEVICE))
    p.eval()
    return g, p


# ------------------------------------------------------------------ 1. leaderboard
def leaderboard(pilots):
    rows = {}
    for name, p in pilots.items():
        rows[name] = evaluate(p, T=1500, B=64, seed=999)
    for name, kw in [("autopilot + gyro", dict(use_gyro=True, k_att=400, k_rate=32)),
                     ("autopilot, no gyro", dict(use_gyro=False, k_att=50, k_rate=16))]:
        r = run_baseline(lambda env: autopilot(env, **kw), T=1500, B=64, seed=999)
        rows[name] = {"fruit_per_min": r["fruit/min"], "crashes_per_min": r["crashes/min"]}
    r = run_baseline(lambda env: (lambda o: torch.randn(o.shape[0], 4) * 0.5), T=1500, B=64, seed=999)
    rows["random"] = {"fruit_per_min": r["fruit/min"], "crashes_per_min": r["crashes/min"]}
    return rows


# ------------------------------------------------------------------ 2. sensory ablations
def port_ablations(g, p, seed=0):
    rng = np.random.default_rng(seed)
    base = evaluate(p, T=1000, B=48, seed=4321)
    pool = np.setdiff1d(np.arange(g.n_nodes), np.concatenate([g.port_idx, g.out_idx]))
    rows = []
    for port in PORTS:
        p.reset_manipulations()
        p.silence_port(port)
        les = evaluate(p, T=1000, B=48, seed=4321)
        n = len(g.port(port))
        p.reset_manipulations()
        p.lesion_mask[torch.as_tensor(rng.choice(pool, n, replace=False), device=DEVICE)] = 0
        rnd = evaluate(p, T=1000, B=48, seed=4321)
        p.reset_manipulations()
        rows.append({"port": port, "label": PORT_LABELS[port], "n": int(n),
                     "fruit": les["fruit_per_min"], "crashes": les["crashes_per_min"],
                     "random_fruit": rnd["fruit_per_min"], "random_crashes": rnd["crashes_per_min"]})
        print(f"  silence {port:8s} n={n:4d} | fruit/min {les['fruit_per_min']:5.2f} "
              f"(random {rnd['fruit_per_min']:5.2f}) | crashes/min {les['crashes_per_min']:5.2f} "
              f"(random {rnd['crashes_per_min']:5.2f})")
    return base, rows


# ------------------------------------------------------------------ 3. gust impulse test
@torch.no_grad()
def gust_impulse(g, p, B=64, torque=90.0, settle=50, after=60, seed=11, silence=None, cut_edges=None):
    """Hover, then hit every drone with the same-size torque kick in a random horizontal
    direction for 6 steps (120 ms). Returns tilt/rate traces and per-group activity."""
    env = DroneEnv(batch=B, seed=seed, gust_prob=0.0, turbulence=0.0, torque_noise=0.0,
                   mean_wind=0.0, n_trees=0, target_dist=(0.01, 0.02), reach_radius=0.0)
    obs = env.reset()
    env.target = env.pos.clone()                       # hover in place
    env.prev_dist = torch.zeros(B)
    p.reset_manipulations()
    saved = None
    if silence:
        p.silence_port(silence)
    if cut_edges is not None:
        saved = p.w_gain.data[cut_edges].clone()
        p.w_gain.data[cut_edges] = 0
    grp = neuron_groups(g)
    gidx = {k: torch.as_tensor(np.where(grp == k)[0], device=DEVICE) for k in GROUP_COLORS}
    h = p.init_state(B)
    tilt, rate, acts, alive = [], [], {k: [] for k in gidx}, torch.ones(B, dtype=torch.bool)
    for t in range(settle + after):
        if t == settle:
            ang = torch.rand(B) * 2 * np.pi
            env.gust = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros(B)], -1) * torque
            env.gust_left[:] = 6
        env.target = env.pos.detach().clone() if t < settle else env.target
        h, mean = p(h, obs.to(DEVICE))
        obs, _, info = env.step(mean.cpu())
        alive &= ~info["done"]
        if t >= settle - 10:
            tilt.append(np.degrees(info["tilt"].numpy()))
            rate.append(env.w.norm(dim=-1).numpy())
            for k, idx in gidx.items():
                acts[k].append(h[:, idx].mean(1).cpu().numpy())
    if saved is not None:
        p.w_gain.data[cut_edges] = saved
    p.reset_manipulations()
    tilt, rate = np.array(tilt), np.array(rate)
    return {"tilt": tilt, "rate": rate, "acts": {k: np.array(v) for k, v in acts.items()},
            "crashed_frac": float(1 - alive.float().mean()), "t": (np.arange(len(tilt)) - 10) * DT * 1000}


def direct_haltere_edges(g, p):
    """Indices (in the pilot's edge order) of synapses from haltere afferents onto wing MNs."""
    src = g.src[p.edge_order]
    dst = g.dst[p.edge_order]
    return np.where(np.isin(src, g.port("haltere")) & np.isin(dst, g.out_idx))[0]


def onset_latency(trace, t):
    """Time (ms after the kick) at which a group's mean response peaks. Every group starts
    responding within the first 20 ms control step, so onset is below our time resolution;
    the peak time is what separates the fast reflex from the slower visual loop."""
    base = trace[:10].mean(0)
    dev = np.abs(trace - base).mean(1)[10:]
    if dev.max() <= 1e-6:
        return None
    return float(t[10 + int(np.argmax(dev))])


# ------------------------------------------------------------------ figures
def fig_leaderboard(rows):
    order = ["autopilot + gyro", "real connectome", "rewired control", "autopilot, no gyro", "random"]
    order = [o for o in order if o in rows]
    colors = {"real connectome": C_REAL, "rewired control": C_RW}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title in [(axes[0], "fruit_per_min", "Fruit collected per minute  (higher is better)"),
                           (axes[1], "crashes_per_min", "Crashes per minute  (lower is better)")]:
        vals = [rows[o][key] for o in order]
        ax.barh(order[::-1], vals[::-1], color=[colors.get(o, "#b9c6c8") for o in order[::-1]], height=0.62)
        for y, v in enumerate(vals[::-1]):
            ax.text(v, y, f"  {v:.1f}", va="center", fontsize=9, color=INK)
        style(ax); ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.margins(x=0.18)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "leaderboard.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_ablations(base, rows, label):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(rows))
    ax.barh(y + 0.18, [r["fruit"] for r in rows], height=0.34, color=C_ABL, label="port silenced")
    ax.barh(y - 0.18, [r["random_fruit"] for r in rows], height=0.34, color="#b9c6c8", label="same number of random neurons silenced")
    ax.axvline(base["fruit_per_min"], color=INK, linestyle="--", linewidth=1)
    ax.text(base["fruit_per_min"], len(rows) - 0.4, " intact", color=INK, fontsize=9)
    ax.set_yticks(y, [f"{r['label']}  (n={r['n']})" for r in rows])
    ax.invert_yaxis()
    style(ax); ax.grid(axis="y", visible=False)
    ax.set_xlabel("fruit collected per minute", color=MUTED, fontsize=9)
    ax.set_title(f"Which senses the {label} pilot depends on", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "sense_ablations.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_gust(res, lat):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for key, lab, col in [("intact", "intact", C_REAL), ("no_halteres", "halteres silenced (cannot even hover)", C_ABL),
                          ("cut_arc", "direct haltere→MN synapses cut", "#7c3aed"),
                          ("rewired", "rewired control", C_RW)]:
        if key not in res:
            continue
        r = res[key]
        m = np.median(r["tilt"], 1)
        lo, hi = np.percentile(r["tilt"], [25, 75], axis=1)
        ax.plot(r["t"], m, color=col, linewidth=2, label=f"{lab}  (crashed {100*r['crashed_frac']:.0f}%)")
        ax.fill_between(r["t"], lo, hi, color=col, alpha=0.12, linewidth=0)
    ax.axvspan(0, 120, color="#fbbf24", alpha=0.15, linewidth=0)
    ax.text(4, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1, "gust", color="#92400e", fontsize=9)
    style(ax)
    ax.set_xlabel("ms after the gust hits", color=MUTED, fontsize=9)
    ax.set_ylabel("tilt from level (°, median, IQR)", color=MUTED, fontsize=9)
    ax.set_title("Recovering from a 90 rad/s² gust", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    r = res["intact"]
    for k, col in GROUP_COLORS.items():
        tr = r["acts"][k]
        dev = np.abs(tr - tr[:10].mean(0)).mean(1)
        if dev.max() < 1e-6:
            continue
        lab = f"{k}  ({lat[k]:.0f} ms)" if lat.get(k) is not None else k
        ax.plot(r["t"], dev / dev.max(), color=col, linewidth=1.8, label=lab)
    ax.axvspan(0, 120, color="#fbbf24", alpha=0.15, linewidth=0)
    style(ax)
    ax.set_xlim(-200, 600)
    ax.set_xlabel("ms after the gust hits", color=MUTED, fontsize=9)
    ax.set_ylabel("response (normalised change in activity)", color=MUTED, fontsize=9)
    ax.set_title("Neural response to the gust (real connectome)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, title="group  (time to peak)", title_fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "gust_response.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_gains(pilots):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(-1, 3, 90)
    for name, p in pilots.items():
        gvals = p.w_gain.detach().cpu().numpy()
        ax.hist(np.clip(gvals, -1, 3), bins=bins, alpha=0.6, label=name,
                color=C_REAL if "real" in name else C_RW)
    ax.axvline(1, color=INK, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    style(ax)
    ax.set_xlabel("learned synapse gain  (1 = unchanged from the connectome)", color=MUTED, fontsize=9)
    ax.set_ylabel("synapses (log)", color=MUTED, fontsize=9)
    ax.set_title("How far training moved each synapse from biology", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "synapse_gains.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    g, p = load("real")
    pilots = {"real connectome": p}
    have_rw = os.path.exists(os.path.join(OUT, "rewired_s0", "pilot.pt"))
    if have_rw:
        g_rw, p_rw = load("rewired")
        pilots["rewired control"] = p_rw

    print("1. leaderboard")
    board = leaderboard(pilots)
    for k, v in board.items():
        print(f"  {k:22s} fruit/min {v['fruit_per_min']:6.2f} | crashes/min {v['crashes_per_min']:6.2f}")
    fig_leaderboard(board)

    print("2. sensory ablations (real connectome)")
    base, abl = port_ablations(g, p)
    fig_ablations(base, abl, "real-connectome")

    print("3/4. gust impulse test")
    cut = direct_haltere_edges(g, p)
    res = {"intact": gust_impulse(g, p), "no_halteres": gust_impulse(g, p, silence="haltere"),
           "cut_arc": gust_impulse(g, p, cut_edges=torch.as_tensor(cut, device=DEVICE))}
    if have_rw:
        res["rewired"] = gust_impulse(g_rw, p_rw)
    lat = {k: onset_latency(res["intact"]["acts"][k], res["intact"]["t"]) for k in GROUP_COLORS}
    summary_gust = {}
    for k, r in res.items():
        post = r["tilt"][10:]
        summary_gust[k] = {"peak_tilt_deg": float(np.median(post.max(0))),
                           "tilt_at_300ms_deg": float(np.median(post[15])),
                           "crashed_frac": r["crashed_frac"]}
        print(f"  {k:12s} peak tilt {summary_gust[k]['peak_tilt_deg']:6.1f}° | "
              f"tilt @300ms {summary_gust[k]['tilt_at_300ms_deg']:6.1f}° | crashed {100*r['crashed_frac']:.0f}%")
    print("  response latency (ms):", {k: v for k, v in lat.items()})
    print(f"  direct haltere->wing MN synapses cut: {len(cut)} connections")
    fig_gust(res, lat)

    print("5. synapse drift")
    fig_gains(pilots)
    drift = {n: {"frac_moved_10pct": float((pp.w_gain.detach().cpu() - 1).abs().gt(0.1).float().mean()),
                 "median_abs_change": float((pp.w_gain.detach().cpu() - 1).abs().median())}
             for n, pp in pilots.items()}
    print(" ", drift)

    json.dump({"leaderboard": board, "ablation_base": base, "ablations": abl, "gust": summary_gust,
               "latency_ms": lat, "n_direct_haltere_mn_edges": int(len(cut)), "synapse_drift": drift},
              open(os.path.join(OUT, "analysis.json"), "w"), indent=2)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
