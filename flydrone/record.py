"""Record episodes for the 3D viewer (viewer/index.html).

Writes into viewer/data/ as plain <script> files, so the viewer opens straight from disk
(file://) with no server:
    brain.js        neuron positions, groups and names (shared by all episodes)
    ep_<name>.js    drone trajectory, world, events, per-group traces, and the uint8 activity of
                    every neuron (frames x neurons, base64)
    episodes.js     index the viewer reads first

Usage:
    python -m flydrone.record                     # intact / halteres removed / rewired
"""
import argparse
import base64
import json
import os

import numpy as np
import torch

from .connectome import PORTS, PORT_LABELS
from .env import DroneEnv, DT
from .train import make_pilot, OUT, DEVICE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "viewer", "data")

GROUPS = list(PORTS) + ["descending", "ascending", "brain", "vnc", "wing_mn", "other"]
GROUP_LABELS = {**PORT_LABELS, "descending": "Descending neurons", "ascending": "Ascending neurons",
                "brain": "Brain interneurons", "vnc": "VNC interneurons",
                "wing_mn": "Wing motor neurons", "other": "Other"}


def _write_js(fname, var, obj):
    with open(os.path.join(DATA, fname), "w") as f:
        if var.startswith("FLY_EP["):
            f.write("window.FLY_EP = window.FLY_EP || {};\n")
        f.write(f"window.{var} = ")
        json.dump(obj, f, separators=(",", ":"))
        f.write(";\n")


def write_index(index):
    _write_js("episodes.js", "FLY_EPISODES", index)


def bundle():
    """Inline every data script into one self-contained viewer/haltere_pilot.html."""
    idx_js = open(os.path.join(DATA, "episodes.js")).read()
    names = json.loads(idx_js.split("=", 1)[1].rstrip().rstrip(";"))
    parts = [idx_js, open(os.path.join(DATA, "brain.js")).read()]
    parts += [open(os.path.join(DATA, f"ep_{e['name']}.js")).read() for e in names]
    html = open(os.path.join(ROOT, "viewer", "index.html")).read()
    inline = "".join(f"<script>{p}</script>\n" for p in parts)
    out = os.path.join(ROOT, "viewer", "haltere_pilot.html")
    with open(out, "w") as f:
        f.write(html.replace('<script type="importmap">', inline + '<script type="importmap">', 1))
    print(f"bundled -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def neuron_groups(g):
    grp = np.full(g.n_nodes, "other", dtype=object)
    sc = g.superclass
    grp[np.isin(sc, ["cb_intrinsic", "visual_projection", "visual_centrifugal", "ol_intrinsic"])] = "brain"
    grp[sc == "vnc_intrinsic"] = "vnc"
    grp[sc == "ascending_neuron"] = "ascending"
    grp[sc == "descending_neuron"] = "descending"
    grp[g.out_idx] = "wing_mn"
    for p in PORTS:
        grp[g.port(p)] = p
    return grp.astype(str)


def brain_dict(g):
    """Neuron positions, groups and names, in the format the viewer expects."""
    grp = neuron_groups(g)
    xyz = g.xyz.astype(np.float64)
    # male CNS voxel space: z runs head -> tail along the body, x is left/right, y dorsal/ventral.
    # Display frame: X = body axis (head at -X), Y = up, Z = left/right.
    c = np.median(xyz, 0)
    s = np.percentile(np.abs(xyz - c), 99)
    pos = np.stack([(xyz[:, 2] - c[2]), -(xyz[:, 1] - c[1]), (xyz[:, 0] - c[0])], 1) / s
    labels = {gname: GROUP_LABELS[gname] for gname in GROUPS}
    brain = {
        "n": int(g.n_nodes),
        "pos": np.round(pos, 3).ravel().tolist(),
        "group": [GROUPS.index(x) for x in grp],
        "groups": GROUPS, "group_labels": labels,
        "type": g.type.tolist(),
        "neuropil": g.neuropil.tolist(),
        "out_group": {int(i): str(gr) for i, gr in zip(g.out_idx, g.out_group)},
    }
    return brain


def write_brain(g):
    _write_js("brain.js", "FLY_BRAIN", brain_dict(g))
    return neuron_groups(g)


@torch.no_grad()
def record(pilot, g, name, title, seed=7, T=1200, every=4, manip=None, ref_range=None):
    env = DroneEnv(batch=1, seed=seed)
    obs = env.reset()
    pilot.reset_manipulations()
    if manip == "no_halteres":
        pilot.silence_port("haltere")
    h = pilot.init_state(1)
    grp = neuron_groups(g)
    frames, acts, events = [], [], []
    fruit = crashes = 0
    world = {"trees": env.trees[0].tolist(), "arena": env.cfg["arena"]}
    for t in range(T):
        was_gust = bool(env.gust_left[0] > 0)
        h, mean = pilot(h, obs.to(DEVICE))
        obs, r, info = env.step(mean.cpu())
        tt = round((t + 1) * DT, 3)
        if info["gust"][0] and not was_gust:
            events.append({"t": tt, "kind": "gust"})
        if info["reached"][0]:
            fruit += 1
            events.append({"t": tt, "kind": "fruit", "n": fruit})
        if info["done"][0] and not info["timeout"][0]:
            crashes += 1
            why = "tree" if info["hit_tree"][0] else ("ground" if info["hit_ground"][0] else "out of bounds")
            events.append({"t": tt, "kind": "crash", "why": why, "n": crashes})
            h.zero_()
            world_now = env.trees[0].tolist()
            if world_now != world["trees"]:
                events[-1]["trees"] = world_now
        if t % every == 0:
            s = env.snapshot(0)
            s["t"] = tt
            frames.append(s)
            acts.append(h[0].float().cpu().numpy())
    pilot.reset_manipulations()

    A = np.stack(acts)                                   # frames x neurons, in [0, 1)
    q = np.clip(np.round(A * 255), 0, 255).astype(np.uint8)
    traces = {gr: np.round(A[:, grp == gr].mean(1), 4).tolist() for gr in GROUPS if (grp == gr).any()}
    # bar scale for the viewer: taken from the intact flight so episodes are comparable
    ref_range = ref_range or {gr: [float(np.percentile(v, 2)), float(np.percentile(v, 98)) + 1e-6]
                              for gr, v in traces.items()}
    ep = {"name": name, "title": title, "dt": DT * every, "n_frames": len(frames),
          "n_neurons": int(g.n_nodes), "frames": frames, "events": events, "world": world,
          "traces": traces, "ref_range": ref_range, "mean": np.round(A.mean(0), 3).tolist(),
          "fruit": fruit, "crashes": crashes, "manip": manip or "none",
          "act_b64": base64.b64encode(q.tobytes()).decode()}
    _write_js(f"ep_{name}.js", f"FLY_EP[{json.dumps(name)}]", ep)
    print(f"{name:14s} fruit {fruit:3d} | crashes {crashes:3d} | {len(frames)} frames")
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--T", type=int, default=1200)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    index = []
    g, pilot = make_pilot("real")
    pilot.load_state_dict(torch.load(os.path.join(OUT, "real_s0", "pilot.pt"), map_location=DEVICE))
    pilot.eval()
    write_brain(g)
    ref = record(pilot, g, "intact", "Real connectome", seed=args.seed, T=args.T)["ref_range"]
    record(pilot, g, "no_halteres", "Halteres removed", seed=args.seed, T=args.T,
           manip="no_halteres", ref_range=ref)
    index += [{"name": "intact", "title": "Real connectome"},
              {"name": "no_halteres", "title": "Halteres removed"}]

    rw_ckpt = os.path.join(OUT, "rewired_s0", "pilot.pt")
    if os.path.exists(rw_ckpt):
        g_rw, pilot_rw = make_pilot("rewired")
        pilot_rw.load_state_dict(torch.load(rw_ckpt, map_location=DEVICE))
        pilot_rw.eval()
        record(pilot_rw, g_rw, "rewired", "Rewired control", seed=args.seed, T=args.T, ref_range=ref)
        index.append({"name": "rewired", "title": "Rewired control"})
    write_index(index)
    bundle()


if __name__ == "__main__":
    main()
