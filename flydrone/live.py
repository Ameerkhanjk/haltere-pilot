"""Live simulation: the drone, its world and the fly brain run in real time in Python and
stream to the 3D viewer in your browser. Nothing is pre-recorded and nothing loops: the drone
flies until it crashes, respawns, and carries on.

    python3 -m flydrone.live               # macOS / Linux
    python  -m flydrone.live               # Windows
    ... --no-open                          # don't open a browser
    ... --port 9000                        # use another port

Runs on Apple silicon (MPS), CUDA or plain CPU; one control step of the 9,235-neuron network
costs ~5 ms on CPU, so the 50 Hz loop keeps real time on a laptop without a GPU.

While it runs, the viewer lets you change things live: switch a sense organ off (try the
halteres), kick the drone with a gust, change the wind, swap the real fly brain for the
rewired control. Press Ctrl+C in the Terminal to stop.

Everything stays on your computer: the server listens on 127.0.0.1 only.
"""
import argparse
import base64
import json
import os
import queue
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import torch

from .connectome import PORTS
from .env import DT, DroneEnv
from .record import GROUPS, brain_dict, neuron_groups
from .train import DEVICE, OUT, make_pilot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
SEND_EVERY = 2                    # a frame to the browser every 2 physics steps (25 per second)
NEVER = 10 ** 9                   # no time limit: the flight only ends when the drone crashes

SENSES = [  # id, label shown in the viewer  (ids are the connectome.PORTS names)
    ("haltere", "Halteres · gyroscope"),
    ("jo", "Antenna · wind"),
    ("lptc", "Optic flow"),
    ("ocelli", "Ocelli · horizon"),
    ("lc10", "LC10 · sees the fruit"),
    ("loom", "Looming · trees"),
]
BRAINS = [("real", "Real connectome"), ("rewired", "Rewired control")]
RASTER_QUOTA = {"haltere": 16, "jo": 8, "lptc": 8, "ocelli": 4, "lc10": 10, "loom": 8,
                "descending": 14, "vnc": 10, "wing_mn": 18}


class Sim:
    def __init__(self):
        self.subs, self.subs_lock = [], threading.Lock()
        self.cmds = queue.Queue()
        self.pilots = {}
        for name, _ in BRAINS:
            ckpt = os.path.join(OUT, f"{name}_s0", "pilot.pt")
            if not os.path.exists(ckpt):
                if name == "real":
                    raise SystemExit(f"Missing {ckpt}\nTrain it first:  python3 -m flydrone.train --graph real")
                continue
            g, pilot = make_pilot(name)
            pilot.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            pilot.eval()
            self.pilots[name] = (g, pilot)
        g = self.pilots["real"][0]
        grp = neuron_groups(g)
        self.gidx = {gr: np.where(grp == gr)[0] for gr in GROUPS if (grp == gr).any()}
        self.N = g.n_nodes

        self.brain = "real"
        self.senses = {p: True for p in PORTS}
        self.paused, self.speed = False, 1.0
        self.wind_mag, self.gust_rate = 1.5, 0.006
        self.hello = None

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def calibrate(self, steps=600):
        """Fly the real brain for 12 simulated seconds, unpaced, to learn what 'normal' activity
        looks like: bar scales for each neuron group, per-neuron baselines, and which 96 neurons
        make the most interesting raster."""
        pilot = self.pilots["real"][1]
        pilot.reset_manipulations()
        env = DroneEnv(batch=1, seed=7, max_steps=NEVER)
        obs, h = env.reset(), pilot.init_state(1)
        s1, s2 = np.zeros(self.N), np.zeros(self.N)
        tr = {gr: [] for gr in self.gidx}
        for _ in range(steps):
            h, mean = pilot(h, obs.to(DEVICE))
            obs, _, info = env.step(mean.cpu())
            a = h[0].float().cpu().numpy()
            s1 += a
            s2 += a * a
            for gr, idx in self.gidx.items():
                tr[gr].append(a[idx].mean())
            if info["done"][0]:
                h.zero_()
        mean_act = s1 / steps
        var = s2 / steps - mean_act ** 2
        rows = []
        for gr, k in RASTER_QUOTA.items():
            idx = self.gidx[gr]
            for i in idx[np.argsort(-var[idx])[:k]]:
                rows.append({"i": int(i), "g": gr})
        ref = {gr: [float(np.percentile(v, 2)), float(np.percentile(v, 98)) + 1e-6] for gr, v in tr.items()}
        g = self.pilots["real"][0]
        self.hello = {
            "live": True, "brain": brain_dict(g), "raster": rows, "ref_range": ref,
            "mean": np.round(mean_act, 3).tolist(),
            "senses": [{"id": i, "label": l} for i, l in SENSES],
            "brains": [{"id": i, "label": l} for i, l in BRAINS if i in self.pilots],
        }

    # ------------------------------------------------------------------ clients
    def subscribe(self):
        q = queue.Queue(maxsize=6)
        with self.subs_lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.subs_lock:
            if q in self.subs:
                self.subs.remove(q)

    def broadcast(self, payload):
        with self.subs_lock:
            for q in self.subs:
                if q.full():
                    try:
                        q.get_nowait()          # a slow browser drops old frames, never blocks the sim
                    except queue.Empty:
                        pass
                q.put_nowait(payload)

    # ------------------------------------------------------------------ state + commands
    def state(self):
        return {"brain": self.brain, "senses": self.senses, "paused": self.paused, "speed": self.speed,
                "wind": self.wind_mag, "gust_rate": self.gust_rate}

    def _apply_manipulations(self):
        for _, pilot in self.pilots.values():
            pilot.reset_manipulations()
            for port, on in self.senses.items():
                if not on:
                    pilot.silence_port(port)

    def _apply_wind(self):
        """Keep the wind's random direction from each spawn, but use the wind speed from the slider."""
        mw = self.env.mean_wind[0, :2]
        n = float(mw.norm())
        if n < 1e-6:
            ang = float(torch.rand(1)) * 2 * np.pi
            d = torch.tensor([np.cos(ang), np.sin(ang)], dtype=torch.float32)
        else:
            d = mw / n
        self.env.mean_wind[0, :2] = d * self.wind_mag

    def _respawn(self, new_world=True):
        self.obs = self.env.reset()
        self._apply_wind()
        self.env.wind[0] = self.env.mean_wind[0]
        self.h = self.pilots[self.brain][1].init_state(1)
        self.tv += 1
        self.jump = True

    def _handle(self, cmd):
        kind = cmd.get("cmd")
        if kind == "brain" and cmd.get("value") in self.pilots:
            self.brain = cmd["value"]
            self.h = self.pilots[self.brain][1].init_state(1)     # new brain, fresh neuron state
        elif kind == "sense" and cmd.get("port") in self.senses:
            self.senses[cmd["port"]] = bool(cmd.get("on"))
            self._apply_manipulations()
        elif kind == "gust":
            self.env.trigger_gust(torch.ones(1, dtype=torch.bool))
            self.events.append({"kind": "gust", "manual": True})
        elif kind == "reset":
            self.fruit = self.crashes = self.step_i = 0
            self._respawn()
            self.events.append({"kind": "reset"})
            self.force_frame = True
        elif kind == "pause":
            self.paused = bool(cmd.get("on"))
            self.force_frame = True
        elif kind == "speed" and cmd.get("value") in (0.25, 0.5, 1, 1.0):
            self.speed = float(cmd["value"])
        elif kind == "wind":
            self.wind_mag = float(min(4.0, max(0.0, cmd.get("value", 0))))
            self._apply_wind()
        elif kind == "gust_rate":
            self.gust_rate = float(min(0.1, max(0.0, cmd.get("value", 0))))
            self.env.cfg["gust_prob"] = self.gust_rate

    # ------------------------------------------------------------------ frames
    def _frame(self):
        env = self.env
        a = self.h[0].float().cpu().numpy()
        q = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
        snap = env.snapshot(0)
        frame = {
            "t": round(self.step_i * DT, 3), "pos": snap["pos"], "R": snap["R"], "w": snap["w"],
            "wind": snap["wind"], "target": snap["target"], "gust": snap["gust"], "act": snap["act"],
            "groups": {gr: round(float(a[idx].mean()), 4) for gr, idx in self.gidx.items()},
            "n": base64.b64encode(q.tobytes()).decode(),
            "fruit": self.fruit, "crashes": self.crashes, "events": self.events, "jump": self.jump,
            "tv": self.tv, "trees": env.trees[0].tolist(), "state": self.state(),
        }
        self.events, self.jump, self.force_frame = [], False, False
        self.broadcast(("data: " + json.dumps(frame, separators=(",", ":")) + "\n\n").encode())

    # ------------------------------------------------------------------ the simulation loop
    @torch.no_grad()
    def run(self):
        self.env = DroneEnv(batch=1, seed=int(time.time()) % 100000, max_steps=NEVER)
        self.env.cfg["gust_prob"] = self.gust_rate
        self.tv, self.jump, self.events, self.force_frame = 0, False, [], False
        self.fruit = self.crashes = self.step_i = 0
        self.obs = self.env.reset()
        self._apply_wind()
        self.env.wind[0] = self.env.mean_wind[0]
        self.h = self.pilots[self.brain][1].init_state(1)
        self._apply_manipulations()
        next_t = time.perf_counter()
        while True:
            while True:
                try:
                    self._handle(self.cmds.get_nowait())
                except queue.Empty:
                    break
            if self.force_frame:
                self._frame()
            if self.paused:
                time.sleep(0.03)
                next_t = time.perf_counter()
                continue

            env, pilot = self.env, self.pilots[self.brain][1]
            self.h, mean = pilot(self.h, self.obs.to(DEVICE))
            pos_before = env.pos[0].tolist()
            was_gust = bool(env.gust_left[0] > 0)
            self.obs, _, info = env.step(mean.cpu())
            self.step_i += 1
            if info["gust"][0] and not was_gust:
                self.events.append({"kind": "gust"})
            if info["reached"][0]:
                self.fruit += 1
                self.events.append({"kind": "fruit", "n": self.fruit})
            if info["done"][0]:
                self.crashes += 1
                why = "tree" if info["hit_tree"][0] else ("ground" if info["hit_ground"][0] else "out of bounds")
                self.events.append({"kind": "crash", "why": why, "n": self.crashes, "pos": pos_before})
                self.h.zero_()
                self._apply_wind()
                env.wind[0] = env.mean_wind[0]
                self.tv += 1
                self.jump = True
            if self.step_i % SEND_EVERY == 0:
                self._frame()

            next_t += DT / self.speed                      # keep to real time
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                next_t = time.perf_counter()               # fell behind (slow machine): don't sprint to catch up


# ---------------------------------------------------------------------- web server
class Handler(SimpleHTTPRequestHandler):
    sim = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=VIEWER, **kwargs)

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _local_only(self):
        """Reject requests that aren't addressed to this machine (DNS-rebinding) or that come from
        another website's page."""
        host = (self.headers.get("Host") or "").split(":")[0]
        origin = self.headers.get("Origin")
        ok = host in ("127.0.0.1", "localhost") and (
            not origin or urlparse(origin).netloc == self.headers.get("Host"))
        if not ok:
            self.send_error(403)
        return ok

    def do_GET(self):
        if not self._local_only():
            return
        path = self.path.split("?")[0]
        if path == "/api/hello":
            body = json.dumps({**self.sim.hello, "state": self.sim.state()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            q = self.sim.subscribe()
            try:
                while True:
                    try:
                        chunk = q.get(timeout=1.0)
                    except queue.Empty:
                        chunk = b": keepalive\n\n"
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                self.sim.unsubscribe(q)
        else:
            super().do_GET()

    def do_POST(self):
        if not self._local_only():
            return
        if self.path.split("?")[0] != "/api/cmd":
            self.send_error(404)
            return
        try:
            n = min(int(self.headers.get("Content-Length", 0)), 4096)
            cmd = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(cmd, dict):
                raise ValueError
        except ValueError:
            self.send_error(400)
            return
        self.sim.cmds.put(cmd)
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description="Live fly-brain drone simulation")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser automatically")
    args = ap.parse_args()

    print("Loading the fly brain ...", flush=True)
    sim = Sim()
    print("Warming up (about 5 seconds) ...", flush=True)
    sim.calibrate()
    Handler.sim = sim
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        raise SystemExit(f"Port {args.port} is already in use. Is the simulation already running?\n"
                         f"Close the other Terminal window, or use:  python3 -m flydrone.live --port {args.port + 1}")
    threading.Thread(target=sim.run, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nThe live simulation is running at {url}\n"
          f"Leave this window open. Press Ctrl+C here to stop.\n", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
