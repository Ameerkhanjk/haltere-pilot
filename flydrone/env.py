"""Batched quadcopter world, simulated in torch so thousands of steps/s run on the GPU.

The fly sits on the drone. Its senses are wired to the drone's sensors:

    haltere afferents  <- gyroscope (body angular velocity)            no delay
    Johnston's organ   <- airspeed + specific force (antenna bend)     no delay
    ocelli             <- horizon (roll / pitch from the sky)          OCELLI_DELAY steps
    optic-flow cells   <- rotation + translation optic flow            VISUAL_DELAY steps
    LC10               <- where the target fruit is in view             VISUAL_DELAY steps
    looming detectors  <- nearest tree approaching                      VISUAL_DELAY steps

The delays are the point: in a real fly, vision is slow (tens of ms through the optic lobe)
while the haltere pathway is one of the fastest reflexes in the animal. Gusts torque the drone
faster than vision can report it, so a controller that ignores the gyro should struggle.
Note the antenna, like any accelerometer, feels *specific force* (thrust + drag), not gravity:
in flight it says little about tilt, so the only instant rotation sense is the haltere.

Body frame is FLU: x forward, y left, z up. World frame is ENU with gravity along -z.
The action is [collective thrust, roll, pitch, yaw torque], each squashed to [-1, 1].
"""
import math
import torch

G = 9.81
DT = 0.02                 # 50 Hz control
SUBSTEPS = 2
VISUAL_DELAY = 3          # 60 ms  optic lobe -> LPTC / LC
OCELLI_DELAY = 2          # 40 ms  ocellar pathway is faster but noisy
OCELLI_NOISE = 0.08
OBS_NOISE = 0.02

# observation layout, in the same order as connectome.PORTS
CHANNELS = {"haltere": 3, "jo": 6, "lptc": 6, "ocelli": 2, "lc10": 4, "loom": 4}
N_OBS = sum(CHANNELS.values())
N_ACT = 4
N_PRIV = 26

DEFAULTS = dict(
    arena=10.0, ceiling=8.0, n_trees=10, tree_radius=(0.25, 0.6),
    # fast rotational dynamics, like a real fly's: small inertia, little passive damping,
    # sharp gusts. Tuned so a hand-written autopilot loses most of its performance when it
    # has to rely on delayed vision instead of a gyro (see flydrone/baselines.py).
    thrust_range=0.8, torque_max=(120.0, 120.0, 30.0), rot_damping=0.5, drag=0.35,
    mean_wind=1.5, turbulence=0.8, gust_prob=0.006, gust_torque=(50.0, 110.0),
    gust_steps=6, torque_noise=15.0,
    target_dist=(3.0, 6.0), reach_radius=0.7, max_steps=1500,
    progress_w=4.0, reach_bonus=5.0, crash_penalty=5.0, alive_bonus=0.02,
    rate_cost=0.002, action_cost=0.01,
    # ablations / manipulations (used by analysis, never during training)
    gyro_gain=1.0, vision_gain=1.0,
)


def _skew(w):
    z = torch.zeros_like(w[:, 0])
    return torch.stack([torch.stack([z, -w[:, 2], w[:, 1]], -1),
                        torch.stack([w[:, 2], z, -w[:, 0]], -1),
                        torch.stack([-w[:, 1], w[:, 0], z], -1)], 1)


def _rodrigues(w, dt):
    """Rotation matrix for rotating by body rate w for time dt."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9) * dt
    k = _skew(w / (w.norm(dim=-1, keepdim=True).clamp_min(1e-9)))
    eye = torch.eye(3, device=w.device).expand_as(k)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return eye + s * k + (1 - c) * (k @ k)


def _orthonormalize(R):
    x, y = R[:, :, 0], R[:, :, 1]
    x = x / x.norm(dim=-1, keepdim=True)
    y = y - (x * y).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def _rot_z(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1),
                        torch.stack([z, z, o], -1)], 1)


class DroneEnv:
    def __init__(self, batch=64, device="cpu", seed=0, **cfg):
        self.B, self.device = batch, device
        self.cfg = {**DEFAULTS, **cfg}
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.torque_max = torch.tensor(self.cfg["torque_max"], device=device)

    # ------------------------------------------------------------------ helpers
    def _u(self, *shape, lo=0.0, hi=1.0):
        return (torch.rand(*shape, generator=self.g) * (hi - lo) + lo).to(self.device)

    def _n(self, *shape):
        return torch.randn(*shape, generator=self.g).to(self.device)

    def _sample_free(self, m, around=None):
        """Sample xy positions for the envs in mask m: inside the arena, clear of trees."""
        c = self.cfg
        n, L = int(m.sum()), c["arena"] - 1.0
        trees = self.trees[m]
        out = ok = None
        for _ in range(12):
            if around is None:
                xy = self._u(n, 2, lo=-L, hi=L)
            else:
                ang = self._u(n, lo=0, hi=2 * math.pi)
                r = self._u(n, lo=c["target_dist"][0], hi=c["target_dist"][1])
                xy = (around[:, :2] + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * r[:, None]).clamp(-L, L)
            good = ((xy[:, None, :] - trees[:, :, :2]).norm(dim=-1) - trees[:, :, 2] > 1.0).all(-1)
            if out is None:
                out, ok = xy, good
            else:
                out = torch.where((good & ~ok)[:, None], xy, out)
                ok = ok | good
            if ok.all():
                break
        return out

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.pos = torch.zeros(self.B, 3, device=self.device)
        self.vel = torch.zeros_like(self.pos)
        self.R = torch.eye(3, device=self.device).repeat(self.B, 1, 1)
        self.w = torch.zeros_like(self.pos)
        self.wind = torch.zeros_like(self.pos)
        self.mean_wind = torch.zeros_like(self.pos)
        self.dist_torque = torch.zeros_like(self.pos)
        self.gust = torch.zeros_like(self.pos)
        self.gust_left = torch.zeros(self.B, device=self.device)
        self.trees = torch.zeros(self.B, self.cfg["n_trees"], 3, device=self.device)
        self.target = torch.zeros_like(self.pos)
        self.t = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.last_act = torch.zeros(self.B, N_ACT, device=self.device)
        self.acc = torch.zeros_like(self.pos)
        self._vis_hist = None
        self._oc_hist = None
        self._reset_envs(torch.ones(self.B, dtype=torch.bool, device=self.device))
        return self._obs()

    def _reset_envs(self, m):
        n = int(m.sum())
        if n == 0:
            return
        c = self.cfg
        L = c["arena"]
        trees = torch.cat([self._u(n, c["n_trees"], 2, lo=-L, hi=L),
                           self._u(n, c["n_trees"], 1, lo=c["tree_radius"][0], hi=c["tree_radius"][1])], -1)
        self.trees[m] = trees
        start = self._sample_free(m)
        z0 = self._u(n, lo=1.5, hi=3.5)
        self.pos[m] = torch.cat([start, z0[:, None]], -1)
        self.vel[m] = self._n(n, 3) * 0.3
        tilt = _rodrigues(self._n(n, 3) * torch.tensor([0.15, 0.15, 0.0], device=self.device), 1.0)
        self.R[m] = _rot_z(self._u(n, lo=-math.pi, hi=math.pi)) @ tilt
        self.w[m] = self._n(n, 3) * 0.3
        ang = self._u(n, lo=0, hi=2 * math.pi)
        mag = self._u(n, lo=0, hi=c["mean_wind"])
        self.mean_wind[m] = torch.stack([torch.cos(ang) * mag, torch.sin(ang) * mag,
                                         torch.zeros_like(mag)], -1)
        self.wind[m] = self.mean_wind[m]
        self.dist_torque[m] = 0
        self.gust[m] = 0
        self.gust_left[m] = 0
        self.t[m] = 0
        self.last_act[m] = 0
        self._new_target(m)
        self.prev_dist = (self.target - self.pos).norm(dim=-1)

    def _new_target(self, m):
        n = int(m.sum())
        if n == 0:
            return
        xy = self._sample_free(m, around=self.pos[m])
        z = self._u(n, lo=1.2, hi=4.5)
        self.target[m] = torch.cat([xy, z[:, None]], -1)

    # ------------------------------------------------------------------ sensing
    def _nearest_loom(self):
        if self.trees.shape[1] == 0:
            z = torch.zeros(self.B, device=self.device)
            return torch.zeros(self.B, 3, device=self.device), z, z + 99
        rel = self.trees[:, :, :2] - self.pos[:, None, :2]
        dist = (rel.norm(dim=-1) - self.trees[:, :, 2]).clamp_min(0.15)
        dirn = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        approach = (self.vel[:, None, :2] * dirn).sum(-1)          # m/s toward the tree
        loom = (approach / dist).clamp(0, 6) / 6 + (0.6 / dist).clamp(0, 1) * 0.3
        k = loom.argmax(-1)
        idx = torch.arange(self.B, device=self.device)
        d3 = torch.cat([dirn[idx, k], torch.zeros(self.B, 1, device=self.device)], -1)
        return d3, loom[idx, k], dist[idx, k]

    def _obs(self):
        c = self.cfg
        Rt = self.R.transpose(1, 2)
        g_body = -Rt[:, :, 2]                     # gravity direction in the body frame
        f_spec = (Rt @ (self.acc + torch.tensor([0, 0, G], device=self.device))[..., None]).squeeze(-1) / G
        v_body = (Rt @ self.vel[..., None]).squeeze(-1)
        vair_body = (Rt @ (self.vel - self.wind)[..., None]).squeeze(-1)

        rel = self.target - self.pos
        d = rel.norm(dim=-1, keepdim=True)
        tdir = (Rt @ (rel / d.clamp_min(1e-6))[..., None]).squeeze(-1)
        close = torch.exp(-d / 5.0)
        in_view = (tdir[:, :1] > -0.3).float() * 0.7 + 0.3   # dimmer when behind the fly
        ldir, loom, _ = self._nearest_loom()
        ldir = (Rt @ ldir[..., None]).squeeze(-1)

        vis_now = torch.cat([self.w / 6.0, v_body / 5.0,               # lptc
                             tdir * in_view, close,                    # lc10
                             ldir * loom[:, None], loom[:, None]], -1)  # loom
        if self._vis_hist is None:
            self._vis_hist = [vis_now] * (VISUAL_DELAY + 1)
        self._vis_hist = self._vis_hist[1:] + [vis_now]
        vis = self._vis_hist[0] * c["vision_gain"]
        horizon = -g_body[:, :2] + OCELLI_NOISE * self._n(self.B, 2)
        if self._oc_hist is None:
            self._oc_hist = [horizon] * (OCELLI_DELAY + 1)
        self._oc_hist = self._oc_hist[1:] + [horizon]

        obs = torch.cat([
            self.w / 6.0 * c["gyro_gain"],                                # haltere
            torch.cat([vair_body / 5.0, f_spec - torch.tensor([0, 0, 1.0], device=self.device)], -1),  # jo
            vis[:, 0:6],                                                  # lptc
            self._oc_hist[0] * c["vision_gain"],                          # ocelli (horizon)
            vis[:, 6:10],                                                 # lc10
            vis[:, 10:14],                                                # loom
        ], -1)
        obs = obs + OBS_NOISE * self._n(*obs.shape)
        return obs

    def privileged(self):
        """Full state for the critic only -- the fly's brain never sees this."""
        ldir, loom, ldist = self._nearest_loom()
        return torch.cat([
            (self.target - self.pos) / 5.0, self.vel / 5.0, self.R.reshape(self.B, 9),
            self.w / 6.0, self.wind / 3.0, self.pos[:, 2:3] / 4.0,
            ldir[:, :2] * loom[:, None], (ldist[:, None] / 3.0).clamp(max=2),
            (self.t[:, None].float() / self.cfg["max_steps"]),
        ], -1)

    # ------------------------------------------------------------------ dynamics
    def trigger_gust(self, mask):
        """Start a gust now on the envs in `mask`: a sharp torque kick in a random direction,
        plus a shove of the wind. (Also used by the live viewer's 'kick' button.)"""
        c = self.cfg
        n = int(mask.sum())
        if n == 0:
            return
        dirn = self._n(n, 3) * torch.tensor([1.0, 1.0, 0.4], device=self.device)
        dirn = dirn / dirn.norm(dim=-1, keepdim=True)
        self.gust[mask] = dirn * self._u(n, 1, lo=c["gust_torque"][0], hi=c["gust_torque"][1])
        self.gust_left[mask] = c["gust_steps"]
        self.wind[mask] = self.wind[mask] + self._n(n, 3) * torch.tensor([2.0, 2.0, 0.5], device=self.device)

    def step(self, action):
        c = self.cfg
        a = torch.tanh(action)
        self.last_act = a
        thrust = G * (1.0 + c["thrust_range"] * a[:, 0])
        torque_cmd = a[:, 1:] * self.torque_max

        # turbulence (Ornstein-Uhlenbeck) and discrete gusts that kick the attitude
        sq = math.sqrt(DT)
        self.wind = self.wind + 1.0 * (self.mean_wind - self.wind) * DT + c["turbulence"] * sq * self._n(self.B, 3) * torch.tensor([1, 1, 0.3], device=self.device)
        self.dist_torque = self.dist_torque - 2.0 * self.dist_torque * DT + c["torque_noise"] * 2 * sq * self._n(self.B, 3)
        start = (self._u(self.B) < c["gust_prob"]) & (self.gust_left <= 0)
        if start.any():
            self.trigger_gust(start)
        gust_on = (self.gust_left > 0).float()[:, None]
        self.gust_left = self.gust_left - 1

        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            dw = torque_cmd + self.dist_torque + self.gust * gust_on - c["rot_damping"] * self.w
            self.w = self.w + dw * h
            self.R = self.R @ _rodrigues(self.w, h)
            acc = self.R[:, :, 2] * thrust[:, None] - c["drag"] * (self.vel - self.wind)
            acc = acc + torch.tensor([0, 0, -G], device=self.device)
            self.vel = self.vel + acc * h
            self.acc = acc
            self.pos = self.pos + self.vel * h
        self.R = _orthonormalize(self.R)
        self.t = self.t + 1

        # ---- events
        L = c["arena"]
        horiz = (self.pos[:, None, :2] - self.trees[:, :, :2]).norm(dim=-1) - self.trees[:, :, 2]
        hit_tree = (horiz < 0.25).any(-1)
        out = (self.pos[:, :2].abs() > L + 1).any(-1) | (self.pos[:, 2] > c["ceiling"])
        hit_ground = self.pos[:, 2] < 0.05
        crashed = hit_tree | out | hit_ground
        timeout = self.t >= c["max_steps"]

        dist = (self.target - self.pos).norm(dim=-1)
        reached = (dist < c["reach_radius"]) & ~crashed

        reward = (c["progress_w"] * (self.prev_dist - dist)
                  + c["alive_bonus"]
                  - c["rate_cost"] * (self.w ** 2).sum(-1).clamp(max=100)
                  - c["action_cost"] * (a[:, 1:] ** 2).sum(-1)
                  + c["reach_bonus"] * reached.float()
                  - c["crash_penalty"] * crashed.float())

        info = {"crashed": crashed, "reached": reached, "timeout": timeout,
                "hit_tree": hit_tree, "hit_ground": hit_ground, "dist": dist,
                "gust": gust_on[:, 0] > 0, "tilt": torch.acos(self.R[:, 2, 2].clamp(-1, 1))}

        if reached.any():
            self._new_target(reached)
        self.prev_dist = (self.target - self.pos).norm(dim=-1)
        done = crashed | timeout
        if done.any():
            self._reset_envs(done)
            if self._vis_hist is not None:   # the new episode starts with a fresh visual history
                self._vis_hist = [torch.where(done[:, None], torch.zeros_like(v), v) for v in self._vis_hist]
                self._oc_hist = [torch.where(done[:, None], torch.zeros_like(v), v) for v in self._oc_hist]
            self.acc[done] = 0
        info["done"] = done
        return self._obs(), reward, info

    # ------------------------------------------------------------------ for rendering
    def snapshot(self, i=0):
        return {"pos": self.pos[i].tolist(), "R": self.R[i].reshape(-1).tolist(),
                "vel": self.vel[i].tolist(), "w": self.w[i].tolist(),
                "wind": self.wind[i].tolist(), "target": self.target[i].tolist(),
                "gust": bool(self.gust_left[i] > 0), "act": self.last_act[i].tolist()}


# ---------------------------------------------------------------------- reference pilots
def autopilot(env, use_gyro=True, k_att=12.0, k_rate=3.0):
    """A hand-written cascaded controller with perfect state (position loop -> attitude loop
    -> rate loop). Upper bound on what's achievable, and a check the task is flyable.
    With use_gyro=False it only gets what a fly without halteres has: attitude from the ocelli
    (OCELLI_DELAY) and rotation from optic flow (VISUAL_DELAY)."""
    hist, R_hist = [], []

    def ctrl(_obs):
        rel = env.target - env.pos
        dist = rel.norm(dim=-1, keepdim=True)
        v_des = rel / dist.clamp_min(1e-6) * dist.clamp(max=3.0) * 0.9
        acc_des = 2.0 * (v_des - env.vel) + torch.tensor([0, 0, G], device=env.device)
        acc_des[:, :2] = acc_des[:, :2].clamp(-6, 6)
        z_des = acc_des / acc_des.norm(dim=-1, keepdim=True)
        R_hist.append(env.R.clone())
        R = env.R if use_gyro else R_hist[max(0, len(R_hist) - 1 - OCELLI_DELAY)]
        b3 = R[:, :, 2]
        thrust = (acc_des * b3).sum(-1)
        e_world = torch.cross(b3, z_des, dim=-1)          # rotate b3 toward z_des
        e_body = (R.transpose(1, 2) @ e_world[..., None]).squeeze(-1)
        w_meas = env.w.clone()
        hist.append(w_meas)
        del hist[:-8], R_hist[:-8]
        if not use_gyro:
            w_meas = hist[max(0, len(hist) - 1 - VISUAL_DELAY)]
        torque = k_att * e_body - k_rate * w_meas
        torque[:, 2] = -k_rate * 0.5 * w_meas[:, 2]
        a_thr = ((thrust / G - 1.0) / env.cfg["thrust_range"]).clamp(-0.999, 0.999)
        a_tq = (torque / env.torque_max).clamp(-0.999, 0.999)
        return torch.atanh(torch.cat([a_thr[:, None], a_tq], -1))
    return ctrl
