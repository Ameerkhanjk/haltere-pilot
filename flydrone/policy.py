"""The fly's brain as the drone's pilot.

Each neuron is a leaky, rectified rate unit driven by its real presynaptic partners:

    r_t      = tanh(relu(gain * sum_j w_ij r_j + b_i + sensory drive_i))
    h_{t+1}  = (1 - a_i) h_t + a_i r_t

w_ij starts at the connectome's synapse count (log-scaled, normalised per neuron) with the
sign of the presynaptic neuron's predicted neurotransmitter. Training learns a multiplicative
gain per synapse (how far RL moves each synapse from biology), a bias and a time constant
per neuron, how each port neuron is tuned to its sensor, and a linear readout from the wing
motor neurons to the drone's four commands. It cannot add connections that don't exist.

Sensor channels only reach the port neurons of their own modality (the input weight matrix is
block-masked), so the gyroscope can only reach the brain through the haltere afferents.
"""
import numpy as np
import torch
import torch.nn as nn

from .connectome import PORTS
from .env import CHANNELS

ACTIONS = ["thrust", "roll", "pitch", "yaw"]


class ConnectomePilot(nn.Module):
    def __init__(self, graph, ticks=3, gain=1.5, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.N, self.E, self.ticks, self.gain = graph.n_nodes, graph.n_edges, ticks, gain
        # edges sorted by destination: the scatter below is then memory-local, ~3x faster
        order = np.argsort(graph.dst, kind="stable")
        self.edge_order = order
        self.register_buffer("src", torch.as_tensor(graph.src[order], dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(graph.dst[order], dtype=torch.long))
        self.register_buffer("w_base", torch.as_tensor((graph.normalized_base() * graph.sign)[order]))
        self.w_gain = nn.Parameter(torch.ones(self.E))
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.alpha_raw = nn.Parameter(torch.zeros(self.N))

        # ---- sensory ports: channel block c -> only that modality's neurons
        n_obs = sum(CHANNELS.values())
        in_idx, W, M = [], [], []
        c0 = 0
        for p in PORTS:
            idx = graph.port(p)
            d = CHANNELS[p]
            w = np.zeros((len(idx), n_obs), np.float32)
            w[:, c0:c0 + d] = rng.normal(0, 2.0 / np.sqrt(d), (len(idx), d))
            side = graph.side[idx]
            # bilateral structure: left-eye neurons prefer things on the left (+y), and the
            # left/right HS cells have opposite yaw preference
            lr = np.where(side == "L", 1.0, np.where(side == "R", -1.0, 0.0))
            if p in ("lc10", "loom"):
                w[:, c0 + 1] += 1.5 * lr
            if p == "lptc":
                w[:, c0 + 2] += 1.5 * lr
            m = np.zeros_like(w)
            m[:, c0:c0 + d] = 1
            in_idx.append(idx); W.append(w); M.append(m)
            c0 += d
        self.register_buffer("in_idx", torch.as_tensor(np.concatenate(in_idx), dtype=torch.long))
        self.W_in = nn.Parameter(torch.as_tensor(np.concatenate(W)))
        self.register_buffer("in_mask", torch.as_tensor(np.concatenate(M)))
        self.port_slices = {}
        s = 0
        for p, idx in zip(PORTS, in_idx):
            self.port_slices[p] = (s, s + len(idx))
            s += len(idx)

        # ---- motor readout from wing motor neurons, initialised with textbook muscle roles
        self.register_buffer("out_idx", torch.as_tensor(graph.out_idx, dtype=torch.long))
        grp, side = graph.out_group, graph.side[graph.out_idx]
        lr = np.where(side == "L", 1.0, np.where(side == "R", -1.0, 0.0))
        R = rng.normal(0, 0.02, (4, len(grp))).astype(np.float32)
        norm = lambda m: m / max(m.sum(), 1)
        R[0] += 2.0 * norm(grp == "power")                       # thrust: power muscles
        R[1] += 2.0 * norm(grp == "b") * lr * 2                  # roll: basalare L-R
        R[2] += 2.0 * (norm(grp == "iii") - norm(grp == "b"))    # pitch: iii vs basalare
        R[3] += 2.0 * norm(grp == "i") * lr * 2                  # yaw: first axillary L-R
        self.readout = nn.Linear(len(grp), 4)
        with torch.no_grad():
            self.readout.weight.copy_(torch.as_tensor(R))
            self.readout.bias.zero_()
        self.log_std = nn.Parameter(torch.full((4,), -0.7))

        self.register_buffer("lesion_mask", torch.ones(self.N))
        self.register_buffer("port_mask", torch.ones(len(self.in_idx)))

    def init_state(self, B):
        return torch.zeros(B, self.N, device=self.w_base.device)

    def forward(self, h, obs):
        """h: (B, N) neuron state, obs: (B, n_obs). Runs `ticks` synaptic updates per
        20 ms control step (~7 ms per hop). Internally neuron-major (N, B) so each synapse
        gathers one contiguous row."""
        hT = h.T
        drive = ((self.W_in * self.in_mask) @ obs.T) * self.port_mask[:, None]
        w = (self.w_base * self.w_gain)[:, None]
        alpha = torch.sigmoid(self.alpha_raw)[:, None]
        bias = self.bias[:, None]
        lesion = self.lesion_mask[:, None]
        for _ in range(self.ticks):
            syn = torch.zeros_like(hT).index_add_(0, self.dst, hT.index_select(0, self.src) * w)
            pre = (syn * self.gain + bias).index_add(0, self.in_idx, drive)
            hT = ((1 - alpha) * hT + alpha * torch.tanh(torch.relu(pre))) * lesion
        mean = self.readout(hT.index_select(0, self.out_idx).T)
        return hT.T, mean

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())

    # ---- manipulations used by the analysis
    def silence_port(self, name, on=True):
        a, b = self.port_slices[name]
        self.port_mask[a:b] = 0.0 if on else 1.0

    def reset_manipulations(self):
        self.lesion_mask.fill_(1.0)
        self.port_mask.fill_(1.0)


class Critic(nn.Module):
    """Value function on the privileged simulator state. Only used to train; not part of the fly."""

    def __init__(self, n_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)
