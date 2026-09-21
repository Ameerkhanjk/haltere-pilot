"""Build the flight-control subnetwork of the male CNS connectome.

Instead of dumping observations into generic "sensory" neurons and reading actions from
generic "descending" neurons, every drone sensor enters the network at the neuron class a
real fly uses for that signal, and actions are read from the real wing motor neurons:

    drone sensor              fly neurons it enters through
    ------------------------  ------------------------------------------------------------
    gyroscope (body rates)    haltere afferents (SApp, SNpp*)      -- the fly's gyroscope
    airspeed + gravity        Johnston's organ wind/gravity neurons (JO-C/E)
    optic flow                lobula plate tangential cells (HS, VS, H2)
    horizon / attitude        ocellar projection neurons (OCG01-03)
    target in view            LC10 object-tracking projection neurons
    obstacle approaching      looming detectors (LPLC2, LC4)

    actions  <-  wing motor neurons (DLM/DVM power muscles, b1-3, i1-2, iii1/3, hg1-4 ...)

The optic lobe's front end (photoreceptors -> T4/T5 motion detectors) is replaced by the
computation it is known to perform: optic flow is injected at its *output* neurons. Everything
downstream of the ports is the real wiring.

The subnetwork keeps every neuron that lies on a short path (<= K synaptic hops) from any
input port to a wing motor neuron. K=3 gives ~9k neurons / 333k connections: small enough
to train on a laptop GPU, and every neuron is within 3 hops of a wing motor neuron, so with
3 synaptic ticks per control step every neuron can influence the very next wingbeat command.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# connectome_cache/ at the repo root; connectome/connectome_cache/ is also accepted
CACHE_DIR = next((d for d in (os.path.join(ROOT, "connectome_cache"),
                              os.path.join(ROOT, "connectome", "connectome_cache")) if os.path.isdir(d)),
                 os.path.join(ROOT, "connectome_cache"))
RAW_CACHE = os.path.join(CACHE_DIR, "male-cns_v1_0_w5.npz")
SOMA_CACHE = os.path.join(CACHE_DIR, "male-cns_v1_0_soma.npz")

MIN_SYNAPSES = 5
EXCITATORY = {"acetylcholine"}
INHIBITORY = {"gaba", "glutamate"}   # glutamate is mostly inhibitory (GluCl) in the fly CNS

# port name -> (column, regex). Order matters: it is the order of the input channel blocks.
PORTS = {
    "haltere": ("subclass", r"^haltere$"),
    "jo":      ("subclass", r"^wind_gravity$"),
    "lptc":    ("type", r"^(?:HS[ENST]|VS|VSm|VST[12]|H2)$"),
    "ocelli":  ("type", r"^OCG0[1-3]"),
    "lc10":    ("type", r"^LC10"),
    "loom":    ("type", r"^(?:LPLC2|LC4)$"),
}
PORT_LABELS = {
    "haltere": "Haltere afferents",
    "jo": "Johnston's organ (wind/gravity)",
    "lptc": "Optic-flow cells (HS/VS/H2)",
    "ocelli": "Ocellar neurons",
    "lc10": "LC10 object tracking",
    "loom": "Looming detectors",
}

# Wing motor neurons grouped by the muscle they drive. The functional roles follow the
# fly flight literature (Dickinson & Muijres 2016; Lindsay et al. 2017) and are used only to
# initialise the readout -- training is free to change them.
WING_MN_GROUPS = {
    "power":   r"^(?:DLMn|DVMn)",          # asynchronous power muscles -> overall lift
    "b":       r"^b[123] MN$",           # basalare: stroke amplitude  (bilateral diff -> roll/yaw)
    "i":       r"^i[12] MN$",            # first axillary: stroke deviation (yaw)
    "iii":     r"^iii[13] MN$",          # third axillary: stroke retraction (pitch / turn)
    "hg":      r"^hg[1-4] MN$",          # fourth axillary
    "tp_ps":   r"^(?:tp[12n]|ps[12]) MN$", # tension / pleurosternal
    "other":   r"^(?:MNwm|STTMm|TTMn)",
}


def _load_raw():
    z = np.load(RAW_CACHE, allow_pickle=True)
    neurons = pd.DataFrame(z["neurons"].item())
    conns = pd.DataFrame(z["conns"].item())
    return neurons, conns


def _match(df, col, pattern):
    return df[col].astype(str).str.contains(pattern, regex=True, na=False).values


def _hops(adj, seeds, max_k):
    """Multi-source BFS hop count over a CSR adjacency."""
    dist = np.full(adj.shape[0], np.inf)
    frontier = np.where(seeds)[0]
    dist[frontier] = 0
    for k in range(1, max_k + 1):
        if len(frontier) == 0:
            break
        nb = np.unique(adj[frontier].indices)
        nb = nb[np.isinf(dist[nb])]
        dist[nb] = k
        frontier = nb
    return dist


class FlightGraph:
    """Arrays describing the chosen subnetwork. Everything is indexed 0..n_nodes-1."""

    FIELDS = ["src", "dst", "weight", "sign", "body_id", "type", "superclass", "side",
              "neuropil", "xyz", "hops_in", "hops_out", "out_idx", "out_group",
              "port_idx", "port_of"]

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.n_nodes = len(self.body_id)

    @property
    def n_edges(self):
        return len(self.src)

    def port(self, name):
        return self.port_idx[self.port_of == name]

    def normalized_base(self):
        """Per-destination normalisation (same as the notebook): each neuron's summed input
        weight is ~1 regardless of how many partners it has."""
        w = np.log1p(self.weight)
        tot = np.zeros(self.n_nodes)
        np.add.at(tot, self.dst, w)
        return (w / np.maximum(tot, 1e-6)[self.dst]).astype(np.float32)

    def summary(self):
        ports = ", ".join(f"{p} {int((self.port_of == p).sum())}" for p in PORTS)
        return (f"{self.n_nodes:,} neurons | {self.n_edges:,} connections | "
                f"{100 * (self.sign < 0).mean():.1f}% inhibitory | "
                f"inputs: {ports} | outputs: {len(self.out_idx)} wing motor neurons")

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.FIELDS})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(**{k: z[k] for k in cls.FIELDS})

    def rewired(self, seed=0):
        """Configuration-model control: every neuron keeps its out-degree, synapse counts and
        sign; the in-degree distribution is preserved as a whole; *who talks to whom* is
        shuffled. Ports and motor neurons stay where they are."""
        rng = np.random.default_rng(seed)
        new_dst = self.dst.copy()
        rng.shuffle(new_dst)
        keep = new_dst != self.src
        kw = {k: getattr(self, k) for k in self.FIELDS}
        kw.update(src=self.src[keep], dst=new_dst[keep], weight=self.weight[keep],
                  sign=self.sign[keep])
        return FlightGraph(**kw)


def build_flight_graph(max_hops=3, max_neurons=20_000, max_edges=800_000, verbose=True):
    cache = os.path.join(CACHE_DIR, f"flight_graph_k{max_hops}_n{max_neurons}_e{max_edges}.npz")
    if os.path.exists(cache):
        g = FlightGraph.load(cache)
        if verbose:
            print("loaded", os.path.basename(cache), "|", g.summary())
        return g

    log = print if verbose else (lambda *a, **k: None)
    neurons, conns = _load_raw()
    neurons = neurons.drop_duplicates("bodyId").reset_index(drop=True)
    agg = conns.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    agg = agg[(agg.weight >= MIN_SYNAPSES) & (agg.bodyId_pre != agg.bodyId_post)]
    ids = pd.Index(neurons.bodyId.values)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    agg = agg[agg.bodyId_pre.isin(ids) & agg.bodyId_post.isin(ids)]
    s, d = pos[agg.bodyId_pre].values, pos[agg.bodyId_post].values
    w = agg.weight.values.astype(np.float32)
    N = len(ids)
    A = sp.csr_matrix((w, (s, d)), shape=(N, N))
    log(f"full connectome: {N:,} neurons, {len(s):,} connections (>= {MIN_SYNAPSES} synapses)")

    # ---- ports and outputs -------------------------------------------------------------
    port_mask = {p: _match(neurons, col, pat) for p, (col, pat) in PORTS.items()}
    is_wing_mn = (neurons.superclass.astype(str).eq("vnc_motor").values
                  & neurons.subclass.astype(str).eq("wm").values)
    any_port = np.zeros(N, bool)
    for m in port_mask.values():
        any_port |= m

    hops_in = _hops(A, any_port, max_hops)
    hops_out = _hops(A.T.tocsr(), is_wing_mn, max_hops)
    on_path = (hops_in + hops_out) <= max_hops
    keep = on_path | any_port | is_wing_mn
    log(f"neurons on a <= {max_hops}-hop sensor -> wing-motor path: {int(keep.sum()):,}")

    if keep.sum() > max_neurons:
        # shortest paths first, then the most strongly connected within the candidate set
        cand = np.where(keep)[0]
        sub = A[cand][:, cand]
        strength = np.asarray(sub.sum(0)).ravel() + np.asarray(sub.sum(1)).ravel()
        must = any_port[cand] | is_wing_mn[cand]
        plen = np.nan_to_num(hops_in[cand] + hops_out[cand], posinf=99)
        order = np.lexsort((-strength, plen, ~must))
        keep = np.zeros(N, bool)
        keep[cand[order[:max_neurons]]] = True
        log(f"  capped to {max_neurons:,} (shortest paths first, then strongest)")

    kept = np.where(keep)[0]
    sub = A[kept][:, kept].tocoo()
    src, dst, wt = sub.row, sub.col, sub.data.astype(np.float32)
    if len(src) > max_edges:
        top = np.argsort(-wt)[:max_edges]
        src, dst, wt = src[top], dst[top], wt[top]
        log(f"  kept the {max_edges:,} strongest connections")

    meta = neurons.iloc[kept].reset_index(drop=True)
    n = len(kept)

    # ---- signs from predicted neurotransmitter -----------------------------------------
    nt = meta.consensusNt.astype(str).str.lower()
    nt = nt.where(~nt.isin(["none", "nan", "unclear", ""]), meta.predictedNt.astype(str).str.lower())
    node_sign = np.where(nt.isin(INHIBITORY), -1.0, 1.0).astype(np.float32)

    # ---- dominant input neuropil ------------------------------------------------------
    body = meta.bodyId.values
    cin = conns[conns.bodyId_post.isin(body)]
    top_roi = (cin.groupby(["bodyId_post", "roi"])["weight"].sum().reset_index()
                  .sort_values("weight", ascending=False).drop_duplicates("bodyId_post")
                  .set_index("bodyId_post")["roi"])
    cout = conns[conns.bodyId_pre.isin(body)]
    top_roi_out = (cout.groupby(["bodyId_pre", "roi"])["weight"].sum().reset_index()
                      .sort_values("weight", ascending=False).drop_duplicates("bodyId_pre")
                      .set_index("bodyId_pre")["roi"])
    neuropil = top_roi.reindex(body).fillna(top_roi_out.reindex(body)).fillna("unknown")
    neuropil = neuropil.values.astype(str)

    # ---- 3D positions: soma if it has one, else the mean position of its partners -----
    zs = np.load(SOMA_CACHE)
    soma = pd.DataFrame(zs["xyz"], index=zs["id"]).reindex(body).values
    xyz = soma.copy()
    und = sp.csr_matrix((wt, (src, dst)), shape=(n, n))
    und = (und + und.T).tocsr()
    for _ in range(4):
        missing = np.isnan(xyz[:, 0])
        if not missing.any():
            break
        have = ~np.isnan(xyz[:, 0])
        M = und[missing][:, have]
        tot = np.asarray(M.sum(1)).ravel()
        est = (M @ xyz[have]) / np.maximum(tot, 1e-9)[:, None]
        est[tot == 0] = np.nan
        xyz[missing] = est
    xyz = np.nan_to_num(xyz, nan=np.nanmean(xyz, 0)[0]).astype(np.float32)

    # ---- ports, outputs ---------------------------------------------------------------
    port_idx, port_of = [], []
    for p in PORTS:
        idx = np.where(port_mask[p][kept])[0]
        port_idx.append(idx)
        port_of += [p] * len(idx)
    port_idx = np.concatenate(port_idx).astype(np.int64)
    port_of = np.array(port_of)

    out_idx = np.where(is_wing_mn[kept])[0].astype(np.int64)
    types = meta.type.astype(str).values
    out_group = np.array(["other"] * len(out_idx), dtype=object)
    for g, pat in WING_MN_GROUPS.items():
        hit = pd.Series(types[out_idx]).str.contains(pat, regex=True).values
        out_group[hit & (out_group == "other")] = g
    out_group = out_group.astype(str)

    inst = meta.instance.astype(str)
    side = np.where(inst.str.endswith("_L") | inst.str.contains(r"_L\b|\(L\)"), "L",
                    np.where(inst.str.endswith("_R") | inst.str.contains(r"_R\b|\(R\)"), "R", "M"))
    # some afferents (e.g. SNpp haltere neurons) carry no side label: infer it from position
    xl, xr = xyz[side == "L", 0].mean(), xyz[side == "R", 0].mean()
    unl = (side == "M") & np.isin(np.arange(n), port_idx)
    side[unl] = np.where(np.abs(xyz[unl, 0] - xl) < np.abs(xyz[unl, 0] - xr), "L", "R")

    g = FlightGraph(src=src.astype(np.int64), dst=dst.astype(np.int64), weight=wt,
                    sign=node_sign[src], body_id=body.astype(np.int64), type=types,
                    superclass=meta.superclass.astype(str).values, side=side,
                    neuropil=neuropil, xyz=xyz, hops_in=hops_in[kept].astype(np.float32),
                    hops_out=hops_out[kept].astype(np.float32), out_idx=out_idx,
                    out_group=out_group, port_idx=port_idx, port_of=port_of)
    g.save(cache)
    log("built |", g.summary())
    return g


if __name__ == "__main__":
    g = build_flight_graph()
    print(pd.Series(g.superclass).value_counts().head(12).to_string())
    print(pd.Series(g.out_group).value_counts().to_string())
