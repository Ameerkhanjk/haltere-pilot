"""Download the raw male CNS connectome from neuPrint (only needed to rebuild the flight graph).

The repo already ships the small flight subnetwork (connectome_cache/flight_graph_k3_*.npz),
so training, analysis and the viewer work without this step. Run it if you want to change
which neurons are included (e.g. max_hops in connectome.py).

    export NEUPRINT_TOKEN="paste-your-token-here"   # from https://neuprint.janelia.org -> Account
    python -m flydrone.download
"""
import os
import sys
import time

import numpy as np

from .connectome import CACHE_DIR, RAW_CACHE, SOMA_CACHE, MIN_SYNAPSES

SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
PROPS = ["type", "instance", "class", "subclass", "superclass", "predictedNt", "consensusNt"]


def main():
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        sys.exit("Set your neuPrint token first:  export NEUPRINT_TOKEN=\"...\"  "
                 "(get it at https://neuprint.janelia.org -> Account)")
    from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC

    os.makedirs(CACHE_DIR, exist_ok=True)
    client = Client(SERVER, dataset=DATASET, token=token)
    print("connected to", client.dataset)

    if not os.path.exists(RAW_CACHE):
        t0 = time.time()
        print("downloading all neurons and connections (this takes a while) ...")
        crit = NC(status="Traced", cropped=False)
        neurons, conns = fetch_adjacencies(crit, crit, min_total_weight=MIN_SYNAPSES,
                                           properties=PROPS, client=client)
        np.savez_compressed(RAW_CACHE, neurons=neurons.to_dict("list"), conns=conns.to_dict("list"))
        print(f"  {len(neurons):,} neurons, {len(conns):,} connection rows in {time.time()-t0:.0f}s")
    else:
        print("already have", os.path.basename(RAW_CACHE))

    if not os.path.exists(SOMA_CACHE):
        print("downloading soma positions ...")
        df = client.fetch_custom(
            "MATCH (n:Neuron) WHERE n.somaLocation IS NOT NULL "
            "RETURN n.bodyId AS id, n.somaLocation.x AS x, n.somaLocation.y AS y, n.somaLocation.z AS z")
        np.savez_compressed(SOMA_CACHE, id=df.id.values.astype(np.int64),
                            xyz=df[["x", "y", "z"]].values.astype(np.float32))
        print(f"  {len(df):,} somas")
    else:
        print("already have", os.path.basename(SOMA_CACHE))
    print("done. Delete connectome_cache/flight_graph_*.npz and run  python -m flydrone.connectome  to rebuild.")


if __name__ == "__main__":
    main()
