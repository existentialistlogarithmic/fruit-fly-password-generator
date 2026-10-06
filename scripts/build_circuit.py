"""Extract the right-hemisphere olfactory learning circuit from the FlyWire connectome.

Source data: FlyWire whole-brain connectome, materialization 783
(Dorkenwald et al. 2024, Schlegel et al. 2024; Nature 634). Zenodo record 10676866,
file ``proofread_connections_783.feather`` (~850 MB). Licensed CC-BY 4.0.

Usage:
    python scripts/build_circuit.py --download          # fetch the feather file first
    python scripts/build_circuit.py --feather path/to/proofread_connections_783.feather

Writes ``flypass/data/olfactory_r_783.npz``, a compact circuit the generator ships with.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import requests

ZENODO_URL = (
    "https://zenodo.org/records/10676866/files/"
    "proofread_connections_783.feather?download=1"
)
# Antennal lobe -> mushroom body (calyx, peduncle, lobes) -> lateral horn, right side.
NEUROPILS = ["AL_R", "MB_CA_R", "MB_PED_R", "MB_VL_R", "MB_ML_R", "LH_R"]
MIN_SYNAPSES = 5  # FlyWire's usual threshold for a "real" connection
NT_COLUMNS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
NT_NAMES = np.array(["GABA", "ACh", "Glu", "Oct", "Ser", "DA"])
# GABA and glutamate (via GluCl) are inhibitory in the adult fly central brain.
INHIBITORY = {"GABA", "Glu"}

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "flypass" / "data" / "olfactory_r_783.npz"


def download(dest: pathlib.Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(ZENODO_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def build(feather: pathlib.Path, out: pathlib.Path = OUT) -> None:
    df = pd.read_feather(feather)
    df = df[df.neuropil.isin(NEUROPILS)]

    # Per-neuron output profile across neuropils, used to label regions.
    out_by_np = df.groupby(["pre_pt_root_id", "neuropil"]).syn_count.sum().unstack(fill_value=0)
    in_by_np = df.groupby(["post_pt_root_id", "neuropil"]).syn_count.sum().unstack(fill_value=0)

    # Collapse neuropils into one weighted edge per neuron pair.
    nt_weighted = df[NT_COLUMNS].mul(df.syn_count, axis=0)
    nt_weighted[["pre", "post", "syn"]] = df[["pre_pt_root_id", "post_pt_root_id", "syn_count"]].to_numpy()
    edges = nt_weighted.groupby(["pre", "post"]).sum()
    edges = edges[edges.syn >= MIN_SYNAPSES]

    pre = edges.index.get_level_values(0).to_numpy()
    post = edges.index.get_level_values(1).to_numpy()
    root_ids = np.unique(np.concatenate([pre, post]))
    index = {rid: i for i, rid in enumerate(root_ids)}

    # Each presynaptic neuron gets one transmitter identity (Dale's law).
    nt_per_neuron = edges[NT_COLUMNS].groupby(level=0).sum()
    nt_of_pre = NT_NAMES[np.argmax(nt_per_neuron.to_numpy(), axis=1)]
    nt = np.full(len(root_ids), "ACh", dtype="<U4")
    nt[[index[r] for r in nt_per_neuron.index]] = nt_of_pre

    def dominant(table: pd.DataFrame) -> np.ndarray:
        labels = np.full(len(root_ids), "", dtype="<U8")
        table = table.reindex(root_ids, fill_value=0)
        has = table.sum(axis=1).to_numpy() > 0
        labels[has] = table.columns.to_numpy()[np.argmax(table.to_numpy()[has], axis=1)]
        return labels

    np.savez_compressed(
        out,
        root_ids=root_ids.astype(np.int64),
        pre=np.array([index[r] for r in pre], dtype=np.int32),
        post=np.array([index[r] for r in post], dtype=np.int32),
        syn_count=edges.syn.to_numpy().astype(np.int32),
        nt=nt,
        inhibitory=np.isin(nt, list(INHIBITORY)),
        output_region=dominant(out_by_np),
        input_region=dominant(in_by_np),
        materialization=np.array(783),
    )
    print(f"wrote {out}: {len(root_ids)} neurons, {len(pre)} connections, "
          f"{int(edges.syn.sum())} synapses")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feather", type=pathlib.Path, default=ROOT / ".cache" / "proofread_connections_783.feather")
    p.add_argument("--download", action="store_true", help="download the FlyWire table from Zenodo first")
    p.add_argument("--out", type=pathlib.Path, default=OUT)
    args = p.parse_args(argv)
    if args.download:
        download(args.feather)
    build(args.feather, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
