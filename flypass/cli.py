import argparse
import json
import sys

import numpy as np

from .brain import FlyBrain
from .generator import CHARSETS, generate

SHADES = " .:-=+*#%@"


def _sparkline(spikes: np.ndarray, width: int) -> str:
    bins = np.array_split(spikes.sum(axis=1), width)
    rates = np.array([b.sum() for b in bins], dtype=float)
    if rates.max() == 0:
        return " " * width
    idx = np.ceil(rates / rates.max() * (len(SHADES) - 1)).astype(int)
    return "".join(SHADES[i] for i in idx)


def _brain_report(pw, brain: FlyBrain) -> str:
    c, a = brain.circuit, pw.activity
    counts = a.spike_counts()
    groups = [
        ("antennal lobe in", c.sensory),
        ("Kenyon cells    ", c.kenyon),
        ("lateral horn    ", np.flatnonzero(c.output_region == "LH_R")),
        ("whole circuit   ", np.arange(c.n)),
    ]
    lines = [f"  odor: {len(a.stimulated)} of {len(c.sensory)} sensory neurons stimulated"]
    for name, idx in groups:
        active = int((counts[idx] > 0).sum())
        lines.append(f"  {name} |{_sparkline(a.spikes[:, idx], 50)}| {active}/{len(idx)} fired")
    lines.append(f"  {int(counts.sum())} spikes over {a.spikes.shape[0] * brain.dt:.0f} ms")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="flypass",
        description="Generate passwords by letting a simulated fruit-fly brain "
                    "(real FlyWire connectome) smell random odors.",
    )
    p.add_argument("-n", "--count", type=int, default=1, help="number of passwords (default 1)")
    p.add_argument("-l", "--length", type=int, default=24, help="password length (default 24)")
    p.add_argument("-c", "--charset", default="full",
                   help=f"one of {', '.join(CHARSETS)}, or a literal string of characters (default full)")
    p.add_argument("-s", "--stimulus", default="",
                   help="optional text mixed into the odor the fly smells (flavor only, not a secret)")
    p.add_argument("--no-classes", action="store_true",
                   help="don't require lower/upper/digit/symbol each to appear")
    p.add_argument("--brain", action="store_true", help="show what the fly brain did")
    p.add_argument("--json", action="store_true", help="output JSON")
    args = p.parse_args(argv)
    if args.count < 1 or args.length < 1:
        p.error("count and length must be positive")

    brain = FlyBrain()
    results = []
    for _ in range(args.count):
        try:
            pw = generate(args.length, args.charset, args.stimulus, brain,
                          require_all_classes=not args.no_classes)
        except ValueError as e:
            p.error(str(e))
        results.append(pw)
        if not args.json:
            print(pw.value)
            if args.brain:
                print(_brain_report(pw, brain))
                print(f"  ~{pw.entropy_bits:.0f} bits of entropy\n")

    if args.json:
        json.dump([{"password": r.value, "entropy_bits": round(r.entropy_bits, 1),
                    "spikes": int(r.activity.spike_counts().sum())} for r in results],
                  sys.stdout, indent=2)
        print()
    elif not args.brain:
        print(f"(~{results[0].entropy_bits:.0f} bits of entropy each)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
