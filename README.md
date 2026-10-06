# flypass 🪰🔑

A password generator that runs every password through a simulated fruit-fly brain,
wired from the **real FlyWire connectome** of *Drosophila melanogaster*.

```
$ flypass --brain
8rYqAs]+r.[GWWvGv5Iq?blC
  odor: 188 of 1886 sensory neurons stimulated
  antennal lobe in |  .=*+*=%++=@#=***++*#=*##+*#*+%*+**=%+=++%+#%*+**| 211/1886 fired
  Kenyon cells     |          .:-=#*%%##%%#*#@%%%#%#%%%%%@%%##@%%#%#%%| 1277/2106 fired
  lateral horn     |      ...-+*%%##@%%#%#@#%@#@%%%%%@#@%@%@%%@#%%%%%%| 1000/2617 fired
  whole circuit    |  ......:-=+####@%%#%%%#%@%@%%%%%@%@%@%%%%@%%%%%%%| 2725/9514 fired
  15072 spikes over 100 ms
  ~156 bits of entropy
```

## How it works

1. **Seed.** 512 bits from your OS's cryptographically secure random generator
   (`secrets.token_bytes`).
2. **Odor.** The seed (plus an optional `--stimulus` phrase and a timing nonce) is hashed
   into an "odor": a random 10% of the fly's antennal-lobe input neurons get stimulated.
3. **Brain.** A leaky integrate-and-fire simulation runs for 100 ms on the fly's right-hemisphere
   olfactory learning circuit: **9,514 real neurons, 96,367 connections, 1.3 million synapses**
   (antennal lobe → mushroom body → lateral horn). Synapse counts and excitatory/inhibitory
   signs come from FlyWire, and the neuron parameters come from Shiu et al. 2024's whole-brain fly model.
   Spike timing, membrane noise, and starting voltages are all randomized from the seed.
4. **Extract.** The seed, the odor, and the fly's full spike raster (~238 KB of activity) go into
   SHAKE-256, which yields a stream of bytes.
5. **Characters.** Bytes become characters through rejection sampling, which avoids modulo bias.
   If a password is missing a character class, the whole password is redrawn, so the result is uniform
   over every password that meets the policy.

The mushroom body is the fly's real "neural algorithm" for hashing. Kenyon cells turn a dense odor
code into a sparse, random-looking tag
([Dasgupta, Stevens & Navlakha 2017, *Science*](https://doi.org/10.1126/science.aam9868)). Here that
same circuit sits inside a password pipeline.

## The honest security model

**The CSPRNG seed makes the passwords unpredictable. The fly brain does not.**
A neural simulation is a deterministic computation. If someone knew all of its inputs, they could
predict all of its outputs, and a brain does not create entropy. The design takes this into account:

- The 512-bit OS seed goes **directly** into the final hash, alongside the brain activity. An attacker
  who has this code, the full connectome, and your stimulus phrase still has to guess 2⁵¹² seeds.
- The brain therefore **can't weaken** the output. If the simulation did something degenerate
  (some odors barely activate it, which you can watch happen with `--brain`), the password would still
  be as strong as the seed.
- A password's strength is capped by its length and character set. A 24-character password from the
  default 90-character set has about 156 bits of entropy, which is far beyond brute force. The tool
  reports the exact number, adjusted for the "must contain every character class" rule.
- `--stimulus` adds flavor, not secrecy. Don't put a real secret in it.

So the passwords are as strong as any good generator's (1Password, Bitwarden, `openssl rand`).
The fly makes the process more interesting. It does not make the output stronger.

## Install & use

```bash
pip install .            # needs numpy + scipy; the 308 KB circuit is bundled
flypass                  # one 24-char password
flypass -n 5 -l 32       # five 32-char passwords
flypass -c unambiguous   # no Il1O0o look-alikes
flypass -c hex -l 64     # 256-bit hex key
flypass -s "ripe banana" --brain   # pick the fly's odor and watch it think
flypass --json
```

As a library:

```python
from flypass import generate
pw = generate(length=24, charset="full")
pw.value, pw.entropy_bits, pw.activity.spikes  # spikes: [time, neuron] bool array
```

## Rebuilding the circuit from raw FlyWire data

`flypass/data/olfactory_r_783.npz` was extracted from FlyWire materialization 783:

```bash
pip install ".[build]"
python scripts/build_circuit.py --download   # fetches the 850 MB table from Zenodo
```

To try another brain region, edit `NEUROPILS` in `scripts/build_circuit.py` (for example `EB`, `FB`, `PB`
for the central complex, or `ME_R`, `LO_R` for the optic lobe).

## Tests

```bash
pip install ".[test]" && pytest
```

The tests check that the bundled data is the FlyWire circuit and follows Dale's law, that odors spread
through the network, and that one flipped spike changes the password. They also check that the
character draw is uniform (a chi-square test), that the policy-entropy math is correct, and that the CLI works.

## Credits & license

Connectome data: FlyWire Consortium, CC-BY 4.0. Dorkenwald et al. 2024, "Neuronal wiring diagram of an
adult brain", *Nature* 634; Schlegel et al. 2024, *Nature* 634; Zenodo
[10676866](https://zenodo.org/records/10676866). Neuron model parameters: Shiu et al. 2024, "A Drosophila
computational brain model reveals sensorimotor processing", *Nature* 634.
