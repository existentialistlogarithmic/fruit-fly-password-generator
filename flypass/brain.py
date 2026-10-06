"""Leaky integrate-and-fire simulation of a FlyWire connectome subcircuit.

Neuron and synapse parameters follow the whole-brain LIF model of
Shiu et al. 2024 ("A Drosophila computational brain model reveals sensorimotor
processing", Nature 634), integrated with forward Euler.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy import sparse

CIRCUIT_FILE = "olfactory_r_783.npz"


@dataclass(frozen=True)
class Circuit:
    root_ids: np.ndarray       # FlyWire root id per neuron
    weights: sparse.csr_matrix  # signed synapse counts, [post, pre]
    nt: np.ndarray             # dominant neurotransmitter per neuron
    output_region: np.ndarray  # neuropil receiving most of the neuron's output
    input_region: np.ndarray   # neuropil providing most of the neuron's input
    sensory: np.ndarray        # indices of antennal-lobe input neurons
    kenyon: np.ndarray         # indices of neurons fed in the mushroom-body calyx

    @property
    def n(self) -> int:
        return len(self.root_ids)


@lru_cache(maxsize=None)
def load_circuit(path: str | None = None) -> Circuit:
    if path is None:
        with resources.as_file(resources.files("flypass") / "data" / CIRCUIT_FILE) as p:
            d = dict(np.load(p))
    else:
        d = dict(np.load(path))
    n = len(d["root_ids"])
    sign = np.where(d["inhibitory"][d["pre"]], -1.0, 1.0)
    w = sparse.csr_matrix((sign * d["syn_count"], (d["post"], d["pre"])), shape=(n, n))

    syn_out = np.bincount(d["pre"], weights=d["syn_count"], minlength=n)
    syn_in = np.bincount(d["post"], weights=d["syn_count"], minlength=n)
    # Sensory-like: output lands in the antennal lobe and the neuron is mostly a
    # sender rather than a receiver within this circuit (receptor and projection neurons).
    sensory = np.flatnonzero((d["output_region"] == "AL_R") & (syn_in < 0.5 * syn_out))
    kenyon = np.flatnonzero(d["input_region"] == "MB_CA_R")
    return Circuit(d["root_ids"], w, d["nt"], d["output_region"], d["input_region"], sensory, kenyon)


@dataclass
class Activity:
    spikes: np.ndarray   # bool [steps, neurons]
    stimulated: np.ndarray

    def spike_counts(self) -> np.ndarray:
        return self.spikes.sum(axis=0)

    def raster_bytes(self) -> bytes:
        return np.packbits(self.spikes).tobytes()


class FlyBrain:
    # Shiu et al. 2024 parameters (mV, ms).
    V_REST = -52.0
    V_RESET = -52.0
    V_THRESH = -45.0
    TAU_M = 20.0
    TAU_SYN = 5.0
    T_REF = 2.2
    W_SYN = 0.275      # mV per synapse
    INPUT_RATE = 150.0  # Hz Poisson drive on stimulated sensory neurons
    W_INPUT = 68.75 * 0.275  # one input spike pushes a neuron well past threshold

    def __init__(self, circuit: Circuit | None = None, dt: float = 0.5):
        self.circuit = circuit or load_circuit()
        self.dt = dt
        self._w = self.circuit.weights * self.W_SYN

    def run(self, rng: np.random.Generator, duration_ms: float = 100.0,
            odor_fraction: float = 0.1, noise_mv: float = 0.5) -> Activity:
        """Present a random "odor" (a random subset of sensory neurons) and simulate.

        All randomness — which neurons smell the odor, Poisson input timing, membrane
        noise and initial voltages — comes from ``rng``.
        """
        c, dt = self.circuit, self.dt
        steps = int(round(duration_ms / dt))
        k = max(1, int(len(c.sensory) * odor_fraction))
        stimulated = rng.choice(c.sensory, size=k, replace=False)

        v = self.V_REST + rng.uniform(0.0, 2.0, c.n)
        g = np.zeros(c.n)
        refractory = np.zeros(c.n)
        spikes = np.zeros((steps, c.n), dtype=bool)
        p_input = self.INPUT_RATE * dt / 1000.0

        for t in range(steps):
            fired = spikes[t - 1] if t else np.zeros(c.n, dtype=bool)
            g += self._w @ fired.astype(float)
            drive = np.zeros(c.n)
            drive[stimulated] = (rng.random(k) < p_input) * self.W_INPUT
            g += drive
            g -= g * (dt / self.TAU_SYN)

            active = refractory <= 0
            dv = (self.V_REST - v + g) * (dt / self.TAU_M) + noise_mv * np.sqrt(dt) * rng.standard_normal(c.n)
            v = np.where(active, v + dv, v)
            refractory -= dt

            now = v >= self.V_THRESH
            spikes[t] = now
            v[now] = self.V_RESET
            refractory[now] = self.T_REF
        return Activity(spikes, stimulated)
