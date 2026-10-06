import math
import os
import string
from collections import Counter

import numpy as np
import pytest

from flypass import CHARSETS, FlyBrain, generate, load_circuit
from flypass.cli import main
from flypass.generator import _draw, _policy_entropy


@pytest.fixture(scope="module")
def brain():
    return FlyBrain()


def test_circuit_is_the_flywire_subcircuit():
    c = load_circuit()
    assert c.n > 9000
    assert c.weights.nnz > 90000
    assert len(c.sensory) > 1000 and len(c.kenyon) > 1000
    # Real FlyWire root ids are 18-digit numbers starting with 7205759406.
    assert str(c.root_ids[0]).startswith("72057594")
    # Inhibitory neurons produce only negative outgoing weights (Dale's law).
    w = c.weights.tocsc()
    for i in range(c.n):
        col = w[:, i].data
        if col.size:
            assert len(set(np.sign(col))) == 1


def test_odor_drives_activity(brain):
    a = brain.run(np.random.default_rng(0), duration_ms=50)
    counts = a.spike_counts()
    assert counts[a.stimulated].sum() > 0
    assert counts.sum() > len(a.stimulated)  # activity spreads beyond the inputs


def test_same_seed_same_password(brain):
    seed = bytes(64)
    a = generate(_seed=seed, brain=brain, duration_ms=20)
    b = generate(_seed=seed, brain=brain, duration_ms=20)
    assert a.value == b.value


def test_stimulus_changes_password(brain):
    seed = bytes(64)
    a = generate(_seed=seed, stimulus="banana", brain=brain, duration_ms=20)
    b = generate(_seed=seed, stimulus="vinegar", brain=brain, duration_ms=20)
    assert a.value != b.value


def test_brain_activity_reaches_the_output(brain, monkeypatch):
    seed = bytes(64)
    a = generate(_seed=seed, brain=brain, duration_ms=20)
    original = brain.run

    def flipped(*args, **kwargs):
        act = original(*args, **kwargs)
        act.spikes[0, 0] ^= True  # a single spike difference
        return act

    monkeypatch.setattr(brain, "run", flipped)
    b = generate(_seed=seed, brain=brain, duration_ms=20)
    assert a.value != b.value


def test_fresh_seeds_are_unique(brain):
    values = {generate(brain=brain, duration_ms=10).value for _ in range(20)}
    assert len(values) == 20


@pytest.mark.parametrize("name", list(CHARSETS))
def test_charset_and_length(brain, name):
    pw = generate(length=16, charset=name, brain=brain, duration_ms=10)
    assert len(pw.value) == 16
    assert set(pw.value) <= set(CHARSETS[name])


def test_all_classes_present(brain):
    for _ in range(10):
        v = generate(length=4, brain=brain, duration_ms=5).value
        assert set(v) & set(string.ascii_lowercase)
        assert set(v) & set(string.ascii_uppercase)
        assert set(v) & set(string.digits)
        assert set(v) - set(string.ascii_letters + string.digits)


def test_rejects_bad_input(brain):
    with pytest.raises(ValueError):
        generate(length=3, brain=brain)
    with pytest.raises(ValueError):
        generate(charset="aab", brain=brain)


def test_draw_is_uniform():
    chars = CHARSETS["full"]  # 90 chars: naive modulo would bias the first 76
    text = _draw(os.urandom, chars, 90 * 2000)
    counts = Counter(text)
    expected = len(text) / len(chars)
    chi2 = sum((counts[c] - expected) ** 2 / expected for c in chars)
    # 89 degrees of freedom; p < 1e-6 beyond ~170.
    assert chi2 < 170


def test_policy_entropy():
    assert _policy_entropy("ab", 3) == pytest.approx(math.log2(8))
    assert _policy_entropy("aB", 2) == pytest.approx(1.0)  # only "aB" and "Ba"
    full = CHARSETS["full"]
    assert _policy_entropy(full, 24) < 24 * math.log2(len(full))
    assert _policy_entropy(full, 24) > 155


def test_cli(capsys):
    assert main(["-n", "2", "-l", "12", "--json"]) == 0
    out = capsys.readouterr().out
    assert out.count('"password"') == 2
