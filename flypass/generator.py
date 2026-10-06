"""Turn fly-brain activity into passwords.

Where the security comes from
-----------------------------
A 512-bit seed from the operating system's CSPRNG (``secrets``) is the root of
every password. It drives the fly brain (which odor it smells, spike timing,
membrane noise), and it is *also* fed directly into the final SHAKE-256
extraction alongside the brain's full spike raster. So:

* Knowing the connectome, the code, and your stimulus phrase gets an attacker
  nothing without the 512-bit seed.
* The simulation can only add mixing; it can never reduce strength below what
  the CSPRNG seed provides, because the seed reaches the hash unchanged.
"""

import hashlib
import math
import secrets
import string
import time
from dataclasses import dataclass

import numpy as np

from .brain import Activity, FlyBrain

SYMBOLS = "!#$%&()*+,-./:;<=>?@[]^_{|}~"
AMBIGUOUS = set("Il1O0o|`'\"")

CHARSETS = {
    "full": string.ascii_lowercase + string.ascii_uppercase + string.digits + SYMBOLS,
    "alnum": string.ascii_lowercase + string.ascii_uppercase + string.digits,
    "hex": "0123456789abcdef",
    "unambiguous": "".join(ch for ch in string.ascii_letters + string.digits + SYMBOLS if ch not in AMBIGUOUS),
}

DOMAIN = b"flypass/v1"


@dataclass
class Password:
    value: str
    entropy_bits: float
    activity: Activity

    def __str__(self) -> str:
        return self.value


def _classes(charset: str) -> list[set[str]]:
    groups = [string.ascii_lowercase, string.ascii_uppercase, string.digits, SYMBOLS]
    return [set(g) & set(charset) for g in groups if set(g) & set(charset)]


def _draw(stream, charset: str, length: int) -> str:
    """Unbiased index draw by rejection sampling over bytes."""
    n = len(charset)
    limit = 256 - (256 % n)
    out = []
    while len(out) < length:
        b = stream(1)[0]
        if b < limit:
            out.append(charset[b % n])
    return "".join(out)


def _policy_entropy(charset: str, length: int) -> float:
    """log2 of the number of passwords containing at least one char of every class.

    Counted by inclusion-exclusion over the missing classes.
    """
    sizes = [len(c) for c in _classes(charset)]
    total = 0
    for mask in range(1 << len(sizes)):
        missing = sum(s for i, s in enumerate(sizes) if mask >> i & 1)
        sign = -1 if bin(mask).count("1") % 2 else 1
        total += sign * (len(charset) - missing) ** length
    return math.log2(total)


def generate(length: int = 24, charset: str = "full", stimulus: str = "",
             brain: FlyBrain | None = None, require_all_classes: bool = True,
             duration_ms: float = 100.0, _seed: bytes | None = None) -> Password:
    """Generate one password.

    ``stimulus`` is optional text (a phrase, keyboard mash) that is mixed into the
    odor the fly smells. It adds flavor, not secrecy — never rely on it.
    ``_seed`` exists only for reproducible tests; leave it unset.
    """
    chars = CHARSETS.get(charset, charset)
    if len(set(chars)) != len(chars) or len(chars) < 2:
        raise ValueError("charset must contain at least two distinct characters, no repeats")
    classes = _classes(chars) if require_all_classes else []
    if length < len(classes):
        raise ValueError(f"length must be at least {len(classes)} to cover every character class")

    brain = brain or FlyBrain()
    seed = _seed if _seed is not None else secrets.token_bytes(64)
    jitter = b"" if _seed is not None else time.perf_counter_ns().to_bytes(8, "little")
    odor = hashlib.shake_256(DOMAIN + b"/odor" + seed + stimulus.encode() + jitter).digest(32)

    rng = np.random.Generator(np.random.PCG64(int.from_bytes(odor, "little")))
    activity = brain.run(rng, duration_ms=duration_ms)

    xof = hashlib.shake_256()
    for part in (DOMAIN + b"/extract", seed, odor, activity.raster_bytes()):
        xof.update(len(part).to_bytes(8, "little") + part)

    # SHAKE is an XOF: read it as a stream, re-deriving longer output on demand.
    offset = 0
    buf = b""

    def stream(k: int) -> bytes:
        nonlocal offset, buf
        while offset + k > len(buf):
            buf = xof.digest(max(256, 2 * len(buf)))
        out = buf[offset:offset + k]
        offset += k
        return out

    while True:
        value = _draw(stream, chars, length)
        if all(set(value) & c for c in classes):
            break
    bits = _policy_entropy(chars, length) if classes else length * math.log2(len(chars))
    return Password(value, min(bits, 8 * len(seed)), activity)
