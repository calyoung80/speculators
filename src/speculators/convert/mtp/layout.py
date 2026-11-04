"""Native -> Speculators key layouts for the MTP converter and stitcher.

A layout is a small, data-only description of one native checkpoint family:
which key prefixes belong to the MTP head, how native names map onto the
training-state names, and which keys are auxiliary metadata that must be
reported but not loaded.  GLM-specific rules live in ``glm_layout.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NativeLayout:
    """Key mapping for one native MTP checkpoint family."""

    name: str
    native_prefixes: tuple[str, ...]
    exact_remap: dict[str, str] = field(default_factory=dict)
    prefix_remap: tuple[tuple[str, str], ...] = ()
    ignored_keys: tuple[str, ...] = ()
    ignored_suffixes: tuple[str, ...] = ()

    def is_native_key(self, key: str) -> bool:
        return any(key.startswith(prefix) for prefix in self.native_prefixes)

    def is_ignored(self, key: str) -> bool:
        if key in self.ignored_keys:
            return True
        return any(key.endswith(suffix) for suffix in self.ignored_suffixes)

    def remap_key(self, key: str) -> str:
        """Map a native key to its training-state name."""
        if key in self.exact_remap:
            return self.exact_remap[key]
        for source, target in self.prefix_remap:
            if key.startswith(source):
                return target + key[len(source) :]
        return key

    def inverse_remap_key(self, key: str) -> str:
        """Map a training-state key back to its native name."""
        inverse_exact = {target: source for source, target in self.exact_remap.items()}
        if key in inverse_exact:
            return inverse_exact[key]
        # ``prefix_remap`` is ordered specific-first and ends with the
        # catch-all native prefix, so the inverse keeps that order too.
        for source, target in self.prefix_remap:
            if key.startswith(target):
                return source + key[len(target) :]
        return key
