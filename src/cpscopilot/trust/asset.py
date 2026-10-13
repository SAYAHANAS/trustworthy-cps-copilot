"""Asset criticality (business / operational / safety impact).

Feature names are mapped to physical assets with ordered regular expressions
(first match wins). For each event the *implicated* assets are those whose
features deviate by at least ``z_threshold`` robust standard deviations; the
event's criticality is the maximum criticality among implicated assets, or the
default criticality when nothing deviates (no attribution possible).

Criticality values in the shipped configs are illustrative and must be set
with plant operations (e.g. from an IEC 62443 / ISA-95 asset inventory).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from cpscopilot.config import AssetRule

__all__ = ["AssetRegistry", "Attribution"]

UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class Attribution:
    """Vectorised attribution result."""

    criticality: np.ndarray  # (n,)
    asset: np.ndarray  # (n,) object - most critical implicated asset
    n_implicated: np.ndarray  # (n,) number of deviating features


class AssetRegistry:
    """Regex-based mapping from base feature names to assets."""

    def __init__(self, rules: list[AssetRule], default_criticality: float) -> None:
        self.rules = list(rules)
        self.default_criticality = float(default_criticality)
        self._compiled = [(re.compile(r.pattern), r) for r in self.rules]

    def lookup(self, base_feature: str) -> AssetRule | None:
        for pattern, rule in self._compiled:
            if pattern.search(base_feature):
                return rule
        return None

    def asset_of(self, base_feature: str) -> str:
        rule = self.lookup(base_feature)
        return rule.asset if rule else UNATTRIBUTED

    def describe(self, asset: str) -> str:
        for rule in self.rules:
            if rule.asset == asset:
                return rule.description
        return ""

    def attribute(self, z_abs: np.ndarray, base_cols: list[str], z_threshold: float) -> Attribution:
        """Most critical implicated asset per row."""
        rules = [self.lookup(c) for c in base_cols]
        crit = np.array([r.criticality if r else self.default_criticality for r in rules])
        names = np.array([r.asset if r else UNATTRIBUTED for r in rules], dtype=object)
        implicated = z_abs >= z_threshold
        masked = np.where(implicated, crit[None, :], -1.0)
        best = masked.argmax(axis=1)
        has_any = implicated.any(axis=1)
        criticality = np.where(has_any, crit[best], self.default_criticality)
        asset = np.where(has_any, names[best], UNATTRIBUTED)
        return Attribution(criticality=criticality.astype(np.float64), asset=asset,
                           n_implicated=implicated.sum(axis=1))
