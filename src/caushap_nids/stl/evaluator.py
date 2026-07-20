"""STL robustness evaluator — quantitative semantics (Donzé & Maler, CAV 2010).

Each STLNode computes a per-step robustness vector ρ over the n-flow window:
  - ρ > 0  ⟹  formula satisfied with margin ρ
  - ρ ≤ 0  ⟹  formula violated (or exactly on boundary)
  - global_robustness = min(ρ[0..n-1])

Signals are plain numpy float64 arrays of shape (n,), keyed by the NF-v2
column name they were derived from.  No fitting or thresholds are learned —
all constants encode domain/protocol knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

Signals = dict[str, np.ndarray]  # col_name → float64 shape (n,)

_EMPTY = np.empty(0, dtype=np.float64)


class STLNode(ABC):
    """Abstract STL formula node."""

    @abstractmethod
    def _rho(self, s: Signals) -> np.ndarray:
        """Per-step robustness vector, shape (n,)."""

    def global_robustness(self, s: Signals) -> float:
        """min-over-time robustness (standard STL global semantics)."""
        rho = self._rho(s)
        if rho.size == 0:
            return -np.inf
        return float(rho.min())

    def satisfied(self, s: Signals) -> bool:
        return self.global_robustness(s) >= 0.0


# ── Atomic propositions ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ge(STLNode):
    """signal ≥ threshold.  ρ[t] = signal[t] − threshold."""
    sig: str
    thr: float

    def _rho(self, s: Signals) -> np.ndarray:
        return s[self.sig] - self.thr


@dataclass(frozen=True)
class Le(STLNode):
    """signal ≤ threshold.  ρ[t] = threshold − signal[t]."""
    sig: str
    thr: float

    def _rho(self, s: Signals) -> np.ndarray:
        return self.thr - s[self.sig]


@dataclass(frozen=True)
class Eq(STLNode):
    """signal == value.  ρ[t] = +1 if equal, −1 if not."""
    sig: str
    val: float

    def _rho(self, s: Signals) -> np.ndarray:
        return np.where(s[self.sig] == self.val, 1.0, -1.0).astype(np.float64)


# ── Boolean connectives ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Neg(STLNode):
    """¬φ.  ρ[t] = −ρφ[t]."""
    phi: STLNode

    def _rho(self, s: Signals) -> np.ndarray:
        return -self.phi._rho(s)


@dataclass(frozen=True)
class And(STLNode):
    """φ₁ ∧ … ∧ φₙ.  ρ[t] = min(ρ₁[t], …, ρₙ[t])."""
    children: tuple[STLNode, ...]

    def _rho(self, s: Signals) -> np.ndarray:
        return np.minimum.reduce([c._rho(s) for c in self.children])


@dataclass(frozen=True)
class Or(STLNode):
    """φ₁ ∨ … ∨ φₙ.  ρ[t] = max(ρ₁[t], …, ρₙ[t])."""
    children: tuple[STLNode, ...]

    def _rho(self, s: Signals) -> np.ndarray:
        return np.maximum.reduce([c._rho(s) for c in self.children])


# ── Temporal operators (window-scoped) ────────────────────────────────────────

@dataclass(frozen=True)
class Globally(STLNode):
    """G φ — min over all n steps.  ρ = np.full(n, min(ρφ))."""
    phi: STLNode

    def _rho(self, s: Signals) -> np.ndarray:
        inner = self.phi._rho(s)
        if inner.size == 0:
            return inner
        return np.full_like(inner, inner.min())


@dataclass(frozen=True)
class Eventually(STLNode):
    """F φ — max over all n steps.  ρ = np.full(n, max(ρφ))."""
    phi: STLNode

    def _rho(self, s: Signals) -> np.ndarray:
        inner = self.phi._rho(s)
        if inner.size == 0:
            return inner
        return np.full_like(inner, inner.max())


# ── Factory helpers (clean call-site syntax) ──────────────────────────────────

def ge(sig: str, thr: float) -> Ge:
    return Ge(sig=sig, thr=thr)


def le(sig: str, thr: float) -> Le:
    return Le(sig=sig, thr=thr)


def eq(sig: str, val: float) -> Eq:
    return Eq(sig=sig, val=val)


def neg(phi: STLNode) -> Neg:
    return Neg(phi=phi)


def and_(*children: STLNode) -> And:
    if len(children) < 2:
        raise ValueError("and_() requires at least 2 children")
    return And(children=children)


def or_(*children: STLNode) -> Or:
    if len(children) < 2:
        raise ValueError("or_() requires at least 2 children")
    return Or(children=children)


def globally(phi: STLNode) -> Globally:
    return Globally(phi=phi)


def eventually(phi: STLNode) -> Eventually:
    return Eventually(phi=phi)
