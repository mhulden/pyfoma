"""FST -> regex via state elimination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import random


class RE:
    """Base class for internal regex AST."""

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        raise NotImplementedError


_PREC_UNION = 1
_PREC_CONCAT = 2
_PREC_STAR = 3


@dataclass(frozen=True)
class Empty(RE):
    """Empty language."""

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        return "'' - ''"


@dataclass(frozen=True)
class Eps(RE):
    """Epsilon language."""

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        return "''"


def _format_symbol(sym: str) -> str:
    if sym == "":
        return "''"
    if len(sym) == 1 and sym.isascii() and sym.isalnum():
        return sym
    esc = sym.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{esc}'"


@dataclass(frozen=True)
class Label(RE):
    """One transition label, potentially n-tape."""

    value: Tuple[str, ...]

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        if len(self.value) == 1:
            return _format_symbol(self.value[0])
        return ":".join(_format_symbol(sym) for sym in self.value)


@dataclass(frozen=True)
class LabelExpr(RE):
    """A pre-rendered label regex expression."""

    expr: str

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        return self.expr


@dataclass(frozen=True)
class Star(RE):
    r: RE

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        inner = self.r.to_pyfoma(_PREC_STAR)
        out = f"{inner}*"
        if ctx_prec > _PREC_STAR:
            return f"( {out} )"
        return out


@dataclass(frozen=True)
class ConcatRE(RE):
    parts: Tuple[RE, ...]

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        s = " ".join(
            p.to_pyfoma(_PREC_CONCAT) if not isinstance(p, UnionRE) else f"( {p.to_pyfoma(_PREC_UNION)} )"
            for p in self.parts
        )
        if ctx_prec > _PREC_CONCAT:
            return f"( {s} )"
        return s


@dataclass(frozen=True)
class UnionRE(RE):
    alts: Tuple[RE, ...]

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        s = " | ".join(a.to_pyfoma(_PREC_UNION) for a in self.alts)
        if ctx_prec > _PREC_UNION:
            return f"( {s} )"
        return s


_EMPTY = Empty()
_EPS = Eps()


def _is_empty(r: RE) -> bool:
    return isinstance(r, Empty)


def _is_eps(r: RE) -> bool:
    return isinstance(r, Eps)


def union(a: RE, b: RE) -> RE:
    """Union with flattening and deduplication."""
    if _is_empty(a):
        return b
    if _is_empty(b):
        return a
    if a == b:
        return a

    alts: List[RE] = []

    def add(x: RE):
        if isinstance(x, UnionRE):
            for y in x.alts:
                add(y)
        else:
            alts.append(x)

    add(a)
    add(b)

    seen = set()
    uniq: List[RE] = []
    for x in alts:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    if not uniq:
        return _EMPTY
    if len(uniq) == 1:
        return uniq[0]
    return UnionRE(tuple(uniq))


def concat(*xs: RE) -> RE:
    """Concatenation with flattening and epsilon/empty handling."""
    parts: List[RE] = []

    for x in xs:
        if _is_empty(x):
            return _EMPTY
        if _is_eps(x):
            continue
        if isinstance(x, ConcatRE):
            parts.extend(list(x.parts))
        else:
            parts.append(x)

    if not parts:
        return _EPS
    if len(parts) == 1:
        return parts[0]
    return ConcatRE(tuple(parts))


def star(x: RE) -> RE:
    if _is_empty(x) or _is_eps(x):
        return _EPS
    if isinstance(x, Star):
        return x
    return Star(x)


def re_size(r: RE) -> int:
    """A simple syntactic size measure used for elimination ordering."""
    if isinstance(r, Empty):
        return 0
    if isinstance(r, (Eps, Label, LabelExpr)):
        return 1
    if isinstance(r, Star):
        return 1 + re_size(r.r)
    if isinstance(r, UnionRE):
        k = len(r.alts)
        return (k - 1) + sum(re_size(a) for a in r.alts)
    if isinstance(r, ConcatRE):
        k = len(r.parts)
        return (k - 1) + sum(re_size(p) for p in r.parts)
    raise TypeError(f"Unknown RE node type: {type(r)}")


def state_weight(R: List[List[RE]], q: int, active: List[bool]) -> int:
    """Delgado-Morais repeated state-weight heuristic."""
    n = len(R)

    in_edges: List[RE] = []
    for i in range(n):
        if not active[i] or i == q:
            continue
        if not isinstance(R[i][q], Empty):
            in_edges.append(R[i][q])

    out_edges: List[RE] = []
    for j in range(n):
        if not active[j] or j == q:
            continue
        if not isinstance(R[q][j], Empty):
            out_edges.append(R[q][j])

    incoming = len(in_edges)
    outgoing = len(out_edges)
    if incoming == 0 or outgoing == 0:
        return 0

    loop = R[q][q]
    loop_size = re_size(loop) if not isinstance(loop, Empty) else 0
    in_size = sum(re_size(x) for x in in_edges)
    out_size = sum(re_size(x) for x in out_edges)

    return (in_size * outgoing) + (out_size * incoming) + (loop_size * incoming * outgoing)


def _symbol_union_expr(symbols: List[str]) -> str:
    rendered = [_format_symbol(sym) for sym in symbols]
    if not rendered:
        return "'' - ''"
    if len(rendered) == 1:
        return rendered[0]
    return "( " + " | ".join(rendered) + " )"


def _template_sort_key(item):
    pos, fixed = item
    return (pos, tuple("" if sym is None else sym for sym in fixed))


def _labels_to_re(labels: Set[Tuple[str, ...]], sigma: Set[str]) -> RE:
    """Convert all labels between one state-pair into one RE edge label."""
    terms: RE = _EMPTY
    by_arity: Dict[int, Set[Tuple[str, ...]]] = {}
    for label in labels:
        by_arity.setdefault(len(label), set()).add(label)

    for arity in sorted(by_arity):
        group = by_arity[arity]
        concrete: Set[Tuple[str, ...]] = set()
        wildcard_templates = {}

        for label in group:
            if all(sym == "" for sym in label):
                terms = union(terms, _EPS)
                continue

            dot_positions = [idx for idx, sym in enumerate(label) if sym == "."]
            if not dot_positions:
                concrete.add(label)
                continue
            if len(dot_positions) > 1:
                raise ValueError(
                    "to_regex currently supports wildcard labels with at most one '.' per transition label."
                )
            pos = dot_positions[0]
            fixed = tuple(sym if idx != pos else None for idx, sym in enumerate(label))
            wildcard_templates[(pos, fixed)] = True

        covered_concrete: Set[Tuple[str, ...]] = set()
        for pos, fixed in sorted(wildcard_templates.keys(), key=_template_sort_key):
            allowed_in_sigma: Set[str] = set()
            for label in concrete:
                if all(idx == pos or label[idx] == fixed[idx] for idx in range(arity)):
                    wildcard_side = label[pos]
                    if wildcard_side != "":
                        covered_concrete.add(label)
                    if wildcard_side in sigma:
                        allowed_in_sigma.add(wildcard_side)

            excluded = sorted(sigma - allowed_in_sigma)
            wildcard_component = "." if not excluded else f"( . - {_symbol_union_expr(excluded)} )"
            components = []
            for idx in range(arity):
                if idx == pos:
                    components.append(wildcard_component)
                else:
                    components.append(_format_symbol(fixed[idx]))
            wildcard_expr = components[0] if arity == 1 else ":".join(components)
            terms = union(terms, LabelExpr(wildcard_expr))

        for label in sorted(concrete):
            if label in covered_concrete:
                continue
            terms = union(terms, Label(label))

    return terms


def _build_gnfa(fst):
    """Build GNFA adjacency matrix R from a pyfoma FST object."""
    statenums = fst.number_unnamed_states(force=True)
    n = len(fst.states)

    num_to_state = [None] * n
    for state in fst.states:
        num_to_state[statenums[id(state)]] = state

    start_num = statenums[id(fst.initialstate)]
    finals = {statenums[id(state)] for state in fst.finalstates}

    start = n
    accept = n + 1
    total = n + 2
    sigma = {sym for sym in fst.alphabet if sym not in {".", ""}}

    R: List[List[RE]] = [[_EMPTY for _ in range(total)] for __ in range(total)]
    edge_labels: Dict[Tuple[int, int], Set[Tuple[str, ...]]] = {}

    for i in range(n):
        state = num_to_state[i]
        for label, tset in state.transitions.items():
            if not isinstance(label, tuple) or len(label) == 0:
                raise ValueError(f"to_regex expects tuple labels with arity >= 1, got {label!r}")
            for transition in tset:
                j = statenums[id(transition.targetstate)]
                edge_labels.setdefault((i, j), set()).add(label)

    for (i, j), labels in edge_labels.items():
        R[i][j] = union(R[i][j], _labels_to_re(labels, sigma))

    R[start][start_num] = union(R[start][start_num], _EPS)
    for final in finals:
        R[final][accept] = union(R[final][accept], _EPS)

    return R, start, accept


def _eliminate_once(
    R0: List[List[RE]],
    start: int,
    accept: int,
    *,
    mode: str,
    rng: Optional[random.Random] = None,
    best_k: int = 3,
) -> RE:
    """Perform one elimination run on a GNFA adjacency matrix."""
    if mode not in ("dm", "random"):
        raise ValueError("mode must be 'dm' or 'random'")

    n = len(R0)
    R: List[List[RE]] = [row[:] for row in R0]
    active = [True] * n
    remaining = [k for k in range(n) if k not in (start, accept)]

    while remaining:
        if mode == "random":
            if rng is None:
                raise ValueError("mode='random' requires rng")
            k = rng.choice(remaining)
        elif rng is None:
            k = min(remaining, key=lambda q: (state_weight(R, q, active), q))
        else:
            weighted = [(state_weight(R, q, active), q) for q in remaining]
            weighted.sort()
            k_choices = [q for _, q in weighted[:max(1, min(best_k, len(weighted)))]]
            k = rng.choice(k_choices)

        col_empty = True
        row_empty = True
        for i in range(n):
            if not active[i] or i == k:
                continue
            if not isinstance(R[i][k], Empty):
                col_empty = False
                break
        for j in range(n):
            if not active[j] or j == k:
                continue
            if not isinstance(R[k][j], Empty):
                row_empty = False
                break
        if col_empty and row_empty:
            active[k] = False
            remaining.remove(k)
            continue

        loop_star = star(R[k][k])
        for i in range(n):
            if not active[i] or i == k:
                continue
            Rik = R[i][k]
            if isinstance(Rik, Empty):
                continue
            for j in range(n):
                if not active[j] or j == k:
                    continue
                Rkj = R[k][j]
                if isinstance(Rkj, Empty):
                    continue
                add = concat(Rik, loop_star, Rkj)
                R[i][j] = union(R[i][j], add)

        for i in range(n):
            R[i][k] = _EMPTY
            R[k][i] = _EMPTY
        R[k][k] = _EMPTY
        active[k] = False
        remaining.remove(k)

    return R[start][accept]


def to_regex(
    fst,
    *,
    n: int = 1,
    mode: str = "dm",
    seed: Optional[int] = None,
    best_k: int = 3,
) -> str:
    """Convert a pyfoma FST into an equivalent pyfoma regex string.

    Assumptions:
      - tuple labels with arity >= 1
      - wildcard labels may contain at most one '.' per transition label
      - weights are ignored
    """
    if n < 1:
        raise ValueError("to_regex: n must be >= 1")
    if mode not in ("dm", "random"):
        raise ValueError("to_regex: mode must be 'dm' or 'random'")
    if best_k < 1:
        raise ValueError("to_regex: best_k must be >= 1")

    work = fst.copy_mod().trim().epsilon_remove().trim()
    if work.arity() == 1:
        work = work.determinize_unweighted().minimize().trim()

    R0, start, accept = _build_gnfa(work)
    best_str: Optional[str] = None
    base_rng = random.Random(seed)

    if mode == "dm":
        re0 = _eliminate_once(R0, start, accept, mode="dm", rng=None, best_k=best_k)
        best_str = re0.to_pyfoma(0)
        for _run in range(1, n):
            rng = random.Random(base_rng.randint(0, 2**31 - 1))
            rerun = _eliminate_once(R0, start, accept, mode="dm", rng=rng, best_k=best_k)
            candidate = rerun.to_pyfoma(0)
            if best_str is None or len(candidate) < len(best_str):
                best_str = candidate
    else:
        for _run in range(n):
            rng = random.Random(base_rng.randint(0, 2**31 - 1))
            rerun = _eliminate_once(R0, start, accept, mode="random", rng=rng, best_k=best_k)
            candidate = rerun.to_pyfoma(0)
            if best_str is None or len(candidate) < len(best_str):
                best_str = candidate

    return best_str if best_str is not None else "'' - ''"
