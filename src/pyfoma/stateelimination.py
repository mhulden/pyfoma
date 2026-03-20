"""FST -> regex via state elimination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
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
        if ctx_prec >= _PREC_STAR and any(op in self.expr for op in (":", "|", "-")):
            return f"( {self.expr} )"
        return self.expr


@dataclass(frozen=True)
class OptionalRE(RE):
    r: RE

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        inner = self.r.to_pyfoma(_PREC_STAR)
        out = f"{inner}?"
        if ctx_prec > _PREC_STAR:
            return f"( {out} )"
        return out


@dataclass(frozen=True)
class PlusRE(RE):
    r: RE

    def to_pyfoma(self, ctx_prec: int = 0) -> str:
        inner = self.r.to_pyfoma(_PREC_STAR)
        out = f"{inner}+"
        if ctx_prec > _PREC_STAR:
            return f"( {out} )"
        return out


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


@dataclass(frozen=True)
class RewriteRule:
    name: str
    phase: str
    fn: Callable[[RE], Optional[RE]]


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
    if isinstance(r, (OptionalRE, PlusRE)):
        return 1 + re_size(r.r)
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


def _from_parts(parts: Tuple[RE, ...]) -> RE:
    if len(parts) == 0:
        return _EPS
    if len(parts) == 1:
        return parts[0]
    return ConcatRE(parts)


def _as_parts(r: RE) -> Tuple[RE, ...]:
    if isinstance(r, ConcatRE):
        return r.parts
    return (r,)


def _score_key(r: RE) -> Tuple[int, int, str]:
    rendered = r.to_pyfoma(0)
    return (re_size(r), len(rendered), rendered)


def _rule_alt_eps_to_optional(node: RE) -> Optional[RE]:
    if not isinstance(node, UnionRE):
        return None
    non_eps = [alt for alt in node.alts if not isinstance(alt, Eps)]
    if len(non_eps) == len(node.alts):
        return None
    base = _EMPTY
    for alt in non_eps:
        base = union(base, alt)
    if isinstance(base, Empty):
        return _EPS
    return OptionalRE(base)


def _rule_concat_to_plus_left(node: RE) -> Optional[RE]:
    if not isinstance(node, ConcatRE):
        return None
    parts = list(node.parts)
    for i in range(len(parts) - 1):
        if isinstance(parts[i + 1], Star) and parts[i] == parts[i + 1].r:
            new_parts = parts[:i] + [PlusRE(parts[i])] + parts[i + 2:]
            return concat(*new_parts)
    return None


def _rule_concat_to_plus_right(node: RE) -> Optional[RE]:
    if not isinstance(node, ConcatRE):
        return None
    parts = list(node.parts)
    for i in range(len(parts) - 1):
        if isinstance(parts[i], Star) and parts[i].r == parts[i + 1]:
            new_parts = parts[:i] + [PlusRE(parts[i + 1])] + parts[i + 2:]
            return concat(*new_parts)
    return None


def _rule_union_plus_or_star_with_eps(node: RE) -> Optional[RE]:
    if not isinstance(node, UnionRE):
        return None
    if len(node.alts) != 2:
        return None
    if not any(isinstance(alt, Eps) for alt in node.alts):
        return None
    other = next(alt for alt in node.alts if not isinstance(alt, Eps))
    if isinstance(other, Star):
        return other
    if isinstance(other, PlusRE):
        return Star(other.r)
    return None


def _rule_nested_closure_optional_plus(node: RE) -> Optional[RE]:
    if isinstance(node, Star):
        if isinstance(node.r, Star):
            return node.r
        if isinstance(node.r, OptionalRE):
            return Star(node.r.r)
        if isinstance(node.r, PlusRE):
            return Star(node.r.r)
    if isinstance(node, OptionalRE) and isinstance(node.r, OptionalRE):
        return node.r
    if isinstance(node, OptionalRE):
        if isinstance(node.r, PlusRE):
            return Star(node.r.r)
        if isinstance(node.r, Star):
            return node.r
    if isinstance(node, PlusRE):
        if isinstance(node.r, PlusRE):
            return node.r
        if isinstance(node.r, Star):
            return node.r
    return None


def _charclass_escape(sym: str) -> str:
    if sym in {"\\", "]", "[", "-", "^"}:
        return "\\" + sym
    return sym


def _union_to_charclass(union_node: RE) -> Optional[str]:
    if not isinstance(union_node, UnionRE):
        return None
    chars: Set[str] = set()
    for alt in union_node.alts:
        if not isinstance(alt, Label):
            return None
        if len(alt.value) != 1:
            return None
        sym = alt.value[0]
        if sym in {"", "."}:
            return None
        if len(sym) != 1:
            return None
        chars.add(sym)
    if len(chars) < 2:
        return None
    return "[" + "".join(_charclass_escape(ch) for ch in sorted(chars)) + "]"


def _rule_star_union_to_charclass(node: RE) -> Optional[RE]:
    if not isinstance(node, Star):
        return None
    cls = _union_to_charclass(node.r)
    if cls is None:
        return None
    return LabelExpr(cls + "*")


def _rule_union_to_charclass(node: RE) -> Optional[RE]:
    cls = _union_to_charclass(node)
    if cls is None:
        return None
    return LabelExpr(cls)


def _rule_optional_union_to_charclass(node: RE) -> Optional[RE]:
    if not isinstance(node, OptionalRE):
        return None
    cls = _union_to_charclass(node.r)
    if cls is None:
        return None
    return LabelExpr(cls + "?")


def _rule_plus_union_to_charclass(node: RE) -> Optional[RE]:
    if not isinstance(node, PlusRE):
        return None
    cls = _union_to_charclass(node.r)
    if cls is None:
        return None
    return LabelExpr(cls + "+")


def _rule_optional_union_pluses_to_union_stars(node: RE) -> Optional[RE]:
    if not isinstance(node, OptionalRE):
        return None
    if not isinstance(node.r, UnionRE):
        return None

    out = _EMPTY
    for alt in node.r.alts:
        if isinstance(alt, PlusRE):
            out = union(out, star(alt.r))
            continue
        if isinstance(alt, Star):
            out = union(out, alt)
            continue
        return None

    return out


def _rule_factor_common_prefix(node: RE) -> Optional[RE]:
    if not isinstance(node, UnionRE) or len(node.alts) < 2:
        return None
    alts = list(node.alts)
    for i in range(len(alts) - 1):
        for j in range(i + 1, len(alts)):
            p1 = _as_parts(alts[i])
            p2 = _as_parts(alts[j])
            k = 0
            while k < len(p1) and k < len(p2) and p1[k] == p2[k]:
                k += 1
            if k == 0:
                continue
            prefix = p1[:k]
            rest1 = _from_parts(p1[k:])
            rest2 = _from_parts(p2[k:])
            factored = concat(*prefix, union(rest1, rest2))
            new_alts = alts[:]
            new_alts[i] = factored
            del new_alts[j]
            out = _EMPTY
            for alt in new_alts:
                out = union(out, alt)
            return out
    return None


def _rule_factor_common_suffix(node: RE) -> Optional[RE]:
    if not isinstance(node, UnionRE) or len(node.alts) < 2:
        return None
    alts = list(node.alts)
    for i in range(len(alts) - 1):
        for j in range(i + 1, len(alts)):
            p1 = _as_parts(alts[i])
            p2 = _as_parts(alts[j])
            k = 0
            while k < len(p1) and k < len(p2) and p1[-1 - k] == p2[-1 - k]:
                k += 1
            if k == 0:
                continue
            suffix = p1[len(p1) - k:]
            rest1 = _from_parts(p1[:len(p1) - k])
            rest2 = _from_parts(p2[:len(p2) - k])
            factored = concat(union(rest1, rest2), *suffix)
            new_alts = alts[:]
            new_alts[i] = factored
            del new_alts[j]
            out = _EMPTY
            for alt in new_alts:
                out = union(out, alt)
            return out
    return None


_LOCAL_RULES: Tuple[RewriteRule, ...] = (
    RewriteRule("alt_eps_to_optional", "local", _rule_alt_eps_to_optional),
    RewriteRule("concat_left_to_plus", "local", _rule_concat_to_plus_left),
    RewriteRule("concat_right_to_plus", "local", _rule_concat_to_plus_right),
    RewriteRule("union_plus_or_star_with_eps", "local", _rule_union_plus_or_star_with_eps),
    RewriteRule("nested_closure_optional_plus", "local", _rule_nested_closure_optional_plus),
    RewriteRule("star_union_to_charclass", "local", _rule_star_union_to_charclass),
    RewriteRule("union_to_charclass", "local", _rule_union_to_charclass),
    RewriteRule("optional_union_to_charclass", "local", _rule_optional_union_to_charclass),
    RewriteRule("plus_union_to_charclass", "local", _rule_plus_union_to_charclass),
    RewriteRule(
        "optional_union_pluses_to_union_stars",
        "local",
        _rule_optional_union_pluses_to_union_stars,
    ),
    RewriteRule("factor_common_prefix", "local", _rule_factor_common_prefix),
    RewriteRule("factor_common_suffix", "local", _rule_factor_common_suffix),
)


def _apply_rule_first(node: RE, rule_fn: Callable[[RE], Optional[RE]]) -> Tuple[RE, bool]:
    rewritten = rule_fn(node)
    if rewritten is not None and rewritten != node:
        return rewritten, True

    if isinstance(node, Star):
        child, changed = _apply_rule_first(node.r, rule_fn)
        if changed:
            return star(child), True
        return node, False
    if isinstance(node, OptionalRE):
        child, changed = _apply_rule_first(node.r, rule_fn)
        if changed:
            return OptionalRE(child), True
        return node, False
    if isinstance(node, PlusRE):
        child, changed = _apply_rule_first(node.r, rule_fn)
        if changed:
            return PlusRE(child), True
        return node, False
    if isinstance(node, ConcatRE):
        parts = list(node.parts)
        for i, part in enumerate(parts):
            child, changed = _apply_rule_first(part, rule_fn)
            if changed:
                parts[i] = child
                return concat(*parts), True
        return node, False
    if isinstance(node, UnionRE):
        alts = list(node.alts)
        for i, alt in enumerate(alts):
            child, changed = _apply_rule_first(alt, rule_fn)
            if changed:
                alts[i] = child
                out = _EMPTY
                for candidate in alts:
                    out = union(out, candidate)
                return out, True
        return node, False
    return node, False


def _simplify_local(root: RE, max_steps: int = 200) -> Tuple[RE, List[str]]:
    current = root
    applied: List[str] = []
    for _ in range(max_steps):
        base_score = _score_key(current)
        improved = False
        for rule in _LOCAL_RULES:
            candidate, changed = _apply_rule_first(current, rule.fn)
            if not changed:
                continue
            if _score_key(candidate) < base_score:
                current = candidate
                applied.append(rule.name)
                improved = True
                break
        if not improved:
            break
    return current, applied


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
    simplify: bool = True,
    simplify_level: str = "local",
    max_simplify_steps: int = 200,
) -> str:
    """Convert a pyfoma FST into an equivalent pyfoma regex string.

    This uses GNFA-style state elimination with restart strategies. Before
    elimination, the input machine is normalized with the same core pipeline
    used by the regex compiler:

      ``trim().epsilon_remove().push_weights().determinize_as_dfa().minimize_as_dfa()``

    The resulting machine is then converted to a GNFA matrix and eliminated
    into one regex AST, which is rendered as a pyfoma regex string.

    Args:
      fst:
        Input FST (acceptor or transducer; tuple labels with arity >= 1).
      n:
        Number of elimination runs. The shortest rendered regex is returned.
      mode:
        Elimination strategy.
          - ``"dm"``: Delgado-Morais repeated state-weight heuristic.
            Run 0 is deterministic; runs 1..n-1 use randomized best-k tie
            breaking when ``n > 1``.
          - ``"random"``: purely random elimination order on every run.
      seed:
        Optional random seed for reproducible restart sampling.
      best_k:
        For randomized DM runs, sample uniformly from the ``best_k`` lowest
        weighted candidate states at each elimination step.
      simplify:
        If ``True``, apply local post-elimination rewrite rules to reduce
        output size.
      simplify_level:
        Simplification profile selector. Currently only ``"local"`` is
        implemented.
      max_simplify_steps:
        Hard cap on local simplification rewrite steps.

    Returns:
      A pyfoma-compatible regular expression string.

    Notes / current limits:
      - Transition wildcard support allows at most one ``'.'`` per transition
        label tuple in ``to_regex`` output construction.
      - Weights are currently not encoded in the emitted regex; they are only
        handled during normalization to keep determinization/minimization
        behavior aligned with the compiler path.
    """
    if n < 1:
        raise ValueError("to_regex: n must be >= 1")
    if mode not in ("dm", "random"):
        raise ValueError("to_regex: mode must be 'dm' or 'random'")
    if best_k < 1:
        raise ValueError("to_regex: best_k must be >= 1")
    if simplify_level != "local":
        raise ValueError("to_regex: only simplify_level='local' is currently implemented")
    if max_simplify_steps < 0:
        raise ValueError("to_regex: max_simplify_steps must be >= 0")

    work = (
        fst.copy_mod()
        .trim()
        .epsilon_remove()
        .push_weights()
        .determinize_as_dfa()
        .minimize_as_dfa()
    )

    R0, start, accept = _build_gnfa(work)
    best_str: Optional[str] = None
    base_rng = random.Random(seed)

    if mode == "dm":
        re0 = _eliminate_once(R0, start, accept, mode="dm", rng=None, best_k=best_k)
        if simplify:
            re0, _ = _simplify_local(re0, max_steps=max_simplify_steps)
        best_str = re0.to_pyfoma(0)
        for _run in range(1, n):
            rng = random.Random(base_rng.randint(0, 2**31 - 1))
            rerun = _eliminate_once(R0, start, accept, mode="dm", rng=rng, best_k=best_k)
            if simplify:
                rerun, _ = _simplify_local(rerun, max_steps=max_simplify_steps)
            candidate = rerun.to_pyfoma(0)
            if best_str is None or len(candidate) < len(best_str):
                best_str = candidate
    else:
        for _run in range(n):
            rng = random.Random(base_rng.randint(0, 2**31 - 1))
            rerun = _eliminate_once(R0, start, accept, mode="random", rng=rng, best_k=best_k)
            if simplify:
                rerun, _ = _simplify_local(rerun, max_steps=max_simplify_steps)
            candidate = rerun.to_pyfoma(0)
            if best_str is None or len(candidate) < len(best_str):
                best_str = candidate

    return best_str if best_str is not None else "'' - ''"
