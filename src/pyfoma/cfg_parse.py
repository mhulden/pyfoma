# cfg_parse.py
#
# Exact CFG / weighted CFG parser for pyfoma using a
# Chomsky-Schützenberger-style local regular language + balanced extraction.
#
# Features
# --------
# - Accepts grammars in the same format as `pyfoma.cfg.parse_cfg()`
# - Reuses `pyfoma.cfg.draw_tree()` for tree rendering
# - Supports EPSILON productions
# - Supports weighted parsing (costs; use negative log-probabilities for PCFGs)
# - Two interchangeable builders for the local regular language and h^{-1}(w):
#     * direct low-level graph construction (fast default)
#     * regex + $^restrict() construction (legible / paper-oriented)
# - Best parse via weighted heap extraction
# - n-best parses via a lazy generator wrapper (streaming interface)

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, Iterable, Any, Iterator
import heapq

from pyfoma.fst import FST, State
from pyfoma import cfg

@dataclass(frozen=True)
class BracketInfo:
    rid: int
    pos: int
    side: str  # "OB" | "CB"


@dataclass
class _PairInfo:
    derivs: List[Tuple]


class CFGParse:
    """
    Parse context-free grammars (and weighted CFGs / PCFGs-as-costs) using a
    Chomsky-Schützenberger representation implemented with pyfoma FSTs.

    The parser builds a *local regular language* ``R`` over bracket symbols and
    surface terminals, intersects it with ``h^{-1}(w)`` for an input sentence,
    and then performs balanced extraction over the resulting automaton. The
    grammar input format is exactly the same as in ``pyfoma.cfg.parse_cfg()``.

    Parameters
    ----------
    grammar:
        Grammar string in the same format accepted by ``pyfoma.cfg.parse_cfg``.
    start_symbol:
        Optional explicit start symbol. If omitted, the start symbol inferred by
        ``parse_cfg()`` is used.
    epsilon_symbol:
        Symbol used in the grammar for epsilon productions (default: ``"EPSILON"``).
    local_builder:
        Backend used to build the local regular language ``R``.
        - ``"direct"``: low-level graph construction (usually fastest)
        - ``"regex"``: ``FST.re(...)`` + ``$^restrict(...)`` (paper-readable)
    sent_builder:
        Backend used to build ``h^{-1}(w)`` for each sentence.
        - ``"direct"``
        - ``"regex"``

    Notes
    -----
    **Weights are treated as costs**, not probabilities. To use a PCFG, convert
    rule probabilities to negative log-probabilities before parsing.

    Rule costs are counted exactly once per rule application by placing the cost
    on the first opening bracket transition ``OB(rid,1)``.

    Public methods
    --------------
    - ``parse_ptb(sentence, n=10)``:
      unweighted enumeration of up to ``n`` PTB parses (debug / exploration).
    - ``parse_ptb_best(sentence)``:
      weighted best parse(s) using heap-based balanced extraction.
    - ``iter_ptb_nbest(sentence, ...)``:
      lazy generator yielding weighted parses in increasing cost order
      (enumeration-based ranking with deduplication).
    - ``parse_ptb_nbest(sentence, n=10, ...)``:
      convenience wrapper that consumes the lazy iterator.
    - ``parse_svg(...)`` / ``parse_svg_best(...)``:
      draw parse trees via ``pyfoma.cfg.draw_tree``.

    Paper mapping (Hulden LTC 2009)
    -------------------------------
    The local regular constraints correspond to:
    (a) boundary legality (start/end),
    (b) legality after opening brackets,
    (c) legality after terminals,
    (d) legality after non-final closing brackets,
    (e) legality after final closing brackets.
    
    Reference:
    
    Hulden, Mans. (2009). Parsing CFGs and PCFGs with a Chomsky-Schützenberger representation. In Language and Technology Conference (pp. 151-160). Berlin, Heidelberg: Springer Berlin Heidelberg. 
    """

    def __init__(self, grammar: str, *, start_symbol: Optional[str] = None, epsilon_symbol: str = "EPSILON",
                 local_builder: str = "direct", sent_builder: str = "direct"):
        valid_builders = {"direct", "regex"}
        if local_builder not in valid_builders:
            raise ValueError(f"Invalid local_builder={local_builder!r}; expected one of {sorted(valid_builders)}")
        if sent_builder not in valid_builders:
            raise ValueError(f"Invalid sent_builder={sent_builder!r}; expected one of {sorted(valid_builders)}")

        self.rules, inferred_start = cfg.parse_cfg(grammar)
        self.start_symbol = start_symbol or inferred_start
        self.epsilon_symbol = epsilon_symbol
        self.local_builder = local_builder
        self.sent_builder = sent_builder

        self.nonterminals, self.terminals_all, self.terminals_in, self._seen = self._symbol_sets()
        self.rule_by_id: Dict[int, Any] = {i: r for i, r in enumerate(self.rules)}
        self.rules_by_lhs: Dict[str, List[int]] = defaultdict(list)
        for rid, r in self.rule_by_id.items():
            self.rules_by_lhs[r.lhs].append(rid)

        self.open_sym: Dict[Tuple[int, int], str] = {}
        self.close_sym: Dict[Tuple[int, int], str] = {}
        self.bracket_info: Dict[str, BracketInfo] = {}

        self.allowed_starts: Set[str] = set()   # (a)
        self.allowed_finals: Set[str] = set()   # (a)
        self.next_map: Dict[str, Set[str]] = defaultdict(set)  # (b)-(e)

        self.brackets: Set[str] = set()
        self.local_symbols: Set[str] = set()
        self.surface_terminals: Set[str] = set(self.terminals_in)

        self._sym_quoted_cache: Dict[str, str] = {}
        self._regex_env: Dict[str, FST] = {}

        self._build_symbol_maps_and_local_constraints()
        self._prepare_regex_env()

        self.R = self._build_local_R_regex() if local_builder == "regex" else self._build_local_R_direct()

    def draw_tree(self, sentence: str, *, which: int = 0, style: str = "tree"):
        """
        Backward-compatible alias for `parse_svg()`.

        Draw the `which`-th parse tree for `sentence` using `pyfoma.cfg.draw_tree()`.
        """
        return self.parse_svg(sentence, which=which, style=style)
	
	
    def _symbol_sets(self):
        seen: Dict[str, int] = {}
        for r in self.rules:
            seen[r.lhs] = 1
        for r in self.rules:
            for sym in r.rhs:
                if seen.get(sym) not in (1, 2):
                    seen[sym] = 2
        nts = sorted([s for s, v in seen.items() if v == 1])
        ts_all = sorted([s for s, v in seen.items() if v == 2])
        ts_in = [t for t in ts_all if t != self.epsilon_symbol]
        return nts, ts_all, ts_in, seen

    def _is_nonterminal(self, sym: str) -> bool:
        return sym in self.rules_by_lhs

    def _ob(self, rid: int, pos: int) -> str: return f"OB__{rid}__{pos}"
    def _cb(self, rid: int, pos: int) -> str: return f"CB__{rid}__{pos}"

    def _rule_weight(self, r: Any) -> float:
        for attr in ("weight", "w", "cost"):
            if hasattr(r, attr):
                try:
                    return float(getattr(r, attr))
                except Exception:
                    pass
        return 0.0

    def _build_symbol_maps_and_local_constraints(self) -> None:
        for rid, r in self.rule_by_id.items():
            for i in range(1, len(r.rhs) + 1):
                ob, cb = self._ob(rid, i), self._cb(rid, i)
                self.open_sym[(rid, i)] = ob; self.close_sym[(rid, i)] = cb
                self.bracket_info[ob] = BracketInfo(rid, i, "OB")
                self.bracket_info[cb] = BracketInfo(rid, i, "CB")
                self.brackets |= {ob, cb}
        self.local_symbols = set(self.brackets) | set(self.surface_terminals)

        # (a) boundary legality
        for rid, r in self.rule_by_id.items():
            if r.lhs == self.start_symbol:
                self.allowed_starts.add(self.open_sym[(rid, 1)])
                self.allowed_finals.add(self.close_sym[(rid, len(r.rhs))])

        # (b)-(e) local successor legality
        terminal_followers: Dict[str, Set[str]] = defaultdict(set)
        all_closes = set(self.close_sym.values())
        for rid, r in self.rule_by_id.items():
            n = len(r.rhs)
            for i, sym in enumerate(r.rhs, start=1):
                ob, cb = self.open_sym[(rid, i)], self.close_sym[(rid, i)]

                # (b) after OB(rid,i)
                if sym == self.epsilon_symbol:
                    self.next_map[ob].add(cb)
                elif self._is_nonterminal(sym):
                    for rid2 in self.rules_by_lhs[sym]:
                        self.next_map[ob].add(self.open_sym[(rid2, 1)])
                else:
                    self.next_map[ob].add(sym)
                    terminal_followers[sym].add(cb)

                # (d)/(e) after CB(rid,i)
                if i < n:
                    self.next_map[cb].add(self.open_sym[(rid, i + 1)])
                else:
                    self.next_map[cb] |= all_closes

        # (c) after terminal
        for t, cbs in terminal_followers.items():
            self.next_map[t] |= cbs

    # ---------- regex helpers ----------
    def _q(self, sym: str) -> str:
        if sym in self._sym_quoted_cache:
            return self._sym_quoted_cache[sym]
        q = "'" + sym.replace("\\", "\\\\").replace("'", "\\'") + "'"
        self._sym_quoted_cache[sym] = q
        return q

    def _re_union_syms(self, syms: Iterable[str], empty_ok: bool = False) -> str:
        syms = list(dict.fromkeys(syms))
        if not syms:
            return "''" if empty_ok else "[]"
        if len(syms) == 1:
            return self._q(syms[0])
        return "(" + " | ".join(self._q(s) for s in syms) + ")"

    def _prepare_regex_env(self):
        self._regex_env = {
            "SYM": FST.re(self._re_union_syms(sorted(self.local_symbols))),
            "B": FST.re(self._re_union_syms(sorted(self.brackets))),
            "T": FST.re(self._re_union_syms(sorted(self.surface_terminals), empty_ok=True)),
        }
        if self.allowed_starts:
            self._regex_env["START"] = FST.re(self._re_union_syms(sorted(self.allowed_starts)))
        if self.allowed_finals:
            self._regex_env["FINAL"] = FST.re(self._re_union_syms(sorted(self.allowed_finals)))

    # ---------- low-level builders ----------
    def _build_local_R_direct(self) -> FST:
        fst = FST(alphabet=set(self.local_symbols)); bos = fst.initialstate
        s4: Dict[str, State] = {}
        for sym in sorted(self.local_symbols):
            st = State(name=sym); s4[sym] = st; fst.states.add(st)
        firstw = {self.open_sym[(rid, 1)]: self._rule_weight(r) for rid, r in self.rule_by_id.items()}

        # (a) start boundary
        for sym in sorted(self.allowed_starts):
            bos.add_transition(s4[sym], (sym,), firstw.get(sym, 0.0))
        # (b)-(e) local adjacency
        for sym, nxts in self.next_map.items():
            if sym not in s4:
                continue
            for nxt in sorted(nxts):
                if nxt in s4:
                    s4[sym].add_transition(s4[nxt], (nxt,), firstw.get(nxt, 0.0))
        # (a) end boundary
        fst.finalstates = set()
        for sym in self.allowed_finals:
            s4[sym].finalweight = 0.0
            fst.finalstates.add(s4[sym])
        return fst.trim()

    def _apply_first_open_weights(self, fst: FST) -> FST:
        firstw = {self.open_sym[(rid, 1)]: self._rule_weight(r) for rid, r in self.rule_by_id.items()}
        for s in fst.states:
            for lbl, t in s.all_transitions():
                if len(lbl) == 1:
                    t.weight = float(firstw.get(lbl[0], 0.0))
        return fst

    def _build_local_R_regex(self) -> FST:
        # (a) explicit start/end boundary legality + (b)-(e) successor legality via restrict
        pieces = ["$START $SYM* $FINAL"]
        for x in sorted(self.local_symbols):
            ctxs = []
            nxts = sorted(self.next_map.get(x, set()))
            if nxts:
                ctxs.append(f"_ {self._re_union_syms(nxts)}")
            if x in self.allowed_finals:
                ctxs.append("_ #")
            if not ctxs:
                pieces.append(f"$^restrict({self._q(x)} / _ [])")
            else:
                pieces.append(f"$^restrict({self._q(x)} / {', '.join(ctxs)})")
        rex = " & ".join(f"({p})" for p in pieces)
        R = FST.re(rex, dict(self._regex_env))
        return self._apply_first_open_weights(R)

    def _build_Rw_direct(self, sentence: str) -> FST:
        toks = [t for t in sentence.split() if t]
        bad = [t for t in toks if t not in self.surface_terminals]
        if bad:
            raise ValueError(f"Unknown surface terminal(s): {bad}")

        fst = FST(alphabet=set(self.local_symbols))
        states = [fst.initialstate]
        for i in range(len(toks)):
            st = State(name=f"w{i+1}")
            fst.states.add(st)
            states.append(st)
        for st in states:
            for b in sorted(self.brackets):
                st.add_transition(st, (b,), 0.0)
        for i, tok in enumerate(toks):
            states[i].add_transition(states[i + 1], (tok,), 0.0)
        fst.finalstates = {states[-1]}
        states[-1].finalweight = 0.0
        return fst

    def _build_Rw_regex(self, sentence: str) -> FST:
        toks = [t for t in sentence.split() if t]
        bad = [t for t in toks if t not in self.surface_terminals]
        if bad:
            raise ValueError(f"Unknown surface terminal(s): {bad}")
        if not toks:
            rex = "$B*"
        else:
            parts = []
            for t in toks:
                parts += ["$B*", self._q(t)]
            parts += ["$B*"]
            rex = " ".join(parts)
        return FST.re(rex, dict(self._regex_env))

    def _local_for_sentence(self, sentence: str) -> FST:
        Rw = self._build_Rw_regex(sentence) if self.sent_builder == "regex" else self._build_Rw_direct(sentence)
        return (self.R & Rw).trim()

    # ---------- balanced extraction ----------
    def _extract_balanced(self, local: FST, *, max_derivs_per_pair: int = 200):
        pairinfo: Dict[Tuple[State, State], _PairInfo] = {}
        agenda = deque()
        by_left: Dict[State, Set[Tuple[State, State]]] = defaultdict(set)
        by_right: Dict[State, Set[Tuple[State, State]]] = defaultdict(set)
        incoming_open: Dict[State, List[Tuple[State, str]]] = defaultdict(list)
        outgoing_close: Dict[State, List[Tuple[State, str]]] = defaultdict(list)
        terminals: List[Tuple[State, str, State]] = []

        for s in local.states:
            for lbl, t in s.all_transitions():
                if len(lbl) != 1:
                    continue
                sym = lbl[0]
                if sym in self.bracket_info:
                    if self.bracket_info[sym].side == "OB":
                        incoming_open[t.targetstate].append((s, sym))
                    else:
                        outgoing_close[s].append((t.targetstate, sym))
                else:
                    terminals.append((s, sym, t.targetstate))

        def add(pair, deriv):
            new = False
            if pair not in pairinfo:
                pairinfo[pair] = _PairInfo([])
                new = True
            if deriv not in pairinfo[pair].derivs and len(pairinfo[pair].derivs) < max_derivs_per_pair:
                pairinfo[pair].derivs.append(deriv)
            if new:
                by_left[pair[0]].add(pair)
                by_right[pair[1]].add(pair)
                agenda.append(pair)

        for s in local.states:
            add((s, s), ("empty",))
        for p, sym, q in terminals:
            add((p, q), ("term", sym))

        while agenda:
            p, q = agenda.popleft()
            cur = (p, q)

            for left in list(by_right[p]):
                p0, _ = left
                if left[0] == left[1] or cur[0] == cur[1]:
                    if (p0, q) not in pairinfo:
                        add((p0, q), ("empty",))
                    continue
                add((p0, q), ("concat", left, cur))

            for right in list(by_left[q]):
                _, q0 = right
                if right[0] == right[1] or cur[0] == cur[1]:
                    if (p, q0) not in pairinfo:
                        add((p, q0), ("empty",))
                    continue
                add((p, q0), ("concat", cur, right))

            for p0, ob in incoming_open.get(p, []):
                obi = self.bracket_info[ob]
                for q0, cb in outgoing_close.get(q, []):
                    cbi = self.bracket_info[cb]
                    if obi.rid == cbi.rid and obi.pos == cbi.pos:
                        add((p0, q0), ("wrap", ob, cur, cb))

        complete = [(local.initialstate, f) for f in local.finalstates if (local.initialstate, f) in pairinfo]
        return pairinfo, complete

    def _extract_balanced_best(self, local: FST):
        best_cost: Dict[Tuple[State, State], float] = {}
        best_deriv: Dict[Tuple[State, State], Tuple] = {}
        by_left: Dict[State, Set[Tuple[State, State]]] = defaultdict(set)
        by_right: Dict[State, Set[Tuple[State, State]]] = defaultdict(set)
        incoming_open: Dict[State, List[Tuple[State, str, float]]] = defaultdict(list)
        outgoing_close: Dict[State, List[Tuple[State, str, float]]] = defaultdict(list)
        terminals: List[Tuple[State, str, State, float]] = []

        for s in local.states:
            for lbl, t in s.all_transitions():
                if len(lbl) != 1:
                    continue
                sym = lbl[0]
                tw = float(getattr(t, "weight", 0.0) or 0.0)
                if sym in self.bracket_info:
                    if self.bracket_info[sym].side == "OB":
                        incoming_open[t.targetstate].append((s, sym, tw))
                    else:
                        outgoing_close[s].append((t.targetstate, sym, tw))
                else:
                    terminals.append((s, sym, t.targetstate, tw))

        heap = []
        serial = 0
        finalized = set()

        def push(pair, cost, deriv):
            nonlocal serial
            old = best_cost.get(pair)
            if old is None or cost < old:
                best_cost[pair] = cost
                best_deriv[pair] = deriv
                serial += 1
                heapq.heappush(heap, (cost, serial, pair))

        for s in local.states:
            push((s, s), 0.0, ("empty",))
        for p, sym, q, w in terminals:
            push((p, q), w, ("term", sym))

        while heap:
            cost, _, pair = heapq.heappop(heap)
            if pair in finalized or cost != best_cost.get(pair):
                continue
            finalized.add(pair)
            p, q = pair
            by_left[p].add(pair)
            by_right[q].add(pair)

            for left in list(by_right[p]):
                p0, _ = left
                push((p0, q), best_cost[left] + cost, ("concat", left, pair))
            for right in list(by_left[q]):
                _, q0 = right
                push((p, q0), cost + best_cost[right], ("concat", pair, right))
            for p0, ob, w_ob in incoming_open.get(p, []):
                obi = self.bracket_info[ob]
                for q0, cb, w_cb in outgoing_close.get(q, []):
                    cbi = self.bracket_info[cb]
                    if obi.rid == cbi.rid and obi.pos == cbi.pos:
                        push((p0, q0), w_ob + cost + w_cb, ("wrap", ob, pair, cb))

        complete = [(local.initialstate, f) for f in local.finalstates if (local.initialstate, f) in best_cost]
        complete.sort(key=lambda pr: best_cost[pr])
        return best_cost, best_deriv, complete

    # ---------- reconstruction / scoring ----------
    def _enumerate_encoded(self, pairinfo, pair, *, limit: int = 20):
        memo, inprog = {}, set()

        def rec(pr):
            if pr in memo:
                return memo[pr]
            if pr in inprog:
                return []
            inprog.add(pr)
            out, seen = [], set()
            pinfo = pairinfo.get(pr)
            if pinfo is None:
                memo[pr] = []
                inprog.remove(pr)
                return []
            for d in pinfo.derivs:
                k = d[0]
                if k == "empty":
                    cands = [tuple()]
                elif k == "term":
                    cands = [(d[1],)]
                elif k == "concat":
                    cands = [a + b for a in rec(d[1]) for b in rec(d[2])]
                elif k == "wrap":
                    cands = [(d[1],) + m + (d[3],) for m in rec(d[2])]
                else:
                    raise ValueError(k)
                for seq in cands:
                    if seq not in seen:
                        seen.add(seq)
                        out.append(seq)
                        if len(out) >= limit:
                            memo[pr] = out
                            inprog.remove(pr)
                            return out
            memo[pr] = out
            inprog.remove(pr)
            return out

        return rec(pair)

    def _best_encoded_from_deriv(self, best_deriv, pair):
        memo = {}

        def rec(pr):
            if pr in memo:
                return memo[pr]
            d = best_deriv[pr]
            k = d[0]
            if k == "empty":
                ans = tuple()
            elif k == "term":
                ans = (d[1],)
            elif k == "concat":
                ans = rec(d[1]) + rec(d[2])
            elif k == "wrap":
                ans = (d[1],) + rec(d[2]) + (d[3],)
            else:
                raise ValueError(k)
            memo[pr] = ans
            return ans

        return rec(pair)

    def _cost_of_encoded(self, seq: Iterable[str]) -> float:
        total = 0.0
        for sym in seq:
            bi = self.bracket_info.get(sym)
            if bi and bi.side == "OB" and bi.pos == 1:
                total += self._rule_weight(self.rule_by_id[bi.rid])
        return total

    def _decode_encoded_to_ptb(self, seq: Iterable[str]) -> str:
        toks = list(seq)
        n = len(toks)

        def parse_const(i):
            if i >= n:
                raise ValueError("Unexpected end")
            tok = toks[i]
            if tok not in self.bracket_info or self.bracket_info[tok].side != "OB":
                raise ValueError(f"Expected OB at {i}, got {tok!r}")
            bi = self.bracket_info[tok]
            if bi.pos != 1:
                raise ValueError(f"Expected pos-1 OB, got {tok!r}")
            rid = bi.rid
            r = self.rule_by_id[rid]
            i += 1
            kids = []
            for pos, sym in enumerate(r.rhs, start=1):
                ob, cb = self.open_sym[(rid, pos)], self.close_sym[(rid, pos)]
                if pos != 1:
                    if i >= n or toks[i] != ob:
                        raise ValueError(f"Expected {ob!r}")
                    i += 1
                if sym == self.epsilon_symbol:
                    child = self.epsilon_symbol
                elif self._is_nonterminal(sym):
                    child, i = parse_const(i)
                else:
                    if i >= n or toks[i] != sym:
                        raise ValueError(f"Expected terminal {sym!r}")
                    child = sym
                    i += 1
                if i >= n or toks[i] != cb:
                    raise ValueError(f"Expected {cb!r}")
                i += 1
                kids.append(child)
            return f"({r.lhs} {' '.join(kids)})", i

        ptb, j = parse_const(0)
        if j != n:
            raise ValueError("Extra tokens after parse")
        return ptb

    # ---------- public API ----------
    def parse_encoded(self, sentence: str, *, n: int = 10) -> List[str]:
        local = self._local_for_sentence(sentence)
        pairinfo, complete = self._extract_balanced(local)
        out, seen = [], set()
        for cp in complete:
            for seq in self._enumerate_encoded(pairinfo, cp, limit=max(n, 20)):
                s = " ".join(seq)
                if s not in seen:
                    seen.add(s)
                    out.append(s)
                    if len(out) >= n:
                        return out
        return out

    def parse_ptb(self, sentence: str, *, n: int = 10) -> List[str]:
        """Return up to `n` PTB parses by unweighted enumeration of the parse forest."""
        local = self._local_for_sentence(sentence)
        pairinfo, complete = self._extract_balanced(local)
        out, seen = [], set()
        for cp in complete:
            for seq in self._enumerate_encoded(pairinfo, cp, limit=max(n, 50)):
                if not seq:
                    continue
                try:
                    ptb = self._decode_encoded_to_ptb(seq)
                except Exception:
                    continue
                if ptb not in seen:
                    seen.add(ptb)
                    out.append(ptb)
                    if len(out) >= n:
                        return out
        return out

    def parse_ptb_best(self, sentence: str) -> List[Tuple[float, str]]:
        """Return weighted best parse(s) as ``(cost, ptb)`` using heap-based extraction."""
        local = self._local_for_sentence(sentence)
        best_cost, best_deriv, complete = self._extract_balanced_best(local)
        out, seen = [], set()
        for cp in complete:
            seq = self._best_encoded_from_deriv(best_deriv, cp)
            try:
                ptb = self._decode_encoded_to_ptb(seq)
            except Exception:
                continue
            if ptb not in seen:
                seen.add(ptb)
                out.append((best_cost[cp], ptb))
        out.sort(key=lambda x: x[0])
        return out

    def iter_ptb_nbest(self, sentence: str, *, per_complete_limit: int = 200,
                       include_encoded: bool = False) -> Iterator[Tuple[float, str] | Tuple[float, Tuple[str, ...], str]]:
        """
        Lazily yield weighted parses in increasing cost order.

        This is currently *enumeration-based ranking*, but exposed as a generator so
        callers can stop early. Internally it:
          1. builds the unweighted balanced forest,
          2. enumerates encoded derivations up to `per_complete_limit` per complete span,
          3. scores each derivation by summing rule costs on OB(rid,1),
          4. pushes candidates into a min-heap,
          5. yields distinct PTB parses cheapest-first.

        Parameters
        ----------
        sentence:
            Space-separated surface sentence.
        per_complete_limit:
            Max encoded derivations to enumerate per complete balanced span before ranking.
            Increase this for highly ambiguous grammars when deeper n-best results are needed.
        include_encoded:
            If True, yield ``(cost, encoded_seq, ptb)`` for debugging.
            Otherwise yield ``(cost, ptb)``.
        """
        local = self._local_for_sentence(sentence)
        pairinfo, complete = self._extract_balanced(local)

        heap: List[Tuple[float, int, Tuple[str, ...], str]] = []
        serial = 0
        for cp in complete:
            for seq in self._enumerate_encoded(pairinfo, cp, limit=per_complete_limit):
                if not seq:
                    continue
                try:
                    ptb = self._decode_encoded_to_ptb(seq)
                except Exception:
                    continue
                cost = self._cost_of_encoded(seq)
                serial += 1
                heapq.heappush(heap, (cost, serial, seq, ptb))

        seen_ptb: Set[str] = set()
        while heap:
            cost, _k, seq, ptb = heapq.heappop(heap)
            if ptb in seen_ptb:
                continue
            seen_ptb.add(ptb)
            if include_encoded:
                yield (cost, seq, ptb)
            else:
                yield (cost, ptb)

    def parse_ptb_nbest(self, sentence: str, *, n: int = 10, per_complete_limit: Optional[int] = None) -> List[Tuple[float, str]]:
        """
        Return the n cheapest distinct PTB parses as ``(cost, ptb)``.

        This is a thin wrapper over :meth:`iter_ptb_nbest`.
        """
        if per_complete_limit is None:
            per_complete_limit = max(50, 20 * n)
        out: List[Tuple[float, str]] = []
        for i, item in enumerate(self.iter_ptb_nbest(sentence, per_complete_limit=per_complete_limit)):
            if i >= n:
                break
            out.append(item)
        return out

    def parse(self, sentence: str, *, n: int = 10) -> List[str]:
        """Alias for :meth:`parse_ptb`."""
        return self.parse_ptb(sentence, n=n)

    def parse_svg(self, sentence: str, *, which: int = 0, style: str = "tree"):
        """Draw the `which`-th enumerated parse (PTB) via `pyfoma.cfg.draw_tree()`."""
        parses = self.parse_ptb(sentence, n=which + 1)
        if len(parses) <= which:
            raise ValueError("No parse found.")
        return cfg.draw_tree(parses[which], style=style)

    def parse_svg_best(self, sentence: str, *, style: str = "tree"):
        """Draw the best weighted parse via `pyfoma.cfg.draw_tree()`."""
        best = self.parse_ptb_best(sentence)
        if not best:
            raise ValueError("No parse found.")
        return cfg.draw_tree(best[0][1], style=style)

    # ---------- debug / benchmarking ----------
    def show_symbols(self) -> Dict[str, Any]:
        return {
            "start_symbol": self.start_symbol,
            "nonterminals": list(self.nonterminals),
            "surface_terminals": list(self.surface_terminals),
            "num_rules": len(self.rule_by_id),
            "num_brackets": len(self.brackets),
            "allowed_starts": sorted(self.allowed_starts),
            "allowed_finals": sorted(self.allowed_finals),
            "local_builder": self.local_builder,
            "sent_builder": self.sent_builder,
        }

    def debug_local_stats(self, sentence: str) -> Dict[str, Any]:
        """Return small stats for the local automaton used for `sentence`."""
        local = self._local_for_sentence(sentence)
        return {
            "sentence": sentence,
            "states": len(local.states),
            "transitions": sum(1 for s in local.states for _lbl, _t in s.all_transitions()),
            "finalstates": len(local.finalstates),
            "alphabet_size": len(local.alphabet),
        }

    def show_R_regex_recipe(self) -> str:
        """Human-readable summary of the regex backend construction for R."""
        return ("R = $START $SYM* $FINAL"
                "  & successor $^restrict(...) clauses per symbol for (b)-(e)"
                "  ; then apply weights on OB(rid,1) transitions")


TEST_GRAMMAR = r"""
S -> a S b | a a T b | a U b b | EPSILON
T -> a a T b | EPSILON
U -> a U b b | EPSILON
"""

TEST_GRAMMAR_WEIGHTED = r"""
S -> a S b 1.0 | a a T b 2.0 | a U b b 3.0 | EPSILON 0.5
T -> a a T b 1.7 | EPSILON 0.2
U -> a U b b 2.3 | EPSILON 0.4
"""
