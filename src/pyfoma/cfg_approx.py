"""
CFG subset approximation with finite-state machinery
====================================================

This module implements the CFG subset approximation method in Hulden & Silfverberg (2014)
that turns a CFG into a finite-state grammar relation Gr, and then parses by intersecting
Gr with an input sentence expanded with optional nonterminal/markup insertions.

Hulden, Mans, and Miikka Silfverberg. 2014. Finite-state subset approximation of phrase structure,
ISAIM. 2014.

TL;DR Representation trick (why it works)
-----------------------------------------
Derivations are encoded as a *flat string* over a small auxiliary alphabet:

  - terminals:                      [ d ]
  - rule applications (heads):      [ A -> B b ]
  - threaded RHS material:          { ... }
  - one distinguished root:         ( S -> ... )
  - center-embeddings:               < ... >
  - top tree right-branch separator: ^

The crucial locality property is the ordering of the encoding around each subtree:
material to the *left* of the subtree root is listed in a postorder-like walk, then the
root itself, then material on the *right* in a preorder-like walk. In effect, each subtree
is written as a locally checkable 'spine' where the head bracket acts as a synchronization
point between what has already been expanded (left side) and what remains to be expanded
(right side). This makes it possible to enforce a context-free derivation with *regular*
constraints on bounded neighborhoods.

Subset approximation and the < > markers
----------------------------------------
The approximation is enforced by the constraint LR2, which characterizes locally valid
neighborhoods of rule material inside the flat parse string. To extend the approximation
to deeper recursion, the pipeline repeatedly composes an *iteration transducer* IT
a fixed number of times (parameter `levels`).

IT temporarily inserts and filters bounded subtrees inside brace-context. The markers
< ... > that you see in parse strings are the readable remnants of internal delimiters
used to fence off candidate inserted substrings while checking LR2-compatibility; composing
IT repeatedly refines the approximation by allowing recursively inserted, locally-checked
substructures.

Visualization
-------------
The native parse strings are optimized for locality, not readability. This module therefore
provides:
  - conversion to a PTB-style bracketed representation, and
  - rendering of that PTB tree as an SVG (via `draw_tree()`).

Example: PP attachment ambiguity (unweighted)
---------------------------------------------
Classic ambiguity in “cats eat cats with cats” (PP attaches either to the object NP or VP):

    from pyfoma import cfg, cfg_approx
    grammar = '''
    S  -> NP VP
    NP -> D N | N | NP PP
    VP -> V NP | V NP PP
    PP -> P NP
    D  -> the
    P  -> with
    N  -> cats | crabs | claws | people | walls
    V  -> eat | scratch
    '''

    sent = "cats eat cats with cats"
    g = cfg_approx.CFG(grammar)
    g.compile(levels = 2)

    # Native parse strings:
    print("\n".join(g.parse_pretty(sent, n = 10)))

    # PTB:
    print("\n".join(g.parse_ptb(sent, n = 10)))



Example: weighted grammar (costs per parse)
-------------------------------------------
If the grammar provides trailing numeric weights, we compile Gr unweighted as usual and
then *post-weight* it by composing contextual rewrite rules that put a weight on the arrow
token "->" *only when it occurs as the head of a particular rule*.

    from pyfoma import cfg, cfg_approx
    grammar = '''
    S -> NP VP 1.0
    NP -> D N 1.0 | N 2.0
    NP -> NP RC 1.5
    RC -> NP V 2.5
    VP -> V NP 3.0
    VP -> V 1.0
    D -> the 1.0
    N -> rat 1.0 | cat 1.0 | dog 1.0 | cheese 1.0
    V -> chased 1.0 | killed 1.0 | ate 1.0
    '''

    sent = "the rat the cat the dog chased killed ate the cheese"
    g = cfg_approx.CFG(grammar)
    g.compile(levels = 3)

    # (cost, native parse string):
    for cost, s in g.parse_pretty_with_cost(sent, n = 10):
        print(cost, s)
    
    
PTB strings can also be visualized with draw_tree(ptbstring).

Epsilon (empty) right-hand sides are supported as follows:

LHS -> EPSILON

e.g.:

Det -> the | a | EPSILON
"""


from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pyfoma.fst import FST

from .cfg import (
    parse_cfg,
    _tok,
    funky_to_ptb,
    draw_tree,
)

class CFG:
    """Compile a (weighted) CFG into a finite-state grammar relation Gr and parse by intersection."""

    def __init__(self, grammar: str, *, start_symbol: Optional[str] = None, epsilon_symbol: str = "EPSILON"):
        self.rules, inferred_start = parse_cfg(grammar)
        self.start_symbol = start_symbol or inferred_start
        self.epsilon_symbol = epsilon_symbol
        self.fsts: Dict[str, FST] = {}
        self._compiled_levels: Optional[int] = None

    def _symbol_sets(self) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
        """Return (nonterminals, all-terminals, input-terminals, seen-map).

        input-terminals excludes the configured epsilon symbol so epsilon-productions can be
        represented in the grammar without requiring the user to type EPSILON tokens in the sentence.
        """
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

    def compile(self, *, levels: int = 2, keep_intermediates: bool = False) -> FST:
        fsts: Dict[str, FST] = {}
        nts, ts_all, ts_in, seen = self._symbol_sets()

        # define NT
        fsts["NT"] = FST.re("|".join(_tok(x) for x in nts) if nts else "''")

        # define T
        fsts["T"] = FST.re("|".join(_tok(x) for x in ts_all) if ts_all else "''")

        # terminals that must match the input sentence (exclude EPSILON)
        fsts["T_in"] = FST.re("|".join(_tok(x) for x in ts_in) if ts_in else "''")

        # define RHS
        fsts["RHS"] = FST.re(f"{_tok('->')} ($NT|$T)+", fsts)

        # define PB
        fsts["PB"] = FST.re(f"{_tok('^')}|{_tok('(')}|{_tok(')')}", fsts)

        # literal tokens
        LB, RB = _tok("["), _tok("]")
        LC, RC = _tok("{"), _tok("}")
        CARET = _tok("^")
        LP, RP = _tok("("), _tok(")")
        BARLIT = _tok("|")
        EPS = _tok(self.epsilon_symbol)

        def wstring(sym: str) -> str:
            return " $RHS " if seen.get(sym) != 2 else ""

        # LRules / RRules / Center0
        lrules: List[str] = []
        rrules: List[str] = []
        center: List[str] = []
        NOTPB_LIT = _tok("NOTPB")

        for r in self.rules:
            rhs = list(r.rhs)
            rhs_full = f"{_tok('->')} " + " ".join(_tok(x) for x in rhs)

            # LRules
            lr = []
            for i, sym in enumerate(rhs):
                tail = wstring(sym)
                if i == 0:
                    lr.append(f"{LB} {_tok(sym)}{tail} {RB}")
                else:
                    lr.append(f"{LC} {_tok(sym)}{tail} {RC}")
            lr.append(f"{LB} {_tok(r.lhs)} {rhs_full} {RB}")
            lrules.append(" ".join(lr))

            # RRules
            rr = [f"{LB} {_tok(r.lhs)} {rhs_full} {RB}"]
            for i, sym in enumerate(rhs):
                tail = wstring(sym)
                if i + 1 < len(rhs):
                    rr.append(f"{LC} {_tok(sym)}{tail} {RC}")
                else:
                    rr.append(f"{LB} {_tok(sym)}{tail} {RB}")
            rrules.append(" ".join(rr))

            # Center0 (with NOTPB placeholder)
            c: List[str] = []
            for i, sym in enumerate(rhs):
                if i == 0:
                    if seen.get(sym) != 2:
                        c.append(f"{NOTPB_LIT} {LB} {_tok(sym)} $RHS {RB} {LP} {_tok(r.lhs)} {rhs_full} {RP}")
                    else:
                        c.append(f"{LB} {_tok(sym)} {RB} {LP} {_tok(r.lhs)} {rhs_full} {RP}")
                else:
                    if i > 1:
                        c.append(CARET)
                    if seen.get(sym) != 2:
                        c.append(f"{LB} {_tok(sym)} $RHS {RB} {NOTPB_LIT}")
                    else:
                        c.append(f"{LB} {_tok(sym)} {RB}")
            center.append(" ".join(c))

        fsts["LRules"]  = FST.re(" | ".join(f"({x})" for x in lrules), fsts)
        fsts["RRules"]  = FST.re(" | ".join(f"({x})" for x in rrules), fsts)
        fsts["Center0"] = FST.re(" | ".join(f"({x})" for x in center), fsts)

        # redefine Center: [Center0 .o. [NOTPB:[[\PB-NOTPB]*]|\NOTPB]*].l;
        fsts["NOTPB"] = FST.re(_tok("NOTPB"))
        fsts["PBminusNOTPB_star"] = FST.re("(((.-$PB) - $NOTPB)*)", fsts)
        fsts["SubstNOTPB"] = FST.re("(($NOTPB:$PBminusNOTPB_star) | (.-$NOTPB))*", fsts)
        fsts["Center"] = FST.re("$^output($Center0 @ $SubstNOTPB)", fsts)

        # define Mid
        fsts["Mid"] = FST.re(
            f"({LB} (($NT $RHS) | $T) {RB} | {LC} (($NT $RHS) | $T) {RC})*",
            fsts
        )

        # define Filter
        filter_re = (
            f"({LB} $T {RB} $Mid* {LP} $NT $RHS {RP} "
            f"($Mid* {LB} $T {RB} {CARET})* "
            f"($Mid* {LB} $T {RB})? )"
        )
        fsts["Filter"] = FST.re(filter_re, fsts)


        # define LR0
        fsts["LR0"] = FST.re("$^output($Filter @ (.* '':'|' .* '':'|' .*))", {'Filter': fsts["Filter"]})
        # define LR
        notBAR = f"(.-{BARLIT})"
        notBRACKBAR = f"(.-({LB}|{BARLIT}|{RB}))"
        notCURLYBAR = f"(.-({LC}|{BARLIT}|{RC}))"

        LR_left = (
            f"({notBAR}* {BARLIT} {LB} {notBRACKBAR}* {RB} "
            f"({LC} {notCURLYBAR}* {RC})* "
            f"{LB} {notBRACKBAR}* {RB} {BARLIT} {notBAR}*)"
        )
        LR_bad = (
            f"({notBAR}* {BARLIT} $LRules {BARLIT} {notBAR}* {LP} {notBAR}*"
            f" | {notBAR}* {RP} {notBAR}* {BARLIT} $RRules {BARLIT} {notBAR}*)"
        )
        fsts["LR"] = FST.re(f"((({LR_left}) & $LR0) - ({LR_bad}))", fsts)

        # define LR2 ~`[LR,|,0] & Filter & Center;
        # Implement equivalently as:  (Filter & Center) - [LR,|,0]
		
        fsts["LR_hom"] = fsts["LR"].copy_mod().map_labels({'|':''}).epsilon_remove().determinize_as_dfa().minimize()
        # Work on fresh copies to avoid mutating stored Filter/Center via alphabet harmonization.
        Filter2 = FST.re(filter_re, fsts)
        Center2 = fsts["Center"]  # Center does not contain '.' transitions, safe in practice.

        base = Filter2.intersection(Center2).trim().epsilon_remove().determinize_as_dfa().minimize()
        LR2 = base.difference(fsts["LR_hom"]).trim().epsilon_remove().determinize_as_dfa().minimize()
        fsts["LR2"] = LR2

        fsts["Filter"] = FST.re(filter_re, fsts)

        # define Alphabet
        fsts["Alphabet"] = FST.re(
            f"$NT | $T | {LC} | {RC} | {LB} | {RB} | {CARET} | {_tok('->')}",
            fsts
        )
        notLC = f"(.-{LC})"

        # define IT
        piece1 = (
            f"({notLC} | {LC} {_tok('<')} | "
            f"{LC} '' : ({_tok('<#')} $Alphabet*) "
            f"'' : {LP} $NT $RHS '' : {RP} "
            f"'' : ($Alphabet* {_tok('>#')}) {RC} | "
            f"{LC} $T {RC})*"
        )
        piece2 = f"~(.* {_tok('<#')} ( (.-({_tok('<#')}|{_tok('>#')}))* & ~$LR2 ) {_tok('>#')} .*)"
        rw = f"$^rewrite({_tok('<#')}:{_tok('<')} | {_tok('>#')}:{_tok('>')})"
        env_it = {"Alphabet": fsts["Alphabet"], "LR2": fsts["LR2"], "NT": fsts["NT"], "RHS": fsts["RHS"], "T": fsts["T"]}
        fsts["IT"] = FST.re(f"({piece1}) @ ({piece2}) @ ({rw})", env_it)

        # define Gr
        fsts["StartConstraint"] = FST.re(f"(.* {LP} {_tok(self.start_symbol)} .*)")
        fsts["HasOpenCurlyNT"] = FST.re(f"(.* {LC} $NT .*)", fsts)
        chain = "($LR2 & $StartConstraint)"
        for _ in range(levels):
            chain += " @ $IT"
        chain += " @ ~$HasOpenCurlyNT"
        fsts["Gr"] = FST.re(f"$^output({chain})", fsts)
        if len(fsts["Gr"]) == 0: import warnings; warnings.warn("CFG.compile(): grammar automaton Gr has no final states (empty language / too-low levels / inconsistent start symbol).")

        # Purge auxiliary symbols from sigma now that Gr is built.
        if hasattr(fsts["Gr"], "cleanup_sigma"):
            fsts["Gr"] = fsts["Gr"].cleanup_sigma()

        # Optional post-weighting: attach weights to rule-head "->" tokens via contextual rewrites.
        if any(r.weight is not None for r in self.rules):
            fsts["Gr"] = self._weight_grammar(fsts["Gr"])
            fsts["Gr"] = fsts["Gr"].cleanup_sigma()
                
        # Terminals/nonterminals for yield extraction.
        # T is already the grammar terminals acceptor.
        fsts["T_in"] = fsts["T"]
        fsts["NT_in"] = fsts["NT"]

        fsts['yield_inv'] = FST.re("$^invert("
        "$^rewrite(('{' '<'| '>' '}'):'') @ ("                        # injection markers (delete)
        f"'[':'' ($T_in|{EPS}:'') ']':'' | '{{':'' ($T_in|{EPS}:'') '}}':'' | "                # nonterminals in tree, remove markup
        "'[':'' ($NT_in:'')+ '->':'' (($NT_in|$T_in):'')+ ']':'' | "  # bracketed head (delete)
        "'(':'' ($NT_in:'')+ '->':'' (($NT_in|$T_in):'')+ ')':'' | "  # parenthesized head (deleted)
        "'^':'')*"                                                    # caret (delete)
        ")", fsts)
        fsts['yield'] = FST.re("$^invert($yield_inv)", fsts)

        # Store string language separately
        fsts["strings"] = FST.re(
            f"$^output($Gr @ $yield @ $^rewrite({EPS}:''))",
            {'Gr': fsts["Gr"].copy_mod(), 'yield': fsts['yield']},
        )

        # Persist compiled artifacts.
        self._compiled_levels = levels
        if keep_intermediates:
            self.fsts = fsts
        else:
            self.fsts = {"Gr": fsts["Gr"], "T": fsts["T"], "T_in": fsts["T_in"], "NT": fsts["NT"], "yield_inv":fsts["yield_inv"], "yield": fsts["yield"], "strings": fsts["strings"]}
        return self.fsts["Gr"]

    def parse(self, sentence: str) -> FST:
        """Return an acceptor for the parse forest of `sentence` (native H&S encoding)."""
        if "Gr" not in self.fsts:
            self.compile(levels=2)

        env: Dict[str, FST] = dict(self.fsts)

        toks = [t for t in sentence.split() if t]
        env["X"] = FST.re(" ".join(_tok(t) for t in toks)) if toks else FST(('',))

        # Pull back the surface string into the parse-string space via yield^{-1}.
        # C accepts exactly those parse strings whose yield is the input sentence.
        env["C"] = FST.re("$^output($X @ $yield_inv)", env)

        # Keep only grammatical parse strings.
        return FST.re("$C & $Gr", env)
        

    def _weight_grammar(self, gr: FST) -> FST:
        """
        Post-weight an already-compiled *unweighted* Gr by composing contextual rewrites
        that attach weights to the arrow token "->" in rule heads.

        IMPORTANT: we intentionally avoid compiling the full composition through FST.re(...)
        (which runs global weight pushing) because that can be very expensive on larger
        grammars. Instead we:
          1) compile each small contextual rewrite with FST.re($^rewrite(...))
          2) compose using the low-level FST.compose() method
          3) project to the output tape to keep an acceptor

        For each weighted rule LHS -> RHS with weight w:

          rewrite('->' -> '->'<w>) / ( "("|"[" ) LHS _ RHS ( ")"|"]" )
        """
        # Build LB/RB once.
        LB = FST.re("'('|'['")
        RB = FST.re("')'|']'")

        out = gr
        for r in self.rules:
            if r.weight is None:
                continue
            rhs_re = " ".join(_tok(x) for x in r.rhs)
            rw = FST.re(
                f"$^rewrite('->':('->'<{float(r.weight)}>) / $LB {_tok(r.lhs)} _ {rhs_re} $RB)",
                {"LB": LB, "RB": RB},
            )
            # Compose and keep only the output tape (acceptor).
            out = out.copy_mod().compose(rw).project(dim=-1).trim().epsilon_remove()

        return out


    @staticmethod
    def _fmt_seq(seq, *, drop_eps: bool = False, epsilon_symbol: str = "EPSILON") -> str:
        """Format a label sequence as space-separated symbols (foma print-space style)."""
        out = []
        for tok in seq:
            sym = tok[0] if isinstance(tok, tuple) and len(tok) == 1 else str(tok)
            if drop_eps and sym == epsilon_symbol:
                continue
            out.append(sym)
        return " ".join(out)
        
    def parse_pretty(self, sentence: str, *, n: int = 10, drop_eps: bool = False) -> List[str]:
        """Parse `sentence` and return up to `n` parses as strings (native encoding)."""
        fst = self.parse(sentence)
        if len(fst) == 0: return []
        return [self._fmt_seq(seq, drop_eps=drop_eps, epsilon_symbol=self.epsilon_symbol) for _cost, seq in fst.words_nbest(n)]

    def parse_pretty_with_cost(self, sentence: str, *, n: int = 10, drop_eps: bool = False) -> List[Tuple[float, str]]:
        """Like parse_pretty(), but returns (cost, string) for each parse."""
        fst = self.parse(sentence)
        if len(fst) == 0: return []
        return [(float(cost), self._fmt_seq(seq, drop_eps=drop_eps, epsilon_symbol=self.epsilon_symbol)) for cost, seq in fst.words_nbest(n)]
    
    def parse_ptb_with_cost(self, sentence: str, *, n: int = 10, drop_eps: bool = False) -> List[Tuple[float, str]]:
        fst = self.parse(sentence)
        if len(fst) == 0: return []
        return [(float(cost), funky_to_ptb(self._fmt_seq(seq, drop_eps=drop_eps, epsilon_symbol=self.epsilon_symbol))) for cost, seq in fst.words_nbest(n)]

    def parse_ptb(self, sentence: str, *, n: int = 10, drop_eps: bool = False) -> List[str]:
        """Parse `sentence` and return up to `n` PTB-style bracketed trees."""
        return [funky_to_ptb(s) for s in self.parse_pretty(sentence, n=n, drop_eps=drop_eps)]

    def parse_svg(self, sentence: str, *, n: int = 1, style: str = "tree", drop_eps: bool = False):
        """Parse `sentence` and return SVG renderings (one per parse, up to `n`)."""
        return [draw_tree(t, style=style) for t in self.parse_ptb(sentence, n=n, drop_eps=drop_eps)]

    def parse_graphviz(self, sentence: str, *, n: int = 1, style: str = "tree", drop_eps: bool = False):
        """Backwards-compatible alias."""
        return self.parse_svg(sentence, n=n, style=style, drop_eps=drop_eps)


# ----------------------------------------------------------------------
# Backwards-compatible module-level helpers.
# ----------------------------------------------------------------------

def parse_svg(cfg: CFG, sentence: str, *, n: int = 1, style: str = "tree", drop_eps: bool = False):
    return cfg.parse_svg(sentence, n=n, style=style, drop_eps=drop_eps)

def parse_graphviz(cfg: CFG, sentence: str, *, n: int = 1, style: str = "tree", drop_eps: bool = False):
    return cfg.parse_graphviz(sentence, n=n, style=style, drop_eps=drop_eps)

__all__ = ["CFG", "parse_svg", "parse_graphviz"]
