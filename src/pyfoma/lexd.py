"""Lexd → PyFoma compiler

A Python implementation of the Lexd formalism (inspired by Apertium's C++ lexd),
allowing compilation of Lexd grammars into finite-state transducers (FSTs)
using the PyFoma library.

Lexd is a concise, two-level description language that supports:
  - Multi-segment lexicon entries (for reduplication, circumfixation, etc.)
  - Pattern-based concatenation with alignment (:Lex, Lex:, Lex(i))
  - Tag-based selection/filtering ([count], [-mass], |[a,b], ^[x,y])
  - Anonymous lexicons and patterns
  - Quantifiers (?, *, +) and alternation (|)
  - Sieve operators (<, >)
  - Regular expressions in lexicon entries (/.../)

See: https://github.com/apertium/lexd/blob/main/Usage.md

Main entry point:

    compile(grammar: str) -> pyfoma.FST
        Compile a Lexd grammar string into a (minimized) PyFoma FST.

        Args:
            grammar: The full Lexd source code as a string.

        Returns:
            A pyfoma.FST representing the compiled transducer.


Usage example:


from pyfoma import lexd

grammar = r'''
PATTERNS
Prefix? NounStem NounNumber

LEXICON Prefix
ex-
anti-

LEXICON NounStem
cat
dog

LEXICON NounNumber
<sg>:
<pl>:s
'''

myfst = lexd.compile(grammar)

# Generate surface forms
print(list(myfst.generate("cat<pl>")))
# → ['cats']

# Analyze
print(list(myfst.analyze("ex-dogs")))
# → ['ex-dog<pl>']

print(list(myfst.analyze("cats")))
# → ['cat<pl>']

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set
import re, collections

from pyfoma.fst import FST, State, concatenate, union, kleene_star, kleene_plus


# ----------------------------------------
# Helpers
# ----------------------------------------

def epsilon_fst() -> FST:
    return FST(('',))

def empty_fst() -> FST:
    """An empty-language FST with no transitions and no final states.
    Avoids using empty_fst(), which introduces '[' and ']' into the alphabet.
    """
    return FST()

def _normalize_label(label: Tuple[str, ...]) -> Tuple[str, ...]:
    if len(label) == 2 and label[0] == label[1]:
        return (label[0],)
    return label

def union_all(fsts: List[FST]) -> FST:
    out = None
    for f in fsts:
        out = f if out is None else union(out, f)
    return out if out is not None else FST()

def from_tuples(tuples_iter: Iterable[Iterable[Tuple[str, ...]]]) -> FST:
    """Create FST from iterables of iterables of labels (tuples of symbols)."""
    newfst = FST()
    for tpls in tuples_iter:
        currstate = newfst.initialstate
        for raw_label in tpls:
            label = _normalize_label(tuple(raw_label))
            for sym in label:
                if sym and sym not in newfst.alphabet:
                    newfst.alphabet.add(sym)
            targetstate = State()
            newfst.states.add(targetstate)
            currstate.add_transition(targetstate, label, 0.0)
            currstate = targetstate
        newfst.finalstates.add(currstate)
        currstate.finalweight = 0.0
    return newfst

# ----------------------------------------
# Tag selectors (DNF: OR of AND-clauses)
# ----------------------------------------

@dataclass(frozen=True)
class TagSelector:
    clauses: Tuple[Tuple[frozenset[str], frozenset[str]], ...]

    @staticmethod
    def any() -> "TagSelector":
        return TagSelector(((frozenset(), frozenset()),))

    def matches(self, tags: Set[str]) -> bool:
        for must, mustnot in self.clauses:
            if must.issubset(tags) and mustnot.isdisjoint(tags):
                return True
        return False

    def and_selector(self, other: "TagSelector") -> "TagSelector":
        new_clauses = []
        for (m1, n1) in self.clauses:
            for (m2, n2) in other.clauses:
                new_clauses.append((m1 | m2, n1 | n2))
        return TagSelector(tuple(new_clauses))

def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts

def _split_selector_suffix(tok: str) -> tuple[str, str | None]:
    """Split a trailing tag-selector suffix '[...]' from a token, supporting nested brackets.

    Examples:
      'A[count]' -> ('A', '[count]')
      'A[|[a,b]]' -> ('A', '[|[a,b]]')
      '(A B)[^[x,y]]' (handled at expr level, but same logic)
    """
    tok = tok.strip()
    if not tok.endswith(']'):
        return tok, None
    depth = 0
    for i in range(len(tok) - 1, -1, -1):
        c = tok[i]
        if c == ']':
            depth += 1
        elif c == '[':
            depth -= 1
            if depth == 0:
                return tok[:i], tok[i:]
    return tok, None

def parse_tag_selector(raw: str) -> TagSelector:
    raw = raw.strip()
    if not raw:
        return TagSelector.any()

    components = _split_top_level_commas(raw)
    sel = TagSelector.any()

    for comp in components:
        comp = comp.strip()
        if not comp:
            continue

        m = re.fullmatch(r"\|\[(.*)\]", comp)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            comp_sel = TagSelector(tuple((frozenset([it]), frozenset()) for it in items))
            sel = sel.and_selector(comp_sel)
            continue

        m = re.fullmatch(r"\^\[(.*)\]", comp)
        if m:
            items = [x.strip() for x in m.group(1).split(",") if x.strip()]
            clauses = []
            for it in items:
                clauses.append((frozenset([it]), frozenset(set(items) - {it})))
            comp_sel = TagSelector(tuple(clauses))
            sel = sel.and_selector(comp_sel)
            continue

        if comp.startswith("-"):
            comp_sel = TagSelector(((frozenset(), frozenset([comp[1:]])),))
        else:
            comp_sel = TagSelector(((frozenset([comp]), frozenset()),))
        sel = sel.and_selector(comp_sel)

    return sel

# ----------------------------------------
# Lexicon parsing helpers
# ----------------------------------------

_TAG_RE = re.compile(r"\[([^\]]*)\]\s*$")

def _split_tags(s: str) -> Tuple[str, Set[str]]:
    s = s.rstrip()
    m = _TAG_RE.search(s)
    if not m:
        return s, set()
    tagraw = m.group(1)
    base = s[: m.start()].rstrip()
    tags = set(t.strip() for t in tagraw.split(",") if t.strip())
    return base, tags

def _quote_multichar_for_pyfoma_regex(regex: str) -> str:
    """Rewrite lexd-style <...> and {...} tokens into PyFoma multichar symbols for regex compilation.

    PyFoma treats anything inside single quotes as a multichar symbol, so we rewrite:
      <sent>  -> '<sent>'
      {A}     -> '{A}'
    but we do NOT touch content already inside single quotes.
    """
    out = []
    i = 0
    in_q = False
    while i < len(regex):
        ch = regex[i]
        if ch == "'":
            in_q = not in_q
            out.append(ch)
            i += 1
            continue
        if not in_q and ch == "<":
            j = regex.find(">", i + 1)
            if j == -1:
                raise ValueError(f"Unclosed <...> in regex: {regex!r}")
            token = regex[i:j+1]
            out.append("'" + token + "'")
            i = j + 1
            continue
        if not in_q and ch == "{":
            j = regex.find("}", i + 1)
            if j == -1:
                raise ValueError(f"Unclosed {{...}} in regex: {regex!r}")
            token = regex[i:j+1]
            out.append("'" + token + "'")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)

def _split_top_level_colon(regex: str) -> tuple[str, str] | None:
    """Split regex at the first top-level ':' (not inside quotes or parentheses/brackets).

    Returns (left, right) or None if no top-level ':' exists.
    """
    in_q = False
    depth_paren = 0
    depth_brack = 0
    esc = False
    for i, ch in enumerate(regex):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == "'":
            in_q = not in_q
            continue
        if in_q:
            continue
        if ch == "(":
            depth_paren += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            continue
        if ch == "[":
            depth_brack += 1
            continue
        if ch == "]":
            depth_brack = max(0, depth_brack - 1)
            continue
        if ch == ":" and depth_paren == 0 and depth_brack == 0:
            return (regex[:i], regex[i+1:])
    return None

def _wrap_top_level_colon_operands(regex: str) -> str:
    """Ensure correct precedence for a single top-level cross-product operator ':'.

    PyFoma's ':' binds tighter than concatenation. Lexd users typically intend
    the whole expression left of ':' and the whole expression right of ':' to be the operands.

    Notes for PyFoma:
      * A missing left or right operand (e.g. 'a:' or ':b') should be treated as epsilon (''), but
        PyFoma's regex parser does not accept a bare missing operand, so we insert '' explicitly.
      * We only rewrite when there is a *top-level* ':' in the regex.
    """
    split = _split_top_level_colon(regex)
    if split is None:
        return regex
    left, right = split
    left = left.strip()
    right = right.strip()

    if left == "":
        left = "''"
    elif not (left.startswith("(") and left.endswith(")")):
        left = f"({left})"

    if right == "":
        right = "''"
    elif not (right.startswith("(") and right.endswith(")")):
        right = f"({right})"

    return f"{left}:{right}"

def _tokenize_symbols(x: str) -> List[str]:
    x = x.strip()
    out: List[str] = []
    i = 0
    while i < len(x):
        if x[i] == "<":
            j = x.find(">", i + 1)
            if j == -1:
                raise ValueError(f"Unclosed <...> tag in {x!r}")
            out.append(x[i : j + 1])
            i = j + 1
            continue
        if x[i] == "{":
            j = x.find("}", i + 1)
            if j == -1:
                raise ValueError(f"Unclosed {{...}} archisymbol in {x!r}")
            out.append(x[i : j + 1])  # keep braces as part of the multichar symbol, e.g. {A}
            i = j + 1
            continue
        out.append(x[i])
        i += 1
    return out

def _entry_to_labels(lexside: str, surfside: str) -> List[Tuple[str, ...]]:
    L = _tokenize_symbols(lexside)
    R = _tokenize_symbols(surfside)
    n = max(len(L), len(R))
    L += [""] * (n - len(L))
    R += [""] * (n - len(R))
    labels: List[Tuple[str, ...]] = []
    for a, b in zip(L, R):
        labels.append(_normalize_label((a, b)))
    return labels

@dataclass
class LexEntry:
    cols: List[str]
    tags: Set[str]

@dataclass
class LexiconDef:
    name: str
    arity: int = 1
    entries: List[LexEntry] = None

# ----------------------------------------
# Pattern AST
# ----------------------------------------

@dataclass(frozen=True)
class TokRef:
    kind: str
    name: str
    # For kind == 'lex': col is Optional[int] or Optional[tuple[int,int]] (for X(i):X(j) same-lex dual-col)
    # For kind == 'pair': col is tuple[int,int] = (left_col, right_col) and left/right store lexicon names.
    col: Optional[object] = None
    side: str = "both"
    selector: TagSelector = TagSelector.any()
    left: Optional[str] = None
    right: Optional[str] = None

class PatExpr:
    pass

@dataclass
class Seq(PatExpr):
    parts: List[PatExpr]

@dataclass
class Alt(PatExpr):
    alts: List[PatExpr]

@dataclass
class Ref(PatExpr):
    token: TokRef

@dataclass
class Quant(PatExpr):
    expr: PatExpr
    q: str

@dataclass(frozen=True)
class Tagged(PatExpr):
    expr: PatExpr
    selector: TagSelector

@dataclass
class ParsedLexd:
    patterns: Dict[str, PatExpr]
    top_patterns: List[PatExpr]
    lexicons: Dict[str, LexiconDef]
    aliases: Dict[str, str]

# ----------------------------------------
# Pattern tokenizer / parser
# ----------------------------------------

_SEL_SUFFIX_RE = re.compile(r"^(.*?)(\[[^\]]*\])$")

def _tokenize_pattern_line(line: str) -> List[str]:
    s = line.strip()
    out: List[str] = []
    i = 0
    n = len(s)

    def skip_ws(k: int) -> int:
        while k < n and s[k].isspace():
            k += 1
        return k

    def read_balanced_brackets(k: int) -> Tuple[str, int]:
        depth = 0
        j = k
        while j < n:
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    return s[k : j + 1], j + 1
            j += 1
        raise ValueError("Unclosed [...] in pattern line")

    while True:
        i = skip_ws(i)
        if i >= n:
            break
        c = s[i]

        if c in ("|", "?", "*", "+", "<", ">" ):
            # lexd quirk: '|' can be used without surrounding whitespace (e.g. X(1)|Y(1)),
            # and in that case it groups tighter than whitespace concatenation. We encode this
            # as a distinct token so the parser can give it higher precedence.
            if c == "|":
                prev = s[i - 1] if i > 0 else " "
                nxt = s[i + 1] if (i + 1) < n else " "
                out.append("__TIGHTOR__" if (not prev.isspace() and not nxt.isspace()) else "|")
            else:
                out.append(c)
            i += 1
            continue
        if c in ("(", ")"):
            out.append(c)
            i += 1
            continue
        if c == "[":
            bracket, j = read_balanced_brackets(i)
            content = bracket[1:-1].strip()
            # Disambiguation:
            #   * If '[' is preceded by whitespace (or is at start), treat as an anonymous lexicon atom: [ ... ]
            #   * Otherwise (e.g. immediately after ')'), treat as a selector postfix on the previous expression.
            if i == 0 or s[i-1].isspace():
                out.append("[")
                if content:
                    out.append(content)
                out.append("]")
            else:
                out.append(f"__POSTSEL__:{content}")
            i = j
            continue

        j = i
        while j < n and (not s[j].isspace()) and s[j] not in ("|", "?", "*", "+", "<", ">", "[", "]", "(", ")"):
            j += 1
        tok = s[i:j]

        # Special lexd syntax: NAME?(N) means the NAME(N) token is optional.
        # We tokenize it as: NAME(N) followed by '?' so the parser sees a Ref with column,
        # then applies the '?' quantifier to that Ref.
        if j < n and s[j] == "?" and (j + 1) < n and s[j + 1] == "(":
            k = j + 2
            while k < n and s[k].isdigit():
                k += 1
            if k > j + 2 and k < n and s[k] == ")":
                tok = tok + s[j + 1 : k + 1]  # append "(N)" but not the '?'
                j = k + 1
                out.append(tok)
                out.append("?")
                i = j
                continue

        if j < n and s[j] == "(":
            k = j + 1
            while k < n and s[k].isdigit():
                k += 1
            if k > j + 1 and k < n and s[k] == ")":
                tok = tok + s[j : k + 1]
                j = k + 1

        if j < n and s[j] == "[":
            bracket, j2 = read_balanced_brackets(j)
            tok = tok + bracket
            j = j2

        # A ':' immediately after a token can be either:
        #  - a one-sided marker (e.g. Lex:) when followed by whitespace/end/operator, OR
        #  - an internal ':' (e.g. x(1):y(2)) which must remain inside the token.
        if j < n and s[j] == ":":
            # Suffix ':' (one-sided marker)
            nxt = s[j + 1] if (j + 1) < n else ""
            if (j + 1) == n or nxt.isspace() or nxt in ("|", "?", "*", "+", "<", ">", "[", "]", "(", ")"):
                tok = tok + ":"
                j += 1
            else:
                # Internal ':': keep consuming ':' + following segment(s) as part of this token.
                while j < n and s[j] == ":":
                    tok += ":"
                    j += 1
                    k = j
                    while k < n and (not s[k].isspace()) and s[k] not in ("|", "?", "*", "+", "<", ">", "[", "]"):
                        k += 1
                    tok += s[j:k]
                    j = k

        out.append(tok)
        i = j

    return out

def _expand_sieve_line(line: str) -> List[str]:
    """Expand sieve operators < and > into equivalent PATTERN lines."""
    s = line.strip()
    if "<" not in s and ">" not in s:
        return [s]

    def has_toplevel_or(seg: str) -> bool:
        depth_par = 0
        depth_sq = 0
        in_regex = False
        esc = False
        for ch in seg:
            if esc:
                esc = False
                continue
            if ch == "\\":  # escape
                esc = True
                continue
            if in_regex:
                if ch == "/":
                    in_regex = False
                continue
            if ch == "/":
                in_regex = True
                continue
            if ch == "(":
                depth_par += 1
                continue
            if ch == ")":
                depth_par = max(0, depth_par - 1)
                continue
            if ch == "[":
                depth_sq += 1
                continue
            if ch == "]":
                depth_sq = max(0, depth_sq - 1)
                continue
            if ch == "|" and depth_par == 0 and depth_sq == 0:
                return True
        return False

    def protect(seg: str) -> str:
        seg = seg.strip()
        if has_toplevel_or(seg):
            return f"({seg})"
        return seg

    segs: List[str] = []
    ops: List[str] = []

    buf: List[str] = []
    depth_par = 0
    depth_sq = 0
    in_regex = False
    esc = False

    def flush_buf() -> None:
        seg = "".join(buf).strip()
        buf.clear()
        if seg:
            segs.append(seg)

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if esc:
            buf.append(ch)
            esc = False
            i += 1
            continue
        if ch == "\\":  # escape
            buf.append(ch)
            esc = True
            i += 1
            continue
        if in_regex:
            buf.append(ch)
            if ch == "/":
                in_regex = False
            i += 1
            continue
        if ch == "/":
            buf.append(ch)
            in_regex = True
            i += 1
            continue
        if ch == "(":
            depth_par += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth_par = max(0, depth_par - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "[":
            depth_sq += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "]":
            depth_sq = max(0, depth_sq - 1)
            buf.append(ch)
            i += 1
            continue

        # Detect sieve operator at top level, requiring whitespace around.
        if depth_par == 0 and depth_sq == 0 and ch in ("<", ">"):
            prev = s[i - 1] if i > 0 else " "
            nxt = s[i + 1] if (i + 1) < n else " "
            if prev.isspace() and nxt.isspace():
                flush_buf()
                ops.append(ch)
                i += 1
                continue

        buf.append(ch)
        i += 1

    flush_buf()

    if not ops:
        return [s]
    if len(ops) != max(0, len(segs) - 1):
        return [s]

    out_lines: List[str] = []
    k = len(segs)
    for start in range(k):
        if start > 0 and ops[start - 1] == ">":
            continue
        for end in range(start, k):
            if end < k - 1 and ops[end] == "<":
                continue
            pieces = [protect(x) for x in segs[start : end + 1]]
            out_lines.append(" ".join(pieces))
    return out_lines

def _parse_token_ref(tok: str) -> TokRef:
    side = "both"
    if tok.startswith(":"):
        side = "out"
        tok = tok[1:]
    if tok.endswith(":"):
        if side == "both":
            side = "in"
        tok = tok[:-1]

    selector = TagSelector.any()
    base, sel = _split_selector_suffix(tok)
    if sel is not None:
        tok = base
        selector = parse_tag_selector(sel[1:-1])

    # Special lexd syntax: X(1):X(2) binds the same lexicon entry while using different columns
    # on the input/output side. (This is NOT the regex cross-product operator.)
    if ":" in tok:
        left, right = tok.split(":", 1)
        m1 = re.match(r"^(.+?)\((\d+)\)$", left)
        m2 = re.match(r"^(.+?)\((\d+)\)$", right)
        if m1 and m2 and m1.group(1) == m2.group(1):
            name = m1.group(1)
            col_in = int(m1.group(2))
            col_out = int(m2.group(2))
            return TokRef(kind="lex", name=name, col=(col_in, col_out), side="both", selector=selector)

    # Cross-lexicon pairing syntax: x(i):y(j)
    mxy = re.match(r"^(.+?)\((\d+)\):(.+?)\((\d+)\)$", tok)
    if mxy:
        lx, ci, ly, co = mxy.group(1), int(mxy.group(2)), mxy.group(3), int(mxy.group(4))
        return TokRef(kind="pair", name=f"{lx}:{ly}", col=(ci, co), side="both", selector=selector, left=lx, right=ly)

    col = None
    m = re.match(r"^(.+?)\((\d+)\)$", tok)
    if m:
        tok = m.group(1)
        col = int(m.group(2))

    return TokRef(kind="lex", name=tok, col=col, side=side, selector=selector)

def _parse_pattern_expr(tokens: List[str], pos: int = 0) -> Tuple[PatExpr, int]:
    def parse_atom(p: int) -> Tuple[PatExpr, int]:
        if tokens[p] == "(":
            e, p2 = parse_alt(p + 1)
            if p2 >= len(tokens) or tokens[p2] != ")":
                raise ValueError("missing )")
            return e, p2 + 1

        if tokens[p] == "[":
            raw = ""
            if p + 1 < len(tokens) and tokens[p + 1] != "]":
                raw = tokens[p + 1]
                p = p + 1
            if p + 1 >= len(tokens) or tokens[p + 1] != "]":
                raise ValueError("missing ] in anonymous lexicon")
            anon_name = f"__ANONLEX__:{raw.strip()}"
            return Ref(TokRef(kind="anonlex", name=anon_name)), p + 2

        return Ref(_parse_token_ref(tokens[p])), p + 1

    def parse_postfix(p: int) -> Tuple[PatExpr, int]:
        e, p = parse_atom(p)
        while p < len(tokens) and tokens[p] in ("?", "*", "+"):
            e = Quant(e, tokens[p])
            p += 1
        return e, p

    def parse_concat(p: int) -> Tuple[PatExpr, int]:
        parts: List[PatExpr] = []
        while p < len(tokens) and tokens[p] not in (")", "|", "<", ">" ):
            e, p = parse_postfix(p)
            # Higher-precedence OR for '|' with no surrounding whitespace (tokenized as __TIGHTOR__).
            if p < len(tokens) and tokens[p] == "__TIGHTOR__":
                alts = [e]
                while p < len(tokens) and tokens[p] == "__TIGHTOR__":
                    e2, p = parse_postfix(p + 1)
                    alts.append(e2)
                e = Alt(alts)

            # Postfix selector(s) on the expression we just parsed: (...)[selector]
            while p < len(tokens) and tokens[p].startswith("__POSTSEL__:"):
                raw = tokens[p].split(":", 1)[1]
                sel = parse_tag_selector(raw.strip())
                e = Tagged(e, sel)
                p += 1

            parts.append(e)

        if not parts:
            return Seq([]), p
        if len(parts) == 1:
            return parts[0], p
        return Seq(parts), p

    def parse_alt(p: int) -> Tuple[PatExpr, int]:
        e, p = parse_concat(p)
        alts = [e]
        while p < len(tokens) and tokens[p] == "|":
            e2, p = parse_concat(p + 1)
            alts.append(e2)
        if len(alts) == 1:
            return alts[0], p
        return Alt(alts), p

    return parse_alt(pos)

# ----------------------------------------
# Selector distribution
# ----------------------------------------

def _selector_to_atomic_and_list(selector: TagSelector) -> List[TagSelector]:
    if len(selector.clauses) != 1:
        return [selector]
    must, mustnot = selector.clauses[0]
    atoms: List[TagSelector] = []
    for t in sorted(must):
        atoms.append(TagSelector(((frozenset([t]), frozenset()),)))
    for t in sorted(mustnot):
        atoms.append(TagSelector(((frozenset(), frozenset([t])),)))
    return atoms

def _apply_selector_distribution(expr: PatExpr, selector: TagSelector) -> PatExpr:
    if selector == TagSelector.any():
        return expr
    if len(selector.clauses) > 1:
        return Alt([_apply_selector_distribution(expr, TagSelector((cl,))) for cl in selector.clauses])
    out = expr
    for atom in _selector_to_atomic_and_list(selector):
        out = _apply_selector_distribution_single(out, atom)
    return out

def _apply_selector_distribution_single(expr: PatExpr, selector: TagSelector) -> PatExpr:
    must, mustnot = selector.clauses[0]
    if not must and not mustnot:
        return expr

    def apply_to_ref(r: Ref) -> Ref:
        t = r.token
        return Ref(TokRef(
            kind=t.kind, name=t.name, col=t.col, side=t.side,
            selector=t.selector.and_selector(selector)
        ))

    if isinstance(expr, Ref):
        return apply_to_ref(expr)
    if isinstance(expr, Quant):
        return Quant(_apply_selector_distribution_single(expr.expr, selector), expr.q)
    if isinstance(expr, Alt):
        return Alt([_apply_selector_distribution_single(a, selector) for a in expr.alts])

    if isinstance(expr, Seq):
        if mustnot and not must:
            return Seq([_apply_selector_distribution_single(p, selector) for p in expr.parts])
        if must and not mustnot:
            alts: List[PatExpr] = []
            for i in range(len(expr.parts)):
                new_parts: List[PatExpr] = []
                for j, pp in enumerate(expr.parts):
                    new_parts.append(_apply_selector_distribution_single(pp, selector) if j == i else pp)
                alts.append(Seq(new_parts))
            return Alt(alts)
        return Seq([_apply_selector_distribution_single(p, selector) for p in expr.parts])

    return expr

# ----------------------------------------
# parse_lexd
# ----------------------------------------

_SECTION_RE = re.compile(r"^(PATTERNS|PATTERN|LEXICON|ALIAS)\b")

def _strip_inline_comment(s: str) -> str:
    r"""Strip an inline '#' comment, but only when the '#' is not escaped.

    IMPORTANT: we *preserve* backslash escapes in the returned string (e.g. "ya\ ngáí"),
    because later parsing needs to see them to keep escaped spaces inside a single column.
    """
    out = []
    esc = False
    for ch in s:
        if esc:
            # keep the escaped character, and keep the preceding backslash as well
            out.append("\\")
            out.append(ch)
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == "#":
            break
        out.append(ch)
    if esc:
        # trailing backslash: keep it literally
        out.append("\\")
    return "".join(out).rstrip()

def _expand_optional_segmented_lexicons(line: str) -> List[str]:
    """Expand lexd syntax NAME?(N) meaning the whole lexicon NAME is optional in this PATTERNS line."""
    rx = re.compile(r'(:?)([A-Za-z_][\w\-]*)\?\((\d+)\)(:?)')
    bases = sorted({m.group(2) for m in rx.finditer(line)})
    if not bases:
        return [line]
    expanded = [line]
    for base in bases:
        new_expanded = []
        def incl(m: re.Match) -> str:
            pre, name, idx, post = m.group(1), m.group(2), m.group(3), m.group(4)
            if name != base:
                return m.group(0)
            return f"{pre}{name}({idx}){post}"
        def excl(m: re.Match) -> str:
            if m.group(2) != base:
                return m.group(0)
            return ""
        for ln in expanded:
            inc = " ".join(rx.sub(incl, ln).split())
            exc = " ".join(rx.sub(excl, ln).split())
            if inc:
                new_expanded.append(inc)
            if exc:
                new_expanded.append(exc)
        expanded = new_expanded
    # dedupe preserve order
    out=[]
    seen=set()
    for ln in expanded:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out

def _split_escaped_fields(line: str) -> List[str]:
    r"""Split a line into whitespace-separated fields, honoring backslash escapes.

    Example: 'ya\ ngáí <tag>:' -> ['ya ngáí', '<tag>:']
    Backslash escapes the next character (space, '#', backslash, etc.).
    """
    fields: List[str] = []
    buf: List[str] = []
    esc = False
    in_ws = True
    for ch in line:
        if esc:
            buf.append(ch)
            esc = False
            in_ws = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch.isspace():
            if not in_ws:
                fields.append("".join(buf))
                buf = []
                in_ws = True
            continue
        buf.append(ch)
        in_ws = False
    if esc:
        buf.append("\\")
    if buf:
        fields.append("".join(buf))
    return fields

def _parse_line_to_exprs(line: str, *, for_patterns_section: bool) -> List[PatExpr]:
    exprs: List[PatExpr] = []
    for ln in _expand_optional_segmented_lexicons(line) if for_patterns_section else [line]:
        for ln2 in _expand_sieve_line(ln):
            toks = _tokenize_pattern_line(ln2)
            expr, p = _parse_pattern_expr(toks, 0)
            if p != len(toks):
                sect = "PATTERNS" if for_patterns_section else "PATTERN"
                raise ValueError(f"Could not parse full {sect} line: {ln!r}")
            exprs.append(expr)
    return exprs

def parse_lexd(lexdstring: str) -> ParsedLexd:
    lines = lexdstring.splitlines()
    mode = None
    curr_name: Optional[str] = None
    curr_block_default_tags: Set[str] = set()

    patterns: Dict[str, PatExpr] = {}
    top_patterns: List[PatExpr] = []
    lexicons: Dict[str, LexiconDef] = {}
    aliases: Dict[str, str] = {}

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        line = _strip_inline_comment(line)
        if not line or line.startswith("#"):
            continue

        m = _SECTION_RE.match(line)
        if m:
            head = m.group(1)
            if head == "PATTERNS":
                mode = "PATTERNS"
                curr_name = None
                continue
            if head == "PATTERN":
                mode = "PATTERN"
                curr_name = line.split()[1]
                patterns[curr_name] = Seq([])
                continue
            if head == "LEXICON":
                mode = "LEXICON"
                rest_raw = line.split(None, 1)[1]

                # Lexicon definition tags may appear as:
                #   LEXICON A[x]
                # or with side-specific defaults:
                #   LEXICON B[x]:[y]
                # These tags are defaults for the block (not emitted as symbols).
                out_tags: List[str] = []
                if ":[" in rest_raw:
                    idxc = rest_raw.find(":[")
                    left_raw = rest_raw[:idxc]
                    right_raw = rest_raw[idxc + 1:]  # starts with '[...]'
                    _, out_tags = _split_tags(right_raw)
                    rest = left_raw
                else:
                    rest = rest_raw

                name_part, default_tags = _split_tags(rest)
                default_tags = list(set(default_tags) | set(out_tags))

                arity = 1
                m2 = re.match(r"^(.+?)\((\d+)\)$", name_part)
                if m2:
                    name_part = m2.group(1)
                    arity = int(m2.group(2))

                if name_part not in lexicons:
                    lexicons[name_part] = LexiconDef(name=name_part, arity=arity, entries=[])
                else:
                    if lexicons[name_part].arity != arity:
                        raise ValueError(f"Lexicon {name_part} arity mismatch across blocks.")

                curr_name = name_part
                curr_block_default_tags = set(default_tags)
                continue
            if head == "ALIAS":
                _, src, dst = line.split()
                aliases[dst] = src
                continue

        if mode == "PATTERNS":
            top_patterns.extend(_parse_line_to_exprs(line, for_patterns_section=True))
            continue

        if mode == "PATTERN":
            exprs = _parse_line_to_exprs(line, for_patterns_section=False)
            expr = exprs[0] if len(exprs) == 1 else Alt(exprs)
            prev = patterns[curr_name]
            if isinstance(prev, Seq) and prev.parts == []:
                patterns[curr_name] = expr
            else:
                if isinstance(prev, Alt):
                    patterns[curr_name] = Alt(prev.alts + [expr])
                else:
                    patterns[curr_name] = Alt([prev, expr])
            continue

        if mode == "LEXICON":
            base, tags = _split_tags(line)
            lex = lexicons[curr_name]

            merged = set(curr_block_default_tags)
            merged |= {t for t in tags if not t.startswith("-")}
            merged -= {t[1:] for t in tags if t.startswith("-")}

            cols = _split_escaped_fields(base) if base else []
            # Lexicon-side tags: tags can appear attached to individual columns like "a[tag]  b".
            # These tags should NOT be part of the symbol string; they only constrain combinations.
            if cols:
                new_cols = []
                for c in cols:
                    c_base, c_tags = _split_tags(c)
                    new_cols.append(c_base)
                    merged |= {t for t in c_tags if not t.startswith("-")}
                    merged -= {t[1:] for t in c_tags if t.startswith("-")}
                cols = new_cols
            if lex.arity != 1 and cols and len(cols) != lex.arity:
                raise ValueError(
                    f"Lexicon {lex.name} expects {lex.arity} columns, got {len(cols)} in {line!r}"
                )
            if lex.arity == 1 and cols:
                cols = [" ".join(cols)]
            lex.entries.append(LexEntry(cols=cols, tags=merged))
            continue

        raise ValueError(f"Line outside a section: {line!r}")

    return ParsedLexd(patterns=patterns, top_patterns=top_patterns, lexicons=lexicons, aliases=aliases)

# ----------------------------------------
# Lexicon compilation
# ----------------------------------------

def _compile_lexicon_entry_variant(
    lex: LexiconDef,
    entry: LexEntry,
    col: Optional[int],
    side: str,
) -> FST:
    if lex.arity == 1:
        content = entry.cols[0] if entry.cols else ""
    else:
        if col is None:
            content = "".join(entry.cols)
        else:
            if not (1 <= col <= lex.arity):
                raise ValueError(f"Column {col} out of range for {lex.name}({lex.arity})")
            content = entry.cols[col - 1]
    content = content.strip()

    if content.startswith("/") and content.endswith("/"):
        regex = content[1:-1]
        # Rewrite lexd multichar tokens (<...>, {...}) into PyFoma-quoted multichar symbols,
        # and fix precedence around a top-level ':' (cross-product binds tighter than concat).
        regex = _quote_multichar_for_pyfoma_regex(regex)
        regex = _wrap_top_level_colon_operands(regex)
        return FST.re(regex)

    if ":" in content:
        lexside, surfside = content.split(":", 1)
    else:
        lexside, surfside = content, content
    lexside = lexside.strip()
    surfside = surfside.strip()

    if side == "both":
        labels = _entry_to_labels(lexside, surfside)
    elif side == "out":
        use = surfside if surfside != "" else lexside
        labels = [_normalize_label(("", s)) for s in _tokenize_symbols(use)]
    elif side == "in":
        labels = [_normalize_label((l, "")) for l in _tokenize_symbols(lexside)]
    else:
        raise ValueError(side)

    fst = from_tuples([tuple(labels)])
    try:
        return fst.determinize().minimize_as_dfa()
    except Exception:
        return fst.determinize()

def _compile_lexicon_variant(
    lex: LexiconDef,
    col: Optional[int],
    side: str,
    selector: TagSelector,
) -> FST:
    paths: List[Tuple[Tuple[str, ...], ...]] = []
    regex_union: Optional[FST] = None

    # Support lexd syntax X(i):X(j) where a single lexicon entry is used,
    # but its (i)th column is placed on the input side and its (j)th column on the output side.
    # We encode this by compiling a temporary 1-column entry of the form "L:R" and compiling it normally.
    if isinstance(col, tuple):
        col_in, col_out = col
        if not (1 <= col_in <= lex.arity) or not (1 <= col_out <= lex.arity):
            raise ValueError(f"Column pair {col!r} out of range for {lex.name}({lex.arity})")
        tmp_lex = LexiconDef(name=lex.name, arity=1, entries=[])
        variants: List[FST] = []
        for e in lex.entries:
            if not selector.matches(e.tags):
                continue
            left = e.cols[col_in - 1] if col_in - 1 < len(e.cols) else ""
            right = e.cols[col_out - 1] if col_out - 1 < len(e.cols) else ""
            tmp_entry = LexEntry(cols=[f"{left}:{right}"], tags=set(e.tags))
            variants.append(_compile_lexicon_entry_variant(tmp_lex, tmp_entry, col=1, side="both"))
        f = union_all(variants) if variants else FST()
        return f.determinize().minimize_as_dfa()

    for e in lex.entries:
        if not selector.matches(e.tags):
            continue

        if lex.arity == 1:
            content = e.cols[0] if e.cols else ""
        else:
            if col is None:
                content = "".join(e.cols)
            else:
                if not (1 <= col <= lex.arity):
                    raise ValueError(f"Column {col} out of range for {lex.name}({lex.arity})")
                content = e.cols[col - 1]
        content = content.strip()

        if content.startswith("/") and content.endswith("/"):
            rf = _compile_lexicon_entry_variant(lex, e, col, side)
            regex_union = rf if regex_union is None else union(regex_union, rf)
            continue

        if ":" in content:
            lexside, surfside = content.split(":", 1)
        else:
            lexside, surfside = content, content

        lexside = lexside.strip()
        surfside = surfside.strip()

        if side == "both":
            labels = _entry_to_labels(lexside, surfside)
        elif side == "out":
            use = surfside if surfside != "" else lexside
            labels = [_normalize_label(("", s)) for s in _tokenize_symbols(use)]
        elif side == "in":
            labels = [_normalize_label((l, "")) for l in _tokenize_symbols(lexside)]
        else:
            raise ValueError(side)

        paths.append(tuple(labels))

    outfst: Optional[FST] = None
    if paths:
        outfst = from_tuples(paths)
    if regex_union is not None:
        outfst = regex_union if outfst is None else union(outfst, regex_union)

    if outfst is None:
        return empty_fst()
    try:
        return outfst.determinize().minimize_as_dfa()
    except Exception:
        return outfst.determinize()


# ----------------------------------------
# Compilation
# ----------------------------------------

def compile(grammar: str) -> FST:
    """Compile a lexd grammar and return a pyfoma FST."""
    return compile_lexd(parse_lexd(grammar))


def compile_lexd(parsed: ParsedLexd) -> FST:
    def resolve_name(name: str) -> str:
        return parsed.aliases.get(name, name)

    anon_counter = 0
    anon_map: Dict[str, str] = {}
    lex_cache: Dict[Tuple[str, Optional[int], str, Tuple], FST] = {}
    pat_cache: Dict[Tuple[str, Tuple], FST] = {}

    def compile_tok(tok: TokRef) -> FST:
        nonlocal anon_counter

        if tok.kind == "anonlex":
            raw = tok.name.split(":", 1)[1]
            if tok.name not in anon_map:
                anon_counter += 1
                anon_name = f"__anonlex_{anon_counter}"
                base, tags = _split_tags(raw)
                parsed.lexicons[anon_name] = LexiconDef(
                    name=anon_name,
                    arity=1,
                    entries=[LexEntry(cols=[base], tags=set(tags))],
                )
                anon_map[tok.name] = anon_name
            name = anon_map[tok.name]
        else:
            name = tok.name

        # Cross-lexicon paired reference: x(i):y(j)
        # Handled in compile_seq_aligned so we can bind the paired row index across multiple occurrences.
        if tok.kind == "pair":
            raise RuntimeError("Internal: pair tokens must be compiled in compile_seq_aligned()")

        if name in parsed.patterns and resolve_name(name) not in parsed.lexicons:
            key = (name, tok.selector.clauses)
            if key in pat_cache:
                return pat_cache[key]
            expr = parsed.patterns[name]
            expr = _apply_selector_distribution(expr, tok.selector)
            f = compile_expr(expr, env={})
            pat_cache[key] = f
            return f

        base = resolve_name(name)
        if base not in parsed.lexicons:
            # Internal: some anon-pattern expansions use a __POSTSEL__: prefix for temporary atoms.
            # Treat these as anonymous literal patterns (identity transducer).
            # See test-anonpat-modifier for the type of pattern where this is needed
            if name.startswith("__POSTSEL__:"): 
                lit = name[len("__POSTSEL__:"):]
                syms = _tokenize_symbols(lit)
                labels = [(s, s) for s in syms]
                fst = from_tuples([labels])
                return fst.determinize().minimize_as_dfa()
            
            raise KeyError(f"Unknown lexicon/pattern: {name}")
        cache_key = (base, tok.col, tok.side, tok.selector.clauses)
        if cache_key in lex_cache:
            return lex_cache[cache_key]
        f = _compile_lexicon_variant(parsed.lexicons[base], tok.col, tok.side, tok.selector)
        lex_cache[cache_key] = f
        return f

    def should_bind(lexdef: LexiconDef, tok: TokRef, env: Dict[str, int], base: str) -> bool:
        # Bind if:
        #  - explicitly column-referenced (tok.col set), OR
        #  - this lexicon repeats in the current sequence scope (__FORCE_BIND__), OR
        #  - one-sided binding is in effect (tok.side != 'both')
        force = env.get("__FORCE_BIND__", set())
        if tok.name in force:
            return True
        if tok.col is not None:
            return True
        if tok.side != "both":
            return True
        # multi-column lexicons are bound by construction
        return lexdef.arity > 1

    def compile_seq_aligned(parts: List[PatExpr], env: Dict[str, int]) -> FST:
        # Binding: if a lexicon name appears multiple times in the *current* sequence scope,
        # its choice must be coherent across those occurrences.
        if "__FORCE_BIND__" not in env:
            counts = collections.Counter()
            for e in parts:
                if isinstance(e, Ref) and isinstance(e.token, TokRef):
                    t = e.token
                    if t.kind == "lex":
                        counts[t.name] += 1
                    elif t.kind == "pair" and t.left and t.right:
                        counts[("pair", t.left, t.right)] += 1
            env["__FORCE_BIND__"] = {k for k, c in counts.items() if c > 1}

        if not parts:
            return epsilon_fst()

        head, *tail = parts

        if isinstance(head, Ref):
            tok = head.token
            if tok.kind in ("lex", "anonlex"):
                base = resolve_name(tok.name)
                if base in parsed.lexicons:
                    lexdef = parsed.lexicons[base]
                    if should_bind(lexdef, tok, env, base):
                        if base in env:
                            entry = lexdef.entries[env[base]]
                            if not tok.selector.matches(entry.tags):
                                return empty_fst()
                            fst_head = _compile_lexicon_entry_variant(lexdef, entry, tok.col, tok.side)
                            return concatenate(fst_head, compile_seq_aligned(tail, env))

                        out = None
                        for idx, entry in enumerate(lexdef.entries):
                            if not tok.selector.matches(entry.tags):
                                continue
                            fst_head = _compile_lexicon_entry_variant(lexdef, entry, tok.col, tok.side)
                            env2 = dict(env)
                            env2[base] = idx
                            path = concatenate(fst_head, compile_seq_aligned(tail, env2))
                            out = path if out is None else union(out, path)
                        return out if out is not None else empty_fst()

        # Special: paired token x(i):y(j) binds a row index across occurrences.
        if isinstance(head, Ref) and isinstance(head.token, TokRef) and head.token.kind == "pair":
            tok = head.token
            if not tok.left or not tok.right:
                raise ValueError(f"Malformed pair token: {tok}")
            lx = resolve_name(tok.left)
            ly = resolve_name(tok.right)
            lex_x = parsed.lexicons.get(lx)
            lex_y = parsed.lexicons.get(ly)
            if lex_x is None or lex_y is None:
                raise KeyError(f"Unknown lexicon in pair: {tok.left}:{tok.right}")

            ci, co = tok.col  # type: ignore[misc]
            if not (isinstance(ci, int) and isinstance(co, int)):
                raise ValueError(f"Bad pair columns in {tok}")
            if not (1 <= ci <= lex_x.arity) or not (1 <= co <= lex_y.arity):
                raise ValueError(
                    f"Pair columns out of range in {tok}: {tok.left}({lex_x.arity}) {tok.right}({lex_y.arity})"
                )

            pair_key = ("__PAIR__", tok.left, tok.right)
            # If already bound, compile only that paired row.
            if pair_key in env:
                k = env[pair_key]
                ex = lex_x.entries[k]
                ey = lex_y.entries[k]
                tag_union = set(ex.tags) | set(ey.tags)
                if tok.selector and not tok.selector.matches(tag_union):
                    return empty_fst()
                left_str = ex.cols[ci - 1] if ci - 1 < len(ex.cols) else ""
                right_str = ey.cols[co - 1] if co - 1 < len(ey.cols) else ""
                left_syms = _tokenize_symbols(left_str)
                right_syms = _tokenize_symbols(right_str)
                L = max(len(left_syms), len(right_syms))
                labels: List[Tuple[str, str]] = []
                for i in range(L):
                    a = left_syms[i] if i < len(left_syms) else ""
                    b = right_syms[i] if i < len(right_syms) else ""
                    labels.append((a, b))
                fst_head = from_tuples([labels])
                return concatenate(fst_head, compile_seq_aligned(tail, env))

            # Otherwise, branch over paired rows (zip semantics).
            out = None
            max_k = min(len(lex_x.entries), len(lex_y.entries))
            for k in range(max_k):
                ex = lex_x.entries[k]
                ey = lex_y.entries[k]
                tag_union = set(ex.tags) | set(ey.tags)
                if tok.selector and not tok.selector.matches(tag_union):
                    continue
                left_str = ex.cols[ci - 1] if ci - 1 < len(ex.cols) else ""
                right_str = ey.cols[co - 1] if co - 1 < len(ey.cols) else ""
                left_syms = _tokenize_symbols(left_str)
                right_syms = _tokenize_symbols(right_str)
                L = max(len(left_syms), len(right_syms))
                labels: List[Tuple[str, str]] = []
                for i in range(L):
                    a = left_syms[i] if i < len(left_syms) else ""
                    b = right_syms[i] if i < len(right_syms) else ""
                    labels.append((a, b))
                fst_head = from_tuples([labels])
                env2 = dict(env)
                env2[pair_key] = k
                path = concatenate(fst_head, compile_seq_aligned(tail, env2))
                out = path if out is None else union(out, path)

            return out if out is not None else empty_fst()

        fst_head = compile_expr(head, env)
        return concatenate(fst_head, compile_seq_aligned(tail, env))

    def compile_expr(expr: PatExpr, env: Dict[str, int]) -> FST:
        if isinstance(expr, Ref):
            return compile_tok(expr.token)

        if isinstance(expr, Seq):
            return compile_seq_aligned(expr.parts, env)

        if isinstance(expr, Alt):
            out = None
            for a in expr.alts:
                af = compile_expr(a, dict(env))
                out = af if out is None else union(out, af)
            return out if out is not None else empty_fst()

        if isinstance(expr, Quant):
            base = compile_expr(expr.expr, env={})
            if expr.q == "?":
                return union(epsilon_fst(), base)
            if expr.q == "*":
                return kleene_star(base)
            if expr.q == "+":
                return kleene_plus(base)
            raise ValueError(expr.q)

        if isinstance(expr, Tagged):
            distributed = _apply_selector_distribution(expr.expr, expr.selector)
            return compile_expr(distributed, env)

        raise ValueError(f"Unhandled node: {expr!r}")

    outfst = None
    for expr in parsed.top_patterns:
        f = compile_expr(expr, env={})
        outfst = f if outfst is None else union(outfst, f)

    if outfst is None:
        outfst = empty_fst()

    try:
        outfst = outfst.determinize().minimize_as_dfa()
    except Exception:
        outfst = outfst.determinize()
    return outfst
