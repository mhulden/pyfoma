# cfg.py

"""Shared grammar parsing and tree rendering utilities for cfg_parse and cfg_approx."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re as pyre

from pyfoma import FST, State

_NUM_RE = pyre.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _tok(sym: str) -> str:
    """Token for PyFoma regex. One-char alnum => as-is; else single-quoted literal symbol."""
    if len(sym) == 1 and sym.isalnum():
        return sym
    sym = sym.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{sym}'"

@dataclass(frozen=True)
class Rule:
    lhs: str
    rhs: Tuple[str, ...]
    weight: Optional[float] = None


def parse_cfg(grammar: str) -> Tuple[List[Rule], str]:
    """
    Parse a CFG from a string.

    Syntax (weights optional):
      LHS -> RHS1 RHS2 ...  [weight]
      LHS -> RHSalt ... [weight] | RHSalt2 ... [weight] | ...

    The optional weight is a trailing numeric token, either as 1.23 or <1.23>.
    """
    rules: List[Rule] = []
    start: Optional[str] = None

    num_re = pyre.compile(r"^<\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*>$|^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$")

    for line in grammar.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" not in line:
            raise ValueError(f"Bad rule line (missing '->'): {line!r}")

        lhs, rhs_all = [x.strip() for x in line.split("->", 1)]
        if not lhs:
            raise ValueError(f"Empty LHS in rule line: {line!r}")
        if start is None:
            start = lhs

        # Split alternations
        for alt in rhs_all.split("|"):
            alt = alt.strip()
            if not alt:
                continue
            toks = [t for t in alt.split() if t]
            weight: Optional[float] = None
            if toks:
                m = num_re.match(toks[-1])
                if m:
                    weight = float(m.group(1) or m.group(2))
                    toks = toks[:-1]
            if not toks:
                raise ValueError(f"Empty RHS in rule: {line!r}")
            rules.append(Rule(lhs=lhs, rhs=tuple(toks), weight=weight))

    if start is None:
        raise ValueError("No rules found in grammar.")
    return rules, start


# ====================================
# Parse-string visualization utilities
# ====================================

@dataclass
class _Tree:
    label: str
    children: List["_Tree | str"]

@dataclass
class _Pending:
    label: str
    remaining: List[str]
    children: List["_Tree | str"]

def _ptb_escape(tok: str) -> str:
    # Minimal PTB-friendly escaping (optional)
    return tok

def funky_to_ptb(s: str) -> str:
    class Node:
        __slots__ = ("label", "children")
        def __init__(self, label, children): self.label, self.children = label, children

    def toks(txt: str):
        out, i, n = [], 0, len(txt)
        while i < n:
            if txt[i].isspace(): i += 1; continue
            c = txt[i]
            if c in "[({":
                close = {"[": "]", "(": ")", "{": "}"}[c]
                if c == "{":  # allow nested { ... }
                    depth, j = 0, i
                    while j < n:
                        if txt[j] == "{": depth += 1
                        elif txt[j] == "}":
                            depth -= 1
                            if depth == 0: j += 1; break
                        j += 1
                    if depth: raise ValueError("unmatched '{'")
                    out.append(("brace", txt[i+1:j-1].strip()))
                    i = j
                else:
                    j = txt.find(close, i+1)
                    if j < 0: raise ValueError(f"unmatched '{c}'")
                    out.append(("paren" if c == "(" else "brack", txt[i+1:j].strip()))
                    i = j + 1
            else:
                j = i
                while j < n and not txt[j].isspace(): j += 1
                out.append(("bare", txt[i:j]))
                i = j
        return out

    def rule(x: str):
        if "->" not in x: return x.strip(), None
        a, b = x.split("->", 1)
        return a.strip(), [t for t in b.strip().split() if t]

    def parse(txt: str):
        T = toks(txt)
        r = next((i for i, (t, _) in enumerate(T) if t == "paren"), None)
        if r is None: raise ValueError("no ( ... ) root found")

        left, right = T[:r], T[r+1:]
        root, rhs = rule(T[r][1])
        if not rhs: raise ValueError("root must be A -> ...")

        def brace(inner: str):
            inner = inner.strip()
            if inner.startswith("^"): inner = inner[1:].strip()
            if "<" in inner and ">" in inner:
                return parse(inner[inner.find("<")+1:inner.rfind(">")].strip())
            return inner

        # left (postorder): stack-reduce
        st = []
        for typ, inner in left:
            if typ == "brace": st.append(brace(inner))
            elif typ == "brack":
                lab, r = rule(inner)
                if r is None: st.append(lab)
                else:
                    k = len(r); kids = st[-k:]; del st[-k:]
                    st.append(Node(lab, kids))
            else:
                st.append(inner)
        left_sub = st[-1] if st else None

        def strip_hat(typ, inner):
            inner = inner.strip()
            if inner.startswith("^"): inner = inner[1:].strip()
            return typ, inner

        def skip_hats(i):
            while i < len(right) and right[i][0] == "bare" and right[i][1].strip() == "^":
                i += 1
            return i

        def side(i):
            i = skip_hats(i)
            typ, inner = strip_hat(*right[i])
            if typ == "brace": return brace(inner), i + 1
            if typ == "brack":
                lab, r = rule(inner)
                if r is None: return lab, i + 1
                return spine(i)
            return inner, i + 1

        def spine(i):
            i = skip_hats(i)
            typ, inner = strip_hat(*right[i])
            if typ == "brace": return brace(inner), i + 1
            if typ != "brack": raise ValueError("expected [ ... ] on spine")
            lab, r = rule(inner)
            if r is None: return lab, i + 1
            kids, j = [], i + 1
            for _ in r[:-1]:
                c, j = side(j); kids.append(c)
            c, j = spine(j); kids.append(c)
            return Node(lab, kids), j

        j, rights = 0, []
        for _ in rhs[1:]:
            j = skip_hats(j)
            c, j = spine(j)
            rights.append(c)

        return Node(root, ([left_sub] if left_sub is not None else []) + rights)

    def ptb(x):
        if isinstance(x, str): return x
        return f"({x.label} {' '.join(ptb(c) for c in x.children)})"

    return ptb(parse(s))

def draw_tree(ptb_str, style='tree', node_width=20, row_height=30):
    """
    Render a PTB-style bracketed tree string as SVG.

    Returns an IPython.display.SVG object when available; otherwise returns the SVG XML string.
    """
    try:
        from IPython.display import SVG as _SVG  # type: ignore
    except Exception:
        _SVG = None

    # 1. Tokenize and Build Tree
    tokens = pyre.findall(r'\(|\)|[^\s()]+', ptb_str)

    def build_tree():
        if not tokens:
            return None
        token = tokens.pop(0)
        if token == '(':
            label = tokens.pop(0)
            children = []
            while tokens and tokens[0] != ')':
                children.append(build_tree())
            if tokens:
                tokens.pop(0)
            return {'label': label, 'children': children}
        return {'label': token, 'children': []}

    tree = build_tree()
    if tree is None:
        raise ValueError("Empty PTB string.")

    # 2. Layout Logic
    memo = []

    def layout(node, x_offset, depth):
        # Calculate width based on label size + small padding
        label_w = len(node['label']) * 9

        if not node['children']:
            x = x_offset + max(node_width, label_w) / 2
            memo.append({'label': node['label'], 'x': x, 'y': depth * row_height, 'kids': [], 'term': True})
            return max(node_width, label_w)

        child_x = x_offset
        kid_centers = []
        for child in node['children']:
            w = layout(child, child_x, depth + 1)
            kid_centers.append(child_x + w / 2)
            # Tighter sibling spacing: only add width of the child
            child_x += w

        width = max(child_x - x_offset, label_w)
        current_x = x_offset + width / 2
        memo.append({'label': node['label'], 'x': current_x, 'y': depth * row_height, 'kids': kid_centers, 'term': False})
        return width

    total_width = layout(tree, 0, 1)
    max_y = max(m['y'] for m in memo) + row_height

    # 3. Render SVG
    svg = [f'<svg width="{total_width + 30}" height="{max_y}" xmlns="http://www.w3.org/2000/svg">']
    svg.append('<style>text{font:12px sans-serif;text-anchor:middle;fill:#2c3e50;} line{stroke:#95a5a6;stroke-width:1;}</style>')

    for m in memo:
        # Draw lines to children
        for k_x in m['kids']:
            svg.append(f'<line x1="{m["x"]}" y1="{m["y"]+5}" x2="{k_x}" y2="{m["y"] + row_height - 15}" />')

        # Draw Box if requested
        if style == 'boxes':
            rw, rh = len(m['label']) * 8 + 8, 18
            svg.append(f'<rect x="{m["x"]-rw/2}" y="{m["y"]-12}" width="{rw}" height="{rh}" fill="white" stroke="#34495e" rx="2"/>')

        # Draw Text
        weight = "bold" if not m['term'] else "normal"
        svg.append(f'<text x="{m["x"]}" y="{m["y"]}" font-weight="{weight}">{m["label"]}</text>')

    svg.append('</svg>')
    svg_xml = "\n".join(svg)
    return _SVG(svg_xml) if _SVG is not None else svg_xml
