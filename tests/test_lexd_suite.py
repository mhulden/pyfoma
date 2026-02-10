#!/usr/bin/env python3
"""
test_lexd_suite.py

Regression test suite for Lexd implementation:
- 7 hand-written tests from lexd usage documentation 
- +36 feature tests embedded from the reference C++ test bundle (self-contained here)

Usage documentation tests from:
https://github.com/apertium/lexd/blob/main/Usage.md

Feature tests from:
https://github.com/apertium/lexd/tree/main/tests/feature

Usage:
  python test_lexd_suite.py
  python test_lexd_suite.py --only 3
  python test_lexd_suite.py --only "cpp:test-alt"
  python test_lexd_suite.py --list
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Union
import argparse, sys

from pyfoma import lexd


# -------------------------
# Test definitions
# -------------------------

@dataclass
class PairTest:
    name: str
    grammar: str
    # One per line, either "A:B" or "A/B" (we infer which side is the input by probing fst.apply).
    expected_lines: List[str]
    forbidden_alphabet_symbols: Optional[Set[str]] = None


@dataclass
class AcceptRejectTest:
    name: str
    grammar: str
    cases: Dict[str, bool]
    require_identity_outputs: bool = True


@dataclass
class GoldStringsTest:
    """Compare the transducer language serialized in foma-like 'string' form.

    The gold files in the C++ suite are lists of strings where each path is serialized by concatenating
    per-arc labels:
      - identity pair (a,a) is printed as 'a'
      - (a,ε) is printed as 'a:'
      - (ε,b) is printed as ':b'
      - (a,b) is printed as 'a:b'
    with multichar symbols kept intact (e.g. '<sent>').
    """
    name: str
    grammar: str
    gold_strings: List[str]
    max_depth: Optional[int] = None
    max_strings: int = 200000


Test = Union[PairTest, AcceptRejectTest, GoldStringsTest]


# -------------------------
# Hand tests (7)
# -------------------------

HAND_TESTS: List[Test] = [
    PairTest(
        name="Test 1",
        grammar=r"""PATTERNS
VerbRoot VerbInfl

LEXICON VerbRoot
sing
walk
dance

LEXICON VerbInfl
<v><pres>:
<v><pres><p3><sg>:s
""",
        expected_lines=[
            "sing<v><pres>:sing",
            "sing<v><pres><p3><sg>:sings",
            "walk<v><pres>:walk",
            "walk<v><pres><p3><sg>:walks",
            "dance<v><pres>:dance",
            "dance<v><pres><p3><sg>:dances",
        ],
    ),
    PairTest(
        name="Test 2",
        grammar=r"""PATTERNS
:VerbInfl VerbRoot VerbInfl:
:VerbInfl :VerbRoot VerbRoot VerbInfl: Redup

LEXICON VerbRoot
bloop
vroom

LEXICON VerbInfl
<v><pres>:en

LEXICON Redup
<redup>:
""",
        expected_lines=[
            "enbloop/bloop<v><pres>",
            "envroom/vroom<v><pres>",
            "enbloopbloop/bloop<v><pres><redup>",
            "envroomvroom/vroom<v><pres><redup>",
        ],
    ),
    PairTest(
        name="Test 3",
        grammar=r"""PATTERNS
C(1) :V(1) C(2) :V(2) C(3) V(2):

LEXICON C(3)
sh m r
y sh v

LEXICON V(2)
:a <v><p3><sg>:a
:o <v><pprs>:e
""",
        expected_lines=[
            "shamar/shmr<v><p3><sg>",
            "shomer/shmr<v><pprs>",
            "yashav/yshv<v><p3><sg>",
            "yoshev/yshv<v><pprs>",
        ],
    ),
    PairTest(
        name="Test 4",
        grammar=r"""PATTERNS
NounStem NounInfl
NounStem NounInflComp Comp NounStem2 NounInfl

LEXICON Comp
<comp>+:

LEXICON NounStem
shoop
blarg

ALIAS NounStem NounStem2

LEXICON NounInfl
<n><sg>:
<n><pl>:ah

LEXICON NounInflComp
<n>:a
""",
        expected_lines=[
            "shoop/shoop<n><sg>",
            "shoopah/shoop<n><pl>",
            "shoopashoop/shoop<n><comp>+shoop<n><sg>",
            "shoopashoopah/shoop<n><comp>+shoop<n><pl>",
            "shoopablarg/shoop<n><comp>+blarg<n><sg>",
            "shoopablargah/shoop<n><comp>+blarg<n><pl>",
            "blarg/blarg<n><sg>",
            "blargah/blarg<n><pl>",
            "blargashoop/blarg<n><comp>+shoop<n><sg>",
            "blargashoopah/blarg<n><comp>+shoop<n><pl>",
            "blargablarg/blarg<n><comp>+blarg<n><sg>",
            "blargablargah/blarg<n><comp>+blarg<n><pl>",
        ],
    ),
    PairTest(
        name="Test 5",
        grammar=r"""PATTERNS
NounStem [<n>:] NounNumber

LEXICON NounStem
sock
ninja

LEXICON NounNumber
<sg>:
<pl>:s
""",
        expected_lines=[
            "ninja/ninja<n><sg>",
            "ninjas/ninja<n><pl>",
            "sock/sock<n><sg>",
            "socks/sock<n><pl>",
        ],
    ),
    PairTest(
        name="Test 6",
        grammar=r"""PATTERNS
(NounStem CaseEnding)[^[Decl1,Decl2],^[N,M,F]]

LEXICON NounStem
mensa:mens[Decl1,F]     # table
poeta:poet[Decl1,M]     # poet
dominus:domin[Decl2,M]  # master
bellum:bell[Decl2,N]    # war

LEXICON CaseEnding[Decl2]
<nom>:>us[M]
<nom>:>um[N]
<acc>:>um    # M or N

LEXICON CaseEnding[Decl1]
<nom>:>a     # any gender
<acc>:>am    # any gender
""",
        expected_lines=[
            "poeta<nom>/poet>a",
            "poeta<acc>/poet>am",
            "mensa<nom>/mens>a",
            "mensa<acc>/mens>am",
            "bellum<nom>/bell>um",
            "bellum<acc>/bell>um",
            "dominus<nom>/domin>us",
            "dominus<acc>/domin>um",
        ],
        forbidden_alphabet_symbols={"Decl1", "Decl2", "M", "F", "N", "[", "]"},
    ),
    AcceptRejectTest(
        name="Test 7",
        grammar=r"""PATTERNS
SomeLexicon

LEXICON SomeLexicon
/x(y|zz)?[n-p]/
""",
        cases={
            "xn": True,
            "xo": True,
            "xp": True,
            "xyn": True,
            "xyo": True,
            "xyp": True,
            "xzzn": True,
            "xzzo": True,
            "xzzp": True,
        },
        require_identity_outputs=True,
    ),
]


# -------------------------
# Helpers
# -------------------------

def _split_pair_line(line: str) -> Tuple[str, str]:
    if "/" in line:
        a, b = line.split("/", 1)
        return a, b
    if ":" in line:
        a, b = line.split(":", 1)
        return a, b
    raise ValueError(f"Expected pair line with '/' or ':': {line!r}")


def _serialize_path(inp: str, out: str) -> str:
    """Serialize a path to the C++ gold format.

    - Pure identity (inp == out) -> inp
    - Input-only (out == '') -> inp + ':'
    - Output-only (inp == '') -> ':' + out
    - General -> inp + ':' + out
    """
    if inp == out:
        return inp
    if out == "":
        return inp + ":"
    if inp == "":
        return ":" + out
    return inp + ":" + out

def _enumerate_gold_strings(fst, max_depth: int, max_strings: int) -> Set[str]:
    """Enumerate strings from fst in the same format as the C++ gold suite.

    We accumulate the full input and output strings along each path, then serialize with minimal colons:
      - identity: 'abc'
      - input-only: 'abc:'
      - output-only: ':abc'
      - general: 'in:out'
    This matches the lexd C++ test gold files (e.g. 'xX:Yy', not 'x:YX:y').
    """
    initial = fst.initialstate
    finals = set(fst.finalstates)

    out: Set[str] = set()

    # stack items: (state, in_str, out_str, depth)
    stack: List[Tuple[object, str, str, int]] = [(initial, "", "", 0)]

    while stack:
        st, ins, outs, d = stack.pop()

        if st in finals:
            out.add(_serialize_path(ins, outs))
            if len(out) >= max_strings:
                break

        if d >= max_depth:
            continue

        for label, trans_set in getattr(st, "transitions", {}).items():
            for tr in trans_set:
                if len(label) == 1:
                    a = label[0]
                    in2 = ins + a
                    out2 = outs + a
                elif len(label) == 2:
                    a, b = label
                    in2 = ins + a if a != "" else ins
                    out2 = outs + b if b != "" else outs
                else:
                    raise ValueError(f"Unexpected label tuple length: {label!r}")
                stack.append((tr.targetstate, in2, out2, d + 1))

        if len(out) >= max_strings:
            break

    return out


def _infer_max_depth_from_gold(gold: List[str]) -> int:
    # crude but safe: allow some headroom
    m = max((len(x) for x in gold), default=0)
    return max(10, m + 10)



CPP_TESTS: List[Test] = [
        GoldStringsTest(
            name="cpp:test-alt",
            grammar=r"""PATTERNS
pattern1

PATTERN pattern1
A | B
C:|:D

LEXICON A
a

LEXICON B
b

LEXICON C
c

LEXICON D
d
""",
            gold_strings=[
                        ":d",
                        "a",
                        "b",
                        "c:"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonlex",
            grammar=r"""PATTERNS
[ a ]
""",
            gold_strings=[
                        "a"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonlex-modifier",
            grammar=r"""PATTERNS
[a] [b]? [c]
""",
            gold_strings=[
                        "abc",
                        "ac"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonpat",
            grammar=r"""PATTERNS
( a b ) | c

LEXICON a
a

LEXICON b
b

LEXICON c
c
""",
            gold_strings=[
                        "ab",
                        "c"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonpat-filter",
            grammar=r"""PATTERNS
(Adj Noun)[nofruit,-nocolor]
(Adj Noun)[-nofruit,nocolor]

LEXICON Adj
bright[nofruit]
green
tasty[nocolor]
impetuous[nofruit,nocolor]

LEXICON Noun
apple[nocolor]
orange
green[nofruit]
cat[nofruit,nocolor]
""",
            gold_strings=[
                        "brightgreen",
                        "brightorange",
                        "greenapple",
                        "greengreen",
                        "tastyapple",
                        "tastyorange"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonpat-filter-ops",
            grammar=r"""PATTERNS
(A)[^[a,b]]

LEXICON A
apple[a]
banana[b]
orange[a,b]
""",
            gold_strings=[
                        "apple",
                        "banana"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonpat-modifier",
            grammar=r"""PATTERNS
[a] ([b])? [c]
""",
            gold_strings=[
                        "abc",
                        "ac"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonpat-nospaces",
            grammar=r"""PATTERNS
(A)

LEXICON A
a
""",
            gold_strings=[
                        "a"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-anonpat-ops",
            grammar=r"""PATTERNS
A|(C?)

LEXICON A
a

LEXICON C
c
""",
            gold_strings=[
                        "",
                        "a",
                        "c"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-conflicting-tags",
            grammar=r"""PATTERNS
Verbs-IV

PATTERN VerbStemBase
V-IV [<v><iv>:>[nonpunct]] V-Aspect-Hab
V-IV [<v><iv>:>[punct]] V-Aspect-Punct   # error reported here

PATTERN VerbStem
VerbStemBase[^[A1,A2]]

PATTERN Verbs-IV
:V-Agent VerbStem[-nonpunct] V-Agent:
:V-Agent VerbStem[stat] V-Agent:

LEXICON V-IV
stem

LEXICON V-Aspect-Hab
<hab>:{a}haʔ[A1]
<hab>:{a}s[A2]

LEXICON V-Aspect-Punct
<punct>:{a}{ʔ}[A1]
<punct>:{a}{ʔ}[A2]

LEXICON V-Agent
<a1sg>:{G}{e}
<a1duexcl>:{y}ag{n}{I}
""",
            gold_strings=[
                        "stem<v><iv><punct><a1duexcl>:{y}ag{n}{I}stem>{a}{\u0294}",
                        "stem<v><iv><punct><a1sg>:{G}{e}stem>{a}{\u0294}"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-diacritic",
            grammar=r"""PATTERNS
X
Y(2)

LEXICON X
\ַ
:ֶ
:\ֻ
x\ַ

LEXICON Y(2)
a ַ
""",
            gold_strings=[
                        ":\u05b6",
                        ":\u05bb",
                        "x\u05b7",
                        "\u05b7"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-disjoint-opt",
            grammar=r"""PATTERNS
A?(1) B A?(1)

LEXICON A
a
aa

LEXICON B
b
bb
""",
            gold_strings=[
                        "aabaa",
                        "aabbaa",
                        "aba",
                        "abba",
                        "b",
                        "bb"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-empty",
            grammar=r"""
""",
            gold_strings=[

            ],
        ),
        GoldStringsTest(
            name="cpp:test-empty-patterns",
            grammar=r"""PATTERNS
Case[t]
Case[s] # comment to get the correct answer

LEXICON Obl
<suff>

LEXICON OblCase
<case>[t]

PATTERN Case
Obl OblCase
""",
            gold_strings=[
                        "<suff><case>"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-filter-crosstalk",
            grammar=r"""PATTERNS
Phrase[nofruit,-nocolor]
Phrase[-nofruit,nocolor]

PATTERN Phrase
Adj Noun

LEXICON Adj
bright[nofruit]

LEXICON Noun
apple[nocolor]
orange[nofruit]
""",
            gold_strings=[
                        "brightorange"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-lexdeftag",
            grammar=r"""PATTERNS
A[x]
B[x]

LEXICON A[x]
apple
banana[-x]

LEXICON A
orange
pear[x]

LEXICON B[x]:[y]
nope[-x]
yep:yep[-y,x] # left side gets x
""",
            gold_strings=[
                        "apple",
                        "pear",
                        "yep"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-lexicon-side-tags",
            grammar=r"""PATTERNS
X(1):X(2)[tag]


LEXICON X(2)
a[tag]	b
""",
            gold_strings=[
                        "a:b"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-lexname-space",
            grammar=r"""PATTERNS
[a]

LEXICON X
blah

LEXICON Y 
bloop

LEXICON Z(2)
x y

LEXICON W(3) 
a b c
""",
            gold_strings=[
                        "a"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-lexnegtag",
            grammar=r"""PATTERNS
[a] A[s,-r]
[b] A[-t,s]
[c] A[-s]

LEXICON A
a[t,s]
b[s,r]
c[t,r]
""",
            gold_strings=[
                        "aa",
                        "bb",
                        "cc"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-lextag",
            grammar=r"""PATTERNS
[a] A[t]
[b] A[r]
[ab] A[s]

LEXICON A
a[t,s]
b[s,r]
""",
            gold_strings=[
                        "aa",
                        "aba",
                        "abb",
                        "bb"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-nontree",
            grammar=r"""PATTERNS
parentpat
parentpat2

PATTERN parentpat
childpat

PATTERN parentpat2
childpat

PATTERN childpat
child

LEXICON child
x
""",
            gold_strings=[
                        "x"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-oneside",
            grammar=r"""PATTERNS
A:
:A

LEXICON A
a1:a2
""",
            gold_strings=[
                        ":a2",
                        "a1:"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-opt",
            grammar=r"""PATTERNS
pattern1

PATTERN pattern1
A? B?
C:? :D?

LEXICON A
a

LEXICON B
b

LEXICON C
c

LEXICON D
d
""",
            gold_strings=[
                        "",
                        ":d",
                        "a",
                        "ab",
                        "b",
                        "c:",
                        "c:d"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-or-filter",
            grammar=r"""PATTERNS
A[|[a,b]]

LEXICON A
apple[a]
banana[b]
orange
delaware[notafruit]
""",
            gold_strings=[
                        "apple",
                        "banana"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-pairs",
            grammar=r"""PATTERNS
pattern

PATTERN pattern
x(1):y(2) x(2):y(1)

LEXICON x(2)
x X

LEXICON y(2)
y Y
""",
            gold_strings=[
                        "xX:Yy"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-pattag",
            grammar=r"""PATTERNS
[t] A[t]
[nott] A[-t]

PATTERN A
B
C

LEXICON B
a[t]

LEXICON C
b[s]
""",
            gold_strings=[
                        "nottb",
                        "ta"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-pattag-coherent",
            grammar=r"""PATTERNS

B(1)[x] A(1)[x] B(1)
B(1)[-x] A(1)[-x] B(1)

LEXICON A

a-no-x
a-x[x]

LEXICON B

b-no-x
b-x[x]
""",
            gold_strings=[
                        "b-no-xa-no-xb-no-x",
                        "b-xa-xb-x"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-pattag-details",
            grammar=r"""PATTERNS
X[t,-s]

PATTERN X
A B
C

LEXICON A
a
at[t]
as[s]
ast[s,t]

LEXICON B
b
bt[t]
bs[s]
bst[t,s]

LEXICON C
ct[t]
cs[s]
cst[t,s]
""",
            gold_strings=[
                        "abt",
                        "atb",
                        "atbt",
                        "ct"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-pattern-independence",
            grammar=r"""PATTERNS
X A
B B

PATTERN X
A A

LEXICON A
a1
a2

LEXICON B
b1
b2
""",
            gold_strings=[
                        "a1a1a1",
                        "a1a1a2",
                        "a2a2a1",
                        "a2a2a2",
                        "b1b1",
                        "b2b2"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-regex",
            grammar=r"""PATTERNS
[/a/]
RE
COLRE(1) COLRE(2)
TWOSIDED

LEXICON RE
/<b>/
/c[d-f]g/
/h(i)?/
/j|k/
/(l(m)?)?/

LEXICON COLRE(2)
/n[op]/ q
r /[s-u]v/

LEXICON TWOSIDED
/w:x[yz]/
""",
            gold_strings=[
                        "",
                        "<b>",
                        "a",
                        "cdg",
                        "ceg",
                        "cfg",
                        "h",
                        "hi",
                        "j",
                        "k",
                        "l",
                        "lm",
                        "noq",
                        "npq",
                        "rsv",
                        "rtv",
                        "ruv",
                        "w:xy",
                        "w:xz"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-revsieve",
            grammar=r"""PATTERNS
b(1) < b(2) < b(3)

LEXICON b(3)
b1 b2 b3
""",
            gold_strings=[
                        "b1b2b3",
                        "b2b3",
                        "b3"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-sieve",
            grammar=r"""PATTERNS
pattern

LEXICON f(3)
f1 f2 f3
g1 g2 g3

LEXICON b(3)
b1 b2 b3
c1 c2 c3

LEXICON m
m

PATTERN pattern
f(1) > f(2) > f(3)
b(1) < b(2) < b(3)
b(1) < m > f(1)
""",
            gold_strings=[
                        "b1b2b3",
                        "b1m",
                        "b1mf1",
                        "b1mg1",
                        "b2b3",
                        "b3",
                        "c1c2c3",
                        "c1m",
                        "c1mf1",
                        "c1mg1",
                        "c2c3",
                        "c3",
                        "f1",
                        "f1f2",
                        "f1f2f3",
                        "g1",
                        "g1g2",
                        "g1g2g3",
                        "m",
                        "mf1",
                        "mg1"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-sieveopt",
            grammar=r"""PATTERNS
A > B|C

LEXICON A
a

LEXICON B
b

LEXICON C
c
""",
            gold_strings=[
                        "a",
                        "ab",
                        "ac"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-slots-and-operators-nospace",
            grammar=r"""PATTERNS
X(1)|Y(1) [z]

LEXICON X(2)
x1 x2

LEXICON Y(2)
y1 y2
""",
            gold_strings=[
                        "x1z",
                        "y1z"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-xor-filter",
            grammar=r"""PATTERNS
A[^[a,b]]

LEXICON A
apple[a]
banana[b]
orange[a,b]
""",
            gold_strings=[
                        "apple",
                        "banana"
            ],
        ),
        GoldStringsTest(
            name="cpp:test-xor-multi",
            grammar=r"""PATTERNS
Phrase[^[nofruit,nocolor]]

PATTERN Phrase
Adj Noun

LEXICON Adj
bright[nofruit]
green
tasty[nocolor]
impetuous[nofruit,nocolor]

LEXICON Noun
apple[nocolor]
orange
green[nofruit]
cat[nofruit,nocolor]
""",
            gold_strings=[
                        "brightgreen",
                        "brightorange",
                        "greenapple",
                        "greengreen",
                        "tastyapple",
                        "tastyorange"
            ],
        ),
]





# -------------------------
# Runners
# -------------------------

def run_pair_test(t: PairTest) -> None:
    parsed = lexd.parse_lexd(t.grammar)
    fst = lexd.compile_lexd(parsed)

    if t.forbidden_alphabet_symbols:
        bad = set(t.forbidden_alphabet_symbols) & set(getattr(fst, "alphabet", set()))
        if bad:
            raise AssertionError(f"{t.name}: forbidden symbols in alphabet: {sorted(bad)}")

    expected_map: Dict[str, Set[str]] = {}
    for line in t.expected_lines:
        left, right = _split_pair_line(line)
        outs_left = set(fst.apply(left))
        if right in outs_left:
            inp, out = left, right
        else:
            outs_right = set(fst.apply(right))
            if left in outs_right:
                inp, out = right, left
            else:
                raise AssertionError(
                    f"{t.name} FAILED: neither apply({left!r}) contains {right!r} nor apply({right!r}) contains {left!r}."
                )
        expected_map.setdefault(inp, set()).add(out)

    missing_lines = []
    extra_lines = []
    for inp, exp_outs in expected_map.items():
        got_outs = set(fst.apply(inp))
        missing = exp_outs - got_outs
        extra = got_outs - exp_outs
        for m in sorted(missing):
            missing_lines.append(f"{inp} -> {m}")
        for e in sorted(extra):
            extra_lines.append(f"{inp} -> {e}")

    if missing_lines or extra_lines:
        msg = [f"{t.name} FAILED"]
        if missing_lines:
            msg.append("Missing:")
            msg.extend("  " + x for x in missing_lines)
        if extra_lines:
            msg.append("Extra:")
            msg.extend("  " + x for x in extra_lines)
        raise AssertionError("\n".join(msg))


def run_accept_reject_test(t: AcceptRejectTest) -> None:
    parsed = lexd.parse_lexd(t.grammar)
    fst = lexd.compile_lexd(parsed)

    for inp, should_accept in t.cases.items():
        outs = list(fst.apply(inp))
        accepted = len(outs) > 0
        if accepted != should_accept:
            raise AssertionError(
                f"{t.name} FAILED on {inp!r}: expected accept={should_accept}, got accept={accepted}, outs={outs}"
            )
        if should_accept and t.require_identity_outputs:
            if inp not in outs:
                raise AssertionError(
                    f"{t.name} FAILED on {inp!r}: expected identity output {inp!r} among outs={outs}"
                )


def run_gold_strings_test(t: GoldStringsTest) -> None:
    parsed = lexd.parse_lexd(t.grammar)
    fst = lexd.compile_lexd(parsed)

    gold = set(t.gold_strings)
    max_depth = t.max_depth or _infer_max_depth_from_gold(t.gold_strings)
    got = _enumerate_gold_strings(fst, max_depth=max_depth, max_strings=t.max_strings)

    if got != gold:
        missing = sorted(gold - got)
        extra = sorted(got - gold)
        msg = [f"{t.name} FAILED (gold strings mismatch)"]
        if missing:
            msg.append("Missing:")
            msg.extend("  " + m for m in missing[:200])
            if len(missing) > 200:
                msg.append(f"  ... ({len(missing)-200} more)")
        if extra:
            msg.append("Extra:")
            msg.extend("  " + e for e in extra[:200])
            if len(extra) > 200:
                msg.append(f"  ... ({len(extra)-200} more)")
        raise AssertionError("\n".join(msg))


def run_test(test: Test) -> None:
    if isinstance(test, PairTest):
        run_pair_test(test)
    elif isinstance(test, AcceptRejectTest):
        run_accept_reject_test(test)
    else:
        run_gold_strings_test(test)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Run only tests matching this name/substring or number (1-based index in --list).")
    ap.add_argument("--list", action="store_true", help="List tests and exit.")
    args = ap.parse_args(argv)

    tests: List[Test] = []
    tests.extend(HAND_TESTS)
    tests.extend(CPP_TESTS)

    if args.list:
        for i, t in enumerate(tests, 1):
            print(f"{i:02d}. {t.name}")
        return 0

    selected: List[Test] = tests
    if args.only:
        key = args.only.strip()
        if key.isdigit():
            idx = int(key)
            if not (1 <= idx <= len(tests)):
                print(f"--only index out of range: {idx}", file=sys.stderr)
                return 2
            selected = [tests[idx - 1]]
        else:
            selected = [t for t in tests if t.name == key]
            if not selected:
                selected = [t for t in tests if key.lower() in t.name.lower()]
            if not selected:
                print(f"No tests matched --only {key!r}", file=sys.stderr)
                return 2

    failures = 0
    for t in selected:
        try:
            run_test(t)
            print(f"PASS: {t.name}")
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.name} ({type(e).__name__}: {e})")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
