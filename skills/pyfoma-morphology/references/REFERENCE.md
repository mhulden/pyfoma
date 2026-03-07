# PyFoma morphology reference

This document is a practical reference for writing analyzers/generators in PyFoma.

For detailed lexd syntax and semantics, see `LEXD.md`.

## 1. Core build model

Recommended architecture:
1. Build lexical transducer (morphotactics + lexical tags) with `lexd.compile(...)`.
2. Build morphophonological rules as separate FSTs with `FST.re("$^rewrite(...)")`.
3. Compose in linguistic order:

```python
grammar = FST.re("$lexicon @ $rule1 @ $rule2 @ $rule3 @ $cleanup", fsts)
```

Treat composition order as part of grammar correctness.

## 1.1 Analyzer design goal

A morphological analyzer should represent a compact system of:
- productive morphological rules
- lexical inventory
- explicit exceptions

It should not behave like a disguised lookup table where most behavior is encoded
as lexeme-specific classes or one-off rules.

## 2. Core APIs

```python
from pyfoma import FST, lexd

fst = FST.re("...")
lex = lexd.compile("""...""")
```

- `FST.re(regex, defined=None, functions=None, multichar_symbols=None)`.
- `lexd.compile(grammar)`.
- `fst.generate(lexical_string)` yields surfaces.
- `fst.analyze(surface_string)` yields analyses.

`generate`/`analyze` are generators. Use `list(...)` in tests.

If `lexd` is unavailable in the runtime, build the lexical transducer with `FST.re`
from reusable parts (stems + inflection/tag mapping) and keep the same design
discipline: productive rules + explicit exceptions.

## 3. PyFoma regex essentials

### 3.1 Symbols and quoting

- Whitespace in regex is insignificant unless escaped or quoted.
- Epsilon is `''`.
- Multichar symbols must be quoted in regex:
  - `'<N>'`, `'<Pl>'`, `'+'`, `'{A}'`

### 3.2 Reusing compiled FSTs

```python
fsts = {}
fsts['stem'] = FST.re("house|spy|church")
fsts['infl'] = FST.re("('<N>' '<Sg>'):'' | ('<N>' '<Pl>'):('+' s)")
fsts['lexicon'] = FST.re("$stem $infl", fsts)
```

### 3.3 Operators used most in morphology

- Concatenation: adjacency (`A B`)
- Union: `A | B`
- Cross-product: `A:B`
- Optional cross-product: `A:?B`
- Composition: `A @ B`
- Optional/repetition: `?`, `*`, `+`, `{m,n}`

### 3.4 Wildcard caution

`.` is not a normal literal period. It is wildcard behavior tied to automaton alphabet semantics.
Completely avoid use of literal periods. If you need a syllable separator, use a hyphen or some other symbol instead.

## 4. Rewrite rules

### 4.1 Basic usage

```python
FST.re("$^rewrite(y:(ie) / _ '+' s)")
FST.re("$^rewrite('':e / (sh|ch|x|s|z) _ '+' s)")
FST.re("$^rewrite('+':'')")
```

### 4.2 Context and directionality

- Context form: `... / LEFT _ RIGHT`
- Optional rewrite behavior: use optional cross-product in rule mapping (e.g. `a:?b`).
- Directed overlap control:
  - `leftmost=True`
  - `longest=True`
  - `shortest=True`

Test directed rules explicitly on overlapping inputs.

### 4.3 Rule-writing style

- One linguistic process per rule.
- Keep intermediate markers (like `+`) until cleanup.
- Add one comment example per rule.

## 5. Practical grammar design

### 5.1 Separation of concerns

- `lexd`: morphotactics, paradigms, lexical selection logic, irregular listing.
- `FST.re` rules: alternation, insertion, deletion, assimilation, cleanup.

### 5.2 Lexical-vs-surface side contract

- Lexical side may contain grammatical tags and temporary class markers.
- Final surface side should contain only real orthographic wordforms.
- Never expose grammatical payload in outputs (no `<N>`, `<Pl>`, `+`, `SG`, `PL`, etc.).

### 5.3 Productive vs irregular

Keep productive morphology and irregular paradigms separate in lexical design.

### 5.4 Intermediate representations

Use explicit intermediate symbols (such as `+` boundary or temporary markers) when they make rule triggering predictable.

Preferred style for inflectional morphology:
- represent canonical morphemes on the intermediate surface (`+s`, `+es`, etc.)
- keep grammatical features (`<Pl>`, `<Sg>`) on the lexical side

Release default:
- do not use abstract placeholders like `+PL` in final grammars.

Exception policy:
- allow abstract placeholders only when canonical morph representation is demonstrably impractical in the runtime.
- if used, document the technical reason and keep the placeholder scope minimal.

### 5.5 Avoid hard-coding productive alternation

If a pattern applies to many lexemes, encode it as a general rule, not as lexeme-by-lexeme surface listings.

Bad direction:
- listing plural outputs per noun directly in lexd for regular classes.

Good direction:
- keep lemma inventory in lexd
- use shared class markers when needed
- apply one plural allomorphy/orthography rule per process in the cascade

### 5.6 Class-marker policy

Class markers are allowed, but should be controlled:
- Use them only when behavior is not cleanly expressible by reusable contexts.
- Give each marker a linguistic meaning (not ad hoc labels like “group1/group2”).
- Prefer a small explicit exception lexicon over proliferating marker classes.
- Document why each marker class exists and which lexemes it covers.

If a marker class has very small support and no clear linguistic rationale, treat
it as an exception path instead of productive grammar.

### 5.7 Rule specificity policy

Rules should primarily reference structural contexts (symbol classes, boundaries,
morpheme position), not tiny handpicked substrings.

Warning signs of overfitting:
- many narrowly targeted rules to force individual words to pass
- rules that only work because of custom per-lemma marker injection
- no clear productive default behavior statement

## 6. Testing protocol

Minimum viable checks for each grammar revision:
1. Positive generation tests.
2. Positive analysis tests.
3. Negative invalid-form tests.
4. Regression set for previously fixed cases.
5. Lexicon-stage sanity checks (before phonological rules).
6. Model-shape checks (productive rules vs exceptions).

Do not rely on output order from `generate`/`analyze`.

Lexicon-stage sanity checks should verify that productive alternation is not already hard-coded.
Example: regular plural candidates at lexicon stage should still carry intermediate markers,
not already-final allomorphs.

Model-shape checks should verify:
- you can name the productive default(s) in one or two sentences
- exception list is explicit and bounded
- rule count is not inflated by many one-off fixes
- intermediate representation uses canonical morphemes unless a justified exception exists

### 6.1 Gold-set automaton checks

Besides per-form `generate`/`analyze` assertions, do global checks by composing
the grammar against a finite automaton of gold forms.

```python
from pyfoma import FST

fsts['gold'] = FST.re("house|houses|bag|bags|spy|spies|church|churches|wish|wishes|buzz|buzzes")
```

Useful checks:
- `FST.re("$grammar @ $gold", fsts)`:
  isolates grammar behavior on the gold language.
- `FST.re("$grammar @ ~$gold", fsts)`:
  isolates generated/analyzed behavior outside gold (overgeneration candidates).

Use these as debugging networks and inspect with `generate`/`analyze` on targeted items.

If you want explicit missing/extra output sets, compare against output projection:

```python
fsts['generated_out'] = FST.re("$^output($grammar)", fsts)
fsts['missing_out'] = FST.re("$gold - $generated_out", fsts)
fsts['extra_out'] = FST.re("$generated_out - $gold", fsts)
```

## 7. Common failure modes

- Unquoted multichar symbols in `FST.re`.
- Rule order mismatch (rule expects context destroyed earlier).
- Overuse of wildcard `.` causing overgeneration.
- Encoding phonological processes in lexd instead of rules.
- Forgetting lexd repeated-lexicon binding behavior in one sequence.
- Surface forms leaking grammar markers (`+`, tags, `SG/PL`, class symbols).
- Overfitting by marker proliferation (many per-lemma classes instead of productive contexts).
- Micro-rules for tiny subsets instead of explicit exception handling.

## 7.1 Generalization audit checklist

Before shipping a grammar, answer:
1. What are the productive default inflection patterns?
2. Which forms are exceptions, and where are they encoded?
3. Does each class marker correspond to a real linguistic class?
4. Could any marker-specific rule be rewritten as a broader structural rule?
5. If all exceptions were removed, would the remaining grammar still be coherent?

## 8. End-to-end example

```python
from pyfoma import FST, lexd

fsts = {}

fsts['lexicon'] = lexd.compile(r'''
PATTERNS
NounRoot NounInfl

LEXICON NounRoot
house
bag
spy
church
wish
buzz

LEXICON NounInfl
<N><Sg>:
<N><Pl>:+s
''')

fsts['y_to_ie'] = FST.re("$^rewrite(y:(ie) / _ '+' s)")
fsts['insert_e'] = FST.re("$^rewrite('':e / (sh|ch|x|s|z) _ '+' s)")
fsts['cleanup'] = FST.re("$^rewrite('+':'')")

grammar = FST.re("$lexicon @ $y_to_ie @ $insert_e @ $cleanup", fsts)

print(list(grammar.generate("spy<N><Pl>")))     # ['spies']
print(list(grammar.analyze("churches")))        # ['church<N><Pl>']
```

Add a hygiene assertion block in tests:

```python
for lex in ["spy<N><Pl>", "church<N><Pl>"]:
    for surf in grammar.generate(lex):
        assert "<" not in surf and ">" not in surf
        assert "+" not in surf
        assert "SG" not in surf and "PL" not in surf
```

## 9. When to open `LEXD.md`

Open `LEXD.md` whenever you need any of:
- `ALIAS`, selectors, multi-column lexicons
- `X(i):X(j)` or `X(i):Y(j)` pairing
- reduplication/root-pattern patterns
- sieve operators `<` and `>`
- anonymous lexicons/patterns
- subtle parsing details and escaping rules
