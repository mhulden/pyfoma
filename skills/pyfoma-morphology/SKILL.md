---
name: pyfoma-morphology
description: Build and review complex morphological analyzers/generators in PyFoma with clear separation of lexd morphotactics and FST.re morphophonology, including robust handling of tagging, rule ordering, reduplication, root-pattern morphology, and regression testing.
---

# pyfoma-morphology skill

This skill is for implementing or reviewing production-quality morphology grammars in PyFoma.

Use with:
- `references/REFERENCE.md` for PyFoma regex/rewrite details and design patterns.
- `references/LEXD.md` for full lexd syntax and semantics in this repository.

## Output expectations

When asked to build or revise a grammar, produce:
1. Grammar code (`lexd` + rule cascade).
2. A minimal executable test snippet that verifies generation and analysis.
3. Short notes on rule ordering and known edge cases.
4. A brief generalization note:
   - productive pattern(s)
   - explicit exception list
   - why each lexical class marker is needed (if any)

## Analyzer objectives

A morphological analyzer is not a memorized pair list. It should:
- encode productive inflection with general rules
- keep lexical listing for stems and true irregular exceptions
- map lexical strings with tags to clean orthographic outputs
- map orthographic outputs back to lexical analyses

## Workflow

### 1. Define lexical contract before coding

Freeze these first:
- Tag inventory: `<N>`, `<V>`, `<Past>`, `<Pl>`, etc.
- Boundary symbols used during intermediate forms (often `+`).
- Intermediate morpheme policy:
  - prefer canonical morphs with boundaries (for example `+s`, `+es`)
  - default rule: do not use abstract markers like `+PL`
  - exception only if canonical morph representation is demonstrably impractical in the runtime, and the reason is documented
- Expected API usage:
  - `generate(lexical)` for generation
  - `analyze(surface)` for analysis
- Side contract:
  - lexical side may contain tags and temporary class markers
  - final surface side must contain only orthographic wordforms (no tags, no `SG/PL` markers, no `+`)

If this contract is unstable, delay rule-writing.

### 2. Build morphotactics with lexd

Use `lexd.compile(...)` for:
- Paradigm structure and affix sequencing
- Lexically conditioned alternation choices (via tags/selectors)
- Non-concatenative structure supported by this implementation (reduplication/root-pattern patterns)

Keep phonological alternation out of lexd unless it is true suppletion/irregular listing.

Anti-pattern to avoid:
- Do not list productive pluralized outputs directly per noun in lexd.
- Do not encode singular/plural as literal output strings like `catSG`, `catPL`, etc.
- If multiple lemmas share behavior, model it once with a rule and (if needed) a lexical class marker.
- Do not create many ad hoc lexical class markers when the behavior can be expressed by orthographic/phonological context rules.
- Do not use one-off marker classes unless they are true exceptions and documented as such.

### 2.1 If `lexd` is unavailable

Fallback to `FST.re` without changing grammar design principles:
- Build a stem inventory FST.
- Build an inflection/tag transducer FST.
- Compose them into a lexical transducer.
- Keep productive alternations in rewrite rules.

Even in fallback mode:
- do not collapse into per-lemma hard-coding
- keep an explicit productive/default vs exception split

### 3. Validate morphotactics in isolation

Before adding rewrite rules:
- Test at least 10 positive lexical->surface pairs.
- Test at least 10 positive surface->lexical pairs.
- Test at least 5 negative forms.
- Verify morphotactic stage still contains expected intermediate markers (if any) and has not baked in full productive alternations.

Do not proceed to morphophonology until these pass.

### 4. Add rewrite rules incrementally

Write one process per rule using `FST.re("$^rewrite(...)")`.

Required discipline:
- Keep each rule name descriptive.
- Add one comment example per rule.
- Preserve boundaries/tags until final cleanup.
- Prefer rules stated over natural symbol classes/contexts (e.g., vowel-final vs consonant-final), not rules tied to tiny hand-built categories.
- If a rule only handles one or two lexemes, move that behavior to an explicit exception path instead of pretending it is productive.
- Prefer rewriting against canonical intermediate morphs (e.g., `+s`, `+es`) rather than abstract grammatical placeholders when possible.
- If abstract placeholders are used temporarily during development, convert to canonical morph representation before final delivery unless a documented exception applies.

Compose strictly as a left-to-right cascade:

```python
final = FST.re("$lexicon @ $rule1 @ $rule2 @ $rule3 @ $cleanup", fsts)
```

### 5. Test each rule after composition

After each new rule:
- Re-run small focused tests for the new phenomenon.
- Re-run previous regression cases.
- Verify both `generate` and `analyze`.

For overlapping matches, explicitly test default vs directed behavior:
- `leftmost=True`
- `longest=True`
- `shortest=True`

### 6. Add irregularity explicitly

Prefer separate lexicon paths for irregulars instead of forcing them into productive rules.

Example pattern:
- Productive stems in `NounRoot` + productive inflection.
- Irregular whole forms in `IrregularNoun` lexicon block.

### 7. Final QA

Pass criteria:
- Stable tag scheme and readable grammar layout.
- No rule relies on symbols already deleted by earlier rules.
- No unexplained nondeterminism in expected deterministic cases.
- Regression set includes productive, irregular, and edge cases.
- Gold-set automaton checks are run for coverage and overgeneration:
  `grammar @ gold` and `grammar @ ~gold`, plus projection-set diffs if needed
  (`gold - output(grammar)`, `output(grammar) - gold`).
- Surface-side hygiene checks pass:
  - no `<...>` tags in generated forms
  - no `+` boundary symbols in generated forms
  - no grammatical marker suffixes/prefixes like `SG`, `PL` in generated forms
- Overfitting checks pass:
  - productive rules are few and broad, not many micro-rules for tiny subsets
  - each lexical marker class is linguistically named and justified
  - exception handling is explicit and separated from productive rules
- Intermediate representation checks pass:
  - no abstract inflection placeholders in the final grammar unless explicitly justified

## Compact lexd guide (full details in `references/LEXD.md`)

Core sections:
- `PATTERNS`
- `PATTERN Name`
- `LEXICON Name`
- `ALIAS Source Alias`

Frequent operators/features:
- `?`, `*`, `+`, `|`
- Grouping: `( ... )`
- Anonymous lexicon: `[ ... ]`
- Selectors: `[tag]`, `[-tag]`, `[|[a,b]]`, `[^[a,b]]`
- Columns: `Root(2)`
- Side markers: `:Lex`, `Lex:`
- Paired columns: `X(1):X(2)`, `X(1):Y(2)`
- Optional segmented lexicon: `Name?(1)`
- Sieve expansion: `<`, `>`

Critical semantic points:
- Repeating the same lexicon in one sequence usually binds the same chosen entry.
- Use `ALIAS` when repeated positions should vary independently.
- Tag selectors constrain combinations but do not emit symbols.

Design-quality points:
- Lexical class markers are a last resort, not the first strategy.
- Prefer one productive rule plus a small exception list over many marker-specific rules.

## Guardrails

- In `FST.re`, always quote multichar symbols: `'<N>'`, `'+'`, `'{A}'`.
- Treat wildcard `.` as advanced; avoid accidental use.
- Keep morphotactics and morphophonology separate.
- Prefer several small rules to one unreadable mega-rule.
- Never assume generator/analyzer output order is stable.

## Minimal starter template

```python
from pyfoma import FST, lexd

fsts = {}

fsts['lexicon'] = lexd.compile(r'''
PATTERNS
NounRoot NounInfl

LEXICON NounRoot
house
spy
church

LEXICON NounInfl
<N><Sg>:
<N><Pl>:+s
''')

fsts['y_to_ie'] = FST.re("$^rewrite(y:(ie) / _ '+' s)")
fsts['epenthesis'] = FST.re("$^rewrite('':e / (sh|ch|x|s|z) _ '+' s)")
fsts['cleanup'] = FST.re("$^rewrite('+':'')")

grammar = FST.re("$lexicon @ $y_to_ie @ $epenthesis @ $cleanup", fsts)

print(list(grammar.generate("spy<N><Pl>")))
print(list(grammar.analyze("churches")))
```
