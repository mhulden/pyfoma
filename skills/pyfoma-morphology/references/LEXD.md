# lexd for PyFoma: practical reference

This file documents the lexd implementation in this repository for AI agents and grammar authors.

Main entry point:

```python
from pyfoma import lexd
fst = lexd.compile(grammar_text)
```

The result is one transducer suitable for `generate()` and `analyze()`.

## Table of contents

1. [What lexd is for](#1-what-lexd-is-for)
2. [File structure and sections](#2-file-structure-and-sections)
3. [LEXICON entries](#3-lexicon-entries)
4. [PATTERNS and PATTERN expressions](#4-patterns-and-pattern-expressions)
5. [Tag selectors](#5-tag-selectors)
6. [Columns, side placement, and pairing](#6-columns-side-placement-and-pairing)
7. [Binding semantics](#7-binding-semantics)
8. [Advanced examples](#8-advanced-examples)
9. [Escaping, comments, and parsing rules](#9-escaping-comments-and-parsing-rules)
10. [Common mistakes and fixes](#10-common-mistakes-and-fixes)
11. [Minimal validation script](#11-minimal-validation-script)

## 1. What lexd is for

Use lexd to define morphotactics and lexical tagging, then compose with rewrite-rule FSTs for morphophonology.

Recommended split:
- lexd: what morphemes can combine
- rewrite rules: how combinations surface

Non-negotiable output principle:
- Final generated surface forms must be plain orthography.
- Do not leave grammatical markers or temporary symbols in the surface side.

Non-negotiable modeling principle:
- lexd should encode morphotactic structure and lexical inventory, not memorize
  productive inflected outputs entry by entry.

## 2. File structure and sections

Supported top-level section headers:
- `PATTERNS`
- `PATTERN Name`
- `LEXICON Name`
- `ALIAS Source Alias`

### 2.1 `PATTERNS`

Each line is a top-level pattern. The compiled grammar is the union of all lines.

```text
PATTERNS
NounRoot NounInfl
VerbRoot VerbInfl
```

### 2.2 `PATTERN Name`

Define reusable named pattern fragments.

```text
PATTERN VerbStem
VerbRoot
VerbRoot Causative
AuxRoot

PATTERNS
VerbStem Tense PersonNumber
```

### 2.3 `LEXICON Name`

Defines a lexicon block of entries.

```text
LEXICON NounRoot
house
spy
church
```

### 2.4 `ALIAS Source Alias`

`ALIAS A B` means `B` resolves to lexicon `A`.

Use this when you want the same lexicon contents but independent choice behavior in patterns.

```text
ALIAS NounStem NounStem2
```

## 3. LEXICON entries

### 3.1 Basic entry forms

Allowed entry forms include:
- `stem` (same lexical and surface string)
- `lex:surf` (different lexical/surface sides)
- `/.../` regex entry
- multi-column row for `LEXICON X(n)`

Example:

```text
LEXICON Infl
<N><Sg>:
<N><Pl>:+s
```

Preferred inflection style:
- lexical side: grammatical features (`<N><Pl>`)
- intermediate/surface side: canonical morphs with boundaries (`+s`, `+es`)

Less preferred:
- abstract placeholders like `+PL` as the main intermediate morph token.

Important:
- Entries may introduce intermediate symbols for later rules.
- Cleanup rules must remove intermediate symbols before final output.

### 3.2 Multi-column lexicons

Declare arity with `LEXICON Name(n)` and provide `n` columns per entry.

```text
LEXICON Root(3)
k t b
s l m
```

Columns are referenced with `(i)` from patterns (see section 6).

### 3.3 Regex entries

If an entry starts and ends with `/`, it is treated as a regex and compiled through `FST.re`.

```text
LEXICON SomeLexicon
/x(y|zz)?[n-p]/
```

This yields strings such as `xn`, `xyo`, `xzzp`.

### 3.4 Lexicon default tags

You can assign default tags to all entries in a lexicon block.

```text
LEXICON NounRoot[count]
sock
rice[mass,-count]
sand[mass]
```

Entry tags override defaults by adding/removing tags (`-tag`).

### 3.5 Side defaults in lexicon header

This implementation supports:

```text
LEXICON A[x]:[y]
...
```

Header tags are defaults for filtering only; they are not emitted symbols.

### 3.6 What not to encode directly in lexd

Do not hard-code productive alternation outcomes per lemma when they can be generalized.

Avoid:
- many entries that directly list regular plural outputs as if they were irregular.

Prefer:
- lexical stem inventory + inflection tags in lexd
- plural allomorphy and orthographic alternations in rewrite rules

Additional guidance:
- If you find yourself assigning many custom class markers directly in stem entries,
  pause and check whether a single structural rule can replace them.
- Keep marker classes interpretable (for example, “takes-es plural” or “true irregular”),
  not arbitrary bucket names.

## 4. PATTERNS and PATTERN expressions

### 4.1 Core operators

- Sequence by whitespace: `A B C`
- Alternation: `A | B`
- Grouping: `( ... )`
- Quantifiers: `?`, `*`, `+`

```text
PATTERNS
Negation? Adjective
(VerbRoot Causative?) | AuxRoot Tense PersonNumber
```

### 4.2 Anonymous lexicons

Use `[ ... ]` inside patterns for single inline entries.

```text
PATTERNS
NounStem [<n>:] NounNumber
```

### 4.3 Optional segmented lexicon syntax

`Name?(k)` means the whole `Name(k)` segment is optional in that pattern line.

```text
PATTERNS
OptionalCircumfix?(1) Stem OptionalCircumfix?(2)
```

### 4.4 Sieve operators `<` and `>`

Sieve operators expand one pattern line into multiple lines.

```text
PATTERNS
VerbStem > Nominalization > Case
```

Equivalent expansions:
- `VerbStem`
- `VerbStem Nominalization`
- `VerbStem Nominalization Case`

Important: `<` and `>` are treated as sieve operators only when surrounded by whitespace at top level.

### 4.5 Anonymous pattern combinations

Nested grouped expressions are supported and can be quantified.

```text
PATTERNS
NounRoot ([<n>:] (Number Case)?) | (Verbalizer Tense)
```

## 5. Tag selectors

Selectors constrain which entries may participate.

### 5.1 Selector forms

- Positive: `[tag]`
- Negative: `[-tag]`
- Union: `[|[a,b]]` meaning tag `a` or tag `b`
- Symmetric difference: `[^[a,b]]` meaning exactly one of `a`, `b`

Selectors can attach to lexicon refs and grouped expressions.

### 5.2 Example

```text
PATTERNS
NounRoot[count] [<n>:] Number
NounRoot[mass] [<n>:]
NounRoot[-count] [<n>:]
```

### 5.3 Selector distribution over grouped patterns

For grouped expression `(A B)[x]`, selector distribution follows:
- positive selector distributes as alternatives where one component satisfies it
- negative selector is applied to all components

Practical takeaway: grouped selectors can change pattern expansion non-trivially. Test them explicitly.

## 6. Columns, side placement, and pairing

### 6.1 Column reference

Use `X(i)` to reference column `i` of lexicon `X(n)`.

```text
PATTERNS
Root(1) Root(2)
```

### 6.2 Side placement markers

- `:Lex` places lexicon material on output side.
- `Lex:` places lexicon material on input side.
- `Lex` uses both sides according to entry form.

### 6.3 Same-lexicon paired columns

`X(i):X(j)` binds one chosen entry from `X` and maps its column `i` to input and column `j` to output.

### 6.4 Cross-lexicon paired columns

`X(i):Y(j)` pairs rows by index across two lexicons.

Behavior details:
- Pairing iterates row index `k` up to `min(len(X), len(Y))`.
- Repeated occurrences of the same pair in one sequence share the same chosen row index.

## 7. Binding semantics

This is essential for correct grammar design.

### 7.1 Repeated lexicon references in one sequence

Within one sequence scope, repeated references to the same lexicon name generally bind to the same selected entry.

This enables reduplication-like patterns naturally.

### 7.2 Breaking shared choice

If repeated positions should select independently, use a different name via `ALIAS`.

```text
ALIAS NounStem NounStem2
```

Then `NounStem ... NounStem2` does not force same entry.

### 7.3 Multi-column and one-sided refs

Multi-column references and one-sided refs also engage binding behavior. Keep this in mind when debugging unexpected coupling.

## 8. Advanced examples

### 8.1 Reduplication pattern

```text
PATTERNS
:VerbInfl VerbRoot VerbInfl:
:VerbInfl :VerbRoot VerbRoot VerbInfl: Redup

LEXICON VerbRoot
bloop
vroom

LEXICON VerbInfl
<v><pres>:en

LEXICON Redup
<redup>:
```

Examples:
- `bloop<v><pres> -> enbloop`
- `bloop<v><pres><redup> -> enbloopbloop`

### 8.2 Root-pattern style interdigitation

```text
PATTERNS
C(1) :V(1) C(2) :V(2) C(3) V(2):

LEXICON C(3)
sh m r
y sh v

LEXICON V(2)
:a <v><p3><sg>:a
:o <v><pprs>:e
```

Examples:
- `shmr<v><p3><sg> -> shamar`
- `yshv<v><pprs> -> yoshev`

### 8.3 Compounding with independent stem choices

```text
PATTERNS
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
```

Examples:
- `shoop<n><sg> -> shoop`
- `shoop<n><comp>+blarg<n><pl> -> shoopablargah`

### 8.4 Selector-heavy pattern

```text
PATTERNS
(NounStem CaseEnding)[^[Decl1,Decl2],^[N,M,F]]

LEXICON NounStem
mensa:mens[Decl1,F]
poeta:poet[Decl1,M]
dominus:domin[Decl2,M]
bellum:bell[Decl2,N]

LEXICON CaseEnding[Decl2]
<nom>:>us[M]
<nom>:>um[N]
<acc>:>um

LEXICON CaseEnding[Decl1]
<nom>:>a
<acc>:>am
```

## 9. Escaping, comments, and parsing rules

### 9.1 Comments

`#` starts an inline comment.

To keep a literal `#` in content, escape it: `\#`.

### 9.2 Escaped spaces and fields

Within lexicon lines, backslash escapes the next character.

Example:
- `ya\ ngai` is parsed as one field containing a space.

### 9.3 Tokenization caveats

- Bracketed `[...]` may be selector or anonymous lexicon depending on context.
- Sieve operators `<` `>` require top-level whitespace to be recognized.
- `Name?(k)` has special expansion behavior in `PATTERNS` lines.

## 10. Common mistakes and fixes

1. **Mistake:** forgetting `ALIAS` when two positions should vary independently.
   **Fix:** add alias and use the alias name in one position.

2. **Mistake:** using wrong column index in `X(i)`.
   **Fix:** verify `LEXICON X(n)` arity and indices `1..n`.

3. **Mistake:** accidental selector attachment to wrong expression.
   **Fix:** parenthesize target expression explicitly.

4. **Mistake:** putting phonological alternation in lexd entries.
   **Fix:** keep productive alternation in rewrite rules.

5. **Mistake:** expecting deterministic output order.
   **Fix:** compare sets/lists ignoring order in tests.

6. **Mistake:** generated outputs still contain grammar payload (`<...>`, `+`, `SG`, `PL`, class markers).
   **Fix:** add explicit cleanup rules and test assertions for forbidden symbols.

7. **Mistake:** productive noun plural behavior listed per noun in lexd.
   **Fix:** move productive behavior to general rewrite rules; keep only real irregulars listed separately.

8. **Mistake:** introducing many stem markers to force rule behavior for small subsets.
   **Fix:** refactor toward broader context-based rules and isolate true exceptions.

9. **Mistake:** using lexd as the main place for allomorphic logic.
   **Fix:** keep lexd focused on morphotactics; push productive allomorphy to rule cascade.

10. **Mistake:** using abstract grammatical placeholders as the main intermediate morph string (for example `+PL`) when canonical morphs would work.
    **Fix:** prefer canonical morpheme representations (`+s`, `+es`, etc.) and keep feature tags on lexical side.

### 10.1 Design QA for lexd blocks

Use this quick check:
1. Can you describe each `LEXICON` as a morphotactic class, not a spelling hack?
2. Are most entries plain stems/tags rather than precomputed inflected forms?
3. Are marker classes minimal and linguistically interpretable?
4. Are true exceptions explicit and bounded?

## 11. Minimal validation script

```python
from pyfoma import lexd

grammar = r'''
PATTERNS
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
'''

fst = lexd.compile(grammar)

assert 'shoopah' in set(fst.generate('shoop<n><pl>'))
assert 'shoop<n><comp>+blarg<n><pl>' in set(fst.analyze('shoopablargah'))
```
