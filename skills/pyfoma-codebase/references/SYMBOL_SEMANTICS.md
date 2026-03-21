# Symbol Semantics and Escaping

## Wildcard vs literal period

There are two distinct concepts:

1. Regex wildcard `.`  
   Means “any symbol” in regex language.

2. Transition symbol `.` in compiled FST  
   Means “any symbol outside current alphabet” during matching logic.

Literal period symbol support uses internal token `r"\."`.

Current expected user forms for a literal period in regex:
- `\.`
- `'.'`

## foma vs pyfoma wildcard mapping

For shorthand, foma symbols:
- `@` = `@_IDENTITY_SYMBOL_@`
- `?` = `@_UNKNOWN_SYMBOL_@`

Current practical mapping:

| foma form | Meaning | pyfoma representation today |
|---|---|---|
| `a:?` | map `a` to some symbol outside sigma | `('a', '.')` |
| `?:a` | map some outside-sigma symbol to `a` | `('.', 'a')` |
| `@:@` | repeat same outside-sigma symbol | `('.')` |
| `?:?` | map outside-sigma symbol to (possibly different) outside-sigma symbol | `('.', '.')` |

Important distinction:
- `?:?` is **not** identity-by-definition.
- pyfoma now preserves this distinction with:
  - `('.')` for identity wildcard behavior
  - `('.', '.')` for unknown-to-unknown wildcard behavior

Identity-test implication:
- In identity-discrepancy algorithms, UNKNOWN-style behavior (`?` semantics / two-sided unknown mapping) is treated as non-identity-safe by default.
- Practical consequence in pyfoma tests: wildcard use on multi-tape labels (for example `a:.`, `.:a`, `.:.`) fails `is_identity()`.

Regex-level consequence:
- `.:.` means any->any (identity + non-identity).
- `.:. - .` means strict non-identity substitution wildcard.

## Runtime application behavior

`apply` tokenization is split by input type:
- `str` input: tokenized against alphabet
- `list[str]` input: treated as pre-tokenized symbols

Literal dot parity rule:
- both paths normalize `"."` to `r"\."` for literal-period matching.

Output rendering rule:
- internal `r"\."` is emitted as `\.` in string output
- `tokenize_outputs=True` returns symbol lists unchanged
- one-tape wildcard `('.')` may echo consumed outside-sigma symbols
- two-tape wildcard `('.', '.')` yields wildcard placeholder output `.` (not copied input symbol)

## Quoted symbol unescaping

Quoted regex tokens should only unescape escaped characters, not delete all backslashes.

Correct behavior example:
- `r"'\\'"` -> symbol `\`
- `r"'a\\b'"` -> symbol `a\b`
- `r"'NO\\'UN'"` -> symbol `NO'UN`

## Serialization constraints

### JSON (`todict/fromdict`)
- symbols escaped for `\` and `|`
- transitions may encode target as:
  - `dest` for zero-weight arcs
  - `[dest, weight]` for weighted arcs
- finals preserve final weights

### JS (`tojs`)
- reads both arc encodings above
- `maxlen` uses UTF-16 length

### foma format
- roundtrips wildcard distinctions:
  - `@:@` <-> `('.')`
  - `?:?` <-> `('.', '.')`

## Gotchas to watch

- Changing tokenizer semantics can break built-in helper regexes in methods like `ignore()` and `rewrite()`.
- String input and token-list input can drift if normalization is only done in one path.
- Literal dot and wildcard dot must be tested together in rewrite contexts.
- Property extraction methods (`nonidentity_domain`, `ambiguous_domain`) use internal marker symbols; ensure markers do not leak into final external alphabets.
