---
name: pyfoma-codebase
description: Rapidly onboard to pyfoma core internals (regex compiler + FST algorithms), make safe code changes, and avoid common semantic and performance regressions in fst.py, regexparse.py, atomic.py, algorithms.py, paradigm.py, and partition_refinement.py.
---

# pyfoma-codebase skill

Use this skill when you need to modify, review, or debug pyfoma core behavior.

This skill is focused on:
- `src/pyfoma/fst.py`
- `src/pyfoma/_private/regexparse.py`
- `src/pyfoma/atomic.py`
- `src/pyfoma/algorithms.py`
- `src/pyfoma/paradigm.py`
- `src/pyfoma/_private/partition_refinement.py`
- `src/pyfoma/flag.py` (secondary but frequently involved at runtime)
- `hulden_diss/chapter_properties.tex` (theory for transducer properties APIs)
- `foma/structures.c` (reference implementation lineage)

Not in scope for this skill:
- `cfg_*` modules
- `lexd.py` internals
- OSTIA internals (unless explicitly asked)

## What This Codebase Is

pyfoma has two main layers:

1. **Regex compiler layer**  
   `FST.re(...)` -> tokenize -> parse -> compile to an `FST` object, then normalize:
   `trim -> epsilon_remove -> push_weights -> determinize_as_dfa -> minimize_as_dfa -> label_states_topology -> cleanup_sigma`.

2. **FST runtime/algorithm layer**  
   `FST` methods implement composition, union, concatenation, rewrite/context restriction, determinization, minimization, projection, application (`generate` / `analyze` / `apply`), and serialization (`foma`, `att`, JSON/JS).

## Fast Onboarding Workflow

Read in this order:
1. `references/CORE_ARCHITECTURE.md`
2. `references/SYMBOL_SEMANTICS.md`
3. `references/CHANGE_PLAYBOOK.md`

Then inspect code in this order:
1. `src/pyfoma/fst.py` (operation contracts and data flow)
2. `src/pyfoma/_private/regexparse.py` (language surface and parsing)
3. `src/pyfoma/atomic.py` (state/transition storage and indexes)
4. `src/pyfoma/algorithms.py` + `src/pyfoma/_private/partition_refinement.py`
5. `src/pyfoma/paradigm.py` and `src/pyfoma/flag.py` if task touches analysis output or flag behavior

## Core Mental Model

- `FST` objects are graph objects (`State` + `Transition`) with an explicit alphabet.
- Most public `FST` methods are **non-mutating** in practice (they return a new FST copy).
- Some helper methods mutate local copies or temporary defs internally.
- `harmonize_alphabet` is a central decorator for binary operations; wildcard handling consistency depends on it.
- `State.transitions_by_input` and `State.transitions_by_output` are lazy caches and must stay valid after mutations.

## Critical Invariants

- `.` in regex is wildcard; in compiled transitions it means "symbol outside current alphabet".
- Literal period symbol support exists via `\.` and `'.'` (internally represented as `r"\."`).
- foma wildcard mapping is now explicit:
  - foma `a:?` ~= pyfoma `('a', '.')`
  - foma `?:a` ~= pyfoma `('.', 'a')`
  - foma `@:@` ~= pyfoma `('.')` (outside-sigma identity on one tape)
  - foma `?:?` ~= pyfoma `('.', '.')`
- regex `.:.` denotes any->any (identity + non-identity). Use `.:. - .` for strict change-only behavior.
- Epsilon is `''` as a symbol in labels.
- Transition labels are tuples, even for one tape.
- Composition and wildcard behavior depend on alphabet harmonization + transition index caches.
- Rewrite internals use several helper regexes; changing symbol semantics can break them in non-obvious ways.
- Transducer property methods follow Hulden (2009):
  - functionality via `invert().compose(self).is_identity()`
  - ambiguity via path-encoding transducer + identity/nonidentity tests
  - ambiguity and nonidentity domain extraction are existential ("there exists a non-identity/non-unique path"), not universal.
- Equivalence boundary is semantic, not implementation-only:
  - both non-functional transducers => undecidable (method raises)
  - one functional, one non-functional => immediately non-equivalent.

## High-Risk Areas

- `FST.rewrite()` and `context_restrict()`:
  heavy intermediate machines, sensitive to wildcard/literal semantics and compose direction.
- `harmonize_alphabet` decorator:
  easy place to accidentally mutate caller inputs or explode transition count.
- Tokenizer changes in `regexparse.py`:
  can silently alter semantics of many built-ins (`ignore`, `rewrite`, `restrict`).
- Property APIs in `fst.py` (`is_identity`, `nonidentity_domain`, `is_functional`, `is_unambiguous`, `ambiguous_domain`, `ambiguous_part`, `unambiguous_part`, `is_equivalent`):
  coupled to subtle discrepancy/path-label encoding logic; small changes can alter decidability behavior or over/under-generate debug domains.
- Serialization (`todict/fromdict`, `tojs`, `from_fomastring/to_fomastring`, `save_att`):
  escaping, weights, and determinism/reproducibility details are easy to regress.

## Standard Validation

Primary:
```bash
PYTHONPATH=src pytest -q
```

If full discovery fails because `foma/python` native bindings are unavailable in the environment, run:
```bash
PYTHONPATH=src pytest -q tests
```

If `pytest` is unavailable in a shell, fallback:
```bash
PYTHONPATH=src python3 -m unittest -q
```

For risky parser/runtime edits, also run focused tests in `tests/test_pyfoma.py`:
- `test_tokenizer`
- `test_rewrite*`
- `test_literal_period_*`
- `test_is_identity`
- `test_nonidentity_domain`
- `test_is_functional`
- `test_is_unambiguous`
- `test_ambiguous_domain`
- `test_ambiguous_and_unambiguous_parts`
- `test_is_equivalent`
- JSON/JS and ATT related tests

## Expected Output When Using This Skill

When implementing or reviewing a change, produce:
1. A short impact summary (which semantics changed and which stayed unchanged).
2. A minimal regression test that fails before and passes after.
3. Confirmation of full or targeted test execution with command used.
4. Any residual risk callouts (performance, compatibility, ambiguity).

## Practical Guidance

- Prefer minimal, local fixes that preserve public semantics.
- If touching regex tokenization or wildcard behavior, inspect built-in function regexes in `fst.py` for accidental semantic drift.
- Keep symbol escaping/unescaping symmetric across compile, runtime apply, and serialization.
- Treat memory blow-up in rewrite paths as a first-class regression signal; isolate with compile-only probes before broad test runs.

For details, use:
- `references/CORE_ARCHITECTURE.md`
- `references/SYMBOL_SEMANTICS.md`
- `references/CHANGE_PLAYBOOK.md`
