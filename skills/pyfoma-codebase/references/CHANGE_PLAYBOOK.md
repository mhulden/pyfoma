# Change Playbook (Safe Edits)

## 1) Triage the change type

Classify first:
- Regex surface syntax change (`regexparse.py`)
- Runtime apply/generate/analyze behavior (`fst.py`)
- Core binary operation behavior (`harmonize_alphabet`, `compose`, `union`, etc.)
- Serialization/import-export (`todict/fromdict/tojs/save_att/foma`)
- Flags/paradigm behavior (`flag.py`, `paradigm.py`)

This classification determines where regressions likely appear.

## 2) Make smallest viable change

Preferred strategy:
- keep semantics stable
- patch one layer only when possible
- avoid touching rewrite internals unless necessary
- avoid broad refactors in `fst.py` unless task explicitly requests architecture work

## 3) Add focused regression tests

For each bug:
- add one minimal reproducer test
- ensure it fails before fix and passes after fix
- place test near related section in `tests/test_pyfoma.py`

Useful hot test groups:
- tokenizer and quoting
- rewrite + directed flags
- literal period vs wildcard
- wildcard transduction semantics (`.:.`, `.:. - .`) including unknown-symbol behavior
- JSON/foma/att roundtrip behavior
- compose after mutation (transition index cache correctness)

## 4) Validate in layers

Recommended sequence:
1. run just targeted tests for changed behavior
2. run full suite

Commands:
```bash
PYTHONPATH=src pytest -q
```

Fallback:
```bash
PYTHONPATH=src python3 -m unittest -q
```

## 5) Common pitfalls checklist

- Did we accidentally mutate input operands in decorated binary ops?
- Did cache invalidation still occur on every transition mutation path?
- Did quoted token handling preserve backslashes correctly?
- Did string input and tokenized-list input behave the same for the changed symbol semantics?
- Did internal helper regexes still mean what they used to after parser changes?
- Did serialization roundtrip preserve weights and escaped symbols?

## 6) Performance/memory safety notes

- Rewrite and context restriction can explode if a helper expression broadens unexpectedly.
- For risky rewrite edits:
  - do compile-only probes first
  - avoid converting potentially huge generators to `set(...)` unless bounded
  - if needed, run probes under a memory cap and timeout

## 7) Code review priorities

When reviewing pyfoma core patches, prioritize:
1. semantic correctness for wildcard/epsilon/labels
2. mutation safety and copy contracts
3. serialization compatibility
4. performance blow-up risk in rewrite/compose
5. test coverage for reproducer and neighbors
