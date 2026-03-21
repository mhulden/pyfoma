# Core Architecture (pyfoma)

## 1) Main execution path

`FST.re(regex, defined, functions, multichar_symbols)` does:
1. optional multichar pre-quoting
2. `RegexParse(...).tokenize()`
3. parse to postfix
4. compile postfix stack into FST operations
5. normalize compiled FST:
   - `trim`
   - `epsilon_remove`
   - `push_weights`
   - `determinize_as_dfa`
   - `minimize_as_dfa`
   - `label_states_topology`
   - `cleanup_sigma`

This means most regex outputs are already normalized.

## 2) FST data model

- `FST` stores:
  - `initialstate: State`
  - `states: set[State]`
  - `finalstates: set[State]`
  - `alphabet: set[str]`
- `State` stores:
  - `transitions: dict[label_tuple, set[Transition]]`
  - lazy indexes:
    - `transitions_by_input`
    - `transitions_by_output`
- `Transition` stores:
  - `targetstate`
  - `label` (tuple)
  - `weight` (float)

## 3) Operation shape

Most public `FST` operations build and return new FSTs, not mutate caller objects.

Central categories in `fst.py`:
- Application: `generate/analyze/apply`
- Core algebra: `union/intersection/difference/compose/concatenate/product/shuffle`
- Rewriting: `rewrite/context_restrict/ignore`
- Normalization/transforms: `epsilon_remove/determinize/minimize/project/invert/reverse`
- Serialization:
  - foma format (`to_fomastring` / `from_fomastring`)
  - AT&T (`save_att`)
  - JSON/JS (`todict/fromdict/tojs`)

## 4) Key helpers by file

- `atomic.py`
  - state/transition primitives
  - reverse index and epsilon closure helpers
  - transition index cache invalidation on mutation
- `algorithms.py`
  - SCC (Tarjan), Dijkstra, OSTIA entry point
- `_private/partition_refinement.py`
  - Hopcroft-style partition refinement support for minimization
- `paradigm.py`
  - convenience extraction of lemma/tag/form triples from grammar FSTs
- `flag.py`
  - flag diacritic parsing and runtime filters used by `apply`

## 5) Rewrite architecture

`rewrite()` builds multiple helper FSTs in `defs`:
- crossproduct expansion
- aux symbol machinery
- context-restricted center
- worsener language for directed choices (`longest`, `leftmost`, `shortest`)
- final rule filtering and cleanup

It is composition-heavy and is the most sensitive code path for:
- wildcard semantics
- alphabet harmonization behavior
- intermediate automaton size/memory use

## 6) Architectural pressure points

- `fst.py` is intentionally monolithic: many concerns in one file.
- Symbol semantics are cross-cutting:
  - parser representation
  - application-time tokenization
  - serialization escapes
- Bugs often come from mismatch between those layers rather than from a single method.

## 7) Property APIs architecture (Hulden-style)

Key methods in `fst.py` are direct constructions from Hulden (2009):

- Identity:
  - `is_identity()` uses a DFS discrepancy propagation check.
  - Accept only if every successful path is identity on first/last tapes.
- Non-identity domain:
  - `nonidentity_domain()` marks discrepancy-causing arcs with a fresh marker,
    then extracts input domain of paths containing that marker.
  - This is existential: words are included if at least one path can be non-identity.
- Functionality:
  - `is_functional()` is `invert(self).compose(self).is_identity()`.
- Ambiguity:
  - `_path_encode_transducer()` rewrites output labels to unique per-path symbols.
  - `is_unambiguous()` is identity on `path_fst^-1 .o. path_fst`.
  - `ambiguous_domain()` is `dom(path_fst .o. NotID(path_fst^-1 .o. path_fst))`.
  - `ambiguous_part()` / `unambiguous_part()` are domain-splices:
    `A_amb .o. T` and `~A_amb .o. T`.
- Equivalence (`is_equivalent()`):
  - fast structural/canonical shortcuts first.
  - if both are acceptors, falls back to language symmetric-difference.
  - transducer case follows functional-equivalence boundary:
    - one functional, one non-functional => `False`
    - both non-functional => undecidable (`ValueError`)
    - both functional => equal domains and identity of inverse-compositions.
