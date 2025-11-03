import logging
import time
from collections import defaultdict
from typing import Literal, Optional, cast, List, Tuple, Dict, Set, Union

from pyfoma.atomic import State, Transition
from pyfoma.fst import FST
from tqdm import tqdm

logger = logging.getLogger(__file__)


def prefix(word: List[str]):
    return [word[:i] for i in range(len(word) + 1)]


def lcp(strs: List[str]):
    """Longest common prefix"""
    assert len(strs) > 0
    prefix = ""
    for i in range(min(len(s) for s in strs)):
        if len(set(s[i] for s in strs)) == 1:
            prefix += strs[0][i]
        else:
            break
    return prefix


def build_prefix_tree(samples: List[Tuple[List[str], List[str]]]):
    """Builds a tree sequential transducer (prefix tree),
    where outputs are delayed until the word-final symbol (#)."""

    # Create states for each prefix in the inputs
    state_labels: Set[tuple] = set()
    alphabet = set("#")
    # Track outputs so we can look them up quickly (O(n) space, O(1) time)
    label_to_outputs: Dict[tuple, str] = dict()
    for sample in samples:
        if "#" in sample[0]:
            raise ValueError("Inputs should not contain reserved `#` symbol")
        state_labels.update(tuple(p) for p in prefix(sample[0] + ["#"]))
        alphabet.update(sample[0] + sample[1])
        if (label := tuple(sample[0])) not in label_to_outputs:
            label_to_outputs[label] = "".join(sample[1])
        elif label_to_outputs[label] != "".join(sample[1]):
            logger.warning(
                f"Provided samples are ambiguous for input {label}!! Ignoring subsequent samples."
            )


    states = {label: State(name="".join(label)) for label in state_labels}

    # Create transitions to form prefix tree
    for label, state in states.items():
        if len(label) >= 1:
            prior_state = states[label[:-1]]
            # Determine the output: empty string unless final state
            if label[-1] == "#":
                state.finalweight = 0
                transition_output = label_to_outputs[label[:-1]]
            else:
                transition_output = ""
            prior_state.add_transition(state, label=(label[-1], transition_output))

    fst = FST(alphabet=alphabet)
    fst.states = set(states.values())
    fst.initialstate = states[()]
    fst.finalstates = {
        s for label, s in states.items() if len(label) > 0 and label[-1] == "#"
    }
    return fst


def convert_to_otst(fst: FST):
    """Converts a tree sequential transducer into an onward tree sequential transducer (OTST)."""

    def process_state(state: State):
        if len(state.transitions) == 0:
            return ""

        new_transitions: List[Transition] = []
        for transitions in state.transitions.values():
            for transition in transitions:
                in_label, out_label = transition.label
                downstream_prefix = process_state(transition.targetstate)
                new_out_label = out_label + downstream_prefix
                transition.label = (in_label, new_out_label)
                new_transitions.append(transition)

        # Find and remove the common prefix
        transition_outputs: Set[str] = {
            transition.label[1] for transition in new_transitions
        }
        common_prefix = lcp(list(transition_outputs))
        updated_transition_dict = defaultdict(lambda: set())
        for transition in new_transitions:
            new_label = (
                transition.label[0],
                transition.label[1][len(common_prefix):],
            )
            transition.label = new_label
            updated_transition_dict[new_label].add(transition)
        state.transitions = updated_transition_dict
        return common_prefix

    process_state(fst.initialstate)
    return fst


def dedupe_transitions(
    state: State, incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]]
) -> State:
    new_transitions_dict = defaultdict(set)
    for label, transitions in state.transitions.items():
        for transition in transitions:
            if not any(
                t.targetstate == transition.targetstate
                for t in new_transitions_dict[label]
            ):
                new_transitions_dict[label].add(transition)
            else:
                # Transition is a dupe, remove from lookup
                assert transition.targetstate.name is not None
                incoming_transition_lookup[transition.targetstate.name].remove(
                    (transition, state)
                )
    state.transitions = new_transitions_dict
    return state


def merge(
    fst: FST,
    p: State,
    q: State,
    incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]],
) -> FST:
    """Merges state q into state p"""
    # Any incoming edges to q will now go to p
    needs_deduping = set()
    assert q.name is not None and p.name is not None
    assert q.name in incoming_transition_lookup and q in fst.states
    assert p.name in incoming_transition_lookup and p in fst.states
    for transition, source_state in list(incoming_transition_lookup[q.name]):
        transition.targetstate = p
        incoming_transition_lookup[p.name].add((transition, source_state))
        needs_deduping.add(source_state)

    # Copy all outgoing from q to p
    for label, transition in list(q.all_transitions()):
        assert transition.targetstate.name is not None
        incoming_transition_lookup[transition.targetstate.name].remove((transition, q))
        if label not in p.transitions:
            p.transitions[label] = set()
        p.transitions[label].add(transition)
        incoming_transition_lookup[transition.targetstate.name].add((transition, p))

    needs_deduping.add(p)
    for state in needs_deduping:
        dedupe_transitions(state, incoming_transition_lookup=incoming_transition_lookup)

    # Clean up
    del incoming_transition_lookup[q.name]
    if q is fst.initialstate:
        fst.initialstate = p
    if q in fst.finalstates:
        fst.finalstates.add(p)
        fst.finalstates.remove(q)
    fst.states.remove(q)
    del q
    return fst


def subseq_violations(fst: FST) -> Optional[Tuple[State, Tuple[Transition, Transition]]]:
    """Returns None if the transducer is subsequential, and a tuple of (source state, two edges) that violate the determinism condition if it is not subsequential"""
    for state in fst.states:
        state._transitions_by_input = None
        for _, transitions in state.transitions_by_input.items():
            if len(transitions) > 1:
                violating_transitions = list(t[1] for t in transitions)
                violating_transitions = sorted(
                    violating_transitions, key=lambda t: t.targetstate.name
                )
                return (state, tuple(violating_transitions[:2]))
    return None


def push_back(
    fst: FST,
    suffix: str,
    incoming: Transition,
    source: State,
    incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]],
) -> FST:
    """Removes a (output-side) suffix from the incoming edge and
    preprends it to all outgoing edges"""
    old_label = incoming.label
    new_label = (incoming.label[0], incoming.label[1][:-len(suffix)])
    incoming.label = new_label
    source.transitions[old_label].remove(incoming)
    source.transitions[new_label].add(incoming)
    dedupe_transitions(source, incoming_transition_lookup=incoming_transition_lookup)

    new_transition_dict = defaultdict(set)
    for label, transition in incoming.targetstate.all_transitions():
        new_label = (label[0], suffix + label[1])
        transition.label = new_label
        new_transition_dict[new_label].add(transition)
    incoming.targetstate.transitions = new_transition_dict
    incoming.targetstate = dedupe_transitions(
        incoming.targetstate, incoming_transition_lookup=incoming_transition_lookup
    )
    return fst


def ostia(
    samples: List[Tuple[Union[str, List[str]], Union[str, List[str]]]],
    merging_order: Literal["lex", "dd"],
):
    """Runs the [OSTIA](https://www.jeffreyheinz.net/classes/24F/655/materials/Oncina-et-al-1993-OSTIA.pdf) algorithm to infer an FST from a dataset.

    Args:
        samples: A list of paired input/output strings, where each string is a `List[str]` or `str`.
        merging_order: "lex" for the lexicographic order of Oncina (1991), "dd" for the data-driven approach of Oncina (1998)
    """
    samples_as_lists = [
        (
            list(input) if isinstance(input, str) else input,
            list(output) if isinstance(output, str) else output,
        )
        for input, output in samples
    ]
    logger.info("Building prefix tree")
    T = build_prefix_tree(samples_as_lists)
    logger.info(f"Built prefix tree with {len(T.states)} states")
    logger.info("Converting to onward tree sequential transducer")
    T = convert_to_otst(T)
    logger.info(f"Built OTST with {len(T.states)} states")

    def build_lookup_tables(fst: FST):
        """Helper function to build lookup tables:

        Returns:
            state_lookup (Dict[str, State]): Look up states by name
            incoming_transition_lookup (Dict[str, Set[Tuple[Transition, State]]]): Look up incoming transitions (and corr. source states) by state name
        """
        state_lookup = {s.name: s for s in fst.states}
        incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]] = {
            s.name: set() for s in fst.states if s.name is not None
        }
        for state in fst.states:
            for _, transition in state.all_transitions():
                if transition.targetstate.name is not None:
                    incoming_transition_lookup[transition.targetstate.name].add(
                        (transition, state)
                    )
        return state_lookup, incoming_transition_lookup

    logger.info("Merging states")

    def count_output_symbols(T: FST):
        count = 0
        for state in T.states:
            for t in state.all_transitions():
                count += len(t[0][1])
        return count

    # This function is shared between lex OSTIA and DD-OSTIA
    def try_merge(
        T: FST,
        q_state_name: str,
        p_state_name: str,
        state_lookup: Dict[Optional[str], State],
        incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]],
        dry_run=False,
    ) -> (
        Union[
            Tuple[
                bool, FST, Dict[Optional[str], State], Dict[str, Set[Tuple[Transition, State]]]
            ],
            Optional[int]
        ]
    ):
        """Attempts to merge state q into state p. If the merge fails, returns None (and none of the inputs should be modified).

        If `dry_run==True`, the merge will not actually be performed, but the equivalence score (an integer) will be returned.
        Returns a tuple (FST, state_lookup, incoming_transition_lookup)
        """
        logger.debug(f"try_merge with {q_state_name=}, {p_state_name=}")
        start = time.time()
        T_bar = T.__copy__()  # Keep a copy in case we need to revert changes
        initial_output_sym_count = count_output_symbols(T)
        logger.debug(f"Copying took {time.time() - start} s")

        # Make sure these states still exist
        # may have been deleted before we got to them
        q = state_lookup.get(q_state_name)
        p = state_lookup.get(p_state_name)
        if (
            not q or q not in T.states or p is None or p not in T.states
        ):  # State must have been deleted, no change
            logger.debug("Aborting due to missing q or p")
            return False, T, state_lookup, incoming_transition_lookup

        logger.debug(f"Trying merge '{q.name}' -> '{p.name}'")
        T = merge(T, p, q, incoming_transition_lookup=incoming_transition_lookup)
        # Try to merge to fix all violations
        while (violations := subseq_violations(T)) is not None:
            source_state, violating_edges = violations
            a, v = violating_edges[0].label
            s = violating_edges[0].targetstate
            w = violating_edges[1].label[1]
            t = violating_edges[1].targetstate
            if ((v != w) and (a == "#")) or (
                (s.name or "") < (q.name or "") and v not in prefix(w)
            ):
                break
            u = lcp([v, w])
            T = push_back(
                T,
                suffix=v[len(u):],
                incoming=violating_edges[0],
                source=source_state,
                incoming_transition_lookup=incoming_transition_lookup,
            )
            T = push_back(
                T,
                suffix=w[len(u):],
                incoming=violating_edges[1],
                source=source_state,
                incoming_transition_lookup=incoming_transition_lookup,
            )
            logger.debug(f"2nd-order merge '{t.name}' -> '{s.name}")
            T = merge(
                T, p=s, q=t, incoming_transition_lookup=incoming_transition_lookup
            )
            del state_lookup[t.name]

        # If T is subsequent, we're good to go to the next merge
        # If not, revert
        if not dry_run:
            if subseq_violations(T) is None:
                logger.debug(
                    f"Merged successfully -- FST now has {len(T.states)} states"
                )
                del state_lookup[q_state_name]
                return True, T, state_lookup, incoming_transition_lookup
            else:
                logger.debug("Aborting merge")
                start = time.time()
                state_lookup, incoming_transition_lookup = build_lookup_tables(T_bar)
                return False, T_bar, state_lookup, incoming_transition_lookup
        else:
            if subseq_violations(T) is None:
                new_output_symbol_count = count_output_symbols(T)
                return initial_output_sym_count - new_output_symbol_count
            else:
                return None

    # Use appropriate heuristic for merging
    state_lookup, incoming_transition_lookup = build_lookup_tables(T)
    if merging_order == "lex":
        # Consider merging states in lexicographic order
        state_names_sorted = sorted([s.name or "" for s in T.states])
        for q_index, q_state_name in enumerate(tqdm(state_names_sorted)):
            # Find p < q where q can merge into p
            for p_state_name in state_names_sorted[:q_index]:
                did_merge, T, state_lookup, incoming_transition_lookup = try_merge(  # type:ignore
                    T=T,
                    q_state_name=q_state_name,
                    p_state_name=p_state_name,
                    state_lookup=state_lookup,
                    incoming_transition_lookup=incoming_transition_lookup,
                )
                if did_merge:
                    break
    elif merging_order == "dd":
        # https://scispace.com/pdf/the-data-driven-approach-applied-to-the-ostia-algorithm-1hl6z7m4m6.pdf
        # Consolidated states
        C: Set[str] = set([T.initialstate.name])  # type:ignore
        # Frontier states
        F: Set[str] = {
            cast(str, t.targetstate.name) for _, t in T.initialstate.all_transitions()
        }
        # Table {f->name: (c->name, equiv_score(f,c))}
        score_memo_table: Dict[str, Dict[str, Optional[int]]] = defaultdict(lambda: dict())
        while len(F) > 0:
            # We could technically cut this down from O(|C||F|) to like O(most out transitions on a single state)
            # but ¯\_(ツ)_/¯
            top_scoring: Optional[Tuple[int, str, str]] = None
            to_move_to_C = set()
            for f in F:
                for c in C:
                    if f in score_memo_table and c in score_memo_table[f]:
                        # Memoized
                        score = score_memo_table[f][c]
                    else:
                        score = try_merge(
                            T=T,
                            q_state_name=f,
                            p_state_name=c,
                            state_lookup=state_lookup,
                            incoming_transition_lookup=incoming_transition_lookup,
                            dry_run=True,
                        )
                        assert isinstance(score, int) or score is None
                        score_memo_table[f][c] = score
                    if score and (not top_scoring or score > top_scoring[0]):
                        top_scoring = (score, f, c)
                if all(score is None for score in score_memo_table[f].values()):
                    # f could not be merged anywhere, so now it goes to C
                    to_move_to_C.add(f)

            # First, actually merge the top merge (if any)
            if top_scoring:
                (_, q, p) = top_scoring
                did_merge, T, state_lookup, incoming_transition_lookup = try_merge(  # type:ignore
                    T=T,
                    q_state_name=q,
                    p_state_name=p,
                    state_lookup=state_lookup,
                    incoming_transition_lookup=incoming_transition_lookup,
                )
                assert did_merge
            for f in to_move_to_C:
                C.add(f)
                F.remove(f)
                # Add downstream to frontier F
                for transition in state_lookup[f].all_transitions():
                    F.add(transition[1].targetstate.name)  # type:ignore

    return T


if __name__ == "__main__":
    fst = ostia(
        [
            ("", ""),
            ("a", "bb"),
            ("aa", "bbc"),
            ("aaa", "bbbb"),
            ("aaaa", "bbbbc"),
        ],
        merging_order="lex",
    )
    fst.render()
