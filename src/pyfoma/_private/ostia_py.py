import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

from pyfoma import FST, State, Transition
from tqdm import tqdm

logger = logging.getLogger(__file__)

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)


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
        label = tuple(sample[0])
        if (label) not in label_to_outputs:
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
                transition.label[1][len(common_prefix) :],
            )
            transition.label = new_label
            updated_transition_dict[new_label].add(transition)
        state.transitions = updated_transition_dict
        return common_prefix

    process_state(fst.initialstate)
    return fst


def merge(
    fst: FST,
    p: State,
    q: State,
    incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]],
) -> Tuple[bool, FST, Dict[str, Set[Tuple[Transition, State]]]]:
    """Merges state q into state p"""
    # Any incoming edges to q will now go to p
    assert q.name is not None and p.name is not None
    assert q.name in incoming_transition_lookup and q in fst.states
    assert p.name in incoming_transition_lookup and p in fst.states
    for transition, source_state in list(incoming_transition_lookup[q.name]):
        transition.targetstate = p
        incoming_transition_lookup[p.name].add((transition, source_state))

    del incoming_transition_lookup[q.name]
    success = fold(fst, p, q, incoming_transition_lookup)
    return success, fst, incoming_transition_lookup


def fold(
    fst: FST,
    p: State,
    q: State,
    incoming_transition_lookup: Dict[str, Set[Tuple[Transition, State]]],
) -> bool:
    """Recursively fold q subtree into p"""
    logger.debug(f"Folding '{q.name}' into '{p.name}'")
    if q in fst.finalstates:
        fst.finalstates.add(p)
        fst.finalstates.remove(q)

    # Move outgoing from q to p
    for label, q_transition in list(q.all_transitions()):
        assert q_transition.targetstate.name is not None
        trans_in: str = label[0]
        q_trans_out: str = label[1]

        p._invalidate_transition_indexes()
        if trans_in in p.transitions_by_input:
            p_transition = list(p.transitions_by_input[trans_in])[0][1]
            p_trans_out = p_transition.label[1]
            logger.debug(
                f"Conflicting transitions for input {trans_in}: p has ('{p_trans_out}', '{p_transition.targetstate.name}'), q has ('{q_trans_out}', '{q_transition.targetstate.name}')"
            )
            q_out_prefixes = ["".join(s) for s in prefix(list(q_trans_out))]
            if trans_in == "#" and p_trans_out != q_trans_out:
                # We've reached the end and can't resolve
                return False
            # Check if outputs are compatible
            if (
                p_transition.targetstate.name < q.name
                and p_trans_out not in q_out_prefixes
            ):
                return False
            shared_prefix = lcp([p_trans_out, q_trans_out])
            push_back(fst, p_trans_out[len(shared_prefix) :], p_transition, p)
            push_back(fst, q_trans_out[len(shared_prefix) :], q_transition, q)
            if q_transition.targetstate.name < p_transition.targetstate.name:
                s = q_transition.targetstate
                t = p_transition.targetstate
                p_transition.targetstate = s
                incoming_transition_lookup[s.name].add((p_transition, p))  # type:ignore
                incoming_transition_lookup[s.name].remove(  # type:ignore
                    (q_transition, q)
                )
            else:
                s = p_transition.targetstate
                t = q_transition.targetstate
            logger.debug(f"Recursively folding '{t.name}' into '{s.name}'.")
            del incoming_transition_lookup[t.name]  # type:ignore
            submerge_succeeded = fold(fst, s, t, incoming_transition_lookup)
            if not submerge_succeeded:
                return False
        else:
            # No existing transition, so we can just re-use this one
            del q.transitions[label]
            p.transitions[label] = set([q_transition])
            incoming_transition_lookup[q_transition.targetstate.name].remove(  # type:ignore
                (q_transition, q)
            )
            incoming_transition_lookup[q_transition.targetstate.name].add(
                (q_transition, p)
            )

    logger.debug(f"Deleting state {q.name}")
    # del incoming_transition_lookup[q.name]  # type:ignore
    fst.states.remove(q)
    del q
    return True


def push_back(
    fst: FST,
    suffix: str,
    incoming: Transition,
    source: State,
) -> FST:
    """Removes a (output-side) suffix from the incoming edge and
    preprends it to all outgoing edges"""
    if len(suffix) == 0:
        return fst
    logger.debug(
        f"Pushing suffix '{suffix}' to outgoing from {incoming.targetstate.name}"
    )
    old_label = incoming.label
    new_label = (incoming.label[0], incoming.label[1][: -len(suffix)])
    incoming.label = new_label
    del source.transitions[old_label]
    source.transitions[new_label] = set([incoming])

    new_transition_dict = defaultdict(set)
    for label, transition in incoming.targetstate.all_transitions():
        new_label = (label[0], suffix + label[1])
        transition.label = new_label
        new_transition_dict[new_label].add(transition)
    incoming.targetstate.transitions = new_transition_dict
    return fst


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
):
    """Attempts to merge state q into state p. If the merge fails, returns None (and none of the inputs should be modified).

    If `dry_run==True`, the merge will not actually be performed, but the equivalence score (an integer) will be returned.
    Returns a tuple (FST, state_lookup, incoming_transition_lookup)
    """
    logger.debug(f"⍰ try_merge with {q_state_name=}, {p_state_name=}")
    start = time.time()
    T_bar = T.__copy__()  # Keep a copy in case we need to revert changes
    logger.debug(f"Copying took {time.time() - start} s")

    # Make sure these states still exist
    # may have been deleted before we got to them
    q = state_lookup.get(q_state_name)
    p = state_lookup.get(p_state_name)
    if (
        not q or q not in T.states or p is None or p not in T.states
    ):  # State must have been deleted, no change
        logger.debug("🛑 Aborting due to missing q or p")
        return False, T, state_lookup, incoming_transition_lookup

    logger.debug(f"Trying merge '{q.name}' -> '{p.name}'")
    success, T, incoming_transition_lookup = merge(
        T, p, q, incoming_transition_lookup=incoming_transition_lookup
    )
    if success:
        logger.debug("✅ Merged successfully")
        return True, T, state_lookup, incoming_transition_lookup
    state_lookup, incoming_transition_lookup = build_lookup_tables(T_bar)
    return False, T_bar, state_lookup, incoming_transition_lookup


def ostia(samples: List[Tuple[Union[str, List[str]], Union[str, List[str]]]]):
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

    # Use appropriate heuristic for merging
    state_lookup, incoming_transition_lookup = build_lookup_tables(T)
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
    )
    fst.render()
