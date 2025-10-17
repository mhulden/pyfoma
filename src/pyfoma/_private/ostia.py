import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

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
    state_labels: set[tuple] = set()
    alphabet = set("#")
    # Track outputs so we can look them up quickly (O(n) space, O(1) time)
    label_to_outputs: dict[tuple, str] = dict()
    for sample in samples:
        if "#" in sample[0]:
            raise ValueError("Inputs should not contain reserved `#` symbol")
        state_labels.update(tuple(p) for p in prefix(sample[0] + ["#"]))
        alphabet.update(sample[0] + sample[1])
        if (label := tuple(sample[0])) not in label_to_outputs:
            label_to_outputs[label] = "".join(sample[1])
        elif label_to_outputs[label] != "".join(sample[1]):
            logger.warning(
                "Provided samples are ambiguous for input {label}!! Ignoring subsequent samples."
            )
            # raise ValueError(f"Provided samples are ambiguous for input {label}!!")

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

        new_transitions: list[Transition] = []
        for transitions in state.transitions.values():
            for transition in transitions:
                in_label, out_label = transition.label
                downstream_prefix = process_state(transition.targetstate)
                new_out_label = out_label + downstream_prefix
                transition.label = (in_label, new_out_label)
                new_transitions.append(transition)

        # Find and remove the common prefix
        transition_outputs: set[str] = {
            transition.label[1] for transition in new_transitions
        }
        common_prefix = lcp(list(transition_outputs))
        updated_transition_dict = defaultdict(lambda: set())
        for transition in new_transitions:
            new_label = (
                transition.label[0],
                transition.label[1].removeprefix(common_prefix),
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
    new_label = (incoming.label[0], incoming.label[1].removesuffix(suffix))
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


def ostia(samples: List[Tuple[Union[str, List[str]], Union[str, List[str]]]]):
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
        state_lookup = {s.name: s for s in fst.states}
        incoming_transition_lookup: dict[str, set[tuple[Transition, State]]] = {
            s.name: set() for s in fst.states if s.name is not None
        }
        for state in fst.states:
            for _, transition in state.all_transitions():
                if transition.targetstate.name is not None:
                    incoming_transition_lookup[transition.targetstate.name].add(
                        (transition, state)
                    )
        return state_lookup, incoming_transition_lookup

    # Consider merging states in lexicographic order ()
    logger.info("Merging states")
    state_names_sorted = sorted([s.name or "" for s in T.states])
    state_lookup, incoming_transition_lookup = build_lookup_tables(T)

    for q_index, q_state_name in enumerate(tqdm(state_names_sorted)):
        q = state_lookup.get(q_state_name)
        if not q or q not in T.states:  # State must have been deleted
            continue
        # Find p < q where q can merge into p
        for p_state_name in state_names_sorted[:q_index]:
            p = state_lookup.get(p_state_name)
            if p is None or p not in T.states:
                continue
            start = time.time()
            T_bar = T.__copy__()
            logger.debug(f"Copying took {time.time() - start} s")
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
                    suffix=v.removeprefix(u),
                    incoming=violating_edges[0],
                    source=source_state,
                    incoming_transition_lookup=incoming_transition_lookup,
                )
                T = push_back(
                    T,
                    suffix=w.removeprefix(u),
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
            if subseq_violations(T) is None:
                logger.debug(
                    f"Merged successfully -- FST now has {len(T.states)} states"
                )
                if q_state_name in state_lookup:
                    del state_lookup[q_state_name]
                break
            else:
                logger.debug("Aborting merge")
                start = time.time()
                T = T_bar
                state_lookup, incoming_transition_lookup = build_lookup_tables(T)
                q = state_lookup[q_state_name]
                logger.debug(f"Aborting took {time.time() - start}s")
    return T


if __name__ == "__main__":
    fst = ostia(
        [
            ("", ""),
            (["AaA"], "bb"),
            (["AaA", "AaA"], "bbc"),
            (["AaA", "AaA", "AaA"], "bbbb"),
            (["AaA", "AaA", "AaA", "AaA"], "bbbbc"),
        ]
    )
    fst.render()
