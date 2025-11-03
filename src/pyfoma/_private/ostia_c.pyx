import logging
from copy import deepcopy
from collections import defaultdict
from tqdm import trange
from pyfoma.atomic import State, Transition
from pyfoma.fst import FST
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy


logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


cpdef enum Mode:
    lexicographic = 1
    data_driven = 2

cdef struct C_Transition:
    int idx
    int source_state_idx
    int target_state_idx
    # Define linked lists for each state's in/out
    int next_out_idx
    int next_in_idx

cdef struct C_State:
    int idx
    int out_head_idx
    int in_head_idx
    bint deleted

cdef class C_FST:
    cdef C_State* states
    cdef C_Transition* transitions
    cdef public int n_states
    cdef public list state_labels
    cdef public dict state_label_to_idx
    cdef public list transition_in_labels
    cdef public list transition_out_labels

    def __cinit__(self):
        self.states = NULL
        self.transitions = NULL
        self.n_states = 0
        self.state_labels = []
        self.state_label_to_idx = {}
        self.transition_in_labels = []
        self.transition_out_labels = []

    def __dealloc__(self):
        if self.states != NULL:
            free(self.states)
            self.states = NULL
        if self.transitions != NULL:
            free(self.transitions)
            self.transitions = NULL

    cpdef clone(self):
        cdef C_FST new_fst = C_FST()

        # Immutable Python objects -- shallow reuse
        new_fst.n_states = self.n_states
        new_fst.state_labels = self.state_labels
        new_fst.state_label_to_idx = self.state_label_to_idx

        # Mutable Python objects -- deepcopy
        new_fst.transition_in_labels = deepcopy(self.transition_in_labels)
        new_fst.transition_out_labels = deepcopy(self.transition_out_labels)

        # Raw copy C buffers
        new_fst.states = <C_State*> malloc(self.n_states * sizeof(C_State))
        new_fst.transitions = <C_Transition*> malloc(self.n_states * sizeof(C_Transition))
        memcpy(new_fst.states, self.states, self.n_states * sizeof(C_State))
        memcpy(new_fst.transitions, self.transitions, self.n_states * sizeof(C_Transition))
        return new_fst


cpdef ostia(
    samples: list[tuple[str | list[str], str | list[str]]],
    Mode mode=Mode.lexicographic
):
    samples_as_lists = [
        (
            list(input) if isinstance(input, str) else input,
            list(output) if isinstance(output, str) else output,
        )
        for input, output in samples
    ]
    cdef C_FST fst = build_prefix_tree(samples_as_lists)
    logger.info(f"Built prefix tree with {fst.n_states} states")
    logger.info("Converting to onward tree sequential transducer")
    convert_to_otst(fst)
    logger.info(f"Built OTST")

    logger.info("Merging states")
    cdef C_State *state
    cdef int transition_idx
    cdef C_Transition *transition
    if mode == Mode.lexicographic:
        for q_index in trange(fst.n_states, desc="Merging"):
            # Find p < q where q can merge into p
            for p_index in range(q_index):
                did_merge, fst, _ = try_merge(fst, p_index, q_index, False)
                if did_merge:
                    break
    elif mode == Mode.data_driven:
        # https://scispace.com/pdf/the-data-driven-approach-applied-to-the-ostia-algorithm-1hl6z7m4m6.pdf
        C = set([0])
        # Add all outgoing from initial state to F (frontier)
        F = set()
        state = &fst.states[0]
        transition_idx = state.out_head_idx
        while transition_idx != -1:
            transition = &fst.transitions[transition_idx]
            F.add(transition.target_state_idx)
            transition_idx = transition.next_out_idx
        # Main loop
        while len(F) > 0:
            # (p, q, score)
            top_scoring: Optional[Tuple[int, int, int]] = None
            to_move_to_C = set()
            for f in F:
                viable_merge = False
                for c in C:
                    _, _, score = try_merge(fst, c, f, True)
                    if score:
                        viable_merge = True
                        if not top_scoring or score > top_scoring[2]:
                            top_scoring = (c, f, score)
                if not viable_merge:
                    # f could not be merged anywhere, so now it goes to C
                    to_move_to_C.add(f)
            if top_scoring:
                (p, q, _) = top_scoring
                did_merge, fst, _ = try_merge(fst, p, q, False)
            for f in to_move_to_C:
                C.add(f)
                F.remove(f)
                state = &fst.states[f]
                transition_idx = state.out_head_idx
                while transition_idx != -1:
                    transition = &fst.transitions[transition_idx]
                    if transition.target_state_idx not in C:
                        F.add(transition.target_state_idx)
                    transition_idx = transition.next_out_idx
    else:
        raise ValueError("Unrecognized mode")

    # Convert back to PyFoma
    return convert_to_pyfoma(fst)



cdef C_FST build_prefix_tree(list samples: list[tuple[list[str], list[str]]]):
    """Builds a tree sequential transducer (prefix tree),
    where outputs are delayed until the word-final symbol (#)."""
    # Create states for each prefix in the inputs
    state_labels: set[tuple] = set()
    label_to_outputs: dict[tuple, str] = dict()
    for sample in samples:
        if "#" in sample[0]:
            raise ValueError("Inputs should not contain reserved `#` symbol")
        state_labels.update(tuple(p) for p in prefix(sample[0] + ["#"]))
        if (label := tuple(sample[0])) not in label_to_outputs:
            label_to_outputs[label] = "".join(sample[1])
        elif label_to_outputs[label] != "".join(sample[1]):
            logger.warning(
                f"Provided samples are ambiguous for input {label}!! Ignoring subsequent samples."
            )

    cdef C_FST fst = C_FST()
    # 1. Create empty C arrays for states and transitions
    fst.n_states = len(state_labels)
    fst.states = <C_State*> calloc(len(state_labels), sizeof(C_State))
    fst.transitions = <C_Transition*> calloc(len(state_labels), sizeof(C_Transition))

    # 2. Set up Python label lookups
    fst.state_labels = sorted(state_labels)
    fst.state_label_to_idx = dict()
    fst.transition_in_labels = []
    fst.transition_out_labels = []
    for label in fst.state_labels:
        if len(label) > 0:
            fst.transition_in_labels.append(label[-1])
            if label[-1] == "#":
                fst.transition_out_labels.append(label_to_outputs[label[:-1]])
            else:
                fst.transition_out_labels.append("")
        else:
            fst.transition_in_labels.append("")
            fst.transition_out_labels.append("")

    # 3. Initialize all indices with -1
    cdef int idx
    for idx in range(fst.n_states):
        fst.states[idx].in_head_idx = -1
        fst.states[idx].out_head_idx = -1
        fst.transitions[idx].source_state_idx = -1
        fst.transitions[idx].target_state_idx = -1
        fst.transitions[idx].next_in_idx = -1
        fst.transitions[idx].next_out_idx = -1

    # 4. Update C objects with correct IDs and transitions
    for idx, label in enumerate(fst.state_labels):
        fst.states[idx].idx = idx
        fst.state_label_to_idx[label] = idx
        fst.transitions[idx].idx = idx
        if len(label) > 0:
            # Create incoming transition
            source_state_idx = fst.state_label_to_idx[label[:-1]]
            fst.transitions[idx].source_state_idx = source_state_idx
            fst.transitions[idx].target_state_idx = idx
            # Add ingoing transition
            fst.states[idx].in_head_idx = idx
            fst.transitions[idx].next_in_idx = -1
            # Add outgoing transition (push to front of LL)
            fst.transitions[idx].next_out_idx = fst.states[source_state_idx].out_head_idx
            fst.states[source_state_idx].out_head_idx = idx
    return fst


cdef void convert_to_otst(C_FST fst):
    """Converts a tree sequential transducer into an onward tree sequential transducer (OTST)."""
    _otst_process_state(fst.states, fst) # this works because the initial state *must* be the first state in the array

cdef inline str _otst_process_state(C_State* state, C_FST fst):
    """Helper function to recursively process states"""
    if state.out_head_idx == -1:
        # No outgoing transitions
        return ""
    cdef list all_outputs = []
    # 1. Process future states recursively, and attach the "pulled forward" prefix
    cdef C_Transition* transition
    cdef str prior_out_label, new_out_label
    cdef C_State* target_state
    if state.out_head_idx != -1:
        transition = &fst.transitions[state.out_head_idx]
        while True:
            prior_out_label = fst.transition_out_labels[transition.idx]
            target_state = &fst.states[transition.target_state_idx]
            new_out_label = prior_out_label + _otst_process_state(target_state, fst)
            fst.transition_out_labels[transition.idx] = new_out_label
            all_outputs.append(new_out_label)
            if transition.next_out_idx == -1:
                break
            else:
                transition = &fst.transitions[transition.next_out_idx]

    # 2. Remove common prefix and return it
    cdef str common_prefix = lcp(list(all_outputs))
    if state.out_head_idx != -1:
        transition = &fst.transitions[state.out_head_idx]
        while True:
            prior_out_label = fst.transition_out_labels[transition.idx]
            fst.transition_out_labels[transition.idx] = prior_out_label[len(common_prefix):]
            if transition.next_out_idx == -1:
                break
            else:
                transition = &fst.transitions[transition.next_out_idx]
    return common_prefix


cdef tuple try_merge(C_FST fst, int p_idx, int q_idx, bint dry_run):
    """Attempts a merge q->p.

    If `dry_run==True`, the merge will not actually be performed, but the equivalence score (an integer) will be returned.

    Returns:
        (can_merge: bool, C_FST, score: int | None)
    """
    cdef C_State* p = &fst.states[p_idx]
    cdef C_State* q = &fst.states[q_idx]
    if p.deleted or q.deleted:
        return (False, fst, None)

    cdef int initial_output_sym_count
    if dry_run:
        initial_output_sym_count = count_output_symbols(fst)
    cdef C_FST T_bar = fst.clone()

    logger.debug(f"Trying merge '{fst.state_labels[q_idx]}' -> '{fst.state_labels[p_idx]}'")
    merge(fst, p, q)
    cdef C_Transition* t1
    cdef C_Transition* t2
    while True:
        violations = subsequent_violations(fst)
        t1, t2 = violations
        if t1 == NULL or t2 == NULL:
            break
        a = fst.transition_in_labels[t1.idx]
        v = fst.transition_out_labels[t1.idx]
        w = fst.transition_out_labels[t2.idx]
        prefixes = ["".join(s) for s in prefix(list(w))]
        if (v != w and a == "#") or (t1.target_state_idx < q_idx and v not in prefixes):
            break
        u = lcp([v, w])
        push_back(fst, v[len(u):], t1)
        push_back(fst, w[len(u):], t2)
        logger.debug(f"Second-order merge '{fst.state_labels[t2.target_state_idx]}' -> '{fst.state_labels[t1.target_state_idx]}'")
        merge(fst, &fst.states[t1.target_state_idx], &fst.states[t2.target_state_idx])

    t1, t2 = subsequent_violations(fst)
    if dry_run:
        # Always return the copy so we don't mutate
        if t1 == NULL or t2 == NULL:
            score = initial_output_sym_count - count_output_symbols(fst)
            return (True, T_bar, score)
        else:
            return (False, T_bar, None)
    else:
        if t1 == NULL or t2 == NULL:
            logger.debug("Merged successfully")
            return (True, fst, None)
        else:
            logger.debug("Bad merge, reverting")
            return (False, T_bar, None)


cdef int count_output_symbols(C_FST fst):
    cdef int total = 0
    cdef Py_ssize_t i, n = len(fst.transition_out_labels)
    for i in range(n):
        total += len(fst.transition_out_labels[i])
    return total


cdef void merge(C_FST fst, C_State* p, C_State* q):
    # All incoming to q should now go to p
    cdef int transition_idx = q.in_head_idx
    cdef C_Transition *transition
    while transition_idx != -1:
        transition = &fst.transitions[transition_idx]
        transition_idx = transition.next_in_idx
        if check_for_dupe(fst, transition.source_state_idx, p.idx, fst.transition_in_labels[transition.idx], fst.transition_out_labels[transition.idx]):
            remove_transition(fst, transition)
            continue
        transition.target_state_idx = p.idx
        # Add to head of LL for p
        transition.next_in_idx = p.in_head_idx
        p.in_head_idx = transition.idx

    # All outgoing from q should now come from p
    transition_idx = q.out_head_idx
    while transition_idx != -1:
        transition = &fst.transitions[transition_idx]
        transition_idx = transition.next_out_idx
        if check_for_dupe(fst, p.idx, transition.target_state_idx, fst.transition_in_labels[transition.idx], fst.transition_out_labels[transition.idx]):
            remove_transition(fst, transition)
            continue
        transition.source_state_idx = p.idx
        transition.next_out_idx = p.out_head_idx
        p.out_head_idx = transition.idx
    q.deleted = True

cdef (C_Transition*, C_Transition*) subsequent_violations(C_FST fst):
    """Identifies any subseqentiality violations and returns two offending edges, or (null, null)"""
    cdef C_State* state
    cdef int transition_idx
    cdef C_Transition* transition
    cdef C_Transition* conflicting_transition
    cdef dict input_to_transition_idx
    cdef str input_string, output_string
    for state_idx in range(fst.n_states):
        state = &fst.states[state_idx]
        if state.deleted:
            continue
        input_to_transition_idx = dict() # Keep dict of transitions for a given input, which should only be one unless violation
        transition_idx = state.out_head_idx
        while transition_idx != -1:
            transition = &fst.transitions[transition_idx]
            input_string = fst.transition_in_labels[transition_idx]
            if input_string in input_to_transition_idx:
                conflicting_transition = &fst.transitions[input_to_transition_idx[input_string]]
                if fst.state_labels[transition.target_state_idx] < fst.state_labels[conflicting_transition.target_state_idx]:
                    return transition, conflicting_transition
                else:
                    return conflicting_transition, transition
            input_to_transition_idx[input_string] = transition.idx
            transition_idx = transition.next_out_idx
    return NULL, NULL


cdef void push_back(C_FST fst, str suffix, C_Transition* incoming):
    """Removes a (output-side) suffix from the incoming edge and
    preprends it to all outgoing edges"""
    if suffix == "":
        return
    # Remove suffix from incoming edge
    cdef str new_label = fst.transition_out_labels[incoming.idx][:-len(suffix)]
    # Check if this modified transition would be a duplicate. If so, remove it.
    if check_for_dupe(fst, incoming.source_state_idx, incoming.target_state_idx, fst.transition_in_labels[incoming.idx], new_label):
        remove_transition(fst, incoming)
    else:
        fst.transition_out_labels[incoming.idx] = new_label

    # Prepend suffix to outgoing edges
    cdef C_State* state = &fst.states[incoming.target_state_idx]
    cdef str old_label
    cdef int outgoing_transition_idx
    cdef C_Transition* outgoing_transition
    outgoing_transition_idx = state.out_head_idx
    while outgoing_transition_idx != -1:
        outgoing_transition = &fst.transitions[outgoing_transition_idx]
        old_label = fst.transition_out_labels[outgoing_transition_idx]
        fst.transition_out_labels[outgoing_transition_idx] = suffix + old_label
        outgoing_transition_idx = outgoing_transition.next_out_idx


cdef bint check_for_dupe(C_FST fst, int source_state_idx, int target_state_idx, str in_label, str out_label):
    """Returns True if an identical transition already exists"""
    cdef C_State* state = &fst.states[source_state_idx]
    cdef int transition_idx = state.out_head_idx
    cdef C_Transition* transition
    while transition_idx != -1:
        transition = &fst.transitions[transition_idx]
        if (fst.transition_in_labels[transition_idx] == in_label and
            fst.transition_out_labels[transition_idx] == out_label and
            transition.target_state_idx == target_state_idx):
            return True
        transition_idx = transition.next_out_idx
    return False


cdef void remove_transition(C_FST fst, C_Transition* target_transition):
    """Removes a transition from the ingoing and outgoing linked lists. Does not actually delete the transition object."""
    # Remove from outgoing list
    cdef C_State* state = &fst.states[target_transition.source_state_idx]
    cdef int transition_idx = state.out_head_idx
    cdef C_Transition *transition, *previous_transition = NULL
    while transition_idx != -1:
        transition = &fst.transitions[transition_idx]
        if transition.idx == target_transition.idx:
            if previous_transition == NULL:
                # Must be the head
                state.out_head_idx = transition.next_out_idx
            else:
                previous_transition.next_out_idx = transition.next_out_idx
            transition.next_out_idx = -1
            break
        transition_idx = transition.next_out_idx
        previous_transition = transition

    # Same for incoming
    state = &fst.states[transition.target_state_idx]
    transition_idx = state.in_head_idx
    previous_transition = NULL
    while transition_idx != -1:
        transition = &fst.transitions[transition_idx]
        if transition.idx == target_transition.idx:
            if previous_transition == NULL:
                # Must be the head
                state.in_head_idx = transition.next_in_idx
            else:
                previous_transition.next_in_idx = transition.next_in_idx
            transition.next_in_idx = -1
            break
        transition_idx = transition.next_in_idx
        previous_transition = transition


cdef object convert_to_pyfoma(C_FST fst):
    cdef list states = []
    alphabet = set("".join(fst.transition_in_labels + fst.transition_out_labels))
    finalstates = set()
    cdef C_State *c_state
    for idx in range(fst.n_states):
        c_state = &fst.states[idx]
        if c_state.deleted:
            states.append(None)
            continue
        state_label = fst.state_labels[idx]
        s = State(name="".join(state_label))
        states.append(s)
        if len(state_label) > 0 and state_label[-1] == "#":
            finalstates.add(s)
    py_fst = FST(alphabet=alphabet)
    py_fst.states = set([s for s in states if s is not None])
    py_fst.initialstate = states[0]
    py_fst.finalstates = finalstates

    # Add transitions
    cdef C_Transition *transition
    cdef int transition_idx
    for idx in range(fst.n_states):
        c_state = &fst.states[idx]
        if c_state.deleted:
            continue
        transition_idx = c_state.out_head_idx
        while transition_idx != -1:
            transition = &fst.transitions[transition_idx]
            states[idx].add_transition(
                states[transition.target_state_idx],
                label=(fst.transition_in_labels[transition_idx], fst.transition_out_labels[transition_idx]),
            )
            transition_idx = transition.next_out_idx
    return py_fst


cdef list prefix(list word):
    """All prefixes of a word (as a list of chars)"""
    return [word[:i] for i in range(len(word) + 1)]


cdef str lcp(list strs):
    """Computes the LCP of a list of strings"""
    cdef Py_ssize_t n = len(strs)
    if n == 0:
        return ""
    cdef str s0 = strs[0]
    cdef Py_ssize_t i, j, m = len(s0)
    cdef Py_ssize_t len_i
    cdef str s
    for i in range(1, n):
        len_i = len(strs[i])
        if len_i < m:
            m = len_i
    for j in range(m):
        for i in range(1, n):
            if strs[i][j] != s0[j]:
                return s0[:j]
    return s0[:m]
