#!/usr/bin/env python

"""Defines common algorithms over FSTs"""
from pyfoma.fst import FST, State, Transition
import pyfoma.private.partition_refinement as partition_refinement

import heapq, operator, itertools, functools
from collections import deque
from typing import Dict, Callable


# region Function Wrappers
def _copy_param(func):
    """Automatically uses a copy of the FST parameter instead of the original value, in order to avoid mutating the
    object. Use on any method that returns a modified version of an FST."""
    @functools.wraps(func)
    def wrapper_decorator(fst: 'FST', *args, **kwargs):
        return func(fst.__copy__(), *args, **kwargs)

    return wrapper_decorator


def _harmonize_alphabet(func):
    """A wrapper for expanding .-symbols when operations of arity 2 are performed.
       For example, if calculating the union of FSM1 and FSM2, and both contain
       .-symbols, the transitions with . are expanded to include the symbols that
       are present in the other FST."""
    @functools.wraps(func)
    def wrapper_decorator(fst1: 'FST', fst2: 'FST', **kwargs):
        for A, B in [(fst1, fst2), (fst2, fst1)]:
            if '.' in A.alphabet and (A.alphabet - {'.'}) != (B.alphabet - {'.'}):
                Aexpand = B.alphabet - A.alphabet - {'.', ''}
                if A == fst2:
                    A, _ = fst2.copy_filtered()
                    fst2 = A # Need to copy to avoid mutating other
                for s, l, t in list(A.all_transitions(A.states)):
                    if '.' in l:
                        for sym in Aexpand:
                            newl = tuple(lbl if lbl != '.' else sym for lbl in l)
                            s.add_transition(t.targetstate, newl, t.weight)

        newalphabet = fst1.alphabet | fst2.alphabet
        value = func(fst1, fst2, **kwargs)
        # Do something after
        value.alphabet = newalphabet
        return value
    return wrapper_decorator

# endregion

# region Algorithms


@_copy_param
def trimmed(fst: 'FST') -> 'FST':
    """Returns a modified FST, removing states that aren't both accessible and coaccessible."""
    return filtered_coaccessible(filtered_accessible(fst))


@_copy_param
def filtered_accessible(fst: 'FST') -> 'FST':
    """Returns a modified FST, removing states that are not on a path from the initial state."""
    explored = { fst.initialstate }
    stack = deque([fst.initialstate])
    while stack:
        source = stack.pop()
        for label, transition in source.all_transitions():
            if transition.targetstate not in explored:
                explored.add(transition.targetstate)
                stack.append(transition.targetstate)

    fst.states = explored
    fst.finalstates &= fst.states
    return fst


@_copy_param
def filtered_coaccessible(fst: 'FST') -> 'FST':
    """Returns a modified FST, removing states and transitions to states that have no path to a final state."""
    explored = {fst.initialstate}
    stack = deque([fst.initialstate])
    inverse = {s: set() for s in fst.states}  # store all preceding arcs here
    while stack:
        source = stack.pop()
        for target in source.all_targets():
            inverse[target].add(source)
            if target not in explored:
                explored.add(target)
                stack.append(target)

    stack = deque([s for s in fst.finalstates])
    coaccessible = {s for s in fst.finalstates}
    while stack:
        source = stack.pop()
        for previous in inverse[source]:
            if previous not in coaccessible:
                coaccessible.add(previous)
                stack.append(previous)

    coaccessible.add(fst.initialstate)  # Let's make an exception for the initial
    for s in fst.states:  # Need to also remove transitions to non-coaccessibles
        s.remove_transitions_to_targets(fst.states - coaccessible)

    fst.states &= coaccessible
    fst.finalstates &= fst.states
    return fst


def scc(fst: 'FST') -> set:
    """Calculate the strongly connected components of an FST.

       This is a basic implementation of Tarjan's (1972) algorithm.
       Tarjan, R. E. (1972), "Depth-first search and linear graph algorithms",
       SIAM Journal on Computing, 1 (2): 146–160.

       Returns a set of frozensets of states, one frozenset for each SCC."""

    index = 0
    S = deque([])
    sccs, indices, lowlink, onstack = set(), {}, {}, set()

    def _strongconnect(state):
        nonlocal index, indices, lowlink, onstack, sccs
        indices[state] = index
        lowlink[state] = index
        index += 1
        S.append(state)
        onstack.add(state)
        targets = state.all_targets()
        for target in targets:
            if target not in indices:
                _strongconnect(target)
                lowlink[state] = min(lowlink[state], lowlink[target])
            elif target in onstack:
                lowlink[state] = min(lowlink[state], indices[target])
        if lowlink[state] == indices[state]:
            currscc = set()
            while True:
                target = S.pop()
                onstack.remove(target)
                currscc.add(target)
                if state == target:
                    break
            sccs.add(frozenset(currscc))

    for s in fst.states:
        if s not in indices:
            _strongconnect(s)

    return sccs


@_copy_param
def pushed_weights(fst: 'FST') -> 'FST':
    """Returns a modified FST, pushing weights toward the initial state. Calls dijkstra and maybe scc."""
    potentials = {s:dijkstra(fst, s) for s in fst.states}
    for s, _, t in fst.all_transitions(fst.states):
        t.weight += potentials[t.targetstate] - potentials[s]
    for s in fst.finalstates:
        s.finalweight = s.finalweight - potentials[s]
    residualweight = potentials[fst.initialstate]
    if residualweight != 0.0:
        # Add residual to all exits of initial state SCC and finals in that SCC
        mainscc = next(s for s in scc(fst) if fst.initialstate in s)
        for s, _, t in fst.all_transitions(mainscc):
            if t.targetstate not in mainscc: # We're exiting the main SCC
                t.weight += residualweight
        for s in mainscc & fst.finalstates: # Add res w to finals in initial SCC
            s.finalweight += residualweight
    return fst


@_copy_param
def mapped_labels(fst: 'FST', map: dict) -> 'FST':
    """Returns a modified FST, relabeling the transducer with new labels from dictionary mapping.

    Example: ``map_labels(myfst, {'a':'', 'b':'a'})``"""
    for s in fst.states:
        newlabelings = []
        for lbl in s.transitions.keys():
            if any(l in lbl for l in map):
                newlabel = tuple(map[lbl[i]] if lbl[i] in map else lbl[i] for i in range(len(lbl)))
                newlabelings.append((lbl, newlabel))
        for old, new in newlabelings:
            s.rename_label(old, new)
    fst.alphabet = fst.alphabet - map.keys() | set(map.values()) - {''}
    return fst


def epsilon_removed(fst: 'FST') -> 'FST':
    """Returns a modified FST, creating new epsilon-free FSM equivalent to original."""
    # For each state s, figure out the min-cost w' to hop to a state t with epsilons
    # Then, add the (non-e) transitions of state t to s, adding w' to their cost
    # Also, if t is final and s is not, make s final with cost t.finalweight ⊗ w'
    # If s and t are both final, make s's finalweight s.final ⊕ (t.finalweight ⊗ w')

    eclosures = {s:epsilon_closure(fst, s) for s in fst.states}
    if all(len(ec) == 0 for ec in eclosures.values()): # bail, no epsilon transitions
        return fst.__copy__()
    newfst, mapping = fst.copy_filtered(labelfilter = lambda lbl: any(len(sublabel) != 0 for sublabel in lbl))
    for state, ec in eclosures.items():
        for target, cost in ec.items():
            # copy target's transitions to source
            for label, t in target.all_transitions():
                if all(len(sublabel) == 0 for sublabel in label): # is epsilon: skip
                    continue
                mapping[state].add_transition(mapping[t.targetstate], label, cost + t.weight)
            if target in fst.finalstates:
                if state not in fst.finalstates:
                    newfst.finalstates.add(mapping[state])
                    mapping[state].finalweight = 0.0
                mapping[state].finalweight += cost + target.finalweight
    return newfst


def epsilon_closure(fst: 'FST', state) -> dict:
    """Finds, for a state the set of states reachable by epsilon-hopping."""
    explored, cntr = {}, itertools.count()
    q = [(0.0, next(cntr), state)]
    while q:
        cost, _, source = heapq.heappop(q)
        if source not in explored:
            explored[source] = cost
            for target, weight in source.all_epsilon_targets_cheapest().items():
                heapq.heappush(q, (cost + weight, next(cntr), target))
    explored.pop(state) # Remove the state where we started from
    return explored


def dijkstra(fst: 'FST', state) -> float:
    """The cost of the cheapest path from state to a final state. Go Edsger!"""
    explored, cntr = {state}, itertools.count()  # decrease-key is for wusses
    Q = [(0.0, next(cntr), state)] # Middle is dummy cntr to avoid key ties
    while Q:
        w, _ , s = heapq.heappop(Q)
        if s == None:       # First None we pull out is the lowest-cost exit
            return w
        explored.add(s)
        if s in fst.finalstates:
            # now we push a None state to signal the exit from a final
            heapq.heappush(Q, (w + s.finalweight, next(cntr), None))
        for trgt, cost in s.all_targets_cheapest().items():
            if trgt not in explored:
                heapq.heappush(Q, (cost + w, next(cntr), trgt))
    return float("inf")


@_copy_param
def labelled_states_topology(fst: 'FST', mode = 'BFS') -> 'FST':
    """Returns a modified FST, topologically sorting and labelling states with numbers.
    Keyword arguments:
    mode -- 'BFS', i.e. breadth-first search by default. 'DFS' is depth-first.
    """
    cntr = itertools.count()
    Q = deque([fst.initialstate])
    inqueue = {fst.initialstate}
    while Q:
        s = Q.popleft() if mode == 'BFS' else Q.pop()
        s.name = str(next(cntr))
        for label, t in s.all_transitions():
            if t.targetstate not in inqueue:
                Q.append(t.targetstate)
                inqueue.add(t.targetstate)
    return fst


def words_nbest(fst: 'FST', n) -> list:
    """Finds the n cheapest word in an FST, returning a list."""
    return list(itertools.islice(words_cheapest(fst), n))


def words_cheapest(fst: 'FST'):
    """A generator to yield all words in order of cost, cheapest first."""
    cntr = itertools.count()
    Q = [(0.0, next(cntr), fst.initialstate, [])]
    while Q:
        cost, _, s, seq = heapq.heappop(Q)
        if s is None:
            yield cost, seq
        else:
            if s in fst.finalstates:
                heapq.heappush(Q, (cost + s.finalweight, next(cntr), None, seq))
            for label, t in s.all_transitions():
                heapq.heappush(Q, (cost + t.weight, next(cntr), t.targetstate, seq + [label]))


@_copy_param
def determinized_unweighted(fst: 'FST') -> 'FST':
    """Returns a modified FST, determinized with all zero weights."""
    return determinized(fst, staterep = lambda s, w: (s, 0.0), oplus = lambda *x: 0.0)


def determinized_as_dfa(fst: 'FST') -> 'FST':
    """Returns a modified FST, determinized as a DFA with weight as part of label, then apply unweighted det."""
    newfst = fst.copy_mod(modlabel = lambda l, w: l + (w,), modweight = lambda l, w: 0.0)
    determinized = determinized_unweighted(newfst) # run det, then move weights back
    return determinized.copy_mod(modlabel = lambda l, _: l[:-1], modweight = lambda l, _: l[-1])


def determinized(fst: 'FST', staterep = lambda s, w: (s, w), oplus = min) -> 'FST':
    """Returns a modified FST, by weighted determinization of FST."""
    newfst = FST(alphabet = fst.alphabet.copy())
    firststate = frozenset({staterep(fst.initialstate, 0.0)})
    statesets = {firststate:newfst.initialstate}
    if fst.initialstate in fst.finalstates:
        newfst.finalstates = {newfst.initialstate}
        newfst.initialstate.finalweight = fst.initialstate.finalweight

    Q = deque([firststate])
    while Q:
        currentQ = Q.pop()
        collectlabels = {} # temp dict of label:all transitions {(src1, trans1),...}
        for s, _ in currentQ:
            for label, transitions in s.transitions.items():
                for t in transitions:
                    collectlabels[label] = collectlabels.get(label, set()) | {(s, t)}

        residuals = {s:r for s, r in currentQ}
        for label, tset in collectlabels.items():
            # wprime is the maximum amount the matching outgoing arcs share -
            # some paths may therefore accumulate debt which needs to be passed on
            # and stored in the next state representation for future discharge
            wprime = oplus(t.weight + residuals[s] for s, t in tset)
            # Note the calculation of the weight debt we pass forward, reused w/ finals below
            newQ = frozenset(staterep(t.targetstate, t.weight + residuals[s] - wprime) for s, t in tset)
            if newQ not in statesets:
                Q.append(newQ)
                newstate = State()
                statesets[newQ] = newstate
                newfst.states.add(statesets[newQ])
                #statesets[newQ].name = {(s.name, w) if w != 0.0 else s.name for s, w in newQ}
            else:
                newstate = statesets[newQ]
            statesets[currentQ].add_transition(newstate, label, wprime)
            if any(t.targetstate in fst.finalstates for _, t in tset):
                newfst.finalstates.add(newstate)
                # State was final, so we discharge the maximum debt we can
                newstate.finalweight = oplus(t.targetstate.finalweight + t.weight + \
                    residuals[s] - wprime for s, t in tset if t.targetstate in fst.finalstates)
    return newfst


def minimized_as_dfa(fst: 'FST') -> 'FST':
    """Returns a modified FST, minimized as a DFA with weight as part of label, then apply unweighted min."""
    newfst = fst.copy_mod(modlabel = lambda l, w: l + (w,), modweight = lambda l, w: 0.0)
    minimized_fst = minimized(newfst) # minimize, and shift weights back
    return minimized_fst.copy_mod(modlabel = lambda l, _: l[:-1], modweight = lambda l, _: l[-1])


@_copy_param
def minimized(fst: 'FST') -> 'FST':
    """Returns a modified FST, minimized by constrained reverse subset construction, Hopcroft-ish."""
    reverse_index = create_reverse_index(fst)
    finalset, nonfinalset = fst.finalstates.copy(), fst.states - fst.finalstates
    initialpartition = [x for x in (finalset, nonfinalset) if len(x) > 0]
    P = partition_refinement.PartitionRefinement(initialpartition)
    Agenda = {id(x) for x in (finalset, nonfinalset) if len(x) > 0}
    while Agenda:
        S = P.sets[Agenda.pop()] # convert id to the actual set it corresponds to
        for label, sourcestates in find_sourcestates(fst, reverse_index, S):
            splits = P.refine(sourcestates) # returns list of (A & S, A - S) tuples
            Agenda |= {new for new, _ in splits} # Only place A & S on Agenda
    equivalenceclasses = P.astuples()
    if len(equivalenceclasses) == len(fst.states):
        return fst # we were already minimal, no need to reconstruct

    return merging_equivalent_states(fst, equivalenceclasses)


def merging_equivalent_states(fst: 'FST', equivalenceclasses: set) -> 'FST':
    """Merge equivalent states given as a set of sets."""
    eqmap = {s[i]:s[0] for s in equivalenceclasses for i in range(len(s))}
    representerstates = set(eqmap.values())
    newfst = FST(alphabet = fst.alphabet.copy())
    statemap = {s:State() for s in fst.states if s in representerstates}
    newfst.initialstate = statemap[eqmap[fst.initialstate]]
    for s, lbl, t in fst.all_transitions(fst.states):
        if s in representerstates:
            statemap[s].add_transition(statemap[eqmap[t.targetstate]], lbl, t.weight)
    newfst.states = set(statemap.values())
    newfst.finalstates = {statemap[s] for s in fst.finalstates if s in representerstates}
    for s in fst.finalstates:
        if s in representerstates:
            statemap[s].finalweight = s.finalweight
    return newfst


def find_sourcestates(fst: 'FST', index, stateset):
    """Create generator that yields sourcestates for a set of target states.
       Yields the label, and the set of sourcestates."""
    all_labels = {l for s in stateset for l in index[s].keys()}
    for l in all_labels:
        sources = set()
        for state in stateset:
            if l in index[state]:
                sources |= index[state][l]
        yield l, sources


def create_reverse_index(fst: 'FST') -> dict:
    """Returns dictionary of transitions in reverse (indexed by state)."""
    idx = {s:{} for s in fst.states}
    for s, lbl, t in fst.all_transitions(fst.states):
        idx[t.targetstate][lbl] = idx[t.targetstate].get(lbl, set()) | {s}
    return idx


def minimized_brz(fst: 'FST') -> 'FST':
    """Returns a modified FST, minimized through Brzozowski's trick."""
    return determinized(reversed_e(determinized(reversed_e(epsilon_removed(fst)))))


def kleene_closure(fst: 'FST', mode = 'star') -> 'FST':
    """Returns a modified FST, applying self*. No epsilons here. If mode == 'plus', calculate self+."""
    q1 = {k:State() for k in fst.states}
    newfst = FST(alphabet = fst.alphabet.copy())

    for lbl, t in fst.initialstate.all_transitions():
        newfst.initialstate.add_transition(q1[t.targetstate], lbl, t.weight)

    for s, lbl, t in fst.all_transitions(fst.states):
        q1[s].add_transition(q1[t.targetstate], lbl, t.weight)

    for s in fst.finalstates:
        for lbl, t in fst.initialstate.all_transitions():
            q1[s].add_transition(q1[t.targetstate], lbl, t.weight)
        q1[s].finalweight = s.finalweight

    newfst.finalstates = {q1[s] for s in fst.finalstates}
    if mode != 'plus' or fst.initialstate in fst.finalstates:
        newfst.finalstates |= {newfst.initialstate}
        newfst.initialstate.finalweight = 0.0
    newfst.states = set(q1.values()) | {newfst.initialstate}
    return newfst


def kleene_star(fst: 'FST') -> 'FST':
    """Returns a modified FST, applying self*."""
    return kleene_closure(fst, mode='star')


def kleene_plus(fst: 'FST') -> 'FST':
    """Returns a modified FST, applying self+."""
    return kleene_closure(fst, mode='plus')

@_copy_param
def added_weight(fst: 'FST', weight) -> 'FST':
    """Returns a modified FST, adding weight to the set of final states in the FST."""
    for s in fst.finalstates:
        s.finalweight += weight
    return fst


@_copy_param
def optional(fst: 'FST') -> 'FST':
    """Returns a modified FST, calculated as T|'' ."""
    if fst.initialstate in fst.finalstates:
        return fst
    newinitial = State()

    for lbl, t in fst.initialstate.all_transitions():
        newinitial.add_transition(t.targetstate, lbl, t.weight)

    fst.initialstate = newinitial
    fst.states.add(newinitial)
    fst.finalstates.add(newinitial)
    newinitial.finalweight = 0.0
    return fst


@_harmonize_alphabet
def concatenate(fst1: 'FST', fst2: 'FST') -> 'FST':
    """Concatenation of T1T2. No epsilons. May produce non-accessible states."""
    ocopy, _ = fst2.copy_filtered() # Need to copy since self may equal other
    q1q2 = {k:State() for k in fst1.states | ocopy.states}

    for s, lbl, t in fst1.all_transitions(q1q2.keys()):
        q1q2[s].add_transition(q1q2[t.targetstate], lbl, t.weight)
    for s in fst1.finalstates:
        for lbl2, t2 in ocopy.initialstate.all_transitions():
            q1q2[s].add_transition(q1q2[t2.targetstate], lbl2, t2.weight + s.finalweight)

    newfst = FST()
    newfst.initialstate = q1q2[fst1.initialstate]
    newfst.finalstates = {q1q2[f] for f in ocopy.finalstates}
    for s in ocopy.finalstates:
        q1q2[s].finalweight = s.finalweight
    if ocopy.initialstate in ocopy.finalstates:
        newfst.finalstates |= {q1q2[f] for f in fst1.finalstates}
        for f in fst1.finalstates:
            q1q2[f].finalweight = f.finalweight + ocopy.initialstate.finalweight
    newfst.states = set(q1q2.values())
    return newfst


@_harmonize_alphabet
def cross_product(fst1: 'FST', fst2: 'FST', optional: bool = False) -> 'FST':
    """Perform the cross-product of T1, T2 through composition.
       Keyword arguments:
       optional -- if True, calculates T1:T2 | T1."""
    newfst_a = fst1.copy_mod(modlabel = lambda l, _: l + ('',))
    newfst_b = fst2.copy_mod(modlabel = lambda l, _: ('',) + l)
    if optional == True:
        return union(compose(newfst_a, newfst_b), fst1)
    else:
        return compose(newfst_a, newfst_b)


@_harmonize_alphabet
def compose(fst1: 'FST', fst2: 'FST') -> 'FST':
    """Composition of A,B; will expand an acceptor into 2-tape FST on-the-fly."""

    def _mergetuples(x: tuple, y: tuple) -> tuple:
        if len(x) == 1:
            t = x + y[1:]
        elif len(y) == 1:
            t = x[:-1] + y
        else:
            t = x[:-1] + y[1:]
        if all(t[i] == t[0] for i in range(len(t))):
            t = (t[0],)
        return t

    # Mode 0: allow A=x:0 B=0:y (>0), A=x:y B=y:z (>0), A=x:0 B=wait (>1) A=wait 0:y (>2)
    # Mode 1: x:0 B=wait (>1), x:y y:z (>0)
    # Mode 2: A=wait 0:y (>2), x:y y:z (>0)

    newfst = FST()
    Q = deque([(fst1.initialstate, fst2.initialstate, 0)])
    S = {(fst1.initialstate, fst2.initialstate, 0): newfst.initialstate}
    while Q:
        A, B, mode = Q.pop()
        currentstate = S[(A, B, mode)]
        currentstate.name = "({},{},{})".format(A.name, B.name, mode)
        if A in fst1.finalstates and B in fst2.finalstates:
            newfst.finalstates.add(currentstate)
            currentstate.finalweight = A.finalweight + B.finalweight # TODO: oplus
        for matchsym in A.transitionsout.keys():
            if mode == 0 or matchsym != '': # A=x:y B=y:z, or x:0 0:y (only in mode 0)
                for outtrans in A.transitionsout.get(matchsym, ()):
                    for intrans in B.transitionsin.get(matchsym, ()):
                        target1 = outtrans[1].targetstate # Transition
                        target2 = intrans[1].targetstate  # Transition
                        if (target1, target2, 0) not in S:
                            Q.append((target1, target2, 0))
                            S[(target1, target2, 0)] = State()
                            newfst.states.add(S[(target1, target2, 0)])
                        # Keep intermediate
                        # currentstate.add_transition(S[(target1, target2)], outtrans[1].label[:-1] + intrans[1].label, outtrans[1].weight + intrans[1].weight)
                        newlabel = _mergetuples(outtrans[1].label, intrans[1].label)
                        currentstate.add_transition(S[(target1, target2, 0)], newlabel, outtrans[1].weight + intrans[1].weight)
        for outtrans in A.transitionsout.get('', ()): # B waits
            if mode == 2:
                break
            target1, target2 = outtrans[1].targetstate, B
            if (target1, target2, 1) not in S:
                Q.append((target1, target2, 1))
                S[(target1, target2, 1)] = State()
                newfst.states.add(S[(target1, target2, 1)])
            newlabel = outtrans[1].label
            currentstate.add_transition(S[(target1, target2, 1)], newlabel, outtrans[1].weight)
        for intrans in B.transitionsin.get('', ()): # A waits
            if mode == 1:
                break
            target1, target2 = A, intrans[1].targetstate
            if (target1, target2, 2) not in S:
                Q.append((target1, target2, 2))
                S[(target1, target2, 2)] = State()
                newfst.states.add(S[(target1, target2, 2)])
            newlabel = intrans[1].label
            currentstate.add_transition(S[(target1, target2, 2)], newlabel, intrans[1].weight)
    return newfst


@_copy_param
def inverted(fst: 'FST') -> 'FST':
    """Returns a modified FST, calculating the inverse of a transducer, i.e. flips label tuples around."""
    for s in fst.states:
        s.transitions  = {lbl[::-1]:tr for lbl, tr in s.transitions.items()}
    return fst


def ignore(fst1: 'FST', fst2: 'FST') -> 'FST':
    """A, ignoring intervening instances of B."""
    newfst = FST.re("$^output($A @ ('.'|'':$B)*)", {'A': fst1, 'B': fst2})
    return newfst


def rewritten(fst: 'FST', *contexts, **flags) -> 'FST':
    """Returns a modified FST, rewriting self in contexts in parallel, controlled by flags."""
    defs = {'crossproducts': fst}
    defs['br'] = FST.re("'@<@'|'@>@'")
    defs['aux'] = FST.re(". - ($br|#)", defs)
    defs['dotted'] = FST.re(".*-(.* '@<@' '@>@' '@<@' '@>@' .*)")
    defs['base'] = FST.re("$dotted @ # ($aux | '@<@' $crossproducts '@>@')* #", defs)
    if len(contexts) > 0:
        center = FST.re("'@<@' (.-'@>@')* '@>@'")
        lrpairs = ([l.ignore(defs['br']), r.ignore(defs['br'])] for l,r in contexts)
        defs['rule'] = center.context_restrict(*lrpairs, rewrite = True).compose(defs['base'])
    else:
        defs['rule'] = defs['base']
    defs['remrewr'] = FST.re("'@<@':'' (.-'@>@')* '@>@':''") # worsener
    worseners = [FST.re(".* $remrewr (.|$remrewr)*", defs)]
    if flags.get('longest', False) == 'True':
        worseners.append(FST.re(".* '@<@' $aux+ '':('@>@' '@<@'?) $aux ($br:''|'':$br|$aux)* .*", defs))
    if flags.get('leftmost', False) == 'True':
        worseners.append(FST.re(\
             ".* '@<@':'' $aux+ ('':'@<@' $aux* '':'@>@' $aux+ '@>@':'' .* | '':'@<@' $aux* '@>@':'' $aux* '':'@>@' .*)", defs))
    if flags.get('shortest', False) == 'True':
        worseners.append(FST.re(".* '@<@' $aux* '@>@':'' $aux+ '':'@>@' .*", defs))
    defs['worsen'] = functools.reduce(lambda x, y: x.union(y), worseners).determinize_unweighted().minimize()
    defs['rewr'] = FST.re("$^output($^input($rule) @ $worsen)", defs)
    final = FST.re("(.* - $rewr) @ $rule", defs)
    newfst = final.map_labels({s:'' for s in ['@<@','@>@','#']}).epsilon_remove().determinize_as_dfa().minimize()
    return newfst


@_copy_param
def context_restricted(fst: 'FST', *contexts, rewrite = False) -> 'FST':
    """Returns a modified FST, where self only allowed in the context L1 _ R1, or ... , or  L_n _ R_n."""
    for fsm in itertools.chain.from_iterable(contexts):
        fsm.alphabet.add('@=@') # Add aux sym to contexts so they don't match .
    fst.alphabet.add('@=@')    # Same for self
    if not rewrite:
        cs = (FST.re("$lc '@=@' (.-'@=@')* '@=@' $rc", \
             {'lc':lc.copy_mod().map_labels({'#': '@#@'}),\
             'rc':rc.copy_mod().map_labels({'#': '@#@'})}) for lc, rc in contexts)
    else:
        cs = (FST.re("$lc '@=@' (.-'@=@')* '@=@' $rc", {'lc':lc, 'rc':rc}) for lc, rc in contexts)
    cunion = functools.reduce(lambda x, y: x.union(y), cs).determinize().minimize()
    r = FST.re("(.-'@=@')* '@=@' $c '@=@' (.-'@=@')* - ((.-'@=@')* $cunion (.-'@=@')*)",\
                   {'c':fst, 'cunion':cunion})
    r = r.map_labels({'@=@':''}).epsilon_remove().determinize_as_dfa().minimize()
    for fsm in itertools.chain.from_iterable(contexts):
        fsm.alphabet -= {'@=@'} # Remove aux syms from contexts
    r = FST.re(".? (.-'@#@')* .? - $r", {'r': r})
    newfst = r.map_labels({'@#@':''}).epsilon_remove().determinize_as_dfa().minimize()
    return newfst


@_copy_param
def projected(fst: 'FST', dim = 0) -> 'FST':
    """Returns a modified FST, by projecting fst. dim = -1 will get output proj regardless of # of tapes."""
    sl = slice(-1, None) if dim == -1 else slice(dim, dim+1)
    newalphabet = set()
    for s in fst.states:
        newtransitions = {}
        for lbl, tr in s.transitions.items():
            newtransitions[lbl[sl]] = newtransitions.get(lbl[sl], set()) | tr
            for t in tr:
                t.label = lbl[sl]
                newalphabet |= {sublabel for sublabel in lbl[sl]}
        s.transitions = newtransitions
    fst.alphabet = newalphabet
    return fst


def reversed(fst: 'FST') -> 'FST':
    """Returns a modified FST, reversing the FST, epsilon-free."""
    newfst = FST(alphabet = fst.alphabet.copy())
    newfst.initialstate = State()
    mapping = {k:State() for k in fst.states}
    newfst.states = set(mapping.values()) | {newfst.initialstate}
    newfst.finalstates = {mapping[fst.initialstate]}
    if fst.initialstate in fst.finalstates:
        newfst.finalstates.add(newfst.initialstate)
        newfst.initialstate.finalweight = fst.initialstate.finalweight
    mapping[fst.initialstate].finalweight = 0.0

    for s, lbl, t in fst.all_transitions(fst.states):
        mapping[t.targetstate].add_transition(mapping[s], lbl, t.weight)
        if t.targetstate in fst.finalstates:
            newfst.initialstate.add_transition(mapping[s], lbl, t.weight + \
                                               t.targetstate.finalweight)
    return newfst


def reversed_e(fst: 'FST') -> 'FST':
    """Returns a modified FST, reversing the FST, using epsilons."""
    newfst = FST(alphabet = fst.alphabet.copy())
    newfst.initialstate = State(name = tuple(k.name for k in fst.finalstates))
    mapping = {k:State(name = k.name) for k in fst.states}
    for t in fst.finalstates:
        newfst.initialstate.add_transition(mapping[t], ('',), t.finalweight)

    for s, lbl, t in fst.all_transitions(fst.states):
        mapping[t.targetstate].add_transition(mapping[s], lbl, t.weight)

    newfst.states = set(mapping.values()) | {newfst.initialstate}
    newfst.finalstates = {mapping[fst.initialstate]}
    mapping[fst.initialstate].finalweight = 0.0
    return newfst


@_harmonize_alphabet
def union(fst1: 'FST', fst2: 'FST') -> 'FST':
    """Epsilon-free calculation of union of fst1 and fst2."""
    mapping = {k:State() for k in fst1.states | fst2.states}
    newfst = FST() # Get new initial state
    newfst.states = set(mapping.values()) | {newfst.initialstate}
    # Copy all transitions from old initial states to new initial state
    for lbl, t in itertools.chain(fst1.initialstate.all_transitions(), fst2.initialstate.all_transitions()):
        newfst.initialstate.add_transition(mapping[t.targetstate], lbl, t.weight)
    # Also add all transitions from old FSMs to new FSM
    for s, lbl, t in itertools.chain(fst1.all_transitions(fst1.states), fst2.all_transitions(fst2.states)):
        mapping[s].add_transition(mapping[t.targetstate], lbl, t.weight)
    # Make old final states final in new FSM
    for s in fst1.finalstates | fst2.finalstates:
        newfst.finalstates.add(mapping[s])
        mapping[s].finalweight = s.finalweight
    # If either initial state was final, make new initial final w/ weight min(f1w, f2w)
    newfst.finalstates = {mapping[s] for s in fst1.finalstates|fst2.finalstates}
    if fst1.initialstate in fst1.finalstates or fst2.initialstate in fst2.finalstates:
        newfst.finalstates.add(newfst.initialstate)
        newfst.initialstate.finalweight = min(fst1.initialstate.finalweight, fst2.initialstate.finalweight)

    return newfst


def intersection(fst1: 'FST', fst2: 'FST') -> 'FST':
    """Intersection of self and other. Uses the product algorithm."""
    return product(fst1, fst2, finalf = all, oplus = operator.add, pathfollow = lambda x,y: x & y)


def difference(fst1: 'FST', fst2: 'FST') -> 'FST':
    """Returns self-other. Uses the product algorithm."""
    return product(fst1, fst2, finalf = lambda x: x[0] and not x[1],\
                       oplus = lambda x,y: x, pathfollow = lambda x,y: x)


@_harmonize_alphabet
def product(fst1: 'FST', fst2: 'FST', finalf = any, oplus = min, pathfollow = lambda x,y: x|y) -> 'FST':
    """Generates the product FST from fst1, fst2. The helper functions by default
       produce fst1|fst2."""
    newfst = FST()
    Q = deque([(fst1.initialstate, fst2.initialstate)])
    S = {(fst1.initialstate, fst2.initialstate): newfst.initialstate}
    dead1, dead2 = State(finalweight = float("inf")), State(finalweight = float("inf"))
    while Q:
        t1s, t2s = Q.pop()
        currentstate = S[(t1s, t2s)]
        currentstate.name = (t1s.name, t2s.name,)
        if finalf((t1s in fst1.finalstates, t2s in fst2.finalstates)):
            newfst.finalstates.add(currentstate)
            currentstate.finalweight = oplus(t1s.finalweight, t2s.finalweight)
        # Get all outgoing labels we want to follow
        for lbl in pathfollow(t1s.transitions.keys(), t2s.transitions.keys()):
            for outtr in t1s.transitions.get(lbl, (Transition(dead1, lbl, float('inf')), )):
                for intr in t2s.transitions.get(lbl, (Transition(dead2, lbl, float('inf')), )):
                    if (outtr.targetstate, intr.targetstate) not in S:
                        Q.append((outtr.targetstate, intr.targetstate))
                        S[(outtr.targetstate, intr.targetstate)] = State()
                        newfst.states.add(S[(outtr.targetstate, intr.targetstate)])
                    currentstate.add_transition(S[(outtr.targetstate, intr.targetstate)], lbl, oplus(outtr.weight, intr.weight))
    return newfst
# endregion


# Defines a list of functions that should be added as instance methods to the FST class dynamically
_algorithms_to_add: Dict[str, Callable] = {
    'trim': trimmed,
    'filter_accessible': filtered_accessible,
    'filter_coaccessible': filtered_coaccessible,
    'scc': scc,
    'push_weights': pushed_weights,
    'map_labels': mapped_labels,
    'epsilon_remove': epsilon_removed,
    'epsilon_closure': epsilon_closure,
    'dijkstra': dijkstra,
    'label_states_topology': labelled_states_topology,
    'words_nbest': words_nbest,
    'words_cheapest': words_cheapest,
    'determinize_unweighted': determinized_unweighted,
    'determinize_as_dfa': determinized_as_dfa,
    'determinize': determinized,
    'minimize_as_dfa': minimized_as_dfa,
    'minimize': minimized,
    'merge_equivalent_states': merging_equivalent_states,
    'find_sourcestates': find_sourcestates,
    'create_reverse_index': create_reverse_index,
    'minimize_brz': minimized_brz,
    'kleene_closure': kleene_closure,
    'add_weight': added_weight,
    'optional': optional,
    'concatenate': concatenate,
    'cross_product': cross_product,
    'compose': compose,
    'invert': inverted,
    'ignore': ignore,
    'rewrite': rewritten,
    'context_restrict': context_restricted,
    'project': projected,
    'reverse': reversed,
    'reverse_e': reversed_e,
    'union': union,
    'intersection': intersection,
    'difference': difference,
    'product': product
}