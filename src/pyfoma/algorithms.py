#!/usr/bin/env python

"""Defines common algorithms over FSTs"""
import heapq
import itertools
from collections import deque
from typing import TYPE_CHECKING, List, Literal, Tuple, Union

from pyfoma._private.exceptions import NoFinalStatesException

if TYPE_CHECKING:
    from .fst import FST


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



def dijkstra(fst: 'FST', state) -> float:
    """The cost of the cheapest path from state to a final state. Go Edsger!"""
    if len(fst.finalstates) == 0:
        raise NoFinalStatesException()

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

def best_word(fst: 'FST') -> List:
    explored, cntr = {fst.initialstate}, itertools.count()
    Q = [(0.0, next(cntr), fst.initialstate, [])]
    while Q:
        w, _ , s, seq = heapq.heappop(Q)
        if s is None:
            return seq
        explored.add(s)
        if s in fst.finalstates:
            heapq.heappush(Q, (w + s.finalweight, next(cntr), None, seq))

        for label, transitions in s.transitions.items():
            cheapest_transition = min(transitions, key=lambda t: t.weight)
            if (target_state := cheapest_transition.targetstate) not in explored:
                heapq.heappush(Q, (cheapest_transition.weight + w, next(cntr), target_state, seq + [label]))
    return []

def ostia(samples: List[Tuple[Union[str, List[str]], Union[str, List[str]]]], merging_order: Literal["lex", "dd"], use_cython=True) -> 'FST':
    """Runs the [OSTIA](https://www.jeffreyheinz.net/classes/24F/655/materials/Oncina-et-al-1993-OSTIA.pdf) algorithm to infer an FST from a dataset.

    Args:
        samples: A list of paired input/output strings, where each string is a `list[str]` or `str`.
    """
    if use_cython:
        from ._private.ostia import ostia as _ostia, Mode
        if merging_order == 'lex':
            mode = Mode.lexicographic
        elif merging_order == 'dd':
            mode = Mode.data_driven
        else:
            raise ValueError()
        return _ostia(samples, mode)
    else:
        from ._private.ostia_py import ostia as _ostia
        return _ostia(samples, merging_order)
