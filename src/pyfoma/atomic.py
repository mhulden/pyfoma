from collections import defaultdict
from typing import Dict, Iterable, Optional, Set
import itertools
import heapq
from typing_extensions import Tuple

class Transition:
    __slots__ = ['targetstate', 'label', 'weight']
    def __init__(self, targetstate: "State", label, weight):
        self.targetstate = targetstate
        self.label = label
        self.weight = weight


class State:
    __slots__ = 'transitions', '_transitionsin', '_transitionsout', 'finalweight', 'name'

    def __init__(self, finalweight: Optional[float] = None, name: Optional[str] = None):
        # Index both the first and last elements lazily (e.g. compose needs it)
        self.transitions: Dict[Tuple, Set[Transition]] = dict()     # (l_1,...,l_n):{transition1, transition2, ...}
        self._transitionsin = None    # l_1:(label, transition1), (label, transition2), ... }
        self._transitionsout = None   # l_n:(label, transition1), (label, transition2, ...)}
        if finalweight is None:
            finalweight = float("inf")
        self.finalweight = finalweight
        self.name = name

    @property
    def transitionsin(self) -> dict:
        """Returns a dictionary of the transitions from a state, indexed by the input
           label, i.e. the first member of the label tuple."""
        _transitionsin = defaultdict(set)
        for label, newtrans in self.transitions.items():
            for t in newtrans:
                _transitionsin[label[0]] |= {(label, t)}
        return _transitionsin

    @property
    def transitionsout(self):
        """Returns a dictionary of the transitions from a state, indexed by the output
           label, i.e. the last member of the label tuple."""
        _transitionsout = defaultdict(set)
        for label, newtrans in self.transitions.items():
            for t in newtrans:
                _transitionsout[label[-1]] |= {(label, t)}
        return _transitionsout

    def rename_label(self, original, new):
        """Changes labels in a state's transitions from original to new."""
        for t in self.transitions[original]:
            t.label = new
        self.transitions[new] = self.transitions.get(new, set()) | self.transitions[original]
        self.transitions.pop(original)

    def remove_transitions_to_targets(self, targets):
        """Remove all transitions from self to any state in the set targets."""
        newt = {}
        for label, transitions in self.transitions.items():
            newt[label] = {t for t in transitions if t.targetstate not in targets}
            if len(newt[label]) == 0:
                newt.pop(label)
        self.transitions = newt

    def add_transition(self, other: 'State', label, weight=0.0):
        """Add transition from self to other with label and weight."""
        newtrans = Transition(other, label, weight)
        self.transitions[label] = self.transitions.get(label, set()) | {newtrans}
        return newtrans

    def all_transitions(self):
        """Generator for all transitions out from a given state."""
        for label, transitions in self.transitions.items():
            for t in transitions:
                yield label, t

    def all_targets(self) -> set:
        """Returns the set of states a state has transitions to."""
        return {t.targetstate for tr in self.transitions.values() for t in tr}

    def all_epsilon_targets_cheapest(self) -> dict:
        """Returns a dict of states a state transitions to (cheapest) with epsilon."""
        targets = defaultdict(lambda: float("inf"))
        for lbl, tr in self.transitions.items():
            if all(len(sublabel) == 0 for sublabel in lbl): # funky epsilon-check
                for s in tr:
                    targets[s.targetstate] = min(targets[s.targetstate], s.weight)
        return targets

    def all_targets_cheapest(self) -> dict:
        """Returns a dict of states a state transitions to (cheapest)."""
        targets = defaultdict(lambda: float("inf"))
        for tr in self.transitions.values():
            for s in tr:
                targets[s.targetstate] = min(targets[s.targetstate], s.weight)
        return targets

def all_transitions(states: Iterable[State]):
    """Enumerate all transitions (state, label, Transition) for an iterable of states."""
    for state in states:
        for label, transitions in state.transitions.items():
            for t in transitions:
                yield state, label, t

def all_transitions_by_label(states: Iterable[State]):
    """Enumerate all transitions by label. Each yield produces a label, and those
        the target states. 'states' is an iterable of source states."""
    all_labels = {l for s in states for l in s.transitions.keys()}
    for l in all_labels:
        targets = set()
        for state in states:
            if l in state.transitions:
                for transition in state.transitions[l]:
                    targets.add(transition.targetstate)
        yield l, targets

def find_sourcestates(index, stateset):
   """Create generator that yields sourcestates for a set of target states.
      Yields the label, and the set of sourcestates."""
   all_labels = {l for s in stateset for l in index[s].keys()}
   for l in all_labels:
       sources = set()
       for state in stateset:
           if l in index[state]:
               sources |= index[state][l]
       yield l, sources

def create_reverse_index(states: Iterable[State]) -> dict:
    """Returns dictionary of transitions in reverse (indexed by state)."""
    idx = {s:{} for s in states}
    for s, lbl, t in all_transitions(states):
        idx[t.targetstate][lbl] = idx[t.targetstate].get(lbl, set()) | {s}
    return idx

def epsilon_closure(state: State) -> dict:
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
