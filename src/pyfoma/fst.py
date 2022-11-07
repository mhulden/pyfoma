#!/usr/bin/env python

"""PyFoma Finite-State Tool."""

import heapq, operator, itertools, re as pyre, functools
from collections import deque, defaultdict
from typing import Callable, Dict, Any

def re(*args, **kwargs):
    return FST.re(*args, **kwargs)


# TODO: Move all algorithm functions to the algorithms module
class FST:

    # region Initialization Methods
    def __init__(self, label:tuple=None, weight=0.0, alphabet=set()):
        """Creates an FST-structure with a single state.

        :param label: create a two-state FST that accepts label
        :param weight: add a weight to the final state
        :param alphabet: declare an alphabet explicitly

        If 'label' is given, a two-state automaton is created with label as the
        only transition from the initial state to the final state.

        If 'weight' is also given, the final state will have that weight.
        Labels are always tuples internally, so a two-state automaton
        that only accepts 'a' should have label = ('a',).

        If label is the empty string, i.e. ('',), the second state will not be
        created, but the initial state will be made final with weight 'weight'.
        """

        self.alphabet = alphabet
        """The alphabet used by the FST"""
        self.initialstate = State()
        """The initial (start) state of the FST"""
        self.states = {self.initialstate}
        """A set of all states in the FST"""
        self.finalstates = set()
        """A set of all final (accepting) states of the FST"""

        if label == ('',):  # EPSILON
            self.finalstates.add(self.initialstate)
            self.initialstate.finalweight = weight
        elif label is not None:
            self.alphabet = {s for s in label}
            targetstate = State()
            self.states.add(targetstate)
            self.finalstates = {targetstate}
            targetstate.finalweight = weight
            self.initialstate.add_transition(targetstate, label, 0.0)

    @classmethod
    def character_ranges(cls, ranges, complement = False) -> 'FST':
        """Returns a two-state FSM from a list of unicode code point range pairs.
           Keyword arguments:
           complement -- if True, the character class is negated, i.e. [^ ... ], and
           a two-state FST is returned with the single label . and all the symbols in
           the character class are put in the alphabet.
           """
        newfst = cls()
        secondstate = State()
        newfst.states.add(secondstate)
        newfst.finalstates = {secondstate}
        secondstate.finalweight = 0.0
        alphabet = set()
        for start, end in ranges:
            for symbol in range(start, end + 1):
                if symbol not in alphabet:
                    alphabet |= {chr(symbol)}
                    if not complement:
                        newfst.initialstate.add_transition(secondstate, (chr(symbol),), 0.0)
        if complement:
            newfst.initialstate.add_transition(secondstate, ('.',), 0.0)
            alphabet.add('.')
        newfst.alphabet = alphabet
        return newfst

    @classmethod
    def regex(cls, regularexpression, defined = {}, functions = set()):
        """Compile a regular expression and return the resulting FST.
           Keyword arguments:
           defined -- a dictionary of defined FSTs that the compiler can access whenever
                      a defined network is referenced in the regex, e.g. $vowel
           functions -- a set of Python functions that the compiler can access when a function
                       is referenced in the regex, e.g. $^myfunc(...)
        """
        import pyfoma.private.regexparse as regexparse

        myregex = regexparse.RegexParse(regularexpression, defined, functions)
        return myregex.compiled

    re = regex


    @classmethod
    def from_strings(cls, strings):
        """Create an automaton that accepts words in the iterable 'strings'."""
        Grammar = {"Start":((w, "#") for w in strings)}
        return FST.rlg(Grammar, "Start").determinize_as_dfa().minimize().label_states_topology()

    @classmethod
    def rlg(cls, grammar, startsymbol):
        """Compile a (weighted) right-linear grammar into an FST, similarly to lexc."""
        def _rlg_tokenize(w):
            if w == '':
                return ['']
            tokens = []
            tok_re = r"'(?P<multi>([']|[^']*))'|\\(?P<esc>(.))|(?P<single>(.))"
            for mo in pyre.finditer(tok_re, w):
                token = mo.group(mo.lastgroup)
                if token == " " and mo.lastgroup == 'single':
                    token = ""  # normal spaces for alignment, escaped for actual
                tokens.append(token)
            return tokens

        newfst = FST(alphabet = set())
        statedict = {name:State(name = name) for name in grammar.keys() | {"#"}}
        newfst.initialstate = statedict[startsymbol]
        newfst.finalstates = {statedict["#"]}
        statedict["#"].finalweight = 0.0
        newfst.states = set(statedict.values())

        for lexstate in statedict.keys() - {"#"}:
            for rule in grammar[lexstate]:
                currstate = statedict[lexstate]
                lhs = (rule[0],) if isinstance(rule[0], str) else rule[0]
                target = rule[1]
                i = _rlg_tokenize(lhs[0])
                o = i if len(lhs) == 1 else _rlg_tokenize(lhs[1])
                newfst.alphabet |= {sym for sym in i + o if sym != ''}
                for ii, oo, idx in itertools.zip_longest(i, o, range(max(len(i), len(o))),
                    fillvalue = ''):
                    w = 0.0
                    if idx == max(len(i), len(o)) - 1:  # dump weight on last transition
                        targetstate = statedict[target] # before reaching another lexstate
                        w = 0.0 if len(rule) < 3 else float(rule[2])
                    else:
                        targetstate = State()
                        newfst.states.add(targetstate)
                    newtuple = (ii,) if ii == oo else (ii, oo)
                    currstate.add_transition(targetstate, newtuple, w)
                    currstate = targetstate
        return newfst

    # endregion

    # region Utility Methods

    def __copy__(self):
        """Copy an FST. Actually calls copy_filtered()."""
        return self.copy_filtered()[0]

    def __len__(self):
        """Return the number of states."""
        return len(self.states)

    def __str__(self):
        """Generate an AT&T string representing the FST."""
        # Number states arbitrarily based on id()
        ids = [id(s) for s in self.states if s != self.initialstate]
        statenums = {ids[i]:i+1 for i in range(len(ids))}
        statenums[id(self.initialstate)] = 0 # The initial state is always 0
        st = ""
        for s in self.states:
            if len(s.transitions) > 0:
                for label in s.transitions.keys():
                    for transition in s.transitions[label]:
                        st += '{}\t{}\t{}\t{}\n'.format(statenums[id(s)],\
                        statenums[id(transition.targetstate)], '\t'.join(label),\
                        transition.weight)
        for s in self.states:
            if s in self.finalstates:
                st += '{}\t{}\n'.format(statenums[id(s)], s.finalweight)
        return st

    def __and__(self, other):
        """Intersection."""
        return self.intersection(other)

    def __or__(self, other):
        """Union."""
        return self.union(other)

    def __sub__(self, other):
        """Set subtraction."""
        return self.difference(other)

    def __pow__(self, other):
        """Cross-product."""
        return self.cross_product(other)

    def __mul__(self, other):
        """Concatenation."""
        return self.concatenate(other)

    def __matmul__(self, other):
        """Composition."""
        return self.compose(other)

    def become(self, other):
        """Hacky or what? We use this to mutate self for those algorithms that don't directly do it."""
        self.alphabet, self.initialstate, self.states, self.finalstates = \
        other.alphabet, other.initialstate, other.states, other.finalstates
        return self

    def number_unnamed_states(self, force = False) -> dict:
        """Sequentially number those states that don't have the 'name' attribute.
           If 'force == True', number all states."""
        cntr = itertools.count()
        ordered = [self.initialstate] + list(self.states - {self.initialstate})
        return {id(s):(next(cntr) if s.name == None or force == True else s.name) for s in ordered}

    def cleanup_sigma(self):
        """Remove symbols if they are no longer needed, including . ."""
        seen = {sym for _, lbl, _ in self.all_transitions(self.states) for sym in lbl}
        if '.' not in seen:
            self.alphabet &= seen
        return self

    def view(self, raw=False, show_weights=False, show_alphabet=True) -> 'graphviz.Digraph':
        """Creates a 'graphviz.Digraph' object to view the FST. Will automatically display the FST in Jupyter.

            :param raw: if True, show label tuples and weights unformatted
            :param show_weights: force display of weights even if 0.0
            :param show_alphabet: displays the alphabet below the FST
            :return: A Digraph object which will automatically display in Jupyter.

           If you would like to display the FST from a non-Jupyter environment, please use :code:`FST.render`
        """
        import graphviz
        def _float_format(num):
            if not show_weights:
                return ""
            s = '{0:.2f}'.format(num).rstrip('0').rstrip('.')
            s = '0' if s == '-0' else s
            return "/" + s

        def _str_fmt(s):  # Use greek lunate epsilon symbol U+03F5
            return (sublabel if sublabel != '' else 'ϵ' for sublabel in s)

        #        g = graphviz.Digraph('FST', filename='fsm.gv')

        sigma = "Σ: {" + ','.join(sorted(a for a in self.alphabet)) + "}" \
            if show_alphabet else ""
        g = graphviz.Digraph('FST', graph_attr={"label": sigma, "rankdir": "LR"})
        statenums = self.number_unnamed_states()
        if show_weights == False:
            if any(t.weight != 0.0 for _, _, t in self.all_transitions(self.states)) or \
                    any(s.finalweight != 0.0 for s in self.finalstates):
                show_weights = True

        g.attr(rankdir='LR', size='8,5')
        g.attr('node', shape='doublecircle', style='filled')
        for s in self.finalstates:
            g.node(str(statenums[id(s)]) + _float_format(s.finalweight))
            if s == self.initialstate:
                g.node(str(statenums[id(s)]) + _float_format(s.finalweight), style='filled, bold')

        g.attr('node', shape='circle', style='filled')
        for s in self.states:
            if s not in self.finalstates:
                g.node(str(statenums[id(s)]), shape='circle', style='filled')
                if s == self.initialstate:
                    g.node(str(statenums[id(s)]), shape='circle', style='filled, bold')
            grouped_targets = defaultdict(set)  # {states}
            for label, t in s.all_transitions():
                grouped_targets[t.targetstate] |= {(t.targetstate, label, t.weight)}
            for target, tlabelset in grouped_targets.items():
                if raw == True:
                    labellist = sorted((str(l) + '/' + str(w) for t, l, w in tlabelset))
                else:
                    labellist = sorted((':'.join(_str_fmt(label)) + _float_format(w) for _, label, w in tlabelset))
                printlabel = ', '.join(labellist)
                if s in self.finalstates:
                    sourcelabel = str(statenums[id(s)]) + _float_format(s.finalweight)
                else:
                    sourcelabel = str(statenums[id(s)])
                if target in self.finalstates:
                    targetlabel = str(statenums[id(target)]) + _float_format(target.finalweight)
                else:
                    targetlabel = str(statenums[id(target)])
                g.edge(sourcelabel, targetlabel, label=graphviz.nohtml(printlabel))
        return g

    def render(self, view=True, filename: str='FST', format='pdf'):
        """
        Renders the FST to a file and optionally opens the file.
        :param view: If True, the rendered file will be opened.
        :param format: The file format for the Digraph. Typically 'pdf', 'png', or 'svg'. View all formats: https://graphviz.org/docs/outputs/
        """
        digraph = self.view()
        digraph.format = format
        digraph.render(view=view, filename=filename, cleanup=True)

    def all_transitions(self, states):
        """Enumerate all transitions (state, label, Transition) for an iterable of states."""
        for state in states:
            for label, transitions in state.transitions.items():
                for t in transitions:
                    yield state, label, t

    def all_transitions_by_label(self, states):
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

    def copy_mod(self, modlabel=lambda l, w: l, modweight=lambda l, w: w):
        """Copies an FSM and possibly modifies labels and weights through functions.
           Keyword arguments:
           modlabel -- a function that modifies the label, takes label, weight as args.
           modweights -- a function that modifies the weight, takes label, weight as args.
        """
        newfst = FST(alphabet=self.alphabet.copy())
        q1q2 = {k: State(name=k.name) for k in self.states}
        newfst.states = set(q1q2.values())
        newfst.finalstates = {q1q2[s] for s in self.finalstates}
        newfst.initialstate = q1q2[self.initialstate]

        for s, lbl, t in self.all_transitions(q1q2.keys()):
            q1q2[s].add_transition(q1q2[t.targetstate], modlabel(lbl, t.weight), modweight(lbl, t.weight))

        for s in self.finalstates:
            q1q2[s].finalweight = s.finalweight

        return newfst

    def copy_filtered(self, labelfilter = lambda x: True):
        """Create a copy of self, possibly filtering out labels where them
           optional function 'labelfilter' returns False."""
        newfst = FST(alphabet = self.alphabet.copy())
        q1q2 = {k:State() for k in self.states}
        for s in self.states:
            q1q2[s].name = s.name
        newfst.states = set(q1q2.values())
        newfst.finalstates = {q1q2[s] for s in self.finalstates}
        newfst.initialstate = q1q2[self.initialstate]

        for s, lbl, t in self.all_transitions(q1q2.keys()):
            if labelfilter(lbl):
                q1q2[s].add_transition(q1q2[t.targetstate], lbl, t.weight)

        for s in self.finalstates:
            q1q2[s].finalweight = s.finalweight

        return newfst, q1q2

    def generate(self: 'FST', word, weights=False):
        """Pass word through FST and return generator that yields all outputs."""
        yield from self.apply(word, inverse=False, weights=weights)

    def analyze(self: 'FST', word, weights=False):
        """Pass word through FST and return generator that yields all inputs."""
        yield from self.apply(word, inverse=True, weights=weights)

    def apply(self: 'FST', word, inverse=False, weights=False):
        """Pass word through FST and return generator that yields outputs.
           if inverse == True, map from range to domain.
           weights is by default False. To see the cost, set weights to True."""
        IN, OUT = [-1, 0] if inverse else [0, -1]  # Tuple positions for input, output
        cntr = itertools.count()
        w = self.tokenize_against_alphabet(word)
        Q, output = [], []
        heapq.heappush(Q, (0.0, 0, next(cntr), [], self.initialstate))  # (cost, -pos, output, state)
        while Q:
            cost, negpos, _, output, state = heapq.heappop(Q)
            if state == None and -negpos == len(w):
                if weights == False:
                    yield ''.join(output)
                else:
                    yield (''.join(output), cost)
            elif state != None:
                if state in self.finalstates:
                    heapq.heappush(Q, (cost + state.finalweight, negpos, next(cntr), output, None))
                for lbl, t in state.all_transitions():
                    if lbl[IN] == '':
                        heapq.heappush(Q, (cost + t.weight, negpos, next(cntr), output + [lbl[OUT]], t.targetstate))
                    elif -negpos < len(w):
                        nextsym = w[-negpos] if w[-negpos] in self.alphabet else '.'
                        appendedsym = w[-negpos] if nextsym == '.' else lbl[OUT]
                        if nextsym == lbl[IN]:
                            heapq.heappush(Q, (
                            cost + t.weight, negpos - 1, next(cntr), output + [appendedsym], t.targetstate))

    def words(self: 'FST'):
        """A generator to yield all words. Yay BFS!"""
        Q = deque([(self.initialstate, 0.0, [])])
        while Q:
            s, cost, seq = Q.popleft()
            if s in self.finalstates:
                yield cost + s.finalweight, seq
            for label, t in s.all_transitions():
                Q.append((t.targetstate, cost + t.weight, seq + [label]))

    def tokenize_against_alphabet(self: 'FST', word) -> list:
        """Tokenize a string using the alphabet of the automaton."""
        tokens = []
        start = 0
        while start < len(word):
            t = word[start]  # Default is length 1 token unless we find a longer one
            for length in range(1, len(word) - start + 1):  # TODO: limit to max length
                if word[start:start + length] in self.alphabet:  # of syms in alphabet
                    t = word[start:start + length]
            tokens.append(t)
            start += len(t)
        return tokens
    # endregion


class Transition:
    __slots__ = ['targetstate', 'label', 'weight']
    def __init__(self, targetstate, label, weight):
        self.targetstate = targetstate
        self.label = label
        self.weight = weight


class State:
    def __init__(self, finalweight = None, name = None):
        __slots__ = ['transitions', '_transitionsin', '_transitionsout', 'finalweight', 'name']
        # Index both the first and last elements lazily (e.g. compose needs it)
        self.transitions = dict()     # (l_1,...,l_n):{transition1, transition2, ...}
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
        if self._transitionsin is None:
            self._transitionsin = defaultdict(set)
            for label, newtrans in self.transitions.items():
                for t in newtrans:
                    self._transitionsin[label[0]] |= {(label, t)}
        return self._transitionsin

    @property
    def transitionsout(self):
        """Returns a dictionary of the transitions from a state, indexed by the output
           label, i.e. the last member of the label tuple."""
        if self._transitionsout is None:
            self._transitionsout = defaultdict(set)
            for label, newtrans in self.transitions.items():
                for t in newtrans:
                    self._transitionsout[label[-1]] |= {(label, t)}
        return self._transitionsout

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

    def add_transition(self, other, label, weight):
        """Add transition from self to other with label and weight."""
        newtrans = Transition(other, label, weight)
        self.transitions[label] = self.transitions.get(label, set()) | {newtrans}

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
