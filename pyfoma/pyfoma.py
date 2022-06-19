#!/usr/bin/env python

"""PyFoma Finite-State Tool."""

import heapq, operator, itertools, re as pyre, functools
from collections import deque, defaultdict

__author__     = "Mans Hulden"
__copyright__  = "Copyright 2022"
__credits__    = ["Mans Hulden"]
__license__    = "Apache"
__version__    = "2.0"
__maintainer__ = "Mans Hulden"
__email__      = "mans.hulden@gmail.com"
__status__     = "Prototype"

# Module-level functions
def concatenate(x, y):
    return x.copy_mod().concatenate(y)

def union(x, y):
    return x.copy_mod().union(y)

def intersection(x, y):
    return x.copy_mod().intersection(y)

def kleene_star(x):
    return x.copy_mod().kleene_closure()

def kleene_plus(x):
    return x.copy_mod().kleene_closure(mode = 'plus')

def difference(x, y):
    return x.copy_mod().difference(y)

def cross_product(x, y):
    return x.copy_mod().cross_product(y)

def compose(x, y):
    return x.copy_mod().compose(y)

def optional(x):
    return x.copy_mod().optional()

def ignore(x, y):
    return x.copy_mod().ignore(y)

def project(x, dim = 0):
    return x.copy_mod().project(dim = dim)

def invert(x):
    return x.copy_mod().invert()

def reverse(x):
    return x.copy_mod().reverse()

def reverse_e(x):
    return x.copy_mod().reverse_e()

def minimize(x):
    return x.copy_mod().minimize()

def minimize_as_dfa(x):
    return x.copy_mod().minimize_as_dfa()

def determinize_as_dfa(x):
    return x.copy_mod().determinize_as_dfa()

def minimize_unweighted(x):
    return x.copy_mod().determinize_unweighted()

def re(*args, **kwargs):
    return FST.re(*args, **kwargs)

regex = re


class regexparse:

    shortops = {'|':'UNION', '-':'MINUS', '&':'INTERSECTION', '*':'STAR', '+':'PLUS',
                '(':'LPAREN', ')':'RPAREN', '?':'OPTIONAL', ':':'CP', ':?': 'CPOPTIONAL',
                '~':"COMPLEMENT", '@':"COMPOSE", ',': 'COMMA', '/':'CONTEXT', '_':'PAIRUP'}
    builtins = {'reverse': lambda x: FST.reverse(x),
                'invert':lambda x: FST.invert(x),
                'minimize': lambda x: FST.minimize(x),
                'determinize': lambda x: FST.determinize(x),
                'ignore': lambda x,y: FST.ignore(x,y),
                'rewrite': lambda *args, **kwargs: FST.rewrite(*args, **kwargs),
                'restrict': lambda *args, **kwargs: FST.context_restrict(*args, **kwargs),
                'project': lambda *args, **kwargs: FST.project(*args, dim = int(kwargs.get('dim', '-1'))),
                'input': lambda x: FST.project(x, dim = 0),
                'output': lambda x: FST.project(x, dim = -1)}
    precedence = {"FUNC": 11, "COMMA":1, "PARAM":1, "COMPOSE":3, "UNION":5, "INTERSECTION":5,
                  "MINUS":5, "CONCAT":6, "STAR":9, "PLUS":9, "OPTIONAL":9, "WEIGHT":9,
                  "CP":10, "CPOPTIONAL":10, "RANGE":9, "CONTEXT":1, "PAIRUP":2}
    operands  = {"SYMBOL", "VARIABLE", "ANY", "EPSILON", "CHAR_CLASS"}
    operators = set(precedence.keys())
    unarypost = {"STAR", "PLUS", "WEIGHT", "OPTIONAL", "RANGE"}
    unarypre  = {"COMPLEMENT"}

    def __init__(self, regularexpression, defined, functions):
        """Tokenize, parse, and compile regex into FST.
        'I define UNIX as 30 definitions of regular expressions living under one roof.'
        - Don Knuth, Digital Typography, ch. 33, p. 649 (1999)"""

        self.defined = defined
        self.functions = {f.__name__:f for f in functions} # Who you gonna call?
        self.expression = regularexpression
        self.tokenized = self._insert_invisibles(self.tokenize())
        self.parsed = self.parse()
        self.compiled = self.compile()

    def character_class_parse(self, charclass):
        """Parse a character class into range pairs, e.g. 'a-zA' => [(97,122), (65,65)].
           'Writing clear and unambiguous specifications for character classes is tough,
           and implementing them perfectly is worse, requiring a lot of tedious and
           uninstructive coding.' -Brian Kernighan (in "Beautiful Code", 2007). """

        negated = False
        if charclass[0] == '^':
            negated = True
            charclass = charclass[1:]

        clncc, escaped = [], set()
        j = 0
        for letter in charclass: # remove escape chars and index those escaped positions
            if letter != '\\':
                clncc.append(letter)
                j += 1
            else:
                escaped.add(j)
        # Mark positions with a range (i.e. mark where the "-"-symbol is)
        marks = [True if (clncc[i] == '-' and i not in escaped and i != 0 and \
                      i != len(clncc)-1) else False for i in range(len(clncc))]
        ranges = [(ord(clncc[i-1]), ord(clncc[i+1])) for i in range(len(clncc)) if marks[i]]

        # 3-convolve over marks to figure out where the non-range characters are
        singles = [any(m) for m in zip(marks, marks[1:] + [False], [False] + marks)]
        for i in range(len(singles)):
            if singles[i] == False:
                ranges.append((ord(clncc[i]), ord(clncc[i]))) # dummy range, e.g (65, 65)

        if any(start > end for start, end in ranges):
            raise SyntaxError("End must be larger than start in character class range.")
        return ranges, negated

    def compile(self) -> 'FST':
        """Put it all together!
        'If you lie to the compiler, it will have its revenge.' — Henry Spencer."""

        def _stackcheck(s):
            if not s:
                self._error_report(SyntaxError, "You stopped making sense!", line_num, column)
            return s
        def _pop(s):
            return _stackcheck(s).pop()[0]
        def _peek(s): # For unaries we just directly mutate the FSM on top of the stack
            return _stackcheck(s)[-1][0]
        def _append(s, element):
            s.append([element])
        def _merge(s):      # since we keep the FSTs inside lists, we need to do some`
            _stackcheck(s)  # reshuffling with a COMMA token so that the top 2 elements
            one = s.pop()   # get merged into one list that ends up on top of the stack.
            _stackcheck(s)  # [[FST1], [FST2], [FST3]] => [[FST1], [FST2, FST3]]
            s.append(s.pop() + one)
        def _pairup(s):     # take top two on stack and merge inside list as 2-tuple
            _stackcheck(s)  # [ [FST1], [FST2], [FST3] ] => [ [FST1], [(FST2, FST3)] ]
            one = s.pop()
            _stackcheck(s)
            s.append([tuple(s.pop() + one)])
        def _getargs(s):
            return _stackcheck(s).pop()
        stack, parameterstack = [], []
        for op, value, line_num, column in self.parsed:
            if op == 'FUNC':
                if value in self.functions:  # Look in user-defined functions first ...
                    _append(stack, self.functions[value](*_getargs(stack), **dict(parameterstack)))
                elif value in self.builtins: # ... then in built-ins
                    _append(stack, self.builtins[value](*_getargs(stack), **dict(parameterstack)))
                else:
                    self._error_report(SyntaxError, "Function \"" + value + \
                                        "\" not defined.", line_num, column)
                parameterstack = []
            elif op == 'LPAREN':
                self._error_report(SyntaxError, "Missing closing parenthesis.", line_num, column)
            elif op == 'COMMA':   # Collect arguments in single list on top of stack
                _merge(stack)
            elif op == 'PARAM':
                parameterstack.append(value)
            elif op == 'PAIRUP':  # Collect argument pairs as 2-tuples on top of stack
                _pairup(stack)
            elif op == 'CONTEXT': # Same as COMMA, possible future expansion
                _merge(stack)
            elif op == 'UNION':
                _append(stack, _pop(stack).union(_pop(stack)))
            elif op == 'MINUS':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.difference(arg2.determinize_unweighted()))
            elif op == 'INTERSECTION':
                _append(stack, _pop(stack).intersection(_pop(stack)).coaccessible())
            elif op == 'CONCAT':
                second = _pop(stack)
                _append(stack, _pop(stack).concatenate(second).accessible())
            elif op == 'STAR':
                _append(stack, _pop(stack).kleene_closure())
            elif op == 'PLUS':
                _append(stack, _pop(stack).kleene_closure(mode = 'plus'))
            elif op == 'COMPOSE':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.compose(arg2).coaccessible())
            elif op == 'OPTIONAL':
                _peek(stack).optional()
            elif op == 'RANGE':
                rng = value.split(',')
                lang = _pop(stack)
                if len(rng) == 1:  # e.g. {3}
                    _append(stack, functools.reduce(lambda x, y: concatenate(x, y), [lang]*int(value)))
                elif rng[0] == '': # e.g. {,3}
                    lang = lang.optional()
                    _append(stack, functools.reduce(lambda x, y: concatenate(x, y), [lang]*int(rng[1])))
                elif rng[1] == '': # e.g. {3,}
                    _append(stack, functools.reduce(lambda x, y: concatenate(x, y), [lang]*int(rng[0])).concatenate(lang.kleene_closure()))
                else:              # e.g. {1,4}
                    if int(rng[0] > rng[1]):
                        self._error_report(SyntaxError, "n must be greater than m in {m,n}", line_num, column)
                    lang1 = functools.reduce(lambda x, y: concatenate(x, y), [lang]*int(rng[0]))
                    lang2 = functools.reduce(lambda x, y: concatenate(x, y), [lang.optional()]*(int(rng[1])-int(rng[0])))
                    _append(stack, lang1.concatenate(lang2))
            elif op == 'CP':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.cross_product(arg2).coaccessible())
            elif op == 'CPOPTIONAL':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.cross_product(arg2, optional = True).coaccessible())
            elif op == 'WEIGHT':
                _peek(stack).add_weight(float(value)).push_weights()
            elif op == 'SYMBOL':
                _append(stack, FST(label = (value,)))
            elif op == 'ANY':
                _append(stack, FST(label = ('.',)))
            elif op == 'VARIABLE':
                if value not in self.defined:
                    self._error_report(SyntaxError, "Defined FST \"" + value + \
                                                    "\" not found.", line_num, column)
                _append(stack, self.defined[value].copy_mod())
            elif op == 'CHAR_CLASS':
                charranges, negated = self.character_class_parse(value)
                _append(stack, FST.character_ranges(charranges, complement = negated))
        if len(stack) != 1: # If there's still stuff on the stack, that's a syntax error
            self._error_report(SyntaxError,\
              "Something's happening here, and what it is ain't exactly clear...", 1, 0)
        return _pop(stack).trim().epsilon_remove().push_weights().determinize_as_dfa().minimize_as_dfa().label_states_topology().cleanup_sigma()

    def tokenize(self) -> list:
        """Token, token, token, though the stream is broken... ride 'em in, tokenize!"""
        # prematch (skip this), groupname, core regex (capture this), postmatch (skip)
        token_regexes = [
        (r"\\"  , 'ESCAPED',    r".",                        r""),          # Esc'd sym
        (r", *" , 'PARAM',      r"\w+ *= *[+-]? *\w+",       r""),          # Parameter
        (r"'"   , 'QUOTED',     r"(\\[']|[^'])*",            r"'"),         # Quoted sym
        (r""    , 'SKIPWS',     r"[ \t]+",                   r""),          # Skip ws
        (r""    , 'SHORTOP',    r"(:\?|[|\-&*+()?:@,/_])",   r""),          # main ops
        (r"\$\^", 'FUNC',       r"\w+",                      r"(?=\s*\()"), # Functions
        (r"\$"  , 'VARIABLE',   r"\w+",                      r""),          # Variables
        (r"<"   , 'WEIGHT',     r"[+-]?[0-9]*(\.[0-9]+)?",   r">"),         # Weight
        (r"\{"  , 'RANGE',      r"\d+,(\d+)?|,?\d+",         r"\}"),        # {(m),(n)}
        (r"\["  , 'CHAR_CLASS', r"\^?(\\]|[^\]])+",          r"\]"),        # Char class
        (r""    , 'NEWLINE',    r"\n",                       r""),          # Line end
        (r""    , 'SYMBOL',     r".",                        r"")           # Single sym
    ]
        tok_regex = '|'.join('%s(?P<%s>%s)%s' % mtch for mtch in token_regexes)
        line_num, line_start, res = 1, 0, []
        for mo in pyre.finditer(tok_regex, self.expression):
            op = mo.lastgroup
            value = mo.group(op)
            column = mo.start() - line_start
            if op == 'SKIPWS':
                continue
            elif op == 'ESCAPED' or op == 'QUOTED':
                op = 'SYMBOL'
                value = value.replace("\\","")
            elif op == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
                continue
            elif op == 'SHORTOP':
                op = self.shortops[value]
            elif op == 'PARAM':
                value = value.replace(" ","").split('=')
            res.append((op, value, line_num, column))
        return res

    def _insert_invisibles(self, tokens: list) -> list:
        """Idiot hack or genius? We insert explicit CONCAT tokens before parsing.

           'I now avoid invisible infix operators almost entirely. I do remember a few
           texts dealing with theorems about strings in which concatenation was denoted
           by juxtaposition.' (EWD 1300-9)"""

        resetters = self.operators - self.unarypost
        counter, result = 0, []
        for token, value, line_num, column in tokens: # It's a two-state FST!
            if counter == 1 and token in {'LPAREN', 'COMPLEMENT'} | self.operands:
                result.append(('CONCAT', '', line_num, column))
                counter = 0
            if token in self.operands:
                counter = 1
            if token in resetters: # No, really, it is!
                counter = 0
            result.append((token, value, line_num, column))
        # Add epsilon to rewrites, restrict with missing contexts, .e.g "(missing) _ #"
        newresult, prevt = [], None
        for token, value, line_num, column in result:
            if ((token == 'COMMA' or token == 'PARAM') and prevt == 'PAIRUP') or \
               (token == 'PAIRUP' and (prevt == 'CONTEXT' or prevt == 'COMMA')) or \
               (token == 'RPAREN' and prevt == 'PAIRUP'):
                newresult.append(('SYMBOL', '', line_num, column))
            newresult.append((token, value, line_num, column))
            prevt = token
        return newresult

    def _error_report(self, errortype, errorstring, line_num, column):
        raise errortype(errorstring, ("", line_num, column, self.expression))

    def parse(self) -> list:
        """Attention! Those who don't speak reverse Polish will be shunted!
        'Simplicity is a great virtue but it requires hard work to achieve it and
        education to appreciate it. And to make matters worse: complexity sells better.'
        - E. Dijkstra """
        output, stack = [], []
        for token, value, line_num, column in self.tokenized:
            if token in self.operands or token in self.unarypost:
                output.append((token, value, line_num, column))
            elif token in self.unarypre or token == "FUNC" or token == "LPAREN":
                stack.append((token, value, line_num, column))
                #if token == "LPAREN":
                #    output.append("STARTP")
            elif token == "RPAREN":
                while True:
                    if not stack:
                        self._error_report(SyntaxError, "Too many closing parentheses.", line_num, column)
                    if stack[-1][0] == 'LPAREN':
                        break
                    output.append(stack.pop())
                #output.append("ENDP")
                stack.pop()
                if stack and stack[-1][0] == "FUNC":
                    output.append(stack.pop())
            elif token in self.operators: # We don't have any binaries that assoc right.
                while stack and stack[-1][0] in self.operators and \
                      self.precedence[stack[-1][0]] >= self.precedence[token]:
                    output.append(stack.pop())
                stack.append((token, value, line_num, column))
        while stack:
            output.append(stack.pop())
        return output


class Paradigm:

    def __init__(self, grammar, regexfilter, tagfilter = lambda x: x.startswith('[') and x.endswith(']')):
        """Extract a 'paradigm' from a grammar FST. Available as a list in attr para.
           regexfilter -- a regex which is composed on the input side to filter out
                          a specific lexeme or set of lexemes, e.g. 'run.*'
           Keyword arguments:
           tagfilter -- a function to identify tags, by default bracketed symbols [ ... ]
           """
        self.FSM = grammar
        self.regexfilter = regexfilter # a regex used for filtering input side
        self.tagfilter = tagfilter # func to identify tags vs. other symbols
        self.tables = {} # indexed by citation form of lexeme
        self.filtered = FST.re(regexfilter + " @ $grammar", {'grammar': grammar})
        self.words = self.filtered.words()
        para = []
        for weight, pairlist in self.words:
            lemma, tags, output = [], [], []
            for io in pairlist:
                if len(io) == 1:
                    i, o = io[0], io[0]
                else:
                    i, o = io[0], io[-1]
                if tagfilter(i):
                    tags.append(i)
                else:
                    lemma.append(i)
                output.append(o)
            para.append([''.join(lemma), ''.join(tags), ''.join(output)])
        self.para = sorted(para)

    def __str__(self):
        """Return a formatted table with lemma, tags, wordform."""
        maxlens = (max(len(w) for w in cols) for cols in zip(*self.para)) # max for each col
        fmtstr = "".join("{:<" + str(ml+2) + "}" for ml in maxlens) + "\n"
        return "".join(fmtstr.format(*cols) for cols in self.para)

class PartitionRefinement:

    """Basic partition refinement using dicts. A pared down version of D. Eppstein's
       implementation. https://www.ics.uci.edu/~eppstein/PADS/PartitionRefinement.py"""

    def __init__(self, S):
        """Create a new partition refinement data structure for the given
        items.  Initially, all items belong to the same subset.
        """
        self.sets = {id(s):s for s in S}
        self.partition = {x:s for s in S for x in s}

    def refine(self, S):
        """Refine each set A in the partition to the two sets
        A & S, A - S.  Return a list of pairs (A & S, A - S)
        for each changed set.  Within each pair, A & S will be
        a newly created set, while A - S will be a modified
        version of an existing set in the partition.
        Not a generator because we need to perform the partition
        even if the caller doesn't iterate through the results.
        """
        hit = {}
        output = []
        for x in S:
            if x in self.partition:
                Ax = self.partition[x]
                hit.setdefault(id(Ax), set()).add(x)
        for A, AS in hit.items():
            A = self.sets[A]
            if AS != A:
                self.sets[id(AS)] = AS
                for x in AS:
                    self.partition[x] = AS
                A -= AS
                output.append((id(AS), id(A)))
        return output

    def astuples(self):
        """Get current partitioning and convert to set of tuples."""
        return {tuple(s) for s in self.sets.values()}


class FST:

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
        myregex = regexparse(regularexpression, defined, functions)
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

    def __init__(self, label = None, weight = 0.0, alphabet = set()):
        """Calling FST() creates an FST-structure with a single state.
           Keyword arguments:
           label -- create a two-state FST that accepts label
           weight -- add a weight to the final state
           alphabet -- declare an alphabet explicitly
           If 'label' is given, a two-state automaton is created with label as the
           only transition from the initial state to the final state.
           If 'weight' is also given, the final state will have that weight.
           Labels are always tuples internally, so a two-state automaton
           that only accepts 'a' should have label = ('a',).
           If label is the empty string, i.e. ('',), the second state will not be
           created, but the initial state will be made final with weight 'weight'."""

        self.alphabet = alphabet
        self.initialstate = State()
        self.states = {self.initialstate}
        self.finalstates = set()
        if label == ('',): # EPSILON
            self.finalstates.add(self.initialstate)
            self.initialstate.finalweight = weight
        elif label is not None:
            self.alphabet = {s for s in label}
            targetstate = State()
            self.states.add(targetstate)
            self.finalstates = {targetstate}
            targetstate.finalweight = weight
            self.initialstate.add_transition(targetstate, label, 0.0)

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

    def harmonize_alphabet(func):
        """A wrapper for expanding .-symbols when operations of arity 2 are performed.
           For example, if calculating the union of FSM1 and FSM2, and both contain
           .-symbols, the transitions with . are expanded to include the symbols that
           are present in the other FST."""
        @functools.wraps(func)
        def wrapper_decorator(self, other, **kwargs):
            for A, B in [(self, other), (other, self)]:
                if '.' in A.alphabet and (A.alphabet - {'.'}) != (B.alphabet - {'.'}):
                    Aexpand = B.alphabet - A.alphabet - {'.', ''}
                    if A == other:
                        A, _ = other.copy_filtered()
                        other = A # Need to copy to avoid mutating other
                    for s, l, t in list(A.all_transitions(A.states)):
                        if '.' in l:
                            for sym in Aexpand:
                                newl = tuple(lbl if lbl != '.' else sym for lbl in l)
                                s.add_transition(t.targetstate, newl, t.weight)

            newalphabet = self.alphabet | other.alphabet
            value = func(self, other, **kwargs)
            # Do something after
            value.alphabet = newalphabet
            return value
        return wrapper_decorator

    def trim(self):
        """Remove states that aren't both accessible and coaccessible."""
        return self.accessible().coaccessible()

    def accessible(self):
        """Remove states that are not on a path from the initial state."""
        explored = {self.initialstate}
        stack = deque([self.initialstate])
        while stack:
            source = stack.pop()
            for label, transition in source.all_transitions():
                if transition.targetstate not in explored:
                    explored.add(transition.targetstate)
                    stack.append(transition.targetstate)

        self.states = explored
        self.finalstates &= self.states
        return self

    def coaccessible(self):
        """Remove states and transitions to states that have no path to a final state."""
        explored = {self.initialstate}
        stack = deque([self.initialstate])
        inverse = {s:set() for s in self.states} # store all preceding arcs here
        while stack:
            source = stack.pop()
            for target in source.all_targets():
                inverse[target].add(source)
                if target not in explored:
                    explored.add(target)
                    stack.append(target)

        stack = deque([s for s in self.finalstates])
        coaccessible = {s for s in self.finalstates}
        while stack:
            source = stack.pop()
            for previous in inverse[source]:
                if previous not in coaccessible:
                    coaccessible.add(previous)
                    stack.append(previous)

        coaccessible.add(self.initialstate) # Let's make an exception for the initial
        for s in self.states: # Need to also remove transitions to non-coaccessibles
            s.remove_transitions_to_targets(self.states - coaccessible)

        self.states &= coaccessible
        self.finalstates &= self.states
        return self

    def cleanup_sigma(self):
        """Remove symbols if they are no longer needed, including . ."""
        seen = {sym for _, lbl, _ in self.all_transitions(self.states) for sym in lbl}
        if '.' not in seen:
            self.alphabet &= seen
        return self

    def view(self, raw = False, show_weights = False, show_alphabet = True):
        """Graphviz viewing and display in Jupyter.
           Keyword arguments:
           raw -- if True, show label tuples and weights unformatted
           show_weights -- force display of weights even if 0.0
           show_alphabet -- displays the alphabet below the FST
        """
        import graphviz
        from IPython.display import display
        def _float_format(num):
            if not show_weights:
                return ""
            s = '{0:.2f}'.format(num).rstrip('0').rstrip('.')
            s = '0' if s == '-0' else s
            return "/" + s
        def _str_fmt(s): # Use greek lunate epsilon symbol U+03F5
            return (sublabel if sublabel != '' else 'ϵ' for sublabel in s)


#        g = graphviz.Digraph('FST', filename='fsm.gv')

        sigma = "Σ: {" + ','.join(sorted(a for a in self.alphabet)) + "}" \
            if show_alphabet else ""
        g = graphviz.Digraph('FST', graph_attr={ "label": sigma, "rankdir": "LR" })
        statenums = self.number_unnamed_states()
        if show_weights == False:
            if any(t.weight != 0.0 for _, _, t in self.all_transitions(self.states)) or \
                  any(s.finalweight != 0.0 for s in self.finalstates):
                  show_weights = True

        g.attr(rankdir='LR', size='8,5')
        g.attr('node', shape='doublecircle', style = 'filled')
        for s in self.finalstates:
            g.node(str(statenums[id(s)]) + _float_format(s.finalweight))
            if s == self.initialstate:
                g.node(str(statenums[id(s)]) + _float_format(s.finalweight), style = 'filled, bold')

        g.attr('node', shape='circle', style = 'filled')
        for s in self.states:
            if s not in self.finalstates:
                g.node(str(statenums[id(s)]), shape='circle', style = 'filled')
                if s == self.initialstate:
                    g.node(str(statenums[id(s)]), shape='circle', style = 'filled, bold')
            grouped_targets = defaultdict(set) # {states}
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
                g.edge(sourcelabel, targetlabel, label = graphviz.nohtml(printlabel))
        display(graphviz.Source(g))

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


    def scc(self) -> set:
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

        for s in self.states:
            if s not in indices:
                _strongconnect(s)

        return sccs

    def push_weights(self):
        """Pushes weights toward the initial state. Calls dijkstra and maybe scc."""
        potentials = {s:self.dijkstra(s) for s in self.states}
        for s, _, t in self.all_transitions(self.states):
            t.weight += potentials[t.targetstate] - potentials[s]
        for s in self.finalstates:
            s.finalweight = s.finalweight - potentials[s]
        residualweight = potentials[self.initialstate]
        if residualweight != 0.0:
            # Add residual to all exits of initial state SCC and finals in that SCC
            mainscc = next(s for s in self.scc() if self.initialstate in s)
            for s, _, t in self.all_transitions(mainscc):
                if t.targetstate not in mainscc: # We're exiting the main SCC
                    t.weight += residualweight
            for s in mainscc & self.finalstates: # Add res w to finals in initial SCC
                s.finalweight += residualweight
        return self

    def map_labels(self, map):
        """Relabel a transducer with new labels from dictionary mapping."""
        # Example: myfst.map_labels({'a':'', 'b':'a'})
        for s in self.states:
            newlabelings = []
            for lbl in s.transitions.keys():
                if any(l in lbl for l in map):
                    newlabel = tuple(map[lbl[i]] if lbl[i] in map else lbl[i] for i in range(len(lbl)))
                    newlabelings.append((lbl, newlabel))
            for old, new in newlabelings:
                s.rename_label(old, new)
        self.alphabet = self.alphabet - map.keys() | set(map.values()) - {''}
        return self

    def copy_mod(self, modlabel = lambda l, w: l, modweight = lambda l, w: w):
        """Copies an FSM and possibly modifies labels and weights through functions.
           Keyword arguments:
           modlabel -- a function that modifies the label, takes label, weight as args.
           modweights -- a function that modifies the weight, takes label, weight as args.
        """
        newfst = FST(alphabet = self.alphabet.copy())
        q1q2 = {k:State(name = k.name) for k in self.states}
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

    def epsilon_remove(self):
        """Create new epsilon-free FSM equivalent to original."""
        # For each state s, figure out the min-cost w' to hop to a state t with epsilons
        # Then, add the (non-e) transitions of state t to s, adding w' to their cost
        # Also, if t is final and s is not, make s final with cost t.finalweight ⊗ w'
        # If s and t are both final, make s's finalweight s.final ⊕ (t.finalweight ⊗ w')

        eclosures = {s:self.epsilon_closure(s) for s in self.states}
        if all(len(ec) == 0 for ec in eclosures.values()): # bail, no epsilon transitions
            return self
        newfst, mapping = self.copy_filtered(labelfilter = lambda lbl: any(len(sublabel) != 0 for sublabel in lbl))
        for state, ec in eclosures.items():
            for target, cost in ec.items():
                # copy target's transitions to source
                for label, t in target.all_transitions():
                    if all(len(sublabel) == 0 for sublabel in label): # is epsilon: skip
                        continue
                    mapping[state].add_transition(mapping[t.targetstate], label, cost + t.weight)
                if target in self.finalstates:
                    if state not in self.finalstates:
                        newfst.finalstates.add(mapping[state])
                        mapping[state].finalweight = 0.0
                    mapping[state].finalweight += cost + target.finalweight
        return self.become(newfst)

    def epsilon_closure(self, state) -> dict:
        """Find, for a state the set of states reachable by epsilon-hopping."""
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

    def dijkstra(self, state) -> float:
        """The cost of the cheapest path from state to a final state. Go Edsger!"""
        explored, cntr = {state}, itertools.count()  # decrease-key is for wusses
        Q = [(0.0, next(cntr), state)] # Middle is dummy cntr to avoid key ties
        while Q:
            w, _ , s = heapq.heappop(Q)
            if s == None:       # First None we pull out is the lowest-cost exit
                return w
            explored.add(s)
            if s in self.finalstates:
                # now we push a None state to signal the exit from a final
                heapq.heappush(Q, (w + s.finalweight, next(cntr), None))
            for trgt, cost in s.all_targets_cheapest().items():
                if trgt not in explored:
                    heapq.heappush(Q, (cost + w, next(cntr), trgt))
        return float("inf")

    def words(self):
        """A generator to yield all words. Yay BFS!"""
        Q = deque([(self.initialstate, 0.0, [])])
        while Q:
            s, cost, seq = Q.popleft()
            if s in self.finalstates:
                yield cost + s.finalweight, seq
            for label, t in s.all_transitions():
                Q.append((t.targetstate, cost + t.weight, seq + [label]))

    def label_states_topology(self, mode = 'BFS'):
        """Topologically sort and label states with numbers.
        Keyword arguments:
        mode -- 'BFS', i.e. breadth-first search by default. 'DFS' is depth-first.
        """
        cntr = itertools.count()
        Q = deque([self.initialstate])
        inqueue = {self.initialstate}
        while Q:
            s = Q.popleft() if mode == 'BFS' else Q.pop()
            s.name = str(next(cntr))
            for label, t in s.all_transitions():
                if t.targetstate not in inqueue:
                    Q.append(t.targetstate)
                    inqueue.add(t.targetstate)
        return self

    def words_nbest(self, n) -> list:
        """Finds the n cheapest word in an FST, returning a list."""
        return list(itertools.islice(self.words_cheapest(), n))

    def words_cheapest(self):
        """A generator to yield all words in order of cost, cheapest first."""
        cntr = itertools.count()
        Q = [(0.0, next(cntr), self.initialstate, [])]
        while Q:
            cost, _, s, seq = heapq.heappop(Q)
            if s is None:
                yield cost, seq
            else:
                if s in self.finalstates:
                    heapq.heappush(Q, (cost + s.finalweight, next(cntr), None, seq))
                for label, t in s.all_transitions():
                    heapq.heappush(Q, (cost + t.weight, next(cntr), t.targetstate, seq + [label]))

    def tokenize_against_alphabet(self, word) -> list:
        """Tokenize a string using the alphabet of the automaton."""
        tokens = []
        start = 0
        while start < len(word):
            t = word[start] # Default is length 1 token unless we find a longer one
            for length in range(1, len(word) - start + 1):    # TODO: limit to max length
                if word[start:start+length] in self.alphabet: # of syms in alphabet
                    t = word[start:start+length]
            tokens.append(t)
            start += len(t)
        return tokens

    def generate(self, word, weights = False):
        """Pass word through FST and return generator that yields all outputs."""
        yield from self.apply(word, inverse = False, weights = weights)

    def analyze(self, word, weights = False):
        """Pass word through FST and return generator that yields all inputs."""
        yield from self.apply(word, inverse = True, weights = weights)

    def apply(self, word, inverse = False, weights = False):
        """Pass word through FST and return generator that yields outputs.
           if inverse == True, map from range to domain.
           weights is by default False. To see the cost, set weights to True."""
        IN, OUT = [-1, 0] if inverse else [0, -1] # Tuple positions for input, output
        cntr = itertools.count()
        w = self.tokenize_against_alphabet(word)
        Q, output = [], []
        heapq.heappush(Q, (0.0, 0, next(cntr), [], self.initialstate)) # (cost, -pos, output, state)
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
                            heapq.heappush(Q, (cost + t.weight, negpos - 1, next(cntr), output + [appendedsym], t.targetstate))

    def determinize_unweighted(self):
        """Determinize with all zero weights."""
        self = self.determinize(staterep = lambda s, w: (s, 0.0), oplus = lambda *x: 0.0)
        return self

    def determinize_as_dfa(self):
        """Determinize as a DFA with weight as part of label, then apply unweighted det."""
        newfst = self.copy_mod(modlabel = lambda l, w: l + (w,), modweight = lambda l, w: 0.0)
        determinized = newfst.determinize_unweighted() # run det, then move weights back
        self = determinized.copy_mod(modlabel = lambda l, _: l[:-1], modweight = lambda l, _: l[-1])
        return self

    def determinize(self, staterep = lambda s, w: (s, w), oplus = min):
        """Weighted determinization of FST."""
        newfst = FST(alphabet = self.alphabet.copy())
        firststate = frozenset({staterep(self.initialstate, 0.0)})
        statesets = {firststate:newfst.initialstate}
        if self.initialstate in self.finalstates:
            newfst.finalstates = {newfst.initialstate}
            newfst.initialstate.finalweight = self.initialstate.finalweight

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
                if any(t.targetstate in self.finalstates for _, t in tset):
                    newfst.finalstates.add(newstate)
                    # State was final, so we discharge the maximum debt we can
                    newstate.finalweight = oplus(t.targetstate.finalweight + t.weight + \
                        residuals[s] - wprime for s, t in tset if t.targetstate in self.finalstates)
        return self.become(newfst)

    def minimize_as_dfa(self):
        """Minimize as a DFA with weight as part of label, then apply unweighted min."""
        newfst = self.copy_mod(modlabel = lambda l, w: l + (w,), modweight = lambda l, w: 0.0)
        minimized = newfst.minimize() # minimize, and shift weights back
        self = minimized.copy_mod(modlabel = lambda l, _: l[:-1], modweight = lambda l, _: l[-1])
        return self

    def minimize(self):
        """Minimize FSM by constrained reverse subset construction, Hopcroft-ish."""
        reverse_index = self.reverse_index()
        finalset, nonfinalset = self.finalstates.copy(), self.states - self.finalstates
        initialpartition = [x for x in (finalset, nonfinalset) if len(x) > 0]
        P = PartitionRefinement(initialpartition)
        Agenda = {id(x) for x in (finalset, nonfinalset) if len(x) > 0}
        while Agenda:
            S = P.sets[Agenda.pop()] # convert id to the actual set it corresponds to
            for label, sourcestates in self.find_sourcestates(reverse_index, S):
                splits = P.refine(sourcestates) # returns list of (A & S, A - S) tuples
                Agenda |= {new for new, _ in splits} # Only place A & S on Agenda
        equivalenceclasses = P.astuples()
        if len(equivalenceclasses) == len(self.states):
            return self # we were already minimal, no need to reconstruct
        minimized = self._mergestatesets(equivalenceclasses)
        return self.become(minimized)

    def _mergestatesets(self, equivalenceclasses: set) -> 'FST':
        """Merge equivalent states given as a set of sets."""
        eqmap = {s[i]:s[0] for s in equivalenceclasses for i in range(len(s))}
        representerstates = set(eqmap.values())
        newfst = FST(alphabet = self.alphabet.copy())
        statemap = {s:State() for s in self.states if s in representerstates}
        newfst.initialstate = statemap[eqmap[self.initialstate]]
        for s, lbl, t in self.all_transitions(self.states):
            if s in representerstates:
                statemap[s].add_transition(statemap[eqmap[t.targetstate]], lbl, t.weight)
        newfst.states = set(statemap.values())
        newfst.finalstates = {statemap[s] for s in self.finalstates if s in representerstates}
        for s in self.finalstates:
            if s in representerstates:
                statemap[s].finalweight = s.finalweight
        return newfst

    def find_sourcestates(self, index, stateset):
        """Create generator that yields sourcestates for a set of target states.
           Yields the label, and the set of sourcestates."""
        all_labels = {l for s in stateset for l in index[s].keys()}
        for l in all_labels:
            sources = set()
            for state in stateset:
                if l in index[state]:
                    sources |= index[state][l]
            yield l, sources

    def reverse_index(self) -> dict:
        """Returns dictionary of transitions in reverse (indexed by state)."""
        idx = {s:{} for s in self.states}
        for s, lbl, t in self.all_transitions(self.states):
            idx[t.targetstate][lbl] = idx[t.targetstate].get(lbl, set()) | {s}
        return idx

    def minimize_brz(self):
        """Minimize through Brzozowski's trick."""
        return self.epsilon_remove().reverse_e().determinize().reverse_e().determinize()

    def kleene_closure(self, mode = 'star'):
        """self*. No epsilons here. If mode == 'plus', calculate self+."""
        q1 = {k:State() for k in self.states}
        newfst = FST(alphabet = self.alphabet.copy())

        for lbl, t in self.initialstate.all_transitions():
            newfst.initialstate.add_transition(q1[t.targetstate], lbl, t.weight)

        for s, lbl, t in self.all_transitions(self.states):
            q1[s].add_transition(q1[t.targetstate], lbl, t.weight)

        for s in self.finalstates:
            for lbl, t in self.initialstate.all_transitions():
                q1[s].add_transition(q1[t.targetstate], lbl, t.weight)
            q1[s].finalweight = s.finalweight

        newfst.finalstates = {q1[s] for s in self.finalstates}
        if mode != 'plus' or self.initialstate in self.finalstates:
            newfst.finalstates |= {newfst.initialstate}
            newfst.initialstate.finalweight = 0.0
        newfst.states = set(q1.values()) | {newfst.initialstate}
        return self.become(newfst)

    def add_weight(self, weight):
        """Adds weight to the set of final states in the FST."""
        for s in self.finalstates:
            s.finalweight += weight
        return self

    def optional(self):
        """Same as T|'' ."""
        if self.initialstate in self.finalstates:
            return self
        newinitial = State()

        for lbl, t in self.initialstate.all_transitions():
            newinitial.add_transition(t.targetstate, lbl, t.weight)

        self.initialstate = newinitial
        self.states.add(newinitial)
        self.finalstates.add(newinitial)
        newinitial.finalweight = 0.0
        return self

    @harmonize_alphabet
    def concatenate(self, other):
        """Concatenation of T1T2. No epsilons. May produce non-accessible states."""
        ocopy, _ = other.copy_filtered() # Need to copy since self may equal other
        q1q2 = {k:State() for k in self.states | ocopy.states}

        for s, lbl, t in self.all_transitions(q1q2.keys()):
            q1q2[s].add_transition(q1q2[t.targetstate], lbl, t.weight)
        for s in self.finalstates:
            for lbl2, t2 in ocopy.initialstate.all_transitions():
                q1q2[s].add_transition(q1q2[t2.targetstate], lbl2, t2.weight + s.finalweight)

        newfst = FST()
        newfst.initialstate = q1q2[self.initialstate]
        newfst.finalstates = {q1q2[f] for f in ocopy.finalstates}
        for s in ocopy.finalstates:
            q1q2[s].finalweight = s.finalweight
        if ocopy.initialstate in ocopy.finalstates:
            newfst.finalstates |= {q1q2[f] for f in self.finalstates}
            for f in self.finalstates:
                q1q2[f].finalweight = f.finalweight + ocopy.initialstate.finalweight
        newfst.states = set(q1q2.values())
        return self.become(newfst)

    @harmonize_alphabet
    def cross_product(self, other, optional = False):
        """Perform the cross-product of T1, T2 through composition.
           Keyword arguments:
           optional -- if True, calculates T1:T2 | T1."""
        newfst_a =  self.copy_mod(modlabel = lambda l, _: l + ('',))
        newfst_b = other.copy_mod(modlabel = lambda l, _: ('',) + l)
        if optional == True:
            self = newfst_a.compose(newfst_b).union(self)
        else:
            self = newfst_a.compose(newfst_b)
        return self

    @harmonize_alphabet
    def compose(self, other):
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
        Q = deque([(self.initialstate, other.initialstate, 0)])
        S = {(self.initialstate, other.initialstate, 0): newfst.initialstate}
        while Q:
            A, B, mode = Q.pop()
            currentstate = S[(A, B, mode)]
            currentstate.name = "({},{},{})".format(A.name, B.name, mode)
            if A in self.finalstates and B in other.finalstates:
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
        return self.become(newfst)

    def invert(self):
        """Calculates the inverse of a transducer, i.e. flips label tuples around."""
        for s in self.states:
            s.transitions  = {lbl[::-1]:tr for lbl, tr in s.transitions.items()}
        return self

    def ignore(self, other):
        """A, ignoring intervening instances of B."""
        newfst = FST.re("$^output($A @ ('.'|'':$B)*)", {'A': self, 'B': other})
        return self.become(newfst)

    def rewrite(self, *contexts, **flags):
        """Rewrite self in contexts in parallel, controlled by flags."""
        defs = {'crossproducts': self}
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
        return self.become(newfst)

    def context_restrict(self, *contexts, rewrite = False):
        """self only allowed in the context L1 _ R1, or ... , or  L_n _ R_n."""
        for fsm in itertools.chain.from_iterable(contexts):
            fsm.alphabet.add('@=@') # Add aux sym to contexts so they don't match .
        self.alphabet.add('@=@')    # Same for self
        if not rewrite:
            cs = (FST.re("$lc '@=@' (.-'@=@')* '@=@' $rc", \
                 {'lc':lc.copy_mod().map_labels({'#': '@#@'}),\
                 'rc':rc.copy_mod().map_labels({'#': '@#@'})}) for lc, rc in contexts)
        else:
            cs = (FST.re("$lc '@=@' (.-'@=@')* '@=@' $rc", {'lc':lc, 'rc':rc}) for lc, rc in contexts)
        cunion = functools.reduce(lambda x, y: x.union(y), cs).determinize().minimize()
        r = FST.re("(.-'@=@')* '@=@' $c '@=@' (.-'@=@')* - ((.-'@=@')* $cunion (.-'@=@')*)",\
                       {'c':self, 'cunion':cunion})
        r = r.map_labels({'@=@':''}).epsilon_remove().determinize_as_dfa().minimize()
        for fsm in itertools.chain.from_iterable(contexts):
            fsm.alphabet -= {'@=@'} # Remove aux syms from contexts
        r = FST.re(".? (.-'@#@')* .? - $r", {'r': r})
        newfst = r.map_labels({'@#@':''}).epsilon_remove().determinize_as_dfa().minimize()
        return self.become(newfst)

    def project(self, dim = 0):
        """Let's project! dim = -1 will get output proj regardless of # of tapes."""
        sl = slice(-1, None) if dim == -1 else slice(dim, dim+1)
        newalphabet = set()
        for s in self.states:
            newtransitions = {}
            for lbl, tr in s.transitions.items():
                newtransitions[lbl[sl]] = newtransitions.get(lbl[sl], set()) | tr
                for t in tr:
                    t.label = lbl[sl]
                    newalphabet |= {sublabel for sublabel in lbl[sl]}
            s.transitions = newtransitions
        self.alphabet = newalphabet
        return self

    def reverse(self):
        """Reversal of FST, epsilon-free."""
        newfst = FST(alphabet = self.alphabet.copy())
        newfst.initialstate = State()
        mapping = {k:State() for k in self.states}
        newfst.states = set(mapping.values()) | {newfst.initialstate}
        newfst.finalstates = {mapping[self.initialstate]}
        if self.initialstate in self.finalstates:
            newfst.finalstates.add(newfst.initialstate)
            newfst.initialstate.finalweight = self.initialstate.finalweight
        mapping[self.initialstate].finalweight = 0.0

        for s, lbl, t in self.all_transitions(self.states):
            mapping[t.targetstate].add_transition(mapping[s], lbl, t.weight)
            if t.targetstate in self.finalstates:
                newfst.initialstate.add_transition(mapping[s], lbl, t.weight + \
                                                   t.targetstate.finalweight)
        return self.become(newfst)

    def reverse_e(self):
        """Reversal of FST, using epsilons."""
        newfst = FST(alphabet = self.alphabet.copy())
        newfst.initialstate = State(name = tuple(k.name for k in self.finalstates))
        mapping = {k:State(name = k.name) for k in self.states}
        for t in self.finalstates:
            newfst.initialstate.add_transition(mapping[t], ('',), t.finalweight)

        for s, lbl, t in self.all_transitions(self.states):
            mapping[t.targetstate].add_transition(mapping[s], lbl, t.weight)

        newfst.states = set(mapping.values()) | {newfst.initialstate}
        newfst.finalstates = {mapping[self.initialstate]}
        mapping[self.initialstate].finalweight = 0.0
        return self.become(newfst)

    @harmonize_alphabet
    def union(self, other):
        """Epsilon-free calculation of union of self and other."""
        mapping = {k:State() for k in self.states|other.states}
        newfst = FST() # Get new initial state
        newfst.states = set(mapping.values()) | {newfst.initialstate}
        # Copy all transitions from old initial states to new initial state
        for lbl, t in itertools.chain(self.initialstate.all_transitions(), other.initialstate.all_transitions()):
            newfst.initialstate.add_transition(mapping[t.targetstate], lbl, t.weight)
        # Also add all transitions from old FSMs to new FSM
        for s, lbl, t in itertools.chain(self.all_transitions(self.states), other.all_transitions(other.states)):
            mapping[s].add_transition(mapping[t.targetstate], lbl, t.weight)
        # Make old final states final in new FSM
        for s in self.finalstates | other.finalstates:
            newfst.finalstates.add(mapping[s])
            mapping[s].finalweight = s.finalweight
        # If either initial state was final, make new initial final w/ weight min(f1w, f2w)
        newfst.finalstates = {mapping[s] for s in self.finalstates|other.finalstates}
        if self.initialstate in self.finalstates or other.initialstate in other.finalstates:
            newfst.finalstates.add(newfst.initialstate)
            newfst.initialstate.finalweight = min(self.initialstate.finalweight, other.initialstate.finalweight)

        return self.become(newfst)

    def intersection(self, other):
        """Intersection of self and other. Uses the product algorithm."""
        self = self.product(other, finalf = all, oplus = operator.add, pathfollow = lambda x,y: x & y)
        return self

    def difference(self, other):
        """Returns self-other. Uses the product algorithm."""
        self = self.product(other, finalf = lambda x: x[0] and not x[1],\
                           oplus = lambda x,y: x, pathfollow = lambda x,y: x)
        return self

    @harmonize_alphabet
    def product(self, other, finalf = any, oplus = min, pathfollow = lambda x,y: x|y):
        """Generates the product FST from self, other. The helper functions by default
           produce self|other."""
        newfst = FST()
        Q = deque([(self.initialstate, other.initialstate)])
        S = {(self.initialstate, other.initialstate): newfst.initialstate}
        dead1, dead2 = State(finalweight = float("inf")), State(finalweight = float("inf"))
        while Q:
            t1s, t2s = Q.pop()
            currentstate = S[(t1s, t2s)]
            currentstate.name = (t1s.name, t2s.name,)
            if finalf((t1s in self.finalstates, t2s in other.finalstates)):
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
        return self.become(newfst)


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
