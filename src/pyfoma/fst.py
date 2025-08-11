import heapq, json, itertools, operator, re as pyre
from collections import deque, defaultdict
from typing import Dict, Any, List, TextIO, cast, Optional, Tuple, Sequence
from os import PathLike
from pathlib import Path
import pickle
import functools
import gzip

from pyfoma.flag import FlagStringFilter, FlagOp
from pyfoma._private.states import State, all_transitions
from pyfoma._private import util, algorithms, states, partition_refinement, transition


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
                for s, l, t in list(all_transitions(A.states)):
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


class FST:
    # ==================
    # Initializers
    # ==================

    def __init__(self, label:Optional[Tuple]=None, weight=0.0, alphabet=set()):
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
    def regex(cls, regularexpression, defined = {}, functions = set(), multichar_symbols=None):
        """Compile a regular expression and return the resulting FST.
           Keyword arguments:
           defined -- a dictionary of defined FSTs that the compiler can access whenever
                      a defined network is referenced in the regex, e.g. $vowel
           functions -- a set of Python functions that the compiler can access when a function
                       is referenced in the regex, e.g. $^myfunc(...)
        """
        import pyfoma._private.regexparse as regexparse
        if multichar_symbols is not None:
            escaper = regexparse._multichar_matcher(multichar_symbols)
            regularexpression = escaper.sub(regexparse._multichar_replacer, regularexpression)
        myregex = regexparse.RegexParse(regularexpression, defined, functions)
        return myregex.compiled

    re = regex

    @classmethod
    def from_strings(cls, strings, multichar_symbols=None):
        """Create an automaton that accepts words in the iterable 'strings'."""
        Grammar = {"Start": ((w, "#") for w in strings)}
        lex = FST.rlg(Grammar, "Start", multichar_symbols=multichar_symbols)
        return lex.determinize_as_dfa().minimize().label_states_topology()

    @classmethod
    def rlg(cls, grammar, startsymbol, multichar_symbols=None):
        """Compile a (weighted) right-linear grammar into an FST, similarly to lexc."""
        import pyfoma._private.regexparse as regexparse
        escaper = None
        if multichar_symbols is not None:
            escaper = regexparse._multichar_matcher(multichar_symbols)
        def _rlg_tokenize(w):
            if w == '':
                return ['']
            if escaper is not None:
                w = escaper.sub(regexparse._multichar_replacer, w)
            tokens = []
            tok_re = r"'(?P<multi>'|(?:\\'|[^'])*)'|\\(?P<esc>(.))|(?P<single>(.))"
            for mo in pyre.finditer(tok_re, w):
                token = mo.group(mo.lastgroup) # type:ignore
                if token == " " and mo.lastgroup == 'single':
                    token = ""  # normal spaces for alignment, escaped for actual
                elif mo.lastgroup == "multi":
                    token = token.replace(r"\'", "'")
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
    
    @classmethod
    def from_fomastring(cls, fomastr: str) -> 'FST':
        """
        =====================
        Support for foma i/o
        =====================
        
        The foma file format is a simple string representation of an FST. This representation can be compressed 
        with gzip if stored in a file. A foma-file may contain multiple FSTs, each with its own name (a string).
        
        With some very minor limitations, the foma file format can be used to save/load FSTs in pyfoma as well.
        
        We provide four methods for interchange with foma; two for converting an FST object
        to a string and vice versa, and two for loading/saving foma-files:
        
        to_fomastring(fst: 'FST', fstname:str) -> str
        from_fomastring(fomastr: str) -> 'FST'
        
        load_foma(str: path) -> Dict[str, 'FST']
        save_foma(fsts: Dict)
        
        The file format supports weights in FSTs (which will be ignored if loaded in foma). The main limitation 
        of the file format is that FSTs can be maximally two-tape FSTs, i.e. label tuples cannot be longer than 2.
        
        Foma relies on some information being available in the string representation
        (such as the number of transitions, paths, and final states) so we need to 
        calculate those before putting together a string representation of an FST.
        
        The string format is as follows:
        
        ##foma-net VERSION##
        ##props## 
        PROPERTIES LINE 
        ##weights## 
        ... WEIGHTS LINES corresponding to transition lines ... 
        ##sigma## 
        ...SIGMA LINES... 
        ##states## 
        ...TRANSITION LINES...
        -1 -1 -1 -1 -1
        ##end##
        
        Example (from FST.re("a.<5.0>")):
        
        ##foma-net 1.0##
        ##props##
        1 3 3 5 1 2 1 2 2 1 1 64 0A611F40
        ##weights##
        5.0 inf
        0.0 inf
        0.0
        0.0
        ##sigma##
        1 @_UNKNOWN_SYMBOL_@
        2 @_IDENTITY_SYMBOL_@
        3 a
        ##states##
        0 3 1 0
        1 2 2 0
        3 2
        2 -1 -1 1
        -1 -1 -1 -1 -1
        ##end##
        
        The initial identifier is "##foma-net 1.0##" 
        where 1.0 is the version number for the file format 
        followed by the line "##props##" 
        which is followed by a line of space separated integers 
        which correpond to: 
        
          arity  arccount  statecount  linecount  finalcount  pathcount  is_deterministic 
          is_pruned  is_minimized  is_epsilon_free  is_loop_free  extras  fst_name
        
        where fst_name is used if defined networks are saved/loaded in foma. The booleans
        (like is_deterministic) are 0 = NO, 1 = YES, 2 = UNKNOWN
        
        "extras" compresses a few details used in foma into the 8 least significant bits of
        the integer:
        
          ppqqrrss ; pp = has_weights, qq = arcs_sorted_out, rr = arcs_sorted_in, ss = is_completed
        
        Following the props line, we accept anything (for future expansion)
        until we find ##sigma## or ##weights##
        
        The section beginning with "##sigma##" consists of lines with two space-separated 
        fields: "number string", correponding to the symbol number and the symbol itself.
        Symbols themselves can contain a space character since we split on the first 
        occurrence of space. 
        
        The section beginning with "##states##" consists of lines of ASCII integers 
        with 2-5 fields to avoid some redundancy in every line corresponding to a 
        transition where otherwise state numbers would be unnecessarily repeated and 
        out symbols also (if in = out as is the case for recognizers/simple automata) 
        
        The information depending on the number of fields in the lines is as follows: 
        
          2: in target (here state_no is the same as the last mentioned one and out = in) 
          3: in out target (again, state_no is the same as the last mentioned one) 
          4: state_no in target final_state (where out = in) 
          5: state_no in out target final_state 
         
        States without outgoing transitions are represented as a 4-field: 
        
          state_no -1 -1 final_state 
        
        We add a ##weights## section that has exactly as many lines as the ##states##-section does,
        disregarding the sentinel line "-1 -1 -1 -1 -1".
        Each line consists of one or two space-separated weights, with "inf" representing infinity.
        If there is only one weight, that weight refers to a transition declared on the corresponding 
        ##states##-line or the finalweight for lines that declare a state with no outgoing transitions.
        If there are two weights, the first weight corresponds to the transition, and the second
        corresponds to the final weight on the source state of that line.
        
        For example, here is an automaton with two states declaring the language (ab)*, 3 = a, 4 = b. 
        
        ##states##
        0 3 1 1
        1 4 0 0
        -1 -1 -1 -1 -1   < SENTINEL LINE
        
        If we had a weight of 1.0 for the two transitions, and a weight of 2.0 for the final weight of state 0,
        the ##weights##-lines would look as follows:
        
        ##weights##
        1.0 2.0        < transition has weight 1.0, finalweight for this state is 2.0
        1.0            < transition has weight 1.0
        
        Foma distinguishes between @ (= @_IDENTITY_SYMBOL_@) and ? (= @_UNKNOWN_SYMBOL_@) on transitions
        whereas pyfoma does not - only '.' is special in pyfoma.
        Both get translated to '.' in pyfoma. However, in the other direction e.g. the label ('.', 'a') becomes
        @_UNKNOWN_SYMBOL_@:a, whereas ('.',) becomes "@_IDENTITY_SYMBOL_@".
        pyfoma cannot currently express ?:?, which in foma means "translate some symbol outside the alphabet
        to some other symbol outside the alphabet" (i.e. not an identity relationship).
        """
        
        fomastring = iter(fomastr.split("\n"))
        mode = 'start'
        weightslines = []
        weightlineptr = -1
        for line in fomastring:
            if mode == 'start':                # Get everything before alphabet declaration
                version = line                 # ##foma-net 1.0##
                propsline = next(fomastring)   # ##props##
                propstr = next(fomastring)     # e.g.  2 1 2 3 1 1 1 1 1 1 1 2 4D93F97B
                newfst = FST()
                newfst.alphabet = set()
                sigmastr = ''                  # Skip everything until we find "##weights##" or "##sigma##"
                while sigmastr != '##sigma##' and sigmastr != "##weights##":
                    sigmastr = next(fomastring)
                mode = sigmastr
                alphabet, statedict = {}, {'0':newfst.initialstate} # Initial state must be 0
                
            elif mode == '##weights##': # collect weights (if found)
                if line == "##sigma##":
                    mode = "##sigma##"
                else:
                    weightslines.append(line)
            
            elif mode == '##sigma##':  # collect alphabet
                if line == '##states##':
                    mode = 'states'
                    for key, label in alphabet.items():
                        if label == "@_IDENTITY_SYMBOL_@" or label == "@_UNKNOWN_SYMBOL_@":
                            alphabet[key] = '.'
                        if label == "@_EPSILON_SYMBOL_@":
                            continue     # Unlike foma, don't put epsilon in the alphabet
                        if label == ".":
                            alphabet[key] = '\\.'
                        newfst.alphabet.add(alphabet[key])
                    continue
                number, symbol = line.split(" ", 1) # First space separates the number and the symbol string
                alphabet[number] = symbol
    
            elif mode == 'states': # collect transitions

                if line == '##end##':
                    return newfst
                if line == "-1 -1 -1 -1 -1":
                    continue
                    
                weightlineptr += 1
                if weightslines:
                    weightfields = weightslines[weightlineptr].split()

                final_state = False
                outsym = None
    
                transition = line.split()
                
                if len(transition) == 5:
                    source, insym, outsym, target, final_state = transition
                    
                elif len(transition) == 4:
                    source, insym, target, final_state = transition
                    outsym = None
                    if insym == '-1':
                        if source not in statedict:
                            statedict[source] = State()
                            statedict[source].name = source
                        newfst.states.add(statedict[source])
                        newfst.finalstates.add(statedict[source])
                        if weightslines: # since insym -1 we have a finalweight
                            statedict[source].finalweight = weightfields[0]
                        else:
                            statedict[source].finalweight = 0.0
                        
    
                elif len(transition) == 3:
                    insym, outsym, target = transition
    
                elif len(transition) == 2:
                    insym, target = transition
                    outsym = None
    
                for state in [source, target]:
                    if state not in statedict and state != '-1':
                        # Create new state if not seen before
                        statedict[state] = State()
                        newfst.states.add(statedict[state])
                        statedict[state].name = state
                        statedict[state].finalweight = float('inf') # May be updated later
    
                if final_state == '1':
                    if weightslines:
                        if len(weightfields) == 2:
                            statedict[source].finalweight = float(weightfields[1])
                        else:
                            statedict[source].finalweight = float(weightfields[0])
                    else:
                        statedict[source].finalweight = 0.0
                    newfst.finalstates.add(statedict[source])
    
                if insym != '-1':
                    if outsym == None:
                        label = (alphabet[insym],)
                    else:
                        label = (alphabet[insym], alphabet[outsym])
                    if len(weightslines) == 0:
                        tweight = 0.0
                    else:
                        tweight = float(weightfields[0])
                    statedict[source].add_transition(statedict[target], label, tweight)

    def to_fomastring(self, fstname = None) -> str:
        """Converts an FST to foma's string-based representation."""
    
        NO, YES, UNKNOWN = 0, 1, 2
        outfst = self.label_states_topology()
        arity_ = outfst.arity()
        arccount_ = outfst.arccount()
        statecount_ = len(outfst.states)
        finalcount_ = len(outfst.finalstates)
        pathcount_ = outfst.pathcount()
        is_deterministic_ = YES if outfst.is_deterministic() else NO
        is_pruned_ = UNKNOWN
        is_minimized_ = UNKNOWN
        is_epsilon_free_ = NO if any(label == ('',) for s in outfst.states for label, _ in s.all_transitions()) else YES
        is_loop_free_ = NO if pathcount_ == -1 else YES
        weighted_ = YES if outfst.has_weights() else NO
        extras_ = weighted_ << 6
    
        sigmamap = {}
        cntr = 3
        for symbol in sorted(list(outfst.alphabet)):
            if symbol != '' and symbol != '.':
                sigmamap[symbol] = cntr
                cntr += 1
        if '.' in outfst.alphabet:
            sigmamap['@_UNKNOWN_SYMBOL_@'] = 1
            sigmamap['@_IDENTITY_SYMBOL_@'] = 2
        if any('' in label for s in outfst.states for label, transition in s.all_transitions()):
            sigmamap['@_EPSILON_SYMBOL_@'] = 0
        sigmastr = ["##sigma##"]
        for name, num in sorted(list(sigmamap.items()), key = lambda x: int(x[1])):
            sigmastr.append(f"{num} {name}")
    
        statestr = ["##states##"]
        weightstr = ["##weights##"]
        linecount_ = 1  # Foma needs to know the number of lines in the ##states## section
        for state in [outfst.initialstate] + list(outfst.states - {outfst.initialstate}):
            ts = list(state.all_transitions())
            finalstate = 1 if state in outfst.finalstates else 0
            if len(ts) == 0:            
                statestr.append(f"{state.name} -1 -1 {finalstate}")
                weightstr.append(f"{state.finalweight}")
                linecount_ += 1
            firsttransition = True
            for label, transition in ts:
                if '.' in label:
                    if len(label) == 1:
                        label = ("@_IDENTITY_SYMBOL_@",)
                    else:
                        label = [x if x != '.' else "@_UNKNOWN_SYMBOL_@" for x in label]
                if '' in label:
                    label = [x if x != '' else "@_EPSILON_SYMBOL_@" for x in label]

                insym = sigmamap[label[0]]
                if len(label) == 2:
                    outsym = sigmamap[label[1]]
                    sigstr = f"{insym} {outsym}"
                else:
                    sigstr = f"{insym}"
                if firsttransition:
                    statestr.append(f"{state.name} {sigstr} {transition.targetstate.name} {finalstate}")
                    weightstr.append(f"{transition.weight} {state.finalweight}")
                else:
                    statestr.append(f"{sigstr} {transition.targetstate.name}")
                    weightstr.append(f"{transition.weight}")
                linecount_ += 1
                firsttransition = False
        statestr.append("-1 -1 -1 -1 -1")
    
        intro = ["##foma-net 1.0##", "##props##"]
        if fstname == None:
            fstname = f"{id(outfst):08X}"[-8:] # Generate name if not given
        intro.append(f"{arity_} {arccount_} {statecount_} {linecount_} {finalcount_} {pathcount_} {is_deterministic_} {is_pruned_} {is_minimized_} {is_epsilon_free_} {is_loop_free_} {extras_} {fstname}")
        if weighted_ == YES:
            fststring = "\n".join(intro + weightstr + sigmastr + statestr + ["##end##"]) + "\n"
        else:
            fststring = "\n".join(intro + sigmastr + statestr + ["##end##"]) + "\n"
        return fststring

                
    # ==================
    # Saving and Loading
    # ==================

    def save(self, path: str):
        """Saves the current FST to a file.
        Args:
            path (str): The path to save to (without a file extension)
        """
        if not path.endswith('.fst'):
            path = path + '.fst'
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FST':
        """Loads an FST from a .fst file.
        Args:
            path (str): The path to load from. Must be a `.fst` file
        """
        if not path.endswith('.fst'):
            path = path + '.fst'
        with open(path, 'rb') as f:
            fst = pickle.load(f)
        return fst

    def load_foma(path: str) -> Dict[str, 'FST']:
        """Loads FSTs from a foma-format file.
        Args:
            path (str): The path to load from. Can be gzip-compressed
            or a plain text foma-file. Returns a dictionary of
            FSTs keyed by a name."""

        try:
            gz = gzip.open(path, 'rt')
            fomastrings = gz.read().split("\n")
        except gzip.BadGzipFile:
            gz = open(path, 'rt')
            fomastrings = gz.read().split("\n")
        except Exception as e:
            raise IOError(f"Error reading file {path}: {str(e)}")
    
        splits = [0] + [idx+1 for idx, line in enumerate(fomastrings) if line == "##end##"]
        slices, names = [], []
        for i in range(len(splits)-1):
            slices.append(slice(splits[i], splits[i+1], 1))    # Produce slice of lines corresponding to one FST
            names.append(fomastrings[splits[i]+2].split()[12]) # Grab name from line 3
        fsts = {name:FST.from_fomastring("\n".join(fomastrings[aslice])) for name, aslice in zip(names, slices)}
        return fsts

    def save_foma(fstdict: Dict[str, 'FST'], path: str):
        """Saves a dictionary of named (keys) FSTs (values) into a single foma-format file."""
        with gzip.open(path, 'wt') as f:
            for fstname, fst in fstdict.items():
                f.write(FST.to_fomastring(fst, fstname))


    def save_att(self, base: PathLike, state_symbols=False, epsilon="@0@"):
        """Save to AT&T format files for use with other FST libraries
        (Foma, OpenFST, RustFST, HFST, etc).

        This will, in addition to saving the transitions in `base`,
        also create separate files with the extensions `.isyms` and
        `.osyms` containing the input and output symbol tables (so for
        example if base is `test.fst`, it will create `test.isyms` and
        `test.osyms`)

        Note also that the AT&T format has no mechanism for
        quoting or escaping characters (notably whitespace) in symbols
        and state names, but only tabs are used as field separators by
        default, so any other characters should be acceptable (though
        not always recommended).  The symbol `@0@` is used by default
        for epsilon (but can be changed with the `epsilon` parameter)
        as this is Foma's default, and will always have symbol ID 0 as
        this is required by OpenFST.

        If `state_symbols` is true, the names of states will be
        retained in the output file and a state symbol table created
        with the extension `.ssyms`.  This option is disabled by
        default since it is not compatible with Foma.

        Note also that unreachable states are not included in the output.
        """
        path = Path(base)
        ssympath = path.with_suffix(".ssyms")
        isympath = path.with_suffix(".isyms")
        osympath = path.with_suffix(".osyms")
        # Number states and create state symbol table (see
        # todict() for why we must do this in a repeatable way)
        q = deque([self.initialstate])
        states: List[State] = []
        ssyms: List[str] = []
        ssymtab = {}
        while q:
            state = q.popleft()
            if state.name is None or not state_symbols:
                name = str(len(ssyms))
            else:
                name = state.name
            ssymtab[id(state)] = name
            ssyms.append(name)
            states.append(state)
            # Make sure to sort here too as the order of insertion will
            # vary as a consequence of different ordering of states
            for label, arcs in sorted(state.transitions.items(),
                                      key=operator.itemgetter(0)):
                # FIXME: it is not possible to guarantee the ordering
                # here.  Consider not using `set` for arcs.
                for arc in sorted(arcs, key=operator.attrgetter("weight")):
                    if id(arc.targetstate) not in ssymtab:
                        q.append(arc.targetstate)
        if state_symbols:
            with open(ssympath, "wt") as outfh:
                for idx, name in enumerate(ssyms):
                    print(f"{name}\t{idx}", file=outfh)
        # Do a second pass to output the FST itself (we will always have
        # to do this because of the need to number states)
        isyms = {epsilon: 0}
        osyms = {epsilon: 0}

        def output_state(s: State, outfh: TextIO):
            name = ssymtab[id(s)]
            for label, arcs in sorted(state.transitions.items(),
                                      key=operator.itemgetter(0)):
                if len(label) == 1:
                    isym = osym = (label[0] or epsilon)
                else:
                    isym, osym = ((x or epsilon) for x in label)
                if isym not in isyms:
                    isyms[isym] = len(isyms)
                if osym not in osyms:
                    osyms[osym] = len(osyms)
                for transition in sorted(arcs, key=operator.attrgetter("weight")):
                    dest = ssymtab[id(transition.targetstate)]
                    fields = [
                        name,
                        dest,
                        isym,
                        osym,
                    ]
                    if transition.weight != 0.0:
                        fields.append(str(transition.weight))
                    print("\t".join(fields), file=outfh)
            # NOTE: These are not required to be at the end of the file
            if s in self.finalstates:
                name = ssymtab[id(s)]
                if s.finalweight != 0.0:
                    print(f"{name}\t{s.finalweight}", file=outfh)
                else:
                    print(name, file=outfh)

        with open(path, "wt") as outfh:
            for state in states:
                output_state(state, outfh)
        with open(isympath, "wt") as outfh:
            for name, idx in isyms.items():
                print(f"{name}\t{idx}", file=outfh)
        with open(osympath, "wt") as outfh:
            for name, idx in osyms.items():
                print(f"{name}\t{idx}", file=outfh)

    def todict(self) -> Dict[str, Any]:
        """Create a dictionary form of the FST for export to
        JSON.  May be post-processed for optimization in Javascript."""
        # Traverse, renumbering all the states, because:
        # 1. It removes unreachable states and saves space/bandwidth
        # 2. The JS code requires the initial state to have number 0
        # 3. pyfoma uses a `set` to store states, and sets are not
        #    order-preserving in Python, while dicts are, so two FSTs
        #    created with the same input to `FST.regex` will end up with
        #    different state numberings and thus different JSON unless we
        #    enforce an ordering on them here.
        q = deque([self.initialstate])
        states: List[State] = []
        statenums = {}
        while q:
            state = q.popleft()
            if id(state) in statenums:
                continue
            statenums[id(state)] = len(states)
            states.append(state)
            # Make sure to sort here too as the order of insertion will
            # vary as a consequence of different ordering of states
            for label, arcs in sorted(state.transitions.items(),
                                      key=operator.itemgetter(0)):
                # FIXME: it is not possible to guarantee the ordering
                # here.  Consider not using `set` for arcs.
                for arc in sorted(arcs, key=operator.attrgetter("weight")):
                    if id(arc.targetstate) not in statenums:
                        q.append(arc.targetstate)
        transitions: Dict[int, Dict[str, List[int]]] = {}
        finals = {}
        alphabet: Dict[str, int] = {}
        for src, state in enumerate(states):
            for label, arcs in sorted(state.transitions.items(), key=operator.itemgetter(0)):
                if len(label) == 1:
                    isym = osym = label[0]
                else:
                    isym, osym = label
                for sym in isym, osym:
                    # Omit epsilon from symbol table
                    if sym == "":
                        continue
                    if sym not in alphabet:
                        # Reserve 0, 1, 2 for epsilon, identity, unknown
                        # (actually not necessary)
                        alphabet[sym] = 3 + len(alphabet)
                    sym = pyre.sub(r"\?|", r"\|", sym)
                tlabel = isym if isym == osym else f"{isym}|{osym}"
                # Nothing to do to the symbols beyond that as pyfoma
                # already uses the same convention of epsilon='', and JSON
                # encoding will take care of escaping everything for us.
                for arc in sorted(arcs, key=operator.attrgetter("weight")):
                    transitions.setdefault(src, {}).setdefault(tlabel, []).append(
                        # Ignore weights for now (but will support soon)
                        statenums[id(arc.targetstate)]
                    )
            if state in self.finalstates:
                finals[src] = 1
        return {
            "transitions": transitions,
            "alphabet": alphabet,
            "finals": finals,
        }

    def tojs(self, jsnetname: str = "myNet") -> str:
        """Create Javascript compatible with `foma_apply_down.js`"""
        fstdict = self.todict()
        # Optimize for foma_apply_down.js
        transitions = {}
        for src, out in fstdict["transitions"].items():
            for label, arcs in out.items():
                syms = pyre.split(r"(?<!\\)\|", label)
                isym = syms[0]
                osym = syms[-1]
                transitions.setdefault(f"{src}|{isym}", []).extend(
                    # NOTE: There is no reason for these to be
                    # separate objects, but foma_apply_down.js wants
                    # them that way.
                    {arc: osym} for arc in arcs
                )
        # NOTE: in reality foma_apply_down.js only needs the *input*
        # symbols, so we could further optimize this.
        fstdict["s"] = fstdict["alphabet"]
        del fstdict["alphabet"]
        fstdict["maxlen"] = max(len(k.encode('utf-16le'))
                                for k in fstdict["s"]) // 2
        fstdict["f"] = fstdict["finals"]
        del fstdict["finals"]
        fstdict["t"] = transitions
        del fstdict["transitions"]
        return " ".join(("var", jsnetname, "=",
                         json.dumps(fstdict, ensure_ascii=False), ";"))

    @classmethod
    def fromdict(cls, fstdict: Dict) -> "FST":
        """Recreate an FST from dictionary form."""
        fst = FST(alphabet=set(fstdict["alphabet"].keys()))
        states: List[State] = []

        def add_up_to(state_idx: int):
            while len(states) <= state_idx:
                state = State()
                idx = len(states)
                states.append(state)
                fst.states.add(state)
                if idx == 0:
                    fst.initialstate = state
                if idx in fstdict["finals"]:
                    fst.finalstates.add(state)
                    state.finalweight = fstdict["finals"][idx]
                elif str(idx) in fstdict["finals"]:
                    fst.finalstates.add(state)
                    state.finalweight = fstdict["finals"][str(idx)]
            return states[state_idx]

        for src, arcs in fstdict["transitions"].items():
            src_state = add_up_to(int(src))
            for tlabel, targets in arcs.items():
                labels = tuple(sym.replace(r"\|", "|")
                               for sym
                               in pyre.split(r"(?<!\\)\|", tlabel, maxsplit=2))
                for target in targets:
                    if isinstance(target, tuple):
                        dest, weight = target
                    else:
                        dest = target
                        weight = 0.0
                    dest_state = add_up_to(int(dest))
                    src_state.add_transition(dest_state, labels, weight)
        return fst

    # ==================
    # Rendering
    # ==================

    def view(self, raw=False, show_weights=False, show_alphabet=True) -> 'graphviz.Digraph':
        """Creates a 'graphviz.Digraph' object to view the FST. Will automatically display the FST in Jupyter.

            :param raw: if True, show label tuples and weights unformatted
            :param show_weights: force display of weights even if 0.0
            :param show_alphabet: displays the alphabet below the FST
            :return: A Digraph object which will automatically display in Jupyter.

           If you would like to display the FST from a non-Jupyter environment, please use :code:`FST.render`
        """
        import graphviz
        if not util.check_graphviz_installed():
            raise EnvironmentError("Graphviz executable not found. Please install [Graphviz](https://www.graphviz.org/download/). On macOS, use `brew install graphviz`.")

        def _float_format(num):
            if not show_weights:
                return ""
            s = '{0:.2f}'.format(num).rstrip('0').rstrip('.')
            s = '0' if s == '-0' else s
            return "/" + s

        def _str_fmt(s):  # Use greek lunate epsilon symbol U+03F5
            return (sublabel if sublabel != '' else '&#x03f5;' for sublabel in s)

        #        g = graphviz.Digraph('FST', filename='fsm.gv')

        sigma = "&Sigma;: {" + ','.join(sorted(a for a in self.alphabet)) + "}" \
            if show_alphabet else ""
        g = graphviz.Digraph('FST', graph_attr={"label": sigma, "rankdir": "LR"})
        statenums = self.number_unnamed_states()
        if show_weights == False:
            if any(t.weight != 0.0 for _, _, t in all_transitions(self.states)) or \
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

    def render(self, view=True, filename: str='FST', format='pdf', tight=True):
        """
        Renders the FST to a file and optionally opens the file.
        :param view: If True, the rendered file will be opened.
        :param format: The file format for the Digraph. Typically 'pdf', 'png', or 'svg'. View all formats: https://graphviz.org/docs/outputs/
        :param tight: If False, the rendered file will have whitespace margins around the graph.
        """
        import graphviz
        digraph = cast(graphviz.Digraph, self.view())
        digraph.format = format
        if tight:
            digraph.graph_attr['margin'] = '0' # Remove padding
        digraph.render(view=view, filename=filename, cleanup=True)

    # ==================
    # Application
    # ==================

    def generate(self: 'FST', word, weights=False, tokenize_outputs=False, obey_flags=True, print_flags=False):
        """Pass word through FST and return generator that yields all outputs."""
        yield from self.apply(word, inverse=False, weights=weights, tokenize_outputs=tokenize_outputs, obey_flags=obey_flags, print_flags=print_flags)

    def analyze(self: 'FST', word, weights=False, tokenize_outputs=False, obey_flags=True, print_flags=False):
        """Pass word through FST and return generator that yields all inputs."""
        yield from self.apply(word, inverse=True, weights=weights, tokenize_outputs=tokenize_outputs, obey_flags=obey_flags, print_flags=print_flags)

    def apply(self: 'FST', word, inverse=False, weights=False, tokenize_outputs=False, obey_flags=True, print_flags=False):
        """Pass word through FST and return generator that yields outputs.
           if inverse == True, map from range to domain.
           weights is by default False. To see the cost, set weights to True.
           obey_flags toggles whether invalid flag diacritic
           combinations are filtered out. By default, flags are
           treated as epsilons in the input. print_flags toggels whether flag
           diacritics are printed in the output. """
        IN, OUT = [-1, 0] if inverse else [0, -1]  # Tuple positions for input, output
        cntr = itertools.count()
        w = self.tokenize_against_alphabet(word)
        Q: List[Tuple[float, int, int, List[str], Optional[State]]] = []
        # output = []
        heapq.heappush(Q, (0.0, 0, next(cntr), [], self.initialstate))  # (cost, -pos, output, state)
        flag_filter = FlagStringFilter(self.alphabet) if obey_flags else None

        while Q:
            cost, negpos, _, output, state = heapq.heappop(Q)

            if state == None and -negpos == len(w) and (not obey_flags or flag_filter(output)):
                if not print_flags:
                    output = FlagOp.filter_flags(output)
                yield_output = ''.join(output) if not tokenize_outputs else output
                if weights == False:
                    yield yield_output
                else:
                    yield (yield_output, cost)
            elif state != None:
                if state in self.finalstates:
                    heapq.heappush(Q, (cost + state.finalweight, negpos, next(cntr), output, None))
                for lbl, t in state.all_transitions():
                    if lbl[IN] == '' or FlagOp.is_flag(lbl[IN]):
                        heapq.heappush(Q, (cost + t.weight, negpos, next(cntr), output + [lbl[OUT]], t.targetstate))
                    elif -negpos < len(w):
                        nextsym = w[-negpos] if w[-negpos] in self.alphabet else '.'
                        appendedsym = w[-negpos] if (nextsym == '.' and lbl[OUT] == '.') else lbl[OUT]
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

    def words_nbest(self, n) -> list:
        """Finds the n cheapest word in an FST, returning a list."""
        return list(itertools.islice(self.words_cheapest(), n))

    def words_cheapest(self):
        """A generator to yield all words in order of cost, cheapest first."""
        cntr = itertools.count()
        Q: List[Tuple[float, int, Optional[State], List]] = [(0.0, next(cntr), self.initialstate, [])]
        while Q:
            cost, _, s, seq = heapq.heappop(Q)
            if s is None:
                yield cost, seq
            else:
                if s in self.finalstates:
                    heapq.heappush(Q, (cost + s.finalweight, next(cntr), None, seq))
                for label, t in s.all_transitions():
                    heapq.heappush(Q, (cost + t.weight, next(cntr), t.targetstate, seq + [label]))

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

    # ==================
    # Operations
    # ==================

    def trim(self) -> 'FST':
        """Remove states that aren't both accessible and coaccessible."""
        return self.filter_accessible().filter_coaccessible()

    def filter_accessible(self) -> 'FST':
        """Remove states that are not on a path from the initial state."""
        new_fst = self.__copy__()
        explored = { new_fst.initialstate }
        stack = deque([new_fst.initialstate])
        while stack:
            source = stack.pop()
            for label, transition in source.all_transitions():
                if transition.targetstate not in explored:
                    explored.add(transition.targetstate)
                    stack.append(transition.targetstate)
        new_fst.states = explored
        new_fst.finalstates &= new_fst.states
        return new_fst

    def filter_coaccessible(self) -> 'FST':
        """Remove states and transitions to states that have no path to a final state."""
        new_fst = self.__copy__()
        explored = {new_fst.initialstate}
        stack = deque([new_fst.initialstate])
        inverse = {s: set() for s in new_fst.states}  # store all preceding arcs here
        while stack:
            source = stack.pop()
            for target in source.all_targets():
                inverse[target].add(source)
                if target not in explored:
                    explored.add(target)
                    stack.append(target)

        stack = deque([s for s in new_fst.finalstates])
        coaccessible = {s for s in new_fst.finalstates}
        while stack:
            source = stack.pop()
            for previous in inverse[source]:
                if previous not in coaccessible:
                    coaccessible.add(previous)
                    stack.append(previous)

        coaccessible.add(new_fst.initialstate)  # Let's make an exception for the initial
        for s in new_fst.states:  # Need to also remove transitions to non-coaccessibles
            s.remove_transitions_to_targets(new_fst.states - coaccessible)

        new_fst.states &= coaccessible
        new_fst.finalstates &= new_fst.states
        return new_fst

    def push_weights(self) -> 'FST':
        """Push weights toward the initial state. Calls dijkstra and maybe scc."""
        new_fst = self.__copy__()
        potentials = {s:algorithms.dijkstra(new_fst, s) for s in new_fst.states}
        for s, _, t in all_transitions(new_fst.states):
            t.weight += potentials[t.targetstate] - potentials[s]
        for s in new_fst.finalstates:
            s.finalweight = s.finalweight - potentials[s]
        residualweight = potentials[new_fst.initialstate]
        if residualweight != 0.0:
            # Add residual to all exits of initial state SCC and finals in that SCC
            mainscc = next(s for s in algorithms.scc(new_fst) if new_fst.initialstate in s)
            for s, _, t in all_transitions(mainscc):
                if t.targetstate not in mainscc: # We're exiting the main SCC
                    t.weight += residualweight
            for s in mainscc & new_fst.finalstates: # Add res w to finals in initial SCC
                s.finalweight += residualweight
        return new_fst

    def map_labels(self, map: dict) -> 'FST':
        """Relabel the transducer with new labels from dictionary mapping.

        Example: `fst.map_labels({'a':'', 'b':'a'})`"""
        new_fst = self.__copy__()
        for s in new_fst.states:
            newlabelings = []
            for lbl in s.transitions.keys():
                if any(l in lbl for l in map):
                    newlabel = tuple(map[lbl[i]] if lbl[i] in map else lbl[i] for i in range(len(lbl)))
                    newlabelings.append((lbl, newlabel))
            for old, new in newlabelings:
                s.rename_label(old, new)
        new_fst.alphabet = new_fst.alphabet - map.keys() | set(map.values()) - {''}
        return new_fst

    def epsilon_remove(self) -> 'FST':
        """Create new epsilon-free FSM equivalent to original."""
        # For each state s, figure out the min-cost w' to hop to a state t with epsilons
        # Then, add the (non-e) transitions of state t to s, adding w' to their cost
        # Also, if t is final and s is not, make s final with cost t.finalweight ⊗ w'
        # If s and t are both final, make s's finalweight s.final ⊕ (t.finalweight ⊗ w')

        eclosures = {s:states.epsilon_closure(s) for s in self.states}
        if all(len(ec) == 0 for ec in eclosures.values()): # bail, no epsilon transitions
            return self.__copy__()
        new_fst, mapping = self.copy_filtered(labelfilter = lambda lbl: any(len(sublabel) != 0 for sublabel in lbl))
        for state, ec in eclosures.items():
            for target, cost in ec.items():
                # copy target's transitions to source
                for label, t in target.all_transitions():
                    if all(len(sublabel) == 0 for sublabel in label): # is epsilon: skip
                        continue
                    mapping[state].add_transition(mapping[t.targetstate], label, cost + t.weight)
                if target in self.finalstates:
                    if state not in self.finalstates:
                        new_fst.finalstates.add(mapping[state])
                        mapping[state].finalweight = 0.0
                    mapping[state].finalweight += cost + target.finalweight
        return new_fst

    def label_states_topology(self, mode = 'BFS') -> 'FST':
        """Topologically sort and label states with numbers.
        Keyword arguments:
        mode -- 'BFS', i.e. breadth-first search by default. 'DFS' is depth-first.
        """
        new_fst = self.__copy__()
        cntr = itertools.count()
        Q = deque([new_fst.initialstate])
        inqueue = {new_fst.initialstate}
        while Q:
            s = Q.popleft() if mode == 'BFS' else Q.pop()
            s.name = str(next(cntr))
            for label, t in s.all_transitions():
                if t.targetstate not in inqueue:
                    Q.append(t.targetstate)
                    inqueue.add(t.targetstate)
        return new_fst

    def determinize_unweighted(self) -> 'FST':
        """Determinize with all zero weights."""
        return self.determinize(staterep = lambda s, w: (s, 0.0), oplus = lambda *x: 0.0)

    def determinize_as_dfa(self) -> 'FST':
        """Determinize as a DFA with weight as part of label, then apply unweighted det."""
        new_fst = self.copy_mod(modlabel = lambda l, w: l + (w,), modweight = lambda l, w: 0.0)
        determinized = new_fst.determinize_unweighted() # run det, then move weights back
        return determinized.copy_mod(modlabel = lambda l, _: l[:-1], modweight = lambda l, _: l[-1])

    def determinize(self, staterep = lambda s, w: (s, w), oplus = min) -> 'FST':
        """Weighted determinization of FST."""
        new_fst = FST(alphabet = self.alphabet.copy())
        firststate = frozenset({staterep(self.initialstate, 0.0)})
        statesets = {firststate:new_fst.initialstate}
        if self.initialstate in self.finalstates:
            new_fst.finalstates = {new_fst.initialstate}
            new_fst.initialstate.finalweight = self.initialstate.finalweight

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
                    new_fst.states.add(statesets[newQ])
                    #statesets[newQ].name = {(s.name, w) if w != 0.0 else s.name for s, w in newQ}
                else:
                    newstate = statesets[newQ]
                statesets[currentQ].add_transition(newstate, label, wprime)
                if any(t.targetstate in self.finalstates for _, t in tset):
                    new_fst.finalstates.add(newstate)
                    # State was final, so we discharge the maximum debt we can
                    newstate.finalweight = oplus(t.targetstate.finalweight + t.weight + \
                        residuals[s] - wprime for s, t in tset if t.targetstate in self.finalstates)
        return new_fst

    def minimize_as_dfa(self) -> 'FST':
        """Minimize as a DFA with weight as part of label, then apply unweighted min."""
        new_fst = self.copy_mod(modlabel = lambda l, w: l + (w,), modweight = lambda l, w: 0.0)
        minimized_fst = new_fst.minimize() # minimize, and shift weights back
        return minimized_fst.copy_mod(modlabel = lambda l, _: l[:-1], modweight = lambda l, _: l[-1])

    def minimize(self) -> 'FST':
        """Minimize by constrained reverse subset construction, Hopcroft-ish."""
        reverse_index = states.create_reverse_index(self.states)
        finalset, nonfinalset = self.finalstates.copy(), self.states - self.finalstates
        initialpartition = [x for x in (finalset, nonfinalset) if len(x) > 0]
        P = partition_refinement.PartitionRefinement(initialpartition)
        Agenda = {id(x) for x in (finalset, nonfinalset) if len(x) > 0}
        while Agenda:
            S = P.sets[Agenda.pop()] # convert id to the actual set it corresponds to
            for label, sourcestates in states.find_sourcestates(reverse_index, S):
                splits = P.refine(sourcestates) # returns list of (A & S, A - S) tuples
                Agenda |= {new for new, _ in splits} # Only place A & S on Agenda
        equivalenceclasses = P.astuples()
        if len(equivalenceclasses) == len(self.states):
            return self.__copy__() # we were already minimal, no need to reconstruct

        return self.merge_equivalent_states(equivalenceclasses)

    def merge_equivalent_states(self, equivalenceclasses: set) -> 'FST':
        """Merge equivalent states given as a set of sets."""
        eqmap = {s[i]:s[0] for s in equivalenceclasses for i in range(len(s))}
        representerstates = set(eqmap.values())
        new_fst = FST(alphabet = self.alphabet.copy())
        statemap = {s:State() for s in self.states if s in representerstates}
        new_fst.initialstate = statemap[eqmap[self.initialstate]]
        for s, lbl, t in all_transitions(self.states):
            if s in representerstates:
                statemap[s].add_transition(statemap[eqmap[t.targetstate]], lbl, t.weight)
        new_fst.states = set(statemap.values())
        new_fst.finalstates = {statemap[s] for s in self.finalstates if s in representerstates}
        for s in self.finalstates:
            if s in representerstates:
                statemap[s].finalweight = s.finalweight
        return new_fst

    def minimize_brz(self) -> 'FST':
        """Minimize through Brzozowski's trick."""
        return self.epsilon_remove().reverse_e().determinize().reverse_e().determinize()

    def kleene_closure(self, mode = 'star') -> 'FST':
        """Apply self*. No epsilons here. If mode == 'plus', calculate self+."""
        q1 = {k:State() for k in self.states}
        new_fst = FST(alphabet = self.alphabet.copy())

        for lbl, t in self.initialstate.all_transitions():
            new_fst.initialstate.add_transition(q1[t.targetstate], lbl, t.weight)

        for s, lbl, t in all_transitions(self.states):
            q1[s].add_transition(q1[t.targetstate], lbl, t.weight)

        for s in self.finalstates:
            for lbl, t in self.initialstate.all_transitions():
                q1[s].add_transition(q1[t.targetstate], lbl, t.weight)
            q1[s].finalweight = s.finalweight

        new_fst.finalstates = {q1[s] for s in self.finalstates}
        if mode != 'plus' or self.initialstate in self.finalstates:
            new_fst.finalstates |= {new_fst.initialstate}
            new_fst.initialstate.finalweight = 0.0
        new_fst.states = set(q1.values()) | {new_fst.initialstate}
        return new_fst

    def kleene_star(self) -> 'FST':
        """Apply self*."""
        return self.kleene_closure(mode='star')

    def kleene_plus(self) -> 'FST':
        """Apply self+."""
        return self.kleene_closure(mode='plus')

    def eliminate_flags(self) -> 'FST':
        """Equivalent behavior but no flag diacritics."""
        from ._private.eliminate_flags import eliminate_fst_flags
        return eliminate_fst_flags(self)

    def add_weight(self, weight) -> 'FST':
        """Add weight to the set of final states in the FST."""
        new_fst = self.__copy__()
        for s in new_fst.finalstates:
            s.finalweight += weight
        return new_fst

    def optional(self) -> 'FST':
        """Calculate T|'' ."""
        new_fst = self.__copy__()
        if new_fst.initialstate in new_fst.finalstates:
            return new_fst
        newinitial = State()

        for lbl, t in new_fst.initialstate.all_transitions():
            newinitial.add_transition(t.targetstate, lbl, t.weight)

        new_fst.initialstate = newinitial
        new_fst.states.add(newinitial)
        new_fst.finalstates.add(newinitial)
        newinitial.finalweight = 0.0
        return new_fst

    @harmonize_alphabet
    def concatenate(self, fst2: 'FST') -> 'FST':
        """Concatenation of T1T2. No epsilons. May produce non-accessible states."""
        fst1 = self.__copy__()
        ocopy, _ = fst2.copy_filtered() # Need to copy since self may equal other
        q1q2 = {k:State() for k in fst1.states | ocopy.states}

        for s, lbl, t in all_transitions(q1q2.keys()):
            q1q2[s].add_transition(q1q2[t.targetstate], lbl, t.weight)
        for s in fst1.finalstates:
            for lbl2, t2 in ocopy.initialstate.all_transitions():
                q1q2[s].add_transition(q1q2[t2.targetstate], lbl2, t2.weight + s.finalweight)

        new_fst = FST()
        new_fst.initialstate = q1q2[fst1.initialstate]
        new_fst.finalstates = {q1q2[f] for f in ocopy.finalstates}
        for s in ocopy.finalstates:
            q1q2[s].finalweight = s.finalweight
        if ocopy.initialstate in ocopy.finalstates:
            new_fst.finalstates |= {q1q2[f] for f in fst1.finalstates}
            for f in fst1.finalstates:
                q1q2[f].finalweight = f.finalweight + ocopy.initialstate.finalweight
        new_fst.states = set(q1q2.values())
        return new_fst

    @harmonize_alphabet
    def cross_product(self, fst2: 'FST', optional: bool = False) -> 'FST':
        """Perform the cross-product of T1, T2 through composition.
        Keyword arguments:
        optional -- if True, calculates T1:T2 | T1."""
        newfst_a = self.copy_mod(modlabel = lambda l, _: l + ('',))
        newfst_b = fst2.copy_mod(modlabel = lambda l, _: ('',) + l)
        if optional == True:
            return newfst_a.compose(newfst_b).union(self)
        else:
            return newfst_a.compose(newfst_b)

    @harmonize_alphabet
    def compose(self, fst2: 'FST') -> 'FST':
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
        fst1 = self.__copy__()
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

    def invert(self) -> 'FST':
        """Calculate the inverse of a transducer, i.e. flips label tuples around."""
        new_fst = self.__copy__()
        for s in new_fst.states:
            for lbl, tr in s.transitions.items():
                for t in tr:
                    t.label = t.label[::-1]
            s.transitions = {lbl[::-1]:tr for lbl, tr in s.transitions.items()}
        return new_fst

    def ignore(self, fst2: 'FST') -> 'FST':
        """A, ignoring intervening instances of B."""
        new_fst = FST.re("$^output($A @ ('.'|'':$B)*)", {'A': self, 'B': fst2})
        return new_fst

    def rewrite(self, *contexts, **flags) -> 'FST':
        """Rewrite self in contexts in parallel, controlled by flags."""
        defs = {'crossproducts': self.__copy__()}
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

    def context_restrict(self, *contexts, rewrite = False) -> 'FST':
        """Only allow self in the context L1 _ R1, or ... , or  L_n _ R_n."""
        for fsm in itertools.chain.from_iterable(contexts):
            fsm.alphabet.add('@=@') # Add aux sym to contexts so they don't match .
        fst = self.__copy__()
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

    def project(self, dim = 0) -> 'FST':
        """Project fst. dim = -1 will get output proj regardless of # of tapes."""
        sl = slice(-1, None) if dim == -1 else slice(dim, dim+1)
        newalphabet = set()
        new_fst = self.__copy__()
        for s in new_fst.states:
            newtransitions = {}
            for lbl, tr in s.transitions.items():
                newtransitions[lbl[sl]] = newtransitions.get(lbl[sl], set()) | tr
                for t in tr:
                    t.label = lbl[sl]
                    newalphabet |= {sublabel for sublabel in lbl[sl]}
            s.transitions = newtransitions
        if '.' not in newalphabet: # Preserve . semantics if it occurs on the tape we extract
            new_fst.alphabet = newalphabet
        return new_fst

    def reverse(self) -> 'FST':
        """Reverse the FST, epsilon-free."""
        newfst = FST(alphabet = self.alphabet.copy())
        newfst.initialstate = State()
        mapping = {k:State() for k in self.states}
        newfst.states = set(mapping.values()) | {newfst.initialstate}
        newfst.finalstates = {mapping[self.initialstate]}
        if self.initialstate in self.finalstates:
            newfst.finalstates.add(newfst.initialstate)
            newfst.initialstate.finalweight = self.initialstate.finalweight
        mapping[self.initialstate].finalweight = 0.0

        for s, lbl, t in all_transitions(self.states):
            mapping[t.targetstate].add_transition(mapping[s], lbl, t.weight)
            if t.targetstate in self.finalstates:
                newfst.initialstate.add_transition(mapping[s], lbl, t.weight + \
                                                t.targetstate.finalweight)
        return newfst

    def reverse_e(self) -> 'FST':
        """Reverse the FST, using epsilons."""
        newfst = FST(alphabet = self.alphabet.copy())
        newfst.initialstate = State(name = tuple(k.name for k in self.finalstates))
        mapping = {k:State(name = k.name) for k in self.states}
        for t in self.finalstates:
            newfst.initialstate.add_transition(mapping[t], ('',), t.finalweight)

        for s, lbl, t in all_transitions(self.states):
            mapping[t.targetstate].add_transition(mapping[s], lbl, t.weight)

        newfst.states = set(mapping.values()) | {newfst.initialstate}
        newfst.finalstates = {mapping[self.initialstate]}
        mapping[self.initialstate].finalweight = 0.0
        return newfst

    @harmonize_alphabet
    def union(self, fst2: 'FST') -> 'FST':
        """Epsilon-free calculation of union of self and fst2."""
        fst1 = self.__copy__()
        mapping = {k:State() for k in fst1.states | fst2.states}
        newfst = FST() # Get new initial state
        newfst.states = set(mapping.values()) | {newfst.initialstate}
        # Copy all transitions from old initial states to new initial state
        for lbl, t in itertools.chain(fst1.initialstate.all_transitions(), fst2.initialstate.all_transitions()):
            newfst.initialstate.add_transition(mapping[t.targetstate], lbl, t.weight)
        # Also add all transitions from old FSMs to new FSM
        for s, lbl, t in itertools.chain(all_transitions(fst1.states), all_transitions(fst2.states)):
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

    def intersection(self, fst2: 'FST') -> 'FST':
        """Intersection of self and other. Uses the product algorithm."""
        return self.product(fst2, finalf = all, oplus = operator.add, pathfollow = lambda x,y: x & y)

    def difference(self, fst2: 'FST') -> 'FST':
        """Returns self-other. Uses the product algorithm."""
        return self.product(fst2, finalf = lambda x: x[0] and not x[1],\
                            oplus = lambda x,y: x, pathfollow = lambda x,y: x)

    def complement(self) -> 'FST':
        """Returns the complement of an FST."""
        return FST.re(".* - $X", {"X": self})

    @harmonize_alphabet
    def product(self, fst2: 'FST', finalf = any, oplus = min, pathfollow = lambda x,y: x|y) -> 'FST':
        """Generates the product FST from fst1, fst2. The helper functions by default
        produce fst1|fst2."""
        fst1 = self.__copy__()
        fst2 = fst2.__copy__()
        fst1.number_unnamed_states()
        fst2.number_unnamed_states()
        new_fst = FST()
        Q = deque([(fst1.initialstate, fst2.initialstate)])
        S = {(fst1.initialstate, fst2.initialstate): new_fst.initialstate}
        dead1, dead2 = State(finalweight = float("inf")), State(finalweight = float("inf"))
        while Q:
            t1s, t2s = Q.pop()
            currentstate = S[(t1s, t2s)]
            currentstate.name = (t1s.name, t2s.name,)
            if finalf((t1s in fst1.finalstates, t2s in fst2.finalstates)):
                new_fst.finalstates.add(currentstate)
                currentstate.finalweight = oplus(t1s.finalweight, t2s.finalweight)
            # Get all outgoing labels we want to follow
            for lbl in pathfollow(t1s.transitions.keys(), t2s.transitions.keys()):
                for outtr in t1s.transitions.get(lbl, (transition.Transition(dead1, lbl, float('inf')), )):
                    for intr in t2s.transitions.get(lbl, (transition.Transition(dead2, lbl, float('inf')), )):
                        if (outtr.targetstate, intr.targetstate) not in S:
                            Q.append((outtr.targetstate, intr.targetstate))
                            S[(outtr.targetstate, intr.targetstate)] = State()
                            new_fst.states.add(S[(outtr.targetstate, intr.targetstate)])
                        currentstate.add_transition(S[(outtr.targetstate, intr.targetstate)], lbl, oplus(outtr.weight, intr.weight))
        return new_fst

    # ==================
    # Magic Methods
    # ==================

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
                    if len(label) == 1:
                        att_label = (label[0], label[0])
                    else:
                        att_label = label
                    # You get Foma's default here since it cannot be configured
                    att_label = ["@0@" if sym == "" else sym for sym in att_label]
                    for transition in s.transitions[label]:
                        st += '{}\t{}\t{}\t{}\n'.format(statenums[id(s)],\
                        statenums[id(transition.targetstate)], '\t'.join(att_label),\
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

    # ==================
    # Utilities
    # ==================

    def become(self, other: 'FST'):
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

    def cleanup_sigma(self) -> 'FST':
        """Remove symbols if they are no longer needed, including . .
        Returns a new FST with the cleaned alphabet."""
        new_fst = self.__copy__()
        seen = {sym for _, lbl, _ in all_transitions(new_fst.states) for sym in lbl}
        if '.' not in seen:
            new_fst.alphabet &= seen
        return new_fst

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

        for s, lbl, t in all_transitions(q1q2.keys()):
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

        for s, lbl, t in all_transitions(q1q2.keys()):
            if labelfilter(lbl):
                q1q2[s].add_transition(q1q2[t.targetstate], lbl, t.weight)

        for s in self.finalstates:
            q1q2[s].finalweight = s.finalweight

        return newfst, q1q2
        

    def pathcount(self):
        """Count distinct paths from initialstate to any final state."""

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {}           # state -> WHITE/GRAY/BLACK (default WHITE)
        topo_rev = []        # reverse topological order
        reachable = set()

        def child_iter(s):
            # Unique children only
            return iter(s.all_targets())
    
        # Iterative DFS with early cycle detection
        stack = []
        color[self.initialstate] = GRAY
        reachable.add(self.initialstate)
        stack.append((self.initialstate, child_iter(self.initialstate)))

        while stack:
            u, it = stack[-1]
            try:
                v = next(it)
                c = color.get(v, WHITE)
                if c == WHITE:
                    color[v] = GRAY
                    reachable.add(v)
                    stack.append((v, child_iter(v)))
                elif c == GRAY:
                    # back edge => cycle
                    return -1
                # if BLACK: ignore
            except StopIteration:
                stack.pop()
                color[u] = BLACK
                topo_rev.append(u)

        # acyclic: DP along topological order
        topo = topo_rev[::-1]
        dp = {s:0 for s in reachable}
        dp[self.initialstate] = 1
        for s in topo:
            for _, t in s.all_transitions():
                tgt = t.targetstate
                if tgt in dp:
                    dp[tgt] += dp[s]

        return sum(dp[f] for f in self.finalstates if f in dp)

    def arity(self):
        """Return the maximum length of symbol tuples in the FST's transitions.
        Returns 1 if there are no transitions."""
        try:
            return max(len(label) for s in self.states for label, _ in s.all_transitions())
        except ValueError:
            return 1  # No transitions exist
    
    def arccount(self):
        """Counts number of transitions in FST."""
        return sum(len(list(s.all_transitions())) for s in self.states)
    
    def is_deterministic(self):
        """Return True if the FST is deterministic in the DFA sense, False otherwise."""
        for state in self.states:
            # Track input symbols seen for this state
            input_labels = set()
            for label, _ in state.all_transitions():
                if label in input_labels or label == ('',):
                    return False  # Non-deterministic: multiple transitions with same input or epsilon
                input_labels.add(label)
        return True
    
    def has_weights(self):
        """Determines if FST has non-trivial weights, i.e. not all 0.0 for transitions
           and final states."""
        if any((t.weight != 0.0 and t.weight != float('inf')) for _, _, t in all_transitions(self.states)) or \
           any((s.finalweight != 0.0 and s.finalweight != float('inf')) for s in self.finalstates):
                return True
        return False

# ==================
# Global Functions
# ==================
def reverse(fst: 'FST'):
    return fst.reverse()

def invert(fst: 'FST'):
    return fst.invert()

def minimize(fst: 'FST'):
    return fst.minimize()

def determinize(fst: 'FST'):
    return fst.determinize()

def ignore(fst1: 'FST', fst2: 'FST'):
    return fst1.ignore(fst2)

def rewrite(fst: 'FST', *contexts, **flags):
    return fst.rewrite(*contexts, **flags)

def context_restrict(fst: 'FST', *contexts, rewrite = False):
    return fst.context_restrict(*contexts, rewrite = rewrite)

def project(fst: 'FST', dim = 0):
    return fst.project(dim = dim)

def concatenate(fst1: 'FST', fst2: 'FST'):
    return fst1.concatenate(fst2)
