#!/usr/bin/env python

from pyfoma.fst import FST
from pyfoma.flag import FlagOp, FlagStringFilter

class Paradigm:

    def __init__(self, grammar, regexfilter, tagfilter = lambda x: x.startswith('[') and x.endswith(']'),
                 obey_flags=True, print_flags=False):
        """Extract a 'paradigm' from a grammar FST. Available as a list in attr para.
           regexfilter -- a regex which is composed on the input side to filter out
                          a specific lexeme or set of lexemes, e.g. 'run.*'
           Keyword arguments:
           tagfilter -- a function to identify tags, by default bracketed symbols [ ... ]
           obey_flags -- whether to exlcude input-output pairs with invalid flag diacritic combinations
           print_flags -- whether to print flag diacritics in output
           """
        self.FSM = grammar
        self.regexfilter = regexfilter # a regex used for filtering input side
        self.tagfilter = tagfilter # func to identify tags vs. other symbols
        self.tables = {} # indexed by citation form of lexeme
        self.filtered = FST.re(regexfilter + " @ $grammar", {'grammar': grammar})
        self.words = self.filtered.words()
        self.flag_filter = FlagStringFilter(self.filtered.alphabet) if obey_flags else None
        para = []
        for weight, pairlist in self.words:
            lemma, tags, output = [], [], []
            if obey_flags:
                tapes = zip(*pairlist)
                if False in map(self.flag_filter,tapes):
                    continue
            for io in pairlist:
                if len(io) == 1:
                    i, o = io[0], io[0]
                else:
                    i, o = io[0], io[-1]                    
                if print_flags or not FlagOp.is_flag(i):
                    if tagfilter(i):
                        tags.append(i)
                    else:
                        lemma.append(i)
                if print_flags or not FlagOp.is_flag(o):
                    output.append(o)
            para.append([''.join(lemma), ''.join(tags), ''.join(output)])
        self.para = sorted(para)

    def __str__(self):
        """Return a formatted table with lemma, tags, wordform."""
        maxlens = (max(len(w) for w in cols) for cols in zip(*self.para)) # max for each col
        fmtstr = "".join("{:<" + str(ml+2) + "}" for ml in maxlens) + "\n"
        return "".join(fmtstr.format(*cols) for cols in self.para)
