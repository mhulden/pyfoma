from itertools import product
from functools import reduce
import re

from pyfoma.fst import FST
from pyfoma.flag import FlagOp, FLAGRE2, FLAGRE3, EMPTY

def set_pos(X, y):
    """ Set variable X to value y (i.e. [[$X=y]] | [[$X?=y]]). """
    return FST.re(f"('[[${X}={y}]]' | '[[${X}?={y}]]')")

def set_any(X, ys):
    """ Set variable X to any value in ys. """
    res = [set_pos(X,y) for y in ys]
    return reduce(lambda x,y: FST.re("$x | $y", {"x":x, "y":y}), res)

def set_neg(X, y, ys):
    """ Set variable X to any value in ys except y. """
    return FST.re("$any - $set", {"any":set_any(X, ys), "set":set_pos(X, y)})

def value_restr(X, y, ys, pos):
    """ Return minimal condition for [[$X==y]] (and [[$X?=y]]) or [[$X!=y]] to fail. """
    op = f"('[[${X}=={y}]]'|'[[${X}?={y}]]')" if pos else f"'[[${X}!={y}]]'"
    setval = set_neg(X,y,ys) if pos else set_pos(X,y)
    return FST.re(f".* $set (. - $any)* {op} .*", {"set":setval, "any":set_any(X,ys)})

def empty_restr(X, y, ys, pos):
    """ Return minimal condition for [[$X=={}]] (and [[$X?={}]]) or [[$X!={}]] to fail. """
    op = f"('[[${X}=={y}]]'|'[[${X}?={y}]]')" if pos else f"'[[${X}!={y}]]'"
    return FST.re(f"(. - $any)* {op} .*", {"any":set_any(X,ys)})

def get_value_tests(Xs, ys):
    """ Return a list of tests for [[$X==y]], [[$X?=y]] and [[$X!=y]] flags which 
        valid strings have to pass. """
    tests = []  
    for X in Xs:
        for y in ys:
            tests.append(value_restr(X, y, ys, pos=True))
            tests.append(value_restr(X, y, ys, pos=False))
            if y == EMPTY:
                tests.append(empty_restr(X, y, ys, pos=False))
            else:
                tests.append(empty_restr(X, y, ys, pos=True))
    return [FST.re(".* - $r", {"r":r}) for r in tests]

def substitute_no_val_flags(fst):
    """ Substitute all flags of form:
        * [[$VAR]] with [[$VAR!={}]]
        * [[!$VAR]] with [[$VAR=={}]]
        * [[$VAR=]] with [[$VAR={}]] 
    
        This is done to ensure that all flags have the format VAR_OP_VAL """

    subst = {}
    for sym in fst.alphabet:
        m = re.match(FLAGRE2,sym)
        if m:
            subst[sym] = f"[[{m.group(1)}={{}}]]"
        m = re.match(FLAGRE3,sym)
        if re.match(FLAGRE3,sym):
            if m.group(1) == "!":
                subst[sym] = f"[[{m.group(2)}=={{}}]]"
            else:
                subst[sym] = f"[[{m.group(2)}!={{}}]]"
    return fst.map_labels(subst)

def eliminate_flags(fst, Xs=None):
    """Eliminate all flag diacritics from an FST. 

    :param fst: An FST.
    :param Xs: The variables to be eliminated. If None, then all
    variables will be eliminated.

    :return: An FST without flag diacritics with equivalent behavior
    to 'fst'
    """
    fst = substitute_no_val_flags(fst)
    flags = [FlagOp(sym) for sym in fst.alphabet if FlagOp.is_flag(sym)]        
    if Xs == None:
        Xs = set(flag.var[1:] for flag in flags)
    if len(Xs) == 0:
        return fst
    ys = set(flag.val for flag in flags if flag.var[1:] in Xs)
    ys.add(EMPTY)

    tests = get_value_tests(Xs, ys)
    flag_filter = reduce(lambda x, y: FST.re("$x & $y",{"x":x, "y":y}),tests)
    flags = [sym for sym in fst.alphabet if FlagOp.is_flag(sym)]
    clean = reduce(lambda x, y: FST.re("$x @ $y",{"x":x, "y":y}), 
                   [FST.re(f"$^rewrite('{flag}':'')") for flag in flags])
    fst = FST.re("$^invert($fst @ $filter @ $clean)", 
                 {"fst":fst, "filter":flag_filter, "clean":clean})
    fst = FST.re("$^invert($fst @ $clean)", {"fst":fst, "clean":clean})
    return fst

################################
#                              #
#            TESTS             #
#                              #
################################
import unittest

def equivalent(fst1, fst2):
    comp = FST.re("($fst1 - $fst2) | ($fst2 - $fst1)", {"fst1":fst1, "fst2":fst2})
    return len(comp.finalstates) == 0

class TestValueFlags(unittest.TestCase):
    def test_pos(self):
        pos_flag_re = ["a",
                       "a b",
                       "a '[[$X=x]]' b",
                       "a '[[!$X]]' b",
                       "a '[[$X=x]]' b '[[$X=]]' '[[!$X]]'",
                       "a '[[$X=x]]' '[[$X=y]]' b",
                       "a '[[$X=x]]' '[[!$Y]]' b",
                       "a '[[$X=x]]' '[[$Y!=x]]' b",
                       "a '[[$X=x]]' '[[$Y!=y]]' b",
                       "a '[[$X=x]]' b '[[$X=y]]'",
                       "a '[[$X=x]]' b '[[$X==x]]'",
                       "a '[[$X=x]]' b '[[$X!=y]]'",
                       "a '[[$X=x]]' b '[[$X?=x]]'"]
        pos_elim_re = ["a",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b",
                       "a b"]
        assert(len(pos_flag_re) == len(pos_elim_re))
        for re1, re2 in zip(pos_flag_re, pos_elim_re):
            print("With flags:",re1, "Eliminated:", re2)
            fst1 = eliminate_flags(FST.re(re1))
            fst2 = FST.re(re2)
            self.assertTrue(equivalent(fst1, fst2))

    def test_neg(self):
        neg_flag_re = [
            "a '[[$X==x]]'",
            "a '[[$X==x]]'",
            "a '[[$X]]'",
            "a '[[$X=x]]' b '[[!$X]]'",
            "a '[[$X=x]]' b '[[$X=]]' '[[$X]]'",
            "a '[[$X=x]]' b '[[$X!=x]]'",
            "a '[[$X=x]]' b '[[$X?=y]]'",
            "a '[[$X=x]]' b '[[$X==y]]'",
        ]
        for re1 in neg_flag_re:
            print("With flags:",re1, "Should give an empty FST")
            fst1 = eliminate_flags(FST.re(re1))
            fst2 = FST()
            self.assertTrue(equivalent(fst1, fst2))
            
if __name__=="__main__":
    Grammar = {}
    Grammar["S"] = [("","A")]
    Grammar["A"] = [("a'[[$X=a]]'", "B"), ("b'[[$X=b]]'", "B"), ("c", "B")]
    Grammar["B"] = [("a'[[$X==a]]'", "C"), ("b'[[$X==b]]'", "C")]
    Grammar["C"] = [("a'[[$Y=a]]'", "D"), ("b'[[$Y=b]]'", "D")]
    Grammar["D"] = [("'[[$Y==$X]]'", "#")]
    Lexicon = FST.rlg(Grammar, "S").epsilon_remove().minimize()
    Lexicon = eliminate_flags(Lexicon, "X Y".split())
    unittest.main()
