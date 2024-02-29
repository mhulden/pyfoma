from itertools import product
from functools import reduce
import re

from pyfoma.fst import FST
from pyfoma.flag import FlagOp

EMPTYVAL="{}"

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
    """ Return minimal condition for [[$X==y]] (and [[$X?=y]]) or [[$X!=y]] to fail. 
        Precondition: y != "{}". """
    op = f"('[[${X}=={y}]]'|'[[${X}?={y}]]')" if pos else f"'[[${X}!={y}]]'"
    setval = set_neg(X,y,ys) if pos else set_pos(X,y)
    return FST.re(f".* $set (. - $any)* {op} .*", {"set":setval, "any":set_any(X,ys)})

def empty_restr(X, y, ys, pos):
    """ Return minimal condition for [[$X=={}]] (and [[$X?={}]]) or [[$X!={}]] to fail. """
    op = f"('[[${X}=={y}]]'|'[[${X}?={y}]]')" if pos else f"'[[${X}!={y}]]'"
    return FST.re(f"(. - $any)* {op} .*", {"any":set_any(X,ys)})

def eq_restr(X1, X2, y1, y2, ys):
    """ Return minimal condition for [[$X1==$X2]] or [[$X1!=$X2]] to fail given 
        values y1 and y2 (and y1 != "{}"). """
    set1 = FST.re(".* $set (. - $any)*", {"set":set_pos(X1,y1), "any":set_any(X1, ys)})
    set2 = FST.re(".* $set (. - $any)*", {"set":set_pos(X2,y2), "any":set_any(X2, ys)})
    op = f"'[[${X1}!=${X2}]]'" if y1 == y2 else f"'[[${X1}==${X2}]]'"
    return FST.re(f"($set1 & $set2) {op} .*", {"set1": set1, "set2": set2})

def empty_eq_restr(X1, X2, ys):
    """ Return minimal condition for [[$X1==$X2]] or [[$X1!=$X2]] to fail given 
        values that the value of X1 is "{}". """
    set1 = FST.re("(. - $any)*", {"any":set_any(X1,ys)})
    set2 = FST.re(".* $set (. - $any)*", 
                  {"set":set_neg(X2,EMPTYVAL,ys), "any":set_any(X2, ys)})
    return FST.re("($set1 & $set2) '[[${X1}==${X2}]]' .*", {"set1":set1, "set2":set2})

def get_value_tests(Xs, ys):
    """ Return a list of tests for [[$X==y]], [[$X?=y]] and [[$X!=y]] flags which 
        valid strings have to pass. """
    tests = []  
    for X in Xs:
        for y in ys:
            tests.append(value_restr(X, y, ys, pos=True))
            tests.append(value_restr(X, y, ys, pos=False))
            if y == EMPTYVAL:
                tests.append(empty_restr(X, y, ys, pos=False))
            else:
                tests.append(empty_restr(X, y, ys, pos=True))
    return [FST.re(".* - $r", {"r":r}) for r in tests]

def get_eq_tests(Xs, ys):
    """ Return a list of tests for [[$X==$Y]], [[$X!=$Y]] flags which valid strings 
        have to pass. """
    tests = []
    for X1, X2 in product(Xs, repeat=2):
        if X1 != X2:
            for y1, y2 in product(ys, ys):
                tests.append(eq_restr(X1, X2, y1, y2, ys))
            tests.append(empty_eq_restr(X1, X2, ys))
    return [FST.re(".* - $r", {"r":r}) for r in tests]

def eliminate_flags(fst, Xs=None):
    """Eliminate all flag diacritics from an FST. 

    :param fst: An FST.
    :param Xs: The variables to be eliminated. If None, then all
    variables will be eliminated.

    :return: An FST without flag diacritics with equivalent behavior
    to 'fst'
    """
    if Xs == None:
        flags = [FlagOp(sym) for sym in fst.alphabet if FlagOp.is_flag(sym)]
        Xs = set(flag.var for flag in flags)
        ys = set(flag.val for flag in flags)
    if len(Xs) == 0:
        return fst
    tests = get_value_tests(Xs, ys) + get_eq_tests(Xs, ys)
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

class TestValueFlags(unittest.TestCase):
    def test_neg(self):
        fst = FST.re("a")
        fst = eliminate_flags(fst)
        self.assertTrue(fst == FST.re("a"))
        fst = FST.re("a '[[$X==x]]' b")
        fst = eliminate_flags(fst)
        self.assertEqual(fst, FST.re("a b"))
        
if __name__=="__main__":
    Grammar = {}
    Grammar["S"] = [("","A")]
    Grammar["A"] = [("a'[[$X=a]]'", "B"), ("b'[[$X=b]]'", "B"), ("c", "B")]
    Grammar["B"] = [("a'[[$X==a]]'", "C"), ("b'[[$X==b]]'", "C")]
    Grammar["C"] = [("a'[[$Y=a]]'", "D"), ("b'[[$Y=b]]'", "D")]
    Grammar["D"] = [("'[[$Y==$X]]'", "#")]
    Lexicon = FST.rlg(Grammar, "S").epsilon_remove().minimize()
    Lexicon = eliminate_flags(Lexicon, "X Y".split(), "a b {}".split())
    print(Lexicon)
    unittest.main()
