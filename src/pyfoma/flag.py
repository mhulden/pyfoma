import re

class FlagOp:
    def __init__(self, sym):
        """Creates a Flag diacritic

        :param sym: String representation of flag diacritic 

        The parameter 'sym' should follow the format [[XYZ]], for
        example "[[$Num=Sg]]", where:

        1. X is a variable name matching the regex "[$]\w+" 
        2. Y is one of the operators "=" (set value), "==" (check that
        value equals), "!=" (check that value does not equal) or "$="
        (unify to value) 
        3. Z is a value matching the regex "[$]?\w+". If the value
        starts with $, then it refers to a variable.
        """

        match = re.match(r"\[\[(\$\w+)([?!=]?=)(\$?\w+|{})\]\]",sym)
        self.var, self.op, self.val = match.group(1,2,3)
        self.eq_flag = (self.val[0] == "$")
        self.op_func = {"=":self.setv,
                        "==":self.check,
                        "!=":self.neg_check,
                        "?=":self.unify}[self.op]

    @staticmethod
    def is_flag(sym):
        """Check that 'sym' matches the format required by FlagOp.__init__

        :param sym: A string

        :return: True is 'sym' satisfies the requirement for
        FlagOp.__init__. False, otherwise.
        """
        return re.match(r"\[\[\$\w+[?!=]?=(\$?\w+|{})\]\]",sym) != None
    
    def setv(self, config, val):
        """ The operator "=" """
        config[self.var] = val
        return True

    def check(self, config, val):
        """ The operator "==" """
        return config[self.var] == val

    def neg_check(self, config, val):
        """ The operator "!=" """
        return config[self.var] != val

    def unify(self, config, val):
        """ The operator "?=" """
        if config[self.var] in ["{}", val]:
            config[self.var] = val
            return True
        return False

    def __call__(self, config):
        """Perform test/operation specified by this flag 

        :param config: A dictionary of variable:value pairs
        (e.g. {"$var1":"val", "var2":"{}"})

        :return: True if test/operation succeeds. False, otherwise.

        The state of 'config' will change to reflect the operation
        specified by this flag (when the operator is "=" or "?=").
        """
        val = config[self.val] if self.eq_flag else self.val
        return self.op_func(config, val)        
        
class FlagFilter:
    def __init__(self, alphabet):
        """ Create FlagFilter from an FST alphabet
        
        :param alphabet: A symbol set (containing strings)
        """ 
        self.flags = {}
        self.alphabet = alphabet
        for sym in alphabet:
            if FlagOp.is_flag(sym):
                self.flags[sym] = FlagOp(sym)
        self.vars = {flag.var for flag in self.flags.values()}
        
class FlagStringFilter(FlagFilter):
    def __call__(self, seq):
        """Check that flag diacritic configuration is valid

        :param seq: A list of string symbols in self.alphabet
        
        :return: True if the combination of flag diactritics in
        'seq' is valid. False, otherwise.

        Raises KeyError when 'seq' contains symbols which are
        absent from the FST alphabet.
        """
        config = {var:"{}" for var in self.vars}
        for sym in seq:
            if not sym in self.alphabet:
                raise KeyError(sym)
            if sym in self.flags:
                if not self.flags[sym](config):
                    return False
        return True

class FlagStreamFilter(FlagFilter):
    def __init__(self, alphabet):
        """ Create FlagStreamFilter from an FST alphabet
        
        :param alphabet: A symbol set (containing strings)
        """
        super().__init__(alphabet)
        self.reset()

    def reset(self):
        """ Reset all variables to empty value "{}" """
        self.config = {var:"{}" for var in self.vars}
        self.has_failed = False
        
    def check(self, sym):
        """Read next symbol and check fla diacritic configuration

        :param sym: A symbol in self.alphabet
        
        :return: True if the combination of flag diactritics upto this
        point is valid. False, otherwise.

        Raises KeyError when 'sym' is missing from the FST alphabet.
        """
        if not sym in self.alphabet:
            raise KeyError(sym)
        if self.has_failed:
            return False
        if sym in self.flags:
            if not self.flags[sym](self.config):
                self.has_failed = True
                return False
        return True

####################################
###                              ###
###            TESTS             ###
###                              ###
####################################

import unittest
import time

class TestFlagOp(unittest.TestCase):
    def test_init(self):
        for op in "?= == != =".split():
            for val in "val {}".split():
                fo = FlagOp(f"[[$var{op}{val}]]")
                self.assertEqual(fo.var, "$var")
                self.assertEqual(fo.op, op)
                self.assertEqual(fo.val, val)

    def test_is_flag(self):
        self.assertFalse(FlagOp.is_flag(""))
        for op in "?= == != =".split():
            self.assertFalse(FlagOp.is_flag("[$var{op}val]"))
            self.assertFalse(FlagOp.is_flag("[[var{op}val]]"))
        self.assertFalse(FlagOp.is_flag("[[$var]]"))

        for op in "?= == != =".split():
            for val in "val {} $var2".split():
                self.assertTrue(FlagOp.is_flag(f"[[$var1{op}{val}]]"))

    def test_init_non_flag(self):
        self.assertRaises(AttributeError, FlagOp, sym="")
        for op in "?= == != =".split():
            self.assertRaises(AttributeError, FlagOp, sym=f"[$var{op}val]")
            self.assertRaises(AttributeError, FlagOp, sym=f"[[var{op}val]]")
        self.assertRaises(AttributeError, FlagOp, sym="[[$var]]")

    def test_call(self):
        for val in "{} foo".split():
            config = {"$var1":val}
            pos_op_val = FlagOp(f"[[$var1=={val}]]")
            neg_op_val = FlagOp(f"[[$var1!={val}]]")
            pos_op_oval = FlagOp(f"[[$var1==bar]]")
            neg_op_oval = FlagOp(f"[[$var1!=bar]]")
            self.assertTrue(pos_op_val(config))
            self.assertFalse(neg_op_val(config))
            self.assertFalse(pos_op_oval(config))
            self.assertTrue(neg_op_oval(config))
            self.assertEqual(config, {"$var1":val})
            
        for val in "{} foo".split():
            config = {"$var1":"{}"}
            set_op = FlagOp(f"[[$var1={val}]]")
            self.assertTrue(set_op(config))
            self.assertEqual(config, {"$var1":val})

            config = {"$var1":"bar"}
            set_op = FlagOp(f"[[$var1={val}]]")
            self.assertTrue(set_op(config))
            self.assertEqual(config, {"$var1":val})

            config = {"$var1":"{}"}
            unify_op = FlagOp(f"[[$var1?={val}]]")
            self.assertTrue(unify_op(config))
            self.assertEqual(config, {"$var1":val})

            config = {"$var1":val}
            unify_op = FlagOp(f"[[$var1?={val}]]")
            self.assertTrue(unify_op(config))
            self.assertEqual(config, {"$var1":val})

            config = {"$var1":"bar"}
            unify_op = FlagOp(f"[[$var1?={val}]]")
            self.assertFalse(unify_op(config))

            config = {"$var1":"foo", "$var2":"foo"}
            pos_op = FlagOp("[[$var1==$var2]]")
            neg_op = FlagOp("[[$var1!=$var2]]")
            self.assertTrue(pos_op(config))
            self.assertFalse(neg_op(config))
            
            for val2 in "bar {}":
                config = {"$var1":"foo", "$var2":val2}
                self.assertFalse(pos_op(config))            
                self.assertTrue(neg_op(config))

            config = {"$var1":"foo", "$var2":"bar"}
            set_op = FlagOp("[[$var1=$var2]]")
            self.assertTrue(set_op(config))
            self.assertEqual(config["$var1"], "bar")
            self.assertEqual(config["$var2"], "bar")

            config = {"$var1":"foo", "$var2":"bar"}
            unify_op = FlagOp("[[$var1?=$var2]]")
            self.assertFalse(unify_op(config))
            
            config = {"$var1":"foo", "$var2":"{}"}
            unify_op = FlagOp("[[$var1?=$var2]]")
            self.assertFalse(unify_op(config))

            config = {"$var1":"{}", "$var2":"foo"}
            unify_op = FlagOp("[[$var1?=$var2]]")
            self.assertTrue(unify_op(config))
            self.assertEqual(config["$var1"], "foo")
            self.assertEqual(config["$var2"], "foo")

            config = {"$var1":"foo", "$var2":"foo"}
            unify_op = FlagOp("[[$var1?=$var2]]")
            self.assertTrue(unify_op(config))
            self.assertEqual(config["$var1"], "foo")
            self.assertEqual(config["$var2"], "foo")
            
class TestFlagFilter(unittest.TestCase):
    def test_init(self):
        flags = {f"[[{var}{op}{val}]]"
                 for var in "$var1 $var2".split()
                 for op in "= == != ?=".split()
                 for val in "foo bar {} $var1 $var2".split()}
        ffilter = FlagFilter(flags.union({"a"}))
        self.assertEqual(set(ffilter.flags.keys()), flags)

    def test_strings_pos(self):
        flags = {f"[[{var}{op}{val}]]"
                 for var in "$var1 $var2".split()
                 for op in "= == != ?=".split()
                 for val in "foo bar {} $var1 $var2".split()}
        ffilter = FlagStringFilter(flags.union({"a","b"}))
        self.assertTrue(ffilter(""))
        self.assertTrue(ffilter("a"))
        self.assertTrue(ffilter("[[$var1=foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]]".split()))
        self.assertTrue(ffilter("a [[$var1={}]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] b".split()))
        self.assertTrue(ffilter("a [[$var1?=foo]] b".split()))
        self.assertTrue(ffilter("a [[$var1!=foo]] b".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1==foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1?=foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1!=bar]]".split()))
        self.assertTrue(ffilter("a [[$var1?=foo]] [[$var1==foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1={}]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1={}]] [[$var1=={}]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1={}]] [[$var1!=foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var1={}]] [[$var1?=bar]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var2=$var1]] [[$var2!={}]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var2=$var1]] [[$var2==foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var2?=$var1]] [[$var2!={}]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var2?=$var1]] [[$var2==foo]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var2=foo]] [[$var1==$var2]]".split()))
        self.assertTrue(ffilter("a [[$var1=foo]] [[$var2=bar]] [[$var1!=$var2]]".split()))
        
    def test_strings_neg(self):
        flags = {f"[[{var}{op}{val}]]"
                 for var in "$var1 $var2".split()
                 for op in "= == != ?=".split()
                 for val in "foo bar {} $var1 $var2".split()}
        ffilter = FlagStringFilter(flags.union({"a","b"}))
        self.assertFalse(ffilter("a [[$var1==foo]] b".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var1!=foo]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var1=={}]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var1?=bar]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var1?=$var2]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var2=bar]] [[$var1?=$var2]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var2=bar]] [[$var1==bar]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var2=bar]] [[$var1==$var2]]".split()))
        self.assertFalse(ffilter("a [[$var1=foo]] [[$var2=$var1]] [[$var1!=$var2]]".split()))

    def test_stream(self):
        flags = {f"[[{var}{op}{val}]]"
                 for var in "$var1 $var2".split()
                 for op in "= == != ?=".split()
                 for val in "foo bar {} $var1 $var2".split()}
        ffilter = FlagStreamFilter(flags.union({"a","b"}))
        for i,sym in enumerate("a [[$var1=foo]] [[$var2=$var1]] [[$var1!=$var2]]".split()):
            if i == 3:
                self.assertFalse(ffilter.check(sym))
            else:
                self.assertTrue(ffilter.check(sym))
        self.assertTrue(ffilter.has_failed)

        ffilter.reset()
        for sym in "a [[$var1=foo]] [[$var2=$var1]] [[$var1==$var2]]".split():
            ffilter.check(sym)
        self.assertFalse(ffilter.has_failed)
        
def time_flag_execution(trials):
    flags = {f"[[{var}{op}{val}]]"
             for var in "$var1 $var2".split()
             for op in "= == != ?=".split()
             for val in "foo bar {} $var1 $var2".split()}
    ffilter = FlagStringFilter(flags.union({"a","b"}))

    seq_a = ["a" for i in range(10)]
    seq_set = ["[[$var1=foo]]"]
    seq_check_pos = ["[[$var1==foo]]"]
    seq_check_neg = ["[[$var1!=foo]]"]
    seq_unify = ["[[$var1?=$var2]]"]

    for ex in [seq_a, seq_set, seq_check_pos, seq_check_neg, seq_unify]:
        start = time.time_ns()
        for i in range(trials):
            ffilter(seq_a)
        stop = time.time_ns()
        print(f"Processed {trials} {ex} examples in {(stop-start)/10**6}ms")
    
if __name__=="__main__":
    ### Test main ###
    print("Run performance tests")
    time_flag_execution(1000000)
    print()
    print("Run unit tests")
    unittest.main()
