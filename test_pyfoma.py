import unittest
from pyfoma import algorithms
from pyfoma.fst import FST

class TestFST(unittest.TestCase):

    def test_rewrite(self):
        f1 = FST.re("$^rewrite((ab):x / a b _ a)")
        self.assertEqual(set(f1.generate("abababa")), {"abxxa"})

        bigrule = FST.re('@'.join("$^rewrite([mnŋ]:%s / _ %s)" % (nas, stop)\
                                          for nas, stop in zip('mnŋ','ptk')))
        self.assertEqual(list(bigrule.apply('anpinkamto'))[0], "ampiŋkanto")

    def test_rewrite_weights(self):
        f1 = FST.re("$^rewrite(a:?(b<1.0>))")
        res = list(f1.analyze('bbb', weights = True))
        self.assertEqual(len(res), 8)
        self.assertEqual(sum(e[1] for e in res), 12.0)

    def test_rewrite_directed(self):
        f1 = FST.re("$^rewrite((ab|ba):x)")
        self.assertEqual(len(f1), 5)
        self.assertEqual(set(f1.generate("aba")), {"ax", "xa"})
        f2 = FST.re("$^rewrite((ab|ba):x, leftmost = True)")
        self.assertEqual(len(f2), 5)
        self.assertEqual(set(f2.generate("aba")), {"xa"})

        f3 = FST.re("$^rewrite((ab|ba|aba):x)")
        self.assertEqual(set(f3.generate("aba")), {"ax", "xa", "x"})

        f4 = FST.re("$^rewrite((ab|ba|aba):x, longest = True)")
        self.assertEqual(set(f4.generate("aba")), {"ax", "x"})

        f5 = FST.re("$^rewrite((ab|ba|aba):x, leftmost = True)")
        self.assertEqual(set(f5.generate("aba")), {"x", "xa"})

        f6 = FST.re("$^rewrite((ab|ba|aba):x, shortest = True)")
        self.assertEqual(set(f6.generate("aba")), {"xa", "ax"})

        f7 = FST.re("$^rewrite((ab|ba|aba):x, longest = True, leftmost = True)")
        self.assertEqual(set(f7.generate("aba")), {"x"})

        f8 = FST.re("$^rewrite((ab|ba|aba):x, shortest = True, leftmost = True)")
        self.assertEqual(set(f8.generate("aba")), {"xa"})

    def test_cross_product(self):
        f1 = FST.re("'':x")
        self.assertEqual(set(f1.generate("")), {"x"})
        f2 = FST.re("'':?x")
        self.assertEqual(set(f2.generate("")), {"x",""})

    def test_union(self):
        f1 = FST.re("a|''")
        f2 = FST.re("a?")
        self.assertEqual(list(f1.words()), list(f2.words()))

    def test_intersection(self):
        f1 = FST.re("a b* & a* b")
        f2 = FST.re("a b")
        self.assertEqual(list(f1.words()), list(f2.words()))

    def test_difference(self):
        f1 = FST.re("(a b) - (a b) (a b)+")
        f2 = FST.re("a b")
        self.assertEqual(list(f1.words()), list(f2.words()))

    def test_tokenizer(self):
        f1 = FST.regex(r"'[NOUN]' '[VERB]'")
        self.assertEqual(f1.alphabet, {"[NOUN]", "[VERB]"})
        f2 = FST.regex(r"'[NO\'UN]' '[VERB]'")
        self.assertEqual(f2.alphabet, {"[NO'UN]", "[VERB]"})

    def test_complement(self):
        """Test the complement operator / method"""
        f1 = FST.regex("~a")
        self.assertEqual(0, len(list(f1.generate("a"))))
        f1 = FST.regex("~(cat | dog)")
        self.assertEqual(0, len(list(f1.generate("cat"))))
        self.assertEqual(0, len(list(f1.generate("dog"))))
        self.assertEqual(1, len(list(f1.generate("octopus"))))
        # ~ binds tighter than concatenation
        f1 = FST.regex("~(cat | dog)s")
        self.assertEqual(0, len(list(f1.generate("cats"))))
        self.assertEqual(0, len(list(f1.generate("dogs"))))
        self.assertEqual(1, len(list(f1.generate("octopus"))))
        self.assertEqual(1, len(list(f1.generate("catdogs"))))
        # * binds tighter than ~
        f1 = FST.regex("~(cat | dog)*s")
        self.assertEqual(0, len(list(f1.generate("catdogs"))))
        self.assertEqual(0, len(list(f1.generate("catdogcats"))))
        # Verify that the new algorithm/method works too
        f1 = FST.regex("octopus")
        self.assertEqual(0, len(list(f1.generate("dog"))))
        self.assertEqual(1, len(list(f1.generate("octopus"))))
        # Non-mutating
        f2 = algorithms.complement(f1)
        self.assertEqual(1, len(list(f2.generate("dog"))))
        self.assertEqual(0, len(list(f2.generate("octopus"))))
        # Mutating
        f1.complement()
        self.assertEqual(1, len(list(f1.generate("dog"))))
        self.assertEqual(0, len(list(f1.generate("octopus"))))
        f1.complement()
        self.assertEqual(0, len(list(f1.generate("dog"))))
        self.assertEqual(1, len(list(f1.generate("octopus"))))

    def test_methods(self):
        """Verify that generated methods work as expected"""
        f1 = FST.regex("(cat):(dog)")
        f2 = FST.regex("(dog):(octopus)")
        f3 = algorithms.compose(f1, f2)
        f4 = algorithms.inverted(f1)
        self.assertEqual("dog", next(f1.generate("cat")))
        self.assertEqual("octopus", next(f2.generate("dog")))
        self.assertEqual("octopus", next(f3.generate("cat")))
        self.assertEqual("cat", next(f4.generate("dog")))
        # This is mutating (maybe not what you expect?)
        f1.compose(f2)
        self.assertEqual("octopus", next(f1.generate("cat")))
        # So is this!
        f1.invert()
        self.assertEqual("cat", next(f1.generate("octopus")))


class TestSymbols(unittest.TestCase):
    MULTICHAR_SYMBOLS = "u: ch ll x̌ʷ".split()

    def test_rlg_multichar(self):
        """Verify multi-character symbols in lexicons (lexc style)"""
        # Hopefully the third one is not an actual word for anyone
        words = ["hecho", "llama", "xu:x̌ʷ"]
        lex = FST.rlg({
            "Root": [(word, "#") for word in words]
        }, "Root", multichar_symbols=self.MULTICHAR_SYMBOLS)
        for sym in self.MULTICHAR_SYMBOLS:
            self.assertTrue(sym in lex.alphabet)
        for word in words:
            self.assertEqual(word, next(lex.generate(word)))

    def test_from_strings_multichar(self):
        """Verify multi-character symbols in from_strings"""
        # Hopefully the third one is not an actual word for anyone
        words = ["hecho", "llama", "xu:x̌ʷ"]
        lex = FST.from_strings(words, multichar_symbols=self.MULTICHAR_SYMBOLS)
        for sym in self.MULTICHAR_SYMBOLS:
            self.assertTrue(sym in lex.alphabet)
        for word in words:
            self.assertEqual(word, next(lex.generate(word)))

    def test_rewrite_multichar(self):
        """Verify multi-character symbols in rewrite rules"""
        # Yes, you can put forbidden characters in symbols now (just
        # because you can doesn't necessarily mean you should)
        rule = FST.regex("$^rewrite(x̌ʷ:x / u: _ #)",
                         multichar_symbols=self.MULTICHAR_SYMBOLS)
        self.assertEqual("xu:x", next(rule.generate("xu:x̌ʷ")))
        self.assertEqual("xux̌ʷ", next(rule.generate("xux̌ʷ")))


if __name__ == "__main__":
    unittest.main()
