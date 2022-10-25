import unittest
from pyfoma.fst import FST
# import sys
#
# sys.path.append('../src')

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


if __name__ == "__main__":
    unittest.main()
