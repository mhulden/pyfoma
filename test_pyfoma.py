import json
import re
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from pyfoma import algorithms
from pyfoma.fst import FST

class TestFST(unittest.TestCase):
    """Test basic FST functionality"""
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
    """Test multi-character symbol feature"""
    MULTICHAR_SYMBOLS = "u: ch ll x̌ʷ".split()

    def test_rlg(self):
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

    def test_from_strings(self):
        """Verify multi-character symbols in from_strings"""
        words = ["hecho", "llama", "xu:x̌ʷ"]
        lex = FST.from_strings(words, multichar_symbols=self.MULTICHAR_SYMBOLS)
        for sym in self.MULTICHAR_SYMBOLS:
            self.assertTrue(sym in lex.alphabet)
        for word in words:
            self.assertEqual(word, next(lex.generate(word)))

    def test_quotes(self):
        """Make sure already-quoted things stay quoted correctly and we
        can pathologically escape quotes everywhere"""
        words = ["'HACKEM'MUCHE", r"FOOBIE'BL\'ETCH'", r"'''"]
        lex = FST.from_strings(words, multichar_symbols=["CH", "BL"])
        self.assertTrue("HACKEM" in lex.alphabet)
        self.assertTrue("BL'ETCH" in lex.alphabet)
        self.assertTrue("'" in lex.alphabet)
        self.assertTrue("CH" in lex.alphabet)
        # Ensure that we don't introduce multichar_symbols inside
        # already quoted symbols
        self.assertTrue("BL" not in lex.alphabet)
        self.assertEqual("HACKEMMUCHE", next(lex.generate("HACKEMMUCHE")))
        self.assertEqual("FOOBIEBL'ETCH", next(lex.generate("FOOBIEBL'ETCH")))
        # Escaped quotes in explicit multichar symbol
        rule = FST.regex(r"$^rewrite('n\'t':(' 'not) / is _)")
        self.assertEqual("is not", next(rule.generate("isn't")))
        # Escaped quotes in multichar_symbols
        rule = FST.regex(r"$^rewrite(n't:(' 'not) / is _)",
                         multichar_symbols=["n't"])
        self.assertEqual("is not", next(rule.generate("isn't")))
        # Escaped quotes in explicit multichar symbol not destroyed by
        # multichar_symbols escaping
        rule = FST.regex(r"$^rewrite('n\'t':(' 'not) / is _)",
                         multichar_symbols=["n't"])
        self.assertEqual("is not", next(rule.generate("isn't")))

    def test_single_quotes(self):
        """Test that literal single quotes work everywhere, in a
        multitude of different ways."""
        lex = FST.from_strings(["foo'''bar"])
        self.assertTrue("'" in lex.alphabet)
        self.assertEqual("foo'bar", next(lex.generate("foo'bar")))
        # such escaping, so wow
        f1 = FST.regex("foo ''' bar")
        self.assertTrue("'" in f1.alphabet)
        self.assertEqual("foo'bar", next(f1.generate("foo'bar")))
        f2 = FST.regex(r"foo \' bar")
        self.assertTrue("'" in f2.alphabet)
        self.assertEqual("foo'bar", next(f2.generate("foo'bar")))
        f3 = FST.regex(r"foo '\'' bar")
        self.assertTrue("'" in f3.alphabet)
        self.assertEqual("foo'bar", next(f3.generate("foo'bar")))

    def test_rewrite(self):
        """Verify multi-character symbols in rewrite rules"""
        # Yes, you can put forbidden characters in symbols now (just
        # because you can doesn't necessarily mean you should)
        rule = FST.regex("$^rewrite(x̌ʷ:x / u: _ #)",
                         multichar_symbols=self.MULTICHAR_SYMBOLS)
        self.assertEqual("xu:x", next(rule.generate("xu:x̌ʷ")))
        self.assertEqual("xux̌ʷ", next(rule.generate("xux̌ʷ")))

    def test_longest_match(self):
        """Verify that longest multichar symbols are matched"""
        rule = FST.regex("$^rewrite(ABC:D)",
                         multichar_symbols=["A", "B", "C", "AB", "ABC"])
        self.assertEqual("ABD", next(rule.generate("ABABC")))


class TestUtil(unittest.TestCase):
    """Test utility functions."""
    fst = FST.regex(r"'[NO\'UN]' '[VERB]'<1> (cat):(dog)? 'ROTFLMAO🤣'")

    def verify_att_format(self, att, epsilon="@0@"):
        """Verify some expected states and such in AT&T FST"""
        self.assertIn("[NO'UN]\t[NO'UN]", att)
        self.assertIn("[VERB]\t[VERB]", att)
        self.assertIn(f"c\t{epsilon}", att)
        self.assertIn(f"a\t{epsilon}", att)
        self.assertIn(f"t\t{epsilon}", att)
        self.assertIn("c\td", att)
        self.assertIn("a\to", att)
        self.assertIn("t\tg", att)

    def test_to_str(self):
        """Test simple AT&T format conversion."""
        att = str(self.fst)
        self.verify_att_format(att)

    def test_to_att(self):
        """Test more complete AT&T format conversion."""
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir)
            f = self.fst
            # Verify expected path behaviour
            f.save_att(path / "test")
            self.assertTrue((path / "test").exists())
            self.assertTrue((path / "test.isyms").exists())
            self.assertTrue((path / "test.osyms").exists())
            f.save_att(path / "test.att", epsilon="<eps>")
            self.assertTrue((path / "test.att").exists())
            self.assertTrue((path / "test.isyms").exists())
            self.assertTrue((path / "test.osyms").exists())
            # Now verify contents
            with open(path / "test.att", "rt") as infh:
                att = infh.read()
                self.verify_att_format(att, epsilon="<eps>")
                # If you have OpenFST you can verify this with:
                #   fstcompile --isymbols=test.isyms \
                #      --osymbols=test.osyms --keep_isymbols \
                #      --keep_osymbols --keep_state_numbering \
                #      test.att  | fstprint
            # Check state symbols get output too
            f = FST.rlg({"Root": [("", "Sublex")],
                         "Sublex": [(("foo", "bar"), "#")]}, "Root")
            f.save_att(path / "test_st.fst", state_symbols=True)
            self.assertTrue((path / "test_st.fst").exists())
            self.assertTrue((path / "test_st.isyms").exists())
            self.assertTrue((path / "test_st.osyms").exists())
            self.assertTrue((path / "test_st.ssyms").exists())
            with open(path / "test_st.fst", "rt") as infh:
                att = infh.read()
                self.assertIn("Root\tSublex\t@0@\t@0@", att)
                self.assertIn("#\n", att)
                # If you have OpenFST you can verify this with:
                #   fstcompile --ssymbols=test_st.ssyms \
                #      --isymbols=test_st.isyms \
                #      --osymbols=test_st.osyms --keep_isymbols \
                #      --keep_osymbols --keep_state_numbering \
                #      test_st.fst | fstprint

    def test_todict(self):
        """Ensure that json is the same for equivalent FSTs"""
        rx = (r"""
        $^rewrite(s:(s | 'š')
            | c:(c | 'č')
            | \?:Ɂ
            | 7:Ɂ
            | ʔ:Ɂ)
        """)
        fst1 = FST.regex(rx)
        fst2 = FST.regex(rx)
        assert json.dumps(fst1.todict()) == json.dumps(fst2.todict())

    def test_to_js_on(self):
        # Has no maxlen anymore, downstream code should do that
        d = self.fst.todict()
        self.assertIn(0, d["transitions"])
        self.assertEqual(1, len(d["finals"]))
        js = self.fst.tojs()
        self.assertIn('"maxlen": 10', js)
        self.assertIn(""""0|[NO'UN]": [{""", js)
        # Ensure the JavaScript is the same FST
        js_dict = re.sub(r"^var \w+ = (.*);", r"\1", js)
        js_json = json.loads(js_dict)
        self.assertIn(str(next(iter(d["finals"]))), js_json["f"])
        self.assertEqual(d["alphabet"], js_json["s"])
        for src, out in d["transitions"].items():
            for label, arcs in out.items():
                syms = re.split(r"(?<!\\)\|", label)
                self.assertIn(f"{src}|{syms[0]}", js_json["t"])
                for arc in arcs:
                    self.assertIn({str(arc): syms[-1]},
                                  js_json["t"][f"{src}|{syms[0]}"])
        # Compare with output of foma2js.perl, sort of (it has various
        # issues, which have been fixed then dumped to JSON)
        with open("test_foma.json", "rt") as infh:
            foma_json = json.load(infh)
            self.assertEqual(js_json["s"].keys(), foma_json["s"].keys())
            # We are not bug-compatible (foma2js.perl has 9, but
            # that's not the actual length of 'ROTFLMAO🤣' in UTF-16)
            self.assertEqual(10, js_json["maxlen"])
            Q = deque([(0, 0)])
            while Q:
                src, jsrc = Q.popleft()
                if src in d["finals"]:
                    self.assertIn(jsrc, foma_json["f"])
                if src not in d["transitions"]:
                    continue
                for label in d["transitions"][src]:
                    syms = re.split(r"(?<!\\)\|", label)
                    for arc in d["transitions"][src][label]:
                        for jarc in foma_json["t"][f"{jsrc}|{syms[0]}"]:
                            for jdst, jsym in jarc.items():
                                if jsym != syms[-1]:
                                    continue
                                Q.append((arc, jdst)) 

if __name__ == "__main__":
    unittest.main()
