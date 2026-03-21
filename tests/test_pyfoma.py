import json
import pathlib
import re
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from pyfoma.fst import FST


class TestFST(unittest.TestCase):
    """Test basic FST functionality"""

    def test_rewrite(self):
        f1 = FST.re("$^rewrite((ab):x / a b _ a)")
        self.assertEqual(set(f1.generate("abababa")), {"abxxa"})

        bigrule = FST.re(
            "@".join(
                "$^rewrite([mnŋ]:%s / _ %s)" % (nas, stop)
                for nas, stop in zip("mnŋ", "ptk")
            )
        )
        self.assertEqual(list(bigrule.apply("anpinkamto"))[0], "ampiŋkanto")

    def test_rewrite_outputcontexts(self):
        f1 = FST.re("$^rewrite((ab):x / ab _ a)")
        f2 = FST.re("$^rewrite((ab):x / ab _ a, outputcontexts = True)")
        self.assertEqual(set(f1.generate("abababa")), {"abxxa"})
        self.assertEqual(set(f2.generate("abababa")), {"abxaba", "ababxa"})

    def test_rewrite_weights(self):
        f1 = FST.re("$^rewrite(a:?(b<1.0>))")
        res = list(f1.analyze("bbb", weights=True))
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
        self.assertEqual(set(f2.generate("")), {"x", ""})

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

    def test_shuffle(self):
        f1 = FST.re("$^shuffle(cat,if)")
        expected = {
            "catif", "caitf", "caift", "ciatf", "ciaft",
            "cifat", "icatf", "icaft", "icfat", "ifcat",
        }
        generated = {''.join(lbl[0] for lbl in seq) for _, seq in f1.words()}
        self.assertEqual(generated, expected)

        f2 = FST.re("$^shuffle(a<1.0>,b<2.0>)")
        weighted = {''.join(lbl[0] for lbl in seq): cost for cost, seq in f2.words()}
        self.assertEqual(weighted, {"ab": 3.0, "ba": 3.0})

    def test_tokenizer(self):
        f1 = FST.regex(r"'[NOUN]' '[VERB]'")
        self.assertEqual(f1.alphabet, {"[NOUN]", "[VERB]"})
        f2 = FST.regex(r"'[NO\'UN]' '[VERB]'")
        self.assertEqual(f2.alphabet, {"[NO'UN]", "[VERB]"})
        f4 = FST.regex(r"'\\'")
        self.assertEqual(f4.alphabet, {"\\"})
        self.assertEqual(set(f4.generate("\\")), {"\\"})
        f5 = FST.regex(r"'a\\b'")
        self.assertEqual(f5.alphabet, {r"a\b"})
        self.assertEqual(set(f5.generate(r"a\b")), {r"a\b"})
        with self.assertRaises(SyntaxError):
            FST.regex("'ab")
        with self.assertRaises(SyntaxError):
            FST.regex("a'")
        f3 = FST.regex("'''")
        self.assertEqual(f3.alphabet, {"'"})

    def test_literal_period_symbol(self):
        f_lit_esc = FST.regex(r"\.")
        f_lit_q = FST.regex("'.'")
        self.assertEqual(set(f_lit_esc.generate(".")), {r"\."})
        self.assertEqual(set(f_lit_q.generate(".")), {r"\."})
        self.assertEqual(list(f_lit_esc.generate("a")), [])

        f_any = FST.regex(".")
        self.assertEqual(set(f_any.generate("x")), {"x"})
        self.assertEqual(set(f_any.generate(".")), {r"\."})

    def test_literal_period_tokenized_input_parity(self):
        f_lit = FST.regex(r"\.")
        self.assertEqual(set(f_lit.generate(".")), {r"\."})
        self.assertEqual(
            list(f_lit.generate(["."], tokenize_outputs=True)),
            [[r"\."]],
        )

    def test_rewrite_literal_period_vs_wildcard(self):
        r_lit = FST.re(r"$^rewrite(\.:a / _ . #)")
        r_any = FST.re(r"$^rewrite(.:a / _ . #)")

        self.assertEqual(set(r_lit.generate("abababa")), {"abababa"})
        self.assertEqual(set(r_lit.generate("ab.c")), {"abac"})

        self.assertEqual(set(r_any.generate("abxc")), {"abac"})
        self.assertEqual(set(r_any.generate("ab.c")), {"abac"})

    def test_wildcard_cross_product_semantics(self):
        f_any = FST.re(".:.")
        self.assertEqual(set(f_any.generate("x")), {"x", "."})

        f_nonid = FST.re(".:. - .")
        self.assertEqual(set(f_nonid.generate("x")), {"."})

        defs = {
            "ab": FST.re("a|b"),
            "nonid": f_nonid,
        }
        sub = FST.re("$ab @ $nonid", defs)
        self.assertEqual(set(sub.generate("a")), {"b", "."})
        self.assertEqual(set(sub.generate("b")), {"a", "."})

    def test_foma_wildcard_roundtrip(self):
        f_id = FST.re(".")
        f_nonid = FST.re(".:. - .")

        rt_id = FST.from_fomastring(f_id.to_fomastring())
        rt_nonid = FST.from_fomastring(f_nonid.to_fomastring())

        self.assertEqual(set(rt_id.generate("x")), {"x"})
        self.assertEqual(set(rt_nonid.generate("x")), {"."})

    def test_spell_corrector_unknown_symbols(self):
        fsts = {}
        fsts["changeone"] = FST.re(".* ( (.:. - .) | .:'' | '':. ) .*")
        fsts["lexicon"] = FST.re("cat|dog|mouse|rat")
        fsts["correct"] = FST.re("$lexicon @ $changeone", fsts)

        self.assertEqual(set(fsts["correct"].analyze("house")), {"mouse"})
        self.assertEqual(set(fsts["correct"].analyze("rouse")), {"mouse"})
        self.assertEqual(set(fsts["correct"].analyze("fat")), {"cat", "rat"})
        self.assertEqual(set(fsts["correct"].analyze("oat")), {"cat", "rat"})

    def test_range_quantifiers(self):
        f0 = FST.regex("a{0}")
        self.assertEqual(set(f0.generate("")), {""})
        self.assertEqual(list(f0.generate("a")), [])

        f1 = FST.regex("a{2,2}")
        self.assertEqual(set(f1.generate("aa")), {"aa"})
        self.assertEqual(list(f1.generate("a")), [])

        f2 = FST.regex("a{,0}")
        self.assertEqual(set(f2.generate("")), {""})
        self.assertEqual(list(f2.generate("a")), [])

        f3 = FST.regex("a{0,3}")
        for n in range(0, 4):
            s = "a" * n
            self.assertEqual(set(f3.generate(s)), {s})
        self.assertEqual(list(f3.generate("aaaa")), [])

        f4 = FST.regex("a{2,10}")
        self.assertEqual(set(f4.generate("aa")), {"aa"})
        self.assertEqual(set(f4.generate("aaaaaaaaaa")), {"aaaaaaaaaa"})
        self.assertEqual(list(f4.generate("a")), [])
        self.assertEqual(list(f4.generate("aaaaaaaaaaa")), [])

        with self.assertRaises(SyntaxError):
            FST.regex("a{10,2}")
        with self.assertRaises(SyntaxError):
            FST.regex("a{12,3}")

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
        f2 = f1.complement()
        self.assertEqual(1, len(list(f2.generate("dog"))))
        self.assertEqual(0, len(list(f2.generate("octopus"))))

    def test_methods(self):
        """Verify that generated methods work as expected"""
        f1 = FST.regex("(cat):(dog)")
        f2 = FST.regex("(dog):(octopus)")
        f3 = f1.compose(f2)
        f4 = f1.invert()
        self.assertEqual("dog", next(f1.generate("cat")))
        self.assertEqual("octopus", next(f2.generate("dog")))
        self.assertEqual("octopus", next(f3.generate("cat")))
        self.assertEqual("cat", next(f4.generate("dog")))

    def test_compose_after_mutating_cached_operand(self):
        f1 = FST.regex("a|b")
        f2 = FST.regex("(a):(x)")
        final = next(iter(f2.finalstates))

        # Prime cached transition index, then mutate.
        _ = f2.initialstate.transitions_by_input
        f2.initialstate.add_transition(final, ("b", "y"), 0.0)
        f2.alphabet |= {"b", "y"}

        composed = f1.compose(f2)
        self.assertEqual(set(composed.generate("a")), {"x"})
        self.assertEqual(set(composed.generate("b")), {"y"})

    def test_constructor_defaults_not_shared(self):
        f1 = FST()
        f1.alphabet.add("x")
        f2 = FST()
        self.assertEqual(f2.alphabet, set())

    def test_hash_canonical_structure(self):
        f1 = FST.re("a|''")
        f2 = FST.re("a?")
        self.assertEqual(f1.hash(), f2.hash())

        rgx1 = "$^restrict(a / b _ c)"
        rgx2 = "~((~(.* b) a .*) | (.* a ~(c .*)))"
        self.assertEqual(FST.re(rgx1).hash(), FST.re(rgx2).hash())

        t1 = FST.re("a:x | b:y | '':z")
        t2 = FST.fromdict(json.loads(json.dumps(t1.todict())))
        self.assertEqual(t1.hash(), t2.hash())

        t3 = FST.re("a:x | b:z | '':z")
        self.assertNotEqual(t1.hash(), t3.hash())

        w1 = FST.re("a<1.0>")
        w2 = FST.re("a<2.0>")
        self.assertNotEqual(w1.hash(), w2.hash())

    def test_is_identity(self):
        self.assertTrue(FST.re("a:a").is_identity())
        self.assertTrue(FST.re("a:'' '':a").is_identity())
        self.assertTrue(FST.re("a:'' b:'' '':a '':b").is_identity())
        self.assertTrue(FST.re("(a:'' '':a | b:'' '':b)*").is_identity())
        self.assertTrue(FST.re(".").is_identity())

        # n-tape: only first and last tapes are compared for identity.
        self.assertTrue(FST.re("a:x:a | b:y:b").is_identity())
        self.assertFalse(FST.re("a:x:b").is_identity())

        self.assertFalse(FST.re("a:b").is_identity())
        self.assertFalse(FST.re("a:''").is_identity())  # final with residual debt
        self.assertFalse(FST.re("a:'' b:'' '':a '':c").is_identity())
        self.assertFalse(FST.re("a:'' . '':a").is_identity())  # one-tape wildcard with debt

        # >1 tape wildcard use always fails.
        self.assertFalse(FST.re("a:.").is_identity())
        self.assertFalse(FST.re(".:a").is_identity())
        self.assertFalse(FST.re(".:.").is_identity())
        self.assertFalse(FST.re("a:.:a").is_identity())
        self.assertFalse(FST.re(".:x:y:.").is_identity())

    def test_nonidentity_domain(self):
        id_dom = FST.re("a:a | b:b").nonidentity_domain()
        for word in ["", "a", "b", "ab", "ba"]:
            self.assertFalse(bool(list(id_dom.analyze(word))))

        mixed = FST.re("a:a | a:b").nonidentity_domain()
        self.assertTrue(bool(list(mixed.analyze("a"))))
        self.assertFalse(bool(list(mixed.analyze(""))))
        self.assertFalse(bool(list(mixed.analyze("aa"))))

        unsync = FST.re("a:'' '':a | a:'' '':b").nonidentity_domain()
        self.assertTrue(bool(list(unsync.analyze("a"))))
        self.assertFalse(bool(list(unsync.analyze(""))))

        wild = FST.re("a:.").nonidentity_domain()
        self.assertTrue(bool(list(wild.analyze("a"))))

    def test_is_functional(self):
        self.assertTrue(FST.re("a|b").is_functional())
        self.assertTrue(FST.re("a:'' '':b | a:b").is_functional())
        self.assertFalse(FST.re("a:x | a:y").is_functional())
        self.assertFalse(
            FST.re("$^rewrite(a [^a]* a [^a]* b:(b|c) a [^b]* b)").is_functional()
        )

        ambig_but_functional = FST.re("(a:x a:x)* | a:y (a:y a:y)*")
        self.assertTrue(ambig_but_functional.is_functional())
        for word in ["", "a", "aa", "aaa", "aaaa"]:
            self.assertLessEqual(len(set(ambig_but_functional.generate(word))), 1)

    def test_is_unambiguous(self):
        self.assertTrue(FST.re("a:b").is_unambiguous())
        self.assertFalse(FST.re("a:b | a:'' '':b").is_unambiguous())
        self.assertTrue(FST.re("(a:b)*").is_unambiguous())
        self.assertFalse(FST.re("(a:b)* | (a:'' '':b)*").is_unambiguous())

    def test_ambiguous_domain(self):
        amb = FST.re("a:b | a:'' '':b").ambiguous_domain()
        self.assertTrue(bool(list(amb.analyze("a"))))
        self.assertFalse(bool(list(amb.analyze(""))))
        self.assertFalse(bool(list(amb.analyze("aa"))))

        unamb = FST.re("a:b").ambiguous_domain()
        for word in ["", "a", "aa", "b"]:
            self.assertFalse(bool(list(unamb.analyze(word))))

        star_amb = FST.re("(a:b)* | (a:'' '':b)*").ambiguous_domain()
        self.assertTrue(bool(list(star_amb.analyze("a"))))
        self.assertTrue(bool(list(star_amb.analyze("aa"))))
        self.assertFalse(bool(list(star_amb.analyze("b"))))

    def test_to_regex_roundtrip_acceptor(self):
        probes = [
            "",
            "a",
            "ab",
            "cd",
            "cdef",
            "abcdef",
            "zz",
            "xy",
        ]
        for expr in [
            "(ab|c)d?",
            "[a-z]* & $^restrict(a b / c d _ e f)",
            "$^restrict(ab / cd _ ef)",
        ]:
            fst = FST.re(expr)
            regex = fst.to_regex(n=3, mode="dm", seed=7)
            rebuilt = FST.re(regex)
            for word in probes:
                self.assertEqual(
                    bool(list(fst.analyze(word))),
                    bool(list(rebuilt.analyze(word))),
                )
            self.assertEqual(fst.hash(), rebuilt.hash())

    def test_to_regex_roundtrip_two_tape(self):
        probes = [
            "",
            "a",
            "b",
            "q",
            "cad",
            "cadf",
            "cccddeff",
            "hgd",
            "xyz",
            "cad",
            "ca",
        ]
        for expr in [
            "a:x | b:y | '':z | q:''",
            "[a-h]* @ $^rewrite(a:b / c _ d)",
            "$^rewrite(a:b / c _ d)",
            "$^rewrite(a:. / c_ #)",
        ]:
            fst = FST.re(expr)
            regex = fst.to_regex(n=5, mode="dm", seed=11)
            rebuilt = FST.re(regex)
            for word in probes:
                self.assertEqual(
                    set(fst.generate(word)),
                    set(rebuilt.generate(word)),
                )
            self.assertEqual(fst.hash(), rebuilt.hash())

    def test_to_regex_roundtrip_three_tape(self):
        fst = FST.re("a:b:c | d:e:f | 'x y':'u v':'w'")
        regex = fst.to_regex(n=4, mode="dm", seed=13)
        rebuilt = FST.re(regex)
        self.assertEqual(fst.arity(), 3)
        self.assertEqual(rebuilt.arity(), 3)
        self.assertEqual(
            {tuple(seq) for _, seq in fst.words()},
            {tuple(seq) for _, seq in rebuilt.words()},
        )
        self.assertEqual(fst.hash(), rebuilt.hash())

    def test_to_regex_rejects_multi_wildcard_labels(self):
        fst = FST.re(".:.")
        with self.assertRaises(ValueError):
            fst.to_regex()

    def test_to_regex_local_simplify_optional_plus(self):
        fst_opt = FST.re("a|''")
        regex_unsimplified = fst_opt.to_regex(simplify=False)
        regex_simplified = fst_opt.to_regex(simplify=True)
        self.assertNotIn("?", regex_unsimplified)
        self.assertIn("?", regex_simplified)
        rebuilt = FST.re(regex_simplified)
        for word in ["", "a", "aa", "b"]:
            self.assertEqual(bool(list(fst_opt.analyze(word))), bool(list(rebuilt.analyze(word))))

        fst_plus = FST.re("a a*")
        regex_plus = fst_plus.to_regex(simplify=True)
        self.assertIn("+", regex_plus)
        rebuilt_plus = FST.re(regex_plus)
        for word in ["", "a", "aa", "aaa", "b"]:
            self.assertEqual(bool(list(fst_plus.analyze(word))), bool(list(rebuilt_plus.analyze(word))))

        fst_cls = FST.re("(a|b)*")
        regex_cls = fst_cls.to_regex(simplify=True)
        self.assertIn("[ab]*", regex_cls)
        rebuilt_cls = FST.re(regex_cls)
        for word in ["", "a", "b", "ab", "ba", "abba", "c"]:
            self.assertEqual(bool(list(fst_cls.analyze(word))), bool(list(rebuilt_cls.analyze(word))))

        fst_opt_cls = FST.re("[abc]?")
        regex_opt_cls = fst_opt_cls.to_regex(simplify=True)
        self.assertIn("[abc]?", regex_opt_cls)
        rebuilt_opt_cls = FST.re(regex_opt_cls)
        for word in ["", "a", "b", "c", "ab", "d"]:
            self.assertEqual(bool(list(fst_opt_cls.analyze(word))), bool(list(rebuilt_opt_cls.analyze(word))))

        fst_mix = FST.re("[abcd]? x* y+ [efg]+")
        regex_mix = fst_mix.to_regex(simplify=True)
        self.assertIn("[abcdx]", regex_mix)
        rebuilt_mix = FST.re(regex_mix)
        for word in ["yef", "xyef", "ayyefg", "dxxxygg", "", "ef"]:
            self.assertEqual(bool(list(fst_mix.analyze(word))), bool(list(rebuilt_mix.analyze(word))))

        fst_compl = FST.re("[abcd]* & ~(.* ab .*)")
        regex_compl = fst_compl.to_regex(simplify=True)
        self.assertNotIn("a+?", regex_compl)
        self.assertIn("a*", regex_compl)
        rebuilt_compl = FST.re(regex_compl)
        for word in ["", "a", "aa", "aba", "abb", "acda", "dddc", "abca"]:
            self.assertEqual(bool(list(fst_compl.analyze(word))), bool(list(rebuilt_compl.analyze(word))))

        fst_diff = FST.re(".* - (.* (a b | b a) .*)")
        regex_diff = fst_diff.to_regex(simplify=True)
        self.assertNotIn("a+ )?", regex_diff)
        self.assertNotIn("b+ )?", regex_diff)
        self.assertIn("a*", regex_diff)
        self.assertIn("b*", regex_diff)
        rebuilt_diff = FST.re(regex_diff)
        for word in ["", "a", "b", "aa", "bb", "ab", "ba", "aba", "bbb", "aab", "baa"]:
            self.assertEqual(bool(list(fst_diff.analyze(word))), bool(list(rebuilt_diff.analyze(word))))

    def test_to_regex_simplify_level_validation(self):
        fst = FST.re("a")
        with self.assertRaises(ValueError):
            fst.to_regex(simplify_level="aggressive")


class TestSymbols(unittest.TestCase):
    """Test multi-character symbol feature"""

    MULTICHAR_SYMBOLS = "u: ch ll x̌ʷ".split()

    def test_rlg(self):
        """Verify multi-character symbols in lexicons (lexc style)"""
        # Hopefully the third one is not an actual word for anyone
        words = ["hecho", "llama", "xu:x̌ʷ"]
        lex = FST.rlg(
            {"Root": [(word, "#") for word in words]},
            "Root",
            multichar_symbols=self.MULTICHAR_SYMBOLS,
        )
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
        rule = FST.regex(r"$^rewrite(n't:(' 'not) / is _)", multichar_symbols=["n't"])
        self.assertEqual("is not", next(rule.generate("isn't")))
        # Escaped quotes in explicit multichar symbol not destroyed by
        # multichar_symbols escaping
        rule = FST.regex(
            r"$^rewrite('n\'t':(' 'not) / is _)", multichar_symbols=["n't"]
        )
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
        rule = FST.regex(
            "$^rewrite(x̌ʷ:x / u: _ #)", multichar_symbols=self.MULTICHAR_SYMBOLS
        )
        self.assertEqual("xu:x", next(rule.generate("xu:x̌ʷ")))
        self.assertEqual("xux̌ʷ", next(rule.generate("xux̌ʷ")))

    def test_longest_match(self):
        """Verify that longest multichar symbols are matched"""
        rule = FST.regex(
            "$^rewrite(ABC:D)", multichar_symbols=["A", "B", "C", "AB", "ABC"]
        )
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
            f = FST.rlg(
                {"Root": [("", "Sublex")], "Sublex": [(("foo", "bar"), "#")]}, "Root"
            )
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

    def test_to_att_no_duplicate_lines(self):
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir)
            f = FST.re("ab|cb")
            f.save_att(path / "dupe.att")
            with open(path / "dupe.att", "rt") as infh:
                att = [line for line in infh.read().strip().splitlines() if line]
            b_arcs = [line for line in att if "\tb\tb" in line and line.count("\t") >= 3]
            final_lines = [line for line in att if "\t" not in line]
            self.assertEqual(len(b_arcs), 1)
            self.assertEqual(len(final_lines), 1)

    def test_todict(self):
        """Ensure that json is the same for equivalent FSTs"""
        rx = r"""
        $^rewrite(s:(s | 'š')
            | c:(c | 'č')
            | \?:Ɂ
            | 7:Ɂ
            | ʔ:Ɂ)
        """
        fst1 = FST.regex(rx)
        fst2 = FST.regex(rx)
        assert json.dumps(fst1.todict()) == json.dumps(fst2.todict())

    def test_fromdict(self):
        """Ensure that we can reload from dictionary / json format"""
        rx = r"""
        $^rewrite(s:(s | 'š')
            | c:(c | 'č')
            | \?:Ɂ
            | 7:Ɂ
            | ʔ:Ɂ)
        """
        fst1 = FST.regex(rx)
        fst2 = FST.fromdict(fst1.todict())
        assert json.dumps(fst1.todict()) == json.dumps(fst2.todict())

    def test_fromdict_preserves_pipe_symbols(self):
        fst1 = FST.regex("'a|b'")
        fst2 = FST.fromdict(json.loads(json.dumps(fst1.todict())))
        self.assertEqual(set(fst1.generate("a|b")), {"a|b"})
        self.assertEqual(set(fst2.generate("a|b")), {"a|b"})

    def test_fromdict_preserves_final_weights(self):
        fst1 = FST.regex("''<2.5>")
        fst2 = FST.fromdict(json.loads(json.dumps(fst1.todict())))
        self.assertEqual(list(fst1.generate("", weights=True)), [("", 2.5)])
        self.assertEqual(list(fst2.generate("", weights=True)), [("", 2.5)])

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
                    arc_dest = arc[0] if isinstance(arc, (tuple, list)) else arc
                    self.assertIn(
                        {str(arc_dest): syms[-1]}, js_json["t"][f"{src}|{syms[0]}"]
                    )
        # Compare with output of foma2js.perl, sort of (it has various
        # issues, which have been fixed then dumped to JSON)
        with open(pathlib.Path(__file__).parent / "test_foma.json", "rt") as infh:
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
                        arc_dest = arc[0] if isinstance(arc, (tuple, list)) else arc
                        for jarc in foma_json["t"][f"{jsrc}|{syms[0]}"]:
                            for jdst, jsym in jarc.items():
                                if jsym != syms[-1]:
                                    continue
                                Q.append((arc_dest, jdst))

    def test_to_js_empty_alphabet(self):
        fst = FST.regex("''")
        js = fst.tojs()
        js_dict = re.sub(r"^var \w+ = (.*);", r"\1", js)
        js_json = json.loads(js_dict)
        self.assertEqual(js_json["maxlen"], 0)
        self.assertEqual(js_json["s"], {})
        self.assertEqual(js_json["t"], {})

    def test_fromdict_preserves_transition_weights(self):
        fst1 = FST.regex("a<2.0>")
        fst2 = FST.fromdict(json.loads(json.dumps(fst1.todict())))
        self.assertEqual(list(fst1.generate("a", weights=True)), [("a", 2.0)])
        self.assertEqual(list(fst2.generate("a", weights=True)), [("a", 2.0)])

    def test_to_js_weighted_transitions(self):
        fst = FST.regex("a<2.0>")
        js = fst.tojs()
        js_dict = re.sub(r"^var \w+ = (.*);", r"\1", js)
        js_json = json.loads(js_dict)
        self.assertIn("0|a", js_json["t"])
        self.assertIn({"1": "a"}, js_json["t"]["0|a"])


if __name__ == "__main__":
    unittest.main()
