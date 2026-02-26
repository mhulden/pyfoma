"""Tests for FST construction: __init__, character_ranges, regex/re, from_strings, rlg."""

import unittest
from pyfoma.fst import FST


class TestInit(unittest.TestCase):
    """Test FST.__init__ with various argument combinations."""

    def test_empty_fst(self):
        """No arguments: one state, no final states, empty alphabet."""
        f = FST()
        self.assertEqual(len(f.states), 1)
        self.assertEqual(len(f.finalstates), 0)
        self.assertEqual(f.alphabet, set())

    def test_epsilon_fst(self):
        """label=('',) creates a single-state FST that accepts the empty string."""
        f = FST(label=('',))
        self.assertEqual(len(f.states), 1)
        self.assertIn(f.initialstate, f.finalstates)
        self.assertEqual(list(f.generate("")), [""])
        self.assertEqual(list(f.generate("a")), [])

    def test_single_symbol_fst(self):
        """label=('a',) creates a two-state FST accepting 'a'."""
        f = FST(label=('a',))
        self.assertEqual(len(f.states), 2)
        self.assertEqual(f.alphabet, {'a'})
        self.assertEqual(list(f.generate("a")), ["a"])
        self.assertEqual(list(f.generate("b")), [])

    def test_weighted_final_state(self):
        """weight is applied to the final state."""
        f = FST(label=('a',), weight=3.0)
        results = list(f.generate("a", weights=True))
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][1], 3.0)

    def test_explicit_alphabet(self):
        """An explicitly provided alphabet is stored as-is."""
        f = FST(alphabet={'a', 'b', 'c'})
        self.assertEqual(f.alphabet, {'a', 'b', 'c'})


class TestCharacterRanges(unittest.TestCase):
    """Test FST.character_ranges."""

    def test_basic_range(self):
        """Accepts any symbol whose code point is in the range."""
        f = FST.character_ranges([(ord('a'), ord('z'))])
        self.assertEqual(list(f.generate('a')), ['a'])
        self.assertEqual(list(f.generate('z')), ['z'])
        self.assertEqual(list(f.generate('m')), ['m'])
        self.assertEqual(list(f.generate('A')), [])

    def test_multiple_ranges(self):
        """Accepts symbols from any of the supplied ranges."""
        f = FST.character_ranges([(ord('a'), ord('c')), (ord('x'), ord('z'))])
        for ch in 'abcxyz':
            self.assertEqual(list(f.generate(ch)), [ch])
        self.assertEqual(list(f.generate('d')), [])

    def test_complement(self):
        """Complemented range uses '.' and rejects symbols inside the range."""
        f = FST.character_ranges([(ord('a'), ord('z'))], complement=True)
        self.assertIn('.', f.alphabet)
        self.assertEqual(list(f.generate('A')), ['A'])
        self.assertEqual(list(f.generate('a')), [])


class TestRegex(unittest.TestCase):
    """Test FST.regex / FST.re compilation."""

    def test_basic_sequence(self):
        f = FST.re("a b c")
        self.assertEqual(list(f.generate("abc")), ["abc"])
        self.assertEqual(list(f.generate("ab")), [])

    def test_re_is_alias_for_regex(self):
        f1 = FST.re("a b c")
        f2 = FST.regex("a b c")
        self.assertEqual(list(f1.generate("abc")), list(f2.generate("abc")))

    def test_quoted_multichar_symbols(self):
        """Single-quoted strings become multi-character symbols in the alphabet."""
        f1 = FST.regex(r"'[NOUN]' '[VERB]'")
        self.assertEqual(f1.alphabet, {"[NOUN]", "[VERB]"})
        f2 = FST.regex(r"'[NO\'UN]' '[VERB]'")
        self.assertEqual(f2.alphabet, {"[NO'UN]", "[VERB]"})

    def test_quotes_in_symbols(self):
        """Escaped quotes inside quoted multi-character symbols are preserved."""
        words = ["'HACKEM'MUCHE", r"FOOBIE'BL\'ETCH'", r"'''"]
        lex = FST.from_strings(words, multichar_symbols=["CH", "BL"])
        self.assertIn("HACKEM", lex.alphabet)
        self.assertIn("BL'ETCH", lex.alphabet)
        self.assertIn("'", lex.alphabet)
        self.assertIn("CH", lex.alphabet)
        # multichar_symbols must not split already-quoted symbols
        self.assertNotIn("BL", lex.alphabet)
        self.assertEqual("HACKEMMUCHE", next(lex.generate("HACKEMMUCHE")))
        self.assertEqual("FOOBIEBL'ETCH", next(lex.generate("FOOBIEBL'ETCH")))
        # Escaped quotes in explicit multichar symbol
        rule = FST.regex(r"$^rewrite('n\'t':(' 'not) / is _)")
        self.assertEqual("is not", next(rule.generate("isn't")))
        # Escaped quotes in multichar_symbols
        rule = FST.regex(r"$^rewrite(n't:(' 'not) / is _)", multichar_symbols=["n't"])
        self.assertEqual("is not", next(rule.generate("isn't")))
        # Escaped quotes in explicit symbol not destroyed by multichar_symbols
        rule = FST.regex(
            r"$^rewrite('n\'t':(' 'not) / is _)", multichar_symbols=["n't"]
        )
        self.assertEqual("is not", next(rule.generate("isn't")))

    def test_single_quotes(self):
        """Literal single quotes work in multiple syntactic contexts."""
        lex = FST.from_strings(["foo'''bar"])
        self.assertIn("'", lex.alphabet)
        self.assertEqual("foo'bar", next(lex.generate("foo'bar")))
        f1 = FST.regex("foo ''' bar")
        self.assertIn("'", f1.alphabet)
        self.assertEqual("foo'bar", next(f1.generate("foo'bar")))
        f2 = FST.regex(r"foo \' bar")
        self.assertIn("'", f2.alphabet)
        self.assertEqual("foo'bar", next(f2.generate("foo'bar")))
        f3 = FST.regex(r"foo '\'' bar")
        self.assertIn("'", f3.alphabet)
        self.assertEqual("foo'bar", next(f3.generate("foo'bar")))


class TestFromStrings(unittest.TestCase):
    """Test FST.from_strings."""

    MULTICHAR_SYMBOLS = "u: ch ll x̌ʷ".split()

    def test_basic(self):
        """from_strings creates an acceptor for the given words."""
        words = ["cat", "dog", "fish"]
        f = FST.from_strings(words)
        for w in words:
            self.assertEqual(next(f.generate(w)), w)
        self.assertEqual(list(f.generate("bird")), [])

    def test_multichar_symbols(self):
        """from_strings handles multi-character symbols correctly."""
        words = ["hecho", "llama", "xu:x\u030cʷ"]
        lex = FST.from_strings(words, multichar_symbols=self.MULTICHAR_SYMBOLS)
        for sym in self.MULTICHAR_SYMBOLS:
            self.assertIn(sym, lex.alphabet)
        for word in words:
            self.assertEqual(word, next(lex.generate(word)))


class TestRLG(unittest.TestCase):
    """Test FST.rlg (right-linear grammar / lexc-style lexicon)."""

    MULTICHAR_SYMBOLS = "u: ch ll x̌ʷ".split()

    def test_basic(self):
        """rlg compiles a simple lexicon."""
        lex = FST.rlg({"Root": [("cat", "#"), ("dog", "#")]}, "Root")
        self.assertEqual("cat", next(lex.generate("cat")))
        self.assertEqual("dog", next(lex.generate("dog")))
        self.assertEqual([], list(lex.generate("fish")))

    def test_multichar_symbols(self):
        """rlg handles multi-character symbols correctly."""
        words = ["hecho", "llama", "xu:x\u030cʷ"]
        lex = FST.rlg(
            {"Root": [(word, "#") for word in words]},
            "Root",
            multichar_symbols=self.MULTICHAR_SYMBOLS,
        )
        for sym in self.MULTICHAR_SYMBOLS:
            self.assertIn(sym, lex.alphabet)
        for word in words:
            self.assertEqual(word, next(lex.generate(word)))

    def test_multistate_grammar(self):
        """rlg handles multi-state grammars (sublexicons)."""
        grammar = {
            "Root": [("", "Sublex")],
            "Sublex": [(("foo", "bar"), "#")],
        }
        f = FST.rlg(grammar, "Root")
        self.assertEqual("bar", next(f.generate("foo")))


if __name__ == "__main__":
    unittest.main()
