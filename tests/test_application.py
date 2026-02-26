"""Tests for FST application: generate/analyze/apply, word enumeration,
FST properties/queries, and structural utilities."""

import unittest

from pyfoma.fst import FST


# ---------------------------------------------------------------------------
# Apply: generate, analyze, apply, tokenize_against_alphabet
# ---------------------------------------------------------------------------

class TestApply(unittest.TestCase):
    """Test generate, analyze, apply, and tokenize_against_alphabet."""

    def test_generate_acceptor(self):
        f = FST.re("a b c")
        self.assertEqual(list(f.generate("abc")), ["abc"])
        self.assertEqual(list(f.generate("ab")), [])

    def test_generate_transducer(self):
        f = FST.re("(cat):(dog)")
        self.assertEqual(list(f.generate("cat")), ["dog"])
        self.assertEqual(list(f.generate("dog")), [])

    def test_analyze_transducer(self):
        """analyze maps in the inverse direction."""
        f = FST.re("(cat):(dog)")
        self.assertEqual(list(f.analyze("dog")), ["cat"])
        self.assertEqual(list(f.analyze("cat")), [])

    def test_generate_with_weights(self):
        f = FST.re("a<2.5>")
        results = list(f.generate("a", weights=True))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], ("a", 2.5))

    def test_generate_ambiguous(self):
        f = FST.re("(a):(b) | (a):(c)")
        self.assertEqual(set(f.generate("a")), {"b", "c"})

    def test_generate_empty_fst(self):
        f = FST()
        self.assertEqual(list(f.generate("a")), [])

    def test_apply_pretokenized_list(self):
        """generate/apply accept a pre-tokenized list."""
        f = FST.re("'ch' a t")
        self.assertEqual(list(f.generate(["ch", "a", "t"])), ["chat"])

    def test_tokenize_multichar_symbol(self):
        """Longest multichar symbol wins during tokenization."""
        f = FST.re("'ch' a t")
        self.assertEqual(f.tokenize_against_alphabet("chat"), ["ch", "a", "t"])

    def test_tokenize_single_chars(self):
        f = FST.re("a b c")
        self.assertEqual(f.tokenize_against_alphabet("abc"), ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Word enumeration
# ---------------------------------------------------------------------------

class TestEnumerate(unittest.TestCase):
    """Test words, words_nbest, and words_cheapest."""

    def test_words_count(self):
        f = FST.re("a | b | c")
        self.assertEqual(len(list(f.words())), 3)

    def test_words_empty_fst(self):
        f = FST()
        self.assertEqual(list(f.words()), [])

    def test_words_nbest_length_and_order(self):
        f = FST.re("a<1.0> | b<2.0> | c<3.0>")
        best2 = f.words_nbest(2)
        self.assertEqual(len(best2), 2)
        costs = [cost for cost, _ in best2]
        self.assertEqual(costs, sorted(costs))

    def test_words_cheapest_order(self):
        f = FST.re("a<3.0> | b<1.0> | c<2.0>")
        results = list(f.words_cheapest())
        costs = [cost for cost, _ in results]
        self.assertEqual(costs, sorted(costs))
        self.assertEqual(len(costs), 3)

    def test_words_nbest_cheapest_first(self):
        f = FST.re("a<3.0> | b<1.0> | c<2.0>")
        costs = [cost for cost, _ in f.words_nbest(3)]
        self.assertEqual(costs, sorted(costs))


# ---------------------------------------------------------------------------
# Properties and queries
# ---------------------------------------------------------------------------

class TestProperties(unittest.TestCase):
    """Test arity, arccount, pathcount, is_deterministic, has_weights, len."""

    def test_len_is_state_count(self):
        f = FST.re("a b c")
        self.assertEqual(len(f), len(f.states))
        self.assertGreater(len(f), 0)

    def test_arity_acceptor(self):
        self.assertEqual(FST.re("a b c").arity(), 1)

    def test_arity_transducer(self):
        self.assertEqual(FST.re("(a):(b)").arity(), 2)

    def test_arity_empty_fst(self):
        self.assertEqual(FST().arity(), 1)

    def test_arccount_simple(self):
        """a b c has exactly 3 transitions."""
        self.assertEqual(FST.re("a b c").arccount(), 3)

    def test_arccount_union(self):
        """a | b has at least 2 arcs."""
        self.assertGreaterEqual(FST.re("a | b").arccount(), 2)

    def test_pathcount_union(self):
        f = FST.re("a | b | c")
        self.assertEqual(f.pathcount(), 3)

    def test_pathcount_single(self):
        f = FST.re("a b c")
        self.assertEqual(f.pathcount(), 1)

    def test_pathcount_cyclic(self):
        """Cyclic FSTs return -1."""
        f = FST.re("a*")
        self.assertEqual(f.pathcount(), -1)

    def test_is_deterministic_true(self):
        f = FST.re("a b c").determinize_unweighted()
        self.assertTrue(f.is_deterministic())

    def test_is_deterministic_false(self):
        """An FST with epsilon transitions is non-deterministic."""
        f = FST.re("a b c").reverse_e()
        self.assertFalse(f.is_deterministic())

    def test_has_weights_false(self):
        self.assertFalse(FST.re("a b c").has_weights())

    def test_has_weights_true(self):
        self.assertTrue(FST.re("a<2.0>").has_weights())


# ---------------------------------------------------------------------------
# Structural utilities
# ---------------------------------------------------------------------------

class TestStructuralUtils(unittest.TestCase):
    """Test number_unnamed_states and become."""

    def test_number_unnamed_states_all_assigned(self):
        f = FST.re("a b c")
        nums = f.number_unnamed_states()
        for s in f.states:
            self.assertIn(id(s), nums)

    def test_number_unnamed_states_initial_is_zero(self):
        """A bare FST() has name=None states; the counter starts at 0."""
        f = FST()
        nums = f.number_unnamed_states()
        self.assertEqual(nums[id(f.initialstate)], 0)

    def test_number_unnamed_states_force(self):
        f = FST.re("a b c").label_states_topology()
        nums = f.number_unnamed_states(force=True)
        self.assertEqual(len(nums), len(f.states))

    def test_become_mutates_self(self):
        f1 = FST.re("a b c")
        f2 = FST.re("x y z")
        f1.become(f2)
        self.assertEqual(list(f1.generate("xyz")), ["xyz"])
        self.assertEqual(list(f1.generate("abc")), [])


if __name__ == "__main__":
    unittest.main()
