"""Tests for FST operations: set ops, composition, Kleene, projection,
reversal, rewrite rules, normalization, weights, and label manipulation."""

import operator
import unittest

from pyfoma.fst import FST


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

class TestSetOperations(unittest.TestCase):
    """Test union, intersection, difference, complement, and product."""

    def test_union_simple(self):
        f1 = FST.re("a|''")
        f2 = FST.re("a?")
        self.assertEqual(list(f1.words()), list(f2.words()))

    def test_union_operator(self):
        f1 = FST.re("cat")
        f2 = FST.re("dog")
        f3 = f1 | f2
        self.assertEqual(list(f3.generate("cat")), ["cat"])
        self.assertEqual(list(f3.generate("dog")), ["dog"])
        self.assertEqual(list(f3.generate("bird")), [])

    def test_intersection_simple(self):
        f1 = FST.re("a b* & a* b")
        f2 = FST.re("a b")
        self.assertEqual(list(f1.words()), list(f2.words()))

    def test_intersection_operator(self):
        f1 = FST.re("a b*")
        f2 = FST.re("a* b")
        f3 = f1 & f2
        self.assertEqual(list(f3.generate("ab")), ["ab"])
        self.assertEqual(list(f3.generate("a")), [])
        self.assertEqual(list(f3.generate("b")), [])

    def test_intersection_mismatched_alphabets(self):
        f1 = FST.re("a | b | c")
        f2 = FST.re("b | c | d")
        f3 = f1 & f2
        self.assertEqual(list(f3.generate("a")), [])
        self.assertEqual(list(f3.generate("b")), ["b"])
        self.assertEqual(list(f3.generate("c")), ["c"])
        self.assertEqual(list(f3.generate("d")), [])

    def test_difference_simple(self):
        f1 = FST.re("(a b) - (a b) (a b)+")
        f2 = FST.re("a b")
        self.assertEqual(list(f1.words()), list(f2.words()))

    def test_difference_operator(self):
        f1 = FST.re("a | b | c")
        f2 = FST.re("b")
        f3 = f1 - f2
        self.assertEqual(list(f3.generate("a")), ["a"])
        self.assertEqual(list(f3.generate("b")), [])
        self.assertEqual(list(f3.generate("c")), ["c"])

    def test_complement_basic(self):
        f1 = FST.regex("~a")
        self.assertEqual(0, len(list(f1.generate("a"))))

    def test_complement_union(self):
        f1 = FST.regex("~(cat | dog)")
        self.assertEqual(0, len(list(f1.generate("cat"))))
        self.assertEqual(0, len(list(f1.generate("dog"))))
        self.assertEqual(1, len(list(f1.generate("octopus"))))

    def test_complement_concatenation_precedence(self):
        # ~ binds tighter than concatenation
        f1 = FST.regex("~(cat | dog)s")
        self.assertEqual(0, len(list(f1.generate("cats"))))
        self.assertEqual(0, len(list(f1.generate("dogs"))))
        self.assertEqual(1, len(list(f1.generate("octopus"))))
        self.assertEqual(1, len(list(f1.generate("catdogs"))))

    def test_complement_kleene_precedence(self):
        # * binds tighter than ~
        f1 = FST.regex("~(cat | dog)*s")
        self.assertEqual(0, len(list(f1.generate("catdogs"))))
        self.assertEqual(0, len(list(f1.generate("catdogcats"))))

    def test_complement_method(self):
        f1 = FST.regex("octopus")
        f2 = f1.complement()
        self.assertEqual(1, len(list(f2.generate("dog"))))
        self.assertEqual(0, len(list(f2.generate("octopus"))))

    def test_product_default_is_union(self):
        """product with default arguments computes union."""
        f1 = FST.re("cat")
        f2 = FST.re("dog")
        f3 = f1.product(f2)
        self.assertEqual(list(f3.generate("cat")), ["cat"])
        self.assertEqual(list(f3.generate("dog")), ["dog"])
        self.assertEqual(list(f3.generate("fish")), [])

    def test_product_intersection(self):
        """finalf=all gives intersection."""
        f1 = FST.re("a b*")
        f2 = FST.re("a* b")
        f3 = f1.product(f2, finalf=all, oplus=operator.add,
                        pathfollow=lambda x, y: x & y)
        self.assertEqual(list(f3.generate("ab")), ["ab"])
        self.assertEqual(list(f3.generate("a")), [])
        self.assertEqual(list(f3.generate("b")), [])

    def test_union_with_dot_expands_correctly(self):
        """'.' in one FST is expanded to include symbols from the other."""
        f1 = FST.re(".")
        f2 = FST.re("z")
        f3 = f1 | f2
        self.assertIn("z", set(f3.generate("z")))
        self.assertIn("a", set(f3.generate("a")))


# ---------------------------------------------------------------------------
# Composition, cross-product, invert
# ---------------------------------------------------------------------------

class TestComposition(unittest.TestCase):
    """Test compose, invert, cross_product, and the @, ** operators."""

    def test_compose_basic(self):
        f1 = FST.re("(cat):(dog)")
        f2 = FST.re("(dog):(octopus)")
        f3 = f1.compose(f2)
        self.assertEqual(list(f3.generate("cat")), ["octopus"])

    def test_invert_basic(self):
        f1 = FST.re("(cat):(dog)")
        f4 = f1.invert()
        self.assertEqual(list(f4.generate("dog")), ["cat"])

    def test_matmul_operator(self):
        f1 = FST.re("(cat):(dog)")
        f2 = FST.re("(dog):(fish)")
        f3 = f1 @ f2
        self.assertEqual(list(f3.generate("cat")), ["fish"])

    def test_cross_product_epsilon_input(self):
        f1 = FST.re("'':x")
        self.assertEqual(set(f1.generate("")), {"x"})

    def test_cross_product_optional_output(self):
        f2 = FST.re("'':?x")
        self.assertEqual(set(f2.generate("")), {"x", ""})

    def test_pow_operator(self):
        f1 = FST.re("cat")
        f2 = FST.re("dog")
        f3 = f1 ** f2
        self.assertEqual(list(f3.generate("cat")), ["dog"])


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

class TestConcatenation(unittest.TestCase):
    """Test concatenate and the * operator."""

    def test_basic(self):
        f1 = FST.re("cat")
        f2 = FST.re("s")
        f3 = f1.concatenate(f2)
        self.assertEqual(list(f3.generate("cats")), ["cats"])
        self.assertEqual(list(f3.generate("cat")), [])

    def test_with_kleene_star(self):
        f1 = FST.re("a")
        f2 = FST.re("b*")
        f3 = f1.concatenate(f2)
        self.assertEqual(list(f3.generate("a")), ["a"])
        self.assertEqual(list(f3.generate("ab")), ["ab"])
        self.assertEqual(list(f3.generate("abbb")), ["abbb"])
        self.assertEqual(list(f3.generate("b")), [])

    def test_mul_operator(self):
        f1 = FST.re("cat")
        f2 = FST.re("s")
        self.assertEqual(list((f1 * f2).generate("cats")),
                         list(f1.concatenate(f2).generate("cats")))


# ---------------------------------------------------------------------------
# Kleene and optional
# ---------------------------------------------------------------------------

class TestKleeneAndOptional(unittest.TestCase):
    """Test kleene_star, kleene_plus, kleene_closure, and optional."""

    def test_kleene_star_accepts_empty(self):
        f = FST.re("a").kleene_star()
        self.assertEqual(list(f.generate("")), [""])

    def test_kleene_star_accepts_repetitions(self):
        f = FST.re("a").kleene_star()
        self.assertEqual(list(f.generate("aaa")), ["aaa"])
        self.assertEqual(list(f.generate("b")), [])

    def test_kleene_plus_rejects_empty(self):
        f = FST.re("a").kleene_plus()
        self.assertEqual(list(f.generate("")), [])

    def test_kleene_plus_accepts_one_or_more(self):
        f = FST.re("a").kleene_plus()
        self.assertEqual(list(f.generate("a")), ["a"])
        self.assertEqual(list(f.generate("aaa")), ["aaa"])

    def test_kleene_closure_star_matches_star(self):
        f1 = FST.re("a").kleene_star()
        f2 = FST.re("a").kleene_closure(mode='star')
        self.assertEqual(list(f1.generate("aaa")), list(f2.generate("aaa")))
        self.assertEqual(list(f1.generate("")), list(f2.generate("")))

    def test_kleene_closure_plus_matches_plus(self):
        f1 = FST.re("a").kleene_plus()
        f2 = FST.re("a").kleene_closure(mode='plus')
        self.assertEqual(list(f1.generate("aaa")), list(f2.generate("aaa")))
        self.assertEqual(list(f1.generate("")), list(f2.generate("")))

    def test_optional_adds_epsilon(self):
        f = FST.re("a b").optional()
        self.assertEqual(list(f.generate("")), [""])
        self.assertEqual(list(f.generate("ab")), ["ab"])
        self.assertEqual(list(f.generate("a")), [])

    def test_optional_idempotent(self):
        """optional on an FST that already accepts '' is a no-op."""
        f = FST.re("a?")
        f2 = f.optional()
        self.assertEqual(list(f.generate("")), list(f2.generate("")))
        self.assertEqual(list(f.generate("a")), list(f2.generate("a")))


# ---------------------------------------------------------------------------
# Projection and reversal
# ---------------------------------------------------------------------------

class TestProjectionAndReversal(unittest.TestCase):
    """Test project, reverse, and reverse_e."""

    def test_project_input(self):
        f = FST.re("(cat):(dog)")
        p = f.project(0)
        self.assertEqual(list(p.generate("cat")), ["cat"])
        self.assertEqual(list(p.generate("dog")), [])

    def test_project_output(self):
        f = FST.re("(cat):(dog)")
        p = f.project(-1)
        self.assertEqual(list(p.generate("dog")), ["dog"])
        self.assertEqual(list(p.generate("cat")), [])

    def test_project_acceptor_unchanged(self):
        f = FST.re("a b c")
        p = f.project(0)
        self.assertEqual(list(f.generate("abc")), list(p.generate("abc")))

    def test_reverse_basic(self):
        f = FST.re("a b c").determinize_unweighted().minimize()
        r = f.reverse()
        self.assertEqual(list(r.generate("cba")), ["cba"])
        self.assertEqual(list(r.generate("abc")), [])

    def test_reverse_palindrome(self):
        f = FST.re("a b a").determinize_unweighted().minimize()
        r = f.reverse()
        self.assertEqual(list(r.generate("aba")), ["aba"])

    def test_reverse_e_with_epsilon_remove(self):
        f = FST.re("a b c").determinize_unweighted().minimize()
        r = f.reverse_e().epsilon_remove().determinize_unweighted().minimize()
        self.assertEqual(list(r.generate("cba")), ["cba"])
        self.assertEqual(list(r.generate("abc")), [])


# ---------------------------------------------------------------------------
# Ignore
# ---------------------------------------------------------------------------

class TestIgnore(unittest.TestCase):
    """Test the ignore method."""

    def test_ignore_without_noise(self):
        f = FST.re("a b")
        noise = FST.re("x")
        f2 = f.ignore(noise)
        self.assertEqual(list(f2.generate("ab")), ["ab"])

    def test_ignore_with_interleaved_noise(self):
        f = FST.re("a b")
        noise = FST.re("x")
        f2 = f.ignore(noise)
        self.assertEqual(list(f2.generate("axb")), ["axb"])

    def test_ignore_pure_noise_rejected(self):
        f = FST.re("a b")
        noise = FST.re("x")
        f2 = f.ignore(noise)
        self.assertEqual(list(f2.generate("xx")), [])


# ---------------------------------------------------------------------------
# Context restriction
# ---------------------------------------------------------------------------

class TestContextRestrict(unittest.TestCase):
    """Test context_restrict."""

    def test_allows_in_context(self):
        x = FST.re("x")
        left = FST.re("a")
        right = FST.re("b")
        f = x.context_restrict((left, right))
        self.assertEqual(list(f.generate("axb")), ["axb"])

    def test_rejects_out_of_context(self):
        x = FST.re("x")
        left = FST.re("a")
        right = FST.re("b")
        f = x.context_restrict((left, right))
        self.assertEqual(list(f.generate("x")), [])

    def test_accepts_string_without_restricted_symbol(self):
        x = FST.re("x")
        left = FST.re("a")
        right = FST.re("b")
        f = x.context_restrict((left, right))
        self.assertEqual(list(f.generate("ab")), ["ab"])


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------

class TestRewriteRules(unittest.TestCase):
    """Test $^rewrite in various configurations."""

    _MULTICHAR_SYMBOLS = "u: ch ll x̌ʷ".split()

    def test_basic_rewrite_with_context(self):
        f1 = FST.re("$^rewrite((ab):x / a b _ a)")
        self.assertEqual(set(f1.generate("abababa")), {"abxxa"})

    def test_simultaneous_rewrite(self):
        bigrule = FST.re(
            "@".join(
                "$^rewrite([mnŋ]:%s / _ %s)" % (nas, stop)
                for nas, stop in zip("mnŋ", "ptk")
            )
        )
        self.assertEqual(list(bigrule.apply("anpinkamto"))[0], "ampiŋkanto")

    def test_rewrite_with_weights(self):
        f1 = FST.re("$^rewrite(a:?(b<1.0>))")
        res = list(f1.analyze("bbb", weights=True))
        self.assertEqual(len(res), 8)
        self.assertEqual(sum(e[1] for e in res), 12.0)

    def test_rewrite_directed_default(self):
        f1 = FST.re("$^rewrite((ab|ba):x)")
        self.assertEqual(len(f1), 5)
        self.assertEqual(set(f1.generate("aba")), {"ax", "xa"})

    def test_rewrite_leftmost(self):
        f2 = FST.re("$^rewrite((ab|ba):x, leftmost = True)")
        self.assertEqual(len(f2), 5)
        self.assertEqual(set(f2.generate("aba")), {"xa"})

    def test_rewrite_longest(self):
        f4 = FST.re("$^rewrite((ab|ba|aba):x, longest = True)")
        self.assertEqual(set(f4.generate("aba")), {"ax", "x"})

    def test_rewrite_shortest(self):
        f6 = FST.re("$^rewrite((ab|ba|aba):x, shortest = True)")
        self.assertEqual(set(f6.generate("aba")), {"xa", "ax"})

    def test_rewrite_longest_leftmost(self):
        f7 = FST.re("$^rewrite((ab|ba|aba):x, longest = True, leftmost = True)")
        self.assertEqual(set(f7.generate("aba")), {"x"})

    def test_rewrite_shortest_leftmost(self):
        f8 = FST.re("$^rewrite((ab|ba|aba):x, shortest = True, leftmost = True)")
        self.assertEqual(set(f8.generate("aba")), {"xa"})

    def test_rewrite_multichar_symbols(self):
        rule = FST.regex(
            "$^rewrite(x̌ʷ:x / u: _ #)", multichar_symbols=self._MULTICHAR_SYMBOLS
        )
        self.assertEqual("xu:x", next(rule.generate("xu:x̌ʷ")))
        self.assertEqual("xux̌ʷ", next(rule.generate("xux̌ʷ")))

    def test_rewrite_longest_multichar_match(self):
        rule = FST.regex(
            "$^rewrite(ABC:D)", multichar_symbols=["A", "B", "C", "AB", "ABC"]
        )
        self.assertEqual("ABD", next(rule.generate("ABABC")))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization(unittest.TestCase):
    """Test trim, filter_accessible/coaccessible, epsilon_remove,
    label_states_topology, determinize*, minimize*, and cleanup_sigma."""

    def test_filter_accessible_preserves_language(self):
        f = FST.re("a b c")
        trimmed = f.filter_accessible()
        self.assertEqual(list(f.generate("abc")), list(trimmed.generate("abc")))

    def test_filter_accessible_removes_unreachable(self):
        from pyfoma.atomic import State
        f = FST.re("a b c")
        orphan = State()
        f.states.add(orphan)
        self.assertIn(orphan, f.states)
        trimmed = f.filter_accessible()
        self.assertNotIn(orphan, trimmed.states)

    def test_filter_coaccessible_preserves_language(self):
        f = FST.re("a b c")
        trimmed = f.filter_coaccessible()
        self.assertEqual(list(f.generate("abc")), list(trimmed.generate("abc")))

    def test_trim_preserves_language(self):
        f = FST.re("a b c")
        trimmed = f.trim()
        self.assertEqual(list(f.generate("abc")), list(trimmed.generate("abc")))

    def test_trim_keeps_all_paths(self):
        def hashable_words(fst):
            return {(cost, tuple(tuple(lbl) for lbl in seq))
                    for cost, seq in fst.words()}
        f = FST.re("cat | dog")
        self.assertEqual(hashable_words(f.trim()), hashable_words(f))

    def test_epsilon_remove_preserves_language(self):
        f = FST.re("a b c")
        f_with_eps = f.reverse_e()
        f_no_eps = f_with_eps.epsilon_remove()
        self.assertEqual(set(f_with_eps.generate("cba")), set(f_no_eps.generate("cba")))

    def test_epsilon_remove_noop_on_epsilon_free(self):
        f = FST.re("a b c")
        f2 = f.epsilon_remove()
        self.assertEqual(list(f.generate("abc")), list(f2.generate("abc")))

    def test_label_states_topology_bfs(self):
        f = FST.re("a b c")
        labeled = f.label_states_topology(mode='BFS')
        self.assertTrue(all(s.name is not None for s in labeled.states))

    def test_label_states_topology_dfs(self):
        f = FST.re("a b c")
        labeled = f.label_states_topology(mode='DFS')
        self.assertTrue(all(s.name is not None for s in labeled.states))

    def test_label_states_topology_preserves_language(self):
        f = FST.re("cat | dog")
        labeled = f.label_states_topology()
        self.assertEqual(set(f.generate("cat")), set(labeled.generate("cat")))
        self.assertEqual(set(f.generate("dog")), set(labeled.generate("dog")))

    def test_determinize_unweighted_is_deterministic(self):
        f = FST.re("a | a b")
        det = f.determinize_unweighted()
        self.assertTrue(det.is_deterministic())

    def test_determinize_unweighted_preserves_language(self):
        f = FST.re("a | a b")
        det = f.determinize_unweighted()
        self.assertEqual(set(det.generate("a")), {"a"})
        self.assertEqual(set(det.generate("ab")), {"ab"})

    def test_determinize_as_dfa_preserves_language(self):
        f = FST.re("cat | dog | cat")
        det = f.determinize_as_dfa()
        self.assertGreater(len(list(det.generate("cat"))), 0)
        self.assertGreater(len(list(det.generate("dog"))), 0)

    def test_determinize_tropical_picks_min_cost(self):
        f = FST.re("a<1.0> | a<3.0>")
        det = f.determinize()
        results = list(det.generate("a", weights=True))
        self.assertGreater(len(results), 0)
        self.assertAlmostEqual(results[0][1], 1.0)

    def test_minimize_reduces_states(self):
        f = FST.re("a | a")
        m = f.minimize()
        self.assertLessEqual(len(m.states), len(f.states))
        self.assertEqual(list(m.generate("a")), ["a"])

    def test_minimize_as_dfa_preserves_language(self):
        f = FST.re("cat | dog | cat")
        m = f.minimize_as_dfa()
        self.assertIn(next(m.generate("cat")), ["cat"])
        self.assertIn(next(m.generate("dog")), ["dog"])

    def test_minimize_brz_preserves_language(self):
        f = FST.re("(a | b)*")
        m = f.minimize_brz()
        self.assertEqual(list(m.generate("")), [""])
        self.assertEqual(list(m.generate("ab")), ["ab"])
        self.assertEqual(list(m.generate("ba")), ["ba"])

    def test_cleanup_sigma_removes_unused(self):
        f = FST.re("a b")
        f2, _ = f.copy_filtered()
        f2.alphabet.add('z')
        f3 = f2.cleanup_sigma()
        self.assertNotIn('z', f3.alphabet)
        self.assertIn('a', f3.alphabet)
        self.assertIn('b', f3.alphabet)

    def test_cleanup_sigma_preserves_dot(self):
        f = FST.re(".")
        f2 = f.cleanup_sigma()
        self.assertIn('.', f2.alphabet)


# ---------------------------------------------------------------------------
# Weight operations
# ---------------------------------------------------------------------------

class TestWeightOperations(unittest.TestCase):
    """Test add_weight and push_weights."""

    def test_add_weight_increases_cost(self):
        f = FST.re("a b")
        fw = f.add_weight(5.0)
        results = list(fw.generate("ab", weights=True))
        self.assertAlmostEqual(results[0][1], 5.0)

    def test_add_weight_nondestructive(self):
        f = FST.re("a b")
        _ = f.add_weight(5.0)
        orig = list(f.generate("ab", weights=True))
        self.assertAlmostEqual(orig[0][1], 0.0)

    def test_push_weights_preserves_total_cost(self):
        f = FST.re("a<1.0> b<2.0> c<3.0>")
        pushed = f.push_weights()
        orig_cost = list(f.generate("abc", weights=True))[0][1]
        pushed_cost = list(pushed.generate("abc", weights=True))[0][1]
        self.assertAlmostEqual(orig_cost, pushed_cost, places=5)

    def test_push_weights_preserves_language(self):
        f = FST.re("a<1.0> b<2.0>")
        pushed = f.push_weights()
        self.assertGreater(len(list(pushed.generate("ab"))), 0)
        self.assertEqual(list(pushed.generate("a")), [])

    def test_weighted_union_costs(self):
        f = FST.re("a<1.0> | b<2.0>")
        wa = list(f.generate("a", weights=True))[0][1]
        wb = list(f.generate("b", weights=True))[0][1]
        self.assertAlmostEqual(wa, 1.0)
        self.assertAlmostEqual(wb, 2.0)
        self.assertLess(wa, wb)

    def test_weighted_concatenation_accumulates(self):
        f = FST.re("a<1.0>") * FST.re("b<2.0>")
        results = list(f.generate("ab", weights=True))
        self.assertAlmostEqual(results[0][1], 3.0)

    def test_add_weight_then_push_preserves_cost(self):
        f = FST.re("a b").add_weight(4.0)
        pushed = f.push_weights()
        orig_cost = list(f.generate("ab", weights=True))[0][1]
        pushed_cost = list(pushed.generate("ab", weights=True))[0][1]
        self.assertAlmostEqual(orig_cost, pushed_cost, places=5)


# ---------------------------------------------------------------------------
# Label manipulation
# ---------------------------------------------------------------------------

class TestLabelManipulation(unittest.TestCase):
    """Test map_labels, copy_mod, and copy_filtered."""

    def test_map_labels_relabel_symbol(self):
        f = FST.re("a b c")
        f2 = f.map_labels({'b': 'x'})
        self.assertEqual(list(f2.generate("axc")), ["axc"])
        self.assertEqual(list(f2.generate("abc")), [])

    def test_map_labels_to_epsilon(self):
        """Mapping 'b' to '' creates an epsilon transition; FST accepts 'ac'."""
        f = FST.re("a b c")
        f2 = f.map_labels({'b': ''})
        self.assertIn("ac", list(f2.generate("ac")))

    def test_map_labels_updates_alphabet(self):
        f = FST.re("a b c")
        f2 = f.map_labels({'b': 'x'})
        self.assertIn('x', f2.alphabet)
        self.assertNotIn('b', f2.alphabet)

    def test_copy_mod_identity(self):
        f = FST.re("cat | dog")
        f2 = f.copy_mod()
        self.assertEqual(set(f.generate("cat")), set(f2.generate("cat")))
        self.assertEqual(set(f.generate("dog")), set(f2.generate("dog")))

    def test_copy_mod_modweight(self):
        """Adding 5.0 per transition → total path cost = 10.0 for two-arc path."""
        f = FST.re("a b")
        f2 = f.copy_mod(modweight=lambda l, w: w + 5.0)
        results = list(f2.generate("ab", weights=True))
        self.assertAlmostEqual(results[0][1], 10.0)

    def test_copy_mod_modlabel(self):
        f = FST.re("a b c")
        f2 = f.copy_mod(modlabel=lambda l, w: l + ('x',))
        self.assertEqual(list(f2.generate("abc")), ["xxx"])

    def test_copy_filtered_no_filter(self):
        f = FST.re("a b c")
        f2, _ = f.copy_filtered()
        self.assertEqual(list(f.generate("abc")), list(f2.generate("abc")))

    def test_copy_filtered_removes_transitions(self):
        f = FST.re("a | b | c")
        f2, _ = f.copy_filtered(labelfilter=lambda lbl: lbl != ('b',))
        self.assertEqual(list(f2.generate("a")), ["a"])
        self.assertEqual(list(f2.generate("b")), [])
        self.assertEqual(list(f2.generate("c")), ["c"])


if __name__ == "__main__":
    unittest.main()
