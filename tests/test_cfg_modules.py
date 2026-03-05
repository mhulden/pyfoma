import unittest

from pyfoma import FST, State, cfg_approx, cfg_parse


class TestCFGParse(unittest.TestCase):
    def test_backends_direct_vs_regex_match(self):
        # Mirrors notebook usage while asserting both builders are equivalent.
        grammar = """
        S  -> NP VP
        NP -> D N | N
        VP -> V NP
        D  -> the
        N  -> cat
        V  -> chased
        """
        sent = "the cat chased the cat"

        direct = cfg_parse.CFGParse(grammar, local_builder="direct", sent_builder="direct")
        regex = cfg_parse.CFGParse(grammar, local_builder="regex", sent_builder="regex")

        self.assertEqual(direct.parse_ptb(sent, n=5), regex.parse_ptb(sent, n=5))

    def test_weighted_nbest_orders_by_cost(self):
        # Small weighted ambiguity from the docstring model (costs are additive).
        grammar = """
        S -> A | B
        A -> a 1.0
        B -> a 2.5
        """
        parser = cfg_parse.CFGParse(grammar)

        nbest = parser.parse_ptb_nbest("a", n=5)

        self.assertEqual(
            nbest,
            [(1.0, "(S (A a))"), (2.5, "(S (B a))")],
        )


class TestCFGApprox(unittest.TestCase):
    def test_weighted_parse_ptb_with_cost(self):
        # Adapted from the notebook's weighted approximation example.
        grammar = """
        S -> NP VP 1.0
        NP -> N 0.3
        VP -> V NP 0.7
        N -> cats 0.1
        V -> chase 0.2
        """
        parser = cfg_approx.CFG(grammar)
        parser.compile(levels=1)

        parses = parser.parse_ptb_with_cost("cats chase cats", n=3)

        self.assertEqual(
            parses,
            [(2.7, "(S (NP (N cats)) (VP (V chase) (NP (N cats))))")],
        )

    def test_custom_epsilon_symbol_supported(self):
        # Docstring says epsilon symbol is configurable; verify non-default works.
        parser = cfg_approx.CFG("S -> EPS", epsilon_symbol="EPS")
        parser.compile(levels=1)

        self.assertEqual(parser.parse_ptb("", n=3), ["(S EPS)"])


if __name__ == "__main__":
    unittest.main()
