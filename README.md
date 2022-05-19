# PyFoma
Python Finite-State Toolkit

__PyFoma__ is a an open source (Apache) package for finite-state automaton and transducer modeling and learning. It is implemented entirely in Python with no external dependencies.

__PyFoma__ supports:

- [Compiling both weighted and unweighted automata](./docs/RegularExpressionCompiler.ipynb) and transducers (FSMs) from Perl/Python-like regular expressions.
- All the standard weighted and unweighted automata algorithms: epsilon-removal, determinization, minimization, composition, shortest-path algorithms, extraction of strongly connected components, (co)accessibility, etc.
- Weights in the _tropical semiring_ for automata and transducer construction both for low-level construction methods and regular expressions.
- Integration with Jupyter-style notebooks for automata visualization and debugging.
- Custom extensions to the regular expression parser and compiler using Python.
- Compilation of morphological lexicons as weighted or unweighted right-linear grammars, similarly to the [lexc](https://fomafst.github.io/morphtut.html#The_lexc-script)-formalism.
- A comprehensive replacement-rule formalism to construct string-rewriting transducers.

The PyFoma implementation aims at a level of abstraction where most major finite-state algorithms are implemented clearly and asymptotically optimally in what resembles canonical pseudocode, so the code itself could be used for instructional purposes. Additionally, many algorithms can illustrate how they work. The regular expression and right-linear grammar formalisms are intended to be accessible to linguists and computer scientists alike.

----

## History

As a tool, PyFoma is unrelated to the [__foma__](https://fomafst.github.io) compiler, which is implemented in C and uses the Xerox formalism for regular expressions and which has its own Python extensions, but it inherits many of its FSM construction algorithms. The regular expression formalism is influenced by [The Kleene Programming Language](http://www.kleene-lang.org/).

----
