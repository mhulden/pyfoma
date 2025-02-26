.. pyfoma documentation master file, created by
   sphinx-quickstart on Sun Feb 23 15:43:16 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyFoma
====================

**PyFoma** is a an open source (Apache) package for finite-state automaton and transducer modeling and learning. It is implemented entirely in Python with no external dependencies.

PyFoma supports:

- `Compiling both weighted and unweighted automata <https://github.com/mhulden/pyfoma/docs/examples/RegularExpressionCompiler.ipynb>`_ and transducers (FSMs) from Perl/Python-like regular expressions.
- All the standard weighted and unweighted automata algorithms: epsilon-removal, determinization, minimization, composition, shortest-path algorithms, extraction of strongly connected components, (co)accessibility, etc.
- Weights in the *tropical semiring* for automata and transducer construction both for low-level construction methods and regular expressions.
- Integration with Jupyter-style notebooks for automata visualization and debugging.
- Custom extensions to the regular expression parser and compiler using Python.
- `Compilation of morphological lexicons <https://github.com/mhulden/pyfoma/docs/examples/MorphologicalAnalyzerTutorial.ipynb>`_ as weighted or unweighted right-linear grammars, similarly to the `lexc <https://fomafst.github.io/morphtut.html#The_lexc-script>`_-formalism.
- A comprehensive replacement-rule formalism to construct string-rewriting transducers.

The PyFoma implementation aims for a level of abstraction where most major finite-state algorithms are implemented clearly and asymptotically optimally in what resembles canonical pseudocode, so the code itself could be used for instructional purposes. The regular expression and right-linear grammar formalisms are intended to be accessible to linguists and computer scientists alike.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   get_started
   history
   examples
   modules
