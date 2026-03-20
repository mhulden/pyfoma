# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The PyFoma regex compiler
#
# The regular expression compiler in __PyFoma__ employs a formalism as close as possible to standard regular expressions as in the Python `re`-module. There are some significant differences, however. Since __PyFoma__ also supports transducers and multitape automata, there are many extensions to basic regular expression formalisms.
#
# Generally, one uses the `re` method in the `FST` class to compile a regular expression to a finite-state machine (FSM). Calling the compiler returns an `FST` object, which also has an associated `.view` method, used extensively here to illustrate the output of the compiler.

# %%
from pyfoma import FST
myfst = FST.re("(cat|dog|mouse)s?")
myfst.view()

# %% [markdown]
# By default, the compiler will produce deterministic and minimal automata from regular expressions. Additionally, weighted automata (see section on weights) will have their weights normalized (pushed) by the compiler.
#
# Whitespace is not significant in regular expressions, and will need to be escaped with a backslash. Note that backslashes are expressed differently when the string provided to the compiler is a "raw string" e.g. `r"Hello\ World!"` vs. `"Hello\\ World!"`: the following two regexes are identical.

# %%
FST.re("Hello\\ World!").view() # Needs two backslashes since we aren't passing a raw string

# %%
FST.re(r"Hello\ World!").view() # Raw strings work with one backslash

# %% [markdown]
# # Basic examples 

# %%
myfst1 = FST.re("cat")
myfst2 = FST.re("c a t")                        # same as above
myfst3 = FST.re(r"\+ \* \ ")                    # literalizing special characters
myfst4 = FST.re(r"'+' '*' ' ' '''")             # same as above using single quotes (note literalized single quote)
myfst5 = FST.re("(cat|dog|mouse)s?")    
myfst6 = FST.re("[A-Za-z0-9] - [aeiouAEIOU]")   # all ASCII characters, except vowels (w/ set subtraction)
myfst7 = FST.re("[^aeiou]")                     # negated character class
myfst8 = FST.re("(cat):(gato) @ (gato):(chat)") # cross-product (:), composition (@)
myfst9 = FST.re("cat<1.0>|dog<2.0>|mouse<3.0>") # weights specified by <float>

# %% [markdown]
# ## Multi-character symbols
#
# To treat a sequence of characters as a single symbol in a regular expression, enclose it in single quotes.  If you wish to have literal single quotes in a multi-character symbol, you can escape them with backslash (remember to use raw strings or to escape the backslash as noted above):

# %%
myfst1 = FST.re("(cat'+Pl'):(cats)")
myfst2 = FST.re(r"(is'n\'t'):(is ' ' not)")

# %% [markdown]
# Similarly to the `Multichar_Symbols` section in a `lexc` file, you can also declare a set of multi-character symbols to be automatically quoted in a regular expression.  This may be useful when dealing with Unicode data where a glyph can possibly consist of many characters, or if you wish to treat digraphs as single symbols:

# %%
myfst1 = FST.re("(perro):(dog)", multichar_symbols=["rr"])

# %% [markdown]
# ## Wildcard behavior and FSMs
#
# The regular expression wildcard `.`, as usual, denotes any single symbol. However, since the compiled FSM needs to be logically consistent, the semantics of a `.`-symbol in the resulting FSM is subtly different. In the FSM, a `.`-symbol refers to _any symbol outside the alphabet_. The alphabet is displayed at the bottom when visualizing FSMs, and is also available in the attribute `alphabet'.
#
# Consider:

# %%
x = FST.re("(ab.)*")
x.view()
x.alphabet

# %% [markdown]
# The reason for this difference in regular expression `.` and FSM `.`-symbols can be seen when compiling negated character classes:

# %%
FST.re("[^aeiou]").view() # any single symbol string except [aeiou]

# %% [markdown]
# # Variables: re-using FSMs
#
# When constructing complex expressions, it can be useful to break them up into smaller steps of FSM definitions, and re-use those when constructing a larger FSM. The compiler interprets any string beginning with a `$`-symbol as being a pre-defined FSM.
#
# For example, a (naive) model of a syllable as consisting of (1) zero or more consonants, (2) 1-2 vowels, followed by (3) zero or more consonants, could be built as follows.

# %%
vowel = FST.re("[aeiou]")        # vowels
cons = FST.re("[a-z] - [aeiou]") # define consonant by subtracting vowels from all letters
syllable = FST.re("$cons* $vowel $vowel? $cons*", {'vowel': vowel, 'cons': cons}) # re-use $cons and $vowel
syllable.view()

# %% [markdown]
# Note how line 3 passes a dictionary as its second argument, telling the compiler which Python variables correspond to regex variables `'vowel'` and `'cons'`.  They keys are the strings used for variables inside the regular expression, and the values are the corresponding FSMs.
#
# For many applications, it's more convenient to build a single dictionary step-by-step that contains all the compiled substeps, which can be passed to the compiler as a single argument, as is done here on lines 3 and 4:

# %%
mydefs = {} # Empty dictionary of {name1:FST1, name2:FST2, ...}
mydefs['vowel']    = FST.re("[aeiou]") # Compile and add 'vowel' FST to dict
mydefs['cons']     = FST.re("[a-z] - $vowel", mydefs) # NOTE mydefs passed to compiler as 2nd arg
mydefs['syllable'] = FST.re("$cons* $vowel $vowel? $cons*", mydefs)
mydefs['syllable'].view()

# %% [markdown]
# ## Weights
#
# __PyFoma__ supports weighted FSMs in the [tropical semiring](https://en.wikipedia.org/wiki/Tropical_semiring). In general, all FSMs are weighted, with weights being `0.0` if weights are not specified anywhere. If all weights are zero, the `.view()` method, by default will not show any weights.
#
# Weights are specified as part of regular expressions using `<...>`. For example:
#
#

# %%
wfst = FST.re("a<1.0> b<3.0>* c<5.0> | a<2.0> b<4.0>* d<6.0>")
wfst.view()

# %% [markdown]
# ### Weight pushing
#
# Note the discrepancy between the FSM and the regular expression (which puts a weight of `<1.0>` and `<2.0>` on the first `a`-symbols in the union. The regular expression compiler always calculates the equivalent FSM by pushing the weights so they are discharged as early as possible (through the `FST` method `push_weights()`), and so the outgoing `a`-transitions have weights of `<8.0>` and `<6.0>`, respectively.
#
# ### Determinization in weighted FSMs
#
# Another thing to note in the above example is that weighted automata are not automatically determinized, because the determinization algorithm may not terminate for weighted FSMs. Instead, the compiler performs a kind of pseudo-determinization, treating the label and the weight as a pseudo-label, over which determinization (and minimization) is performed. In the above, `a/8` and `a/6` would be different labels for this determinization algorithm. The method for performing this is `determinize_as_dfa()` in the `FST` class.
#
# The above weighted automaton is also an example of a non-determinizable weighted automaton. Running `determinize()` on it will not terminate.
#
# ```
# wfstdet = wfst.determinize() # Don't run this! Won't terminate!
# ```
#
# The culprits are the two self-loops in states `1` and `2` which aren't [twins](https://en.wikipedia.org/wiki/Finite-state_transducer).<a name="cite_ref-1"></a>[<sup>[1]</sup>](#cite_note-1)
#
# A subtle change to the above will allow for determinization, and also shows how the compiler's pseudo-determinization doesn't always yield an optimal weighted automaton.
#
# <a name="cite_note-1"></a>1. [^](#cite_ref-1) There are [algorithms](https://cs.nyu.edu/~mohri/pub/twins.pdf) for deciding whether a weighted automaton is determinizable, but their complexity is such that running them for every potential determinization step of complex expression is not worth it in general.

# %%
wfst = FST.re("a<1.0> b<3.0>* c<5.0> | a<2.0> b<3.0>* d<6.0>") # Will be pseudo-determinized
wfst.view()
wfstdet = wfst.determinize()
wfstdet.view() # True determinization

# %% [markdown]
# # Summary of basic regular expression operators
#
# |                          | PyFoma           | Python re / Perl  |
# |:-|:-|:-|
# | zero or more             | `a*`             | `a*`            |
# | one or more              | `a+`             | `a+`            |
# | optional                 | `a?`             | `a?`            |
# | quantifiers              | `a{2}`           | `a{2}`          |
# | m to n times             | `a{1,3}`         | `a{1,3}`        |
# | union                    | <code>a&#124;b</code>           | <code>a&#124;b</code>          |
# | concatenation            | `abc` or `a b c` | `abc`           |
# | escapes                  | `\+` or `'+'`    | `\+`            |
# | wildcard (any symbol)    | `.`              | `.`             |
# | grouping parentheses     | `(...)`          | `(...)`         |
# | character class          | `[a-z]`          | `[a-z]`         |
# | negated char. class      | `[^a-z]`         | `[^a-z]`        |
# | epsilon (empty string)   |  `''`            | N/A             |
# | variables                | `$vowel`         | {vowel}         |
# | multi-character symbol   | `'[Noun]'`       | N/A             |
# | intersection             | `&`              | N/A             |
# | subtraction              | `-`              | N/A             |
# | complement               | `~`              | N/A             |
# | cross-product            | `:`              | N/A             |
# | composition              | `@`              | N/A             |
# | add weight to expression | `<float>`        | N/A             |
#

# %% [markdown]
# # Built-in Functions 
#
# To avoid an extensive set of regular expression operators, much of the functionality of the compiler is accessible through built-in functions. These are identified with the sigil `$^`.
#
# For example, the reversal of a set is accessed by `$^reverse(arg)`.
#
# ## reverse

# %%
FST.re("$^reverse(cat | dog)").view()

# %% [markdown]
# ## determinize
#
# The built-in `$^determinize()`-function simply forces the weighted determinization algorithm as referenced above in the section on weights.

# %%
FST.re("$^determinize(a<1.0> b<3.0>* c<5.0> | a<2.0> b<3.0>* d<6.0>)").view()

# %% [markdown]
# ## minimize
#
# The `$^determinize()`-function calls `minimize()` in the `FST` class. Normally all FSMs returned by the regular expression compiler are minimized after compilation, so calling `minimize()` on a regular expression is redundant. However, if you have an `FST` that isn't produced by the compiler, this function my be useful. Also, the compiler does not perform minimization between each step of the compilation. Sometimes, minimizing an intermediate result may result in faster compilation of the whole expression, and the function gives the user more fine-grained control over this kind of intermediate minimization.
#
# Minimization with weighted automata/transducers treats the combination of label tuple/weight as a single label for the purposes of minimization.

# %% [markdown]
# ## ignore
#
# The ignore-function calculates a FSM where the second argument's set of strings can freely occur anywhere between symbols of the first argument's strings:

# %%
FST.re("$^ignore(cat|dog, x)").view()

# %% [markdown]
# # restrict
#
# The built-in `$^restrict()`-function can take multiple arguments. It defines the language where strings from the first argument, if present, is only allowed if it occurs in one of the specified "contexts". In other words, it defines an automaton that accepts any input, except those that would violate the restriction conditions.
#
# For example, defining the language that follows the (incorrect) generalization about when English words are spelled ei or ie - "__i before e except after c__" - could be implemented as follows:

# %%
FST.re("$^restrict(ei / c _) & $^restrict(ie / [^c] | # _)").view()

# %% [markdown]
# The above automaton would accept words such as __receive__, __believe__, __receipt__, but would reject __recieve__, __beleive__, __reciept__.
#
# Multiple allowed contexts can be specified, separating them with a comma. The symbol `#` can be used in the contexts to restrict occurrences to the beginning of a string or the end.
#
# The following restriction would specify all words with the restriction that if an `a` occurs, it must occur either at the beginning, or the end of the string.

# %%
FST.re("$^restrict(a / # _ , _ #)").view()


# %% [markdown]
# # User-defined functions 
#
# __Pyfoma__ also supports user-defined functions. These are Python functions that the regular expression compiler will call with the same number of arguments as passed to the compiler. The user-defined functions are *passed as a set* to the `re`-method with the keyword argument `functions`.
#
# As a simple example, let's say we wanted a function that made all states in an automaton final with weight `0.0`. We define the function in Python; the function is passed the `FST` object as an argument, and changes the object's set of `finalstates` to be all states, and assigns `0.0` to all `finalweight` attributes of each state.
#
# After defining the function, we pass the regex compiler the keyword argument `functions` (in this case only containing the single function `allfinal`).
#

# %%
def allfinal(myfst):
    for s in myfst.states:
        s.finalweight = 0.0
    myfst.finalstates = myfst.states
    return myfst

FST.re("$^allfinal(cat|dog)", functions = {allfinal}).view()


# %% [markdown]
# In many cases, we don't need to access the internals of an `FST` object to create a function. We can use the regular expression compiler itself to define the function's behavior.
#
# For example, suppose we made extensive use of the idiom "does not contain any string from the set `$X` as a substring", which could be expressed by the regular expression `.* - (.* $X .*)`, we could define a function as follows:

# %%
def notcontain(thefst):
    return FST.re(".* - (.* $X .*)", {'X': thefst}) # note passing of dictionary to recycle myfst as $X

FST.re("$^notcontain(cat)", functions = {notcontain}).view()

# %% [markdown]
# The built-in `$^ignore`-function (see above), could be defined exactly with this method, simply using the regex compiler itself to achieve the result:
#
#  ```python
#      def ignore(A, B):
#         """A, ignoring intervening instances of B."""
#         return FST.re("$^output($A @ ('.'|'':$B)*)", {'A': A, 'B': B})
#  ```
#  
#  In the case of naming conflicts, the regular expression compiler will consult the user-defined functions first, and then the built-in ones.

# %% [markdown]
# # Notes on displaying
#
# All automata returned by __PyFoma__ are weighted multitape automata/transducers. In the case that no weights are specified anywhere, and there is only one tape, the `.view` omits the weights and doesn't distinguish tapes. By passing the `raw` keyword argument, you can force display of the underlying structure:

# %%
FST.re("cat").view()
FST.re("cat").view(raw = True)

# %% [markdown]
# As seen above, all the labels on the transitions are internally tuples, and all the transitions have weights (in this case `0.0`).
#
# Without `raw`, only tuples longer than 1 will be shown as `:`-separated strings, and epsilons (the empty string) will be shown as &#x03f5; (`GREEK LUNATE EPSILON SYMBOL U+03F5`).

# %%
FST.re("(cat):(gato)").view()
FST.re("(cat):(gato)").view(raw = True)

# %% [markdown]
# Multiple transitions that have the same source and target states are shown as a single transition, where the different labels are separated by a comma:

# %%
FST.re("a|b").view()

# %% [markdown]
# ____
#
# # Transducer-related operations
#
# ## cross-product
#
# The only primitive operation that creates transducers (2-tape automata) from regular languages is the cross-product operation (`:`).
#
# For example

# %%
FST.re("(dog|cat|rat):(animal|mammal)").view()

# %% [markdown]
# creates a transducer that accepts as inputs the strings `dog`, `cat`, and `rat`, mapping them to both `animal`, and `mammal`.
#
# In effect, it represents the following __relation__:
# ![Cross-product illustration](./images/dogratcat.png)

# %% [markdown]
# ## precedence of cross product
#
# Note that the cross-product operation binds tighter than concatenation. This means that if we want to calculate the cross-product of two strings - a common use case - the strings need to be parenthesized.
#
# The below two regular expressions are __not__ equivalent:

# %%
FST.re("cat:chat").view()      # maps cathat to cachat
FST.re("(cat):(chat)").view()  # maps cat to chat

# %% [markdown]
# ## optional cross-product
#
# Adding `?` after the cross-product makes it *optional*, i.e. input words will either map to themselves, or to the second argument. The below two expressions illustrate different ways of producing the same transducer.

# %%
FST.re("(cat):?(gato)").view()
FST.re("(cat):(gato)|cat").view()

# %% [markdown]
# ## Projections
#
# The built-in functions `$^input` and `$^output` can be used to extract the first (input) tape or the last (output) tape from a transducer.

# %%
FST.re("$^input((dog|cat|rat):(animal|mammal))").view()
FST.re("$^output((dog|cat|rat):(animal|mammal))").view()

# %% [markdown]
# The above are two (redundant) ways of specifying the simple languages `dog|rat|cat` and `mammal|animal`, respectively.

# %% [markdown]
# ## inverse
#
# The inverse of a transducer is calculated by the `$^invert`-function.

# %%
FST.re("$^invert(a:b)").view()

# %% [markdown]
# # Composition
#
# The composition operator takes two transducers and return the resulting composite transducer.
#
# For example, if we have two transducers, one mapping English words to Spanish words, and the second, mapping Spanish to French, their composite is the direct mapping of English to French:
#
# ![Cross-product illustration](./images/compositionengfre.png)
#
# The above can be compiled as follows:

# %%
FST.re("(cat):(gato)|(dog):(perro) @ (gato):(chat) | (perro):(chien)").view()

# %% [markdown]
# Note the low precedence of `@` - union and concatenation both bind tighter. Also note how the alphabet has been pruned from any intermediate symbols: the `p` and `r` which were present during composition since they appear in `perro`, do not show up in the final alphabet.
#
# Although the above example is trivial and only serves the purpose of illustrating how composition works, composition together with constrained rewrite/replacement rules are often used to build complex models in a variety of domains.
#
# # Weighted composition
#
# If the transducers being composed are weighted, the resulting weights will be added (the otimes-operation in the [Tropical Semiring](https://en.wikipedia.org/wiki/Tropical_semiring)).

# %%
FST.re("(cat<1.0>):(gato)|(dog):(perro) @ (gato<2.0>):(chat) | (perro):(chien)").view()

# %% [markdown]
# ----
#
# # Rewrite rules
#
# __Rewrite rules__, also called __conditional replacement rules__, allow one to specify transducers that modify some part of the input, under certain conditions, leaving other parts untouched. In general, the result will be a transducer that accepts arbitrary strings as input, some of which will pass through the transducer untouched, and others which will be modified by the transducer according to the rule.
#
# Earlier tools that have allowed for the compilation of rewrite rules into transducers generally work with a formalism similar to that of context-sensitive rewrite rules:
#
# $$A \rightarrow B~ /~ C~ \_~ D$$
#
# This would correspond to a transducer that rewrites strings from the set *A* to strings from the set *B*, whenever the string from *A* would occur flanked by strings from $C$ (to the left) and $D$ (to the right). Any other strings would be repeated.
#
# __PyFoma__ takes a more minimalistic approach: the `$^rewrite()`-operation only potentially restricts where an arbitrary transduction may occur. The transduction in question is given as the first argument. If the transduction is allowed anywhere within a string, the context argument(s) can be omitted.
#
# Even simple rewrite rules can give rise to complex transducers that aren't interpretable at a glance. Consider the rule (expressed in formal notation):
#
# $$ab~ \rightarrow~ x~ /~ ab~ \_~ a$$
#
# This would map all instances of $ab$ to $x$, if $ab$ occurs between $ab$ (to the left) and $a$ to the right.
#
# For example, the input string `abababa` would map to `abxxa` as the second and third instances of `ab` would meet the contextual requirements:
#
# ```
# a b a b a b a    (input)
# a b  x   x  a    (output)
# ```
#
# This compiles into a transducer with 7 states and 23 transitions, as follows:

# %%
myrule = FST.re("$^rewrite((a b):x / a b _ a)")
myrule.view()
list(myrule.generate("abababa")) # test abababa output with the generate method

# %% [markdown]
# ## Markup rules
#
# Since the first argument to `$^rewrite` is an arbitrary transducer, that transducer can also be rigged to not just change substrings, but also to repeat material and at strategic points insert material. So-called markup transducers can be built to take advantage of this. 
#
# For example, suppose we wanted to apply tags `<VOWEL>` and `</VOWEL>` to all vowels in an input string, but leave the vowel itself unchanged - in effect inserting tags before and after vowels - we could define a rule:
# ```
# $^rewrite('':'<VOWEL>' [aeiou] '':'</VOWEL>')
# ```
#
# Note the cross-products of the empty string ('') and the tags, as well as the character class which is not involved in the cross product. We don't use a context specification since the rule is intended to apply everywhere a vowel is found. 
#
# Markup rules can of course also use arbitrary context specifications since they are not, essentially, a different rule type than ordinary rules, but only reflect a different construction of the transducer argument.

# %%
vowelmarkup = FST.re("$^rewrite('':'<VOWEL>' [aeiou] '':'</VOWEL>')")
vowelmarkup.view()
list(vowelmarkup.generate("sequoia"))

# %% [markdown]
# # "optional" rules
#
# Rules that apply "optionally", i.e. nondeterministically both apply and don't apply in the specified context can be modeled by creating the transducer argument using the optional cross-product notation, `:?`.
#
# For example, to create a rule that optionally deletes word-final vowels, we could issue:
#
# ```
# $^rewrite([aeiou]:?'' / _ #)
# ```
#
# Note the use of `#` to define the word edge in the context spefication. The `#` has a special meaning in rule contexts if it occurs as the last symbol in the right context or the first symbol in the left.

# %%
FST.re("$^rewrite([aeiou]:?'' / _ #)").view()

# %% [markdown]
# # Directed rules
#
# In many cases there could be alternative overlapping positions where a rule could potentially apply. By default, all the overlapping possibilities are non-deterministically applied by the transducer. For example, consider a rule
#
# ```
# $^rewrite((ab|ba):x)
# ```
#
# Now, given an input string `aba` there are two overlapping ways in which the rule could apply.
#
# ```
# a b a    a b a 
# ―――        ―――
#  x  a    a  x
#
# ```
#
# To control for which application should be preferred, there are three positional arguments that can be passed to `$^rewrite()`:
#
# ```
# leftmost = True
# longest = True
# shortest = True
# ```
#
# In the above example, we could pass the `leftmost = True` keyword argument to the compiler and create a different transducer that would unambiguously pick the leftmost application in the case of overlapping possibilities for rewriting.

# %%
# Example contrasting regular and leftmost application. Compose with input "aba" and extract output.
FST.re("$^output(aba @ $^rewrite((ab|ba):x))").view()  # should give xa and ax as outputs
FST.re("$^output(aba @ $^rewrite((ab|ba):x, leftmost = True))").view() # should only give xa

# %%
# Could also get outputs with the .generate()-method
print(list(FST.re("$^rewrite((ab|ba):x)").generate("aba")))
print(list(FST.re("$^rewrite((ab|ba):x, leftmost = True)").generate("aba")))

# %% [markdown]
# Sometimes we also need either the shortest or longest specification, perhaps in addition to leftmost. Consider a rule
#
# ```
# $^rewrite((ab|ba|aba):x)
# ```
#
# This time, compiled in the regular fashion, the rule would produce three outputs for an input `aba`.
#
#
# ```
#  (1)      (2)      (3)
# a b a    a b a    a b a 
# ―――        ―――    ―――――
#  x  a    a  x       x
#
# ```
#
# Depending on which combination of longest/shortest/leftmost we specify, only a subset of these would be possible:
#
# ```
# NO SPECIFICATION                  =>    (1), (2), and (3)
# longest = True                    =>    (2) and (3)
# leftmost = True                   =>    (1) and (3)
# shortest = True                   =>    (1) and (2)
# longest = True, leftmost = True   =>    (3) only
# shortest = True, leftmost = True  =>    (1) only
# ```

# %%
FST.re("$^output(aba @ $^rewrite((ab|ba|aba):x))").view()
FST.re("$^output(aba @ $^rewrite((ab|ba|aba):x, longest = True))").view()
FST.re("$^output(aba @ $^rewrite((ab|ba|aba):x, leftmost = True))").view()
FST.re("$^output(aba @ $^rewrite((ab|ba|aba):x, shortest = True))").view()
FST.re("$^output(aba @ $^rewrite((ab|ba|aba):x, longest = True, leftmost = True))").view()
FST.re("$^output(aba @ $^rewrite((ab|ba|aba):x, shortest = True, leftmost = True))").view()

# %% [markdown]
# # Weights in rewrite rules
#
# Weights can also be attached to some part of the rewrite rule components, usually the output. For example, to swap `a`'s and `b`'s with the cost of 1.0 could be achieved as:

# %%
FST.re("$^rewrite((a:b)<1.0>|(b:a)<1.0>)").view()

# %% [markdown]
# This is equivalent to the following expression which doesn't make use of `$^rewrite()`.

# %%
FST.re("((a:b)<1.0>|(b:a)<1.0>|[^ab])*").view()

# %% [markdown]
# ## Overview of rewriting strategies
#
# |                        |  PyFoma                                               |  foma/Xerox       | 
# |:-|:-|:-|
# | Basic rewriting             | `$^rewrite(a:b)`                                      | `a -> b`          |
# | Optional rewriting          | `$^rewrite(a:?b)`                                     | `a (->) b`         | 
# | Conditional rewriting       | `$^rewrite(a:b / c _ d, e _ f, ...)`   | <code>a -> b &#124;&#124; c _ d, e _ f, ...</code> |
# | Markup                      | `$^rewrite('':x a '':x)`                              | `a -> x ... x`    |
# | Directed rewriting          | `$^rewrite((a+):x, Longest = True, Leftmost = True)`  | `a @-> x`         |
# | Directed (shortest)         | `$^rewrite((a+):x, Shortest = True, Leftmost = True)` | `a @> x`          |
#

# %% [markdown]
# # Operator Precedence
#
# <style>
# table td, table th, table tr {text-align:left !important;}
# </style>
#     
# | Operator precedence (highest to lowest) | 
# |:----------------------------------------|
# | `:` `:?`                                | 
# | `?` `*` `+` `{m,n}` `<weight>`          | 
# | `~`                                     |
# | (concatenation)                         | 
# | <code>&#124;</code> `&` `-`                            | 
# | `@`                                     | 
# | `_` `,` `/`                             |
#
#
#

# %% [markdown]
# # Table of operators and functions
#
# | PyFoma | Foma/(Xerox) | Description | Example |
# |:-|:-|:-|:-|
# |(...)   |[...]       | Grouping     | <code>(a&#124;e&#124;i&#124;o&#124;u)</code> |
# |<...>, e.g. <2.0> | NA | Weight     | `cat<3.0>` |
# |'Symbol'| Symbol     | Multicharacter symbol | `'[Noun]'` |
# | ab     |{ab} _or_ a b    | Single-character symbol sequence | `foma` |
# | #      | .#.        | String edge (in rules, constraints) | `$^rewrite(d:t / _ #)` |
# |A&#124;B    |A&#124;B      | Union       | <code>cat&#124;dog</code> |
# |A\&B    |A\&B      | Intersection       | `a[a-z]* & [a-z]*a` |
# |A\-B    |A\-B      | Subtraction       | `[a-z]* - [a-z]* a` |
# |A\*      |A\*    | Kleene Star | `[aeiou]*` |
# |A\+      |A\+     | Kleene Plus | <code>(a&#124;b)+</code> |
# |\\\_(regex),\\\\\_(str)| %_  | Escape | `the\ cat` |
# |\$\^reverse(A) | A.r  | Reverse | `$^reverse(cat)` |
# |\$\^invert(A)  | A.i  | Invert  | `$^invert(a:b)` |
# |\$\^ignore(A, B)| A/B  | Ignore  | `$^ignore(abc, x)` |
# |\$\^restrict(A, B _ C, ...) | A => B _ C, ... | Context restriction | `$^restrict([ai], # _ )` |
# |\$\^rewrite(A:B / C _ D, ...) | A -> B &#124;&#124; C _ D, ... | Rewriting | `$^rewrite([aeiou]:'' / _ #)` |
# |\$\^rewrite(A:?B / C _ D, ...) | A (->) B &#124;&#124; C _ D, ... | Optional Rewriting | `$^rewrite(b:?p / _ #)` |
# |\$\^input(A)  | A.1, A.u | Input-side projection | `$^input($T)` | 
# |\$\^output(A) | A.2, A.l | Output-side projection | `$^output($T)` |
# |\$\^project(A, dim = n) | N/A | Arbitrary projection | `$^project($T, dim = 3)` |
# |\$\^shuffle(A, B) | A <> B | Shuffle two FSTs | `$^shuffle(abcd, 123)` |
# |A:B           | A:B, A.x.B | Cross-product | `(cat):(gato)` |
# |A:?B           | N/A       | Optional cross-product | `(cat):?(gato)` |
# |A@B           | A .o. B    | Composition   | `a:b @ b:c` |
# |\'\'          | 0, []      | Epsilon       | <code>([aeiou]:'' &#124; [^aeiou])*</code> |
# |\[...\], e.g. \[a-zA-Z\]       | N/A | Character class | `[aeiou]` |
# |.             | ?          | Any symbol/wildcard | `.* cat .*` |
# |\[\^a\]        | \\a       | Term negation       | `[^a-z]` |
# |A?             | (A)       | Optional (zero or one) | `cats?` |
# |\$variable    | Variable   | User-defined variable FSM | `$Vow $Cons $Vow` |
# |\$\^funcname(...) | funcname(...) | User-defined function | `$^syllabify($words)` |
# |\{m,n\},\{m,\},\{,n\},\{m\} | \^\{m,n\}, \^\>m, \^\<n, \^m | Quantifiers | `a{1,2}` |
#
#

# %% [markdown]
# # References
#
# Beesley, K. R. and Karttunen, L. (2003). *Finite State Morphology*. CSLI Publications, Stanford, CA.
#
# Beesley, K. R. (2012). Kleene, a free and open-source language for finite-state programming. In Proceedings of the 10th International Workshop on Finite State Methods and Natural Language Processing (FSMNLP).
#
# Chomsky, N. and Halle, M. (1968). *The Sound Pattern of English*. Harper & Row.
#
# Gerdemann, D. (2009). Mix and Match Replacement Rules. In *Proceedings of the Workshop on Adaptation of Language Resources and Technology* to New Domains (pp. 39-47).
#
# Gorman, K., and Sproat, R. (2021). *Finite-State Text Processing*. Synthesis Lectures on Human Language Technologies.
#
# Hulden, M. (2009a). *Finite-State Machine Construction Methods and Algorithms for Phonology and Morphology*. PhD Dissertation. University of Arizona.
#
# Hulden, M. (2009b). Foma: a finite-state compiler and library. In Proceedings of the EACL 2009 Demonstrations Session, pages 29–32.
#
# Hulden, M. (2017). Rewrite rule grammars with multitape automata. Journal of Language Modelling, 5, pages 107-130.
#
# Kaplan, R. M. and Kay, M. (1994). Regular models of phonological rule systems. *Computational Linguistics*, 20(3):331–378.
#
# Karttunen, L. (1996). Directed replacement. In *Proceedings of the 34th conference on
# Association for Computational Linguistics*, pages 108–115.
#
# Karttunen, L. (1997). The replace operator. In Roche, E. and Schabes, Y., editors, *Finite-State Language Processing*. MIT Press.
#
# Kempe, A. and Karttunen, L. (1996). Parallel replacement in finite state calculus. In *Proceedings of the 34th annual meeting of the Association for Computational Linguistics*.
#
# Mohri, M. and Sproat, R. (1996). An efficient compiler for weighted rewrite rules. In *Proceedings of the 34th conference on Association for Computational Linguistics*, pages 231–238.
#
# Noord, G. V., and Gerdemann, D. (1999). An extendible regular expression compiler for finite-state approaches in natural language processing. In *International Workshop on Implementing Automata* (pp. 122-139). Springer, Berlin, Heidelberg.
#
# Yli-Jyrä, A. (2007). A new method for compiling parallel replace rules. *Lecture Notes in Computer Science*, 4783.
#
# Yli-Jyrä, A. (2008). Transducers from parallel replace rules and modes with generalized lenient composition. In *Proceedings of FSMNLP 2007*.
#
# Yli-Jyrä, A. and Koskenniemi, K. (2004). Compiling contextual restrictions on strings into
# finite-state automata. In *The Eindhoven FASTAR Days Proceedings*.
#
