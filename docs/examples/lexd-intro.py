# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to lexd (with PyFoma)
#
# This notebook is an introduction to the `lexd` formalism
# implemented in PyFoma.
#
# We cover:
# 1. How `lexd` fits into morphological analyzer design.
# 2. Syntax and behavior through runnable examples.
# 3. The same core topics as the terse and technical `skills/foma-morphology/references/LEXD.md` document, written for AI agents.
#
# Most examples are English. A few non-English examples are used when English is not
# a natural fit (circumfixation and root-pattern morphology).

# %%
from pyfoma import FST, lexd

# Helper functions for compact, uniform display in all runnable examples.
def show_generate(fst, lexical_forms):
    """Print generation results in a compact, readable way."""
    for form in lexical_forms:
        print(f"GEN {form} -> {sorted(set(fst.generate(form)))}")


def show_analyze(fst, surface_forms):
    """Print analysis results in a compact, readable way."""
    for form in surface_forms:
        print(f"ANA {form} -> {sorted(set(fst.analyze(form)))}")


# %% [markdown]
# ## 1. Where `lexd` fits in analyzer design
#
# In finite-state morphology, a very common architecture is:
#
# - **lexicon transducer**: maps lexical analysis strings (stems + tags) to an
#   intermediate representation
# - **rewrite cascade**: maps intermediate representation to final surface forms
#
# In practice:
#
# - `lexd` handles **morphotactics** (which morphemes can combine) **and tagging strategy**
# - `FST.re("$^rewrite(...)")` handles morphophonology and cleanup
#
# For tiny grammars you can build the lexicon transducer directly with regular expressions.
# As grammars grow, `lexd` is usually the cleaner and more maintainable route.

# %% [markdown]
# ### 1.1 First lexicon transducer with `lexd`
#
# This first example only builds the lexicon transducer and applies `+` cleanup.
# We will expand it with morphophonological rules in the next section.

# %%
fsts = {}

fsts["lexicon"] = lexd.compile(
    r"""
PATTERNS
NounStem Number

LEXICON NounStem
cat
dog
spy
church
wish
buzz

LEXICON Number
<N><Sg>:
<N><Pl>:+s
"""
)

fsts["cleanup"] = FST.re("$^rewrite('+':'')")
fsts["lexicon_surface"] = FST.re("$lexicon @ $cleanup", fsts)

show_generate(fsts["lexicon"], ["cat<N><Pl>", "spy<N><Pl>"])
show_generate(fsts["lexicon_surface"], ["cat<N><Pl>", "spy<N><Pl>"])
show_analyze(fsts["lexicon_surface"], ["cats", "dogs"])

# %% [markdown]
# At this point we have no spelling-change rules yet, so `spy<N><Pl>` still behaves as
# simple suffixation (`spys` after cleanup).

# %% [markdown]
# ## 2. Expand the same grammar with rewrite rules
#
# Now we compose in two classic rules:
#
# - `y_to_ie`: `y -> ie / _ + s`
# - `e_insert`: insert `e` after sibilants (`sh`, `ch`, `x`, `s`, `z`) before `+s`
#
# Then we clean up `+`.

# %%
fsts["y_to_ie"] = FST.re("$^rewrite(y:(ie) / _ '+' s)")
fsts["e_insert"] = FST.re("$^rewrite('':e / (sh|ch|x|s|z) _ '+' s)")
fsts["grammar"] = FST.re("$lexicon @ $y_to_ie @ $e_insert @ $cleanup", fsts)

show_generate(
    fsts["grammar"],
    [
        "cat<N><Pl>",
        "spy<N><Pl>",
        "church<N><Pl>",
        "wish<N><Pl>",
        "buzz<N><Pl>",
    ],
)
show_analyze(fsts["grammar"], ["cats", "spies", "churches", "wishes", "buzzes"])

# %% [markdown]
# This is the key design pattern for many analyzers:
#
# `lexicon @ rule1 @ rule2 @ ... @ cleanup`

# %% [markdown]
# ## 3. Side-by-side: regex lexicon vs `lexd` lexicon
#
# Below we build the *same* lexical mapping in two ways and compare outputs.

# %%
fsts_cmp = {}

# Plain regex approach
fsts_cmp["stems"] = FST.re("cat|dog|spy|church|wish|buzz")
fsts_cmp["number"] = FST.re("('<N>' '<Sg>'):'' | ('<N>' '<Pl>'):('+' s)")
fsts_cmp["lexicon_re"] = FST.re("$stems $number", fsts_cmp)

# lexd approach
fsts_cmp["lexicon_lexd"] = lexd.compile(
    r"""
PATTERNS
NounStem Number

LEXICON NounStem
cat
dog
spy
church
wish
buzz

LEXICON Number
<N><Sg>:
<N><Pl>:+s
"""
)

for form in ["cat<N><Pl>", "spy<N><Pl>", "church<N><Pl>"]:
    out_re = sorted(set(fsts_cmp["lexicon_re"].generate(form)))
    out_lexd = sorted(set(fsts_cmp["lexicon_lexd"].generate(form)))
    print(form, "regex:", out_re, "lexd:", out_lexd, "same:", out_re == out_lexd)

# %% [markdown]
# For small grammars both strategies can work. For larger grammars, `lexd` gives a cleaner
# organization of patterns and lexicon components.

# %% [markdown]
# ## 4. Anatomy of a `lexd` grammar
#
# Typical sections:
#
# - `PATTERNS`
# - `PATTERN Name` (named reusable patterns)
# - `LEXICON Name`
# - `ALIAS Source Alias`
#
# For the compounding example, we'll borrow the spirit of
# [xkcd 2043: Boathouse and Houseboat](https://xkcd.com/2043/): the two
# positions in a compound can be modeled as "container" vs "contained".

# %%
grammar_anatomy = r"""
PATTERN Nominal
Thing Number

PATTERNS
Nominal
Thing Link Thing2

LEXICON Thing
car
house
boat

ALIAS Thing Thing2

LEXICON Number
<N><Sg>:
<N><Pl>:+s

LEXICON Link
<cmp>+:
"""

fst_anatomy = lexd.compile(grammar_anatomy)
fst_anatomy_clean = FST.re("$A @ $C", {"A": fst_anatomy, "C": FST.re("$^rewrite('+':'')")})

show_generate(
    fst_anatomy_clean,
    [
        "boat<N><Sg>",
        "house<N><Pl>",
        "house<cmp>+boat",
        "boat<cmp>+house",
    ],
)
show_analyze(fst_anatomy_clean, ["boat", "houses", "houseboat", "boathouse"])

# %% [markdown]
# `ALIAS` is useful when you want two positions to reuse the same inventory while keeping
# pattern definitions readable and allowing mixed compounds like `houseboat` and `boathouse`.

# %% [markdown]
# ## 5. LEXICON entry types
#
# Common entry styles:
#
# - `stem`
# - `lex:surf`
# - `/.../` regex entries
# - tagged entries (`[...]`) for selector-based constraints

# %%
grammar_entries = r"""
PATTERNS
Entry

LEXICON Entry
mouse<N><Pl>:mice
goose<N><Pl>:geese
/fox<N><Pl>:foxes/
book<N><Sg>:book
book<N><Pl>:books
"""

fst_entries = lexd.compile(grammar_entries)
show_generate(fst_entries, ["mouse<N><Pl>", "book<N><Pl>", "fox<N><Pl>"])
show_analyze(fst_entries, ["mice", "geese", "books", "foxes"])

# %% [markdown]
# ## 6. Pattern operators (`|`, `?`, `*`, `+`, grouping)

# %%
grammar_ops = r"""
PATTERN NP
Det? Adj* Noun

PATTERNS
NP

LEXICON Det
<DET>:

LEXICON Adj
<ADJ>:big
<ADJ>:small

LEXICON Noun
cat<N>:cat
dog<N>:dog
"""

fst_ops = lexd.compile(grammar_ops)
show_generate(fst_ops, ["<ADJ><ADJ>cat<N>", "<DET><ADJ><ADJ>dog<N>"])
show_analyze(fst_ops, ["cat", "bigdog", "bigsmallcat"])

# %% [markdown]
# ## 7. Anonymous lexicons and sieve operators
#
# Anonymous lexicons: inline entries without defining a named `LEXICON` block.

# %%
grammar_anon = r"""
PATTERNS
Stem [<n>:] Number

LEXICON Stem
book
cat

LEXICON Number
<sg>:
<pl>:+s
"""

fst_anon = lexd.compile(grammar_anon)
show_generate(fst_anon, ["book<n><sg>", "book<n><pl>"])

# %% [markdown]
# Sieve operators (`<` and `>`) expand one line into multiple allowed lengths.
#
# Here we use an English derivation chain:
#
# `anti + dis + establish + ment + ary + an + ism`
#
# and allow shorter realizations around the root via `<` and `>`.

# %%
grammar_sieve = r"""
PATTERNS
P1 < P2 < Root > S1 > S2 > S3 > S4

LEXICON P1
<NEG>:anti+

LEXICON P2
<REV>:dis+

LEXICON Root
establish

LEXICON S1
<NMLZ>:+ment

LEXICON S2
<ADJ>:+ary

LEXICON S3
<ADJ2>:+an

LEXICON S4
<N>:+ism
"""

fst_sieve = lexd.compile(grammar_sieve)

show_generate(
    fst_sieve,
    [
        "establish",
        "<REV>establish",
        "<NEG><REV>establish",
        "<NEG><REV>establish<NMLZ>",
        "<NEG><REV>establish<NMLZ><ADJ><ADJ2><N>",
    ],
)
show_analyze(
    fst_sieve,
    [
        "establish",
        "dis+establish",
        "anti+dis+establish",
        "anti+dis+establish+ment",
        "anti+dis+establish+ment+ary+an+ism",
    ],
)

# %% [markdown]
# ## 8. Tag selectors and lexical conditioning (mass vs count)
#
# Linguistically, count nouns behave differently from mass nouns in many languages.
# Selectors let us state this directly in morphotactics.

# %%
grammar_selectors = r"""
PATTERNS
Noun[mass]
Noun[count] Number

LEXICON Noun
rice[mass]
water[mass]
book[count]
chair[count]

LEXICON Number
<N><Sg>:
<N><Pl>:+s
"""

fst_selectors = lexd.compile(grammar_selectors)
show_generate(fst_selectors, ["book<N><Pl>", "book<N><Sg>", "rice"])
show_analyze(fst_selectors, ["book+s", "water", "rice+s"])

# %% [markdown]
# `rice+s` is rejected here because the pattern only licenses plural number for `[count]` nouns.

# %% [markdown]
# ## 9. Multi-column lexicons and column references
#
# A practical use case is irregular paradigms: keep many irregular mappings in a
# dedicated multi-column lexicon, while productive morphology stays elsewhere.
# This keeps irregular lists explicit without cluttering productive rules.

# %%
grammar_columns = r"""
PATTERNS
Irreg(1):Irreg(2)

LEXICON Irreg(2)
mouse<N><Sg> mouse
mouse<N><Pl> mice
goose<N><Sg> goose
goose<N><Pl> geese
"""

fst_columns = lexd.compile(grammar_columns)
show_generate(fst_columns, ["mouse<N><Sg>", "mouse<N><Pl>", "goose<N><Sg>", "goose<N><Pl>"])
show_analyze(fst_columns, ["mouse", "mice", "goose", "geese"])

# %% [markdown]
# `Irreg(1):Irreg(2)` maps column 1 to column 2 for the same selected row.
#
# Irregular lemmas usually include both singular and plural entries here, so you
# avoid accidentally deriving a regular plural in parallel (for example, `gooses`).
# Two common design options are:
#
# - Keep irregular lemmas out of the regular noun lexicon entirely.
# - Keep them in a shared stem lexicon, but add a class tag (for example
#   `[noplural]` or `[irreg]`) and use selectors to block regular plural paths.

# %% [markdown]
# ## 10. Reduplication: side placement and contrastive focus
#
# English can use reduplication for contrastive focus (for example, `salad-salad`),
# a point often discussed in the "salad-salad" literature:
# https://www.jstor.org/stable/4048061
#
# Here, side placement lets us duplicate the noun on the surface side while adding
# a lexical tag `<CF>` ("contrastive focus").

# %%
grammar_redup = r"""
PATTERNS
Noun
Noun Focus :Noun

LEXICON Noun
salad
coffee
friend

LEXICON Focus
<CF>:-
"""

fst_redup = lexd.compile(grammar_redup)
show_generate(fst_redup, ["salad", "salad<CF>", "coffee<CF>"])
show_analyze(fst_redup, ["salad", "salad-salad", "coffee-coffee"])

# %% [markdown]
# ## 11. Cross-lexicon pairing (`X(i):Y(j)`)
#
# In this implementation, a pair token is easiest to demonstrate inside a sequence.

# %%
grammar_pair_cross = r"""
PATTERNS
EN(1):FR(1) End

LEXICON EN(1)
cat<N>
dog<N>

LEXICON FR(1)
chat
chien

LEXICON End
:
"""

fst_pair_cross = lexd.compile(grammar_pair_cross)
show_generate(fst_pair_cross, ["cat<N>", "dog<N>"])
show_analyze(fst_pair_cross, ["chat", "chien"])

# %% [markdown]
# ## 12. Binding behavior
#
# Repeating a lexicon name in one sequence often binds choices together.
# First, observe behavior without `ALIAS`.

# %%
grammar_binding_no_alias = r"""
PATTERNS
Root Root

LEXICON Root
boo
zap
"""

fst_binding_no_alias = lexd.compile(grammar_binding_no_alias)
show_analyze(fst_binding_no_alias, ["booboo", "zapzap", "boozap", "zapboo"])

# %% [markdown]
# Now add an alias and compare:
#
# - `Root Root` favors same-entry doubling.
# - With `Root Root2` + `ALIAS Root Root2`, mixed combinations are also licensed.

# %%
grammar_binding_with_alias = r"""
PATTERNS
Root Root
Root Root2

LEXICON Root
boo
zap

ALIAS Root Root2
"""

fst_binding_with_alias = lexd.compile(grammar_binding_with_alias)
show_analyze(fst_binding_with_alias, ["booboo", "zapzap", "boozap", "zapboo"])

# %% [markdown]
# ## 13. Circumfixation (German participle style)

# %%
grammar_circumfix = r"""
PATTERNS
Stem
Circ?(1) Stem Circ?(2)

LEXICON Stem
kaufen:kauf

LEXICON Circ(2)
:ge   <PTCP>:t
"""

fst_circ = lexd.compile(grammar_circumfix)
show_generate(fst_circ, ["kaufen", "kaufen<PTCP>"])
show_analyze(fst_circ, ["kauf", "gekauft"])

# %% [markdown]
# ## 14. Root-pattern morphology (templatic example)

# %%
grammar_root_pattern = r"""
PATTERNS
C(1) :V(1) C(2) :V(2) C(3) V(2):

LEXICON C(3)
sh m r
y sh v

LEXICON V(2)
:a <v><p3><sg>:a
:o <v><pprs>:e
"""

fst_root_pattern = lexd.compile(grammar_root_pattern)
show_generate(fst_root_pattern, ["shmr<v><p3><sg>", "yshv<v><pprs>"])
show_analyze(fst_root_pattern, ["shamar", "yoshev"])

# %% [markdown]
# ## 15. Arabic triliteral templates (regex and lexd)
#
# Start from triliteral roots plus tags, and compose with template transducers.
#
# Arabic templatic morphology is often modeled as:
# - a consonantal root (here, three radicals)
# - a vocalic/prosodic template that interleaves with the root
# - grammatical tags (voice/aspect/form) constrained by the template choice
#
# We model Form II / Form III with voice and aspect:
#
# - `<FormII|FormIII>`
# - `<Act|Pass>`
# - `<Perf|Imperf>`

# %%
AR_TEMPLATE_MS = ["<FormII>", "<FormIII>", "<Act>", "<Pass>", "<Perf>", "<Imperf>"]
fsts_tpl = {}

# Root inventory and a small consonant class for this toy example.
fsts_tpl["C"] = FST.re("(b|d|f|h|j|k|l|m|n|r|s|t|w|z)")
fsts_tpl["Roots"] = FST.re("ktb|drs")
fsts_tpl["Lexicon"] = FST.re(
    "$Roots ((<FormII>|<FormIII>) (<Act>|<Pass>) (<Perf>|<Imperf>))",
    fsts_tpl,
    multichar_symbols=AR_TEMPLATE_MS,
)

# C2 doubles the consumed consonant (Form II gemination).
c2_alts = " | ".join(f"({c}):({c} {c})" for c in "bdfhjklmnrstwz")
fsts_tpl["C2"] = FST.re(c2_alts)

# 15.1 PyFoma regex implementation of the templates.
fsts_tpl["FormII"] = FST.re(
    "$C '':a $C2 '':a $C (<FormII> <Act> <Perf>):'' | "
    "$C '':a $C2 '':i $C (<FormII> <Act> <Imperf>):'' | "
    "$C '':u $C2 '':i $C (<FormII> <Pass> <Perf>):'' | "
    "$C '':a $C2 '':a $C (<FormII> <Pass> <Imperf>):''",
    fsts_tpl,
    multichar_symbols=AR_TEMPLATE_MS,
)
fsts_tpl["FormIII"] = FST.re(
    "$C '':(a a) $C '':a $C (<FormIII> <Act> <Perf>):'' | "
    "$C '':(a a) $C '':i $C (<FormIII> <Act> <Imperf>):'' | "
    "$C '':(u u) $C '':i $C (<FormIII> <Pass> <Perf>):'' | "
    "$C '':(u u) $C '':a $C (<FormIII> <Pass> <Imperf>):''",
    fsts_tpl,
    multichar_symbols=AR_TEMPLATE_MS,
)
fsts_tpl["TemplateRegex"] = FST.re("$FormII | $FormIII", fsts_tpl)
fsts_tpl["GrammarRegex"] = FST.re("$Lexicon @ $TemplateRegex", fsts_tpl)

show_generate(
    fsts_tpl["GrammarRegex"],
    [
        "ktb<FormII><Act><Perf>",
        "ktb<FormII><Pass><Perf>",
        "drs<FormIII><Act><Perf>",
        "drs<FormIII><Pass><Perf>",
    ],
)
show_analyze(fsts_tpl["GrammarRegex"], ["kattab", "kuttib", "daaras", "duuris"])

# %%
# 15.2 Same system in lexd using explicit Root(1) Root(2) Root(3) columns.
#
# Note how `:Root(2)` inserts a copy of the second radical (gemination) on
# the surface side without consuming extra lexical input.
fsts_tpl["TemplateLexd"] = lexd.compile(
    r"""
PATTERNS
Root(1) [:a]  Root(2) :Root(2) [:a] Root(3) [<FormII><Act><Perf>:]
Root(1) [:a]  Root(2) :Root(2) [:i] Root(3) [<FormII><Act><Imperf>:]
Root(1) [:u]  Root(2) :Root(2) [:i] Root(3) [<FormII><Pass><Perf>:]
Root(1) [:a]  Root(2) :Root(2) [:a] Root(3) [<FormII><Pass><Imperf>:]

Root(1) [:aa] Root(2) [:a] Root(3) [<FormIII><Act><Perf>:]
Root(1) [:aa] Root(2) [:i] Root(3) [<FormIII><Act><Imperf>:]
Root(1) [:uu] Root(2) [:i] Root(3) [<FormIII><Pass><Perf>:]
Root(1) [:uu] Root(2) [:a] Root(3) [<FormIII><Pass><Imperf>:]

LEXICON Root(3)
k t b
d r s
"""
)

show_generate(
    fsts_tpl["TemplateLexd"],
    [
        "ktb<FormII><Act><Perf>",
        "ktb<FormII><Pass><Perf>",
        "drs<FormIII><Act><Perf>",
        "drs<FormIII><Pass><Perf>",
    ],
)
show_analyze(fsts_tpl["TemplateLexd"], ["kattab", "kuttib", "daaras", "duuris"])

# %% [markdown]
# In this case both implementations are transparent:
#
# - regex composition mirrors the classic template-transducer view
# - lexd columns (`Root(1..3)`) make root-slot structure very readable

# %% [markdown]
# ## 16. Arabic broken plurals (regex-first recommendation)
#
# In Arabic, many plurals are formed by changing the internal vowel template
# (and sometimes stem shape) of the singular rather than by simple suffixation.
# These are traditionally called "broken plurals": the stem pattern changes
# itself, so plural formation is often class-based rather than fully predictable
# from one suffix rule. For a concise learner-facing overview, see:
# https://www.learnarabiconline.com/broken-plurals/
#
# Broken plurals are often easiest to model with direct regex transducers over
# fully vowelled singular forms. If your lexicon already stores full singular
# stems, direct vowel-pattern rewriting is usually the most practical route.
#
# This is a good example of tool fit: lexd is excellent for morphotactics, but
# not every morphophonological mapping becomes clearer when forced into lexd.

# %%
AR_BPL_MS = ["<N>", "<Sg>", "<Pl>", "<BPL1>", "<BPL2>", "<SG>", "<PL>"]
fsts_bpl = {}
fsts_bpl["C"] = FST.re("(b|d|f|h|j|k|l|m|n|r|s|t|w|z)")

bpl_entries = [
    ("kitaab", "BPL1"),
    ("safiina", "BPL1"),
    ("sabiil", "BPL1"),
    ("kalb", "BPL2"),
    ("jamal", "BPL2"),
    ("rajul", "BPL2"),
]
bpl_pairs = []
for lemma, cls in bpl_entries:
    bpl_pairs.append(f"({lemma}<N><Sg>):({lemma}<{cls}><SG>)")
    bpl_pairs.append(f"({lemma}<N><Pl>):({lemma}<{cls}><PL>)")
fsts_bpl["Lexicon"] = FST.re(" | ".join(bpl_pairs), multichar_symbols=AR_BPL_MS)

fsts_bpl["BPL1Pl"] = FST.re(
    "$C (a|i):(u) $C ((a a)|(i i)):(u) $C (a:'')? (<BPL1> <PL>):''",
    fsts_bpl,
    multichar_symbols=AR_BPL_MS,
)
fsts_bpl["BPL2Pl"] = FST.re(
    "$C (a|i):(i) $C ( ((a|u):(a a)) | ('':(a a)) ) $C (<BPL2> <PL>):''",
    fsts_bpl,
    multichar_symbols=AR_BPL_MS,
)
fsts_bpl["BPL1Sg"] = FST.re(
    "$C (a|i) $C ((a a)|(i i)) $C (a)? (<BPL1> <SG>):''",
    fsts_bpl,
    multichar_symbols=AR_BPL_MS,
)
fsts_bpl["BPL2Sg"] = FST.re(
    "( $C a $C $C | $C a $C a $C | $C i $C $C | $C a $C u $C ) (<BPL2> <SG>):''",
    fsts_bpl,
    multichar_symbols=AR_BPL_MS,
)
fsts_bpl["Pluralize"] = FST.re("$BPL1Pl | $BPL2Pl | $BPL1Sg | $BPL2Sg", fsts_bpl)
fsts_bpl["Grammar"] = FST.re("$Lexicon @ $Pluralize", fsts_bpl)

show_generate(
    fsts_bpl["Grammar"],
    ["kitaab<N><Sg>", "kitaab<N><Pl>", "safiina<N><Pl>", "kalb<N><Pl>", "rajul<N><Pl>"],
)
show_analyze(fsts_bpl["Grammar"], ["kitaab", "kutub", "sufun", "kilaab", "rijaal"])

# %% [markdown]
# ## 17. Escaping and comments
#
# `#` introduces comments in lexd source. Backslash escapes spaces and literal `#`.

# %%
grammar_escaping = r"""
PATTERNS
X

LEXICON X
New\ York
C\#Sharp
"""

fst_escaping = lexd.compile(grammar_escaping)
show_analyze(fst_escaping, ["New York", "C#Sharp"])

# %% [markdown]
# ## 18. Common pitfalls
#
# 1. Hard-coding productive outputs entry-by-entry in `LEXICON`.
# 2. Letting grammar markers leak to final surfaces.
# 3. Creating many ad hoc marker classes instead of broad productive rules.
# 4. Forgetting binding behavior of repeated lexicon references.
# 5. Treating morphology as a plain regex task instead of a designed lexicon+rule system.

# %% [markdown]
# ## 19.  See also
#
# The Apertium project's [documentation of lexd syntax](https://github.com/apertium/lexd/blob/main/Usage.md).
#

# %% [markdown]
# ## References
#
# Jila Ghomeshi, Ray Jackendoff, Nicole Rosen, and Kevin Russell. 2004. Contrastive focus reduplication in English (the salad-salad paper). Natural language & linguistic theory, 22(2), 307-357. [[PDF](https://languagelog.ldc.upenn.edu/myl/llog/SaladSalad.pdf)]
#
# Mans Hulden, Michael Ginn, Miikka Silfverberg, and Michael Hammond. 2024. PyFoma: a Python finite-state compiler module. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 258–265, Bangkok, Thailand. Association for Computational Linguistics.
# [[PDF](https://aclanthology.org/2024.acl-demos.24.pdf)]
#
# Daniel Swanson and Nick Howell. 2021. Lexd: A finite state lexicon compiler
# for non-suffixational morphologies. In Hämäläinen, M., Partanen, N., Alnajjar, K.
# (eds.) Multilingual Facilitation (2021), pages 133-146.
# [[PDF](https://pdfs.semanticscholar.org/8467/4b11a67ad4a83ba594376f6a7a2236d84382.pdf)]
#
#
