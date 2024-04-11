#!/usr/bin/env python

import re as pyre, functools

from pyfoma.fst import FST
import pyfoma.algorithms as alg


class RegexParse:
    shortops = {'|': 'UNION', '-': 'MINUS', '&': 'INTERSECTION', '*': 'STAR', '+': 'PLUS',
                '(': 'LPAREN', ')': 'RPAREN', '?': 'OPTIONAL', ':': 'CP', ':?': 'CPOPTIONAL',
                '~': "COMPLEMENT", '@': "COMPOSE", ',': 'COMMA', '/': 'CONTEXT', '_': 'PAIRUP'}

    # Used to set names so that these functions have useful error messages
    _builtins_project_lambda = lambda *args, **kwargs: alg.projected(*args, dim=int(kwargs.get('dim', '-1')))
    _builtins_project_lambda.__name__ = "project"
    _builtins_input_lambda = lambda x: alg.projected(x, dim=0)
    _builtins_input_lambda.__name__ = "input"
    _builtins_output_lambda = lambda x: alg.projected(x, dim=-1)
    _builtins_output_lambda.__name__ = "output"

    builtins = {'reverse': alg.reversed,
                'invert': alg.inverted,
                'minimize': alg.minimized,
                'determinize': alg.determinized,
                'ignore': alg.ignore,
                'rewrite': alg.rewritten,
                'restrict': alg.context_restricted,
                'project': _builtins_project_lambda,
                'input': _builtins_input_lambda,
                'output': _builtins_output_lambda}
    precedence = {"FUNC": 11, "COMMA": 1, "PARAM": 1, "COMPOSE": 3, "UNION": 5, "INTERSECTION": 5,
                  "MINUS": 5, "CONCAT": 6, "STAR": 9, "PLUS": 9, "OPTIONAL": 9, "WEIGHT": 9,
                  "CP": 10, "CPOPTIONAL": 10, "RANGE": 9, "CONTEXT": 1, "PAIRUP": 2}
    operands = {"SYMBOL", "VARIABLE", "ANY", "EPSILON", "CHAR_CLASS"}
    operators = set(precedence.keys())
    unarypost = {"STAR", "PLUS", "WEIGHT", "OPTIONAL", "RANGE"}
    unarypre = {"COMPLEMENT"}

    def __init__(self, regularexpression, defined, functions):
        """Tokenize, parse, and compile regex into FST.
        'I define UNIX as 30 definitions of regular expressions living under one roof.'
        - Don Knuth, Digital Typography, ch. 33, p. 649 (1999)"""

        self.defined = defined
        self.functions = {f.__name__: f for f in functions}  # Who you gonna call?
        self.expression = regularexpression
        self.tokenized = self._insert_invisibles(self.tokenize())
        self.parsed = self.parse()
        self.compiled = self.compile()

    def character_class_parse(self, charclass):
        """Parse a character class into range pairs, e.g. 'a-zA' => [(97,122), (65,65)].
           'Writing clear and unambiguous specifications for character classes is tough,
           and implementing them perfectly is worse, requiring a lot of tedious and
           uninstructive coding.' -Brian Kernighan (in "Beautiful Code", 2007). """

        negated = False
        if charclass[0] == '^':
            negated = True
            charclass = charclass[1:]

        clncc, escaped = [], set()
        j = 0
        for letter in charclass:  # remove escape chars and index those escaped positions
            if letter != '\\':
                clncc.append(letter)
                j += 1
            else:
                escaped.add(j)
        # Mark positions with a range (i.e. mark where the "-"-symbol is)
        marks = [True if (clncc[i] == '-' and i not in escaped and i != 0 and \
                          i != len(clncc) - 1) else False for i in range(len(clncc))]
        ranges = [(ord(clncc[i - 1]), ord(clncc[i + 1])) for i in range(len(clncc)) if marks[i]]

        # 3-convolve over marks to figure out where the non-range characters are
        singles = [any(m) for m in zip(marks, marks[1:] + [False], [False] + marks)]
        for i in range(len(singles)):
            if singles[i] == False:
                ranges.append((ord(clncc[i]), ord(clncc[i])))  # dummy range, e.g (65, 65)

        if any(start > end for start, end in ranges):
            raise SyntaxError("End must be larger than start in character class range.")
        return ranges, negated

    def compile(self) -> 'FST':
        """Put it all together!
        'If you lie to the compiler, it will have its revenge.' — Henry Spencer."""

        def _stackcheck(s):
            if not s:
                self._error_report(SyntaxError, "You stopped making sense!", line_num, column)
            return s

        def _pop(s):
            return _stackcheck(s).pop()[0]

        def _peek(s):  # For unaries we just directly mutate the FSM on top of the stack
            return _stackcheck(s)[-1][0]

        def _append(s, element):
            s.append([element])

        def _merge(s):  # since we keep the FSTs inside lists, we need to do some`
            _stackcheck(s)  # reshuffling with a COMMA token so that the top 2 elements
            one = s.pop()  # get merged into one list that ends up on top of the stack.
            _stackcheck(s)  # [[FST1], [FST2], [FST3]] => [[FST1], [FST2, FST3]]
            s.append(s.pop() + one)

        def _pairup(s):  # take top two on stack and merge inside list as 2-tuple
            _stackcheck(s)  # [ [FST1], [FST2], [FST3] ] => [ [FST1], [(FST2, FST3)] ]
            one = s.pop()
            _stackcheck(s)
            s.append([tuple(s.pop() + one)])

        def _getargs(s):
            return _stackcheck(s).pop()

        stack, parameterstack = [], []
        for op, value, line_num, column in self.parsed:
            if op == 'FUNC':
                if value in self.functions:  # Look in user-defined functions first ...
                    _append(stack, self.functions[value](*_getargs(stack), **dict(parameterstack)))
                elif value in self.builtins:  # ... then in built-ins
                    _append(stack, self.builtins[value](*_getargs(stack), **dict(parameterstack)))
                else:
                    self._error_report(SyntaxError, "Function \"" + value + \
                                       "\" not defined.", line_num, column)
                parameterstack = []
            elif op == 'LPAREN':
                self._error_report(SyntaxError, "Missing closing parenthesis.", line_num, column)
            elif op == 'COMMA':  # Collect arguments in single list on top of stack
                _merge(stack)
            elif op == 'PARAM':
                parameterstack.append(value)
            elif op == 'PAIRUP':  # Collect argument pairs as 2-tuples on top of stack
                _pairup(stack)
            elif op == 'CONTEXT':  # Same as COMMA, possible future expansion
                _merge(stack)
            elif op == 'UNION':
                _append(stack, _pop(stack).union(_pop(stack)))
            elif op == 'MINUS':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.difference(arg2.determinize_unweighted()))
            elif op == 'INTERSECTION':
                _append(stack, _pop(stack).intersection(_pop(stack)).filter_coaccessible())
            elif op == 'CONCAT':
                second = _pop(stack)
                _append(stack, _pop(stack).concatenate(second).filter_accessible())
            elif op == 'STAR':
                _append(stack, _pop(stack).kleene_closure())
            elif op == 'PLUS':
                _append(stack, _pop(stack).kleene_closure(mode='plus'))
            elif op == 'COMPOSE':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.compose(arg2).filter_coaccessible())
            elif op == 'OPTIONAL':
                _peek(stack).optional()
            elif op == 'RANGE':
                rng = value.split(',')
                lang = _pop(stack)
                if len(rng) == 1:  # e.g. {3}
                    _append(stack, functools.reduce(lambda x, y: alg.concatenate(x, y), [lang] * int(value)))
                elif rng[0] == '':  # e.g. {,3}
                    lang = lang.optional()
                    _append(stack, functools.reduce(lambda x, y: alg.concatenate(x, y), [lang] * int(rng[1])))
                elif rng[1] == '':  # e.g. {3,}
                    _append(stack, functools.reduce(lambda x, y: alg.concatenate(x, y), [lang] * int(rng[0])).concatenate(
                        lang.kleene_closure()))
                else:  # e.g. {1,4}
                    if int(rng[0] > rng[1]):
                        self._error_report(SyntaxError, "n must be greater than m in {m,n}", line_num, column)
                    lang1 = functools.reduce(lambda x, y: alg.concatenate(x, y), [lang] * int(rng[0]))
                    lang2 = functools.reduce(lambda x, y: alg.concatenate(x, y),
                                             [lang.optional()] * (int(rng[1]) - int(rng[0])))
                    _append(stack, lang1.concatenate(lang2))
            elif op == 'CP':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.cross_product(arg2).filter_coaccessible())
            elif op == 'CPOPTIONAL':
                arg2, arg1 = _pop(stack), _pop(stack)
                _append(stack, arg1.cross_product(arg2, optional=True).filter_coaccessible())
            elif op == 'WEIGHT':
                _peek(stack).add_weight(float(value)).push_weights()
            elif op == 'SYMBOL':
                _append(stack, FST(label=(value,)))
            elif op == 'ANY':
                _append(stack, FST(label=('.',)))
            elif op == 'VARIABLE':
                if value not in self.defined:
                    self._error_report(SyntaxError, "Defined FST \"" + value + \
                                       "\" not found.", line_num, column)
                _append(stack, self.defined[value].copy_mod())
            elif op == 'CHAR_CLASS':
                charranges, negated = self.character_class_parse(value)
                _append(stack, FST.character_ranges(charranges, complement=negated))
        if len(stack) != 1:  # If there's still stuff on the stack, that's a syntax error
            self._error_report(SyntaxError, \
                               "Something's happening here, and what it is ain't exactly clear...", 1, 0)
        return _pop(
            stack).trim().epsilon_remove().push_weights().determinize_as_dfa().minimize_as_dfa().label_states_topology().cleanup_sigma()

    def tokenize(self) -> list:
        """Token, token, token, though the stream is broken... ride 'em in, tokenize!"""
        # prematch (skip this), groupname, core regex (capture this), postmatch (skip)
        token_regexes = [
            (r"\\", 'ESCAPED', r".", r""),  # Esc'd sym
            (r", *", 'PARAM', r"\w+ *= *[+-]? *\w+", r""),  # Parameter
            (r"'", 'QUOTED', r"(\\[']|[^'])*", r"'"),  # Quoted sym
            (r"", 'SKIPWS', r"[ \t]+", r""),  # Skip ws
            (r"", 'SHORTOP', r"(:\?|[|\-&*+()?:@,/_])", r""),  # main ops
            (r"\$\^", 'FUNC', r"\w+", r"(?=\s*\()"),  # Functions
            (r"\$", 'VARIABLE', r"\w+", r""),  # Variables
            (r"<", 'WEIGHT', r"[+-]?[0-9]*(\.[0-9]+)?", r">"),  # Weight
            (r"\{", 'RANGE', r"\d+,(\d+)?|,?\d+", r"\}"),  # {(m),(n)}
            (r"\[", 'CHAR_CLASS', r"\^?(\\]|[^\]])+", r"\]"),  # Char class
            (r"", 'NEWLINE', r"\n", r""),  # Line end
            (r"", 'SYMBOL', r".", r"")  # Single sym
        ]
        tok_regex = '|'.join('%s(?P<%s>%s)%s' % mtch for mtch in token_regexes)
        line_num, line_start, res = 1, 0, []
        for mo in pyre.finditer(tok_regex, self.expression):
            op = mo.lastgroup
            value = mo.group(op)
            column = mo.start() - line_start
            if op == 'SKIPWS':
                continue
            elif op == 'ESCAPED' or op == 'QUOTED':
                op = 'SYMBOL'
                value = value.replace("\\", "")
            elif op == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
                continue
            elif op == 'SHORTOP':
                op = self.shortops[value]
            elif op == 'PARAM':
                value = value.replace(" ", "").split('=')
            res.append((op, value, line_num, column))
        return res

    def _insert_invisibles(self, tokens: list) -> list:
        """Idiot hack or genius? We insert explicit CONCAT tokens before parsing.
           'I now avoid invisible infix operators almost entirely. I do remember a few
           texts dealing with theorems about strings in which concatenation was denoted
           by juxtaposition.' (EWD 1300-9)"""

        resetters = self.operators - self.unarypost
        counter, result = 0, []
        for token, value, line_num, column in tokens:  # It's a two-state FST!
            if counter == 1 and token in {'LPAREN', 'COMPLEMENT'} | self.operands:
                result.append(('CONCAT', '', line_num, column))
                counter = 0
            if token in self.operands:
                counter = 1
            if token in resetters:  # No, really, it is!
                counter = 0
            result.append((token, value, line_num, column))
        # Add epsilon to rewrites, restrict with missing contexts, .e.g "(missing) _ #"
        newresult, prevt = [], None
        for token, value, line_num, column in result:
            if ((token == 'COMMA' or token == 'PARAM') and prevt == 'PAIRUP') or \
                    (token == 'PAIRUP' and (prevt == 'CONTEXT' or prevt == 'COMMA')) or \
                    (token == 'RPAREN' and prevt == 'PAIRUP'):
                newresult.append(('SYMBOL', '', line_num, column))
            newresult.append((token, value, line_num, column))
            prevt = token
        return newresult

    def _error_report(self, errortype, errorstring, line_num, column):
        raise errortype(errorstring, ("", line_num, column, self.expression))

    def parse(self) -> list:
        """Attention! Those who don't speak reverse Polish will be shunted!
        'Simplicity is a great virtue but it requires hard work to achieve it and
        education to appreciate it. And to make matters worse: complexity sells better.'
        - E. Dijkstra """
        output, stack = [], []
        for token, value, line_num, column in self.tokenized:
            if token in self.operands or token in self.unarypost:
                output.append((token, value, line_num, column))
            elif token in self.unarypre or token == "FUNC" or token == "LPAREN":
                stack.append((token, value, line_num, column))
                # if token == "LPAREN":
                #    output.append("STARTP")
            elif token == "RPAREN":
                while True:
                    if not stack:
                        self._error_report(SyntaxError, "Too many closing parentheses.", line_num, column)
                    if stack[-1][0] == 'LPAREN':
                        break
                    output.append(stack.pop())
                # output.append("ENDP")
                stack.pop()
                if stack and stack[-1][0] == "FUNC":
                    output.append(stack.pop())
            elif token in self.operators:  # We don't have any binaries that assoc right.
                while stack and stack[-1][0] in self.operators and \
                        self.precedence[stack[-1][0]] >= self.precedence[token]:
                    output.append(stack.pop())
                stack.append((token, value, line_num, column))
        while stack:
            output.append(stack.pop())
        return output
