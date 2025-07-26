from dataclasses import dataclass
from functools import reduce
import re
from typing import List, Literal, Optional, Tuple
from pyfoma import FST

@dataclass
class LexdError(Exception):
    message: str
    line_number: int

SectionType = Literal['patterns', 'pattern', 'lexicon']

def parse_lexd(lexd_string: str):
    patterns, lexicons = _parse_lines(lexd_string.splitlines())

    fsts = {}
    for lexicon_name, entries in lexicons.items():
        entry_fsts = [_compile_lexicon(entry) for entry in entries]
        fsts[lexicon_name] = reduce(lambda x, y: x|y, entry_fsts).determinize_as_dfa().minimize()

    breakpoint()


def _parse_lines(lines: List[str]):
    current_section: Optional[Tuple[SectionType, str]] = None
    patterns: dict[str, List[str]] = {"": []} # The empty string is used for unnamed patterns
    lexicons: dict[str, List[str]] = {}
    for line_number, line in enumerate(lines):
        line = line.strip()
        if not line:
            current_section = None
        elif line.startswith('#'):
            continue
        elif line == 'PATTERNS':
            current_section = ('patterns', '')
        elif line.startswith('PATTERN '):
            pattern_name = line.split(None, 1)[1]
            if pattern_name in patterns:
                raise LexdError(f"Pattern `{pattern_name}` is already defined", line_number)
            current_section = ('pattern', pattern_name)
            patterns[pattern_name] = []
        elif line.startswith('LEXICON '):
            lexicon_name = line.split(None, 1)[1]
            if lexicon_name in lexicons:
                raise LexdError(f"Lexicon `{lexicon_name}` is already defined", line_number)
            current_section = ('lexicon', lexicon_name)
            lexicons[lexicon_name] = []
        elif line.startswith('ALIAS '):
            raise NotImplementedError()
        else:
            # We must be in an existing section
            if current_section is None:
                raise LexdError("Definition must be part of a section!", line_number)
            mode, section_name = current_section
            if mode == 'patterns':
                patterns[''].append(line)
            elif mode == 'pattern':
                patterns[section_name].append(line)
            elif mode == 'lexicon':
                lexicons[section_name].append(line)
    return patterns, lexicons

def _compile_lexicon(entry: str) -> FST:
    left_tokens, right_tokens = _tokenize_lex_entry(entry)
    # Regex
    if left_tokens[0] == '/' and left_tokens[-1] == '/':
        return FST.re(''.join(left_tokens[1:-1]))
    left = " ".join("'" + token + "'" for token in left_tokens)
    right = " ".join("'" + token + "'" for token in right_tokens)
    return FST.re(f"({left}):({right})")

def _tokenize_lex_entry(entry: str):
    """Tokenize a lexicon entry into input and output symbol lists."""
    if entry.startswith('/') and entry.endswith('/'):
        return list(entry), list(entry) # regex entry
    if ':' in entry:
        left, right = entry.split(':', 1)
    else:
        left, right = entry, entry
    in_syms = _split_symbols(left.strip())
    # empty right side means epsilon
    right = right.strip()
    out_syms = [] if right == '' else _split_symbols(right)
    # represent epsilon by [''] if no symbols
    if not out_syms:
        out_syms = ['']
    return in_syms, out_syms

def _split_symbols(s: str):
    """Split a string into multicharacter symbols (<...>) or single chars."""
    # This almost works, unless you 
    chars: List[str] = re.findall(r"(?<!\\)<[^>]+(?<!\\)>|(?<!\\)\{[^\}]+(?<!\\)\}|.", s)
    chars = [char for char in chars if char != "\\"]
    return chars


# Example of building a lexicon FST for testing
if __name__ == '__main__':
    example = '''
PATTERNS
Inflection

PATTERN Inflection
VerbRoot VerbInfl
VerbRoot

LEXICON VerbRoot
sing
walk
dance

LEXICON VerbInfl
<prog>:ing
<pres><3p>:s
<inf>:
/(cat)+/
'''
    parse_lexd(example)