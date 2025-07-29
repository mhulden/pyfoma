import re
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Literal, Optional, Set, Tuple

from pyfoma import FST


@dataclass
class LexdError(Exception):
    message: str
    line_number: int


SectionType = Literal["patterns", "pattern", "lexicon"]


def parse_lexd(lexd_string: str):
    patterns, lexicons = _parse_lines(lexd_string.splitlines())

    lexicon_fsts = {}
    for lexicon_name, entries in lexicons.items():
        entry_fsts = [_compile_lexicon(*entry) for entry in entries]
        lexicon_fsts[lexicon_name] = (
            reduce(lambda x, y: x | y, entry_fsts).determinize_as_dfa().minimize()
        )

    # Topological sort (Kahn) on patterns to get an order to evaluate them in
    # Also check for cyclic dependencies
    pattern_names_sorted = _topological_sort(
        patterns, lexicon_names=set(lexicons.keys())
    )

    pattern_fsts: Dict[str, FST] = {}
    for pattern_name in pattern_names_sorted:
        entries = patterns[pattern_name]
        entry_fsts = [
            _compile_pattern(
                entry[0], fsts={**lexicon_fsts, **pattern_fsts}, line_number=entry[1]
            )
            for entry in entries
        ]
        pattern_fsts[pattern_name] = (
            reduce(lambda x, y: x | y, entry_fsts).determinize_as_dfa().minimize()
        )

    final_fst = (
        reduce(lambda x, y: x | y, pattern_fsts.values())
        .determinize_as_dfa()
        .minimize()
    )
    return final_fst


def _parse_lines(lines: List[str]):
    current_section: Optional[Tuple[SectionType, str]] = None
    patterns: dict[str, List[Tuple[str, int]]] = {
        "": []
    }  # The empty string is used for unnamed patterns
    lexicons: dict[str, List[Tuple[str, int]]] = {}
    for line_number, line in enumerate(lines):
        line = line.strip()
        if not line:
            current_section = None
        elif line.startswith("#"):
            continue
        elif line == "PATTERNS":
            current_section = ("patterns", "")
        elif line.startswith("PATTERN "):
            pattern_name = line.split(None, 1)[1]
            if pattern_name in patterns:
                raise LexdError(
                    f"Pattern `{pattern_name}` is already defined", line_number
                )
            current_section = ("pattern", pattern_name)
            patterns[pattern_name] = []
        elif line.startswith("LEXICON "):
            lexicon_name = line.split(None, 1)[1]
            if lexicon_name in lexicons:
                raise LexdError(
                    f"Lexicon `{lexicon_name}` is already defined", line_number
                )
            current_section = ("lexicon", lexicon_name)
            lexicons[lexicon_name] = []
        elif line.startswith("ALIAS "):
            raise NotImplementedError()
        else:
            # We must be in an existing section
            if current_section is None:
                raise LexdError("Definition must be part of a section!", line_number)
            mode, section_name = current_section
            if mode == "patterns":
                patterns[""].append((line, line_number))
            elif mode == "pattern":
                patterns[section_name].append((line, line_number))
            elif mode == "lexicon":
                lexicons[section_name].append((line, line_number))
    return patterns, lexicons


def _compile_lexicon(entry: str, line_number: int) -> FST:
    left_tokens, right_tokens = _tokenize_lex_entry(entry)
    # Regex
    if left_tokens[0] == "/" and left_tokens[-1] == "/":
        return FST.re("".join(left_tokens[1:-1]))
    left = " ".join("'" + token + "'" for token in left_tokens)
    right = " ".join("'" + token + "'" for token in right_tokens)
    return FST.re(f"({left}):({right})")


def _tokenize_lex_entry(entry: str):
    """Tokenize a lexicon entry into input and output symbol lists."""
    if entry.startswith("/") and entry.endswith("/"):
        return list(entry), list(entry)  # regex entry
    if ":" in entry:
        left, right = entry.split(":", 1)
    else:
        left, right = entry, entry
    in_syms = _split_symbols(left.strip())
    # empty right side means epsilon
    right = right.strip()
    out_syms = [] if right == "" else _split_symbols(right)
    # represent epsilon by [''] if no symbols
    if not out_syms:
        out_syms = [""]
    return in_syms, out_syms


def _split_symbols(s: str):
    """Split a string into multicharacter symbols (<...>) or single chars."""
    # This almost works, unless you
    chars: List[str] = re.findall(
        r"(?<!\\)<[^>]+(?<!\\)>|(?<!\\)\{[^\}]+(?<!\\)\}|.", s
    )
    chars = [char for char in chars if char != "\\"]
    return chars


def _topological_sort(
    patterns: Dict[str, List[Tuple[str, int]]], lexicon_names: Set[str]
):
    # Make quick lookups for the graph with a bidict
    pattern_to_downstream_map: Dict[str, Set[str]] = {
        pattern: set() for pattern in patterns
    }
    pattern_to_upstream_map: Dict[str, Set[str]] = {
        pattern: set() for pattern in patterns
    }
    for pattern_name in patterns:
        for pattern, _ in patterns[pattern_name]:
            # Refs to other patterns
            for match in re.finditer(r"\w+", string=pattern):
                if match.group(0) in lexicon_names:
                    continue
                pattern_to_downstream_map[match.group(0)].add(pattern_name)
                pattern_to_upstream_map[pattern_name].add(match.group(0))

    # Actually sort
    sorted_pattern_names = []
    # Set of nodes with no incoming edges (ie upstream deps)
    no_incoming_edges = {
        pattern_name
        for pattern_name, upstream_dependencies in pattern_to_upstream_map.items()
        if len(upstream_dependencies) == 0
    }
    while len(no_incoming_edges) > 0:
        current_node = no_incoming_edges.pop()
        sorted_pattern_names.append(current_node)
        for downstream in pattern_to_downstream_map[current_node]:
            # Delete edge
            pattern_to_upstream_map[downstream].remove(current_node)
            # If downstream has no other incoming edges now
            if len(pattern_to_upstream_map[downstream]) == 0:
                no_incoming_edges.add(downstream)
    # If graph still has edges, must be a cycle
    if sum(len(n) for n in pattern_to_upstream_map.values()) > 0:
        # TODO: Identify the nodes in the cycle
        raise LexdError("Cyclical dependency!", 0)
    return sorted_pattern_names


def _compile_pattern(pattern: str, fsts: Dict[str, FST], line_number: int) -> FST:
    # Check pattern names
    for match in re.finditer(r"\w+", string=pattern):
        if match.group(0) not in fsts:
            raise LexdError(f"Unrecognized reference: {match.group(0)}", line_number)

    # Compile inline lexicons
    inline_lexicon_fsts: Dict[str, FST] = {}

    def _handle_inline_lexicon(match: re.Match):
        name = f"INLINE_{len(inline_lexicon_fsts)}"
        inline_lexicon_fsts[name] = _compile_lexicon(match.group(1), line_number)
        return f"${name}"

    pattern = re.sub(r"\[([^\[\]]+)\]", _handle_inline_lexicon, pattern)

    # TODO: Handle the > operator

    # TODO: Handle tags

    # TODO: Handle one-sided lexicons

    # Replace lexicon names with pyfoma variables
    pattern = re.sub(r"\w+", lambda match: "$" + match.group(0), pattern)

    return FST.re(pattern, {**fsts, **inline_lexicon_fsts})


# Example of building a lexicon FST for testing
if __name__ == "__main__":
    example = """
PATTERN Inflection
VerbRoot VerbInfl
VerbRoot

PATTERNS
Inflection

LEXICON VerbRoot
sing
walk
dance

LEXICON VerbInfl
<prog>:ing
<pres><3p>:s
<inf>:
/(cat)+/
"""
    parse_lexd(example).render()
