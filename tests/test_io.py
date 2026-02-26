"""Tests for FST I/O: pickle, foma format, AT&T format, dict/JSON export."""

import json
import pathlib
import re
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

from pyfoma.fst import FST


class TestPickle(unittest.TestCase):
    """Test FST.save and FST.load (pickle format)."""

    def test_roundtrip(self):
        """save/load preserves FST behaviour."""
        fst = FST.re("(cat | dog) s?")
        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test")
            fst.save(path)
            loaded = FST.load(path)
        self.assertEqual(set(fst.generate("cat")), set(loaded.generate("cat")))
        self.assertEqual(set(fst.generate("dogs")), set(loaded.generate("dogs")))
        self.assertEqual(list(fst.generate("bird")), list(loaded.generate("bird")))

    def test_extension_added_automatically(self):
        """save/load work without an explicit .fst extension."""
        fst = FST.re("a b c")
        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "noext")
            fst.save(path)
            self.assertTrue((Path(tmpdir) / "noext.fst").exists())
            loaded = FST.load(path)
        self.assertEqual(list(loaded.generate("abc")), ["abc"])


class TestFomaFormat(unittest.TestCase):
    """Test to_fomastring / from_fomastring / save_foma / load_foma."""

    def _roundtrip(self, fst):
        return FST.from_fomastring(fst.to_fomastring())

    def test_simple_acceptor(self):
        fst = FST.re("a b c")
        fst2 = self._roundtrip(fst)
        self.assertEqual(list(fst.generate("abc")), list(fst2.generate("abc")))
        self.assertEqual(list(fst.generate("ab")), list(fst2.generate("ab")))

    def test_transducer(self):
        fst = FST.re("(cat):(dog)")
        fst2 = self._roundtrip(fst)
        self.assertEqual(list(fst2.generate("cat")), ["dog"])

    def test_kleene_star(self):
        fst = FST.re("a*")
        fst2 = self._roundtrip(fst)
        self.assertEqual(list(fst.generate("")), list(fst2.generate("")))
        self.assertEqual(list(fst.generate("aaa")), list(fst2.generate("aaa")))

    def test_weighted(self):
        fst = FST.re("a<2.0> b<3.0>")
        fst2 = self._roundtrip(fst)
        self.assertEqual(
            list(fst.generate("ab", weights=True)),
            list(fst2.generate("ab", weights=True)),
        )

    def test_fstname_embedded(self):
        fst = FST.re("a b")
        self.assertIn("myFST", fst.to_fomastring(fstname="myFST"))

    def test_save_load_foma_file(self):
        """save_foma / load_foma roundtrip a dict of named FSTs."""
        fst1 = FST.re("cat | dog")
        fst2 = FST.re("(cat):(dog)")
        with TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.foma")
            FST.save_foma({"animals": fst1, "xducer": fst2}, path)
            loaded = FST.load_foma(path)
        self.assertIn("animals", loaded)
        self.assertIn("xducer", loaded)
        self.assertEqual(list(loaded["animals"].generate("cat")), ["cat"])
        self.assertEqual(list(loaded["animals"].generate("dog")), ["dog"])
        self.assertEqual(list(loaded["xducer"].generate("cat")), ["dog"])


class TestATTFormat(unittest.TestCase):
    """Test save_att and __str__ (AT&T format)."""

    @classmethod
    def setUpClass(cls):
        cls.fst = FST.regex(r"'[NO\'UN]' '[VERB]'<1> (cat):(dog)? 'ROTFLMAO🤣'")

    def _verify_att(self, att, epsilon="@0@"):
        self.assertIn("[NO'UN]\t[NO'UN]", att)
        self.assertIn("[VERB]\t[VERB]", att)
        self.assertIn(f"c\t{epsilon}", att)
        self.assertIn(f"a\t{epsilon}", att)
        self.assertIn(f"t\t{epsilon}", att)
        self.assertIn("c\td", att)
        self.assertIn("a\to", att)
        self.assertIn("t\tg", att)

    def test_str_att_format(self):
        """__str__ returns a valid AT&T format string."""
        self._verify_att(str(self.fst))

    def test_save_att_files_created(self):
        """save_att creates the transition file plus .isyms and .osyms."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.fst.save_att(path / "test")
            self.assertTrue((path / "test").exists())
            self.assertTrue((path / "test.isyms").exists())
            self.assertTrue((path / "test.osyms").exists())

    def test_save_att_custom_epsilon(self):
        """save_att supports a custom epsilon symbol."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            self.fst.save_att(path / "test.att", epsilon="<eps>")
            with open(path / "test.att") as f:
                self._verify_att(f.read(), epsilon="<eps>")

    def test_save_att_state_symbols(self):
        """save_att with state_symbols=True creates a .ssyms file."""
        f = FST.rlg(
            {"Root": [("", "Sublex")], "Sublex": [(("foo", "bar"), "#")]}, "Root"
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            f.save_att(path / "test_st.fst", state_symbols=True)
            self.assertTrue((path / "test_st.ssyms").exists())
            with open(path / "test_st.fst") as fh:
                att = fh.read()
            self.assertIn("Root\tSublex\t@0@\t@0@", att)
            self.assertIn("#\n", att)


class TestDictJSON(unittest.TestCase):
    """Test todict, fromdict, and tojs."""

    _RX = r"""
        $^rewrite(s:(s | 'š')
            | c:(c | 'č')
            | \?:Ɂ
            | 7:Ɂ
            | ʔ:Ɂ)
        """

    def test_todict_deterministic(self):
        """Two FSTs compiled from the same regex have identical dict representations."""
        fst1 = FST.regex(self._RX)
        fst2 = FST.regex(self._RX)
        self.assertEqual(json.dumps(fst1.todict()), json.dumps(fst2.todict()))

    def test_fromdict_roundtrip(self):
        """fromdict(todict()) reproduces the same FST."""
        fst1 = FST.regex(self._RX)
        fst2 = FST.fromdict(fst1.todict())
        self.assertEqual(json.dumps(fst1.todict()), json.dumps(fst2.todict()))

    def test_tojs_structure(self):
        """tojs produces valid JavaScript whose structure matches todict."""
        fst = FST.regex(r"'[NO\'UN]' '[VERB]'<1> (cat):(dog)? 'ROTFLMAO🤣'")
        d = fst.todict()
        self.assertIn(0, d["transitions"])
        self.assertEqual(1, len(d["finals"]))
        js = fst.tojs()
        self.assertIn('"maxlen": 10', js)
        self.assertIn(""""0|[NO'UN]": [{""", js)
        js_dict = re.sub(r"^var \w+ = (.*);", r"\1", js)
        js_json = json.loads(js_dict)
        self.assertIn(str(next(iter(d["finals"]))), js_json["f"])
        self.assertEqual(d["alphabet"], js_json["s"])
        for src, out in d["transitions"].items():
            for label, arcs in out.items():
                syms = re.split(r"(?<!\\)\|", label)
                self.assertIn(f"{src}|{syms[0]}", js_json["t"])
                for arc in arcs:
                    self.assertIn(
                        {str(arc): syms[-1]}, js_json["t"][f"{src}|{syms[0]}"]
                    )
        with open(pathlib.Path(__file__).parent / "test_foma.json", "rt") as infh:
            foma_json = json.load(infh)
            self.assertEqual(js_json["s"].keys(), foma_json["s"].keys())
            self.assertEqual(10, js_json["maxlen"])
            Q = deque([(0, 0)])
            while Q:
                src, jsrc = Q.popleft()
                if src in d["finals"]:
                    self.assertIn(jsrc, foma_json["f"])
                if src not in d["transitions"]:
                    continue
                for label in d["transitions"][src]:
                    syms = re.split(r"(?<!\\)\|", label)
                    for arc in d["transitions"][src][label]:
                        for jarc in foma_json["t"][f"{jsrc}|{syms[0]}"]:
                            for jdst, jsym in jarc.items():
                                if jsym != syms[-1]:
                                    continue
                                Q.append((arc, jdst))


if __name__ == "__main__":
    unittest.main()
