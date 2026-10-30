import unittest, tempfile, json
from pathlib import Path
from evaluation_io import read_rows, append_row


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name) / "rows.jsonl"

    def test_round_trip(self):
        append_row(self.p, {"label": "base", "source_id": 1})
        self.assertEqual(len(read_rows(self.p, {1})), 1)

    def test_partial_recovery(self):
        append_row(self.p, {"label": "base", "source_id": 1})
        with self.p.open("ab") as f:
            f.write(b'{"label":')
        self.assertEqual(len(read_rows(self.p, {1})), 1)
        self.assertTrue(Path(str(self.p) + ".interrupted").exists())
        append_row(self.p, {"label": "tuned", "source_id": 1})
        self.assertEqual(len(read_rows(self.p, {1})), 2)

    def test_duplicate_rejected(self):
        for _ in range(2):
            append_row(self.p, {"label": "base", "source_id": 1})
        with self.assertRaises(ValueError):
            read_rows(self.p, {1})

    def test_unknown_rejected(self):
        append_row(self.p, {"label": "base", "source_id": 2})
        with self.assertRaises(ValueError):
            read_rows(self.p, {1})

    def test_no_final_newline(self):
        self.p.write_text(json.dumps({"label": "base", "source_id": 1}))
        read_rows(self.p, {1})
        self.assertTrue(self.p.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
