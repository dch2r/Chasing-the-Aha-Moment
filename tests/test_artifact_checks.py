import unittest, tempfile, json
from pathlib import Path
from artifact_checks import verify_adapter, sha256


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name)
        for n in [
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]:
            (self.p / n).write_text("{}")
        (self.p / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "test", "bias": "none"})
        )
        self.protocol = {
            "adapter_sha256": sha256(self.p / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256(self.p / "adapter_config.json"),
            "tokenizer_sha256": sha256(self.p / "tokenizer.json"),
            "model": "test",
            "revision": "fixed",
        }

    def test_exact_match(self):
        self.assertTrue(verify_adapter(self.p, self.protocol)["verified"])

    def test_bad_weights(self):
        (self.p / "adapter_model.safetensors").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Fingerprint mismatch"):
            verify_adapter(self.p, self.protocol)

    def test_missing(self):
        (self.p / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify_adapter(self.p, self.protocol)

    def test_wrong_model(self):
        self.protocol["model"] = "other"
        with self.assertRaisesRegex(ValueError, "base model"):
            verify_adapter(self.p, self.protocol)

    def test_incomplete_tokenizer(self):
        (self.p / "tokenizer_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "entire adapter"):
            verify_adapter(self.p, self.protocol)


if __name__ == "__main__":
    unittest.main()
