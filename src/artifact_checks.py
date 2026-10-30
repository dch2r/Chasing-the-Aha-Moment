"""Validate the exact adapter/tokenizer used by the recorded evaluation."""

import hashlib, json
from pathlib import Path

FILES = {
    "adapter_model.safetensors": "adapter_sha256",
    "adapter_config.json": "adapter_config_sha256",
    "tokenizer.json": "tokenizer_sha256",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def verify_adapter(directory, protocol):
    directory = Path(directory)
    errors = []
    for name, key in FILES.items():
        p = directory / name
        if not p.is_file():
            errors.append(f"Missing {name}")
        elif sha256(p) != protocol[key]:
            errors.append(f"Fingerprint mismatch: {name}")
    if errors:
        raise ValueError("; ".join(errors))
    config = json.loads((directory / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != protocol["model"]:
        raise ValueError("Adapter base model does not match protocol")
    if config.get("bias", "none") != "none" or config.get("modules_to_save"):
        raise ValueError(
            "Expected LoRA-only adapter without additional saved base modules"
        )
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError(
            "Missing tokenizer_config.json; copy the entire adapter folder"
        )
    return {
        "verified": True,
        "model": protocol["model"],
        "revision": protocol["revision"],
    }
