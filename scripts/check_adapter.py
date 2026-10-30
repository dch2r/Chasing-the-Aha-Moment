import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artifact_checks import verify_adapter

p = argparse.ArgumentParser(
    description="Verify an externally retained adapter without loading a model."
)
p.add_argument("--adapter", required=True, type=Path)
a = p.parse_args()
try:
    print(
        json.dumps(
            verify_adapter(
                a.adapter,
                json.loads((ROOT / "results/evaluation/protocol.json").read_text()),
            ),
            indent=2,
        )
    )
except (ValueError, OSError) as e:
    p.exit(1, f"Adapter check failed: {e}\n")
