"""Run every packaged CPU evidence check without training or downloading data."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    if sys.flags.optimize:
        raise SystemExit("Run without -O; original task checks require assertions.")
    checks = [
        ("package_and_task1", ROOT, ["verify_package.py"]),
        ("task2", ROOT / "ATML_PA3_Task2", ["reproduce.py", "--verify-only"]),
        ("task3", ROOT / "ATML_PA3_Task3", ["verify_results.py"]),
    ]
    results = {}
    for label, folder, args in checks:
        completed = subprocess.run(
            [sys.executable, "-B", *args], cwd=folder, text=True, capture_output=True)
        if completed.returncode:
            sys.stderr.write(completed.stdout + completed.stderr)
            raise SystemExit(f"{label} failed (exit {completed.returncode}).")
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        results[label] = json.loads(completed.stdout)
        if results[label].get("status", "").lower() not in {"pass", "passed"}:
            raise SystemExit(f"{label} did not report success.")
    print(json.dumps({"status": "PASS", "checks": results, "GPU_used": False}, indent=2))

if __name__ == "__main__":
    main()
