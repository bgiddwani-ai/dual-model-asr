#!/usr/bin/env python3
"""Dependency-free syntax checks for owned Python and shell source."""
import ast
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
count = 0
for folder in ["asr_stack", "scripts", "tests"]:
    for path in (root / folder).glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
        count += 1
for path in (root / "scripts").glob("*.sh"):
    subprocess.run(["bash", "-n", str(path)], check=True)
print(f"Syntax checks passed for {count} Python files and all shell entrypoints")
