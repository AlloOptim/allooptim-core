#!/usr/bin/env python3
"""Check documentation coverage for AlloOptim."""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def check_module(filepath: Path) -> Tuple[int, int, List[str]]:
    """Check docstring coverage in a Python file."""
    with open(filepath, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    total = 0
    documented = 0
    missing = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            # Skip private unless it's __init__
            if node.name.startswith("_") and node.name != "__init__":
                continue

            total += 1
            if ast.get_docstring(node):
                documented += 1
            else:
                missing.append(f"{filepath}::{node.name}")

    return total, documented, missing


def main():
    """Run coverage check."""
    root = Path("allooptim")
    all_missing = []
    total_items = 0
    total_documented = 0

    for pyfile in root.rglob("*.py"):
        if "__pycache__" in str(pyfile):
            continue

        total, documented, missing = check_module(pyfile)
        total_items += total
        total_documented += documented
        all_missing.extend(missing)

    coverage = (total_documented / total_items * 100) if total_items > 0 else 0

    print(f"\n📊 Documentation Coverage: {coverage:.1f}%")
    print(f"   Documented: {total_documented}/{total_items}")

    if all_missing:
        print(f"\n⚠️  Missing docstrings ({len(all_missing)}):")
        for item in sorted(all_missing)[:20]:  # Show first 20
            print(f"   - {item}")
        if len(all_missing) > 20:
            print(f"   ... and {len(all_missing) - 20} more")

    # Fail if coverage below threshold
    if coverage < 60:
        print("\n❌ Coverage below 60% threshold")
        sys.exit(1)
    else:
        print("\n✅ Documentation coverage acceptable")


if __name__ == "__main__":
    main()
