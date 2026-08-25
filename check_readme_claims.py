#!/usr/bin/env python3
"""
check_readme_claims.py -- run from the repo root before committing.

Verifies every rot-prone number in README.md against the actual repo state:
  * [FIX]-tag counts in the verify scripts
  * headline check counts printed by verify_all.py / verify_threepair.py
  * pytest test count
  * existence of every module named in the "What is where" table
  * G(0.9) printed by depolarizing_expansion.py

Prints a PASS/CHECK line per item. Nothing here modifies files; it only reports.
Exit code is nonzero if any hard check fails, so CI can gate on it if you like.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
fail = 0


def report(label, ok, detail=""):
    global fail
    tag = "PASS " if ok else "CHECK"
    if not ok:
        fail += 1
    print(f"[{tag}] {label}" + (f"  --  {detail}" if detail else ""))


def count_tag(path, tag=r"\[FIX\]"):
    p = ROOT / path
    if not p.exists():
        return None
    return len(re.findall(tag, p.read_text()))


# 1. [FIX] tag counts vs the README sentence
n_all = count_tag("verify_all.py")
n_three = count_tag("verify_threepair.py")
report("verify_all.py [FIX] tags == 3", n_all == 3, f"found {n_all}")
report("verify_threepair.py [FIX] tags == 1", n_three == 1, f"found {n_three}")

# whatever the README claims, surface it so you can compare by eye
m = re.search(r"items\s+([\d\-]+)\s+in\s+`verify_all\.py`,\s+item\s+(\d+)", README)
if m:
    print(f"        README currently says: items {m.group(1)} in verify_all.py, "
          f"item {m.group(2)} in verify_threepair.py")

# 2. modules named in the What-is-where table actually exist
for mod in ["paulis", "clifford", "circuits", "susceptibility",
            "classification", "closed_forms"]:
    p = ROOT / "finite_rank_purification" / f"{mod}.py"
    report(f"module finite_rank_purification/{mod}.py exists", p.exists())
for f in ["master.py", "general_n.py", "validate.py", "threepair.py",
          "verify_all.py", "verify_threepair.py", "make_figures.py", "make_fig8.py"]:
    report(f"{f} exists", (ROOT / f).exists())
report("figures/fig8_fourpair.pdf exists", (ROOT / "figures" / "fig8_fourpair.pdf").exists())

# 3. headline check counts -- only if you pass --run (these are slow)
if "--run" in sys.argv:
    for script, claim in [("verify_all.py", 91), ("verify_threepair.py", 38)]:
        try:
            out = subprocess.run([sys.executable, script], cwd=ROOT,
                                 capture_output=True, text=True, timeout=1800)
            m = re.search(r"All\s+(\d+)\s+checks passed", out.stdout)
            got = int(m.group(1)) if m else None
            report(f"{script} prints {claim} checks", got == claim,
                   f"printed {got}; rc={out.returncode}")
        except Exception as e:
            report(f"{script} runs", False, str(e))
    # pytest count
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                             cwd=ROOT, capture_output=True, text=True, timeout=600)
        m = re.search(r"(\d+)\s+passed", out.stdout)
        got = int(m.group(1)) if m else None
        readme_m = re.search(r"(\d+)\s+unit tests", README)
        claim = int(readme_m.group(1)) if readme_m else None
        report(f"pytest count matches README ({claim})", got == claim,
               f"pytest reports {got} passed")
    except Exception as e:
        report("pytest runs", False, str(e))
    # G(0.9)
    try:
        out = subprocess.run([sys.executable, "depolarizing_expansion.py"], cwd=ROOT,
                             capture_output=True, text=True, timeout=600)
        report("depolarizing_expansion prints G(0.9)=1.931906",
               "1.931906" in out.stdout)
    except Exception as e:
        report("depolarizing_expansion runs", False, str(e))
else:
    print("\n(Skipping slow script runs. Re-run with  --run  to execute "
          "verify_all.py, verify_threepair.py, pytest, and depolarizing_expansion.py.)")

print(f"\n{'OK: all fast checks passed' if fail == 0 else f'{fail} item(s) need attention'}")
sys.exit(1 if fail else 0)
