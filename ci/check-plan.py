"""Summarize a Terraform JSON plan from stdin and refuse deletions delivery must not make."""

import argparse
import json
import os
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("stack")
p.add_argument(
    "--replaceable",
    nargs="*",
    default=[],
    help="Resource types whose replacement is part of a normal release",
)
a = p.parse_args()
changes = [
    r
    for r in json.load(sys.stdin).get("resource_changes", [])
    if r["change"]["actions"] not in (["no-op"], ["read"])
]
rows = [f"| `{r['address']}` | {', '.join(r['change']['actions'])} |" for r in changes]
report = f"### Terraform `{a.stack}`\n\n" + (
    "| Resource | Actions |\n| --- | --- |\n" + "\n".join(rows)
    if rows
    else "No changes."
)
print(report)
if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(summary, "a") as f:
        f.write(report + "\n\n")
blocked = [
    r["address"]
    for r in changes
    if "delete" in r["change"]["actions"] and r["type"] not in a.replaceable
]
if blocked:
    sys.exit(
        "Refusing to delete or replace "
        + ", ".join(blocked)
        + "; apply this change from an infrastructure session"
    )
