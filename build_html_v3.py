#!/usr/bin/env python3
"""Assemble the self-contained v3 HTML report with base64-embedded figures."""
import base64
import pathlib

DIRS = [pathlib.Path("figures_v3"), pathlib.Path("figures_v2")]


def b64(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


IMG = {}
for d in DIRS:
    for f in sorted(d.glob("*.png")):
        IMG.setdefault(f.name, b64(f))   # v3 takes precedence on name clash

HTML = pathlib.Path("report_v3_template.html").read_text()
for name, uri in IMG.items():
    HTML = HTML.replace(f"__IMG_{name}__", uri)

out = pathlib.Path("Automation_Economy_Experiments_v3.html")
out.write_text(HTML)
print("written", out, f"({out.stat().st_size/1024:.0f} KB)")
import re
missing = sorted(set(re.findall(r"__IMG_([\w.]+)__", out.read_text())))
print("unreplaced placeholders:", missing or "none")
