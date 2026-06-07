#!/usr/bin/env python3
"""Assemble a self-contained HTML report with base64-embedded figures."""
import base64
import pathlib

FIG = pathlib.Path("figures_v2")


def b64(name):
    data = (FIG / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


IMG = {f.name: b64(f.name) for f in sorted(FIG.glob("*.png"))}

HTML = pathlib.Path("report_template.html").read_text()
for name, uri in IMG.items():
    HTML = HTML.replace(f"__IMG_{name}__", uri)

out = pathlib.Path("Automation_Economy_Experiments.html")
out.write_text(HTML)
print("written", out, f"({out.stat().st_size/1024:.0f} KB)")
remaining = [t for t in IMG if f"__IMG_{t}__" in out.read_text()]
print("unreplaced placeholders:", remaining or "none")
