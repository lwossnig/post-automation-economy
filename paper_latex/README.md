# Simulating a Post-Automation Economy — manuscript package

Two compiled manuscripts plus their LaTeX sources and figures.

| File | What it is |
|---|---|
| `Simulating_a_Post-Automation_Economy__arXiv_full.pdf` | Full preprint: sections 1–10, references, Appendix A, all 28 figures (11pt, single-spaced). |
| `Simulating_a_Post-Automation_Economy__journal_submission.pdf` | Journal-submission format: 12pt, 1.5-spaced, title page with abstract/keywords/JEL. Lean main text; *Further results* (§7) and *Assumptions and limitations* (Appendix A) are relocated to clearly-marked Supplementary Material. |
| `arxiv_main.tex`, `journal_main.tex` | Self-contained LaTeX sources (no `\input`s). |
| `figures_v3/` | All figures referenced by both documents. |

## Build

```
pdflatex arxiv_main.tex      # or: latexmk -pdf arxiv_main.tex
pdflatex journal_main.tex
```

Only base TeX Live packages are used (article class, graphicx, booktabs, longtable, amsmath/amssymb, geometry, hyperref, caption, setspace, ragged2e). No external bibliography tool: references are typeset directly (the paper uses author–year text, not `\cite`).

## Recommended journal

**Primary: *Journal of Economic Dynamics and Control* (JEDC, Elsevier).** It is the natural home for computational and agent-based macro models with a policy question, and it published the benchmark AB-SFC model the paper builds on (Caiani, Godin, Caverzasi, Gallegati, Kinsella & Stiglitz, 2016). The extensive validation machinery in the paper (global Sobol sensitivity, a formal stability analysis, Monte Carlo seed replication) matches its quantitative expectations.

**Strong alternative: *Journal of Evolutionary Economics* (JEE, Springer).** It published the closest single predecessor to this paper (Carvalho & Di Guilmi, 2020 — technological unemployment and inequality in an AB-SFC model), and it is more receptive to the policy/distribution narrative and the conditional, mechanism-over-forecast framing. Choose JEE if reviewers want less emphasis on the computational method and more on the evolutionary/institutional argument.

**Fallbacks:** *Economic Modelling* and *Structural Change and Economic Dynamics* (both Elsevier, policy- and structural-change-oriented; both take simulation work).

Notes on format:
- Both JEDC and JEE impose **no hard length limit**, so the full manuscript is submittable; the journal/supplement split is offered as a leaner option, not a requirement.
- Elsevier's "your paper, your way" policy means initial submissions accept any clean, consistently formatted manuscript, and arXiv accepts the `article` class, so both PDFs use `article`. The Elsevier house class (`elsarticle`) or the Springer template can be applied at the revision stage.
- The author block is a **placeholder** (`[Author name(s) and affiliation to be completed]`) — fill in before submission.

## Reference audit (completed)

Every reference was checked against the published record and **all 26 exist**. Corrections applied to the source:
- **Vivalt et al. (2024), NBER WP 32719** — restored the omitted sixth author, Patrick Krause.
- Added four works that were cited in the text but missing from the list: **Kaldor (1957)**, **Pasinetti (1962)**, **ITIF (2025)**, and the **Gates/Quartz (2017)** robot-tax interview.
- **ITIF (2025)** in-text framing corrected: ITIF documents that digital-services taxes fall on gross revenue but *opposes* them, so it is now cited as a critic of the gross-revenue base rather than as endorsing the rationale.
- Confirmed Brülhart et al. (2022) is cited (an earlier scan missed the `ü` HTML entity), so it is not an orphan.

Spot-verified bibliographic details (journal, volume, pages, year) for the higher-risk entries: Parnreiter et al. (Antipode 56(6):2343–2367), Foster–Haltiwanger–Tuttle (NBER 30491), IMF SDN 2024/002, Jakobsen et al. (NBER 32153, forthcoming AER), Carvalho & Di Guilmi (J Evol Econ 30(1):39–73), Caiani et al. (JEDC 69:375–408), Moll–Rachel–Restrepo (Econometrica 90(6):2645–2683), Oberfield–Raval (Econometrica 89(2):703–732), Guerreiro–Rebelo–Teles (REStud 89(1):279–311), Brülhart et al. (AEJ:Policy 14(4):111–150), the Acemoglu–Restrepo trio (JEP 33(2); JPE 128(6); Econometrica 90(5)), and Kaldor/Pasinetti.
