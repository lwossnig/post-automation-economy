const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, Header, Footer, TabStopType, TabStopPosition
} = require("docx");

const FIG = "figures_v2";
const CW = 9360;               // content width (US Letter, 1" margins) in DXA
const EMU = 9525;              // EMU per pixel at 96 dpi

// scale an image to a target display width in inches, preserving aspect ratio
function img(file, widthInches) {
  const dim = { // px dimensions captured from the files
    "exp1_scenarios.png": [1540, 990], "exp2_speed.png": [880, 550],
    "exp3_rg.png": [880, 550], "exp4_eps.png": [1430, 550],
    "exp5_foreign.png": [880, 550], "exp6_buyer.png": [880, 550],
    "exp7_frontier.png": [880, 660], "exp8_timing.png": [880, 550],
  }[file];
  const wPx = widthInches * 96;
  const hPx = wPx * dim[1] / dim[0];
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(`${FIG}/${file}`),
      transformation: { width: Math.round(wPx), height: Math.round(hPx) },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
  });
}

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

// paragraph from an array of {text, bold?, italics?, code?} runs or a plain string
function P(content, opts = {}) {
  const runs = (typeof content === "string" ? [{ text: content }] : content).map(r =>
    new TextRun({
      text: r.text, bold: r.bold, italics: r.italics,
      font: r.code ? "Consolas" : undefined,
      size: r.size,
    }));
  return new Paragraph({ spacing: { after: 140 }, children: runs, ...opts });
}

function bullet(content) {
  const runs = (typeof content === "string" ? [{ text: content }] : content).map(r =>
    new TextRun({ text: r.text, bold: r.bold, italics: r.italics, font: r.code ? "Consolas" : undefined }));
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs });
}

// simple table from header row + data rows (arrays of strings)
function table(header, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkCell = (txt, w, head) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: txt, bold: !!head, size: 18 })] })],
  });
  const headRow = new TableRow({ tableHeader: true, children: header.map((h, i) => mkCell(h, widths[i], true)) });
  const dataRows = rows.map(r => new TableRow({ children: r.map((c, i) => mkCell(String(c), widths[i], false)) }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths, rows: [headRow, ...dataRows] });
}

const spacer = () => new Paragraph({ spacing: { after: 80 }, children: [new TextRun("")] });

const children = [];

// ---------- Title ----------
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Simulating a Post-Automation Economy", bold: true, size: 40 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: "An agent-based, stock-flow-consistent model with endogenous production, ownership structure, and wealth condensation", size: 24, color: "444444" })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Experiment report (model v2)", italics: true, size: 20, color: "666666" })],
}));

// ---------- Overview ----------
children.push(H1("1  Overview"));
children.push(P("This report documents the experiments run on the v2 model of an automating economy, in which artificial intelligence and robotics progressively shift income from labour to capital, the ownership of that capital may sit with households, the state, or abroad, and the state may fund a universal basic income (UBI) through taxation. The central questions are whether a given taxation and ownership regime yields a stable outcome, who (if anyone) becomes insolvent, and how the distribution of wealth evolves."));
children.push(P("The model couples three layers: a stock-flow-consistent (SFC) accounting backbone that tracks every sector's balance sheet, a task-based constant-elasticity-of-substitution (CES) production block in which the capital share rises endogenously with automation, and a Bouchaud-Mezard kinetic engine that reproduces Pareto-tailed wealth condensation. Each experiment below tests one hypothesis; the design, the figure, the headline numbers, and the interpretation are given together."));
children.push(P([
  { text: "Reproducibility. ", bold: true },
  { text: "All results come from 12-seed Monte Carlo runs. Two accounting invariants (the four sectors' deposits sum to zero; the sum of sector net worths equals real assets) hold to machine precision in every run. Level quantities are reported as a ratio to per-capita output, which is scale-free, because the automation transition raises the absolute scale of the economy substantially." },
]));

// ---------- Prior and related work ----------
children.push(H1("2  Prior and related work"));
children.push(P("The model sits at the intersection of four literatures that have largely developed in isolation. The contribution is to combine them in one framework and use it for ownership and taxation experiments."));

children.push(H2("2.1  Automation and the factor distribution"));
children.push(P([
  { text: "The task-based macroeconomics of automation provides the production side. Acemoglu and Restrepo (2022) model tasks allocated between labour and capital, with automation expanding the set of tasks performed by capital and displacing the workers who held a comparative advantage in them; they attribute a large share of the post-1980 rise in US wage inequality to this task displacement. The CES task block used here is the reduced macro form of that mechanism, with the capital share rising endogenously as automation proceeds." },
]));
children.push(P([
  { text: "The closest analytic antecedent for the wealth dimension is Moll, Rachel and Restrepo (2022), who link automation not just to wages but to the personal income and wealth distributions, and isolate the mechanism by which automation raises the return to capital. In their theory top-tail inequality depends on a return gap that generalises Piketty's r minus g, and accumulation at the top is governed by a random growth process. The present model shares that random-growth-and-return-gap core but replaces the tractable, near-representative closed form with an explicit agent population, adds stock-flow-consistent balance sheets, and makes ownership location and policy the object of experiment rather than holding them fixed." },
]));

children.push(H2("2.2  Kinetic wealth condensation"));
children.push(P([
  { text: "The distributional engine comes from econophysics. Bouchaud and Mezard (2000) introduce a multiplicative exchange model whose stationary distribution has a Pareto tail and which undergoes a phase transition to condensation, noting that taxes and freer exchange reduce inequality. Burda and co-authors (2002) characterise the condensate in closed and open Pareto economies via a balls-in-boxes mechanism in which a finite fraction of wealth is amassed by a single agent, which is the source of the finite-population sensitivity of the very top share. Yakovenko and Rosser (2009) survey the broader statistical-mechanics-of-wealth programme." },
]));

children.push(H2("2.3  Stock-flow-consistent and agent-based macro"));
children.push(P([
  { text: "The accounting backbone follows the stock-flow-consistent tradition of Godley and Lavoie (2007), in which every flow has a source and a sink and every stock is one sector's asset and another's liability. Caiani and co-authors (2016) provide the benchmark agent-based stock-flow-consistent model that anchors the method, and the sfctools toolbox (Baldauf 2023) provides comparable open infrastructure." },
]));
children.push(P([
  { text: "The closest agent-based prior art is Carvalho and Di Guilmi (2019), a stock-flow-consistent agent-based model with endogenous credit and labour-saving technical change that studies technological unemployment and income inequality. The difference is one of object: their inequality emerges on the income and credit side, whereas the present model carries an explicit equity stock and drives concentration with a kinetic process on that stock, and treats the location of ownership (household, state, foreign) as a policy variable. Their model is, in turn, richer on endogenous credit, household debt and Minskian instability, none of which this version includes." },
]));

children.push(H2("2.4  Optimal taxation of automation"));
children.push(P([
  { text: "On the normative side, Guerreiro, Rebelo and Teles (2022) show in a quantitative model with endogenous skill choice that, under the current US tax system, a sustained fall in automation costs can produce a large rise in inequality, and that it is optimal to tax robots while the displaced routine workers remain in the labour force, with the optimal robot tax falling to zero once they retire. The present model is positive and experimental rather than optimal-tax, but its wealth-tax, robot-ownership and citizens'-fund experiments speak to the same policy question from a distributional-dynamics angle." },
]));

children.push(H2("2.5  Open economy and profit repatriation"));
children.push(P([
  { text: "The foreign-ownership dimension uses a rest-of-world balance-sheet sector, standard in open-economy stock-flow-consistent modelling. The empirical motivation is the scale of profit repatriation: Parnreiter, Steinwarder and Kolhoff (2024) estimate that transnational corporations repatriated about one trillion US dollars a year between 2005 and 2020, roughly 4.2% of the global stock of foreign direct investment, with net flows running centripetally towards a few high-income countries. This is the mechanism by which foreign ownership of automated capital erodes the domestic tax base while corporate tax still reaches the profit at source." },
]));

children.push(H2("2.6  Contribution"));
children.push(P([
  { text: "No existing work appears to combine, in a single stock-flow-consistent model used for policy experiments, all three of: automation-driven shifts in the factor distribution; an explicit ownership structure spanning households, the state, and foreign owners; and a kinetic wealth-condensation engine. Moll and co-authors come closest on mechanism but are near-representative and closed-economy, without an explicit ownership or foreign dimension or a citizens'-fund instrument. Carvalho and Di Guilmi come closest on method but focus on income and credit rather than the ownership of automated capital. That intersection is the contribution this prototype is built to explore." },
]));

// ---------- The model ----------
children.push(H1("3  The model"));

children.push(H2("3.1  Sectors and invariants"));
children.push(P("Four sectors carry balance sheets: households (a population of N agents), the firm or capital-owning sector, the government, and the rest of the world (RoW). Two asset classes circulate. Deposits are a liquid claim that, by double-entry construction, sum to zero across the four sectors. Equity represents claims on the firm's productive capital stock K, split at the outset between households, the state, and RoW."));
children.push(P("Because every flow is a matched transfer and every stock is one sector's asset and another's liability, two identities hold every period and are asserted in the test suite:"));
children.push(bullet([{ text: "the four sectors' deposit balances sum to zero;" }]));
children.push(bullet([{ text: "the sum of all sector net worths equals real assets (capital plus the firm's inventories of unsold output)." }]));

children.push(H2("3.2  Production and factor prices"));
children.push(P("Output is a CES aggregate of a labour-task bundle and a capital-task bundle:"));
children.push(P([{ text: "Y = A [ (1 - I)^(1/e) (A_L L)^((e-1)/e) + I^(1/e) (A_K K)^((e-1)/e) ]^(e/(e-1))", code: true, size: 20 }], { alignment: AlignmentType.CENTER }));
children.push(P([
  { text: "Here I in [0,1] is the automation index, interpreted as the share of tasks performed by capital, and e is the elasticity of substitution between the two bundles. Factor prices are marginal products, so the wage bill and capital income sum exactly to output (Euler's theorem). The capital share " },
  { text: "pi = capital_income / Y", code: true },
  { text: " therefore rises endogenously as automation raises I, rather than being imposed. The net return on capital is r = (capital_income / K) - depreciation, and the capital growth rate g is endogenous through investment. This makes the r-versus-g margin, central to fiscal sustainability, an output of the model rather than an assumption." },
]));
children.push(P([
  { text: "Investment is behavioural: firms invest to close the gap between the net return r and a required return, plus replacement of depreciation. This disciplines accumulation so capital does not run away when the marginal product is temporarily high. The no-automation baseline is a verified stationary equilibrium (r equals the required return and g is zero)." },
]));

children.push(H2("3.3  Distribution, ownership, and policy"));
children.push(P([
  { text: "Wealth condenses through two channels. Equity income (dividends and booked retained earnings) accrues pro-rata to equity already held, so wealth begets wealth. On top of this, a conservative Bouchaud-Mezard kernel reshuffles household equity each period, producing a Pareto tail with exponent alpha = 1 + 2J/sigma squared. An optional coupling (parameter kappa) lets the kinetic noise scale with the capital share, so automation can drive the " },
  { text: "speed", italics: true },
  { text: " of condensation, not just its level." },
]));
children.push(P("Policy instruments include a corporate tax, a flat or progressive personal income tax, a wealth tax (levied on the stock and split into an equity leg that transfers ownership to the state and a cash leg that is ordinary revenue), a per-capita UBI, and a government running cost. Ownership can be assigned to households, the state (whose dividends may be kept or rebated per capita as a citizens' fund), or RoW (whose dividends and retained earnings are repatriated abroad)."));

children.push(H2("3.4  Baseline calibration"));
children.push(table(
  ["Parameter", "Value", "Meaning"],
  [
    ["N", "2000", "number of household agents"],
    ["periods", "300", "simulation length"],
    ["eps (e)", "0.6", "elasticity of substitution between task bundles"],
    ["I_base", "0.50", "capital-task share before the AI ramp (gives pi about 0.23)"],
    ["auto_max", "0.45", "automation ramp adds up to this, so I rises 0.50 to about 0.95"],
    ["auto_start / auto_speed", "80 / 0.06", "logistic ramp midpoint and steepness"],
    ["depreciation", "0.05", "capital depreciation per period"],
    ["r_required", "0.04", "required net return; investment closes the gap to it"],
    ["div_payout", "0.6", "share of after-tax profit paid as dividends"],
    ["c_income / c_wealth", "0.80 / 0.03", "consumption out of income and out of net worth"],
    ["phi_equity", "0.5", "share of saving households try to place in equity"],
    ["bm_J / bm_sigma", "0.10 / 0.55", "kinetic mean-reversion and noise (tail alpha about 1.66)"],
    ["kappa", "0.0 (1.0 when coupled)", "coupling of kinetic noise to the capital share"],
    ["gov_cost / r_debt", "0.10 / 0.02", "government running cost; interest on balances"],
  ],
  [2600, 1700, 5060]
));
children.push(spacer());

// ---------- Methodology ----------
children.push(H1("4  Method"));
children.push(P("Each experiment fixes the baseline calibration and varies one lever (a tax instrument, an ownership share, the elasticity, the automation ceiling, or the timing of intervention). Outcomes are read at the end of the 300-period horizon, averaged over the final 40 periods for distributional metrics and taken as the terminal value for stocks, then averaged across 12 random seeds. The metrics are the wealth Gini, the top one per cent wealth share, the labour share of output, government net worth (scaled by output), and rest-of-world net worth (scaled by output, the measure of wealth leakage)."));

// ---------- Experiments ----------
children.push(H1("5  Experiments and results"));

// H1
children.push(H2("5.1  H1 (fiscal sustainability)"));
children.push(P([{ text: "Hypothesis. ", bold: true }, { text: "With endogenous r and g, a UBI funded purely by flow taxes (income and corporate) becomes fiscally fragile as automation proceeds, while stock instruments or state equity restore solvency." }]));
children.push(P([{ text: "Design. ", bold: true }, { text: "Experiment 1 runs all seven regimes over the transition and reads terminal government net worth off the balance sheet. Experiment 3 sweeps the required return (which shifts the steady-state interest rate) and compares the income-tax and wealth-tax regimes, recording mean r, mean g, and terminal government net worth." }]));
children.push(img("exp1_scenarios.png", 6.3));
children.push(caption("Figure 1. Scenario comparison over the automation transition: wealth Gini, top 1% share, government net worth relative to output, and labour share. Twelve-seed representative trajectories."));
children.push(img("exp3_rg.png", 4.6));
children.push(caption("Figure 2. Endogenous net return r and capital growth rate g over the transition: a capital-deepening boom (r and g rise) that relaxes back to the required return."));
children.push(P([{ text: "Results. ", bold: true }, { text: "In the lower-left panel of Figure 1, laissez-faire is the only regime whose government net worth turns and stays negative (about -111 times output), funding its running cost on debt at r greater than g. Every taxed regime accumulates a positive state asset base. Experiment 3 shows the wealth-tax regime gains most from a higher steady-state return: at a required return of 0.06 it reaches government net worth of 263 times output versus 176 for income tax, because the state's accumulated equity earns that return." }]));
children.push(P([{ text: "Interpretation. ", bold: true }, { text: "The hypothesis holds, with a refinement: aggregate insolvency is really a property of the untaxed economy, since any redistributive regime here is solvent. The sharper point is that the wealth tax is simultaneously the most equal and the most solvent regime, because it converts the high transitional return on capital into a durable state asset stream." }]));

// H2
children.push(H2("5.2  H2 (distribution and the speed of condensation)"));
children.push(P([{ text: "Hypothesis. ", bold: true }, { text: "Wealth condensation is governed by stock instruments and portfolio composition, not flow taxes; and once the kinetic noise is coupled to the capital share, the speed of condensation rises with automation." }]));
children.push(P([{ text: "Design. ", bold: true }, { text: "Experiment 1 compares Gini and top-share paths across regimes (Figure 1). Experiment 2 switches the coupling on (kappa = 1) and sweeps the automation ceiling, recording the terminal Gini and t_half, the time to reach the halfway point of the rise." }]));
children.push(img("exp2_speed.png", 4.6));
children.push(caption("Figure 3. Coupled model: deeper automation produces both higher and faster wealth condensation."));
children.push(P([{ text: "Results. ", bold: true }, { text: "In Figure 1 the laissez-faire, income-tax, state-ownership, citizens'-fund and foreign-ownership regimes all settle around a Gini of 0.53 to 0.58, while only the wealth tax (to 0.054) and progressive wealth tax (to 0.006) bend down. Figure 3 shows the speed result: as the automation ceiling rises, the terminal Gini climbs (0.60, 0.69, 0.74, 0.77) and t_half falls (93, 85, 75, 71)." }]));
children.push(P([{ text: "Interpretation. ", bold: true }, { text: "Both parts hold. The level result reproduces the earlier finding under endogenous output: concentration lives in the equity stock, which only a stock instrument reaches. The speed result is the new contribution: tying the concentration engine to the capital share makes automation a rate driver, so deeper automation brings inequality sooner as well as higher." }]));

// H3
children.push(H2("5.3  H3 (foreign ownership and the tax base)"));
children.push(P([{ text: "Hypothesis. ", bold: true }, { text: "Foreign ownership of the automated capital erodes the domestic personal-tax base while corporate tax still reaches the profit at source, shifting the fiscal frontier adversely." }]));
children.push(P([{ text: "Design. ", bold: true }, { text: "Experiment 5 sweeps the rest-of-world ownership share from 0 to 0.8, holding the tax regime fixed, and tracks both repatriated wealth (RoW net worth) and government net worth." }]));
children.push(img("exp5_foreign.png", 4.6));
children.push(caption("Figure 4. Foreign ownership: repatriated wealth rises and government net worth falls roughly linearly with the foreign share."));
children.push(P([{ text: "Results. ", bold: true }, { text: "Rest-of-world net worth rises almost linearly with the foreign share, from 0 to about 92 times output at an 80% foreign share, while government net worth falls in step from 164 to 140." }]));
children.push(P([{ text: "Interpretation. ", bold: true }, { text: "The hypothesis holds. The asymmetry is the point: dividends and retained earnings on foreign-held equity leave the country, shrinking the personal income and wealth bases, while corporate tax still reaches the profit. Foreign ownership is therefore a slow fiscal drain that personal-tax instruments cannot stop, which is the structural case for domestic-ownership requirements or source-based capital taxation." }]));

// H4
children.push(H2("5.4  H4 (the buyer-collapse channel) - a reversal"));
children.push(P([{ text: "Hypothesis. ", bold: true }, { text: "Once labour income vanishes under automation, households can no longer buy into the automated capital, so ownership can only broaden through non-market transfer." }]));
children.push(P([{ text: "Design. ", bold: true }, { text: "Experiment 6 switches on endogenous portfolio choice (households buy newly issued equity out of saving at book value) and tracks household equity purchases against the wage and net investment over the transition." }]));
children.push(img("exp6_buyer.png", 4.6));
children.push(caption("Figure 5. Equity purchases peak then collapse, but the wage per efficiency unit rises rather than falls: the collapse tracks the end of the investment boom, not affordability."));
children.push(P([{ text: "Results. ", bold: true }, { text: "Household equity purchases peak (around period 149) then collapse to zero. But the wage per efficiency unit rises over the transition rather than falling: with endogenous capital deepening the boom lifts labour in absolute terms even as the labour share falls. The purchase collapse is driven by net investment demand ending, not by households being priced out." }]));
children.push(P([{ text: "Interpretation. ", bold: true }, { text: "This is the hypothesis the model overturned, and it is the most instructive result. The original intuition (workers too poor to buy the robots) was an artefact of exogenous output. Once production and accumulation are endogenous, automation makes everyone absolutely richer, and wealth concentrates through the return channel (equity begets equity), not through an affordability barrier to entry. Endogenising production changed the answer, which is precisely why it was worth doing." }]));

// H5
children.push(H2("5.5  H5 (the policy frontier)"));
children.push(P([{ text: "Hypothesis. ", bold: true }, { text: "A citizens' wealth fund (the state holds equity as custodian and rebates its dividends per capita) dominates both income-tax UBI and pure nationalisation on a joint inequality-and-solvency criterion." }]));
children.push(P([{ text: "Design. ", bold: true }, { text: "Experiment 7 plots every regime in (Gini, government net worth) space, so dominance can be read directly." }]));
children.push(img("exp7_frontier.png", 4.4));
children.push(caption("Figure 6. Policy frontier: lower-left is more equal, higher is more solvent. The stock taxes occupy the desirable corner; laissez-faire is alone in the insolvent quadrant."));
children.push(P([{ text: "Results. ", bold: true }, { text: "The progressive wealth tax (Gini 0.006, government net worth 235 times output) and flat wealth tax (0.054, 212) occupy the desirable corner. Laissez-faire is alone in the insolvent quadrant. The citizens' fund lands at (0.530, 60): lower inequality than state-keep (0.536), but the lowest solvency of the taxed regimes because it pays its dividends back out." }]));
children.push(P([{ text: "Interpretation. ", bold: true }, { text: "Only partly supported. The fund beats income-tax UBI on inequality and beats laissez-faire on both axes, but it does not dominate the wealth-tax regimes and explicitly trades solvency for equality. The genuinely dominant policies are the stock taxes; the fund is best read as a distributional instrument that keeps the state lean rather than a free lunch on both axes." }]));

// H6
children.push(H2("5.6  H6 (intervention timing and hysteresis)"));
children.push(P([{ text: "Hypothesis. ", bold: true }, { text: "Condensation exhibits hysteresis, so intervening late is materially more costly than intervening early, for the same final policy." }]));
children.push(P([{ text: "Design. ", bold: true }, { text: "Experiment 8 runs the coupled model and switches on a 3% wealth tax at four onset periods (60, 110, 160, 210), then compares terminal Ginis." }]));
children.push(img("exp8_timing.png", 4.6));
children.push(caption("Figure 7. The later the wealth tax is introduced, the higher the peak inequality reached and the higher the terminal Gini; the latest case does not fully recover within the horizon."));
children.push(P([{ text: "Results. ", bold: true }, { text: "Terminal Gini rises sharply with delay: 0.006 (onset 60), 0.022 (110), 0.081 (160), 0.281 (210). The later the intervention, the higher the peak the Gini reaches before the tax bites and the longer the clawback." }]));
children.push(P([{ text: "Interpretation. ", bold: true }, { text: "The hypothesis holds strongly. Concentrated equity that has already compounded is harder to redistribute than equity caught before it concentrates, so the cost of waiting is convex. Timing a wealth tax relative to the automation wave matters as much as its rate, an argument for acting before condensation rather than after." }]));

// Robustness
children.push(H2("5.7  Robustness (finite size)"));
children.push(P("A finite-size sweep (experiment 9) confirms the Gini is converged from N = 250 (0.550) to N = 4000 (0.551), while the sampling noise in the top one per cent share falls from 7.1 to 1.9 points as N rises. N = 2000 is therefore adequate for the aggregate metrics, with seed replication handling the residual tail noise."));
children.push(table(
  ["N", "Gini (mean)", "Gini (sd)", "top 1% (mean)", "top 1% (sd)"],
  [
    ["250", "0.550", "0.033", "19.2%", "7.11"],
    ["500", "0.547", "0.020", "18.0%", "4.35"],
    ["1000", "0.546", "0.017", "18.2%", "4.05"],
    ["2000", "0.543", "0.010", "17.5%", "2.13"],
    ["4000", "0.551", "0.009", "19.5%", "1.85"],
  ],
  [1500, 1965, 1965, 1965, 1965]
));
children.push(spacer());

// ---------- Summary ----------
children.push(H1("6  Summary of findings"));
children.push(table(
  ["Hypothesis", "Verdict", "Key result"],
  [
    ["H1 fiscal sustainability", "Supported (refined)", "Only the untaxed economy is insolvent; the wealth tax is both most equal and most solvent."],
    ["H2 distribution and speed", "Supported", "Flow taxes leave the Gini near 0.57; stock taxes collapse it; coupling makes condensation faster under deeper automation."],
    ["H3 foreign ownership", "Supported", "Leakage and fiscal erosion rise linearly with the foreign share; corporate tax alone reaches the profit."],
    ["H4 buyer collapse", "Overturned", "Endogenous capital deepening raises wages absolutely, so concentration is via returns, not an affordability barrier."],
    ["H5 citizens' fund dominance", "Partly supported", "The fund helps but does not dominate the stock taxes; it trades solvency for equality."],
    ["H6 timing and hysteresis", "Supported strongly", "Terminal Gini rises from 0.006 to 0.281 as intervention is delayed; the cost of waiting is convex."],
  ],
  [2500, 1900, 4960]
));
children.push(spacer());

children.push(H1("7  Caveats"));
children.push(bullet("Transition magnitudes are large (per-capita output rises about 250-fold) because gross capital deepening compounds over a long horizon; all headline results are reported in shares, ratios and Gini, which are scale-free."));
children.push(bullet("Equity is valued at book (no market price or capital gains); a market-clearing price is a planned extension."));
children.push(bullet("The kinetic concentration channel is coupled to the economy only through the parameter kappa; a fully endogenous return-dispersion process is the natural next modelling step."));
children.push(bullet("A formal eigenvalue stability analysis of the deterministic skeleton is specified but not yet implemented; stability is currently read off trajectories and the verified stationary baseline."));

// ---------- References ----------
children.push(H1("8  References"));
const refs = [
  "Acemoglu, D. and Restrepo, P. (2022). Tasks, Automation, and the Rise in U.S. Wage Inequality. Econometrica, 90(5), 1973-2016.",
  "Baldauf, T. (2023). sfctools: A toolbox for stock-flow consistent, agent-based models. Journal of Open Source Software, 8(87), 4980.",
  "Bouchaud, J.-P. and Mezard, M. (2000). Wealth condensation in a simple model of economy. Physica A, 282, 536-545.",
  "Burda, Z., Johnston, D., Jurkiewicz, J., Kaminski, M., Nowak, M. A., Papp, G. and Zahed, I. (2002). Wealth condensation in Pareto macroeconomies. Physical Review E, 65, 026102.",
  "Caiani, A., Godin, A., Caverzasi, E., Gallegati, M., Kinsella, S. and Stiglitz, J. E. (2016). Agent based-stock flow consistent macroeconomics: Towards a benchmark model. Journal of Economic Dynamics and Control, 69, 375-408.",
  "Carvalho, L. and Di Guilmi, C. (2019). Technological unemployment and income inequality: a stock-flow consistent agent-based approach. Journal of Evolutionary Economics, 29, 39-73.",
  "Godley, W. and Lavoie, M. (2007). Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth. Palgrave Macmillan.",
  "Guerreiro, J., Rebelo, S. and Teles, P. (2022). Should Robots Be Taxed? The Review of Economic Studies, 89(1), 279-311.",
  "Moll, B., Rachel, L. and Restrepo, P. (2022). Uneven Growth: Automation's Impact on Income and Wealth Inequality. Econometrica, 90(6), 2645-2683.",
  "Parnreiter, C., Steinwarder, L. and Kolhoff, K. (2024). Uneven Development through Profit Repatriation: How Capitalism's Class and Geographical Antagonisms Intertwine. Antipode, 56(6), 2343-2367.",
  "Yakovenko, V. M. and Rosser, J. B. (2009). Colloquium: Statistical Mechanics of Money, Wealth, and Income. Reviews of Modern Physics, 81, 1703-1725.",
];
refs.forEach(r => children.push(new Paragraph({
  spacing: { after: 100 },
  indent: { left: 360, hanging: 360 },
  children: [new TextRun({ text: r, size: 20 })],
})));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "44546A" },
        paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Post-automation economy model v2  |  page ", size: 16, color: "888888" }),
                   new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Automation_Economy_Experiments.docx", buffer);
  console.log("written Automation_Economy_Experiments.docx");
});
