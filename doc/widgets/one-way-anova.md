# One-Way ANOVA

Tests whether the means of a numeric variable differ across the levels of a categorical
variable, with optional post hoc comparisons.

## Signals

**Inputs**

- Data: a data table.

**Outputs**

- Test Results: one row, meta columns `Variable` and `Method` (`Classic` or `Welch`), numeric
  columns `F`, `df1`, `df2`, `p-value`, plus `η²` and `ω²` if enabled and, if
  "Include SS and MS" is on, `SS between`, `SS within`, `MS between`, `MS within`.
- Post Hoc: one row per pair, meta columns `Variable`, `Method` (e.g. `Tukey HSD`,
  `Bonferroni (CIs adjusted)`), `Group A`, `Group B`; numeric columns `mean difference`
  (A minus B), `CI low`, `CI high`, `adjusted p-value`. `None` if post hoc is `None` or
  the test is not valid.

## Controls

- **Test variable**: numeric; defaults to the class variable if numeric, else the first numeric.
- **Grouping by**: categorical; defaults to the class variable if categorical, else the first
  categorical.
- **Method**: Classic or Welch.
- **Post hoc**: Tukey HSD, Games-Howell, Bonferroni, None. Default is Tukey for Classic and
  Games-Howell for Welch; switching method moves the post hoc only while it still sits at
  the other method's default.
- **Alpha** (default 0.05; threshold for warnings) and **Confidence (%)** (default 95; post hoc
  CIs and the CIs in the plot).
- **Effect size**: η², ω² (both on by default). Both always use classic sums of squares,
  also under Welch.
- **Include SS and MS in Test Results** (off by default): affects the output only; the
  displayed Test table never has SS/MS columns.

## Tables

**Descriptives by group**: N, Mean, SD, SE per level (levels without observations are skipped).

**Test**: F, df1, df2, p, η², ω² (as enabled). Welch df2 is fractional.

**Post hoc**: Comparison (`A - B`), Mean dif., CI, Adjusted p. Under Bonferroni a caption
"Bonferroni-adjusted CIs" is shown. p-values below 0.001 are shown as `<0.001`.

## Plot

Group means with confidence-interval whiskers (one marker shape per group) and a dashed
line for the grand mean.

## Assumption checks and messages

Shown below the plot: Shapiro-Wilk on the residuals (observation minus group mean) and
Levene (median-centred).

- Warning: residuals not normal (Shapiro-Wilk p < alpha).
- Warning: variances differ (Levene p < alpha), only with the Classic method; suggests
  Welch with Games-Howell.
- Information: only 2 groups (the Independent Samples t-Test fits better); missing values dropped.
- Error: no numeric / no categorical variable; fewer than 2 groups with valid observations;
  `Test undefined: ...` (a group with fewer than 2 values or zero variance).

## Formulas

- SS between = sum n_i (mean_i - grand)^2; SS within = sum (n_i - 1) SD_i^2;
  MS = SS / df; classic F = MS between / MS within, df = (k-1, N-k).
- Welch F (Welch 1951) with weights w_i = n_i / SD_i^2 and fractional df2.
- η² = SS between / SS total; ω² = (SS between - (k-1) MS within) / (SS total + MS within).
- Tukey HSD: studentized range, pooled MS within (scipy).
- Games-Howell: studentized range with Welch-Satterthwaite df per pair; CI uses the
  studentized range at the chosen confidence.
- Bonferroni: pairwise t-tests (pooled MS within for Classic, Welch t-test for Welch),
  p multiplied by the number of comparisons and capped at 1. The CIs are
  Bonferroni-adjusted: level 1 - (1 - confidence) / m.

## Caveats

The plot CIs are unadjusted per-group CIs of each mean (one-sample t), not the post hoc CIs.
The Test table's headers use the symbols η² and ω²; the output uses the same names.
