# One Sample t-Test

Tests whether the mean of each selected numeric variable differs from a value mu0.

## Signals

**Inputs**

- Data: a data table.

**Outputs**

- Test Results: one row per tested variable (meta column `Variable`) with the numeric columns
  `mu0`, `N`, `mean`, `SD`, `SE`, `t`, `df`, `p-value`, `mean difference`, `CI low`, `CI high`
  and, if enabled, `Cohen's d`. The output has exactly the rows of the Test table. It is
  `None` when no test is valid. One-sided intervals have an open end, output as missing.

## Controls

- **Variables**: numeric variables (attributes, class, metas); multiple selection. The first
  numeric variable is selected by default.
- **Alternative hypothesis**: two-sided (mean != mu0), greater, less.
- **Alpha**: threshold for the assumption warning (default 0.05).
- **Confidence (%)**: level of the confidence interval (default 95).
- **Report Cohen's d**: adds the `Cohen's d` column (default on).
- Send automatically / Send.

## Tables

**Descriptives**: Variable, N, Mean, SD, SE.

**Test** (rows in the same order, no name column): mu0, t, df, p, Mean dif., CI, and
Cohen's d if enabled. Double-click a mu0 cell to edit it (default 0); the value is
remembered per variable. p-values below 0.001 are shown as `<0.001`. CI is the interval of
the mean difference (mean - mu0).

## Plot

Histogram of the selected Test row's variable, with mu0 as a dashed line and the mean
as a solid line. The shaded band is the confidence interval of the mean (mu0 + CI of the
difference; open one-sided ends are clipped to the data range). Below the plot: the
Shapiro-Wilk result.

## Assumption checks and messages

- Warning: normality violated, when Shapiro-Wilk p < alpha (variables are named). Advisory only.
- Shapiro-Wilk is skipped, with a note, when N < 3 or N > 5000, or when the variable is constant.
- Error `Data has no numeric variables.`
- Error `Test undefined, variable skipped: ...` for a variable with fewer than 2 valid values
  or zero variance; that variable is omitted from tables and output, the others still run.
- Information: missing values dropped (count per variable).

## Formulas

- t = (mean - mu0) / (SD / sqrt(N)), df = N - 1, SD with N - 1 denominator.
- CI of the mean difference: (mean - mu0) +- t_crit * SE.
- Cohen's d = (mean - mu0) / SD.

## Caveats

p-values and intervals assume approximate normality of the variable. Missing values are
dropped per variable.
