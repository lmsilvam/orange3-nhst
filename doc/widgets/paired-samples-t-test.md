# Paired Samples t-Test

Tests whether the mean difference between two numeric variables measured on the same rows
is zero.

## Signals

**Inputs**

- Data: a data table.

**Outputs**

- Test Results: a single row with meta column `Pair` (`<Variable 1> - <Variable 2>`) and the
  numeric columns `N`, `mean1`, `mean2`, `mean difference`, `SD difference`, `t`, `df`,
  `p-value`, `CI low`, `CI high` and, if enabled, `Cohen's dz`. `None` when the test is not valid.

## Controls

- **Variable 1**, **Variable 2**: numeric variables; default to the first two numeric
  variables. **Swap** exchanges them.
- **Alternative hypothesis**: difference != 0, > 0, < 0 (difference = Variable 1 - Variable 2).
- **Alpha**: threshold for the warning (default 0.05).
- **Confidence (%)**: level of the CI (default 95).
- **Report Cohen's dz** (default on).

## Tables

**Descriptives** (complete pairs only): `Variable 1: <name>`, `Variable 2: <name>` with
N, Mean, SD, SE, and a `Difference` row (mean, SD, SE of the differences).

**Test**: t, df, p, Mean dif., CI, and Cohen's dz if enabled. No name column.
p-values below 0.001 are shown as `<0.001`.

## Plot

Overlaid density curves (Gaussian kernel) of the two variables, with a dashed vertical line
at each mean. The test itself is on the differences.

## Assumption checks and messages

Shapiro-Wilk on the differences is shown below the plot.

- Warning: normality of the differences violated (p < alpha). Advisory only.
- Shapiro-Wilk is skipped, with a note, for N < 3, N > 5000 or zero variance.
- Error: at least two numeric variables are needed; Variable 1 and 2 must differ.
- Error `Test undefined: ...` for fewer than 2 complete pairs or constant differences.
- Information: pairs with missing values dropped (count).

## Formulas

- d_i = x1_i - x2_i; t = mean(d) / (SD(d) / sqrt(N)), df = N - 1.
- CI of the mean difference: mean(d) +- t_crit * SE(d).
- Cohen's dz = mean(d) / SD(d).

## Caveats

Pairs are dropped listwise (a missing value in either variable removes the row).
