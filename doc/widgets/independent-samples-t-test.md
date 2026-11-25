# Independent Samples t-Test

Tests whether the means of a numeric variable differ between two independent groups.
Always Welch's t-test (unequal variances); there is no Student option.

## Signals

**Inputs**

- Data: a data table.

**Outputs**

- Test Results: a single row with meta columns `Variable` and `Test` (always `Welch`) and the
  numeric columns `N1`, `N2`, `mean1`, `mean2`, `SD1`, `SD2`, `t`, `df`, `p-value`,
  `mean difference`, `CI low`, `CI high` and, if enabled, `Hedges' g`. `None` when the test
  is not valid.

## Controls

- **Test variable**: numeric variable. Defaults to the class variable if it is numeric,
  otherwise the first numeric variable.
- **Grouping by**: categorical variable. Defaults to the class variable if it is categorical,
  otherwise the first categorical variable. It must have exactly 2 levels.
  Group 1 is its first level, Group 2 its second (shown under the hypothesis buttons).
- **Alternative hypothesis**: Group 1 != Group 2, Group 1 > Group 2, Group 1 < Group 2.
- **Alpha**: threshold for the warning (default 0.05).
- **Confidence (%)**: level of the CI (default 95).
- **Report Hedges' g** (default on).

## Tables

**Descriptives by group**: `Group 1: <level>` and `Group 2: <level>` with N, Mean, SD, SE.

**Test**: t, df, p, Mean dif., CI, and Hedges' g if enabled. No name column. Mean dif. is
Group 1 minus Group 2; df is the Welch-Satterthwaite value. p-values below 0.001 are shown
as `<0.001`.

## Plot

One box plot per group: box = quartiles, line = median, diamond = mean, whiskers = range
(min to max).

## Assumption checks and messages

Shown under the plot: Shapiro-Wilk per group, and Levene (median-centred) for equal
variances, labelled information only (Welch does not assume equal variances).

- Warning: normality violated (Shapiro-Wilk p < alpha) in the named group(s). Advisory only.
- Shapiro-Wilk is skipped, with a note, for N < 3, N > 5000 or zero variance.
- Error: grouping variable has more than 2 levels / fewer than 2 levels.
- Error: no numeric variable / no categorical variable in the data.
- Error `Test undefined: ...` if a group has fewer than 2 valid values or zero variance.
- Information: missing values dropped (in the test variable within the two groups, and in the
  grouping variable).

## Formulas

- t = (mean1 - mean2) / sqrt(SD1^2/N1 + SD2^2/N2); df by Welch-Satterthwaite.
- CI of mean1 - mean2 from the t distribution with those df.
- Cohen's d = (mean1 - mean2) / SD_pooled, SD_pooled = sqrt(((N1-1)SD1^2 + (N2-1)SD2^2) / (N1+N2-2)).
- Hedges' g = d * (1 - 3 / (4 (N1 + N2 - 2) - 1)).
  Both effect sizes use the pooled SD even though the test is Welch.

## Caveats

For more than two groups use One-Way ANOVA. Rows with a missing test value are dropped
from their group.
