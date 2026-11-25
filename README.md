# Orange3-NHST

Yes I know [Orange Data Mining] (https://orangedatamining.com) frowns upon classical null-hypothesis significance tests but I need them for teaching because I love Orange, so here's a first version of NHST widgets. Yes I used Claude Code of course.

So far we have: 

| Widget | Test |
|---|---|
| One Sample t-Test | mean of one or more variables against a value mu0 |
| Independent Samples t-Test | Welch t-test between the two levels of a grouping variable |
| Paired Samples t-Test | mean difference between two numeric variables |
| One-Way ANOVA | classic or Welch ANOVA with post hoc comparisons |

Widgets show descriptives, test table, plot and assumption checks, and
outputs the results as a data table. Help pages: [doc/widgets](doc/widgets).

## Requirements

- Orange3 >= 3.41
- Python >= 3.11
- numpy, scipy (no other dependencies)

When Orange is installed with pip, a Qt binding is also needed, e.g.
`pip install PyQt6 PyQt6-WebEngine`.

## Installation

From source:

```
pip install -e .
```

From PyPI (not yet published):

```
pip install Orange3-NHST
```

## Example workflow

File -> Select Columns -> (One Sample t-Test | Independent Samples t-Test |
Paired Samples t-Test | One-Way ANOVA) -> Data Table

Connect the widget's *Test Results* output (and *Post Hoc* for One-Way ANOVA) to a Data Table.

## Running the tests

```
pip install pytest
pytest
```

In headless environments set `QT_QPA_PLATFORM=offscreen` first.

## Scope

In scope: one-sample, independent (Welch) and paired t-tests; classic and Welch
one-way ANOVA with Tukey HSD, Games-Howell and Bonferroni post hoc tests; Shapiro-Wilk
and Levene checks; Cohen's d, Hedges' g, Cohen's dz, eta squared, omega squared.

Out of scope: nonparametric alternatives (Mann-Whitney, Wilcoxon, Kruskal-Wallis),
repeated-measures or factorial ANOVA, power analysis, Bayesian tests, translations.

## License

GPL-3.0-or-later, like Orange. See [LICENSE](LICENSE).
