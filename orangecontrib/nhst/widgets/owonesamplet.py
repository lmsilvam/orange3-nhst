from typing import List

import numpy as np

from Orange.data import Table, Variable, ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.nhst.stats.assumptions import shapiro_wilk
from orangecontrib.nhst.stats.ttests import one_sample_t
from orangecontrib.nhst.widgets import utils as u

ALTERNATIVES = ("two-sided", "greater", "less")
MU0_COL, P_COL = 0, 3  # Test table has no name column: row order matches Descriptives


class OWOneSampleT(OWWidget):
    name = "One Sample t-Test"
    description = "Test whether the mean of numeric variables differs from a value mu0."
    icon = "icons/onesamplet.svg"
    priority = 10
    keywords = ["t-test", "one sample", "mean", "nhst", "student"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        test_results = Output("Test Results", Table)

    settingsHandler = DomainContextHandler()
    selected_vars: List[Variable] = ContextSetting([])
    mu0 = ContextSetting({})  # {Variable: test value}; edited in the Test table
    alternative = Setting(0)  # index into ALTERNATIVES
    alpha = Setting(0.05)
    confidence = Setting(95.0)  # percent
    report_d = Setting(True)
    auto_commit = Setting(True)

    graph_name = "plot"

    class Error(OWWidget.Error):
        no_numeric = Msg("Data has no numeric variables.")
        invalid_vars = Msg("Test undefined, variable skipped: {}")

    class Warning(OWWidget.Warning):
        not_normal = Msg("Normality violated (Shapiro-Wilk p < alpha): {}")

    class Information(OWWidget.Information):
        dropped = Msg("Missing values dropped: {}")

    def __init__(self):
        super().__init__()
        self.data = None
        self.results = []  # [(variable, OneSampleResult, NormalityResult)], rows of the Test table

        self.var_model = DomainModel(
            order=(DomainModel.ATTRIBUTES, DomainModel.CLASSES, DomainModel.METAS),
            valid_types=ContinuousVariable)
        box = gui.vBox(self.controlArea, "Variables")
        view = gui.listView(box, self, "selected_vars", model=self.var_model,
                            callback=self.commit.deferred)
        view.setSelectionMode(view.ExtendedSelection)

        gui.radioButtons(
            self.controlArea, self, "alternative", box="Alternative hypothesis",
            btnLabels=("Two-sided (mean ≠ mu0)", "Greater (mean > mu0)",
                       "Less (mean < mu0)"), callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Significance")
        gui.doubleSpin(box, self, "alpha", 0.001, 0.999, 0.01, decimals=3,
                       label="Alpha: ", callback=self.commit.deferred)
        gui.doubleSpin(box, self, "confidence", 1, 99.9, 1, decimals=1,
                       label="Confidence (%): ", callback=self.commit.deferred)
        gui.checkBox(gui.vBox(self.controlArea, "Effect size"), self, "report_d",
                     "Report Cohen's d", callback=self.commit.deferred)
        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        self.desc_model = u.ResultsModel(["Variable", "N", "Mean", "SD", "SE"])
        self.test_model = u.ResultsModel(
            [], p_columns=[P_COL], right_columns=[5], editable_columns=[MU0_COL], on_edit=self._mu0_edited)
        self.desc_view = u.make_table_view(self.desc_model)
        self.test_view = u.make_table_view(self.test_model, selectable_rows=True)
        gui.vBox(self.mainArea, "Descriptives").layout().addWidget(self.desc_view)
        gui.vBox(self.mainArea, "Test (double-click mu0 to edit)").layout().addWidget(
            self.test_view)
        self.plot = u.new_plot(self.mainArea)
        self.mainArea.layout().addWidget(self.plot, 1)
        self.normality_label = gui.widgetLabel(self.mainArea, "")
        self.test_view.selectionModel().selectionChanged.connect(self._update_plot)
        self.mainArea.setMinimumWidth(440)
        self.setStyleSheet(u.pane_qss(self))

    # ---- inputs ----
    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.selected_vars, self.mu0 = [], {}
        self.data = data
        self.Error.no_numeric.clear()
        self.var_model.set_domain(data.domain if data is not None else None)
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(len(data))
            if self.var_model:
                self.selected_vars = [self.var_model[0]]
                self.openContext(data)
            else:
                self.Error.no_numeric()

    def handleNewSignals(self):
        self.commit.deferred()

    def _mu0_edited(self, row, _col, value):
        self.mu0[self.results[row][0]] = value
        self.commit.deferred()

    # ---- computation (all statistics live in orangecontrib.nhst.stats) ----
    @gui.deferred
    def commit(self):
        keep = self._current_var()  # selected row survives recomputation
        for msg in (self.Error.invalid_vars, self.Warning.not_normal,
                    self.Information.dropped):
            msg.clear()
        # Design: a variable whose test is undefined (<2 obs, zero variance) is left out of
        # tables and output and named in an Error; other selected variables still run.
        # If none is valid, the output is cleared (None) as the spec requires.
        results, invalid, dropped, nonnormal = [], [], [], []
        for var in self.selected_vars if self.data is not None else []:
            col = self.data.get_column(var).astype(float)
            n_miss = int(np.isnan(col).sum())
            if n_miss:
                dropped.append(f"{n_miss} in {var.name}")
            try:
                res = one_sample_t(col, self.mu0.get(var, 0.0),
                                   ALTERNATIVES[self.alternative], self.confidence / 100)
            except ValueError as ex:
                invalid.append(f"{var.name} ({ex})")
                continue
            norm = shapiro_wilk(col)
            results.append((var, res, norm))
            if norm.p < self.alpha:  # NaN (skipped) compares False
                nonnormal.append(var.name)
        self.results = results
        for msg, items in ((self.Error.invalid_vars, invalid),
                           (self.Warning.not_normal, nonnormal),
                           (self.Information.dropped, dropped)):
            if items:
                msg(", ".join(items))
        self._show(keep)
        self._send()

    def _show(self, keep):
        self.desc_model.set_rows(
            [[v.name, r.n, r.mean, r.sd, r.se] for v, r, _ in self.results])
        headers = ["mu0", "t", "df", "p", "Mean dif.", "CI"]
        rows = [[r.mu0, r.t, r.df, r.p, r.mean_diff, u.fmt_ci(r.ci_low, r.ci_high)]
                for v, r, _ in self.results]
        if self.report_d:
            headers.append("Cohen's d")
            for row, (_, r, _) in zip(rows, self.results):
                row.append(r.d)
        self.test_model.set_rows(rows, headers)
        if self.results:
            vs = [v for v, *_ in self.results]
            idx = vs.index(keep) if keep in vs else 0
            self.test_view.selectRow(idx)
        else:
            self._update_plot()

    def _send(self):
        if not self.results:
            self.Outputs.test_results.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return
        names = ["mu0", "N", "mean", "SD", "SE", "t", "df", "p-value", "mean difference",
                 "CI low", "CI high"]
        rows = [[r.mu0, r.n, r.mean, r.sd, r.se, r.t, r.df, r.p, r.mean_diff,
                 r.ci_low, r.ci_high] for _, r, _ in self.results]
        if self.report_d:
            names.append("Cohen's d")
            for row, (_, r, _) in zip(rows, self.results):
                row.append(r.d)
        out = u.results_table(names, rows, ["Variable"], [[v.name] for v, *_ in self.results])
        self.Outputs.test_results.send(out)
        self.info.set_output_summary(len(out))

    # ---- plot ----
    def _current_var(self):
        rows = self.test_view.selectionModel().selectedRows()
        return self.results[rows[0].row()][0] if rows and self.results else None

    def _update_plot(self, *_):
        self.plot.clear()
        var = self._current_var()
        if var is None:
            self.normality_label.setText("")
            return
        _, r, norm = next(t for t in self.results if t[0] is var)
        x = self.data.get_column(var).astype(float)
        x = x[~np.isnan(x)]
        u.add_histogram(self.plot, x)
        self.plot.getPlotItem().setLabel("bottom", var.name)
        # CI of the mean difference shifted back to the variable's scale
        # one-sided CIs have an infinite end; clip to the data range or pyqtgraph hangs painting
        span = np.ptp(np.append(x, [r.mu0, r.mean])) or 1.0
        lo_lim = min(x.min(), r.mu0, r.mean) - span
        hi_lim = max(x.max(), r.mu0, r.mean) + span
        u.add_band(self.plot, np.clip(r.mu0 + r.ci_low, lo_lim, hi_lim),
                   np.clip(r.mu0 + r.ci_high, lo_lim, hi_lim))
        u.add_key(self.plot, "mu0", dashed=True)
        u.add_key(self.plot, "Mean")
        u.add_vline(self.plot, r.mu0, dashed=True)
        u.add_vline(self.plot, r.mean)
        self.normality_label.setText(self._normality_text(var, norm))

    @staticmethod
    def _normality_text(var, norm):
        if norm.note:
            return f"{var.name}: {norm.note}"
        return (f"Shapiro-Wilk ({var.name}): W = {u.fmt_num(norm.w)}, "
                f"p {'<0.001' if norm.p < 0.001 else '= ' + u.fmt_num(norm.p)}")

    # ---- report ----
    def send_report(self):
        if not self.results:
            return
        self.report_items("Settings", [
            ("Alternative", ("Two-sided", "Greater", "Less")[self.alternative]),
            ("Alpha", self.alpha), ("Confidence level", f"{self.confidence:g} %")])
        self.report_table("Descriptives", self.desc_view)
        self.report_table("Test", self.test_view)
        var = self._current_var()
        if var is not None:
            norm = next(n for v, _, n in self.results if v is var)
            self.report_paragraph("Normality", self._normality_text(var, norm))
        self.report_plot()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWOneSampleT).run(Table("iris"))
