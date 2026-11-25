import numpy as np
from AnyQt.QtCore import Qt

from Orange.data import Table, ContinuousVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.nhst.stats.assumptions import shapiro_wilk
from orangecontrib.nhst.stats.descriptives import describe
from orangecontrib.nhst.stats.ttests import paired_t
from orangecontrib.nhst.widgets import utils as u

ALTERNATIVES = ("two-sided", "greater", "less")
P_COL = 2  # Test table has no name column (the pair is named in the Descriptives)


class OWPairedT(OWWidget):
    name = "Paired Samples t-Test"
    description = "Test whether the mean difference between two paired numeric variables is zero."
    icon = "icons/pairedt.svg"
    priority = 20
    keywords = ["t-test", "paired", "dependent", "repeated", "nhst", "student"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        test_results = Output("Test Results", Table)

    settingsHandler = DomainContextHandler()
    var1 = ContextSetting(None)
    var2 = ContextSetting(None)
    alternative = Setting(0)  # index into ALTERNATIVES
    alpha = Setting(0.05)
    confidence = Setting(95.0)  # percent
    report_dz = Setting(True)
    auto_commit = Setting(True)

    graph_name = "plot"

    class Error(OWWidget.Error):
        not_enough_numeric = Msg("At least two numeric variables are needed.")
        same_variable = Msg("Variable 1 and Variable 2 must be different.")
        invalid = Msg("Test undefined: {}")

    class Warning(OWWidget.Warning):
        not_normal = Msg("Normality of the differences violated "
                         "(Shapiro-Wilk on the differences, p < alpha).")

    class Information(OWWidget.Information):
        dropped = Msg("Pairs with missing values dropped: {}")

    def __init__(self):
        super().__init__()
        self.data = None
        self.xs = None  # complete-pair values of both variables, for the plot
        self.result = None  # (PairedResult, NormalityResult, differences, desc1, desc2) or None

        self.var_model = DomainModel(
            order=(DomainModel.ATTRIBUTES, DomainModel.CLASSES, DomainModel.METAS),
            valid_types=ContinuousVariable)
        box = gui.vBox(self.controlArea, "Variables")
        gui.comboBox(box, self, "var1", label="Variable 1: ", model=self.var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)
        gui.comboBox(box, self, "var2", label="Variable 2: ", model=self.var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)
        gui.button(box, self, "Swap", callback=self._swap)

        gui.radioButtons(
            self.controlArea, self, "alternative", box="Alternative hypothesis",
            btnLabels=("Two-sided (difference ≠ 0)", "Greater (difference > 0)",
                       "Less (difference < 0)"), callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Significance")
        gui.doubleSpin(box, self, "alpha", 0.001, 0.999, 0.01, decimals=3,
                       label="Alpha: ", callback=self.commit.deferred)
        gui.doubleSpin(box, self, "confidence", 1, 99.9, 1, decimals=1,
                       label="Confidence (%): ", callback=self.commit.deferred)
        gui.checkBox(gui.vBox(self.controlArea, "Effect size"), self, "report_dz",
                     "Report Cohen's dz", callback=self.commit.deferred)
        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        self.desc_model = u.ResultsModel(["", "N", "Mean", "SD", "SE"])
        self.test_model = u.ResultsModel(
            [], p_columns=[P_COL], right_columns=[4])
        self.desc_view = u.make_table_view(self.desc_model)
        self.test_view = u.make_table_view(self.test_model)
        gui.vBox(self.mainArea, "Descriptives").layout().addWidget(self.desc_view)
        gui.vBox(self.mainArea, "Test").layout().addWidget(self.test_view)
        self.plot = u.new_plot(self.mainArea, xlabel="Value", ylabel="Density")
        self.mainArea.layout().addWidget(self.plot, 1)
        self.normality_label = gui.widgetLabel(self.mainArea, "")
        self.mainArea.setMinimumWidth(440)
        self.setStyleSheet(u.pane_qss(self))

    # ---- inputs ----
    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.var1 = self.var2 = None
        self.data = data
        self.Error.not_enough_numeric.clear()
        self.var_model.set_domain(data.domain if data is not None else None)
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
        else:
            self.info.set_input_summary(len(data))
            if len(self.var_model) >= 2:
                self.var1, self.var2 = self.var_model[0], self.var_model[1]
                self.openContext(data)
            else:
                self.Error.not_enough_numeric()

    def handleNewSignals(self):
        self.commit.deferred()

    def _swap(self):
        self.var1, self.var2 = self.var2, self.var1
        self.commit.deferred()

    # ---- computation (all statistics live in orangecontrib.nhst.stats) ----
    @gui.deferred
    def commit(self):
        for msg in (self.Error.same_variable, self.Error.invalid,
                    self.Warning.not_normal, self.Information.dropped):
            msg.clear()
        self.result = None
        if self.data is not None and self.var1 is not None and self.var2 is not None:
            if self.var1 is self.var2:
                self.Error.same_variable()
            else:
                self._compute()
        self._show()
        self._send()

    def _compute(self):
        x1 = self.data.get_column(self.var1).astype(float)
        x2 = self.data.get_column(self.var2).astype(float)
        ok = ~(np.isnan(x1) | np.isnan(x2))
        if not ok.all():
            self.Information.dropped(int((~ok).sum()))
        try:
            res = paired_t(x1, x2, ALTERNATIVES[self.alternative], self.confidence / 100)
        except ValueError as ex:
            self.Error.invalid(ex)
            return
        diff = (x1 - x2)[ok]
        norm = shapiro_wilk(diff)
        self.xs = (x1[ok], x2[ok])
        self.result = (res, norm, diff, describe(x1[ok]), describe(x2[ok]))
        if norm.p < self.alpha:  # NaN (skipped) compares False
            self.Warning.not_normal()

    def _show(self):
        self.plot.clear()
        if self.result is None:
            self.desc_model.set_rows([])
            self.test_model.set_rows([])
            self.normality_label.setText("")
            return
        r, norm, _, d1, d2 = self.result
        self.desc_model.set_rows([
            [f"Variable 1: {self.var1.name}", d1.n, d1.mean, d1.sd, d1.se],
            [f"Variable 2: {self.var2.name}", d2.n, d2.mean, d2.sd, d2.se],
            ["Difference", r.n, r.mean_diff, r.sd_diff, r.se_diff]])
        headers = ["t", "df", "p", "Mean dif.", "CI"]
        row = [r.t, r.df, r.p, r.mean_diff, u.fmt_ci(r.ci_low, r.ci_high)]
        if self.report_dz:
            headers.append("Cohen's dz")
            row.append(r.dz)
        self.test_model.set_rows([row], headers)
        self._update_plot()
        self.normality_label.setText(self._normality_text(norm))

    def _update_plot(self):
        # overlap of the two densities: the more they overlap, the smaller the difference
        _, _, _, d1, d2 = self.result
        x1, x2 = self.xs
        self.plot.addLegend(offset=(-10, 10))
        for i, (x, d, var) in enumerate(((x1, d1, self.var1), (x2, d2, self.var2))):
            u.add_density(self.plot, x, i, name=var.name)
            u.add_vline(self.plot, d.mean, dashed=True, color=u._rgb(i))
        u.add_key(self.plot, "Dashed: mean", dashed=True)

    @staticmethod
    def _normality_text(norm):
        if norm.note:
            return f"Differences: {norm.note}"
        return (f"Shapiro-Wilk (differences): W = {u.fmt_num(norm.w)}, "
                f"p {'<0.001' if norm.p < 0.001 else '= ' + u.fmt_num(norm.p)}")

    def _send(self):
        if self.result is None:
            self.Outputs.test_results.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return
        r = self.result[0]
        names = ["N", "mean1", "mean2", "mean difference", "SD difference", "t", "df",
                 "p-value", "CI low", "CI high"]
        row = [r.n, r.mean1, r.mean2, r.mean_diff, r.sd_diff, r.t, r.df, r.p,
               r.ci_low, r.ci_high]
        if self.report_dz:
            names.append("Cohen's dz")
            row.append(r.dz)
        out = u.results_table(names, [row], ["Pair"],
                              [[f"{self.var1.name} - {self.var2.name}"]])
        self.Outputs.test_results.send(out)
        self.info.set_output_summary(len(out))

    # ---- report ----
    def send_report(self):
        if self.result is None:
            return
        self.report_items("Settings", [
            ("Pair", f"{self.var1.name} - {self.var2.name}"),
            ("Alternative", ("Two-sided", "Greater", "Less")[self.alternative]),
            ("Alpha", self.alpha), ("Confidence level", f"{self.confidence:g} %")])
        self.report_table("Descriptives", self.desc_view)
        self.report_table("Test", self.test_view)
        self.report_paragraph("Normality", self._normality_text(self.result[1]))
        self.report_plot()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWPairedT).run(Table("iris"))
