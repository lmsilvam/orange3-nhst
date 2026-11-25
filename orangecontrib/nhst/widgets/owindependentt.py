import numpy as np
from AnyQt.QtCore import Qt

from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.nhst.stats.assumptions import shapiro_wilk, levene_median
from orangecontrib.nhst.stats.descriptives import describe
from orangecontrib.nhst.stats.ttests import independent_t
from orangecontrib.nhst.widgets import utils as u

ALTERNATIVES = ("two-sided", "greater", "less")
P_COL, CI_COL = 2, 4  # Test table has no name column (groups are named in the Descriptives)


class OWIndependentT(OWWidget):
    name = "Independent Samples t-Test"
    description = "Test whether the means of two independent groups differ."
    icon = "icons/independentt.svg"
    priority = 15
    keywords = ["t-test", "independent", "two sample", "welch", "nhst", "student"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        test_results = Output("Test Results", Table)

    settingsHandler = DomainContextHandler()
    variable = ContextSetting(None)
    grouping = ContextSetting(None)
    alternative = Setting(0)  # index into ALTERNATIVES
    alpha = Setting(0.05)
    confidence = Setting(95.0)  # percent
    report_g = Setting(True)
    auto_commit = Setting(True)

    graph_name = "plot"

    class Error(OWWidget.Error):
        no_numeric = Msg("Data has no numeric variable.")
        no_categorical = Msg("Data has no categorical variable.")
        too_many_levels = Msg("Selected grouping variable has more than 2 levels.")
        too_few_levels = Msg("Selected grouping variable has fewer than 2 levels.")
        invalid = Msg("Test undefined: {}")

    class Warning(OWWidget.Warning):
        not_normal = Msg("Normality violated (Shapiro-Wilk p < alpha) in: {}")

    class Information(OWWidget.Information):
        dropped = Msg("Missing values dropped: {}")

    def __init__(self):
        super().__init__()
        self.data = None
        self.result = None  # (res, desc1, desc2, [norm1, norm2], levene, x1, x2) or None

        order = (DomainModel.ATTRIBUTES, DomainModel.CLASSES, DomainModel.METAS)
        self.var_model = DomainModel(order=order, valid_types=ContinuousVariable)
        self.group_var_model = DomainModel(order=order, valid_types=DiscreteVariable)
        box = gui.vBox(self.controlArea, "Variables")
        gui.comboBox(box, self, "variable", label="Test variable: ", model=self.var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)
        gui.comboBox(box, self, "grouping", label="Grouping by: ", model=self.group_var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)

        self.alt_box = gui.radioButtons(
            self.controlArea, self, "alternative", box="Alternative hypothesis",
            btnLabels=("", "", ""), callback=self.commit.deferred)
        self.groups_label = gui.widgetLabel(self.alt_box, "")
        self.groups_label.setWordWrap(True)
        self._alt_labels()

        box = gui.vBox(self.controlArea, "Significance")
        gui.doubleSpin(box, self, "alpha", 0.001, 0.999, 0.01, decimals=3,
                       label="Alpha: ", callback=self.commit.deferred)
        gui.doubleSpin(box, self, "confidence", 1, 99.9, 1, decimals=1,
                       label="Confidence (%): ", callback=self.commit.deferred)
        gui.checkBox(gui.vBox(self.controlArea, "Effect size"), self, "report_g",
                     "Report Hedges' g", callback=self.commit.deferred)
        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        self.desc_model = u.ResultsModel(["Group", "N", "Mean", "SD", "SE"])
        self.test_model = u.ResultsModel([], p_columns=[P_COL], right_columns=[CI_COL])
        self.desc_view = u.make_table_view(self.desc_model)
        self.test_view = u.make_table_view(self.test_model)
        gui.vBox(self.mainArea, "Descriptives by group").layout().addWidget(self.desc_view)
        gui.vBox(self.mainArea, "Test").layout().addWidget(self.test_view)
        self.plot = u.new_plot(self.mainArea)
        self.mainArea.layout().addWidget(self.plot, 1)
        self.assumption_label = gui.widgetLabel(gui.vBox(self.mainArea, "Assumption checks"), "")
        self.mainArea.setMinimumWidth(440)
        self.setStyleSheet(u.pane_qss(self))

    # ---- inputs ----
    @Inputs.data
    def set_data(self, data):
        self.closeContext()
        self.variable = self.grouping = None
        self.data = data
        for msg in (self.Error.no_numeric, self.Error.no_categorical):
            msg.clear()
        domain = data.domain if data is not None else None
        self.var_model.set_domain(domain)
        self.group_var_model.set_domain(domain)
        if data is None:
            self.info.set_input_summary(self.info.NoInput)
            return
        self.info.set_input_summary(len(data))
        if not self.var_model:
            self.Error.no_numeric()
        if not self.group_var_model:
            self.Error.no_categorical()
        if not self.var_model or not self.group_var_model:
            return
        # defaults: targets first (class variable), else the first suitable variable
        cv = domain.class_var
        self.variable = cv if cv in self.var_model else self.var_model[0]
        self.grouping = cv if cv in self.group_var_model else self.group_var_model[0]
        self.openContext(data)

    def _groups(self):
        """Names of Group 1 and Group 2: the first and second level of the grouping variable."""
        levels = list(self.grouping.values[:2]) if self.grouping is not None else []
        return tuple(levels + ["1", "2"][len(levels):])

    def _alt_labels(self):
        # short radio texts keep the control area narrow; the level names go in a wrapped label
        g1, g2 = self._groups()
        for b, text in zip(self.alt_box.buttons, (
                "Group 1 ≠ Group 2", "Group 1 > Group 2", "Group 1 < Group 2")):
            b.setText(text)
        self.groups_label.setText(f"Group 1 = {g1}, Group 2 = {g2}")

    def handleNewSignals(self):
        self.commit.deferred()

    # ---- computation (all statistics live in orangecontrib.nhst.stats) ----
    @gui.deferred
    def commit(self):
        for msg in (self.Error.too_many_levels, self.Error.too_few_levels, self.Error.invalid,
                    self.Warning.not_normal, self.Information.dropped):
            msg.clear()
        self.result = None
        self._alt_labels()
        if self.data is not None and None not in (self.variable, self.grouping):
            n_levels = len(self.grouping.values)
            if n_levels > 2:
                self.Error.too_many_levels()
            elif n_levels < 2:
                self.Error.too_few_levels()
            else:
                self._compute()
        self._show()
        self._send()

    def _compute(self):
        y = self.data.get_column(self.variable).astype(float)
        g = self.data.get_column(self.grouping).astype(float)
        in_group = [g == 0, g == 1]
        dropped = []
        n_y = int(np.isnan(y[in_group[0] | in_group[1]]).sum())
        if n_y:
            dropped.append(f"{n_y} in {self.variable.name}")
        n_g = int(np.isnan(g).sum())
        if n_g:
            dropped.append(f"{n_g} in {self.grouping.name}")
        if dropped:
            self.Information.dropped(", ".join(dropped))
        x1, x2 = (y[m][~np.isnan(y[m])] for m in in_group)
        try:
            res = independent_t(x1, x2, False, ALTERNATIVES[self.alternative],
                                self.confidence / 100)
        except ValueError as ex:
            self.Error.invalid(ex)
            return
        norms = [shapiro_wilk(x1), shapiro_wilk(x2)]
        lev = levene_median([x1, x2])
        self.result = (res, describe(x1), describe(x2), norms, lev, x1, x2)
        bad = [name for name, nr in zip(self._groups(), norms) if nr.p < self.alpha]
        if bad:  # NaN (skipped) compares False
            self.Warning.not_normal(", ".join(bad))

    def _show(self):
        self.plot.clear()
        if self.result is None:
            self.desc_model.set_rows([])
            self.test_model.set_rows([])
            self.assumption_label.setText("")
            return
        r, d1, d2, _, _, x1, x2 = self.result
        self.desc_model.set_rows([
            [f"Group 1: {self._groups()[0]}", d1.n, d1.mean, d1.sd, d1.se],
            [f"Group 2: {self._groups()[1]}", d2.n, d2.mean, d2.sd, d2.se]])
        headers = ["t", "df", "p", "Mean dif.", "CI"]
        row = [r.t, r.df, r.p, r.mean_diff, u.fmt_ci(r.ci_low, r.ci_high)]
        if self.report_g:
            headers.append("Hedges' g")
            row.append(r.g)
        self.test_model.set_rows([row], headers)
        self.plot.getPlotItem().setLabel("left", self.variable.name)
        self.plot.getPlotItem().setLabel("bottom", self.grouping.name)
        u.add_boxplots(self.plot, [x1, x2], self._groups())
        self.assumption_label.setText(self._assumption_text())

    @staticmethod
    def _p_text(p):
        return "p <0.001" if p < 0.001 else f"p = {u.fmt_num(p)}"

    def _assumption_text(self):
        norms, lev = self.result[3], self.result[4]
        lines = []
        for name, nr in zip(self._groups(), norms):
            lines.append(f"Shapiro-Wilk, {name}: {nr.note}" if nr.note else
                         f"Shapiro-Wilk, {name}: W = {u.fmt_num(nr.w)}, {self._p_text(nr.p)}")
        lines.append(f"Levene (median), equal variances (information only): F = {u.fmt_num(lev.f)}, "
                     f"{self._p_text(lev.p)}")
        return "\n".join(lines)

    def _send(self):
        if self.result is None:
            self.Outputs.test_results.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return
        r = self.result[0]
        names = ["N1", "N2", "mean1", "mean2", "SD1", "SD2", "t", "df", "p-value",
                 "mean difference", "CI low", "CI high"]
        row = [r.n1, r.n2, r.mean1, r.mean2, r.sd1, r.sd2, r.t, r.df, r.p, r.mean_diff,
               r.ci_low, r.ci_high]
        if self.report_g:
            names.append("Hedges' g")
            row.append(r.g)
        out = u.results_table(names, [row], ["Variable", "Test"],
                              [[self.variable.name, r.test]])
        self.Outputs.test_results.send(out)
        self.info.set_output_summary(len(out))

    # ---- report ----
    def send_report(self):
        if self.result is None:
            return
        self.report_items("Settings", [
            ("Variable", self.variable.name), ("Grouping by", self.grouping.name),
            ("Group 1 / Group 2", " / ".join(self._groups())), ("Test", "Welch"),
            ("Alternative", ("Two-sided", "Group 1 greater than Group 2",
                             "Group 1 less than Group 2")[self.alternative]),
            ("Alpha", self.alpha), ("Confidence level", f"{self.confidence:g} %")])
        self.report_table("Descriptives", self.desc_view)
        self.report_table("Test", self.test_view)
        self.report_paragraph("Assumption checks", self._assumption_text().replace("\n", "; "))
        self.report_plot()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWIndependentT).run(Table("iris"))
