import numpy as np
from AnyQt.QtCore import Qt

from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.nhst.stats.anova import one_way_anova
from orangecontrib.nhst.stats.assumptions import shapiro_residuals, levene_median
from orangecontrib.nhst.stats.descriptives import describe
from orangecontrib.nhst.stats.posthoc import tukey_hsd, games_howell, bonferroni
from orangecontrib.nhst.stats.ttests import one_sample_t
from orangecontrib.nhst.widgets import utils as u

METHODS = ("Classic", "Welch")
POSTHOCS = ("Tukey HSD", "Games-Howell", "Bonferroni", "None")
DEFAULT_POSTHOC = (0, 1)  # per method: Tukey for Classic, Games-Howell for Welch
POSTHOC_LABELS = ("Tukey HSD", "Games-Howell", "Bonferroni (CIs adjusted)")
P_COL = 3  # Test table: F, df1, df2, p, ...
PH_CI_COL, PH_P_COL = 2, 3  # Post hoc table: Comparison, Mean dif., CI, Adjusted p
ETA_NOTE = "Uses classic sum of squares"


class OWOneWayANOVA(OWWidget):
    name = "One-Way ANOVA"
    description = "Test whether the means of several independent groups differ."
    icon = "icons/onewayanova.svg"
    priority = 20
    keywords = ["anova", "one-way", "welch", "tukey", "games-howell", "post hoc", "nhst"]
    want_main_area = True

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        test_results = Output("Test Results", Table)
        post_hoc = Output("Post Hoc", Table)

    settingsHandler = DomainContextHandler()
    variable = ContextSetting(None)
    grouping = ContextSetting(None)
    method = Setting(0)  # index into METHODS
    posthoc = Setting(0)  # index into POSTHOCS
    alpha = Setting(0.05)
    confidence = Setting(95.0)  # percent
    report_eta2 = Setting(True)
    report_omega2 = Setting(True)
    include_ss = Setting(False)
    auto_commit = Setting(True)

    graph_name = "plot"

    class Error(OWWidget.Error):
        no_numeric = Msg("Data has no numeric variable.")
        no_categorical = Msg("Data has no categorical variable.")
        too_few_groups = Msg("Fewer than 2 groups with valid observations.")
        invalid = Msg("Test undefined: {}")

    class Warning(OWWidget.Warning):
        not_normal = Msg("Residuals not normal (Shapiro-Wilk p < alpha).")
        unequal_var = Msg("Variances differ (Levene p < alpha): "
                          "consider the Welch method with Games-Howell post hoc.")

    class Information(OWWidget.Information):
        two_groups = Msg("Only 2 groups: the Independent Samples t-Test widget fits better.")
        dropped = Msg("Missing values dropped: {}")

    def __init__(self):
        super().__init__()
        self.data = None
        self.result = None  # dict (see _compute) or None

        order = (DomainModel.ATTRIBUTES, DomainModel.CLASSES, DomainModel.METAS)
        self.var_model = DomainModel(order=order, valid_types=ContinuousVariable)
        self.group_var_model = DomainModel(order=order, valid_types=DiscreteVariable)
        box = gui.vBox(self.controlArea, "Variables")
        gui.comboBox(box, self, "variable", label="Test variable: ", labelWidth=95, model=self.var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)
        gui.comboBox(box, self, "grouping", label="Grouping by: ", labelWidth=95, model=self.group_var_model,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Test")
        gui.comboBox(box, self, "method", label="Method: ", labelWidth=95, items=METHODS,
                     orientation=Qt.Horizontal, callback=self._method_changed)
        gui.comboBox(box, self, "posthoc", label="Post hoc: ", labelWidth=95, items=POSTHOCS,
                     orientation=Qt.Horizontal, callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Significance")
        gui.doubleSpin(box, self, "alpha", 0.001, 0.999, 0.01, decimals=3,
                       label="Alpha: ", callback=self.commit.deferred)
        gui.doubleSpin(box, self, "confidence", 1, 99.9, 1, decimals=1,
                       label="Confidence (%): ", callback=self.commit.deferred)

        box = gui.vBox(self.controlArea, "Effect size")
        gui.checkBox(box, self, "report_eta2", "η²", callback=self.commit.deferred,
                     tooltip=ETA_NOTE)
        gui.checkBox(box, self, "report_omega2", "ω²", callback=self.commit.deferred,
                     tooltip=ETA_NOTE)
        note = gui.widgetLabel(box, ETA_NOTE)
        note.setWordWrap(True)
        note.setToolTip(ETA_NOTE)
        gui.checkBox(self.controlArea, self, "include_ss", "Include SS and MS in Test Results",
                     callback=self.commit.deferred)
        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        self.desc_model = u.ResultsModel(["Group", "N", "Mean", "SD", "SE"])
        self.test_model = u.ResultsModel([], p_columns=[P_COL])
        self.ph_model = u.ResultsModel(
            ["Comparison", "Mean dif.", "CI", "Adjusted p"],
            p_columns=[PH_P_COL], right_columns=[PH_CI_COL])
        self.desc_view = u.make_table_view(self.desc_model)
        self.test_view = u.make_table_view(self.test_model)
        self.ph_view = u.make_table_view(self.ph_model)
        gui.vBox(self.mainArea, "Descriptives by group").layout().addWidget(self.desc_view)
        gui.vBox(self.mainArea, "Test").layout().addWidget(self.test_view)
        box = gui.vBox(self.mainArea, "Post hoc")
        self.ph_caption = gui.widgetLabel(box, "")
        self.ph_caption.hide()
        box.layout().addWidget(self.ph_view)
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

    def handleNewSignals(self):
        self.commit.deferred()

    def _method_changed(self):
        # follow the method only while the post hoc still sits at the previous method's default
        if self.posthoc == DEFAULT_POSTHOC[1 - self.method]:
            self.posthoc = DEFAULT_POSTHOC[self.method]
        self.commit.deferred()

    # ---- computation (all statistics live in orangecontrib.nhst.stats) ----
    @gui.deferred
    def commit(self):
        for msg in (self.Error.too_few_groups, self.Error.invalid, self.Warning.not_normal,
                    self.Warning.unequal_var, self.Information.two_groups,
                    self.Information.dropped):
            msg.clear()
        self.result = None
        if self.data is not None and None not in (self.variable, self.grouping):
            self._compute()
        self._show()
        self._send()

    def _compute(self):
        y = self.data.get_column(self.variable).astype(float)
        g = self.data.get_column(self.grouping).astype(float)
        dropped = []
        n_y = int(np.isnan(y[~np.isnan(g)]).sum())
        if n_y:
            dropped.append(f"{n_y} in {self.variable.name}")
        n_g = int(np.isnan(g).sum())
        if n_g:
            dropped.append(f"{n_g} in {self.grouping.name}")
        if dropped:
            self.Information.dropped(", ".join(dropped))
        groups, labels = [], []
        for i, level in enumerate(self.grouping.values):
            x = y[g == i]
            x = x[~np.isnan(x)]
            if len(x):  # levels without observations are not groups
                groups.append(x)
                labels.append(level)
        if len(groups) < 2:
            self.Error.too_few_groups()
            return
        conf, welch = self.confidence / 100, self.method == 1
        try:
            res = one_way_anova(groups, METHODS[self.method].lower(), labels)
            lev = levene_median(groups, labels)
            shap = shapiro_residuals(groups, labels)
            ph_rows, ph_label = [], None
            if self.posthoc == 0:
                ph_rows = tukey_hsd(groups, labels, conf)
            elif self.posthoc == 1:
                ph_rows = games_howell(groups, labels, conf)
            elif self.posthoc == 2:
                ph_rows = bonferroni(groups, labels, welch, conf)
            if ph_rows:
                ph_label = POSTHOC_LABELS[self.posthoc]
            means = [one_sample_t(x, 0.0, "two-sided", conf) for x in groups]
        except ValueError as ex:
            self.Error.invalid(ex)
            return
        self.result = dict(res=res, levene=lev, shapiro=shap, ph_rows=ph_rows,
                           ph_label=ph_label, labels=labels, desc=[describe(x) for x in groups],
                           means=means, grand=float(np.concatenate(groups).mean()))
        if len(groups) == 2:
            self.Information.two_groups()
        if shap.p < self.alpha:  # NaN (skipped) compares False
            self.Warning.not_normal()
        if self.method == 0 and lev.p < self.alpha:
            self.Warning.unequal_var()

    def _test_columns(self):
        """(display headers, output names, getter) of the optional columns, in order."""
        r = self.result["res"]
        cols = [("F", "F", r.f), ("df1", "df1", r.df1), ("df2", "df2", r.df2),
                ("p", "p-value", r.p)]
        if self.report_eta2:
            cols.append(("η²", "η²", r.eta2))
        if self.report_omega2:
            cols.append(("ω²", "ω²", r.omega2))
        return cols

    @staticmethod
    def _int_if_whole(x):
        return int(x) if float(x).is_integer() else x

    def _show(self):
        self.plot.clear()
        if self.result is None:
            for m in (self.desc_model, self.test_model, self.ph_model):
                m.set_rows([])
            self.ph_caption.hide()
            self.assumption_label.setText("")
            return
        R = self.result
        self.desc_model.set_rows([[lab, d.n, d.mean, d.sd, d.se]
                                  for lab, d in zip(R["labels"], R["desc"])])
        cols = self._test_columns()
        self.test_model.set_rows(
            [[self._int_if_whole(v) if h.startswith("df") else v for h, _, v in cols]],
            [h for h, _, _ in cols])
        self.ph_model.set_rows(
            [[f"{p.group_a} - {p.group_b}", p.mean_diff, u.fmt_ci(p.ci_low, p.ci_high), p.p_adj]
             for p in R["ph_rows"]])
        # label only where it adds information the dropdown lacks
        bonf = (R["ph_label"] or "").startswith("Bonferroni")
        self.ph_caption.setText("Bonferroni-adjusted CIs")
        self.ph_caption.setVisible(bonf)  # no empty strip when there is nothing to say
        self.plot.getPlotItem().setLabel("left", self.variable.name)
        self.plot.getPlotItem().setLabel("bottom", self.grouping.name)
        u.add_mean_plot(self.plot, [m.mean for m in R["means"]],
                        [m.ci_low for m in R["means"]], [m.ci_high for m in R["means"]],
                        R["labels"], R["grand"])
        self.assumption_label.setText(self._assumption_text())

    @staticmethod
    def _p_text(p):
        return "p <0.001" if p < 0.001 else f"p = {u.fmt_num(p)}"

    def _assumption_text(self):
        sh, lev = self.result["shapiro"], self.result["levene"]
        line = sh.note or f"W = {u.fmt_num(sh.w)}, {self._p_text(sh.p)}"
        return (f"Shapiro-Wilk on residuals: {line}\n"
                f"Levene (median), equal variances: F = {u.fmt_num(lev.f)}, "
                f"{self._p_text(lev.p)}")

    def _send(self):
        if self.result is None:
            self.Outputs.test_results.send(None)
            self.Outputs.post_hoc.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return
        R, r = self.result, self.result["res"]
        cols = self._test_columns()
        names, row = [n for _, n, _ in cols], [v for _, _, v in cols]
        if self.include_ss:
            names += ["SS between", "SS within", "MS between", "MS within"]
            row += [r.ss_between, r.ss_within, r.ms_between, r.ms_within]
        out = u.results_table(names, [row], ["Variable", "Method"],
                              [[self.variable.name, r.method]])
        self.Outputs.test_results.send(out)
        self.info.set_output_summary(len(out))
        if R["ph_rows"]:
            ph = u.results_table(
                ["mean difference", "CI low", "CI high", "adjusted p-value"],
                [[p.mean_diff, p.ci_low, p.ci_high, p.p_adj] for p in R["ph_rows"]],
                ["Variable", "Method", "Group A", "Group B"],
                [[self.variable.name, R["ph_label"], p.group_a, p.group_b]
                 for p in R["ph_rows"]])
            self.Outputs.post_hoc.send(ph)
        else:
            self.Outputs.post_hoc.send(None)

    # ---- report ----
    def send_report(self):
        if self.result is None:
            return
        self.report_items("Settings", [
            ("Variable", self.variable.name), ("Grouping by", self.grouping.name),
            ("Method", METHODS[self.method]), ("Post hoc", POSTHOCS[self.posthoc]),
            ("Alpha", self.alpha), ("Confidence level", f"{self.confidence:g} %")])
        self.report_table("Descriptives", self.desc_view)
        self.report_table("Test", self.test_view)
        if self.result["ph_rows"]:
            self.report_table(f"Post hoc: {self.result['ph_label']}", self.ph_view)
        self.report_paragraph("Assumption checks", self._assumption_text().replace("\n", "; "))
        self.report_plot()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWOneWayANOVA).run(Table("iris"))
