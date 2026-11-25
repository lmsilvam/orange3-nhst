import numpy as np
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.nhst.testing import reference_iris
from orangecontrib.nhst.widgets.owonesamplet import OWOneSampleT
from orangecontrib.nhst.widgets import utils as u


def num_table(**cols):
    dom = Domain([ContinuousVariable(n) for n in cols])
    return Table.from_numpy(dom, np.column_stack(list(cols.values())).astype(float))


class TestOWOneSampleT(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWOneSampleT)
        self.iris = reference_iris()
        self.sl, self.sw, self.pl, _ = self.iris.domain.attributes[:4]

    def select(self, *vars_, mu0=None):
        self.widget.selected_vars = list(vars_)
        self.widget.mu0 = dict(mu0 or {})
        self.widget.commit.now()

    def row(self, out, name):
        i = [str(m[0]) for m in out.metas].index(name)
        return {a.name: out.X[i, j] for j, a in enumerate(out.domain.attributes)}

    def test_iris_default(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.selected_vars, [self.sl])
        out = self.get_output(self.widget.Outputs.test_results)
        self.assertEqual(len(out), 1)
        self.assertEqual(self.widget.test_model.rowCount(), 1)

    def test_reference_values(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.select(self.sl, self.sw, self.pl,
                    mu0={self.sl: 5.5, self.sw: 3.0, self.pl: 3.5})
        out = self.get_output(self.widget.Outputs.test_results)
        self.assertEqual([a.name for a in out.domain.attributes],
                         ["mu0", "N", "mean", "SD", "SE", "t", "df", "p-value",
                          "mean difference", "CI low", "CI high", "Cohen's d"])
        self.assertEqual([m.name for m in out.domain.metas], ["Variable"])
        r = self.row(out, "sepal length")
        self.assertEqual((r["N"], r["df"], r["mu0"]), (150, 149, 5.5))
        for k, v in dict(t=5.078, **{"mean difference": 0.343, "CI low": 0.210,
                                     "CI high": 0.477, "Cohen's d": 0.415}).items():
            self.assertAlmostEqual(r[k], v, places=3)
        self.assertAlmostEqual(r["p-value"] / 1.12e-06, 1, delta=0.01)
        self.assertAlmostEqual(self.row(out, "sepal width")["t"], 1.611, places=3)
        r = self.row(out, "petal length")
        self.assertAlmostEqual(r["t"], 1.790, places=3)
        # normality line of the selected row (petal length)
        self.widget.test_view.selectRow(2)
        self.assertIn("W = 0.876", self.widget.normality_label.text())

    def test_none_clears(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.test_results))
        self.assertEqual(self.widget.test_model.rowCount(), 0)
        self.assertEqual(self.widget.desc_model.rowCount(), 0)

    def test_one_sided_paints(self):
        # regression: infinite CI end made pyqtgraph hang while painting the band
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        w.show()
        for alt in (1, 2, 0):
            w.alternative = alt
            w.commit.now()
            w.grab()

    def test_options(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.select(self.sl, mu0={self.sl: 5.5})
        two = self.row(self.get_output(w.Outputs.test_results), "sepal length")
        w.alternative = 1
        w.commit.now()
        g = self.row(self.get_output(w.Outputs.test_results), "sepal length")
        self.assertAlmostEqual(g["p-value"], two["p-value"] / 2, delta=1e-12)
        self.assertTrue(np.isnan(g["CI high"]))  # open end: Orange turns inf into NaN
        w.alternative = 2
        w.commit.now()
        l = self.row(self.get_output(w.Outputs.test_results), "sepal length")
        self.assertAlmostEqual(l["p-value"], 1 - g["p-value"], places=9)
        w.alternative = 0
        w.confidence = 90.0
        w.commit.now()
        c90 = self.row(self.get_output(w.Outputs.test_results), "sepal length")
        self.assertGreater(c90["CI low"], two["CI low"])
        w.report_d = False
        w.commit.now()
        out = self.get_output(w.Outputs.test_results)
        self.assertNotIn("Cohen's d", [a.name for a in out.domain.attributes])
        self.assertEqual(w.test_model.headers[-1], "CI")

    def test_mu0_edit_via_model(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.assertTrue(w.test_model.setData(w.test_model.index(0, 0), "5.5"))
        self.assertFalse(w.test_model.setData(w.test_model.index(0, 0), "abc"))
        self.assertEqual(w.mu0[self.sl], 5.5)
        out = self.get_output(w.Outputs.test_results)
        self.assertAlmostEqual(self.row(out, "sepal length")["t"], 5.078, places=3)
        self.assertFalse(w.test_model.setData(w.test_model.index(0, 1), "1"))  # read-only

    def test_context_settings(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.select(self.sw, self.pl, mu0={self.sw: 3.0, self.pl: 3.5})
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(w.selected_vars, [])
        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual(w.selected_vars, [self.sw, self.pl])
        self.assertEqual(w.mu0, {self.sw: 3.0, self.pl: 3.5})
        out = self.get_output(w.Outputs.test_results)
        self.assertAlmostEqual(self.row(out, "petal length")["t"], 1.790, places=3)
        # settings survive widget re-creation through the handler
        w2 = self.create_widget(OWOneSampleT, stored_settings=w.settingsHandler.pack_data(w))
        self.send_signal(w2.Inputs.data, self.iris, widget=w2)
        self.assertEqual(w2.mu0, {self.sw: 3.0, self.pl: 3.5})

    def test_messages(self):
        w = self.widget
        dom = Domain([DiscreteVariable("c", values=("a", "b"))])
        self.send_signal(w.Inputs.data, Table.from_numpy(dom, np.array([[0.], [1.]])))
        self.assertTrue(w.Error.no_numeric.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.send_signal(w.Inputs.data, self.iris)
        self.assertFalse(w.Error.no_numeric.is_shown())

        # zero variance + <2 observations -> Error, output cleared
        data = num_table(const=[1, 1, 1, 1], few=[1, np.nan, np.nan, np.nan],
                         ok=[1, 2, 4, 9])
        self.send_signal(w.Inputs.data, data)
        v = data.domain.attributes
        self.select(v[0])
        self.assertTrue(w.Error.invalid_vars.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.select(v[1])
        self.assertTrue(w.Error.invalid_vars.is_shown())
        self.assertTrue(w.Information.dropped.is_shown())
        # mixed: bad ones named in Error, valid one still tested, drop info
        self.select(*v)
        self.assertTrue(w.Error.invalid_vars.is_shown())
        self.assertEqual(len(self.get_output(w.Outputs.test_results)), 1)
        self.select(v[2])
        self.assertFalse(w.Error.invalid_vars.is_shown())
        self.assertFalse(w.Information.dropped.is_shown())

        # normality warning raised and cleared with alpha
        self.send_signal(w.Inputs.data, self.iris)
        self.select(self.pl)
        self.assertTrue(w.Warning.not_normal.is_shown())
        w.alpha = 1e-12
        w.commit.now()
        self.assertFalse(w.Warning.not_normal.is_shown())

        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Warning.not_normal.is_shown())

    def test_output_rows_match_table(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.select(self.sl, self.sw, self.pl)
        out = self.get_output(w.Outputs.test_results)
        self.assertEqual(len(out), w.test_model.rowCount())
        self.assertEqual([str(m[0]) for m in out.metas],
                         [r[0] for r in w.desc_model.rows])  # same order as the Test rows
        self.assertEqual(w.test_model.headers[0], "mu0")

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

    def test_formatting(self):
        self.assertEqual(u.fmt_p(0.0004), "<0.001")
        self.assertEqual(u.fmt_p(0.0756), "0.076")
        self.assertEqual(u.fmt_p(float("nan")), "")
        self.assertEqual(u.fmt_num(float("nan")), "")
        self.assertEqual(u.fmt_num(5.0783), "5.078")
