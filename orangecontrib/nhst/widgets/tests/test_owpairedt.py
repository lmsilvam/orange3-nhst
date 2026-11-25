import numpy as np
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.nhst.testing import reference_iris
from orangecontrib.nhst.widgets.owpairedt import OWPairedT


def num_table(**cols):
    dom = Domain([ContinuousVariable(n) for n in cols])
    return Table.from_numpy(dom, np.column_stack(list(cols.values())).astype(float))


class TestOWPairedT(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPairedT)
        self.iris = reference_iris()
        self.sl, self.sw, self.pl, self.pw = self.iris.domain.attributes[:4]

    def pick(self, v1, v2):
        self.widget.var1, self.widget.var2 = v1, v2
        self.widget.commit.now()

    def row(self):
        out = self.get_output(self.widget.Outputs.test_results)
        return {a.name: out.X[0, j] for j, a in enumerate(out.domain.attributes)}

    def test_iris_default(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual((self.widget.var1, self.widget.var2), (self.sl, self.sw))
        self.assertEqual(len(self.get_output(self.widget.Outputs.test_results)), 1)

    def test_reference_values(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.pick(self.sl, self.pl)
        out = self.get_output(w.Outputs.test_results)
        self.assertEqual([a.name for a in out.domain.attributes],
                         ["N", "mean1", "mean2", "mean difference", "SD difference", "t",
                          "df", "p-value", "CI low", "CI high", "Cohen's dz"])
        self.assertEqual([m.name for m in out.domain.metas], ["Pair"])
        self.assertEqual(str(out.metas[0][0]), "sepal length - petal length")
        r = self.row()
        self.assertEqual((r["N"], r["df"]), (150, 149))
        for k, v in {"mean difference": 2.085, "SD difference": 1.120, "t": 22.813,
                     "CI low": 1.905, "CI high": 2.266, "Cohen's dz": 1.863}.items():
            self.assertAlmostEqual(r[k], v, places=3)
        self.assertAlmostEqual(r["p-value"] / 1.80e-50, 1, delta=0.01)
        self.assertAlmostEqual(w.desc_model.rows[2][4], 0.091, places=3)
        self.assertEqual(w.desc_model.rows[2][0], "Difference")
        self.assertEqual(w.test_model.headers[0], "t")  # no name column
        self.assertIn("W = 0.899", w.normality_label.text())
        self.assertIn("differences", w.normality_label.text())

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
        self.pick(self.sl, self.pl)
        two = self.row()
        w.alternative = 2  # mean difference > 0 in the data: "less" has p near 1
        w.commit.now()
        less = self.row()
        self.assertAlmostEqual(less["p-value"], 1 - two["p-value"] / 2, places=9)
        self.assertTrue(np.isnan(less["CI low"]))  # open end: Orange turns inf into NaN
        w.alternative = 1
        w.commit.now()
        g = self.row()
        self.assertAlmostEqual(g["p-value"], two["p-value"] / 2, delta=1e-12)
        self.assertTrue(np.isnan(g["CI high"]))
        w.alternative = 0
        w.confidence = 90.0
        w.commit.now()
        self.assertGreater(self.row()["CI low"], two["CI low"])
        w.report_dz = False
        w.commit.now()
        out = self.get_output(w.Outputs.test_results)
        self.assertNotIn("Cohen's dz", [a.name for a in out.domain.attributes])
        self.assertEqual(w.test_model.headers[-1], "CI")

    def test_swap(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.pick(self.sl, self.pl)
        before = self.row()
        w._swap()
        self.assertEqual((w.var1, w.var2), (self.pl, self.sl))
        after = self.row()
        self.assertAlmostEqual(after["t"], -before["t"], places=9)
        self.assertAlmostEqual(after["mean difference"], -before["mean difference"])
        out = self.get_output(w.Outputs.test_results)
        self.assertEqual(str(out.metas[0][0]), "petal length - sepal length")

    def test_context_settings(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.pick(self.sw, self.pw)
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.var1)
        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual((w.var1, w.var2), (self.sw, self.pw))
        w2 = self.create_widget(OWPairedT, stored_settings=w.settingsHandler.pack_data(w))
        self.send_signal(w2.Inputs.data, self.iris, widget=w2)
        self.assertEqual((w2.var1, w2.var2), (self.sw, self.pw))

    def test_messages(self):
        w = self.widget
        # fewer than two numeric variables
        dom = Domain([ContinuousVariable("a"), DiscreteVariable("c", values=("a", "b"))])
        self.send_signal(w.Inputs.data, Table.from_numpy(dom, np.array([[1., 0], [2., 1]])))
        self.assertTrue(w.Error.not_enough_numeric.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.send_signal(w.Inputs.data, self.iris)
        self.assertFalse(w.Error.not_enough_numeric.is_shown())

        # same variable
        self.pick(self.sl, self.sl)
        self.assertTrue(w.Error.same_variable.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.pick(self.sl, self.sw)
        self.assertFalse(w.Error.same_variable.is_shown())

        # <2 complete pairs, zero variance of differences, dropped pairs
        data = num_table(a=[1, 2, np.nan, np.nan], b=[np.nan, 5, 1, 2],
                         c=[2, 3, 4, 5], d=[1, 2, 3, 4], e=[1, 2, 4, 9])
        self.send_signal(w.Inputs.data, data)
        a, b, c, d, e = data.domain.attributes
        self.pick(a, b)
        self.assertTrue(w.Error.invalid.is_shown())
        self.assertTrue(w.Information.dropped.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.pick(c, d)  # constant difference
        self.assertTrue(w.Error.invalid.is_shown())
        self.assertFalse(w.Information.dropped.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.pick(e, d)
        self.assertFalse(w.Error.invalid.is_shown())

        # normality warning (on the differences) raised and cleared with alpha
        self.send_signal(w.Inputs.data, self.iris)
        self.pick(self.sl, self.pl)
        self.assertTrue(w.Warning.not_normal.is_shown())
        self.assertIn("differences", str(w.Warning.not_normal))
        w.alpha = 1e-12
        w.commit.now()
        self.assertFalse(w.Warning.not_normal.is_shown())
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Warning.not_normal.is_shown())

    def test_output_rows_match_table(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        out = self.get_output(self.widget.Outputs.test_results)
        self.assertEqual(len(out), self.widget.test_model.rowCount())

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()
