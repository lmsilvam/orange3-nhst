import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.nhst.testing import reference_iris
from orangecontrib.nhst.widgets.owindependentt import OWIndependentT


def make_table(a, g, y=None, levels=("u", "v")):
    """Numeric a, categorical g (codes, NaN = missing); optional numeric class y."""
    attrs = [ContinuousVariable("a"), DiscreteVariable("g", values=levels)]
    X = np.column_stack([a, g]).astype(float)
    if y is not None:
        return Table.from_numpy(Domain(attrs, ContinuousVariable("y")), X, np.asarray(y, float))
    return Table.from_numpy(Domain(attrs), X)


def two_level_iris():
    """Reference iris reduced to setosa and versicolor (class has exactly 2 levels)."""
    t = reference_iris()
    cls = DiscreteVariable("iris", values=tuple(t.domain.class_var.values[:2]))
    return Table.from_numpy(Domain(t.domain.attributes, cls), t.X[:100], t.Y[:100])


class TestOWIndependentT(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWIndependentT)
        self.iris3 = reference_iris()
        self.iris = two_level_iris()
        self.sl, self.sw, self.pl, self.pw = self.iris.domain.attributes
        self.cls = self.iris.domain.class_var

    def pick(self, var=None):
        self.widget.variable = var or self.sw
        self.widget.commit.now()

    def row(self):
        out = self.get_output(self.widget.Outputs.test_results)
        return {a.name: out.X[0, j] for j, a in enumerate(out.domain.attributes)}

    def test_iris_default(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        # categorical class -> default grouping; first numeric variable is the test variable
        self.assertEqual((w.variable, w.grouping), (self.sl, self.cls))
        self.assertEqual(w._groups(), ("Iris-setosa", "Iris-versicolor"))
        self.assertEqual(len(self.get_output(w.Outputs.test_results)), 1)

    def test_numeric_class_is_default_variable(self):
        w = self.widget
        data = make_table([1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1], y=[1, 3, 2, 5, 4, 6])
        self.send_signal(w.Inputs.data, data)
        self.assertEqual(w.variable.name, "y")  # not the first numeric ("a")
        self.assertEqual(w.grouping.name, "g")
        self.assertEqual(w._groups(), ("u", "v"))
        self.assertIsNotNone(self.get_output(w.Outputs.test_results))

    def test_categorical_class_is_default_grouping(self):
        w = self.widget
        dom = Domain([ContinuousVariable("a"), DiscreteVariable("g", values=("u", "v"))],
                     DiscreteVariable("k", values=("p", "q")))
        data = Table.from_numpy(dom, np.array([[1., 0], [2, 1], [3, 0], [4, 1]]),
                                np.array([0., 1, 1, 0]))
        self.send_signal(w.Inputs.data, data)
        self.assertEqual(w.grouping.name, "k")
        self.assertEqual(w._groups(), ("p", "q"))

    def test_group_order_is_level_order(self):
        w = self.widget
        # levels ("v", "u"): Group 1 = "v" (code 0), Group 2 = "u"
        data = make_table([10, 11, 12, 1, 2, 3], [0, 0, 0, 1, 1, 1], levels=("v", "u"))
        self.send_signal(w.Inputs.data, data)
        self.assertEqual(w._groups(), ("v", "u"))
        self.assertEqual([r[0] for r in w.desc_model.rows], ["Group 1: v", "Group 2: u"])
        self.assertAlmostEqual(self.row()["mean difference"], 9.0)
        self.assertEqual(self.row()["N1"], 3)

    def test_none_clears(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.assertEqual(w.test_model.rowCount(), 0)
        self.assertEqual(w.desc_model.rowCount(), 0)
        self.assertEqual(w.assumption_label.text(), "")

    def test_reference_welch(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.pick()
        out = self.get_output(w.Outputs.test_results)
        self.assertEqual([a.name for a in out.domain.attributes],
                         ["N1", "N2", "mean1", "mean2", "SD1", "SD2", "t", "df", "p-value",
                          "mean difference", "CI low", "CI high", "Hedges' g"])
        self.assertEqual([m.name for m in out.domain.metas], ["Variable", "Test"])
        self.assertEqual([str(m) for m in out.metas[0]], ["sepal width", "Welch"])
        r = self.row()
        self.assertEqual((r["N1"], r["N2"]), (50, 50))
        for k, v in {"mean1": 3.428, "mean2": 2.770, "SD1": 0.379, "SD2": 0.314,
                     "t": 9.455, "df": 94.698, "mean difference": 0.658, "CI low": 0.520,
                     "CI high": 0.796, "Hedges' g": 1.876}.items():
            self.assertAlmostEqual(r[k], v, places=3, msg=k)
        self.assertAlmostEqual(r["p-value"] / 2.48e-15, 1, delta=0.01)
        self.assertEqual(w.test_model.headers, ["t", "df", "p", "Mean dif.", "CI", "Hedges' g"])
        self.assertEqual(w.test_model.right_columns, {4})
        text = w.assumption_label.text()
        for s in ("setosa: W = 0.972, p = 0.272", "versicolor: W = 0.974, p = 0.338",
                  "F = 0.591, p = 0.444"):
            self.assertIn(s, text)
        self.assertEqual([row[1] for row in w.desc_model.rows], [50, 50])
        self.assertFalse(w.Warning.not_normal.is_shown())

    def test_options(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.pick()
        two = self.row()
        w.alternative = 2  # difference > 0 in the data: "less" has p near 1
        w.commit.now()
        less = self.row()
        self.assertAlmostEqual(less["p-value"], 1 - two["p-value"] / 2, places=9)
        self.assertTrue(np.isnan(less["CI low"]))
        w.alternative = 1
        w.commit.now()
        g = self.row()
        self.assertAlmostEqual(g["p-value"], two["p-value"] / 2, delta=1e-12)
        self.assertTrue(np.isnan(g["CI high"]))
        w.alternative = 0
        w.confidence = 90.0
        w.commit.now()
        self.assertGreater(self.row()["CI low"], two["CI low"])
        w.report_g = False
        w.commit.now()
        out = self.get_output(w.Outputs.test_results)
        self.assertNotIn("Hedges' g", [a.name for a in out.domain.attributes])
        self.assertEqual(w.test_model.headers[-1], "CI")

    def test_radio_labels_show_level_names(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual([b.text() for b in w.alt_box.buttons],
                         ["Group 1 ≠ Group 2", "Group 1 > Group 2", "Group 1 < Group 2"])
        self.assertIn("Iris-setosa", w.groups_label.text())
        self.assertIn("Iris-versicolor", w.groups_label.text())

    def test_paints(self):
        # regression: infinite CI end made pyqtgraph hang while painting
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        w.show()
        for alt in (1, 2, 0):
            w.alternative = alt
            w.commit.now()
            w.grab()

    def test_context_settings(self):
        w = self.widget
        dom = Domain([ContinuousVariable("a"), ContinuousVariable("b"),
                      DiscreteVariable("g", values=("u", "v")),
                      DiscreteVariable("h", values=("x", "y"))])
        data = Table.from_numpy(dom, np.array(
            [[1., 5, 0, 0], [2, 4, 1, 0], [3, 7, 0, 1], [4, 1, 1, 1], [5, 2, 0, 1]]))
        a, b, g, h = dom.attributes
        self.send_signal(w.Inputs.data, data)
        self.assertEqual((w.variable, w.grouping), (a, g))
        w.variable, w.grouping = b, h
        w.commit.now()
        self.assertEqual(w._groups(), ("x", "y"))
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.variable)
        self.send_signal(w.Inputs.data, data)
        self.assertEqual((w.variable, w.grouping), (b, h))
        w2 = self.create_widget(OWIndependentT, stored_settings=w.settingsHandler.pack_data(w))
        self.send_signal(w2.Inputs.data, data, widget=w2)
        self.assertEqual((w2.variable, w2.grouping), (b, h))

    def test_messages(self):
        w = self.widget
        # no numeric / no categorical
        cat = Domain([DiscreteVariable("c", values=("a", "b"))])
        self.send_signal(w.Inputs.data, Table.from_numpy(cat, np.array([[0.], [1]])))
        self.assertTrue(w.Error.no_numeric.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        num = Domain([ContinuousVariable("n")])
        self.send_signal(w.Inputs.data, Table.from_numpy(num, np.array([[0.], [1]])))
        self.assertFalse(w.Error.no_numeric.is_shown())
        self.assertTrue(w.Error.no_categorical.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.send_signal(w.Inputs.data, self.iris)
        self.assertFalse(w.Error.no_categorical.is_shown())

        # more than 2 levels: error, outputs and tables cleared; raised and cleared
        self.send_signal(w.Inputs.data, self.iris3)
        self.assertTrue(w.Error.too_many_levels.is_shown())
        self.assertIn("more than 2 levels", str(w.Error.too_many_levels))
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.assertEqual((w.test_model.rowCount(), w.desc_model.rowCount()), (0, 0))
        self.send_signal(w.Inputs.data, self.iris)
        self.assertFalse(w.Error.too_many_levels.is_shown())
        self.assertIsNotNone(self.get_output(w.Outputs.test_results))
        self.send_signal(w.Inputs.data, self.iris3)
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.too_many_levels.is_shown())

        # fewer than 2 levels
        one = make_table([1, 2, 3], [0, 0, 0], levels=("u",))
        self.send_signal(w.Inputs.data, one)
        self.assertTrue(w.Error.too_few_levels.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.send_signal(w.Inputs.data, self.iris)
        self.assertFalse(w.Error.too_few_levels.is_shown())

        # missing values dropped (per variable), then a group with <2 obs / zero variance
        nan = np.nan
        data = make_table([1, 2, 3, 4, 5, nan, 7], [0, 0, 0, 1, 1, 1, nan])
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(w.Information.dropped.is_shown())
        self.assertIn("1 in a", str(w.Information.dropped))
        self.assertIn("1 in g", str(w.Information.dropped))
        self.assertFalse(w.Error.invalid.is_shown())  # u: 3 obs, v: 2 valid
        data = make_table([1, 2, 3, 4], [0, 0, 0, 1])  # v has a single observation
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(w.Error.invalid.is_shown())
        self.assertFalse(w.Information.dropped.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.assertEqual(w.test_model.rowCount(), 0)
        data = make_table([1, 2, 3, 3, 3], [0, 0, 1, 1, 1])  # v is constant
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(w.Error.invalid.is_shown())

        # Shapiro warning names the group; raised and cleared with alpha
        self.send_signal(w.Inputs.data, self.iris)
        self.pick(var=self.pw)  # petal width: setosa not normal
        self.assertTrue(w.Warning.not_normal.is_shown())
        self.assertIn("Iris-setosa", str(w.Warning.not_normal))
        self.assertFalse(w.Error.invalid.is_shown())
        w.alpha = 1e-12
        w.commit.now()
        self.assertFalse(w.Warning.not_normal.is_shown())
        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Warning.not_normal.is_shown())

    def test_levene_is_information_only(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.pick(var=self.pw)  # unequal variances: Levene significant
        self.assertIn("Levene", w.assumption_label.text())
        self.assertFalse(w.Warning.not_normal.is_shown() and "Levene" in str(w.Warning.not_normal))
        self.assertEqual(len(w.Warning.active), 1)  # only the Shapiro warning

    def test_output_rows_match_table(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        out = self.get_output(self.widget.Outputs.test_results)
        self.assertEqual(len(out), self.widget.test_model.rowCount())

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()
