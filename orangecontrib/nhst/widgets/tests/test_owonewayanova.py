import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.nhst.stats.posthoc import games_howell, bonferroni
from orangecontrib.nhst.testing import reference_iris
from orangecontrib.nhst.widgets.owonewayanova import OWOneWayANOVA


def make_table(a, g, y=None, levels=("u", "v", "w")):
    """Numeric a, categorical g (codes, NaN = missing); optional numeric class y."""
    attrs = [ContinuousVariable("a"), DiscreteVariable("g", values=levels)]
    X = np.column_stack([a, g]).astype(float)
    if y is not None:
        return Table.from_numpy(Domain(attrs, ContinuousVariable("y")), X, np.asarray(y, float))
    return Table.from_numpy(Domain(attrs), X)


def two_level_iris():
    t = reference_iris()
    cls = DiscreteVariable("iris", values=tuple(t.domain.class_var.values[:2]))
    return Table.from_numpy(Domain(t.domain.attributes, cls), t.X[:100], t.Y[:100])


class TestOWOneWayANOVA(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWOneWayANOVA)
        self.iris = reference_iris()
        self.sl = self.iris.domain.attributes[0]
        self.cls = self.iris.domain.class_var
        # sepal length by species, as the groups the stats functions get
        y = self.iris.get_column(self.sl)
        g = self.iris.get_column(self.cls)
        self.groups = [y[g == i] for i in range(3)]
        self.labels = list(self.cls.values)

    def send(self, data):
        self.send_signal(self.widget.Inputs.data, data)

    def row(self):
        out = self.get_output(self.widget.Outputs.test_results)
        return {a.name: out.X[0, j] for j, a in enumerate(out.domain.attributes)}

    def ph(self):
        return self.get_output(self.widget.Outputs.post_hoc)

    def set_method(self, i):
        simulate.combobox_activate_index(self.widget.controls.method, i)

    def set_posthoc(self, i):
        simulate.combobox_activate_index(self.widget.controls.posthoc, i)

    def test_defaults_iris(self):
        w = self.widget
        self.send(self.iris)
        self.assertEqual((w.variable, w.grouping), (self.sl, self.cls))
        self.assertEqual(len(self.get_output(w.Outputs.test_results)), 1)
        self.assertEqual(len(self.ph()), 3)
        self.assertFalse(w.ph_caption.isVisibleTo(w))

    def test_numeric_class_is_default_variable(self):
        w = self.widget
        self.send(make_table([1, 2, 3, 4, 5, 6], [0, 1, 2, 0, 1, 2], y=[1, 3, 2, 5, 4, 6]))
        self.assertEqual((w.variable.name, w.grouping.name), ("y", "g"))
        self.assertIsNotNone(self.get_output(w.Outputs.test_results))

    def test_categorical_class_is_default_grouping(self):
        w = self.widget
        dom = Domain([ContinuousVariable("a"), DiscreteVariable("g", values=("u", "v"))],
                     DiscreteVariable("k", values=("p", "q")))
        data = Table.from_numpy(dom, np.array([[1., 0], [2, 1], [3, 0], [4, 1]]),
                                np.array([0., 1, 1, 0]))
        self.send(data)
        self.assertEqual((w.variable.name, w.grouping.name), ("a", "k"))

    def test_none_clears(self):
        w = self.widget
        self.send(self.iris)
        self.send(None)
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.assertIsNone(self.ph())
        for m in (w.desc_model, w.test_model, w.ph_model):
            self.assertEqual(m.rowCount(), 0)
        self.assertEqual(w.assumption_label.text(), "")

    def test_reference_classic(self):
        w = self.widget
        self.send(self.iris)
        out = self.get_output(w.Outputs.test_results)
        self.assertEqual([a.name for a in out.domain.attributes],
                         ["F", "df1", "df2", "p-value", "η²", "ω²"])
        self.assertEqual([m.name for m in out.domain.metas], ["Variable", "Method"])
        self.assertEqual([str(m) for m in out.metas[0]], ["sepal length", "Classic"])
        r = self.row()
        for k, v in {"F": 119.265, "df1": 2, "df2": 147, "η²": 0.619,
                     "ω²": 0.612}.items():
            self.assertAlmostEqual(r[k], v, places=3, msg=k)
        self.assertAlmostEqual(r["p-value"] / 1.67e-31, 1, delta=0.01)
        self.assertEqual(w.test_model.headers,
                         ["F", "df1", "df2", "p", "η²", "ω²"])
        self.assertEqual(w.test_model.rows[0][1:3], [2, 147])  # whole df shown as ints
        d = w.desc_model.rows
        for row, (m, sd) in zip(d, ((5.006, .352), (5.936, .516), (6.588, .636))):
            self.assertEqual(row[1], 50)
            self.assertAlmostEqual(row[2], m, places=3)
            self.assertAlmostEqual(row[3], sd, places=3)
            self.assertAlmostEqual(row[4], sd / np.sqrt(50), places=3)
        text = w.assumption_label.text()
        self.assertIn("residuals: W = 0.988, p = 0.219", text)
        self.assertIn("F = 6.353, p = 0.002", text)

    def test_ss_ms_columns(self):
        w = self.widget
        self.send(self.iris)
        w.include_ss = True
        w.commit.now()
        r = self.row()
        self.assertAlmostEqual(r["SS between"], 63.212, places=3)
        self.assertAlmostEqual(r["SS within"], 38.956, places=3)
        self.assertAlmostEqual(r["MS between"], 63.212 / 2, places=3)
        self.assertAlmostEqual(r["MS within"], 38.956 / 147, places=4)
        self.assertNotIn("SS between", w.test_model.headers)  # table unchanged
        w.report_eta2 = w.report_omega2 = False
        w.commit.now()
        out = self.get_output(w.Outputs.test_results)
        self.assertNotIn("η²", [a.name for a in out.domain.attributes])
        self.assertEqual(w.test_model.headers, ["F", "df1", "df2", "p"])

    def test_reference_welch_and_autoswitch(self):
        w = self.widget
        self.send(self.iris)
        self.set_method(1)
        self.assertEqual(w.posthoc, 1)  # follows the method
        r = self.row()
        self.assertAlmostEqual(r["F"], 138.908, places=3)
        self.assertAlmostEqual(r["df1"], 2)
        self.assertAlmostEqual(r["df2"], 92.21, places=2)
        self.assertAlmostEqual(r["p-value"] / 1.51e-28, 1, delta=0.01)
        # eta2/omega2 stay classic
        self.assertAlmostEqual(r["η²"], 0.619, places=3)
        self.assertEqual(str(self.get_output(w.Outputs.test_results).metas[0][1]), "Welch")
        self.assertFalse(w.ph_caption.isVisibleTo(w))
        self.set_method(0)
        self.assertEqual(w.posthoc, 0)

    def test_autoswitch_respects_user_choice(self):
        w = self.widget
        self.send(self.iris)
        self.set_posthoc(2)  # Bonferroni, explicit
        self.set_method(1)
        self.assertEqual(w.posthoc, 2)
        self.set_method(0)
        self.assertEqual(w.posthoc, 2)
        self.set_posthoc(1)  # Games-Howell chosen under Classic stays on switching to Welch
        self.set_method(1)
        self.assertEqual(w.posthoc, 1)
        self.set_posthoc(0)  # Tukey chosen under Welch is kept when going Classic
        self.set_method(0)
        self.assertEqual(w.posthoc, 0)

    def test_tukey(self):
        self.send(self.iris)
        ph = self.ph()
        self.assertEqual([a.name for a in ph.domain.attributes],
                         ["mean difference", "CI low", "CI high", "adjusted p-value"])
        self.assertEqual([m.name for m in ph.domain.metas],
                         ["Variable", "Method", "Group A", "Group B"])
        self.assertEqual([str(m) for m in ph.metas[0]],
                         ["sepal length", "Tukey HSD", "Iris-setosa", "Iris-versicolor"])
        ref = [(-0.930, -1.174, -0.686, 2.4e-14), (-1.582, -1.826, -1.338, 2.2e-14),
               (-0.652, -0.896, -0.408, 8.3e-09)]
        for row, vals in zip(ph.X, ref):
            for a, b in zip(row[:3], vals[:3]):
                self.assertAlmostEqual(a, b, places=3)
            self.assertAlmostEqual(row[3] / vals[3], 1, delta=0.15)
        self.assertEqual(self.widget.ph_model.rows[0][0], "Iris-setosa - Iris-versicolor")

    def check_vs(self, expected, label):
        w = self.widget
        ph = self.ph()
        self.assertEqual(len(ph), len(expected))
        for row, meta, e in zip(ph.X, ph.metas, expected):
            np.testing.assert_allclose(
                row, [e.mean_diff, e.ci_low, e.ci_high, e.p_adj], rtol=1e-9)
            self.assertEqual((str(meta[2]), str(meta[3])), (e.group_a, e.group_b))
            self.assertEqual(str(meta[1]), label)
        self.assertEqual(w.ph_caption.isVisibleTo(w), label.startswith("Bonferroni"))
        self.assertEqual(len(ph), w.ph_model.rowCount())

    def test_games_howell_and_bonferroni_match_stats(self):
        # cross-check: widget output vs. direct calls of the stats functions
        w = self.widget
        self.send(self.iris)
        self.set_posthoc(1)
        self.check_vs(games_howell(self.groups, self.labels, .95), "Games-Howell")
        self.set_posthoc(2)
        self.check_vs(bonferroni(self.groups, self.labels, False, .95),
                      "Bonferroni (CIs adjusted)")
        w.confidence = 90.0
        self.set_method(1)  # posthoc stays Bonferroni; now Welch pairwise tests
        self.check_vs(bonferroni(self.groups, self.labels, True, .90),
                      "Bonferroni (CIs adjusted)")

    def test_posthoc_none(self):
        w = self.widget
        self.send(self.iris)
        self.set_posthoc(3)
        self.assertIsNone(self.ph())
        self.assertEqual(w.ph_model.rowCount(), 0)
        self.assertFalse(w.ph_caption.isVisibleTo(w))
        self.assertIsNotNone(self.get_output(w.Outputs.test_results))
        w.send_report()

    def test_options_alpha_confidence(self):
        w = self.widget
        self.send(self.iris)
        lo95 = self.ph().X[0, 1]
        w.confidence = 99.0
        w.commit.now()
        self.assertLess(self.ph().X[0, 1], lo95)
        w.alpha = 0.001
        w.commit.now()
        self.assertFalse(w.Warning.unequal_var.is_shown())  # Levene p 0.0023 > 0.001

    def test_two_groups_information(self):
        w = self.widget
        self.send(two_level_iris())
        self.assertTrue(w.Information.two_groups.is_shown())
        self.assertIn("Independent Samples", str(w.Information.two_groups))
        self.assertIsNotNone(self.get_output(w.Outputs.test_results))
        self.assertEqual(len(self.ph()), 1)
        self.send(self.iris)
        self.assertFalse(w.Information.two_groups.is_shown())

    def test_errors(self):
        w = self.widget
        cat = Domain([DiscreteVariable("c", values=("a", "b"))])
        self.send(Table.from_numpy(cat, np.array([[0.], [1]])))
        self.assertTrue(w.Error.no_numeric.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        num = Domain([ContinuousVariable("n")])
        self.send(Table.from_numpy(num, np.array([[0.], [1]])))
        self.assertFalse(w.Error.no_numeric.is_shown())
        self.assertTrue(w.Error.no_categorical.is_shown())
        self.send(self.iris)
        self.assertFalse(w.Error.no_categorical.is_shown())

        # all data in one level (the others empty): fewer than 2 groups
        self.send(make_table([1, 2, 3], [0, 0, 0]))
        self.assertTrue(w.Error.too_few_groups.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.assertIsNone(self.ph())
        self.assertEqual((w.test_model.rowCount(), w.desc_model.rowCount()), (0, 0))
        self.send(self.iris)
        self.assertFalse(w.Error.too_few_groups.is_shown())

        # a group with a single observation, then zero variance
        self.send(make_table([1, 2, 3, 4], [0, 0, 0, 1]))
        self.assertTrue(w.Error.invalid.is_shown())
        self.assertIsNone(self.get_output(w.Outputs.test_results))
        self.assertIsNone(self.ph())
        self.send(make_table([1, 2, 3, 3, 3], [0, 0, 1, 1, 1]))
        self.assertTrue(w.Error.invalid.is_shown())
        self.assertIn("zero variance", str(w.Error.invalid))
        self.send(self.iris)
        self.assertFalse(w.Error.invalid.is_shown())

    def test_missing_dropped(self):
        w = self.widget
        nan = np.nan
        data = make_table([1, 2, 3, 4, 5, nan, 7, 8, 9], [0, 0, 0, 1, 1, 1, nan, 1, 1])
        self.send(data)
        self.assertTrue(w.Information.dropped.is_shown())
        self.assertIn("1 in a", str(w.Information.dropped))
        self.assertIn("1 in g", str(w.Information.dropped))
        self.assertFalse(w.Error.invalid.is_shown())
        self.send(self.iris)
        self.assertFalse(w.Information.dropped.is_shown())

    def test_warnings(self):
        w = self.widget
        self.send(self.iris)
        # Classic + Levene significant on sepal length
        self.assertTrue(w.Warning.unequal_var.is_shown())
        self.assertIn("Welch", str(w.Warning.unequal_var))
        self.assertFalse(w.Warning.not_normal.is_shown())  # residuals p 0.219
        self.set_method(1)
        self.assertFalse(w.Warning.unequal_var.is_shown())
        self.set_method(0)
        self.assertTrue(w.Warning.unequal_var.is_shown())
        w.alpha = 0.001
        w.commit.now()
        self.assertFalse(w.Warning.unequal_var.is_shown())
        # residual non-normality: petal width
        w.alpha = 0.05
        w.variable = self.iris.domain.attributes[3]
        w.commit.now()
        self.assertTrue(w.Warning.not_normal.is_shown())
        self.assertIn("Residuals", str(w.Warning.not_normal))
        w.alpha = 1e-12
        w.commit.now()
        self.assertFalse(w.Warning.not_normal.is_shown())
        self.send(None)
        self.assertEqual(len(w.Warning.active), 0)

    def test_output_rows_match_tables(self):
        w = self.widget
        self.send(self.iris)
        for method in (0, 1):
            self.set_method(method)
            for ph in (0, 1, 2, 3):
                self.set_posthoc(ph)
                self.assertEqual(len(self.get_output(w.Outputs.test_results)),
                                 w.test_model.rowCount())
                n = 0 if ph == 3 else 3
                self.assertEqual(w.ph_model.rowCount(), n)
                self.assertEqual(0 if self.ph() is None else len(self.ph()), n)

    def test_paints(self):
        # regression: non-finite plotted coordinates made pyqtgraph hang while painting
        w = self.widget
        self.send(self.iris)
        w.show()
        for method in (0, 1):
            self.set_method(method)
            for ph in (0, 1, 2, 3):
                self.set_posthoc(ph)
                w.grab()
        self.send(two_level_iris())
        w.grab()

    def test_context_settings(self):
        w = self.widget
        dom = Domain([ContinuousVariable("a"), ContinuousVariable("b"),
                      DiscreteVariable("g", values=("u", "v", "w")),
                      DiscreteVariable("h", values=("x", "y"))])
        data = Table.from_numpy(dom, np.array(
            [[1., 5, 0, 0], [2, 4, 1, 0], [3, 7, 2, 1], [4, 1, 0, 1], [5, 2, 1, 1],
             [6, 3, 2, 0], [7, 9, 0, 1], [8, 8, 1, 0]]))
        a, b, g, h = dom.attributes
        self.send(data)
        self.assertEqual((w.variable, w.grouping), (a, g))
        w.variable, w.grouping = b, h
        w.commit.now()
        self.send(None)
        self.assertIsNone(w.variable)
        self.send(data)
        self.assertEqual((w.variable, w.grouping), (b, h))
        w2 = self.create_widget(OWOneWayANOVA, stored_settings=w.settingsHandler.pack_data(w))
        self.send_signal(w2.Inputs.data, data, widget=w2)
        self.assertEqual((w2.variable, w2.grouping), (b, h))

    def test_settings_persist(self):
        w = self.widget
        self.send(self.iris)
        self.set_method(1)
        self.set_posthoc(2)
        w.include_ss = True
        w2 = self.create_widget(OWOneWayANOVA, stored_settings=w.settingsHandler.pack_data(w))
        self.assertEqual((w2.method, w2.posthoc, w2.include_ss), (1, 2, True))

    def test_send_report(self):
        self.send(self.iris)
        self.widget.send_report()
        self.send(None)
        self.widget.send_report()

    def test_preview_block_path(self):
        # what `WidgetPreview(OWOneWayANOVA).run(Table("iris"))` does, minus the event loop
        w = self.create_widget(OWOneWayANOVA)
        self.send_signal(w.Inputs.data, Table("iris"), widget=w)
        self.assertIsNotNone(self.get_output(w.Outputs.test_results, widget=w))
