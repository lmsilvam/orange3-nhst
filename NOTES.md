# NOTES: verified API facts (Orange 3.41.0, orange-widget-base 4.29.0)

Environment
- venv `.venv` built from miniconda Python 3.12 (system 3.14 not used: Qt wheels risk).
- `Orange3` does NOT pull a Qt binding. Installed `PyQt6 PyQt6-WebEngine` separately; `from Orange.widgets...` imports fine with `QT_QPA_PLATFORM=offscreen`.
- numpy 2.5.5, scipy 1.18.1, pyqtgraph 0.14.0.

Auto-apply (differs from the prompt's wording)
- `gui.auto_apply(widget, master, value="auto_apply", **kw)` and `gui.auto_send(...)` both wrap `gui.auto_commit(widget, master, value, label, auto_label, box=False, ...)` in `orangewidget/gui.py`. Labels: "Apply" / "Apply Automatically".
- Orange widgets in 3.41 use `gui.auto_send(self.buttonsArea, self, "auto_commit")` with `auto_commit = Setting(True)`, and a method decorated `@gui.deferred def commit(self)`. Trigger with `self.commit.deferred()` (respects auto) or `self.commit.now()` (forces).
- `self.buttonsArea` exists for the bottom button row.

Summaries / reports
- `set_input_summary(summary, details="", icon=..., format=...)` and `set_output_summary(...)` exist on `OWBaseWidget` (`orangewidget/widget.py`).
- `send_report()` is overridden on the widget; reports use `self.report_table(view)` etc. (see `owfeaturestatistics.py`).
- `graph_name = "plot"` class attribute names the graph for Save Image (`owdistributions.py`).

Packaging (from biolab/orange3-example-addon and orange3-associate)
- `setup.py` with `namespace_packages=['orangecontrib']`, `find_packages()`, `package_data={"orangecontrib.<x>.widgets": ["icons/*.svg"]}`.
- `orangecontrib/` has no `__init__.py` in associate (implicit namespace); `orangecontrib/<x>/__init__.py` exists.
- Entry points: `orange3.addon`, `orange.widgets` (`'Category = pkg.widgets'`), `orange.canvas.help` (`html-index = pkg.widgets:WIDGET_HELP_PATH`).
- Category `__init__.py` defines `ICON`, `BACKGROUND`, `WIDGET_HELP_PATH` (tuple of `(url_template, url_or_"")`).

Tests
- `Orange.widgets.tests.base.WidgetTest` (subclass of `WidgetTestBase`): `create_widget`, `send_signal`, `get_output`.

Still to verify while building widgets
- `gui` helper signatures actually used, `DomainModel`, `WidgetPreview.run`, whether `namespace_packages` still works with current setuptools (else switch to implicit namespace + pyproject).
