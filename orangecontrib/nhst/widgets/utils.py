"""Helpers shared by the NHST widgets: number formatting, a read-only table model,
pyqtgraph bits and builders for output Tables."""
import math

import numpy as np
import pyqtgraph as pg
from AnyQt.QtCore import Qt, QAbstractTableModel, QModelIndex
from AnyQt.QtGui import QFont, QPalette
from AnyQt.QtWidgets import QHeaderView, QTableView, QApplication

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.utils.colorpalettes import DefaultRGBColors


# ---- formatting -------------------------------------------------------------

def _missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def fmt_num(x, digits=3):
    """3-decimal number; NaN/None -> empty cell; ints stay ints; text unchanged."""
    if isinstance(x, str):
        return x
    if _missing(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{digits}f}"


def fmt_p(p):
    """p-value rule: '<0.001' below 0.001, else 3 decimals; NaN -> empty."""
    if _missing(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(low, high):
    return "" if _missing(low) or _missing(high) else f"[{fmt_num(low)}, {fmt_num(high)}]"


# ---- table model / view -----------------------------------------------------

class ResultsModel(QAbstractTableModel):
    """Read-only table of raw values (str, int, float, NaN).

    p_columns: column indices formatted with `fmt_p`.
    right_columns: text columns (e.g. CI strings) that are right-aligned anyway.
    editable_columns / on_edit(row, column, float): optional numeric editing.
    """

    def __init__(self, headers=(), rows=(), p_columns=(), right_columns=(),
                 editable_columns=(), on_edit=None, parent=None):
        super().__init__(parent)
        self.headers, self.rows = list(headers), [list(r) for r in rows]
        self.p_columns, self.right_columns = set(p_columns), set(right_columns)
        self.editable_columns, self.on_edit = set(editable_columns), on_edit

    def set_rows(self, rows, headers=None):
        self.beginResetModel()
        if headers is not None:
            self.headers = list(headers)
        self.rows = [list(r) for r in rows]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        r, c = index.row(), index.column()
        v = self.rows[r][c]
        if role == Qt.DisplayRole:
            return fmt_p(v) if c in self.p_columns else fmt_num(v)
        if role == Qt.EditRole:
            return "" if _missing(v) else repr(v)
        if role == Qt.TextAlignmentRole:
            left = isinstance(v, str) and c not in self.right_columns
            return int((Qt.AlignLeft if left else Qt.AlignRight) | Qt.AlignVCenter)
        return None

    def flags(self, index):
        f = super().flags(index) & ~Qt.ItemIsEditable
        if index.column() in self.editable_columns:
            f |= Qt.ItemIsEditable
        return f

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or index.column() not in self.editable_columns:
            return False
        try:
            value = float(str(value).replace(",", "."))
        except ValueError:
            return False
        if not math.isfinite(value):
            return False
        self.rows[index.row()][index.column()] = value
        self.dataChanged.emit(index, index)
        if self.on_edit is not None:
            self.on_edit(index.row(), index.column(), value)
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.headers[section]
            if role == Qt.TextAlignmentRole:
                left = (bool(self.rows) and isinstance(self.rows[0][section], str)
                    and section not in self.right_columns)  # match cells
                return int((Qt.AlignLeft if left else Qt.AlignRight)
                           | Qt.AlignVCenter)
        return None


def make_table_view(model, selectable_rows=False):
    view = QTableView()
    view.setModel(model)
    view.verticalHeader().hide()
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    view.setAlternatingRowColors(True)
    view.setEditTriggers(QTableView.DoubleClicked | QTableView.EditKeyPressed
                         | QTableView.AnyKeyPressed)
    view.setSelectionBehavior(QTableView.SelectRows)
    view.setSelectionMode(QTableView.SingleSelection if selectable_rows
                          else QTableView.NoSelection)
    model.modelReset.connect(lambda: fit_height(view))
    fit_height(view)
    return view


def fit_height(view, min_rows=1):
    """Make a table view just tall enough for its rows."""
    n = max(view.model().rowCount(), min_rows)
    view.setFixedHeight(view.horizontalHeader().sizeHint().height()
                        + n * view.verticalHeader().defaultSectionSize() + 4)


# ---- plotting ---------------------------------------------------------------

def _line_color(widget):
    """Grey used for frames, axes and outlines (theme-aware), instead of black."""
    return widget.palette().color(QPalette.Mid).lighter(125)


def _text_color(widget):
    return widget.palette().color(QPalette.WindowText)


def new_plot(parent, xlabel="", ylabel="Frequency"):
    """pyqtgraph PlotWidget in the style of Orange's distribution plots."""
    plot = pg.PlotWidget(parent, background=None)
    plot.setMinimumHeight(180)
    plot.setMouseEnabled(False, False)
    plot.hideButtons()
    item = plot.getPlotItem()
    item.setLabel("bottom", xlabel)
    item.setLabel("left", ylabel)
    for side in ("bottom", "left"):
        item.getAxis(side).setPen(pg.mkPen(_line_color(plot)))
        item.getAxis(side).setTextPen(pg.mkPen(_text_color(plot)))
    return plot


def add_histogram(plot, x, bins="auto"):
    counts, edges = np.histogram(x, bins=bins)
    color = DefaultRGBColors[0]
    brush = pg.mkBrush(*color.getRgb()[:3], 160) if hasattr(color, "getRgb") \
        else pg.mkBrush(color)
    item = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts,
                           brush=brush, pen=pg.mkPen(_line_color(plot), width=0.5))
    plot.addItem(item)
    return item


def _rgb(i):
    c = DefaultRGBColors[i]
    return tuple(c.getRgb()[:3]) if hasattr(c, "getRgb") else tuple(c)[:3]


def add_density(plot, x, color_index, name=None, points=200):
    """Filled Gaussian-KDE curve (Scott bandwidth) of x; returns the curve item."""
    from scipy.stats import gaussian_kde  # plotting only; tests live in stats/
    x = np.asarray(x, float)
    if np.ptp(x) == 0:  # KDE undefined for constant data
        return None
    pad = 3 * x.std(ddof=1)
    grid = np.linspace(x.min() - pad, x.max() + pad, points)
    r, g, b = _rgb(color_index)
    curve = plot.plot(grid, gaussian_kde(x)(grid), name=name, pen=pg.mkPen((r, g, b), width=2),
                      fillLevel=0, brush=pg.mkBrush(r, g, b, 70))
    return curve


def add_key(plot, name, dashed=False, color=None, width=2):
    """Legend-only entry (empty curve) so reference lines can be named."""
    item = plot.getPlotItem()
    if item.legend is None:
        plot.addLegend(offset=(-10, 10))
    pen = pg.mkPen(color or (_text_color(plot) if dashed else _line_color(plot)), width=width,
                   style=Qt.DashLine if dashed else Qt.SolidLine)
    plot.plot([], [], pen=pen, name=name)


def add_vline(plot, x, dashed=False, width=2, color=None):
    pen = pg.mkPen(color or (_text_color(plot) if dashed else _line_color(plot)), width=width,
                   style=Qt.DashLine if dashed else Qt.SolidLine)
    line = pg.InfiniteLine(pos=x, angle=90, pen=pen, movable=False)
    plot.addItem(line)
    return line


def add_band(plot, low, high):
    color = DefaultRGBColors[1]
    r, g, b = color.getRgb()[:3] if hasattr(color, "getRgb") else tuple(color)[:3]
    band = pg.LinearRegionItem((low, high), movable=False,
                               brush=pg.mkBrush(r, g, b, 90), pen=pg.mkPen(None))
    band.setZValue(-5)
    plot.addItem(band, ignoreBounds=True)  # band must not drive the autorange
    return band


# ---- output Tables ----------------------------------------------------------

def results_table(attr_names, rows, meta_names=(), meta_rows=()):
    """(Orange converts inf, e.g. an open one-sided CI end, to NaN.)
    Table with continuous attributes `attr_names` (rows: sequences of floats)
    and string metas `meta_names` (meta_rows: sequences of str); NaN stays NaN."""
    domain = Domain([ContinuousVariable(n) for n in attr_names],
                    metas=[StringVariable(n) for n in meta_names])
    X = np.array(rows, dtype=float).reshape(len(rows), len(attr_names))
    X[np.isinf(X)] = np.nan
    M = np.array(meta_rows, dtype=object).reshape(len(rows), len(meta_names))
    return Table.from_numpy(domain, X, metas=M)


def add_boxplots(plot, groups, labels):
    """Vertical box plot per group: box = quartiles, line = median, diamond = mean,
    whiskers = range. Group i sits at x = i; x ticks carry `labels`. All coordinates finite."""
    text = _line_color(plot)
    for i, x in enumerate(groups):
        x = np.asarray(x, float)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        r, g, b = _rgb(i)
        plot.addItem(pg.BarGraphItem(x=[i], width=0.5, y0=[q1], y1=[q3],
                                     brush=pg.mkBrush(r, g, b, 160), pen=pg.mkPen(text)))
        plot.plot([i - 0.25, i + 0.25], [med, med], pen=pg.mkPen(text, width=3))
        for lo, hi in ((x.min(), q1), (q3, x.max())):
            plot.plot([i, i], [lo, hi], pen=pg.mkPen(text))
        plot.addItem(pg.ScatterPlotItem([i], [x.mean()], symbol="d", size=11,
                                        brush=pg.mkBrush("w"), pen=pg.mkPen(text)))
    plot.getPlotItem().getAxis("bottom").setTicks([list(enumerate(labels))])
    plot.setXRange(-0.7, len(groups) - 0.3)


MARKERS = ("o", "s", "t", "d", "star", "p", "h", "+")  # distinct shapes, cycled


def add_mean_plot(plot, means, lows, highs, labels, grand_mean=None):
    """Group means with CI whiskers (one marker shape per group) and an optional dashed
    horizontal grand-mean line. Group i sits at x = i; x ticks carry `labels`.
    Caller must pass finite values."""
    text = _line_color(plot)
    k = len(means)
    x = np.arange(k, dtype=float)
    means, lows, highs = (np.asarray(a, float) for a in (means, lows, highs))
    if grand_mean is not None:
        if plot.getPlotItem().legend is None:
            plot.addLegend(offset=(-10, 10))
        plot.plot([-0.5, k - 0.5], [grand_mean, grand_mean], name="Grand mean",
                  pen=pg.mkPen(_text_color(plot), width=1.5, style=Qt.DashLine))
    plot.addItem(pg.ErrorBarItem(x=x, y=means, top=highs - means, bottom=means - lows,
                                 beam=0.15, pen=pg.mkPen(text, width=1.5)))
    plot.addItem(pg.ScatterPlotItem(
        x, means, symbol=[MARKERS[i % len(MARKERS)] for i in range(k)], size=12,
        brush=[pg.mkBrush(*_rgb(i)) for i in range(k)], pen=pg.mkPen(text)))
    plot.getPlotItem().getAxis("bottom").setTicks([list(enumerate(labels))])
    plot.setXRange(-0.7, k - 0.3)


# ---- look -------------------------------------------------------------------

# Qt6's native Windows style draws pane frames and table borders dark; Orange's own look is
# light grey. `widget.setStyleSheet(pane_qss(widget))`; the grey follows light/dark themes.
def pane_qss(widget):
    c = _line_color(widget).name()
    return f"""
QGroupBox {{ border: 1px solid {c}; border-radius: 3px;
            margin-top: 0.6em; padding-top: 0.6em; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                   left: 8px; padding: 0 2px; }}
QTableView {{ border: 1px solid {c}; gridline-color: {c}; }}
QHeaderView::section {{ border: 0; border-right: 1px solid {c};
                       border-bottom: 1px solid {c}; padding: 2px 1px;
                       background: palette(window); }}
"""
