from bokeh.io import output_file, show, output_notebook
from bokeh.palettes import Spectral5, Turbo256
from bokeh import palettes
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import FactorRange
from bokeh.layouts import column, row
from bokeh.io import export_png


def grouped_bar(d, x_super_name, x_sub_name, y_name, title=None, factor_range=None):
    x = [(x1, x2) for x1, x2 in zip(d[x_super_name], d[x_sub_name])]
    assert len(x) == len(list(set(x))), x
    y = d[y_name]
    factor_range = factor_range or FactorRange(*sorted(x))
    x_sub_unique = list(set([x_[1] for x_ in x]))

    if 'colors' not in d:
        cmap = Spectral5 if len(x_sub_unique) <= 5 else Turbo256
        colors = [cmap[x_sub_unique.index(x_[1])] for x_ in x]
    else:
        colors = d['colors']
    source = ColumnDataSource(
        data=dict(x=x, y=y, color=colors, x_super=d[x_super_name], x_sub=d[x_sub_name]))

    tooltips = [(f"{(x_super_name, x_sub_name)}", "@x"), (f"{x_super_name}", "@x_super"),
                (f"{x_sub_name}", "@x_sub"), (f"{y_name}", "@y")]

    p = get_blank_figure(x_range=factor_range,
                         title=title or f"{x_super_name} {y_name} by {x_sub_name}",
                         tooltips=tooltips,
                         tools="save,pan,wheel_zoom,box_zoom,reset")  # toolbar_location=None,

    p.vbar(x='x', top='y', width=0.9, color='color', source=source)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xaxis.major_label_orientation = 1
    p.xgrid.grid_line_color = None

    return p


def get_blank_figure(title, **kwargs):
    plot_height = kwargs.pop('plot_height', 250)
    p = figure(plot_height=plot_height, title=title, **kwargs)
    return p