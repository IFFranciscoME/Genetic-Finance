
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Applications of Genetic Methods for Feature Engineering and Hyperparameter Optimization    -- #
# -- -------- for Neural Networks.                                                                       -- #
# -- script: visualizations.py : python script with functions for plots and tables                       -- #
# -- author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/GeneticMethods                                         -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from functools import reduce
from itertools import product

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# -- -------------------------------------------------------- PLOT: OHLC Price Chart with Vertical Lines -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_ohlc(p_ohlc, p_theme=None, p_vlines=None):
    """
    Timeseries Candlestick with OHLC prices and figures for trades indicator

    Requirements
    ------------
    numpy
    pandas
    plotly

    Parameters
    ----------
    p_ohlc: pd.DataFrame
        that contains the following float or int columns: 'timestamp', 'open', 'high', 'low', 'close'

    p_theme: dict
        with the theme for the visualizations

    p_vlines: list
        with the dates where to visualize the vertical lines, format = pd.to_datetime('2020-01-01 22:15:00')
    
    Returns
    -------
    fig_g_ohlc: plotly
        objet/dictionary to .show() and plot in the browser
    
    References
    ----------
    https://plotly.com/python/candlestick-charts/

    """

    # default value for lables to use in main title, and both x and y axisp_fonts
    if p_theme is not None:
        p_labels = p_theme['p_labels']
    else:
        p_theme = dict(p_colors={'color_1': '#6b6b6b', 'color_2': '#ABABAB', 'color_3': '#ABABAB'},
                       p_fonts={'font_title': 18, 'font_axis': 10, 'font_ticks': 10},
                       p_dims={'width': 900, 'height': 400},
                       p_labels={'title': 'OHLC Prices',
                                 'x_title': 'Dates', 'y_title': 'Historical Prices'})

    # tick values calculation for simetry in y axes
    y0_ticks_vals = np.arange(min(p_ohlc['low']), max(p_ohlc['high']),
                              (max(p_ohlc['high']) - min(p_ohlc['low'])) / 10)
    y0_ticks_vals = np.append(y0_ticks_vals, max(p_ohlc['high']))
    y0_ticks_vals = np.round(y0_ticks_vals, 4)

    # Instantiate a figure object
    fig_g_ohlc = go.Figure()

    # Add layer for OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(name='ohlc', x=p_ohlc['timestamp'], open=p_ohlc['open'],
                                        high=p_ohlc['high'], low=p_ohlc['low'], close=p_ohlc['close'],
                                        opacity=0.7))

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=60, pad=20),
                             xaxis=dict(title_text=p_theme['p_labels']['x_title']),
                             yaxis=dict(title_text=p_theme['p_labels']['y_title']))

    # Color and font type for text in axes
    fig_g_ohlc.update_layout(xaxis=dict(titlefont=dict(color=p_theme['p_colors']['color_1']),
                                        tickfont=dict(color=p_theme['p_colors']['color_1'],
                                                      size=p_theme['p_fonts']['font_axis']), showgrid=False),
                             yaxis=dict(zeroline=False, automargin=True, tickformat='.4f',
                                        titlefont=dict(color=p_theme['p_colors']['color_1']),
                                        tickfont=dict(color=p_theme['p_colors']['color_1'],
                                                      size=p_theme['p_fonts']['font_axis']),
                                        showgrid=True, gridcolor='lightgrey', gridwidth=.05))

    # If parameter vlines is used
    if p_vlines is not None:
        # Dynamically add vertical lines according to the provided list of x dates.
        shapes_list = list()
        for i in p_vlines:
            shapes_list.append({'type': 'line', 'fillcolor': p_theme['p_colors']['color_1'],
                                'line': {'color': p_theme['p_colors']['color_1'],
                                         'dash': 'dashdot', 'width': 3},
                                'x0': i, 'x1': i, 'xref': 'x',
                                'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

        # add v_lines to the layout
        fig_g_ohlc.update_layout(shapes=shapes_list)

    # Update layout for the background
    fig_g_ohlc.update_layout(yaxis=dict(tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis']),
                                        tickvals=y0_ticks_vals),
                             xaxis=dict(tickfont=dict(color='grey', size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the y axis
    fig_g_ohlc.update_xaxes(rangebreaks=[dict(pattern="day of week", bounds=['sat', 'sun'])])

    # Update layout for the background
    fig_g_ohlc.update_layout(title_font_size=p_theme['p_fonts']['font_title'],
                             title=dict(x=0.5, text='<b> ' + p_theme['p_labels']['title'] + ' </b>'),
                             yaxis=dict(title=p_theme['p_labels']['y_title'],
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)),
                             xaxis=dict(title=p_theme['p_labels']['x_title'], rangeslider=dict(visible=False),
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)))

    # Final plot dimensions
    fig_g_ohlc.layout.autosize = True
    fig_g_ohlc.layout.width = p_theme['p_dims']['width']
    fig_g_ohlc.layout.height = p_theme['p_dims']['height']

    return fig_g_ohlc

# -- -------------------------------------------------------------------- PLOT: HeatMap Correlation Plot -- #
# -- --------------------------------------------------------------------------------------------------- -- #


def g_heat_corr(p_data, p_double):
    """
    Generates a heatmap correlation matrix with seaborn library

    Parameters
    ----------

    p_data: pd.DataFrame
        With correlation matrix

        p_data = pd.DataFrame(np.random.randn(10, 10))

    p_double: bool
        True: To generate 2 plots (horizontal axis)
        False: To generate 1 centered plot

        p_double = False

     p_annot: bool
        True: To include annotations in the plot
        False: Not to include annotations in the plot

        p_annot = True

    Returns
    -------
        plt = matplotlib plot object

    References
    ----------
        http://seaborn.pydata.org/generated/seaborn.heatmap.html

    """

    # copy of original data
    g_data = p_data.copy()
    
    # mask = np.triu(np.ones_like(g_data, dtype=bool))
    rLT = g_data.where(np.tril(np.ones(g_data.shape)).astype(np.bool_))
    # mask = np.triu(np.ones_like(g_data, dtype=bool))
    nrLT = g_data.where(np.triu(np.ones(g_data.shape)).astype(np.bool_))
   
    g_heat = go.Figure()
    title = 'Correlation Matrix'

    g_heat = g_heat.add_trace(go.Heatmap(showscale=False,
        z = nrLT*1, 
        x = nrLT.columns.values,
        y = nrLT.columns.values,
        xgap = 0,   # Sets the horizontal gap (in pixels) between bricks
        ygap = 0,
        zmin = -1,  # Sets the lower bound of the color domain
        zmax = +1,
        colorscale = ['#FFFFFF', '#FFFFFF']))

    g_heat = g_heat.add_trace(go.Heatmap(showscale=False,
        z = rLT,
        x = rLT.columns.values,
        y = rLT.columns.values,
        zmin = -1,  # Sets the lower bound of the color domain
        zmax = +1,
        xgap = +1,   # Sets the horizontal gap (in pixels) between bricks
        ygap = +1,
        colorscale = 'Blues'))  

    g_heat = g_heat.update_layout(
        title_text=title,
        title_x = 0.5, 
        title_y = 0.90, 
        width = 900, 
        height = 900,
        xaxis_showgrid = False,
        yaxis_showgrid = False,
        yaxis_autorange = 'reversed')

    z = np.array(rLT.values).tolist()

    def get_att(Mx):
        Mx = z
        att=[]
        Mx = Mx
        a, b = len(Mx), len(Mx[0])
        flat_z = reduce(lambda x, y: x + y, Mx)  # Mx.flat if you deal with numpy
        flat_z = [1 if str(i) == 'nan' else i for i in flat_z]
        colors_z = ['#FAFAFA' if i > 0 else '#6E6E6E' for i in flat_z]
        coords = product(range(a), range(b))
        for pos, elem, color in zip(coords, flat_z, colors_z):
            att.append({'font': {'color': color, 'size':9},
                        'text': str(np.round(elem, 2)), 'showarrow': False,
                        'x': pos[1],
                        'y': pos[0]})
        return att
            
    g_heat.update_layout(annotations=get_att(z))

    return g_heat.show()
