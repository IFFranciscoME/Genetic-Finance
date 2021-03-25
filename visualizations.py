
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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns


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

def heat_corr(Vs):

    plt.figure(figsize=(15,15))

    sns.heatmap(Vs.corr(), cmap='RdBu_r', annot=True, center=0.0)

    plt.show()

Vtitle = 'nan0'


