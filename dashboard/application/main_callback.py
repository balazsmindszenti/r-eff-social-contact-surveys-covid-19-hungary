from copy import deepcopy
from datetime import datetime

from dash import Output, Input, State
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from plotly import graph_objects as go

from dashboard.application.data.results import contact_data, daterange, sim
from dashboard.application.main_app import app


cmap = plt.get_cmap('nipy_spectral')


@app.callback(
    [
        Output(component_id="r_eff_plot", component_property="figure"),
        Output(component_id="contact_scatter", component_property="figure"),
        Output(component_id="infected", component_property="children"),
        Output(component_id='recovered', component_property='figure'),
        Output(component_id='seasonality-fig', component_property='figure')
    ],
    [
        Input(component_id='model-selector', component_property='value')
    ],
    [
        State(component_id="r_eff_plot", component_property="figure"),
        State(component_id='contact_scatter', component_property='figure'),
        State(component_id='recovered', component_property='figure'),
        State(component_id='seasonality-fig', component_property='figure'),
        State(component_id='model-storage', component_property='data')
    ]
)
def plot_updater(values, m_fig, cs_fig, r_fig, s_fig, model_storage):
    # clear figs
    cs_fig["data"] = []
    r_fig["data"] = []
    s_fig["data"] = []
    m_fig["data"] = m_fig["data"][0:2]

    latent, infected = None, None
    for i, val in enumerate(values[::-1]):
        print(val)
        sim_h: dict[str, ...] = model_storage[val]

        m_fig["data"].append(go.Scatter())
        m_fig["data"][-1]["x"] = [datetime.fromtimestamp(t) for t in sim_h['timestamps']]
        m_fig["data"][-1]["y"] = sim_h['r_eff_plot']
        m_fig["data"][-1]["name"] = val

        m_fig["data"][-1]["marker"]["color"] = to_hex(cmap(0.5 + i / len(m_fig["data"]) * 0.5))

        bin_edges = np.array(contact_data.start_ts)
        bin_number = np.digitize(sim_h['timestamps'], bin_edges)

        temp_agg_r_eff = pd.DataFrame([sim_h['timestamps'], sim_h['r_eff_plot'], bin_number]).T
        temp_agg_r_eff.columns = ['ts', 'r_eff', 'binnum']
        temp_agg_r_eff = temp_agg_r_eff.groupby('binnum').agg({'ts': 'min', 'r_eff': 'mean'})
        temp_agg_r_eff["contactnum"] = temp_agg_r_eff.index.map(lambda k: contact_data.loc[k]["avg_contactnum"])
        temp_agg_r_eff['date'] = temp_agg_r_eff['ts'].map(lambda t: str(datetime.fromtimestamp(t).date()))

        cs_fig["data"].append(go.Scatter(mode="markers"))
        cs_fig["data"][-1]["x"] = temp_agg_r_eff.r_eff
        cs_fig["data"][-1]["y"] = temp_agg_r_eff.contactnum
        cs_fig["data"][-1]["text"] = temp_agg_r_eff.date
        cs_fig["data"][-1]["hoverinfo"] = 'text'
        cs_fig["data"][-1]["name"] = val

        r_fig["data"].append(go.Scatter(mode="markers"))
        r_fig["data"][-1]["x"] = [datetime.fromtimestamp(t) for t in sim_h['timestamps']]
        r_fig["data"][-1]["y"] = sim_h['rec_ratio']

        s_fig["data"].append(go.Scatter(mode="markers"))
        s_fig["data"][-1]["x"] = [datetime.fromtimestamp(t) for t in sim_h['timestamps']]
        s_fig["data"][-1]["y"] = sim_h['seasonality_values']

        for fig_ in [cs_fig, r_fig, s_fig]:
            fig_["data"][-1]["marker"]["color"] = to_hex(cmap(0.5 + i / len(m_fig["data"]) * 0.5))
            fig_["data"][-1]["showlegend"] = False

        latent = sim_h['init_latent']
        infected = sim_h['init_infected']

    return m_fig, cs_fig, f'Latent + Infected at 2020.09.13.: {latent:.0f} + {infected:.0f}', r_fig, s_fig


@app.callback(
    [
        Output(component_id='model-selector', component_property='options'),
        Output(component_id='model-selector', component_property='value'),
        Output(component_id='model-storage', component_property='data')
    ],
    [
        Input(component_id="datepicker", component_property="value"),
        Input(component_id='seasonality', component_property='value'),
        Input(component_id='is_r_eff_calc', component_property='on'),
        Input(component_id='baseline_r_0', component_property='value'),
        Input(component_id='initial_r_0', component_property='value'),
        Input(component_id='init_ratio_recovered', component_property='value'),
        Input(component_id='seas_select', component_property='value')
    ],
    [
        State(component_id='model-storage', component_property='data'),
        State(component_id='model-selector', component_property='value')
    ]
)
def select_period(datepicker_range, c, is_r_eff_calc, r0,
                  initial_r0, init_ratio_recovered, seas_select,
                  model_storage, model_selector_value):
    # model_storage = model_storage or {}
    start_time = daterange[datepicker_range[0]]
    end_time = daterange[datepicker_range[1]]

    sim.is_r_eff_calc = is_r_eff_calc
    sim.r0 = r0

    sim.initial_r0 = initial_r0
    sim.init_ratio_recovered = init_ratio_recovered
    sim.seasonality_idx = seas_select

    print("Running simulation...")
    print("\tc", c)
    print("\tis_r_eff_calc", is_r_eff_calc)
    print("\tstart", start_time)
    print("\tend", end_time)
    print("\tR_0", r0)

    sim.simulate(
        start_time=start_time,
        end_time=end_time,
        c=c
    )

    sim_to_store = deepcopy(sim)
    label = "simulated R_eff, R_0=%.1f, c=%.1f, immune %s" % (r0, c, str(is_r_eff_calc))
    if sim_to_store.seasonality_idx == 0:
        seas_str = 'cosine'
    elif sim_to_store.seasonality_idx == 1:
        seas_str = 'piecewise linear'
    else:
        seas_str = 'truncated cosine'
    label += \
        ", initial_r0=%.1f, initial ratio=%.3f, seasonality: %s" \
        % (sim_to_store.initial_r0, sim_to_store.init_ratio_recovered, seas_str)
    model_storage[label] = {
        'r_eff_plot': sim_to_store.r_eff_plot,
        'timestamps': sim_to_store.timestamps,
        'rec_ratio': sim_to_store.rec_ratio,
        'seasonality_values': sim_to_store.seasonality_values,
        'init_latent': sim_to_store.init_latent,
        'init_infected': sim_to_store.init_infected
    }

    options = [
        {'label': k, 'value': k}
        for k, v in model_storage.items()
    ]
    keys = list(model_storage.keys())
    if model_selector_value in keys:
        value = model_selector_value
    else:
        value = [keys[0]] if keys else None
    # value = [label]
    return [options, value, model_storage]


@app.callback(
    Output(component_id='contact_matrix', component_property='figure'),
    [Input(component_id='r_eff_plot', component_property='hoverData')],
    [State(component_id='contact_matrix', component_property='figure')]
)
def display_contact_matrix(hoverdata, cm_fig):
    if hoverdata is not None:
        date = hoverdata['points'][0]['x'].split(' ')[0]
        df = pd.DataFrame(sim.data.contact_data_json).set_index("start_date")
        cm_fig['data'][0]['z'] = np.log10(np.array(df.loc[date]['contact_matrix']) + 1e-4)
        cm_fig['data'][0]['text'] = np.array(df.loc[date]['contact_matrix'])
        cm_fig['layout']['title'] = date
    return cm_fig


@app.callback(
    [Output(component_id='filter-elements', component_property='style'),
     Output(component_id='param-settings', component_property='style')],
    [Input(component_id='param-settings', component_property='n_clicks')],
    [State(component_id='filter-elements', component_property='style'),
     State(component_id='param-settings', component_property='style')]
)
def show_hide_labels(n_clicks, style, button_style):
    print("Callback\tshow_hide_labels")
    if style is None:
        style = {}
    if n_clicks is None or n_clicks % 2 == 0:
        style.update({'display': 'none'})
        button_style['background-color'] = 'white'
        return style, button_style
    else:
        style.update({'display': 'block'})
        button_style['background-color'] = 'lightgrey'
        return style, button_style
