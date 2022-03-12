from copy import deepcopy
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash_folder.template import assets_folder, colors
import plotly.figure_factory as ff
from sklearn.neighbors import KernelDensity
from utils import paths
from utils.dash import get_trigger
import os
import pandas as pd
import numpy as np
import scipy

app = dash.Dash(__name__, assets_folder=assets_folder, title='KDE')
RUN_PORT = 8058

PERFORMERS = [f'p{i}' for i in range(11)]
START_PERFORMER = 0

COLUMNS = ['time_onset', 'time_offset', 'velocity_onset', 'velocity_offset', 'duration', 'inter_onset_interval', 'offset_time_duration']
STD_COLUMNS = [name + '_standardized' for name in COLUMNS]
LABELS = ['Onset Time', 'Offset Time', 'Onset Velocity', 'Offset Velocity', 'Duration', 'Inter Onset Interval', 'Offset Time Duration']

TEST_AMOUNT = 100
GENERATE_POINTS = True
GENERATED_AMOUNT = 200

BANDWIDTH = 0.1
N_SAMPLES = 100

SAMPLE_COLOR = '#1df22f'
POPULATION_COLOR = '#13941e'
WRONG_COLOR = '#f22f1d'
COLORS = ['#333F44', '#37AA9C', '#94F3E4', '#33AA99', '#99FFEE', '#333F44', '#37AA9C', '#94F3E4', '#33AA99', '#99FFEE',
          '#333F44', SAMPLE_COLOR]

DIMENSION_OPTIONS = [{'label': l, 'value': v} for (l, v) in zip(LABELS, COLUMNS)]
PERFORMER_OPTIONS = [{'label': p, 'value': p} for p in PERFORMERS]


def load_data(piece='D960', split=0.8, filter='unstd'):
    training_data = pd.DataFrame()
    test_data = pd.DataFrame()
    deviations_path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'deviations')
    deviations_names = paths.get_files(deviations_path)
    for deviation in deviations_names:
        data_path = os.path.join(deviations_path, deviation)
        performer = deviation.split('-')[0]
        data = pd.read_json(data_path)
        length = int(data.shape[0] * split)
        data['performer'] = performer
        training_data = pd.concat([training_data, data[:length]])
        test_data = pd.concat([test_data, data[length:]])

    # Filtering
    if filter == 'std':
        columns = STD_COLUMNS + ['performer']
    elif filter == 'unstd':
        columns = COLUMNS + ['performer']
    else:
        columns = COLUMNS + STD_COLUMNS + ['performer']

    return training_data[columns], test_data[columns]


def compute_entropies(sample_distributions, performer_distributions):
    return np.array(
        [[scipy.stats.entropy(sample_distributions[i], pdist[i]) for i in range(len(sample_distributions))] for
         pdist in performer_distributions])


def get_sample_distributions(df, bandwidth, min_value, max_value, n_samples):
    sample_distributions = []
    Xs = []
    for column in df.columns:
        if column != 'performer':
            X = df[column].dropna().to_numpy()
            Xs.append(X)
            X = X.reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
            x = np.linspace(min_value[column], max_value[column], n_samples)
            log = kde.score_samples(x.reshape(-1, 1))
            sample_distributions.append(np.exp(log))
    return np.array(sample_distributions), np.array(Xs)


def get_performer_distributions(df, bandwidth, min_value, max_value, n_samples):
    ds = []
    Xs = []
    for performer in PERFORMERS:
        mask = df['performer'] == performer
        data = df[mask]
        performer_ds = []
        performer_Xs = []
        for column in data.columns:
            if column != 'performer':
                X_list = data[column].dropna().to_numpy()
                X = X_list.reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
                x = np.linspace(min_value[column], max_value[column], n_samples)
                log = kde.score_samples(x.reshape(-1, 1))
                performer_ds.append(np.exp(log))
                if GENERATE_POINTS:
                    performer_Xs.append(kde.sample(GENERATED_AMOUNT).flatten())
                else:
                    performer_Xs.append(X_list)
        ds.append(np.array(performer_ds))
        Xs.append(performer_Xs)
    return np.array(ds), Xs


def classify_performer(entropies):
    index = np.argmin(np.array(entropies))
    return PERFORMERS[index]


train, test = load_data()

min_value = train.min()
max_value = train.max()

pdist, pX = get_performer_distributions(train, BANDWIDTH, min_value, max_value, N_SAMPLES)

sample = test[test['performer'] == PERFORMERS[START_PERFORMER]][:TEST_AMOUNT]

sdist, sX = get_sample_distributions(sample, BANDWIDTH, min_value, max_value, N_SAMPLES)

entropies = compute_entropies(sdist, pdist)
pX_transposed = list(map(list, zip(*pX)))

app.layout = html.Div([
    dcc.Graph(
        id='kde',
        style=dict(
            height='85vh',
            width='90vw'
        )),
    html.Div([
        html.Div(id='classification'),
        dcc.Dropdown(
            id='sample-performer',
            clearable=False,
            value=PERFORMER_OPTIONS[0]['value'],
            options=PERFORMER_OPTIONS,
            style={
                'width': '20vw'
            }
        ),
        dcc.Dropdown(
            id='dimension-dd',
            clearable=False,
            value=DIMENSION_OPTIONS[0]['value'],
            options=DIMENSION_OPTIONS,
            style={
                'width': '20vw'
            }
        ),

    ],
        style={
            'display': 'flex',
            'justify-content': 'space-evenly',
            'align-items': 'center',
            'width': '100vw',
            'height': '15vh'
        })
], style={
    'display': 'flex',
    'flex-direction': 'column',
    'align-items': 'center',
    'height': '100vh',
    'width': '100vw',
})


@app.callback(Output('kde', 'figure'),
              Output('classification', 'children'),
              Output('classification', 'style'),
              Input('dimension-dd', 'value'),
              Input('sample-performer', 'value'))
def change_piece_options(dimension, performer):
    global sX
    global entropies
    trigger = get_trigger()

    performer_id = PERFORMERS.index(performer)
    dimension_id = list(train.columns).index(dimension)

    updated_colors = deepcopy(COLORS)
    updated_colors[performer_id] = POPULATION_COLOR

    if trigger == 'sample-performer':
        sample = test[test['performer'] == PERFORMERS[performer_id]][:TEST_AMOUNT]
        sdist, sX = get_sample_distributions(sample, BANDWIDTH, min_value, max_value, N_SAMPLES)
        entropies = compute_entropies(sdist, pdist)

    histograms = pX_transposed[dimension_id] + [sX[dimension_id]]
    names = []
    for i, entropy in enumerate(entropies.transpose()[dimension_id]):
        names.append(f'{PERFORMERS[i]} - {entropy:.2f}')
    names += ['Sa']

    summed_entropies = [sum(e) for e in entropies]
    classified_performer = classify_performer(summed_entropies)

    if classified_performer == performer:
        style = {
            'border-color': SAMPLE_COLOR
        }
    else:
        style = {
            'border-color': WRONG_COLOR
        }

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(histograms, names, show_hist=False, colors=updated_colors)
    return fig, classified_performer, style


if __name__ == '__main__':
    app.run_server(debug=True, port=RUN_PORT)
