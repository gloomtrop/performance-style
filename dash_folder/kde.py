from copy import deepcopy

import dash
import numpy as np
import plotly.figure_factory as ff
import scipy
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.neighbors import KernelDensity

from dash_folder.template import assets_folder
from utils.dash import get_trigger
from utils.loading import load_split

app = dash.Dash(__name__, assets_folder=assets_folder, title='KDE')
RUN_PORT = 8058

PERFORMERS = [f'p{i}' for i in range(11)]
START_PERFORMER = 0

COLUMNS = ['time_onset', 'time_offset', 'velocity_onset', 'velocity_offset', 'duration', 'inter_onset_interval',
           'offset_time_duration']
STD_COLUMNS = [name + '_standardized' for name in COLUMNS]
LABELS = ['Onset Time', 'Offset Time', 'Onset Velocity', 'Offset Velocity', 'Duration', 'Inter Onset Interval',
          'Offset Time Duration']

TEST_AMOUNT = 100
GENERATE_POINTS = True
GENERATED_AMOUNT = 200

BANDWIDTH = 0.6932785319057954
N_SAMPLES = 56
WEIGHTS = np.array(
    [0.9803701032803395, 0.06889461912880954, 0.07764404791412743, 0.3875642462285485, 0.08259937926241562,
     0.6964800043141498, 0.7700041725975063])

SAMPLE_COLOR = '#1df22f'
POPULATION_COLOR = '#13941e'
WRONG_COLOR = '#f22f1d'
COLORS = ['#333F44', '#37AA9C', '#94F3E4', '#33AA99', '#99FFEE', '#333F44', '#37AA9C', '#94F3E4', '#33AA99', '#99FFEE',
          '#333F44', SAMPLE_COLOR]

DIMENSION_OPTIONS = [{'label': l, 'value': v} for (l, v) in zip(LABELS, COLUMNS)]
PERFORMER_OPTIONS = [{'label': p, 'value': p} for p in PERFORMERS]


def compute_entropy_matrix(performer_distributions, sample_distributions):
    return np.array(
        [[scipy.stats.entropy(sample_distributions[i], pdist[i]) for i in range(len(sample_distributions))] for
         pdist in performer_distributions])


def sum_entropies(entropy_matrix, weights):
    return np.dot(entropy_matrix, weights)


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


train, test = load_split()

min_value = train.min()
max_value = train.max()

pdist, pX = get_performer_distributions(train, BANDWIDTH, min_value, max_value, N_SAMPLES)

sample = test[test['performer'] == PERFORMERS[START_PERFORMER]][:TEST_AMOUNT]

sdist, sX = get_sample_distributions(sample, BANDWIDTH, min_value, max_value, N_SAMPLES)

matrix = compute_entropy_matrix(pdist, sdist)
entropies = sum_entropies(matrix, WEIGHTS)
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
    global matrix
    global entropies
    trigger = get_trigger()

    performer_id = PERFORMERS.index(performer)
    dimension_id = list(train.columns).index(dimension)

    updated_colors = deepcopy(COLORS)
    updated_colors[performer_id] = POPULATION_COLOR

    if trigger == 'sample-performer':
        sample = test[test['performer'] == PERFORMERS[performer_id]][:TEST_AMOUNT]
        sdist, sX = get_sample_distributions(sample, BANDWIDTH, min_value, max_value, N_SAMPLES)
        matrix = compute_entropy_matrix(pdist, sdist)

    histograms = pX_transposed[dimension_id] + [sX[dimension_id]]
    names = []
    for i, entropy in enumerate(matrix.transpose()[dimension_id]):
        names.append(f'{PERFORMERS[i]} - {entropy:.2f}')
    names += ['Sa']

    entropies = sum_entropies(matrix, WEIGHTS)
    classified_performer = classify_performer(entropies)

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
