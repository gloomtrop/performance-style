import dash
import plotly.figure_factory as ff
from dash import dcc
from dash import html

from dash_folder.template import assets_folder
from datasets.labelling import LabelledDataSet

app = dash.Dash(__name__, assets_folder=assets_folder, title='KDE')
RUN_PORT = 8058
dataset = LabelledDataSet()
histograms = [list(map(lambda x: x.shape[0], dataset.x))]
colors = COLORS = ['#37AA9C', '#94F3E4', '#33AA99', '#99FFEE']
fig = ff.create_distplot(histograms, group_labels=['Segment length'], show_hist=False, colors=colors)

app.layout = html.Div([
    dcc.Graph(
        id='kde',
        figure=fig,
        style=dict(
            height='100vh',
            width='100vw'
        )),
], style={
    'display': 'flex',
    'flex-direction': 'column',
    'align-items': 'center',
    'height': '100vh',
    'width': '100vw',
})

if __name__ == '__main__':
    app.run_server(debug=True, port=RUN_PORT)
