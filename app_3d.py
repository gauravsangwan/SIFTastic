import dash
from dash import dcc
from dash import html
import json
import plotly.graph_objects as go
import os

# Get a list of all JSON files in the "3dRecons" folder
json_files = [f for f in os.listdir('3dRecons') if f.endswith('.json')]

# Create a Dash app to display the figure
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("3D Reconstruction"),
    html.P("The following application allows you to visualize the 3D reconstruction for the data extracted for the following shown monument The Basilica of the Sacred Heart of Paris"),
    html.P("The dropdown option 3d_reconstruction shows the 3d reconstruction of the monument using the below shown mapping images"),
    html.Img(src='App_images/mapping.jpg', style={'height': '20%', 'width': '20%'}),
    html.P("The dropdown option Localisation1 and 2 shows the mapping of the following query image to the original 3d reconstruction of the monument"),
    html.Img(src='App_images/query.jpg', style={'height': '20%', 'width': '20%'}),
    html.P("Further shows the 3d reconstruction of a house in snowy environment as shown "),
    html.Img(src='App_images/house.jpg', style={'height': '20%', 'width': '20%'}),
    dcc.Dropdown(
        id='json-dropdown',
        options=[{'label': f, 'value': f} for f in json_files],
        value=json_files[0]  # default value
    ),
    dcc.Graph(
        id='plot',
        figure={}  # initialize with an empty figure
    )
])

@app.callback(
    dash.dependencies.Output('plot', 'figure'),
    [dash.dependencies.Input('json-dropdown', 'value')]
)
def update_plot(selected_json):
    # Load the selected JSON data
    with open(os.path.join('3dRecons', selected_json), 'r') as f:
        figure_data = json.load(f)

    # Create the figure using the JSON data
    fig = go.Figure(**figure_data)
    fig.update_layout(width=800, height=600)  # change figure size

    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port = "8070")