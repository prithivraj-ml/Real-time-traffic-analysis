# dashboard.py
import dash
from dash import dcc, html, Input, Output
import requests
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time Traffic Dashboard"),
    dcc.Input(id='junction-input', type='number', placeholder='Enter Junction Number'),
    dcc.Input(id='hours-ahead', type='number', placeholder='Hours Ahead for Prediction'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='prediction-output'),

    dcc.Graph(id='traffic-graph')
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [Input('junction-input', 'value'), Input('hours-ahead', 'value')]
)
def update_output(n_clicks, junction, hours_ahead):
    if n_clicks > 0:
        response = requests.get(f'http://127.0.0.1:5000/predict?junction={junction}&hours_ahead={hours_ahead}')
        data = response.json()
        return f"Predicted traffic: {data['predicted_traffic']} vehicles"

if __name__ == '__main__':
    app.run_server(debug=True)
