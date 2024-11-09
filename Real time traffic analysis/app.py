from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import requests

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:prithivraj@localhost:5432/traffic_data'
db = SQLAlchemy(app)

# Initialize Dash app within Flask
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Database model
class TrafficData(db.Model):
    ID = db.Column(db.Integer, primary_key=True)
    DateTime = db.Column(db.DateTime, nullable=False)
    Junction = db.Column(db.Integer, nullable=False)
    Vehicles = db.Column(db.Integer, nullable=False)

# Function to load CSV data into the database
def load_csv_to_db(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Clean up any extra spaces from column names
    df.columns = df.columns.str.strip()
    
    # Check if the 'DateTime' column exists
    if 'DateTime' not in df.columns:
        print("The 'DateTime' column is missing!")
        return
    
    # Insert the data into the database
    for index, row in df.iterrows():
        # Convert 'DateTime' to datetime object
        date_time = datetime.strptime(row['DateTime'], "%Y-%m-%d %H:%M:%S")  # Adjust format if needed
        junction = row['Junction']
        vehicle_count = row['Vehicles']
        
        # Map the CSV columns to the model columns
        traffic_data = TrafficData(
            DateTime=date_time,  # Model's 'DateTime' field
            Junction=junction,   # Model's 'Junction' field
            Vehicles=vehicle_count  # Model's 'Vehicles' field
        )
        db.session.add(traffic_data)
    
    db.session.commit()
    print("Data inserted successfully!")

# Flask route for traffic prediction
@app.route('/predict', methods=['GET'])
def predict_traffic():
    junction = request.args.get('junction')
    hours_ahead = int(request.args.get('hours_ahead', 1))
    data = TrafficData.query.filter_by(Junction=junction).all()
    
    # Preprocess data for ML model
    times = [d.DateTime for d in data]
    counts = [d.Vehicles for d in data]
    df = pd.DataFrame({'datetime': times, 'vehicle_count': counts})
    df['hour'] = df['datetime'].dt.hour

    # Simple model example
    model = LinearRegression()
    X = np.array(df['hour']).reshape(-1, 1)
    y = df['vehicle_count']
    model.fit(X, y)

    # Predicting future traffic
    future_hour = (datetime.now().hour + hours_ahead) % 24
    predicted_traffic = model.predict(np.array([[future_hour]]))[0]

    return jsonify({'predicted_traffic': predicted_traffic})

# Define the Dash layout with visualizations
dash_app.layout = html.Div([
    html.H1("Real-Time Traffic Analysis Dashboard"),
    dcc.Input(id='junction-input', type='number', placeholder='Enter Junction ID', debounce=True),
    dcc.Graph(id='traffic-graph'),
    dcc.Graph(id='prediction-graph'),
    html.Div(id='prediction-output')
])

# Define the callback to update the graph and display prediction
@dash_app.callback(
    [Output('traffic-graph', 'figure'),
     Output('prediction-graph', 'figure'),
     Output('prediction-output', 'children')],
    [Input('junction-input', 'value')]
)
def update_graph_and_predict(junction):
    # Check if junction input is provided
    if junction is None:
        return go.Figure(), go.Figure(), "Please enter a junction ID."

    # Fetch historical data from the database for the given junction
    data = TrafficData.query.filter_by(Junction=junction).all()
    if not data:
        return go.Figure(), go.Figure(), f"No data available for junction {junction}."

    # Convert data to DataFrame for visualization
    times = [d.DateTime for d in data]
    counts = [d.Vehicles for d in data]
    df = pd.DataFrame({'datetime': times, 'vehicle_count': counts})

    # Create the figure for traffic flow (Line chart)
    fig_traffic = go.Figure(data=[go.Scatter(x=df['datetime'], y=df['vehicle_count'], mode='lines+markers', name='Traffic Flow')])
    fig_traffic.update_layout(title=f'Traffic Flow for Junction {junction}', xaxis_title='Time', yaxis_title='Vehicle Count')

    # Predict future traffic (for prediction graph)
    future_hours = np.array(range(1, 25)).reshape(-1, 1)  # Next 24 hours
    model = LinearRegression()
    X = np.array(df['datetime'].dt.hour).reshape(-1, 1)
    y = df['vehicle_count']
    model.fit(X, y)
    predicted_traffic = model.predict(future_hours)

    # Create the prediction graph (Bar chart)
    fig_prediction = go.Figure(data=[go.Bar(x=np.arange(1, 25), y=predicted_traffic, name='Predicted Traffic')])
    fig_prediction.update_layout(title='Predicted Traffic for Next 24 Hours', xaxis_title='Hour', yaxis_title='Predicted Vehicle Count')

    # Call the prediction endpoint
    response = requests.get(f'http://127.0.0.1:5000/predict?junction={junction}&hours_ahead=1')
    prediction = response.json().get('predicted_traffic', 'N/A')

    # Display the prediction result
    prediction_output = f"Predicted traffic in 1 hour: {prediction} vehicles."

    return fig_traffic, fig_prediction, prediction_output

if __name__ == '__main__':
    # Load CSV file within application context
    with app.app_context():
        load_csv_to_db(r'data/traffic.csv')

    app.run(debug=True)
