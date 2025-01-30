import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import dash_bootstrap_components as dbc
import os
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# Setting random seed for reproducibility
SEED = 2022
torch.manual_seed(SEED)
np.random.seed(SEED)

# Constants
MAX_POINTS = 1000
SCALER_PATH = 'scaler.pkl'
MODEL_PATH = 'cnn_traffic_model.pth'
DATA_PATH = 'packets_per_sec_analysis.csv'

class CNNModel(nn.Module):
    def __init__(self, window_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=4, padding='same')
        self.conv2 = nn.Conv1d(128, 128, kernel_size=4, padding='same')
        self.conv3 = nn.Conv1d(128, 64, kernel_size=4, padding='same')
        self.conv4 = nn.Conv1d(64, 32, kernel_size=4, padding='same')
        self.flatten_size = 32 * window_size
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_dataset_windowed(features, labels, ahead=4, window_size=1, max_window_size=360):
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)
    dataX = np.array([features[(i + max_window_size - window_size):(i + max_window_size), :]
                      for i in range(samples)])
    dataY = labels[ahead + max_window_size - 1 : ahead + max_window_size - 1 + samples]
    return dataX, dataY

def load_model_and_scaler():
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")

    return model, scaler, device

class AdaptivePIDController:
    def __init__(self, initial_kp=0.5, initial_ki=0.1, initial_kd=0.1, dt=0.1, setpoint=0):
        self.kp = initial_kp
        self.ki = initial_ki
        self.kd = initial_kd
        self.dt = dt
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.adaptation_rate = 0.01
        self.min_gain = 0.05
        self.max_gain = 2.0
        self.error_history = deque(maxlen=100)
        self.prediction_errors = deque(maxlen=1000)  # Storing historical prediction errors
        self.mae_history = deque(maxlen=100)  # Tracking MAE over time
        self.last_adaptation_time = 0
        self.adaptation_interval = 10  # Adapting every 10 updates

    def adapt_gains(self):
        if len(self.prediction_errors) < 10:
            return

        current_mae = np.mean(np.abs(list(self.prediction_errors)[-10:]))
        self.mae_history.append(current_mae)

        if len(self.mae_history) > 1:
            mae_trend = self.mae_history[-1] - self.mae_history[-2]

            # Adjusting gains based on error trend
            if mae_trend > 0:  # Error is increasing
                self.kp = np.clip(self.kp * 1.1, self.min_gain, self.max_gain)
                self.ki = np.clip(self.ki * 1.05, self.min_gain * 0.1, self.max_gain * 0.1)
                self.kd = np.clip(self.kd * 1.15, self.min_gain * 0.5, self.max_gain * 0.5)
            else:  # Error is decreasing or stable
                self.kp = np.clip(self.kp * 0.95, self.min_gain, self.max_gain)
                self.ki = np.clip(self.ki * 0.98, self.min_gain * 0.1, self.max_gain * 0.1)
                self.kd = np.clip(self.kd * 0.92, self.min_gain * 0.5, self.max_gain * 0.5)

    def update(self, measured_value, actual_value):
        # Tracking prediction error
        prediction_error = actual_value - measured_value
        self.prediction_errors.append(prediction_error)

        # Standard PID error calculation
        error = self.setpoint - measured_value
        self.error_history.append(error)

        # Adapting gains periodically
        if len(self.prediction_errors) % self.adaptation_interval == 0:
            self.adapt_gains()

        # PID control calculation
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output

    def get_current_mae(self):
        if len(self.prediction_errors) == 0:
            return 0
        return np.mean(np.abs(self.prediction_errors))

# Initializing constants
WINDOW_SIZE = 300
pid = AdaptivePIDController()

# Loading data, model, and scaler
data = pd.read_csv(DATA_PATH)['packets_per_sec']
train_split_index = int(len(data) * 0.45)
val_split_index = int(len(data) * 0.725)

train_dataset = data[:train_split_index]
val_dataset = data[train_split_index:val_split_index]
test_dataset = data[val_split_index:]

train_values = np.asarray(train_dataset.values, dtype=np.float32).reshape(-1, 1)
val_values = np.asarray(val_dataset.values, dtype=np.float32).reshape(-1, 1)
test_values = np.asarray(test_dataset.values, dtype=np.float32).reshape(-1, 1)

train_labels = np.asarray(train_dataset.values, dtype=np.float32)
val_labels = np.asarray(val_dataset.values, dtype=np.float32)
test_labels = np.asarray(test_dataset.values, dtype=np.float32)

model, scaler, device = load_model_and_scaler()
train_scaled = scaler.transform(train_values)
val_scaled = scaler.transform(val_values)
test_scaled = scaler.transform(test_values)

val_predictions = deque(maxlen=MAX_POINTS)
test_predictions = deque(maxlen=MAX_POINTS)
val_actuals = deque(maxlen=MAX_POINTS)
test_actuals = deque(maxlen=MAX_POINTS)
val_pid_predictions = deque(maxlen=MAX_POINTS)
test_pid_predictions = deque(maxlen=MAX_POINTS)
current_index = 0

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = "Traffic Digital Twin Dashboard"

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(dbc.Col(html.H1("Traffic Digital Twin Dashboard", className="text-center text-light my-4"))),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="ahead-selector", options=[{"label": f"{i} seconds ahead", "value": i} for i in [10, 20, 30, 60, 90, 120]], value=10, clearable=False)),
            dbc.Col(dcc.Dropdown(id="update-interval", options=[{"label": "Fast (0.5s)", "value": 500}, {"label": "Normal (1s)", "value": 1000}, {"label": "Slow (2s)", "value": 2000}], value=1000, clearable=False)),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="val-traffic-graph")),
            dbc.Col(dcc.Graph(id="test-traffic-graph")),
        ]),
        dbc.Row([
            dbc.Col(html.Button("Activate PID", id="activate-pid", className="btn btn-primary")),
            dbc.Col(html.Button("Deactivate PID", id="deactivate-pid", className="btn btn-danger")),
        ]),
        dbc.Row([
            dbc.Col(html.Button("Increase Kp", id="increase-kp", className="btn btn-success")),
            dbc.Col(html.Button("Decrease Kp", id="decrease-kp", className="btn btn-warning")),
            dbc.Col(html.Button("Increase Ki", id="increase-ki", className="btn btn-success")),
            dbc.Col(html.Button("Decrease Ki", id="decrease-ki", className="btn btn-warning")),
            dbc.Col(html.Button("Increase Kd", id="increase-kd", className="btn btn-success")),
            dbc.Col(html.Button("Decrease Kd", id="decrease-kd", className="btn btn-warning")),
        ]),
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
    ],
    style={"backgroundColor": "#2c3e50"},
)

@app.callback(
    [
        Output("val-traffic-graph", "figure"),
        Output("test-traffic-graph", "figure"),
    ],
    [Input("interval-component", "n_intervals"), Input("activate-pid", "n_clicks"), Input("deactivate-pid", "n_clicks"),
     Input("increase-kp", "n_clicks"), Input("decrease-kp", "n_clicks"),
     Input("increase-ki", "n_clicks"), Input("decrease-ki", "n_clicks"),
     Input("increase-kd", "n_clicks"), Input("decrease-kd", "n_clicks")],
    [State("ahead-selector", "value")]
)
def update_graphs(n_intervals, activate_clicks, deactivate_clicks, increase_kp, decrease_kp, increase_ki, decrease_ki, increase_kd, decrease_kd, ahead):
    global current_index, val_predictions, test_predictions, val_actuals, test_actuals, val_pid_predictions, test_pid_predictions, pid

    # Determining which button was clicked
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Adjusting PID parameters based on the button clicked
    if button_id == "increase-kp":
        pid.kp = np.clip(pid.kp * 1.1, pid.min_gain, pid.max_gain)
    elif button_id == "decrease-kp":
        pid.kp = np.clip(pid.kp * 0.9, pid.min_gain, pid.max_gain)
    elif button_id == "increase-ki":
        pid.ki = np.clip(pid.ki * 1.1, pid.min_gain * 0.1, pid.max_gain * 0.1)
    elif button_id == "decrease-ki":
        pid.ki = np.clip(pid.ki * 0.9, pid.min_gain * 0.1, pid.max_gain * 0.1)
    elif button_id == "increase-kd":
        pid.kd = np.clip(pid.kd * 1.1, pid.min_gain * 0.5, pid.max_gain * 0.5)
    elif button_id == "decrease-kd":
        pid.kd = np.clip(pid.kd * 0.9, pid.min_gain * 0.5, pid.max_gain * 0.5)

    try:
        # Preparing windowed datasets
        X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels, ahead=ahead, window_size=WINDOW_SIZE)
        X_test_w, r_test_w = create_dataset_windowed(test_scaled, test_labels, ahead=ahead, window_size=WINDOW_SIZE)

        X_val_w = torch.FloatTensor(X_val_w).transpose(1, 2)
        X_test_w = torch.FloatTensor(X_test_w).transpose(1, 2)
        r_val_w = torch.FloatTensor(r_val_w)
        r_test_w = torch.FloatTensor(r_test_w)

        with torch.no_grad():
            val_pred = model(X_val_w[current_index:current_index + 1].to(device)).cpu().numpy()
            test_pred = model(X_test_w[current_index:current_index + 1].to(device)).cpu().numpy()

        val_pred_unscaled = scaler.inverse_transform(val_pred)
        test_pred_unscaled = scaler.inverse_transform(test_pred)
        r_val_w_unscaled = scaler.inverse_transform(r_val_w[current_index:current_index + 1].numpy().reshape(-1, 1))
        r_test_w_unscaled = scaler.inverse_transform(r_test_w[current_index:current_index + 1].numpy().reshape(-1, 1))

        val_predictions.append(val_pred_unscaled[0][0])
        test_predictions.append(test_pred_unscaled[0][0])
        val_actuals.append(r_val_w_unscaled[0][0])
        test_actuals.append(r_test_w_unscaled[0][0])

        pid_active = activate_clicks is not None and (deactivate_clicks is None or activate_clicks > deactivate_clicks)
        val_pid_pred, test_pid_pred = val_pred_unscaled[0][0], test_pred_unscaled[0][0]

        if pid_active:
            val_error = r_val_w_unscaled[0][0] - val_pid_pred
            test_error = r_test_w_unscaled[0][0] - test_pid_pred

            pid.setpoint = r_val_w_unscaled[0][0]
            val_pid_adjustment = pid.update(val_pid_pred,r_val_w_unscaled[0][0])
            test_pid_adjustment = pid.update(test_pid_pred,r_test_w_unscaled[0][0])

            val_pid_pred += val_pid_adjustment
            test_pid_pred += test_pid_adjustment

        val_pid_predictions.append(val_pid_pred)
        test_pid_predictions.append(test_pid_pred)

        current_index = (current_index + 1) % len(X_val_w)

        # Creating figures
        val_fig = go.Figure()
        val_fig.add_trace(go.Scatter(y=list(val_actuals), mode="lines", name="Actual", line=dict(color="yellow", width=2)))
        val_fig.add_trace(go.Scatter(y=list(val_predictions), mode="lines", name="Predicted", line=dict(color="cyan", width=2)))
        if pid_active:
            val_fig.add_trace(go.Scatter(y=list(val_pid_predictions), mode="lines", name="PID Adjusted", line=dict(color="magenta", width=2)))

        val_fig.update_layout(
            title="Validation Traffic Prediction",
            xaxis_title="Time",
            yaxis_title="Packets per Second",
            template="plotly_dark",
            hovermode="x",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        test_fig = go.Figure()
        test_fig.add_trace(go.Scatter(y=list(test_actuals), mode="lines", name="Actual", line=dict(color="yellow", width=2)))
        test_fig.add_trace(go.Scatter(y=list(test_predictions), mode="lines", name="Predicted", line=dict(color="cyan", width=2)))
        if pid_active:
            test_fig.add_trace(go.Scatter(y=list(test_pid_predictions), mode="lines", name="PID Adjusted", line=dict(color="magenta", width=2)))

        test_fig.update_layout(
            title="Test Traffic Prediction",
            xaxis_title="Time",
            yaxis_title="Packets per Second",
            template="plotly_dark",
            hovermode="x",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        return val_fig, test_fig
    except Exception as e:
        print(f"Error in update_graphs: {e}")
        return go.Figure(), go.Figure()

if __name__ == "__main__":
    app.run_server(debug=True)
