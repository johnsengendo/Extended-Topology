import os
import pyshark
import pandas as pd
import nest_asyncio
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# ======= PCAP Processing Function =======
def process_pcap_to_csv():
    """
    Processing the PCAP file to generate a CSV file with aggregated data.
    This function uses pyshark to extract packet timestamps and sizes,
    then groups by timestamp to calculate packets per second and average packet size.
    The resulting CSV is saved with the name 'packets_per_sec_analysis.csv'.
    """
    # Enabling nested asyncio for pyshark
    nest_asyncio.apply()
    
    # Defining the path to the PCAP file (make sure 'server.pcap' is in the working directory)
    pcap_file_path = os.path.join(os.getcwd(), 'flow31.pcap')
    
    # Lists to store timestamps and packet sizes
    timestamps = []
    packet_sizes = []
    
    # Capturing packets and extract data
    cap = pyshark.FileCapture(pcap_file_path)
    for packet in cap:
        timestamps.append(int(packet.sniff_time.timestamp()))
        try:
            # Try to extract length from the Ethernet layer
            packet_sizes.append(int(packet.eth.len))
        except AttributeError:
            try:
                # If Ethernet not available, try the IP layer
                packet_sizes.append(int(packet.ip.len))
            except AttributeError:
                # For other cases, append None
                packet_sizes.append(None)
    cap.close()
    
    # Creating a DataFrame with timestamps and packet sizes
    df = pd.DataFrame({'timestamp': timestamps, 'packet_size': packet_sizes})
    
    # Grouping by timestamp to calculate packets per second and average packet size
    grouped_data = df.groupby('timestamp').agg({'packet_size': ['count', 'mean']})
    grouped_data.columns = ['packets_per_sec', 'avg_packet_size']
    
    # Saving the CSV file to the working directory
    csv_file_path = os.path.join(os.getcwd(), 'packets_per_sec_analysis.csv')
    grouped_data.to_csv(csv_file_path, index=False)
    print("PCAP processing complete. CSV saved to:", csv_file_path)

# ======= Constants and Model Setup =======
# Processing the PCAP file before proceeding so the CSV is up-to-date.
process_pcap_to_csv()

# Defining constants
MAX_POINTS = 1000  # Maximum number of points storing in deques
SCALER_PATH = 'scaler.pkl'  # Path to the scaler file
MODEL_PATH = 'cnn_traffic_model.pth'  # Path to the model file
DATA_PATH = 'packets_per_sec_analysis.csv'  # Path to the dataset
WINDOW_SIZE = 300  # Window size for the CNN model

# Defining the CNN model
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

# Function to create a windowed dataset
def create_dataset_windowed(features, labels, ahead=4, window_size=1, max_window_size=360):
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)
    dataX = np.array([
        features[(i + max_window_size - window_size):(i + max_window_size), :]
        for i in range(samples)
    ])
    dataY = labels[ahead + max_window_size - 1 : ahead + max_window_size - 1 + samples]
    return dataX, dataY

# Function to load the model and scaler
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

# ======= Adaptive PID Controller Definition ======= 
class AdaptivePIDController:
    def __init__(self, initial_kp=0.5, initial_ki=0.1, initial_kd=0.1, dt=0.1, setpoint=0):
        self.kp = initial_kp
        self.ki = initial_ki
        self.kd = initial_kd
        self.dt = dt
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0
        self.min_gain = 0.05
        self.max_gain = 2.0
        self.adaptation_rate = 0.01
        self.error_history = deque(maxlen=100)
        self.prediction_errors = deque(maxlen=1000)
        self.mae_history = deque(maxlen=100)
        self.last_adaptation_time = 0
        self.adaptation_interval = 10

    def adapt_gains(self):
        if len(self.prediction_errors) < 10:
            return
        current_mae = np.mean(np.abs(list(self.prediction_errors)[-10:]))
        self.mae_history.append(current_mae)
        if len(self.mae_history) > 1:
            mae_trend = self.mae_history[-1] - self.mae_history[-2]
            if mae_trend > 0:  # Error increasing
                self.kp = np.clip(self.kp * 1.1, self.min_gain, self.max_gain)
                self.ki = np.clip(self.ki * 1.05, self.min_gain * 0.1, self.max_gain * 0.1)
                self.kd = np.clip(self.kd * 1.15, self.min_gain * 0.5, self.max_gain * 0.5)
            else:
                self.kp = np.clip(self.kp * 0.95, self.min_gain, self.max_gain)
                self.ki = np.clip(self.ki * 0.98, self.min_gain * 0.1, self.max_gain * 0.1)
                self.kd = np.clip(self.kd * 0.92, self.min_gain * 0.5, self.max_gain * 0.5)

    def update(self, measured_value, actual_value):
        prediction_error = actual_value - measured_value
        self.prediction_errors.append(prediction_error)
        error = self.setpoint - measured_value
        self.error_history.append(error)
        if len(self.prediction_errors) % self.adaptation_interval == 0:
            self.adapt_gains()
        self.integral += error * self.dt
        MAX_INTEGRAL = 1000
        self.integral = max(min(self.integral, MAX_INTEGRAL), -MAX_INTEGRAL)
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.integral = 0
        self.prev_error = 0
        self.error_history.clear()
        self.prediction_errors.clear()
        self.mae_history.clear()

# Creating PID controllers for validation set and test set
pid_val = AdaptivePIDController()
pid_test = AdaptivePIDController()

# ======= Loading and Preparing Data =======
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

# Deques for storing predictions and actual values
val_predictions = deque(maxlen=MAX_POINTS)
test_predictions = deque(maxlen=MAX_POINTS)
val_actuals = deque(maxlen=MAX_POINTS)
test_actuals = deque(maxlen=MAX_POINTS)
val_pid_predictions = deque(maxlen=MAX_POINTS)
test_pid_predictions = deque(maxlen=MAX_POINTS)

# Deques for storing absolute errors (for bar charts)
val_errors = deque(maxlen=MAX_POINTS)
val_pid_errors = deque(maxlen=MAX_POINTS)
test_errors = deque(maxlen=MAX_POINTS)
test_pid_errors = deque(maxlen=MAX_POINTS)

current_index = 0

# ======= Defining helper function to style figures =======
def style_figure(fig, title):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=70, b=50),
        hovermode="x"
    )
    return fig

# ======= Defining the Dash App layout =======
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = "Traffic Digital Twin Dashboard"

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(html.H1("Traffic Digital Twin Dashboard", className="text-center text-light my-4"))
        ),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="ahead-selector",
                    options=[{"label": f"{i} seconds ahead", "value": i} for i in [5, 10, 20, 30, 60, 90, 120]],
                    value=10,
                    clearable=False
                ),
                width=4
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="update-interval",
                    options=[
                        {"label": "Fast (0.5s)", "value": 500},
                        {"label": "Normal (1s)", "value": 1000},
                        {"label": "Slow (2s)", "value": 2000}
                    ],
                    value=1000,
                    clearable=False
                ),
                width=4
            ),
            dbc.Col(html.Button("Activate PID", id="activate-pid", className="btn btn-primary"), width=2),
            dbc.Col(html.Button("Deactivate PID", id="deactivate-pid", className="btn btn-danger"), width=2),
        ], className="mb-3"),
        # Defining the first row: Line graphs for validation and test
        dbc.Row([
            dbc.Col(dcc.Graph(id="val-traffic-graph", style={"height": "500px"}), width=6),
            dbc.Col(dcc.Graph(id="test-traffic-graph", style={"height": "500px"}), width=6),
        ]),
        # Defining the second row: Bar charts for absolute errors
        dbc.Row([
            dbc.Col(dcc.Graph(id="val-error-bar-graph", style={"height": "500px"}), width=6),
            dbc.Col(dcc.Graph(id="test-error-bar-graph", style={"height": "500px"}), width=6),
        ]),
        # Defining Fine-tune PID buttons at the bottom of the bar graphs
        dbc.Row([
            dbc.Col(html.Button("Fine-tune PID +", id="increase-ki", className="btn btn-success"), width=2),
            dbc.Col(html.Button("Fine-tune PID -", id="decrease-ki", className="btn btn-warning"), width=2),
        ], className="mb-3"),
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
    ],
    style={"backgroundColor": "#2c3e50"},
)

# ======= Defining Dash Callback to Update Graphs =======
@app.callback(
    [
        Output("val-traffic-graph", "figure"),
        Output("test-traffic-graph", "figure"),
        Output("val-error-bar-graph", "figure"),
        Output("test-error-bar-graph", "figure")
    ],
    [
        Input("interval-component", "n_intervals"),
        Input("activate-pid", "n_clicks"),
        Input("deactivate-pid", "n_clicks"),
        Input("increase-ki", "n_clicks"),
        Input("decrease-ki", "n_clicks"),
    ],
    [State("ahead-selector", "value"), State("update-interval", "value")]
)
def update_graphs(n_intervals, activate_clicks, deactivate_clicks,
                  increase_ki, decrease_ki, ahead, update_interval):
    global current_index
    global val_predictions, test_predictions, val_actuals, test_actuals
    global val_pid_predictions, test_pid_predictions
    global val_errors, val_pid_errors, test_errors, test_pid_errors
    global pid_val, pid_test

    # Updating dt for both PIDs based on selected update interval (ms -> seconds)
    dt_sec = update_interval / 1000.0
    pid_val.dt = dt_sec
    pid_test.dt = dt_sec

    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Resetting both PIDs on activation
    if button_id == "activate-pid":
        pid_val.reset()
        pid_test.reset()

    # Adjusting Ki for both if buttons are clicked
    if button_id == "increase-ki":
        pid_val.ki = np.clip(pid_val.ki * 1.1, pid_val.min_gain * 0.1, pid_val.max_gain * 0.1)
        pid_test.ki = np.clip(pid_test.ki * 1.1, pid_test.min_gain * 0.1, pid_test.max_gain * 0.1)
    elif button_id == "decrease-ki":
        pid_val.ki = np.clip(pid_val.ki * 0.9, pid_val.min_gain * 0.1, pid_val.max_gain * 0.1)
        pid_test.ki = np.clip(pid_test.ki * 0.9, pid_test.min_gain * 0.1, pid_test.max_gain * 0.1)

    # Building windowed datasets for validation and test
    X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels, ahead=ahead, window_size=WINDOW_SIZE)
    X_test_w, r_test_w = create_dataset_windowed(test_scaled, test_labels, ahead=ahead, window_size=WINDOW_SIZE)

    # Checking if data is available; if not, return empty figures on the screen
    if X_val_w.size == 0 or X_test_w.size == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5, y=0.5,
            text="Not enough data for this 'ahead' setting",
            showarrow=False,
            font=dict(color="white", size=16)
        )
        empty_fig.update_layout(template="plotly_dark", xaxis=dict(visible=False), yaxis=dict(visible=False))
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Converting to Torch tensors
    X_val_w = torch.FloatTensor(X_val_w).transpose(1, 2)
    r_val_w = torch.FloatTensor(r_val_w)
    X_test_w = torch.FloatTensor(X_test_w).transpose(1, 2)
    r_test_w = torch.FloatTensor(r_test_w)

    # Resetting current_index if out of range
    if current_index >= len(X_val_w):
        current_index = 0

    # Getting model predictions
    with torch.no_grad():
        val_pred = model(X_val_w[current_index:current_index + 1].to(device)).cpu().numpy()
        test_pred = model(X_test_w[current_index:current_index + 1].to(device)).cpu().numpy()

    # Inversing scale predictions and targets
    val_pred_unscaled = scaler.inverse_transform(val_pred)
    test_pred_unscaled = scaler.inverse_transform(test_pred)
    r_val_w_unscaled = scaler.inverse_transform(r_val_w[current_index:current_index + 1].numpy().reshape(-1, 1))
    r_test_w_unscaled = scaler.inverse_transform(r_test_w[current_index:current_index + 1].numpy().reshape(-1, 1))

    # Appending data for plotting
    val_predictions.append(val_pred_unscaled[0][0])
    test_predictions.append(test_pred_unscaled[0][0])
    val_actuals.append(r_val_w_unscaled[0][0])
    test_actuals.append(r_test_w_unscaled[0][0])

    # Checking if PID is active
    pid_active = (activate_clicks is not None and (deactivate_clicks is None or activate_clicks > deactivate_clicks))

    # Validation PID update
    val_pid_pred = val_pred_unscaled[0][0]
    if pid_active:
        pid_val.setpoint = r_val_w_unscaled[0][0]
        val_adjustment = pid_val.update(val_pid_pred, r_val_w_unscaled[0][0])
        val_pid_pred += val_adjustment
    val_pid_predictions.append(val_pid_pred)

    # Test PID update
    test_pid_pred = test_pred_unscaled[0][0]
    if pid_active:
        pid_test.setpoint = r_test_w_unscaled[0][0]
        test_adjustment = pid_test.update(test_pid_pred, r_test_w_unscaled[0][0])
        test_pid_pred += test_adjustment
    test_pid_predictions.append(test_pid_pred)

    # Computing absolute errors for bar charts
    val_err = abs(r_val_w_unscaled[0][0] - val_pred_unscaled[0][0])
    val_pid_err = abs(r_val_w_unscaled[0][0] - val_pid_pred)
    test_err = abs(r_test_w_unscaled[0][0] - test_pred_unscaled[0][0])
    test_pid_err = abs(r_test_w_unscaled[0][0] - test_pid_pred)

    val_errors.append(val_err)
    val_pid_errors.append(val_pid_err)
    test_errors.append(test_err)
    test_pid_errors.append(test_pid_err)

    # Incrementing current_index
    current_index = (current_index + 1) % len(X_val_w)

    # Building line graphs
    val_fig = go.Figure()
    val_fig.add_trace(go.Scatter(y=list(val_actuals), mode="lines", name="Actual", line=dict(color="yellow", width=2)))
    val_fig.add_trace(go.Scatter(y=list(val_predictions), mode="lines", name="Predicted", line=dict(color="cyan", width=2)))
    if pid_active:
        val_fig.add_trace(go.Scatter(y=list(val_pid_predictions), mode="lines", name="PID Adjusted", line=dict(color="magenta", width=2)))
    val_fig.update_layout(xaxis_title="Time Step", yaxis_title="Packets/sec")
    val_fig = style_figure(val_fig, "Validation Traffic Prediction")

    test_fig = go.Figure()
    test_fig.add_trace(go.Scatter(y=list(test_actuals), mode="lines", name="Actual", line=dict(color="yellow", width=2)))
    test_fig.add_trace(go.Scatter(y=list(test_predictions), mode="lines", name="Predicted", line=dict(color="cyan", width=2)))
    if pid_active:
        test_fig.add_trace(go.Scatter(y=list(test_pid_predictions), mode="lines", name="PID Adjusted", line=dict(color="magenta", width=2)))
    test_fig.update_layout(xaxis_title="Time Step", yaxis_title="Packets/sec")
    test_fig = style_figure(test_fig, "Test Traffic Prediction")

    # Building bar charts (if PID is active)
    if pid_active:
        val_err_fig = go.Figure()
        val_err_fig.add_trace(go.Bar(x=list(range(len(val_errors))), y=list(val_errors),
                                     name="Error (Predicted)", marker_color="cyan"))
        val_err_fig.add_trace(go.Bar(x=list(range(len(val_pid_errors))), y=list(val_pid_errors),
                                     name="Error (PID Adjusted)", marker_color="magenta"))
        val_err_fig.update_layout(barmode="group")
        val_err_fig = style_figure(val_err_fig, "Validation Absolute Error")

        test_err_fig = go.Figure()
        test_err_fig.add_trace(go.Bar(x=list(range(len(test_errors))), y=list(test_errors),
                                      name="Error (Predicted)", marker_color="cyan"))
        test_err_fig.add_trace(go.Bar(x=list(range(len(test_pid_errors))), y=list(test_pid_errors),
                                      name="Error (PID Adjusted)", marker_color="magenta"))
        test_err_fig.update_layout(barmode="group")
        test_err_fig = style_figure(test_err_fig, "Test Absolute Error")
    else:
        val_err_fig = go.Figure()
        val_err_fig = style_figure(val_err_fig, "Validation Absolute Error")
        test_err_fig = go.Figure()
        test_err_fig = style_figure(test_err_fig, "Test Absolute Error")

    return val_fig, test_fig, val_err_fig, test_err_fig

# ======= Running the Dash App =======
if __name__ == "__main__":
    app.run_server(debug=True)
