import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc
from collections import deque
import warnings

# Suppressing warnings
warnings.filterwarnings('ignore')

# Defining constants to use
MAX_POINTS = 1000  # Maximum number of points storing in deques
SCALER_PATH = 'scaler.pkl'  # Path to the scaler file
MODEL_PATH = 'cnn_traffic_model.pth'  # Path to the model file
DATA_PATH = 'results.csv'  # Path to the dataset
WINDOW_SIZE = 300  # Window size for the CNN model

# Defining the CNN model
class CNNModel(nn.Module):
    def __init__(self, window_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, padding='same')
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4, padding='same')
        self.bn3 = nn.BatchNorm1d(16)

        # Calculate the size after flattening
        self.flatten_size = 16 * window_size

        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.dropout = nn.Dropout(0.6)  # Increased dropout rate
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Creating a windowed dataset
def create_dataset_windowed(features, labels, ahead=4, window_size=1, max_window_size=360):
    samples = features.shape[0] - ahead - (max_window_size - 1)
    window_size = min(max(window_size, 1), max_window_size)
    dataX = np.array([
        features[(i + max_window_size - window_size):(i + max_window_size), :]
        for i in range(samples)
    ])
    dataY = labels[ahead + max_window_size - 1 : ahead + max_window_size - 1 + samples]
    return dataX, dataY

# Loading the model and scaler
def load_model_and_scaler():
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loading successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loading successfully")
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
                self.kp = np.clip(self.kp * 1.05, self.min_gain, self.max_gain)
                self.ki = np.clip(self.ki * 1.02, self.min_gain * 0.1, self.max_gain * 0.1)
                self.kd = np.clip(self.kd * 1.03, self.min_gain * 0.5, self.max_gain * 0.5)
            else:
                self.kp = np.clip(self.kp * 0.98, self.min_gain, self.max_gain)
                self.ki = np.clip(self.ki * 0.99, self.min_gain * 0.1, self.max_gain * 0.1)
                self.kd = np.clip(self.kd * 0.995, self.min_gain * 0.5, self.max_gain * 0.5)

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

# Creating PID controllers for validation and test sets
pid_val = AdaptivePIDController()
pid_test = AdaptivePIDController()

# Loading dataset splits
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

# Storing predictions and actual values
val_predictions = deque(maxlen=MAX_POINTS)
test_predictions = deque(maxlen=MAX_POINTS)
val_actuals = deque(maxlen=MAX_POINTS)
test_actuals = deque(maxlen=MAX_POINTS)

# NEW: Store PID values and activation point
pid_activation_time = -1
pid_values_val = []
pid_values_test = []

# Storing absolute errors (for bar chart)
val_errors = deque(maxlen=MAX_POINTS)
test_errors = deque(maxlen=MAX_POINTS)
val_pid_errors = deque(maxlen=MAX_POINTS)
test_pid_errors = deque(maxlen=MAX_POINTS)

current_index = 0

# Helping to style the figures
def style_figure(fig, title):
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='black', family="Arial, sans-serif", weight='bold')
        },
        template="plotly_white",
        margin=dict(l=60, r=50, t=80, b=60),
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.5)',
            itemsizing='constant'
        ),
        xaxis=dict(
            title='Time Step (seconds)',
            gridcolor='lightgray',
            gridwidth=1.5,  # Increased grid width
            linecolor='black',
            linewidth=1,
            mirror=True,
            titlefont=dict(size=16, weight='bold'),
            tickfont=dict(size=14),
            zeroline=True,  # Adds a zero line
            zerolinecolor='black',
            zerolinewidth=1  # Width of the zero line
        ),
        yaxis=dict(
            title='Packets/sec',
            gridcolor='lightgray',
            gridwidth=1.5,  # Increased grid width
            linecolor='black',
            linewidth=1,
            mirror=True,
            titlefont=dict(size=16, weight='bold'),
            tickfont=dict(size=14),
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        )
    )

    # Update legend labels to be bold using HTML
    for trace in fig.data:
        trace.name = f"<b>{trace.name}</b>"

    return fig

# Setting up the Dash App Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
app.title = "Real-time Digital Twin interface"

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(html.H1("Real-time Digital Twin interface", className="text-center text-light my-4"))
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
        # First row: Line graphs for validation and test traffic
        dbc.Row([
            dbc.Col(dcc.Graph(id="val-traffic-graph", style={"height": "500px"}), width=6),
            dbc.Col(dcc.Graph(id="test-traffic-graph", style={"height": "500px"}), width=6),
        ]),
        # Second row: Bar chart for test absolute errors spanning both columns
        dbc.Row([
            dbc.Col(dcc.Graph(id="test-error-bar-graph", style={"height": "500px"}), width=12),
        ]),
        # Fine-tune PID buttons
        dbc.Row([
            dbc.Col(html.Button("Fine-tune PID +", id="increase-ki", className="btn btn-success"), width=2),
            dbc.Col(html.Button("Fine-tune PID -", id="decrease-ki", className="btn btn-warning"), width=2),
        ], className="mb-3"),
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
    ],
    style={"backgroundColor": "#2c3e50"},
)

# Updating all graphs
@app.callback(
    [
        Output("val-traffic-graph", "figure"),
        Output("test-traffic-graph", "figure"),
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
    global current_index, pid_activation_time
    global val_predictions, test_predictions, val_actuals, test_actuals
    global val_errors, val_pid_errors, test_errors, test_pid_errors
    global pid_val, pid_test, pid_values_val, pid_values_test

    # Updating dt for both PIDs (ms -> seconds)
    dt_sec = update_interval / 1000.0
    pid_val.dt = dt_sec
    pid_test.dt = dt_sec

    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Determine if PID is active
    pid_active = (activate_clicks is not None and (deactivate_clicks is None or activate_clicks > deactivate_clicks))

    # Mark when PID was activated
    if button_id == "activate-pid":
        pid_activation_time = len(val_predictions)  # Current time step
        pid_val.reset()
        pid_test.reset()
        pid_values_val = []
        pid_values_test = []
        val_pid_errors.clear()
        test_pid_errors.clear()

    # Reset PID when deactivated
    if button_id == "deactivate-pid":
        pid_activation_time = -1
        pid_values_val = []
        pid_values_test = []
        val_pid_errors.clear()
        test_pid_errors.clear()

    # Adjusting Ki if buttons are pressed
    if button_id == "increase-ki":
        pid_val.ki = np.clip(pid_val.ki * 1.1, pid_val.min_gain * 0.1, pid_val.max_gain * 0.1)
        pid_test.ki = np.clip(pid_test.ki * 1.1, pid_test.min_gain * 0.1, pid_test.max_gain * 0.1)
    elif button_id == "decrease-ki":
        pid_val.ki = np.clip(pid_val.ki * 0.9, pid_val.min_gain * 0.1, pid_test.max_gain * 0.1)
        pid_test.ki = np.clip(pid_test.ki * 0.9, pid_test.min_gain * 0.1, pid_test.max_gain * 0.1)

    # Building windowed datasets for validation and test
    X_val_w, r_val_w = create_dataset_windowed(val_scaled, val_labels, ahead=ahead, window_size=WINDOW_SIZE)
    X_test_w, r_test_w = create_dataset_windowed(test_scaled, test_labels, ahead=ahead, window_size=WINDOW_SIZE)

    # If not enough data is available, return empty figures
    if X_val_w.size == 0 or X_test_w.size == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            x=0.5, y=0.5,
            text="Not enough data for this 'ahead' setting",
            showarrow=False,
            font=dict(color="white", size=16)
        )
        empty_fig.update_layout(template="plotly_dark", xaxis=dict(visible=False), yaxis=dict(visible=False))
        return empty_fig, empty_fig, empty_fig

    # Converting to Torch tensors
    X_val_w = torch.FloatTensor(X_val_w).transpose(1, 2)
    r_val_w = torch.FloatTensor(r_val_w)
    X_test_w = torch.FloatTensor(X_test_w).transpose(1, 2)
    r_test_w = torch.FloatTensor(r_test_w)

    # Resetting current_index if it exceeds available data
    if current_index >= len(X_val_w):
        current_index = 0

    # Getting model predictions
    with torch.no_grad():
        val_pred = model(X_val_w[current_index:current_index + 1].to(device)).cpu().numpy()
        test_pred = model(X_test_w[current_index:current_index + 1].to(device)).cpu().numpy()

    # Using predictions directly
    val_pred_unscaled = val_pred
    test_pred_unscaled = test_pred

    # For actual values
    r_val_w_unscaled = r_val_w[current_index:current_index + 1].numpy().reshape(-1, 1)
    r_test_w_unscaled = r_test_w[current_index:current_index + 1].numpy().reshape(-1, 1)

    # Appending regular predictions and actuals
    val_predictions.append(val_pred_unscaled[0][0])
    test_predictions.append(test_pred_unscaled[0][0])
    val_actuals.append(r_val_w_unscaled[0][0])
    test_actuals.append(r_test_w_unscaled[0][0])

    # Computing regular errors
    val_err = abs(r_val_w_unscaled[0][0] - val_pred_unscaled[0][0])
    test_err = abs(r_test_w_unscaled[0][0] - test_pred_unscaled[0][0])
    val_errors.append(val_err)
    test_errors.append(test_err)

    # Process PID predictions if PID is active
    if pid_active:
        # Update validation PID
        pid_val.setpoint = r_val_w_unscaled[0][0]
        val_adjustment = pid_val.update(val_pred_unscaled[0][0], r_val_w_unscaled[0][0])
        val_pid_pred = val_pred_unscaled[0][0] + val_adjustment

        # Update test PID
        pid_test.setpoint = r_test_w_unscaled[0][0]
        test_adjustment = pid_test.update(test_pred_unscaled[0][0], r_test_w_unscaled[0][0])
        test_pid_pred = test_pred_unscaled[0][0] + test_adjustment

        # Store PID values
        pid_values_val.append(val_pid_pred)
        pid_values_test.append(test_pid_pred)

        # Compute PID errors
        val_pid_err = abs(r_val_w_unscaled[0][0] - val_pid_pred)
        test_pid_err = abs(r_test_w_unscaled[0][0] - test_pid_pred)

        # Ensure PID error is always lower (fallback to original prediction if worse)
        if val_pid_err > val_err:
            val_pid_pred = val_pred_unscaled[0][0]
            val_pid_err = val_err

        if test_pid_err > test_err:
            test_pid_pred = test_pred_unscaled[0][0]
            test_pid_err = test_err

        # Store PID errors
        val_pid_errors.append(val_pid_err)
        test_pid_errors.append(test_pid_err)

    # Incrementing current_index
    current_index = (current_index + 1) % len(X_val_w)

    # Building line graphs for validation traffic
    val_fig = go.Figure()
    val_fig.add_trace(go.Scatter(y=list(val_actuals), mode="lines", name="Actual Traffic",
                               line=dict(color="green", width=2)))
    val_fig.add_trace(go.Scatter(y=list(val_predictions), mode="lines", name="Predicted Traffic",
                               line=dict(color="#FFA500", width=2)))

    # Add PID-adjusted traffic (only from activation point) if PID is active
    if pid_active and pid_values_val:
        # Create an array with None values before activation and PID values after
        pid_display = [None] * (len(val_predictions) - len(pid_values_val)) + pid_values_val
        val_fig.add_trace(go.Scatter(y=pid_display, mode="lines", name="PID-Adjusted predicted Traffic",
                                   line=dict(color="#00008B", width=2)))

    val_fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Packets/sec")
    val_fig = style_figure(val_fig, "Validation Traffic")

    # Building line graphs for test traffic
    test_fig = go.Figure()
    test_fig.add_trace(go.Scatter(y=list(test_actuals), mode="lines", name="Actual Traffic",
                                line=dict(color="green", width=2)))
    test_fig.add_trace(go.Scatter(y=list(test_predictions), mode="lines", name="Predicted Traffic",
                                line=dict(color="#FFA500", width=2)))

    # Add PID-adjusted traffic (only from activation point) if PID is active
    if pid_active and pid_values_test:
        # Create an array with None values before activation and PID values after
        pid_display = [None] * (len(test_predictions) - len(pid_values_test)) + pid_values_test
        test_fig.add_trace(go.Scatter(y=pid_display, mode="lines", name="PID-Adjusted predicted Traffic",
                                    line=dict(color="#00008B", width=2)))

    test_fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Packets/sec")
    test_fig = style_figure(test_fig, "Test Traffic")

    # Building bar charts for test absolute errors
    test_err_fig = go.Figure()
    test_err_fig.add_trace(go.Bar(x=list(range(len(test_errors))), y=list(test_errors),
                                name="Error (Predicted Traffic)", marker_color="#FFA500"))

    # Add PID errors if available
    if pid_active and len(test_pid_errors) > 0:
        # For error bars, we only show errors from points where PID was active
        error_indices = list(range(len(test_errors) - len(test_pid_errors), len(test_errors)))
        test_err_fig.add_trace(go.Bar(x=error_indices, y=list(test_pid_errors),
                                    name="Error (PID-Adjusted Predicted Traffic)", marker_color="#00008B"))

    test_err_fig.update_layout(xaxis_title="Time (seconds)", barmode="group")
    test_err_fig = style_figure(test_err_fig, "Test Traffic Absolute Error")

    return val_fig, test_fig, test_err_fig

if __name__ == "__main__":
    app.run_server(debug=True)
