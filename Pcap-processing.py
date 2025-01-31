'''
Environment setup: Importing and Installing required dependencies

1. Install Python libraries:
   - pyshark: A Python wrapper for TShark, enabling packet capture and analysis.
   - pandas: A data manipulation library for organizing and analyzing packet data in tabular format.

2. Install nest_asyncio:
   - Enables nested asyncio event loops, required for pyshark to function.

3. Update system package list:
   - Ensures the system has the latest information about available software packages.

4. Install TShark:
   - A command-line network protocol analyzer used by pyshark to process PCAP files.
'''
import pyshark
import pandas as pd
import nest_asyncio
from matplotlib import pyplot as plt
import seaborn as sns
import os

# Enabling nested asyncio 
nest_asyncio.apply()

# Defining the path to the PCAP file
pcap_file_path = os.path.join(os.getcwd(), 'server.pcap')

# Lists to store timestamps and packet sizes
timestamps = []
packet_sizes = []

# Capturing packets and extracting data
cap = pyshark.FileCapture(pcap_file_path)
for packet in cap:
    timestamps.append(int(packet.sniff_time.timestamp()))

    # Attempt to extract length from various layers
    try:
        packet_sizes.append(int(packet.eth.len))  # Ethernet layer
    except AttributeError:
        try:
            packet_sizes.append(int(packet.ip.len))  # IP layer
        except AttributeError:
            packet_sizes.append(None)  # Handles other cases

cap.close()

# Creating a DataFrame with timestamps and packet sizes
df = pd.DataFrame({'timestamp': timestamps, 'packet_size': packet_sizes})

# Grouping by timestamp and calculating packets per second and average packet size
grouped_data = df.groupby('timestamp').agg({'packet_size': ['count', 'mean']})
grouped_data.columns = ['packets_per_sec', 'avg_packet_size']

# Defining the path to save the CSV file
csv_file_path = os.path.join(os.getcwd(), 'results.csv')
grouped_data.to_csv(csv_file_path, index=False)

# Plotting the results
def plot_series(series, series_name, series_index=0):
    palette = list(sns.palettes.mpl_palette('Dark2'))
    xs = series.index
    ys = series['packets_per_sec']

    plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = grouped_data.sort_values('timestamp', ascending=True)
plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('timestamp')
plt.ylabel('packets_per_sec')
plt.show()