# **Extended-Topology**  

## **Overview**  
The **Extended-Topology and Live Dashboard** provides a real-time visualization of network traffic, offering insights into realistic traffic monitoring and prediction for Digital Twins.  

### **Key Features**  
- 🌐 **Web-Based Interface** – Accessible via a browser.  
- 📊 **Live-Updating Graphs** – Real-time visualization of traffic data.  
- 🔄 **Dynamic Prediction Intervals** – Adapts to traffic patterns.  
- 🎛 **Interactive PID Control** – Fine-tune parameters for improved accuracy.  

### **Prerequisites**  
- VS Code (Recommended)   

### **Running the Dashboard**  
Steps to set up and run the dashboard:  

1. **Importing the pre-trained module** into the directory where the dashboard script is stored [cnn_traffic_model.pth](model/cnn_traffic_model.pth).
2.   **Placing the Scaler** in the same directory as the dashboard script [Scaler](model/scaler.pkl).
3. **Placing the dataset** in the same directory as the dashboard script [data](data/packets_per_sec_analysis.csv)..  
4. **Running the script** to launch the [dashboard](DigitalTwin.py).
5. Below is the dashboard overview.
![Dashboard Preview](images/dash-board-image.png)

### **Next step for the extended topology to be built, image below** 
![Topology](Extended_topology.png)

## Progress   
`Topology.py` sets up the topology and runs everything while calling the video streaming inside the `video` folder, as well as the video client inside the `video` folder.  

## Pcap File Patch  
During streaming and Iperf data transmission, pcap files are captured. Various flaws detected in the middle link of the topology are stored in the `pcap` folder.  

## Live Dashboard Upgrade  
`DigitalTwin_dashboard.py` is an update of the dashboard, where it works efficiently with the activation of the PID. It also enables the continuous activation, deactivation and fine-tunning of the PID over time.

## Live Dashboard Update  
In this `Live_Dashboard_DigitalTwin.py` script, a pcap file can be loaded immediately, where it is processed, and live traffic is displayed enabling real-time monitoring and the continuous activation, deactivation and fine-tunning of the PID over time
