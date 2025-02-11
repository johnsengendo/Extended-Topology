import argparse
import os
import subprocess
import sys
import time
import threading
from comnetsemu.cli import CLI, spawnXtermDocker
from comnetsemu.net import Containernet, VNFManager
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.node import Controller

# Function to add a streaming container to the VNFManager
def add_streaming_container(manager, name, role, image, shared_dir):
    return manager.addContainer(
        name, role, image, '', docker_args={
            'volumes': {
                shared_dir: {'bind': '/home/pcap/', 'mode': 'rw'}
            }
        }
    )

# Function to start the streaming server
def start_server():
    subprocess.run(['docker', 'exec', '-it', 'streaming_server', 'bash', '-c', 'cd /home && python3 video_streaming.py'])

# Function to start the streaming client
def start_client():
    subprocess.run(['docker', 'exec', '-it', 'streaming_client', 'bash', '-c', 'cd /home && python3 get_video_streamed.py'])

# ----- Iperf Flow Functions -----
def start_iperf_server(host, port):
    host.cmd(f'iperf -s -p {port} -u &')

def start_iperf_client(host, server_ip, port):
    # No fixed duration so the flow runs until terminated.
    host.cmd(f'iperf -c {server_ip} -p {port} -u -b 5M &')

# Function to stop iperf processes on a host
def stop_iperf(host):
    host.cmd('pkill iperf')

# Function to change the properties of the middle link (bottleneck link)
def change_link_properties(link, bw, delay, jitter=0, loss=0):
    info(f'*** Changing link properties: BW={bw} Mbps, Delay={delay} ms, Jitter={jitter} ms, Loss={loss}%\n')
    link.intf1.config(bw=bw, delay=f'{delay}ms', jitter=f'{jitter}ms', loss=loss)
    link.intf2.config(bw=bw, delay=f'{delay}ms', jitter=f'{jitter}ms', loss=loss)

if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='Video streaming application with fixed bandwidth and delay.')
    parser.add_argument('--autotest', dest='autotest', action='store_const', const=True, default=False,
                        help='Enables automatic testing of the topology and closes the streaming application.')
    args = parser.parse_args()
    autotest = args.autotest

    # Shared directory for pcap files and other shared data
    script_directory = os.path.abspath(os.path.dirname(__file__))
    shared_directory = os.path.join(script_directory, 'pcap')
    if not os.path.exists(shared_directory):
        os.makedirs(shared_directory)

    setLogLevel('info')

    # Initializing the network
    net = Containernet(controller=Controller, link=TCLink, xterms=False)
    mgr = VNFManager(net)

    info('*** Adding controller\n')
    net.addController('c0')

    info('*** Creating hosts\n')
    server = net.addDockerHost(
        'server', dimage='dev_test', ip='10.0.0.1', docker_args={'hostname': 'server'}
    )
    client = net.addDockerHost(
        'client', dimage='dev_test', ip='10.0.0.2', docker_args={'hostname': 'client'}
    )

    h1 = net.addHost('h1', ip='10.0.0.3')
    h2 = net.addHost('h2', ip='10.0.0.4')
    h3 = net.addHost('h3', ip='10.0.0.5')  # Iperf Flow 1 client
    h6 = net.addHost('h6', ip='10.0.0.6')  # Iperf Flow 1 server
    h4 = net.addHost('h4', ip='10.0.0.7')  # Iperf Flow 2 server (reverse direction)
    h5 = net.addHost('h5', ip='10.0.0.8')  # Iperf Flow 2 client (reverse direction)

    info('*** Adding switches and links\n')
    switch1 = net.addSwitch('s1')
    switch2 = net.addSwitch('s2')

    net.addLink(switch1, server)
    net.addLink(switch1, h1)
    middle_link = net.addLink(switch1, switch2, bw=10, delay='10ms')
    net.addLink(switch2, client)
    net.addLink(switch2, h2)
    net.addLink(switch1, h3)
    net.addLink(switch2, h6)
    net.addLink(switch1, h4)
    net.addLink(switch2, h5)

    info('\n*** Starting network\n')
    net.start()

    # Testing connectivity: pinging from client to server
    info("*** Client host pings the server to test connectivity: \n")
    reply = client.cmd("ping -c 5 10.0.0.1")
    print(reply)

    # Setting fixed link properties (for example: 100 Mbps, 0 ms delay, 5 ms jitter, 0.1% loss)
    change_link_properties(middle_link, 100, 0, 5, 0.1)

    # ----- Start Separate Tcpdump Captures for Iperf Flows -----
    capture_interface = middle_link.intf1.name

    # For Iperf Flow 1 (from h3 -> h6, port 5001)
    iperf_flow1_capture_file = os.path.join(shared_directory, "iperf_flow1.pcap")
    tcpdump_cmd_flow1 = ["sudo", "tcpdump", "-i", capture_interface, "-s", "96", "udp port 5001", "-w", iperf_flow1_capture_file]
    info(f'*** Starting tcpdump on interface {capture_interface} for iperf flow 1 (port 5001), saving to {iperf_flow1_capture_file}\n')
    tcpdump_proc_flow1 = subprocess.Popen(tcpdump_cmd_flow1)

    # For Iperf Flow 2 (from h5 -> h4, port 5002)
    iperf_flow2_capture_file = os.path.join(shared_directory, "iperf_flow2.pcap")
    tcpdump_cmd_flow2 = ["sudo", "tcpdump", "-i", capture_interface, "-s", "96", "udp port 5002", "-w", iperf_flow2_capture_file]
    info(f'*** Starting tcpdump on interface {capture_interface} for iperf flow 2 (port 5002), saving to {iperf_flow2_capture_file}\n')
    tcpdump_proc_flow2 = subprocess.Popen(tcpdump_cmd_flow2)

    # Adding streaming Docker containers
    streaming_server = add_streaming_container(mgr, 'streaming_server', 'server', 'streaming_server_image', shared_directory)
    streaming_client = add_streaming_container(mgr, 'streaming_client', 'client', 'streaming_client_image', shared_directory)

    # Starting streaming server and client applications in separate threads.
    server_thread = threading.Thread(target=start_server)
    client_thread = threading.Thread(target=start_client)
    server_thread.start()
    client_thread.start()

    # ----- Iperf Communications: Starting Immediately and Running Until Streaming Stops -----
    info('*** Starting iperf communications concurrently with streaming...\n')
    # Iperf Flow 1: from h3 (client) to h6 (server) on port 5001
    start_iperf_server(h6, 5001)
    start_iperf_client(h3, "10.0.0.6", 5001)

    # Iperf Flow 2 (reverse): from h5 (client) to h4 (server) on port 5002
    start_iperf_server(h4, 5002)
    start_iperf_client(h5, "10.0.0.7", 5002)

    # The iperf flows will now run continuously.
    server_thread.join()  # Wait until the streaming server stops.
    client_thread.join()  # Wait until the streaming client stops.

    info('*** Streaming processes have ended. Stopping iperf flows...\n')
    for host in [h3, h6, h4, h5]:
        stop_iperf(host)

    if not autotest:
        CLI(net)

    # Terminating tcpdump captures before cleanup
    info('*** Terminating tcpdump captures\n')
    tcpdump_proc_flow1.terminate()
    tcpdump_proc_flow1.wait()
    tcpdump_proc_flow2.terminate()
    tcpdump_proc_flow2.wait()

    # Cleaning up Docker containers and stopping the network
    mgr.removeContainer('streaming_server')
    mgr.removeContainer('streaming_client')
    net.stop()
    mgr.stop()
