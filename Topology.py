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
    subprocess.run([
        'docker', 'exec', '-it', 'streaming_server',
        'bash', '-c', 'cd /home && python3 video_streaming.py'
    ])

# Function to start the streaming client
def start_client():
    subprocess.run([
        'docker', 'exec', '-it', 'streaming_client',
        'bash', '-c', 'cd /home && python3 get_video_streamed.py'
    ])

# ------------------------------
# iPerf Flow Functions
# ------------------------------
# Flow 1: from h3 -> h6 (UDP port 5001)
def start_iperf_server_flow1(host):
    # iPerf server on port 5001
    host.cmd('iperf -s -p 5001 -u &')

def start_iperf_client_flow1(host):
    # iPerf client
    host.cmd('iperf -c 10.0.0.6 -p 5001 -u -b 5M -t 600 &')

# Flow 2: from h5 -> h4 (UDP port 5002), reverse direction
def start_iperf_server_flow2(host):
    # iPerf server on port 5002
    host.cmd('iperf -s -p 5002 -u &')

def start_iperf_client_flow2(host):
    # iPerf client
    host.cmd('iperf -c 10.0.0.7 -p 5002 -u -b 5M -t 600 &')

# Function to stop iPerf on a host
def stop_iperf(host):
    host.cmd('pkill iperf')

# Function to change link properties (bottleneck link)
def change_link_properties(link, bw, delay, jitter=0, loss=0):
    info(f'*** Changing link properties: BW={bw} Mbps, Delay={delay} ms, Jitter={jitter} ms, Loss={loss}%\n')
    link.intf1.config(bw=bw, delay=f'{delay}ms', jitter=f'{jitter}ms', loss=loss)
    link.intf2.config(bw=bw, delay=f'{delay}ms', jitter=f'{jitter}ms', loss=loss)

if __name__ == '__main__':
    # ------------------------------
    # Parsing command-line arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description='Video streaming + iPerf testbed.')
    parser.add_argument('--autotest', dest='autotest', action='store_const',
                        const=True, default=False,
                        help='Enable automatic testing and close streaming automatically.')
    args = parser.parse_args()
    autotest = args.autotest

    # ------------------------------
    # Setting-up environment + network
    # ------------------------------
    script_directory = os.path.abspath(os.path.dirname(__file__))
    shared_directory = os.path.join(script_directory, 'pcap')
    if not os.path.exists(shared_directory):
        os.makedirs(shared_directory)

    setLogLevel('info')

    net = Containernet(controller=Controller, link=TCLink, xterms=False)
    mgr = VNFManager(net)

    info('*** Adding controller\n')
    net.addController('c0')

    info('*** Creating hosts\n')
    server = net.addDockerHost(
        'server', dimage='dev_test', ip='10.0.0.1',
        docker_args={'hostname': 'server'}
    )
    client = net.addDockerHost(
        'client', dimage='dev_test', ip='10.0.0.2',
        docker_args={'hostname': 'client'}
    )

    # iPerf hosts
    h1 = net.addHost('h1', ip='10.0.0.3')
    h2 = net.addHost('h2', ip='10.0.0.4')
    h3 = net.addHost('h3', ip='10.0.0.5')  # Flow 1 client
    h6 = net.addHost('h6', ip='10.0.0.6')  # Flow 1 server
    h4 = net.addHost('h4', ip='10.0.0.7')  # Flow 2 server
    h5 = net.addHost('h5', ip='10.0.0.8')  # Flow 2 client

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

    # Quick connectivity check between client and server
    info("*** Testing connectivity: client -> server\n")
    reply = client.cmd("ping -c 3 10.0.0.1")
    print(reply)

    # Setting link properties (example: 100 Mbps, 0 ms delay, 5 ms jitter, 0.1% loss)
    change_link_properties(middle_link, 100, 0, 5, 0.1)

    # ------------------------------
    # Starting TCPDump captures
    # ------------------------------
    capture_interface = middle_link.intf1.name

    # Flow 1: port 5001
    iperf_flow1_capture = os.path.join(shared_directory, "iperf_flow1.pcap")
    tcpdump_cmd_flow1 = [
        "sudo", "tcpdump", "-i", capture_interface, "-s", "96",
        "udp port 5001", "-w", iperf_flow1_capture
    ]
    info(f'*** Starting tcpdump for Flow 1 -> {iperf_flow1_capture}\n')
    tcpdump_proc_flow1 = subprocess.Popen(tcpdump_cmd_flow1)

    # Flow 2: port 5002
    iperf_flow2_capture = os.path.join(shared_directory, "iperf_flow2.pcap")
    tcpdump_cmd_flow2 = [
        "sudo", "tcpdump", "-i", capture_interface, "-s", "96",
        "udp port 5002", "-w", iperf_flow2_capture
    ]
    info(f'*** Starting tcpdump for Flow 2 -> {iperf_flow2_capture}\n')
    tcpdump_proc_flow2 = subprocess.Popen(tcpdump_cmd_flow2)

    # Flow 3: capturing all other traffic (NOT iperf port 5001 or 5002)
    flow3_capture = os.path.join(shared_directory, "flow3.pcap")
    tcpdump_cmd_flow3 = [
        "sudo", "tcpdump", "-i", capture_interface, "-s", "96",
        "not (udp port 5001 or udp port 5002)",
        "-w", flow3_capture
    ]
    info(f'*** Starting tcpdump for Flow 3 (non-iperf) -> {flow3_capture}\n')
    tcpdump_proc_flow3 = subprocess.Popen(tcpdump_cmd_flow3)

    # ------------------------------
    # Adding streaming containers
    # ------------------------------
    streaming_server = add_streaming_container(
        mgr, 'streaming_server', 'server',
        'streaming_server_image', shared_directory
    )
    streaming_client = add_streaming_container(
        mgr, 'streaming_client', 'client',
        'streaming_client_image', shared_directory
    )

    # ------------------------------
    # Starting streaming threads
    # ------------------------------
    server_thread = threading.Thread(target=start_server)
    client_thread = threading.Thread(target=start_client)
    server_thread.start()
    client_thread.start()

    # ------------------------------
    # Starting iPerf flows
    # ------------------------------
    def iperf_control():
        info('*** Starting iPerf flows (600s duration)...\n')
        # Flow 1: h3 -> h6 on port 5001
        start_iperf_server_flow1(h6)
        start_iperf_client_flow1(h3)

        # Flow 2: h5 -> h4 on port 5002
        start_iperf_server_flow2(h4)
        start_iperf_client_flow2(h5)

        time.sleep(600)

        info('*** Stopping iPerf flows...\n')
        for host in [h3, h6, h4, h5]:
            stop_iperf(host)

    iperf_thread = threading.Thread(target=iperf_control)
    iperf_thread.start()

    # ------------------------------
    # Waiting for streaming to finish
    # ------------------------------
    server_thread.join()
    client_thread.join()

    # Wait for iPerf thread to finish (after 600 seconds)
    iperf_thread.join()

    if not autotest:
        CLI(net)

    # ------------------------------
    # Terminating tcpdump captures
    # ------------------------------
    info('*** Terminating tcpdump captures\n')
    tcpdump_proc_flow1.terminate()
    tcpdump_proc_flow1.wait()
    tcpdump_proc_flow2.terminate()
    tcpdump_proc_flow2.wait()
    tcpdump_proc_flow3.terminate()
    tcpdump_proc_flow3.wait()

    # ------------------------------
    # Cleanup
    # ------------------------------
    mgr.removeContainer('streaming_server')
    mgr.removeContainer('streaming_client')
    net.stop()
    mgr.stop()
