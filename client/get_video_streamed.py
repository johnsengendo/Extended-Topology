#!/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def get_video_stream():
    """
    Main function to handle video streaming.
    Streams from an RTMP source and writes the output to a file.
    """
    out_file = "stream_output.flv"

    ffmpeg_command = [
        "ffmpeg", "-loglevel", "info", "-stats",
        "-i", "rtmp://10.0.0.1:1935/live/video.flv",
        "-t", "660",              # Stream duration in seconds
        "-probesize", "80000",
        "-analyzeduration", "15",
        "-c:a", "copy",           # Copy audio without re-encoding
        "-c:v", "copy",           # Copy video without re-encoding
        out_file
    ]
    
    subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    get_video_stream()
