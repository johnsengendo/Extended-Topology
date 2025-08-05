#!/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def get_video_stream():
    """
    Streams video from an RTMP source and writes it to a local FLV file.
    """

    # Output file name
    out_file = "stream_output.flv"

    # FFmpeg command for recording RTMP input stream
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "info",      # Show detailed information
        "-stats",                 # Show real-time progress

        "-i", "rtmp://10.0.0.1:1935/live/video.flv",  # RTMP input stream
        # RTMP protocol doc: https://en.wikipedia.org/wiki/Real-Time_Messaging_Protocol

        "-t", "1800",             # Limit capture to 1800 seconds (30 mins)
        # FFmpeg time limit: https://ffmpeg.org/ffmpeg.html#Main-options

        "-probesize", "80000",    # Amount of data to probe for stream info
        "-analyzeduration", "15", # Analyze input for 15 microseconds (can be adjusted)
        # Stream probing: https://ffmpeg.org/ffmpeg.html#Format-Options

        "-c:a", "copy",           # Copy audio without re-encoding
        "-c:v", "copy",           # Copy video without re-encoding
        # Codecs copy: https://ffmpeg.org/ffmpeg-codecs.html

        out_file                  # Output file
    ]
    
    try:
        # Execute the command
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("Stream recording completed.")
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:")
        print(e.stderr)

if __name__ == "__main__":
    get_video_stream()
