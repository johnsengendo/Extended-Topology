#!/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def main():
    """
    Main function to handle video streaming.
    """
    input_file = "video/Jam.mp4"
    loops_number = -1  # Streaming the video multiple times

    ffmpeg_command = [
        "ffmpeg", "-loglevel", "info", "-stats", "-re", "-stream_loop", str(loops_number),
        "-i", input_file,
        "-t", "300",              # Setting the streaming duration (in seconds)
        "-c:v", "copy",           # Coping video stream without re-encoding
        "-c:a", "aac",            # Encode audio using AAC
        "-ar", "44100",           # Audio sample rate
        "-ac", "1",               # Number of audio channels
        "-f", "flv",              # Format set to FLV for RTMP streaming
        "rtmp://localhost:1935/live/video.flv"
    ]

    subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    main()
