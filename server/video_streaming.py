#!/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def main():
    """
    Main function to stream a video file via FFmpeg using RTMP.
    """

    # Local video file path
    input_file = "video/Jam.mp4"

    # -1 means infinite loop of the video file
    # FFmpeg doc: https://ffmpeg.org/ffmpeg.html#Main-options
    loops_number = -1

    # Duration of stream in seconds (30 minutes)
    duration = "18"

    # FFmpeg command with detailed options
    ffmpeg_command = [
        "ffmpeg",
        "-loglevel", "info",   # Show detailed info
        "-stats",              # Print progress stats to terminal
        "-re",                 # Read input in real-time (simulate live input)
        # https://ffmpeg.org/ffmpeg.html#Main-options

        "-stream_loop", str(loops_number),  # Loop input
        # https://ffmpeg.org/ffmpeg.html#Advanced-options

        "-i", input_file,      # Input file
        "-t", duration,        # Duration of output
        # https://ffmpeg.org/ffmpeg.html#Main-options

        "-c:v", "copy",        # Copy video without re-encoding
        # https://ffmpeg.org/ffmpeg-codecs.html

        "-c:a", "aac",         # Encode audio in AAC format
        "-ar", "44100",        # Set audio sample rate
        "-ac", "1",            # Set audio channels to mono
        # https://ffmpeg.org/ffmpeg.html#Audio-Options

        "-f", "flv",           # Output format FLV (used for RTMP)
        # https://ffmpeg.org/ffmpeg-formats.html#flv

        "rtmp://localhost:1935/live/video.flv"  # Output RTMP URL (local streaming server)
        # Typically served by NGINX + RTMP module or a media server
    ]

    try:
        # Run the FFmpeg command
        result = subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("Streaming started successfully.")
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:")
        print(e.stderr)

if __name__ == "__main__":
    main()
