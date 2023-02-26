import streamlit as st
from pytube import YouTube
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# Set the title of the web app
st.title("YouTube Video Downloader and Converter")

# Add a text input widget for the video URL
url = st.text_input("Enter the URL of the YouTube video")

# Add a button to initiate the download and conversion process
if st.button("Download and Convert to MP3"):
    # Create a YouTube object
    yt = YouTube(url)

    # Get the highest resolution stream
    stream = yt.streams.get_highest_resolution()

    # Specify the download directory for this video
    download_dir = os.path.join(os.getcwd(), yt.title)

    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Download the video to the specified directory
    video_path = stream.download(download_dir)

    # Convert the video to an MP3 file
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    mp3_path = os.path.splitext(video_path)[0] + ".mp3"
    audio_clip.write_audiofile(mp3_path)
    audio_clip.close()
    video_clip.close()

    # Display a success message and a link to the MP3 file
    st.success("Video converted to MP3!")
    st.audio(mp3_path, format="audio/mp3")