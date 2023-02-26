from pytube import YouTube
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def download_and_convert_video(video_url):
    # Create a YouTube object
    yt = YouTube(video_url)

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

    # Return the path to the MP3 file
    return mp3_path