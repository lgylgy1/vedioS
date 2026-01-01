import ffmpeg
import os
import tempfile
from typing import Optional


class VideoProcessor:
    """
    A class for processing video files using FFmpeg.
    """
    def __init__(self):
        pass

    def extract_audio(self, video_path: str, output_audio_path: Optional[str] = None) -> str:
        """
        Extract audio from video file.

        Args:
            video_path: Path to the video file
            output_audio_path: Path to save the extracted audio. If None, uses temp file.

        Returns:
            Path to the extracted audio file
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_audio_path is None:
            temp_dir = tempfile.mkdtemp()
            output_audio_path = os.path.join(temp_dir, "audio.wav")

        try:
            # Extract audio using ffmpeg
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, quiet=True)
            return output_audio_path
        except ffmpeg.Error as e:
            raise Exception(f"FFmpeg error: {e}")

    def get_video_duration(self, video_path: str) -> float:
        """
        Get the duration of the video in seconds.

        Args:
            video_path: Path to the video file

        Returns:
            Duration in seconds
        """
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except (ffmpeg.Error, KeyError, IndexError):
            raise Exception(f"Could not get video duration for {video_path}")

    def extract_video_segment(self, video_path: str, start_time: float, duration: float, output_path: str):
        """
        Extract a segment from the video.

        Args:
            video_path: Path to the input video
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Path to save the segment
        """
        try:
            stream = ffmpeg.input(video_path, ss=start_time, t=duration)
            stream = ffmpeg.output(stream, output_path)
            ffmpeg.run(stream, quiet=True)
        except ffmpeg.Error as e:
            raise Exception(f"FFmpeg error extracting segment: {e}")
