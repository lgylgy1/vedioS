import whisper
from typing import List, Dict, Any, Optional
import os


class Transcriber:
    """
    A class for transcribing audio files using Whisper.
    """
    def __init__(self, model_name: str = "base"):
        """
        Initialize the transcriber with a Whisper model.

        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
        """
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file
            language: Language code (optional, auto-detect if None)

        Returns:
            Transcription result with text and segments
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        options = {}
        if language:
            options['language'] = language

        result = self.model.transcribe(audio_path, **options)
        return result

    def get_segments(self, transcription_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract segments from transcription result.

        Args:
            transcription_result: Result from transcribe method

        Returns:
            List of segments with start, end, and text
        """
        return transcription_result.get('segments', [])

    def split_into_chunks(self, transcription_result: Dict[str, Any], chunk_duration: float = 30.0) -> List[Dict[str, Any]]:
        """
        Split transcription into chunks of specified duration.

        Args:
            transcription_result: Transcription result
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of chunks with start, end, text, and metadata
        """
        segments = self.get_segments(transcription_result)
        chunks = []
        current_chunk = {'start': 0.0, 'end': 0.0, 'text': '', 'segments': []}

        for segment in segments:
            if current_chunk['end'] - current_chunk['start'] >= chunk_duration:
                # Save current chunk
                chunks.append(current_chunk.copy())
                # Start new chunk
                current_chunk = {'start': segment['start'], 'end': segment['end'], 'text': segment['text'], 'segments': [segment]}
            else:
                # Add to current chunk
                current_chunk['end'] = segment['end']
                current_chunk['text'] += ' ' + segment['text']
                current_chunk['segments'].append(segment)

        # Add the last chunk
        if current_chunk['text']:
            chunks.append(current_chunk)

        return chunks
