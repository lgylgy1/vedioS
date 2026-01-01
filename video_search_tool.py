import asyncio
from typing import List, Dict, Any, Optional
import os
from video_processor import VideoProcessor
from transcriber import Transcriber
from indexer import VideoIndexer
from configuration import llm_config, video_config
from llm_conversation import LLMConversation


class VideoSearchTool:
    """
    A tool for searching video segments based on natural language queries using Whisper, FAISS, and FFmpeg.
    """
    def __init__(self, whisper_model: Optional[str] = None, embedding_dimension: Optional[int] = None, index_file: Optional[str] = None):
        """
        Initialize the video search tool.

        Args:
            whisper_model: Whisper model to use for transcription (uses config if None)
            embedding_dimension: Dimension of embedding vectors (uses config if None)
            index_file: Path to the FAISS index file (uses config if None)
        """
        whisper_model = whisper_model or video_config.whisper_model
        embedding_dimension = embedding_dimension or video_config.embedding_dimension
        index_file = index_file or video_config.index_file

        self.video_processor = VideoProcessor()
        self.transcriber = Transcriber(whisper_model)
        self.indexer = VideoIndexer(embedding_dimension, index_file)
        self.llm_conversation = LLMConversation()

    async def index_video(self, video_path: str, chunk_duration: float = 30.0, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Index a video file for searching.

        Args:
            video_path: Path to the video file
            chunk_duration: Duration of each chunk in seconds
            language: Language for transcription

        Returns:
            Indexing results
        """
        print(f"Indexing video: {video_path}")

        # Extract audio
        print("Extracting audio...")
        audio_path = self.video_processor.extract_audio(video_path)

        # Transcribe audio
        print("Transcribing audio...")
        transcription = self.transcriber.transcribe(audio_path, language)

        # Split into chunks
        print("Splitting into chunks...")
        chunks = self.transcriber.split_into_chunks(transcription, chunk_duration)

        # Add to index
        print("Adding to index...")
        await self.indexer.add_chunks(chunks, video_path)

        # Clean up temp audio file if it was created
        if audio_path != video_path.replace('.mp4', '.wav'):  # Assuming temp file
            try:
                os.remove(audio_path)
            except:
                pass

        return {
            'video_path': video_path,
            'total_chunks': len(chunks),
            'index_info': self.indexer.get_index_info()
        }

    async def search_videos(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for video segments matching the query.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of matching video segments
        """
        print(f"Searching for: {query}")
        results = await self.indexer.search(query, top_k)

        # Enhance results with LLM if needed
        enhanced_results = []
        for result in results:
            # Optionally use LLM to improve relevance or generate summary
            enhanced_result = result.copy()
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def extract_segment(self, video_path: str, start_time: float, end_time: float, output_path: str):
        """
        Extract a video segment.

        Args:
            video_path: Path to the source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the segment
        """
        duration = end_time - start_time
        self.video_processor.extract_video_segment(video_path, start_time, duration, output_path)

    async def get_relevant_segments(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get the most relevant video segments for a query.

        Args:
            query: Natural language query
            top_k: Number of segments to retrieve

        Returns:
            List of segment information
        """
        results = await self.search_videos(query, top_k)
        return results

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        return self.indexer.get_index_info()


# Example usage
async def main():
    tool = VideoSearchTool()

    # Index a video
    video_path = "path/to/your/video.mp4"
    if os.path.exists(video_path):
        index_result = await tool.index_video(video_path)
        print("Indexing completed:", index_result)

    # Search
    query = "人工智能的发展历程"
    results = await tool.search_videos(query)
    print("Search results:")
    for result in results:
        print(f"- {result['text'][:100]}... (score: {result['score']:.3f})")


if __name__ == "__main__":
    asyncio.run(main())
