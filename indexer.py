import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import os
import pickle
from embedding import emb
import asyncio


class VideoIndexer:
    """
    A class for indexing video transcription chunks using FAISS and embeddings.
    """
    def __init__(self, dimension: int = 1024, index_file: Optional[str] = None):
        """
        Initialize the indexer.

        Args:
            dimension: Dimension of the embedding vectors
            index_file: Path to save/load the FAISS index
        """
        self.dimension = dimension
        self.index_file = index_file or "video_index.faiss"
        self.metadata_file = self.index_file.replace('.faiss', '_metadata.pkl')

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        self.metadata = []  # List of metadata for each vector

        # Load existing index if available
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.load_index()

    async def add_chunks(self, chunks: List[Dict[str, Any]], video_path: str):
        """
        Add transcription chunks to the index.

        Args:
            chunks: List of chunk dictionaries
            video_path: Path to the original video file
        """
        texts = [chunk['text'] for chunk in chunks]

        # Get embeddings for all texts
        embeddings = []
        for text in texts:
            embedding = await emb(text)
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Failed to get embedding for text: {text[:50]}...")
                embeddings.append(np.zeros(self.dimension))  # Fallback

        if not embeddings:
            return

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add to index
        self.index.add(embeddings_array)

        # Add metadata
        for i, chunk in enumerate(chunks):
            metadata_entry = {
                'video_path': video_path,
                'start_time': chunk['start'],
                'end_time': chunk['end'],
                'text': chunk['text'],
                'chunk_index': len(self.metadata) + i
            }
            self.metadata.append(metadata_entry)

        # Save index
        self.save_index()

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on query.

        Args:
            query: Natural language query
            top_k: Number of top results to return

        Returns:
            List of matching chunks with scores
        """
        # Get embedding for query
        query_embedding = await emb(query)
        if not query_embedding:
            return []

        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Search
        scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)

        return results

    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

    def load_index(self):
        """Load the FAISS index and metadata from disk."""
        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_file': self.index_file,
            'metadata_file': self.metadata_file
        }
