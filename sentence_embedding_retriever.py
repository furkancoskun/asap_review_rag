import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm

class DenseRetriever:
    def __init__(self, sentence_transformer_model_name='all-MiniLM-L6-v2'):
        print(f"Loading sentence transformer model: {sentence_transformer_model_name}...")
        self.model = SentenceTransformer(sentence_transformer_model_name)
        print("Model loaded.")
        self.paper_specific_indexes = {} # Stores {'paper_id': {'index': faiss_index, 'chunks_df': paper_chunks_df}}
        self.global_chunk_embeddings = None
        self.global_chunks_df = None

    def build_index_for_paper(self, paper_id, paper_chunks_df):
        """
        Builds a FAISS index for the review chunks of a specific paper.
        Args:
            paper_id (str): The ID of the paper.
            paper_chunks_df (pd.DataFrame): DataFrame containing review chunks ONLY for this paper.
                                            Must have 'text' and 'chunk_id' columns.
        """
        if paper_chunks_df.empty:
            print(f"Warning: No chunks provided for paper {paper_id}. Index not built.")
            return

        print(f"Building FAISS index for paper {paper_id} with {len(paper_chunks_df)} chunks...")
        chunk_texts = paper_chunks_df['text'].tolist()
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=False)

        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension) # Using L2 distance (Euclidean)
        # For larger datasets, consider faiss.IndexIVFFlat or other more advanced indexes
        index.add(np.array(chunk_embeddings, dtype=np.float32))

        self.paper_specific_indexes[paper_id] = {
            'index': index,
            'chunks_df': paper_chunks_df.copy() # Store the specific chunks DF for this paper
        }
        print(f"Index built for paper {paper_id}.")

    def retrieve_from_paper(self, query_text, paper_id, top_k=5):
        """
        Retrieves top_k relevant chunks for a given query from a specific paper's reviews.
        Args:
            query_text (str): The query string.
            paper_id (str): The ID of the paper to retrieve reviews from.
            top_k (int): Number of chunks to retrieve.
        Returns:
            pd.DataFrame: DataFrame of the top_k retrieved chunks for that paper, or empty if no index/chunks.
        """
        if paper_id not in self.paper_specific_indexes:
            print(f"Error: No index found for paper {paper_id}. Please build the index first.")
            return pd.DataFrame()

        paper_data = self.paper_specific_indexes[paper_id]
        index = paper_data['index']
        paper_chunks_df = paper_data['chunks_df']

        if paper_chunks_df.empty:
            return pd.DataFrame()

        query_embedding = self.model.encode([query_text], convert_to_tensor=False, show_progress_bar=False)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        # Ensure top_k is not greater than the number of items in the index
        num_indexed_items = index.ntotal
        actual_top_k = min(top_k, num_indexed_items)

        if actual_top_k == 0:
            return pd.DataFrame()

        distances, indices = index.search(query_embedding_np, actual_top_k)

        retrieved_chunks = paper_chunks_df.iloc[indices[0]].copy()
        retrieved_chunks['similarity_score_L2_distance'] = distances[0] # Lower L2 distance is better
        # You could convert L2 distance to a cosine-like similarity if preferred: 1 / (1 + L2_distance)
        # Or if using IndexFlatIP (Inner Product for normalized embeddings), higher is better.
        return retrieved_chunks.sort_values(by='similarity_score_L2_distance', ascending=True)


    def pre_build_all_paper_indexes(self, all_review_chunks_df):
        """
        Builds FAISS indexes for all papers present in the review_chunks_df.
        Args:
            all_review_chunks_df (pd.DataFrame): DataFrame containing all review chunks.
                                              Must have 'paper_id', 'text', and 'chunk_id'.
        """
        if all_review_chunks_df is None or all_review_chunks_df.empty:
            print("Error: No review chunks data provided to pre_build_all_paper_indexes.")
            return

        print("Pre-building indexes for all papers...")
        grouped_by_paper = all_review_chunks_df.groupby('paper_id')
        for paper_id, paper_chunks_df in tqdm(grouped_by_paper, desc="Building paper indexes"):
            self.build_index_for_paper(paper_id, paper_chunks_df)
        print("All paper-specific indexes built.")
