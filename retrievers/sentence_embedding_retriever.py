import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm
import pickle 

class DenseRetriever:
    def __init__(self, sentence_transformer_model_name='all-MiniLM-L6-v2', storage_path_prefix='retriever_storage/dense'):
        print(f"Loading sentence transformer model: {sentence_transformer_model_name}...")
        self.model = SentenceTransformer(sentence_transformer_model_name)
        print("Model loaded.")
        self.paper_specific_indexes = {} # Stores {'paper_id': {'index_path': path, 'chunks_df_path': path}}
        self.storage_path_prefix = storage_path_prefix
        os.makedirs(self.storage_path_prefix, exist_ok=True)

    def _get_paper_storage_paths(self, paper_id):
        paper_dir = os.path.join(self.storage_path_prefix, paper_id)
        os.makedirs(paper_dir, exist_ok=True)
        index_path = os.path.join(paper_dir, "faiss_index.idx")
        chunks_df_path = os.path.join(paper_dir, "chunks_df.pkl") # Using pickle for DataFrame
        return index_path, chunks_df_path

    def build_index_for_paper(self, paper_id, paper_chunks_df, force_rebuild=False):
        index_path, chunks_df_path = self._get_paper_storage_paths(paper_id)

        if not force_rebuild and os.path.exists(index_path) and os.path.exists(chunks_df_path):
            # print(f"Index for paper {paper_id} already exists. Skipping build. To rebuild, set force_rebuild=True.")
            # No need to load here, load_index_for_paper will handle it on demand
            if paper_id not in self.paper_specific_indexes: # Ensure metadata is stored
                 self.paper_specific_indexes[paper_id] = {'index_path': index_path, 'chunks_df_path': chunks_df_path, 'index_obj': None, 'chunks_df_obj': None}
            return

        if paper_chunks_df.empty:
            print(f"Warning: No chunks provided for paper {paper_id}. Index not built.")
            return

        print(f"Building and saving FAISS index for paper {paper_id} with {len(paper_chunks_df)} chunks...")
        chunk_texts = paper_chunks_df['text'].tolist()
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=False, show_progress_bar=False)

        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings, dtype=np.float32))

        # Save the index and the chunks DataFrame
        faiss.write_index(index, index_path)
        with open(chunks_df_path, 'wb') as f:
            pickle.dump(paper_chunks_df, f)

        self.paper_specific_indexes[paper_id] = {
            'index_path': index_path,
            'chunks_df_path': chunks_df_path,
            'index_obj': index, # Keep in memory after building
            'chunks_df_obj': paper_chunks_df.copy() # Keep in memory
        }
        # print(f"Index built and saved for paper {paper_id}.")

    def _load_index_for_paper(self, paper_id):
        """Loads the FAISS index and chunks_df for a paper if not already in memory."""
        if paper_id not in self.paper_specific_indexes or \
           self.paper_specific_indexes[paper_id].get('index_obj') is None or \
           self.paper_specific_indexes[paper_id].get('chunks_df_obj') is None:

            index_path, chunks_df_path = self._get_paper_storage_paths(paper_id)

            if os.path.exists(index_path) and os.path.exists(chunks_df_path):
                # print(f"Loading FAISS index and chunks for paper {paper_id} from disk...")
                index_obj = faiss.read_index(index_path)
                with open(chunks_df_path, 'rb') as f:
                    chunks_df_obj = pickle.load(f)
                
                # Update or initialize the entry
                if paper_id not in self.paper_specific_indexes:
                    self.paper_specific_indexes[paper_id] = {}
                self.paper_specific_indexes[paper_id].update({
                    'index_path': index_path,
                    'chunks_df_path': chunks_df_path,
                    'index_obj': index_obj,
                    'chunks_df_obj': chunks_df_obj
                })
                return index_obj, chunks_df_obj
            else:
                # print(f"Warning: Index files not found for paper {paper_id} on disk. Cannot load.")
                return None, None
        else: # Already in memory
            return self.paper_specific_indexes[paper_id]['index_obj'], self.paper_specific_indexes[paper_id]['chunks_df_obj']


    def retrieve_from_paper(self, query_text, paper_id, top_k=5):
        index, paper_chunks_df = self._load_index_for_paper(paper_id)

        if index is None or paper_chunks_df is None or paper_chunks_df.empty:
            # print(f"Error: Index or chunks not available for paper {paper_id}. Cannot retrieve.")
            return pd.DataFrame(columns=['chunk_id', 'text', 'similarity_score_L2_distance'])

        query_embedding = self.model.encode([query_text], convert_to_tensor=False, show_progress_bar=False)
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
        num_indexed_items = index.ntotal
        actual_top_k = min(top_k, num_indexed_items)
        if actual_top_k == 0: return pd.DataFrame(columns=['chunk_id', 'text', 'similarity_score_L2_distance'])
        distances, indices = index.search(query_embedding_np, actual_top_k)
        retrieved_chunks = paper_chunks_df.iloc[indices[0]].copy()
        retrieved_chunks['similarity_score_L2_distance'] = distances[0]
        return retrieved_chunks.sort_values(by='similarity_score_L2_distance', ascending=True)

    def pre_build_all_paper_indexes(self, all_review_chunks_df, force_rebuild=False):
        if all_review_chunks_df is None or all_review_chunks_df.empty:
            print("Error: No review chunks data provided to pre_build_all_paper_indexes.")
            return

        print("Pre-building/checking Dense indexes for all papers...")
        grouped_by_paper = all_review_chunks_df.groupby('paper_id')
        for paper_id, paper_chunks_df_group in tqdm(grouped_by_paper, desc="Building/Checking Dense Paper Indexes", leave=False):
            self.build_index_for_paper(paper_id, paper_chunks_df_group, force_rebuild=force_rebuild)
        print("All paper-specific Dense indexes built/checked.")