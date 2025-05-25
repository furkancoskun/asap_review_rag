from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import os
from tqdm.auto import tqdm

class TfidfRetriever:
    def __init__(self, storage_path_prefix='retriever_storage/tfidf'):
        self.paper_specific_tfidf_models = {} # Stores {'paper_id': {'vectorizer_path': path, 'matrix_path': path, 'chunks_df_path': path}}
        self.storage_path_prefix = storage_path_prefix
        os.makedirs(self.storage_path_prefix, exist_ok=True)

    def _get_paper_storage_paths(self, paper_id):
        paper_dir = os.path.join(self.storage_path_prefix, paper_id)
        os.makedirs(paper_dir, exist_ok=True)
        vectorizer_path = os.path.join(paper_dir, "tfidf_vectorizer.pkl")
        matrix_path = os.path.join(paper_dir, "tfidf_matrix.pkl")
        chunks_df_path = os.path.join(paper_dir, "chunks_df.pkl")
        return vectorizer_path, matrix_path, chunks_df_path

    def build_tfidf_for_paper(self, paper_id, paper_chunks_df, force_rebuild=False):
        vectorizer_path, matrix_path, chunks_df_path = self._get_paper_storage_paths(paper_id)

        if not force_rebuild and os.path.exists(vectorizer_path) and \
           os.path.exists(matrix_path) and os.path.exists(chunks_df_path):
            # print(f"TF-IDF model for paper {paper_id} already exists. Skipping build.")
            if paper_id not in self.paper_specific_tfidf_models: # Ensure metadata is stored
                self.paper_specific_tfidf_models[paper_id] = {
                    'vectorizer_path': vectorizer_path, 'matrix_path': matrix_path, 'chunks_df_path': chunks_df_path,
                    'vectorizer_obj': None, 'matrix_obj': None, 'chunks_df_obj': None}
            return

        if paper_chunks_df.empty:
            print(f"Warning: No chunks provided for paper {paper_id} (TF-IDF). Model not built.")
            return

        print(f"Building and saving TF-IDF model for paper {paper_id} with {len(paper_chunks_df)} chunks...")
        chunk_texts_raw = paper_chunks_df['text'].tolist()
        chunk_texts = [str(text) if pd.notna(text) and text is not None else "" for text in chunk_texts_raw]
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)

        with open(vectorizer_path, 'wb') as f_vec, \
             open(matrix_path, 'wb') as f_mat, \
             open(chunks_df_path, 'wb') as f_chunks:
            pickle.dump(vectorizer, f_vec)
            pickle.dump(tfidf_matrix, f_mat)
            pickle.dump(paper_chunks_df, f_chunks)

        self.paper_specific_tfidf_models[paper_id] = {
            'vectorizer_path': vectorizer_path, 'matrix_path': matrix_path, 'chunks_df_path': chunks_df_path,
            'vectorizer_obj': vectorizer, # Keep in memory
            'matrix_obj': tfidf_matrix,   # Keep in memory
            'chunks_df_obj': paper_chunks_df.copy() # Keep in memory
        }
        # print(f"TF-IDF model built and saved for paper {paper_id}.")

    def _load_tfidf_for_paper(self, paper_id):
        """Loads the TF-IDF model, matrix, and chunks_df for a paper if not already in memory."""
        if paper_id not in self.paper_specific_tfidf_models or \
           self.paper_specific_tfidf_models[paper_id].get('vectorizer_obj') is None or \
           self.paper_specific_tfidf_models[paper_id].get('matrix_obj') is None or \
           self.paper_specific_tfidf_models[paper_id].get('chunks_df_obj') is None:

            vectorizer_path, matrix_path, chunks_df_path = self._get_paper_storage_paths(paper_id)
            if os.path.exists(vectorizer_path) and os.path.exists(matrix_path) and os.path.exists(chunks_df_path):
                # print(f"Loading TF-IDF model for paper {paper_id} from disk...")
                with open(vectorizer_path, 'rb') as f_vec, \
                     open(matrix_path, 'rb') as f_mat, \
                     open(chunks_df_path, 'rb') as f_chunks:
                    vectorizer_obj = pickle.load(f_vec)
                    matrix_obj = pickle.load(f_mat)
                    chunks_df_obj = pickle.load(f_chunks)
                
                if paper_id not in self.paper_specific_tfidf_models:
                    self.paper_specific_tfidf_models[paper_id] = {}
                self.paper_specific_tfidf_models[paper_id].update({
                    'vectorizer_path': vectorizer_path, 'matrix_path': matrix_path, 'chunks_df_path': chunks_df_path,
                    'vectorizer_obj': vectorizer_obj, 'matrix_obj': matrix_obj, 'chunks_df_obj': chunks_df_obj
                })
                return vectorizer_obj, matrix_obj, chunks_df_obj
            else:
                # print(f"Warning: TF-IDF model files not found for paper {paper_id} on disk. Cannot load.")
                return None, None, None
        else: # Already in memory
            return self.paper_specific_tfidf_models[paper_id]['vectorizer_obj'], \
                   self.paper_specific_tfidf_models[paper_id]['matrix_obj'], \
                   self.paper_specific_tfidf_models[paper_id]['chunks_df_obj']

    def retrieve_from_paper(self, query_text, paper_id, top_k=5):
        vectorizer, tfidf_matrix, paper_chunks_df = self._load_tfidf_for_paper(paper_id)

        if vectorizer is None or tfidf_matrix is None or paper_chunks_df is None or paper_chunks_df.empty:
            # print(f"Error: TF-IDF model or chunks not available for paper {paper_id}. Cannot retrieve.")
            return pd.DataFrame(columns=['chunk_id', 'text', 'similarity_score_cosine'])

        query_vector = vectorizer.transform([query_text])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        num_indexed_items = tfidf_matrix.shape[0] # Number of documents/chunks in matrix
        actual_top_k = min(top_k, num_indexed_items)
        if actual_top_k == 0: return pd.DataFrame(columns=['chunk_id', 'text', 'similarity_score_cosine'])
        top_indices = cosine_similarities.argsort()[-actual_top_k:][::-1]
        retrieved_chunks = paper_chunks_df.iloc[top_indices].copy()
        retrieved_chunks['similarity_score_cosine'] = cosine_similarities[top_indices]
        return retrieved_chunks.sort_values(by='similarity_score_cosine', ascending=False)

    def pre_build_all_paper_tfidf(self, all_review_chunks_df, force_rebuild=False):
        if all_review_chunks_df is None or all_review_chunks_df.empty:
            print("Error: No review chunks data provided to pre_build_all_paper_tfidf.")
            return
        print("Pre-building/checking TF-IDF models for all papers...")
        grouped_by_paper = all_review_chunks_df.groupby('paper_id')
        for paper_id, paper_chunks_df_group in tqdm(grouped_by_paper, desc="Building/Checking TF-IDF Paper Models", leave=False):
            self.build_tfidf_for_paper(paper_id, paper_chunks_df_group, force_rebuild=force_rebuild)
        print("All paper-specific TF-IDF models built/checked.")